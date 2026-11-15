"""live: 스트림의 시각과 버림, 읽기 스레드, 상태 줄, 웹 페이지, 데모와의 동등성, 이 PC 안의 TCP 왕복, 명령, 끝나는 길.

카메라는 쓰지 않는다. 입력은 얼굴이 없는 합성 그림(영상 파일, 또는 PyAV가 TCP로 보내는 MPEG-TS)이고, 얼굴이 필요한
곳에서는 `drowsy.synth`의 각본 프레임을 돌려주는 가짜 랜드마커를 쓴다. `mediapipe` 표시가 붙은 테스트만 실제 랜드마커를
얼굴 없는 그림에 돌린다. 얼굴 화소는 어디에도 쓰지 않는다(PLAN 1.2절의 결정 D13): 명령을 돌리는 테스트는 출력 폴더
밖에 아무것도 쓰지 않았는지도 본다(`write_watch`).

- 신호(Ctrl+C, kill, 터미널 닫기)는 같은 프로세스 안에서 자기 자신에게 보내 시험한다(`guarded_signals`). 따로 띄운
  `python -m drowsy.live`에 보내는 시험과 가짜 터미널을 닫는 시험은 실제 랜드마커가 있어야 돈다.
- 웹 페이지의 스크립트는 `node`가 있으면 가짜 문서로 돌려 본다(없으면 그 테스트 하나를 건너뛴다. 브라우저에서는 돌려
  보지 못했다).
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import errno
import http.client
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import cv2
import numpy as np
import onnx
import pyarrow.parquet as pq
import pytest
import yaml

from drowsy import demo, extract, live, synth, video_io
from drowsy.export_onnx import exported_spec_path
from drowsy.rules import ZONE_IN, ZONE_OUT, ZONE_UNKNOWN
from drowsy.spec import FeatureSpec, PipelineSpec, RulesConfig
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OK,
    EVENT_CALIBRATING,
    EVENT_MICROSLEEP,
    PHASE_ONSET,
    DriverStateEvent,
    RawFrame,
)

W, H = 160, 120  # 시험 그림의 크기


# ---------------------------------------------------------------------------
# 도우미: 얼굴 없는 그림, 영상 파일, TCP로 보내는 쪽, 가짜 컨테이너, 각본 프레임
# ---------------------------------------------------------------------------


def picture(index: int, width: int = W, height: int = H) -> np.ndarray:
    """얼굴이 아닌 그림: 회색 바탕 위에서 움직이는 사각형."""
    image = np.full((height, width, 3), 110, dtype=np.uint8)
    left = (4 * index) % (width - 40)
    image[30:70, left : left + 40] = (200, 60, 60)
    return image


def write_clip(path: Path, pts_ms: Sequence[int], container_format: str | None = None) -> None:
    """얼굴 없는 그림을 주어진 pts(ms)로 H.264 영상에 쓴다. container_format을 주면 그 형식(예: mpegts)으로 쓴다."""
    time_base = Fraction(1, 1000)
    container = av.open(str(path), "w", format=container_format)
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "23"}
    for index, pts in enumerate(pts_ms):
        frame = av.VideoFrame.from_ndarray(picture(index), format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def frame_times(n_frames: int, fps: int = 30) -> list[int]:
    """n_frames장의 pts(ms): 0, 33, 67, ..."""
    return [int(index * 1000 / fps + 0.5) for index in range(n_frames)]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Sender(threading.Thread):
    """PyAV로 MPEG-TS(H.264, 지연을 줄인 설정)를 TCP로 보내는 쪽. 그림은 얼굴 없는 합성 그림이고 실제 속도로 보낸다.

    받는 쪽이 듣기 시작할 때까지 접속을 되풀이한다. 받는 쪽이 먼저 끊으면(Ctrl+C 시험) 거기서 멈춘다.
    """

    def __init__(self, port: int, n_frames: int, fps: int = 30) -> None:
        super().__init__(name="test-live-sender", daemon=True)
        self._url = f"tcp://127.0.0.1:{port}"
        self._n_frames = n_frames
        self._fps = fps
        self.sent = 0
        self.error: BaseException | None = None

    def _connect(self) -> tuple[Any, Any]:
        """받는 쪽에 접속한 (컨테이너, 스트림). PyAV는 첫 패킷을 쓸 때에야 접속하므로 `start_encoding`으로 접속을 확인한다."""
        deadline = time.monotonic() + 60.0
        while True:
            container = av.open(self._url, "w", format="mpegts")
            options = {"preset": "ultrafast", "tune": "zerolatency", "g": "10"}
            stream = container.add_stream("libx264", rate=self._fps, options=options)
            stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
            try:
                container.start_encoding()
                return container, stream
            except OSError:  # 아직 듣지 않는다(ConnectionRefusedError)
                container.close()
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

    def run(self) -> None:
        try:
            container, stream = self._connect()
            start = time.monotonic()
            try:
                for index in range(self._n_frames):
                    wait = start + index / self._fps - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    frame = av.VideoFrame.from_ndarray(picture(index), format="rgb24")
                    frame.pts = index
                    frame.time_base = Fraction(1, self._fps)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    self.sent += 1
                for packet in stream.encode():
                    container.mux(packet)
            except OSError:  # 받는 쪽이 연결을 닫았다
                pass
            try:
                container.close()
            except OSError:
                pass
        except BaseException as exc:
            self.error = exc


class FakeFrame:
    """디코딩된 프레임 흉내: pts와, 왼쪽 위 화소에 표식 값을 넣은 그림."""

    def __init__(self, pts: int | None, mark: int = 0) -> None:
        self.pts = pts
        self.mark = mark

    def to_ndarray(self, format: str) -> np.ndarray:  # noqa: A002 (PyAV의 인자 이름)
        assert format == "rgb24"
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[0, 0] = self.mark
        return image


class FakePacket:
    def __init__(self, frames: Sequence[FakeFrame] = (), error: BaseException | None = None) -> None:
        self._frames = list(frames)
        self._error = error

    def decode(self) -> list[FakeFrame]:
        if self._error is not None:
            raise self._error
        return self._frames


class FakeContainer:
    """PyAV 입력 컨테이너 흉내. pts 목록(패킷마다 프레임 하나)이나 패킷 목록으로 만든다."""

    def __init__(self, packets: Sequence[FakePacket], time_base: Fraction | None = Fraction(1, 90000)) -> None:
        self.name = "fake"
        self._packets = list(packets)
        self.streams = SimpleNamespace(video=[SimpleNamespace(time_base=time_base)])

    @classmethod
    def of_pts(cls, pts: Sequence[int | None], time_base: Fraction | None = Fraction(1, 90000)) -> FakeContainer:
        return cls([FakePacket([FakeFrame(value, index)]) for index, value in enumerate(pts)], time_base)

    def demux(self, stream: Any) -> Iterator[FakePacket]:
        assert stream is self.streams.video[0]
        yield from self._packets


def live_frame(t_ms: int) -> live.LiveFrame:
    return live.LiveFrame(t_ms, picture(t_ms))


def scripted_frames(spec: PipelineSpec, duration_ms: int, *segments: Any, fps: float = 30.0) -> list[RawFrame]:
    """각본 세션의 프레임(기본 30 fps, 첫 t_ms는 1). 구간을 주지 않은 시간은 눈을 뜨고 정면을 본다."""
    return synth.render(synth.Scenario(segments=tuple(segments), duration_ms=duration_ms, fps=fps), spec)


SESSION_MS = 300_000


def five_minute_session(spec: PipelineSpec, fps: float = 30.0) -> list[RawFrame]:
    """5분 각본: 기준선 수집 중의 감김, 마이크로슬립, 수면(되풀이 경고), PERCLOS를 올리는 짧은 감김들, 긴 시선 이탈."""
    scripted = [
        synth.Closure(1500, start_ms=20_000),
        synth.Closure(1500, start_ms=100_000),
        synth.Closure(6500, start_ms=150_000),
        *(synth.Closure(800, start_ms=start) for start in range(165_000, 195_000, 5_000)),
        synth.HeadTurn(4500, yaw=35.0, start_ms=250_000),
    ]
    busy = [(segment.start_ms - 1500, segment.start_ms + segment.duration_ms + 1500) for segment in scripted]
    blinks = [
        synth.Blink(start_ms=start)
        for start in range(3_000, SESSION_MS, 4_000)
        if not any(low <= start < high for low, high in busy)
    ]
    return scripted_frames(spec, SESSION_MS, *scripted, *blinks, fps=fps)


class ListReader:
    """`LatestFrameReader` 자리에 넣는 것: 준 프레임을 하나도 버리지 않고 차례로 넘긴다(시간과 무관한 시험용)."""

    def __init__(
        self, frames: Sequence[live.LiveFrame], error: BaseException | None = None, dropped: int = 0
    ) -> None:
        self._frames = list(frames)
        self._next = 0
        self.error = error
        self.dropped = dropped  # 밀려서 버렸다고 알리는 수(시험이 정한다)
        self.stopped = False

    @property
    def received(self) -> int:
        return self._next

    @property
    def finished(self) -> bool:
        return self._next >= len(self._frames)

    def get(self, timeout_s: float) -> live.LiveFrame | None:
        if self.finished:
            return None
        self._next += 1
        return self._frames[self._next - 1]

    def stop(self) -> bool:
        self.stopped = True
        return True


def http_get(port: int, path: str, host: str | None = None) -> tuple[int, str, bytes]:
    """(상태 코드, Content-Type, 본문). host를 주면 그 값을 Host 머리말로 보낸다(없으면 `127.0.0.1:<port>`)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", path, headers={} if host is None else {"Host": host})
        answer = connection.getresponse()
        return answer.status, answer.getheader("Content-Type", ""), answer.read()
    finally:
        connection.close()


def status_json(server_or_bytes: Any) -> dict[str, Any]:
    """상태 JSON을 읽어 `run`(서버가 붙이는 실행 표식)을 뗀 dict. 서버를 주면 그 서버가 지금 들고 있는 상태다."""
    data = server_or_bytes.status_bytes() if isinstance(server_or_bytes, live.StatusServer) else server_or_bytes
    payload = json.loads(data)
    assert isinstance(payload.pop("run"), str)
    return payload


def out_files(out_dir: Path) -> tuple[Path, Path, Path]:
    """출력 폴더에 프레임 표, 타임라인, 이벤트 JSON 셋만 있는지 확인하고 그 경로를 돌려준다."""
    names = sorted(path.name for path in out_dir.iterdir())
    tables = [name for name in names if name.endswith(".parquet")]
    assert len(tables) == 1 and tables[0].startswith("live_"), names
    stem = tables[0][: -len(".parquet")]
    assert names == sorted([f"{stem}.parquet", f"{stem}_events.json", f"{stem}_timeline.png"]), names
    return out_dir / f"{stem}.parquet", out_dir / f"{stem}_timeline.png", out_dir / f"{stem}_events.json"


def printed(lines: Sequence[str], prefix: str) -> list[str]:
    return [line for line in lines if line.startswith(prefix)]


def frame_counts(lines: Sequence[str]) -> dict[str, int]:
    """끝의 `frames: decoded 90, skipped 0 (timestamp), dropped 0 (...), left 0 (...), failed 0 (...), processed 90,
    with a face 0`의 수. 디코딩한 수가 나머지의 합인지도 본다(`decoded = skipped + dropped + left + failed + processed`)."""
    (line,) = printed(lines, "frames: ")
    found = re.fullmatch(
        r"frames: decoded (\d+), skipped (\d+) \(timestamp\), dropped (\d+) \(processing was behind\), "
        r"left (\d+) \(unprocessed at the end\), failed (\d+) \(processing error\), processed (\d+), with a face (\d+)",
        line,
    )
    assert found is not None, line
    names = ("decoded", "skipped", "dropped", "left", "failed", "processed", "face")
    counts = dict(zip(names, (int(value) for value in found.groups())))
    assert counts["decoded"] == sum(counts[name] for name in ("skipped", "dropped", "left", "failed", "processed"))
    return counts


def frames_line(decoded: int, skipped: int, dropped: int, left: int, failed: int, processed: int, face: int) -> str:
    """끝의 프레임 계수 줄 그대로(손으로 센 값을 넣는다)."""
    return (
        f"frames: decoded {decoded}, skipped {skipped} (timestamp), dropped {dropped} (processing was behind), "
        f"left {left} (unprocessed at the end), failed {failed} (processing error), processed {processed}, "
        f"with a face {face}"
    )


def assert_frame_metadata(parquet: Path, **expected: int) -> dict[str, Any]:
    """표의 메타데이터에 적힌 프레임 계수(`n_<이름>`)가 expected와 같고, `n_decoded`가 나머지의 합인지 본다."""
    metadata = extract.read_extract_metadata(parquet)
    assert {name: metadata[f"n_{name}"] for name in expected} == expected
    parts = ("n_skipped", "n_dropped", "n_left", "n_failed", "n_processed")
    assert metadata["n_decoded"] == sum(metadata[key] for key in parts)
    return metadata


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_v0(repo_root: Path) -> Path:
    """저장소의 무작위 가중치 모델. 옆에 `feature_spec_v0.json`이 있다."""
    return repo_root / "export" / "drowsiness_v0.onnx"


@pytest.fixture(scope="module")
def model_fn(model_v0: Path, pipeline_spec: PipelineSpec) -> Any:
    features = pipeline_spec.step_features
    return demo.make_model_fn(model_v0, features.window_steps, len(features.names))


@pytest.fixture
def cli_config(tmp_path: Path, config_path: Path, repo_root: Path) -> Path:
    """기본 설정에서 경로를 절대 경로로, 출력 폴더를 임시 폴더로 바꾼 설정 파일."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"]["feature_spec"] = str(repo_root / "config" / "feature_spec.json")
    data["paths"]["rules"] = str(repo_root / "config" / "rules.yaml")
    data["paths"]["model_path"] = str(repo_root / "models" / "face_landmarker.task")
    data["paths"]["export_dir"] = str(repo_root / "export")
    data["demo"]["out_dir"] = str(tmp_path / "configured_out")
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


CLIP_SCRIPT_MS = 12_000
CLIP_CLOSURE = synth.Closure(2500, start_ms=800)  # 0.8초에 시작해 2.5초 머무는 감김: 기준선 완성 전의 임시 microsleep


@pytest.fixture
def scripted_landmarker(monkeypatch: pytest.MonkeyPatch, pipeline_spec: PipelineSpec) -> SimpleNamespace:
    """`extract`의 랜드마커와 추론을 가짜로 바꾼다: 프레임의 시각에 해당하는 각본 프레임(얼굴 있음)을 돌려준다.

    그림은 보지 않는다(얼굴이 없는 그림이다). 각본은 0.8초에 눈을 감아 2.5초 머문다. 돌려주는 RawFrame의 크기는 각본의
    640x480 그대로다.
    """
    scripted = scripted_frames(pipeline_spec, CLIP_SCRIPT_MS, CLIP_CLOSURE)
    state = SimpleNamespace(created=[], seen=[], shapes=[])

    class FakeLandmarker:
        closed = False

        def close(self) -> None:
            self.closed = True

    def make_landmarker(model_path: Path, spec: PipelineSpec | None = None) -> FakeLandmarker:
        state.created.append(FakeLandmarker())
        return state.created[-1]

    def detect_frame(landmarker: FakeLandmarker, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        assert not landmarker.closed and rgb.ndim == 3 and rgb.dtype == np.uint8
        assert t_ms > (state.seen[-1] if state.seen else 0)  # VIDEO 모드는 엄격히 증가하는 양수만 받는다
        state.seen.append(t_ms)
        state.shapes.append(rgb.shape)
        index = min(len(scripted) - 1, int((t_ms - 1) * 30 / 1000 + 0.5))
        return dataclasses.replace(scripted[index], t_ms=t_ms)

    monkeypatch.setattr(extract, "make_landmarker", make_landmarker)
    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    monkeypatch.setattr(extract, "mediapipe_version", lambda: "0.0-fake")
    return state


@pytest.fixture
def status_server() -> Iterator[live.StatusServer]:
    server = live.StatusServer(0)
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def real_landmarker_model(repo_root: Path, pipeline_spec: PipelineSpec) -> Path:
    model = repo_root / "models" / "face_landmarker.task"
    if not model.is_file():
        pytest.skip("models/face_landmarker.task is absent")
    try:
        extract.make_landmarker(model, pipeline_spec).close()
    except extract.ExtractError as exc:
        pytest.skip(f"MediaPipe cannot be used here: {exc}")
    return model


# 쓰기를 지켜보는 감사 훅(`sys.addaudithook`은 뗄 수 없으므로 한 번만 달고, 지켜보는 동안에만 적는다)
_WRITE_LOGS: list[list[str]] = []
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _audit_open(event: str, args: tuple[Any, ...]) -> None:
    if event != "open" or not _WRITE_LOGS:
        return
    path, _mode, flags = args
    if isinstance(flags, int) and flags & _WRITE_FLAGS and not isinstance(path, int):
        name = os.fsdecode(path)
        for log in _WRITE_LOGS:
            log.append(name)


_AUDIT_INSTALLED: list[bool] = []


@pytest.fixture
def write_watch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """명령이 출력 폴더 밖에 아무것도 쓰지 않는지 보는 장치(얼굴 화소가 디스크에 가지 않는다는 결정 D13의 시험).

    - 시스템 임시 폴더(`TMPDIR`)와 작업 폴더를 임시 폴더 안의 빈 폴더 둘로 바꾼다(`systmp`, `cwd`).
    - `watching()` 안에서 Python이 쓰기로 연 경로를 모두 적는다(`sys.addaudithook`의 `open`. `open`, `os.open`,
      `tempfile`, `numpy.save` 등이 여기를 지난다. C 라이브러리가 직접 여는 파일은 보지 못한다: 그쪽은 위의 두 폴더와
      출력 폴더의 목록, 그리고 소스의 호출 목록 검사가 맡는다).
    - `check(out_dir)`: 적힌 경로가 모두 out_dir 아래이고 두 폴더가 비어 있는지 확인한다.
    """
    if not _AUDIT_INSTALLED:
        sys.addaudithook(_audit_open)
        _AUDIT_INSTALLED.append(True)
    systmp, cwd = tmp_path / "systmp", tmp_path / "cwd"
    systmp.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("TMPDIR", str(systmp))
    monkeypatch.setattr(tempfile, "tempdir", None)  # 다음 `gettempdir()`가 TMPDIR을 다시 읽는다
    monkeypatch.chdir(cwd)
    opened: list[str] = []

    @contextlib.contextmanager
    def watching() -> Iterator[None]:
        _WRITE_LOGS.append(opened)
        try:
            yield
        finally:
            _WRITE_LOGS.remove(opened)

    def check(out_dir: Path) -> None:
        outside = [name for name in opened if not Path(name).resolve().is_relative_to(out_dir.resolve())]
        assert outside == [], outside
        assert opened != []  # 감사 훅이 살아 있다: 출력 폴더의 그림과 JSON은 Python이 연다
        assert sorted(path.name for path in systmp.iterdir()) == []
        assert sorted(path.name for path in cwd.iterdir()) == []

    return SimpleNamespace(watching=watching, check=check, opened=opened, systmp=systmp, cwd=cwd)


class PageWatcher(threading.Thread):
    """명령이 도는 동안 웹 페이지를 읽는 쪽: `/status.json`을 되풀이해 읽고, 세션이 돌기 시작하면 미리보기 그림 하나를 읽는다."""

    def __init__(self, port: int, preview: bool = True) -> None:
        super().__init__(name="test-live-page", daemon=True)
        self._port = port
        self._preview = preview
        self.done = threading.Event()
        self.statuses: list[dict[str, Any]] = []
        self.jpeg: bytes | None = None
        self.preview_code: int | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            while not self.done.is_set():
                try:
                    code, _, body = http_get(self._port, "/status.json")
                except OSError:  # 서버가 아직 없거나 이미 멈췄다
                    time.sleep(0.02)
                    continue
                assert code == 200
                status = status_json(body)
                self.statuses.append(status)
                if status["phase"] == "running" and self.preview_code is None:
                    if self._preview:
                        _, _, self.jpeg = read_preview_part(self._port)
                        self.preview_code = 200
                    else:
                        self.preview_code = http_get(self._port, "/preview.mjpg")[0]
                time.sleep(0.05)
        except BaseException as exc:
            self.error = exc

    def phases(self) -> list[str]:
        """읽은 상태의 phase를 바뀐 순서대로."""
        seen: list[str] = []
        for status in self.statuses:
            if not seen or seen[-1] != status["phase"]:
                seen.append(status["phase"])
        return seen


# ---------------------------------------------------------------------------
# (1) 시각: video_io와 같은 규약, 증가하지 않는 프레임은 버리고 센다
# ---------------------------------------------------------------------------


def test_timestamps_start_at_one_and_frames_that_do_not_increase_are_dropped() -> None:
    # 90 kHz 시계(MPEG-TS). 첫 pts가 0이 아니어도 첫 프레임은 1이다. 넷째(같은 pts), 다섯째(되돌아간 pts), 여섯째(pts 없음)는 버린다.
    pts = [126_000, 129_000, 132_003, 132_003, 131_000, None, 135_000]
    counts = live.StreamCounts()
    frames = list(live.iter_live_frames(FakeContainer.of_pts(pts), 0, counts))
    assert [frame.t_ms for frame in frames] == [1, 34, 68, 101]
    assert [int(frame.rgb[0, 0, 0]) for frame in frames] == [0, 1, 2, 6]  # 버린 것은 3, 4, 5번 프레임이다
    assert (counts.decoded, counts.skipped, counts.decode_errors, counts.arrival_clock) == (7, 3, 0, False)
    # 변환은 video_io의 식 그대로다
    assert [frame.t_ms for frame in frames] == [
        video_io.pts_to_ms(value, pts[0], Fraction(1, 90000)) for value in (126_000, 129_000, 132_003, 135_000)
    ]
    # 반올림은 floor(x + 0.5)다: 0.5 ms는 올린다(Python의 round()는 짝수로 간다)
    halves = list(live.iter_live_frames(FakeContainer.of_pts([0, 1, 3, 5], Fraction(1, 2000))))
    assert [frame.t_ms for frame in halves] == [1, 2, 3, 4]

    # 계수를 받지 않아도 돈다. 프레임이 없는 스트림은 아무것도 내지 않는다.
    assert [frame.t_ms for frame in live.iter_live_frames(FakeContainer.of_pts([5, 5, 95]))] == [1, 2]
    assert list(live.iter_live_frames(FakeContainer([]))) == []


def test_a_stream_without_timestamps_uses_the_arrival_time() -> None:
    ticks = iter([10.0, 10.0334, 10.0336, 10.1])  # 프레임마다 시계를 한 번 읽는다. 첫 프레임의 값이 기준이다
    counts = live.StreamCounts()
    container = FakeContainer.of_pts([None, None, None, None])
    frames = list(live.iter_live_frames(container, 0, counts, clock=lambda: next(ticks)))
    # 0 ms → 1, 33.4 ms → 34, 33.6 ms → 35, 100 ms → 101
    assert [frame.t_ms for frame in frames] == [1, 34, 35, 101]
    assert (counts.decoded, counts.skipped, counts.arrival_clock) == (4, 0, True)

    # 같은 ms에 도착한 프레임은 버린다. 첫 프레임에 pts가 없으면 뒤 프레임에 pts가 있어도 도착 시각만 쓴다.
    ticks = iter([5.0, 5.0002, 5.05])
    counts = live.StreamCounts()
    container = FakeContainer.of_pts([None, 900_000, 1_800_000])
    frames = list(live.iter_live_frames(container, 0, counts, clock=lambda: next(ticks)))
    assert [frame.t_ms for frame in frames] == [1, 51]
    assert (counts.decoded, counts.skipped, counts.arrival_clock) == (3, 1, True)

    # 반올림은 floor(x + 0.5)다: 62.5 ms는 63으로 올린다(Python의 round()는 62로 간다). 1/16초는 정확히 표현되는 값이다
    ticks = iter([2.0, 2.0625])
    frames = list(live.iter_live_frames(FakeContainer.of_pts([None, None]), 0, clock=lambda: next(ticks)))
    assert [frame.t_ms for frame in frames] == [1, 64]

    # time_base가 없는 스트림도 도착 시각으로 읽는다
    ticks = iter([1.0, 1.25])
    counts = live.StreamCounts()
    frames = list(live.iter_live_frames(FakeContainer.of_pts([0, 3000], None), 0, counts, clock=lambda: next(ticks)))
    assert [frame.t_ms for frame in frames] == [1, 251] and counts.arrival_clock is True


def test_undecodable_packets_are_skipped_and_counted() -> None:
    broken = av.error.InvalidDataError(1094995529, "Invalid data found when processing input")
    packets = [
        FakePacket(error=broken),
        FakePacket([FakeFrame(0, 1)]),
        FakePacket(error=broken),
        FakePacket([FakeFrame(3000, 2), FakeFrame(6000, 3)]),  # 패킷 하나가 프레임을 둘 낼 수도 있다
        FakePacket([]),  # 디코더를 비우는 마지막 패킷
    ]
    counts = live.StreamCounts()
    frames = list(live.iter_live_frames(FakeContainer(packets), 0, counts))
    assert [frame.t_ms for frame in frames] == [1, 34, 68]
    assert (counts.decoded, counts.skipped, counts.decode_errors) == (3, 0, 2)

    # 그 밖의 읽기 오류(끊긴 연결, 시간 초과)는 그대로 난다
    stalled = av.error.ExitError(1414092869, "Immediate exit requested")
    with pytest.raises(av.error.ExitError):
        list(live.iter_live_frames(FakeContainer([FakePacket([FakeFrame(0)]), FakePacket(error=stalled)])))


def test_rotation_turns_the_picture_counter_clockwise() -> None:
    decoded = FakeFrame(0, 200).to_ndarray("rgb24")  # 높이 4, 너비 6, 표식은 왼쪽 위
    for rotation, quarter_turns in ((0, 0), (90, 1), (180, 2), (270, 3), (-90, 3), (450, 1)):
        (frame,) = live.iter_live_frames(FakeContainer([FakePacket([FakeFrame(0, 200)])]), rotation)
        assert np.array_equal(frame.rgb, np.rot90(decoded, quarter_turns))  # `video_io.rotate_upright`와 같은 방향
        assert frame.rgb.flags["C_CONTIGUOUS"]
    assert frame.rgb.shape == (6, 4, 3) and frame.rgb[5, 0, 0] == 200  # 90도: 왼쪽 위가 왼쪽 아래로 간다
    with pytest.raises(video_io.VideoError, match="multiple of 90"):
        list(live.iter_live_frames(FakeContainer.of_pts([0]), 45))
    empty = FakeContainer([])
    empty.streams.video = []
    with pytest.raises(video_io.VideoError, match="no video stream"):
        list(live.iter_live_frames(empty))


@pytest.mark.parametrize("container_format", [None, "mpegts"])
def test_frames_of_a_file_match_video_io(tmp_path: Path, container_format: str | None) -> None:
    """같은 파일을 실시간 경로로 읽은 프레임(시각과 화소)이 `video_io.iter_video_frames`가 낸 것과 같다."""
    path = tmp_path / ("clip.ts" if container_format else "clip.mp4")
    pts_ms = [0, 33, 67, 100, 133, 167, 200, 233, 267, 300, 333, 367]
    write_clip(path, pts_ms, container_format)
    expected = list(video_io.iter_video_frames(path, rotation=0))
    assert [t_ms for t_ms, _ in expected] == [1 + value for value in pts_ms]

    for rotation in (0, 90):
        reference = list(video_io.iter_video_frames(path, rotation=rotation))
        counts = live.StreamCounts()
        container = live.open_source(str(path), 10.0)
        try:
            frames = list(live.iter_live_frames(container, rotation, counts))
        finally:
            container.close()
        assert [frame.t_ms for frame in frames] == [t_ms for t_ms, _ in reference]
        assert all(np.array_equal(frame.rgb, rgb) for frame, (_, rgb) in zip(frames, reference))
        assert (counts.decoded, counts.skipped, counts.arrival_clock) == (len(pts_ms), 0, False)
    assert frames[0].rgb.shape == (W, H, 3)  # 90도 돌린 그림


# ---------------------------------------------------------------------------
# (1) 소스 열기
# ---------------------------------------------------------------------------


def test_plan_source_picks_the_format_from_the_source_string(tmp_path: Path) -> None:
    default = live.plan_source(live.DEFAULT_SOURCE)
    assert live.DEFAULT_SOURCE == "tcp://127.0.0.1:5599?listen=1"
    assert (default.format, default.listening, default.realtime) == ("mpegts", True, False)
    assert default.options == live.STREAM_OPTIONS and default.options["fflags"] == "nobuffer"
    assert live.plan_source("tcp://127.0.0.1:5599").listening is False
    assert live.plan_source("udp://127.0.0.1:5599?fifo_size=100&listen=1&x=2").listening is True
    assert live.plan_source("tcp://127.0.0.1:5599?listen=10").listening is False
    device = live.plan_source("/dev/video0")
    assert (device.format, device.options, device.listening, device.realtime) == ("v4l2", {}, False, False)
    clip = tmp_path / "clip.mp4"
    write_clip(clip, [0, 33])
    assert live.plan_source(str(clip)) == live.SourcePlan(None, {}, False, True)
    assert live.plan_source(str(tmp_path / "missing.mp4")) == live.SourcePlan(None, {}, False, False)


def no_video_error(source: str, timeout: str) -> str:
    """듣는 소스가 시간 안에 영상을 받지 못했을 때의 오류 문장 그대로: 아무도 접속하지 않았거나, 접속한 쪽이 영상을 보내지
    않았다(프로그램은 둘을 가릴 수 없다)."""
    return (
        f"no video from a sender on {source} within {timeout} s: either no sender connected, or a client stayed "
        "connected and sent no MPEG-TS video (ffmpeg: -f mpegts). Fix: start this program first, then start the "
        "sender while it waits (ffmpeg on Windows sending MPEG-TS with H.264 to the same port on localhost; see "
        "--help for the command)."
    )


def test_open_source_gives_up_when_no_sender_connects() -> None:
    port = free_port()
    source = f"tcp://127.0.0.1:{port}?listen=1"
    started = time.monotonic()
    with pytest.raises(live.LiveError) as caught:
        live.open_source(source, 0.5)
    assert time.monotonic() - started < 30.0  # 120초 기본값이 아니라 준 시간으로 끝난다
    assert str(caught.value) == no_video_error(source, "0.5")
    # 그 뒤에는 포트가 풀려 있다
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def test_open_source_names_an_address_in_use_and_passes_other_errors_on(tmp_path: Path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        with pytest.raises(live.LiveError, match="already in use"):
            live.open_source(f"tcp://127.0.0.1:{port}?listen=1", 5.0)
    with pytest.raises(OSError) as caught:  # 없는 파일은 PyAV의 오류 그대로다
        live.open_source(str(tmp_path / "missing.mp4"), 5.0)
    assert caught.value.errno == errno.ENOENT and not isinstance(caught.value, live.LiveError)


def test_open_source_says_how_long_it_still_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    assert live.WAIT_NOTE_INTERVAL_S == 30.0
    monkeypatch.setattr(live, "WAIT_NOTE_INTERVAL_S", 0.3)
    said: list[str] = []
    source = f"tcp://127.0.0.1:{free_port()}?listen=1"
    with pytest.raises(live.LiveError) as caught:
        live.open_source(source, 1.5, said.append)
    assert str(caught.value) == no_video_error(source, "1.5")
    # 0.3초마다 남은 시간(1.2, 0.9, 0.6초 → "1 s left"). 남은 시간이 0초로 적힐 알림(1.2초와 1.5초 뒤)은 내지 않는다:
    # 시간 초과의 오류 문장이 바로 뒤에 온다.
    assert 1 <= len(said) <= 3
    assert all(re.fullmatch(r"still waiting for the sender \([1-9]\d* s left\)", line) for line in said), said
    assert not any("(0 s left)" in line for line in said)
    # say를 주지 않으면 조용히 기다린다(오류 없이 같은 결과)
    source = f"tcp://127.0.0.1:{free_port()}?listen=1"
    with pytest.raises(live.LiveError) as caught:
        live.open_source(source, 0.5)
    assert str(caught.value) == no_video_error(source, "0.5")


@pytest.mark.parametrize("sends", [b"", b"this is text, not MPEG-TS\n" * 400])
def test_open_source_names_both_cases_when_a_client_stays_connected_without_video(
    monkeypatch: pytest.MonkeyPatch, sends: bytes
) -> None:
    """접속한 채 아무것도 보내지 않거나 MPEG-TS가 아닌 것을 계속 보내는 쪽: 시간 초과의 오류 문장이 그 경우도 적는다."""
    monkeypatch.setattr(live, "WAIT_NOTE_INTERVAL_S", 0.4)
    port = free_port()
    source = f"tcp://127.0.0.1:{port}?listen=1"
    done = threading.Event()
    connected = threading.Event()

    def client() -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not done.is_set():
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            except OSError:
                time.sleep(0.02)
                continue
            connected.set()
            with connection:
                while not done.is_set():  # 끊지 않고 붙어 있는다
                    if sends:
                        with contextlib.suppress(OSError):
                            connection.sendall(sends)
                    time.sleep(0.05)
            return

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    said: list[str] = []
    try:
        with pytest.raises(live.LiveError) as caught:
            live.open_source(source, 2.0, said.append)
    finally:
        done.set()
        thread.join(30.0)
    assert connected.is_set()
    assert str(caught.value) == no_video_error(source, "2")
    # 접속한 쪽이 끊지 않았으므로 다시 듣는다는 줄은 없고, 남은 시간 알림에 0초는 없다
    assert said and all(re.fullmatch(r"still waiting for the sender \([1-9]\d* s left\)", line) for line in said), said


def test_open_source_gives_up_at_the_stated_time_even_when_pyav_stays_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyAV의 열기가 준 시간을 넘겨 막혀 있어도(`udp://`가 그렇다) 준 시간 + `OPEN_GRACE_S`에 그만둔다."""
    assert live.OPEN_GRACE_S == 0.5
    release = threading.Event()
    opened: list[SimpleNamespace] = []

    def blocked_open(source: str, **kwargs: Any) -> SimpleNamespace:
        assert kwargs["timeout"][0] == pytest.approx(0.3, abs=0.2) and kwargs["timeout"][1] == live.STALL_TIMEOUT_S
        assert release.wait(60.0)  # 준 시간(0.3초)을 훨씬 넘겨 막혀 있다
        container = SimpleNamespace(closed=False, streams=SimpleNamespace(video=[object()]))
        container.close = lambda: setattr(container, "closed", True)
        opened.append(container)
        return container

    monkeypatch.setattr(live.av, "open", blocked_open)
    started = time.monotonic()
    with pytest.raises(live.LiveError, match=r"udp://127.0.0.1:5 gave no video within 0.3 s"):
        live.open_source("udp://127.0.0.1:5", 0.3)
    elapsed = time.monotonic() - started
    assert 0.3 + live.OPEN_GRACE_S <= elapsed < 30.0  # 막힌 열기(60초)를 기다리지 않았다
    # 나중에 열린 컨테이너는 여는 스레드가 스스로 닫는다
    release.set()
    deadline = time.monotonic() + 30.0
    while not (opened and opened[0].closed) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(opened) == 1 and opened[0].closed


def test_open_source_listens_again_after_a_client_that_sends_no_video() -> None:
    """포트만 확인해 보는 접속(붙었다 끊는다)이 기다리는 프로그램을 끝내지 않는다: 다시 듣고, 그 뒤의 보내는 쪽을 받는다."""
    port = free_port()
    source = f"tcp://127.0.0.1:{port}?listen=1"
    said: list[str] = []
    sender = Sender(port, n_frames=60)

    def port_check_then_sender() -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1.0).close()  # 붙었다 바로 끊는다
                break
            except OSError:
                time.sleep(0.05)
        sender.start()

    threading.Thread(target=port_check_then_sender, daemon=True).start()
    container = live.open_source(source, 60.0, said.append)
    try:
        frames = list(live.iter_live_frames(container))
    finally:
        container.close()
    sender.join(30.0)
    assert sender.error is None and 0 < len(frames) <= 60 and frames[0].t_ms == 1
    (line,) = [line for line in said if "listening again" in line]
    assert line.startswith(f"a client connected to {source} and closed without sending MPEG-TS video; ")
    assert "-f mpegts" in line


def test_open_source_explains_a_connection_without_video_and_a_stream_without_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    end_of_file = av.error.EOFError(541478725, "End of file")

    def closes_at_once(source: str, **kwargs: Any) -> Any:
        time.sleep(0.05)
        raise end_of_file

    monkeypatch.setattr(live.av, "open", closes_at_once)
    said: list[str] = []
    # 듣는 소스: 남은 시간 동안 다시 듣다가, 시간이 다 되면 할 일을 적은 오류다
    with pytest.raises(live.LiveError) as caught:
        live.open_source("tcp://127.0.0.1:5?listen=1", 0.4, said.append)
    assert str(caught.value) == (
        "a client connected to tcp://127.0.0.1:5?listen=1 but sent no MPEG-TS video. Fix: the sender must send "
        "MPEG-TS (ffmpeg: -f mpegts) with H.264 video; see --help for the command."
    )
    assert len(said) >= 2 and all("listening again" in line for line in said)
    # 듣는 소스가 아니면(이 프로그램이 접속하는 쪽) PyAV의 오류 그대로다
    with pytest.raises(av.error.EOFError):
        live.open_source("tcp://127.0.0.1:5", 0.4)

    # 열렸는데 비디오 스트림이 없다(MPEG-TS가 아닌 것을 계속 보내는 쪽): 닫고, 할 일을 적는다
    empty = SimpleNamespace(closed=False, streams=SimpleNamespace(video=[]))
    empty.close = lambda: setattr(empty, "closed", True)
    monkeypatch.setattr(live.av, "open", lambda source, **kwargs: empty)
    with pytest.raises(live.LiveError) as caught:
        live.open_source("tcp://127.0.0.1:5?listen=1", 5.0)
    assert str(caught.value).startswith("no video stream in tcp://127.0.0.1:5?listen=1. Fix: the sender must send ")
    assert empty.closed
    with pytest.raises(live.LiveError) as caught:
        live.open_source("clip.mp4", 5.0)
    assert str(caught.value) == "no video stream in clip.mp4"


def test_sender_target_is_the_listening_address_as_the_sender_sees_it() -> None:
    assert live.sender_target(live.DEFAULT_SOURCE) == "tcp://localhost:5599"
    assert live.sender_target("tcp://0.0.0.0:6000?listen=1&x=2") == "tcp://localhost:6000"
    assert live.sender_target("udp://127.0.0.1:5599?listen=1") == "udp://localhost:5599"
    assert live.sender_target("tcp://192.168.0.7:5599?listen=1") == "tcp://192.168.0.7:5599"
    assert live.sender_target("tcp://[::1]:5599?listen=1") == "tcp://localhost:5599"


# ---------------------------------------------------------------------------
# (1) 읽기 스레드: 가장 새 프레임만 넘기고, 버린 것을 센다
# ---------------------------------------------------------------------------


def test_reader_hands_over_the_newest_frame_and_counts_the_dropped_ones() -> None:
    all_out = threading.Event()
    release = threading.Event()

    def frames() -> Iterator[live.LiveFrame]:
        for t_ms in (1, 34, 68, 101):
            yield live_frame(t_ms)
        all_out.set()  # 넷째 프레임까지 읽기 스레드가 받아 갔다
        assert release.wait(30.0)
        yield live_frame(134)

    reader = live.LatestFrameReader(frames())
    assert reader.get(0.05) is None and not reader.finished  # 시작 전: 프레임도 없고 끝나지도 않았다
    reader.start()
    assert all_out.wait(30.0)
    # 처리 루프가 한 번도 가져가지 않는 동안 넷이 왔다: 가장 새 것만 남고 앞의 셋은 버려졌다
    assert (reader.received, reader.dropped) == (4, 3)
    newest = reader.get(1.0)
    assert newest is not None and newest.t_ms == 101
    assert reader.get(0.05) is None and not reader.finished  # 칸이 비었고 스트림은 아직 끝나지 않았다
    assert (reader.received, reader.dropped) == (4, 3)

    release.set()
    last = reader.get(30.0)
    assert last is not None and last.t_ms == 134
    assert reader.get(30.0) is None and reader.finished and reader.error is None
    # 가져간 수 = 받은 수 − 버린 수
    assert (reader.received, reader.dropped) == (5, 3)
    assert reader.stop() is True


def test_reader_keeps_the_last_frame_for_the_consumer_and_reports_a_read_error() -> None:
    failure = av.error.ExitError(1414092869, "Immediate exit requested")

    def frames() -> Iterator[live.LiveFrame]:
        yield live_frame(1)
        yield live_frame(34)
        raise failure

    reader = live.LatestFrameReader(frames())
    reader.start()
    deadline = time.monotonic() + 30.0
    while reader.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert reader.error is failure
    assert not reader.finished  # 읽기는 끝났지만 넘길 프레임이 남아 있다
    left = reader.get(1.0)
    assert left is not None and left.t_ms == 34 and reader.finished
    assert (reader.received, reader.dropped) == (2, 1)
    assert reader.stop() is True


def test_reader_plays_a_file_source_at_its_own_frame_timing() -> None:
    # 타임스탬프가 0.15초씩 떨어진 프레임 넷: 마지막 프레임은 첫 프레임보다 0.45초 뒤에야 나온다
    reader = live.LatestFrameReader([live_frame(t_ms) for t_ms in (1, 151, 301, 451)], realtime=True)
    arrivals: dict[int, float] = {}
    reader.start()
    while not reader.finished:
        frame = reader.get(1.0)
        if frame is not None:
            arrivals[frame.t_ms] = time.monotonic()
    # 실제 속도로 내면 처리 루프가 넷을 모두 가져간다(하한만 본다: 바쁜 기계에서는 더 늦을 수 있다)
    assert sorted(arrivals) == [1, 151, 301, 451] and reader.dropped == 0
    assert arrivals[451] - arrivals[1] >= 0.40
    assert arrivals[301] - arrivals[1] >= 0.25

    # 기다리는 중에도 멈추라는 요청을 바로 듣는다(다음 프레임은 60초 뒤다)
    waiting = live.LatestFrameReader([live_frame(1), live_frame(60_001)], realtime=True)
    waiting.start()
    first = waiting.get(30.0)
    assert first is not None and first.t_ms == 1
    started = time.monotonic()
    assert waiting.stop(30.0) is True and time.monotonic() - started < 30.0
    assert waiting.finished and waiting.received == 1


def test_stop_request_ends_a_stream_that_gives_packets_but_no_frames() -> None:
    """첫 키프레임을 기다리는 스트림(패킷은 오는데 프레임이 없다)도 멈추라는 요청에 바로 끝난다."""
    stop = threading.Event()
    counts = live.StreamCounts()
    frames = live.iter_live_frames(FakeContainer.of_pts([0, 3000, 6000, 9000]), 0, counts, stop=stop)
    assert next(frames).t_ms == 1
    stop.set()
    assert list(frames) == [] and counts.decoded == 1  # 다음 패킷은 풀지 않았다

    class EndlessPackets(FakeContainer):
        """프레임을 내지 않는 패킷이 끝없이 온다."""

        def demux(self, stream: Any) -> Iterator[FakePacket]:
            while True:
                time.sleep(0.001)
                yield FakePacket([])

    shared = threading.Event()
    reader = live.LatestFrameReader(live.iter_live_frames(EndlessPackets([]), stop=shared), stop=shared)
    reader.start()
    assert reader.get(0.05) is None and not reader.finished
    started = time.monotonic()
    assert reader.stop(30.0) is True and time.monotonic() - started < 10.0
    assert shared.is_set() and reader.finished and reader.error is None


def test_reader_stop_waits_only_once_for_a_reader_that_is_blocked() -> None:
    """읽기에 막힌 읽기 스레드를 두 번 기다리지 않는다(처리 루프와 `main`이 한 번씩 부른다)."""
    release = threading.Event()

    def frames() -> Iterator[live.LiveFrame]:
        yield live_frame(1)
        assert release.wait(60.0)  # 소켓 읽기에 막힌 것처럼

    reader = live.LatestFrameReader(frames())
    reader.start()
    assert reader.get(30.0) is not None
    started = time.monotonic()
    assert reader.stop(0.3) is False  # 처음: 준 시간만큼 기다린다
    first = time.monotonic() - started
    assert reader.stop(30.0) is False  # 다시: 기다리지 않는다
    assert first >= 0.3 and time.monotonic() - started - first < 10.0
    release.set()
    deadline = time.monotonic() + 30.0
    while not reader.stop() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert reader.stop() is True and reader.finished


# ---------------------------------------------------------------------------
# (1) 상태 줄, 이벤트 줄
# ---------------------------------------------------------------------------


def a_status(**changes: Any) -> live.LiveStatus:
    base = live.LiveStatus(
        t_ms=12_301,
        session_ms=12_300,
        frames=370,
        face=True,
        closure=0.0712,
        closure_valid=True,
        provisional=True,
        gaze_zone=ZONE_UNKNOWN,
        baseline_status=None,
        baseline_ms=60_000,
        events=1,
        alerts=0,
        model_outputs=0,
        fps_received=30.04,
        fps_processed=29.95,
        dropped=2,
        skipped=1,
    )
    return dataclasses.replace(base, **changes)


def test_status_line() -> None:
    assert live.format_status_line(a_status()) == (
        "session 12.3 s, face yes, closure 0.07 (provisional), gaze unknown, baseline 12/60 s, "
        "received 30.0 fps, processed 29.9 fps, dropped 2, skipped 1"
    )
    # 얼굴이 없으면 closure는 뜻이 없다. 기준선이 연장되면 분모가 지금 마감이다.
    assert live.format_status_line(
        a_status(session_ms=75_999, face=False, closure=0.0, closure_valid=False, baseline_ms=90_000)
    ) == (
        "session 76.0 s, face no, closure n/a, gaze unknown, baseline 75/90 s, "
        "received 30.0 fps, processed 29.9 fps, dropped 2, skipped 1"
    )
    # 기준선이 완성된 뒤: 상태와 완성된 세션 시간. closure는 임시가 아니다.
    done = a_status(session_ms=61_000, provisional=False, gaze_zone=ZONE_IN, baseline_status=BASELINE_OK, closure=1.0)
    assert live.format_status_line(done) == (
        "session 61.0 s, face yes, closure 1.00, gaze forward, baseline ok (60 s), "
        "received 30.0 fps, processed 29.9 fps, dropped 2, skipped 1"
    )
    away = a_status(
        session_ms=200_000,
        provisional=False,
        gaze_zone=ZONE_OUT,
        baseline_status=BASELINE_FALLBACK_GLOBAL,
        baseline_ms=180_000,
        fps_received=0.0,
        fps_processed=0.0,
        dropped=0,
        skipped=0,
    )
    assert live.format_status_line(away) == (
        "session 200.0 s, face yes, closure 0.07, gaze away, baseline fallback_global (180 s), "
        "received 0.0 fps, processed 0.0 fps, dropped 0, skipped 0"
    )

    # 모델 출력은 상태 줄에 나오지 않는다(무작위 가중치 모델의 값을 판정처럼 보이게 하지 않는다)
    with_output = a_status(model_outputs=1, new_probs=(demo.ProbPoint(12_301, 12_001, 12, (0.1, 0.2, 0.7)),))
    assert live.format_status_line(with_output) == live.format_status_line(a_status())
    assert "0.7" not in json.dumps(live.status_fields(with_output))

    # 상태 JSON은 그 줄과, 줄을 이루는 값들을 담는다
    fields = live.status_fields(a_status())
    assert fields["line"] == live.format_status_line(a_status())
    assert (fields["session_s"], fields["face"], fields["closure"], fields["provisional"]) == (12.3, True, 0.0712, True)
    assert fields["gaze_zone"] == ZONE_UNKNOWN
    assert fields["baseline"] == {"done": False, "status": None, "elapsed_s": 12, "target_s": 60}
    assert (fields["fps_received"], fields["fps_processed"]) == (30.04, 29.95)
    assert (fields["dropped"], fields["skipped"]) == (2, 1)
    assert live.status_fields(a_status(closure_valid=False))["closure"] is None
    assert live.status_fields(done)["baseline"] == {"done": True, "status": "ok", "elapsed_s": 61, "target_s": 60}
    json.dumps(fields, allow_nan=False)


def test_event_and_alert_lines() -> None:
    event = DriverStateEvent(EVENT_MICROSLEEP, 2, 20_501, 1_000, 1.0, "closure", PHASE_ONSET, True)
    record = demo.EventRecord(21_501, event, True)
    assert live.format_event_line(record, 1) == (
        "event 21.5 s: microsleep onset (severity 2, started 20.5 s, duration 1.0 s, confidence 1.00, "
        "source closure, provisional)"
    )
    final = dataclasses.replace(event, provisional=False, phase="end", duration_ms=1_634, confidence=0.5)
    assert live.format_event_line(demo.EventRecord(22_135, final, False), 1) == (
        "event 22.1 s: microsleep end (severity 2, started 20.5 s, duration 1.6 s, confidence 0.50, source closure)"
    )
    alert = demo.AlertRecord(24_501, dataclasses.replace(event, duration_ms=4_000), demo.ALERT_DEFERRED)
    assert live.format_alert_line(alert, 1) == "alert 24.5 s: microsleep (deferred, severity 2, duration 4.0 s)"

    # 상태 JSON의 항목은 이벤트 JSON의 항목과 같은 값에 세션 시간과 터미널 줄을 더한 것이다
    fields = live.event_fields(record, 1)
    assert (fields["t_ms"], fields["alert"], fields["session_s"]) == (21_501, True, 21.5)
    assert {key: fields[key] for key in dataclasses.asdict(event)} == dataclasses.asdict(event)
    assert fields["line"] == live.format_event_line(record, 1)
    fields = live.alert_fields(alert, 1)
    assert (fields["kind"], fields["session_s"], fields["type"]) == ("deferred", 24.5, "microsleep")
    assert fields["line"] == live.format_alert_line(alert, 1) and "alert" not in fields


SENDER_LINE = (
    "sender example (Windows PowerShell; start it while this program waits): ffmpeg -hide_banner -f dshow "
    '-video_size 640x480 -framerate 30 -i video="<camera name>" -c:v libx264 -preset ultrafast -tune zerolatency '
    "-g 30 -pix_fmt yuv420p -f mpegts tcp://localhost:{port}"
)
CAMERA_LINE = "camera names for the sender: ffmpeg -hide_banner -list_devices true -f dshow -i dummy"
STOP_LINE = (
    "stop: press Ctrl+C in this terminal, or {ends}. The files of the session are written then; keep this window "
    'open until the "wrote" lines have appeared.'
)


def test_startup_lines_say_what_is_running(model_v0: Path, feature_spec: FeatureSpec) -> None:
    model = demo.verify_model(model_v0, feature_spec)
    plan = live.plan_source(live.DEFAULT_SOURCE)
    lines = live.startup_lines(
        model, live.DEFAULT_SOURCE, plan, 0, None, "http://localhost:8765/", True, rotation_given=False
    )
    assert lines == [
        "drowsy.live: desktop test of the streaming pipeline on a camera stream. It verifies nothing about Android. "
        "Assistive guidance only, not a safety system.",
        f"model: {model_v0} (version 0, trained_on: none (random weights)): the model output carries no meaning, "
        "only the rule events do",
        "drowsiness.source: perclos (drowsy events come from PERCLOS, not from the model)",
        "source: tcp://127.0.0.1:5599?listen=1 (listening; waiting up to 120 s for the sender to connect)",
        SENDER_LINE.format(port=5599),
        CAMERA_LINE,
        "rotation: 0 (the default; --rotation was not given; a live stream has no automatic rotation check: look at "
        "the preview)",
        "speed: not given (alerts are not speed-gated)",
        "page: http://localhost:8765/ (status and camera preview; the preview is kept in memory only and served "
        "on 127.0.0.1 only)",
        STOP_LINE.format(ends="stop the sender"),
    ]
    # 웹 페이지의 "This run": 같은 여섯 줄(보내는 쪽의 명령, 페이지 주소, 끝내는 방법은 없다). 소스가 열린 뒤에는
    # 소스 줄이 "기다리는 중"이 아니다.
    waiting = live.run_lines(model, live.DEFAULT_SOURCE, plan, 0, None, rotation_given=False)
    assert waiting == lines[:4] + lines[6:8]
    opened = live.run_lines(model, live.DEFAULT_SOURCE, plan, 0, None, rotation_given=False, opened=True)
    assert opened[3] == "source: tcp://127.0.0.1:5599?listen=1 (listening; the sender is connected)"
    assert opened[:3] + opened[4:] == waiting[:3] + waiting[4:]

    # 학습한 모델이면 무작위 가중치 표시가 없다. 파일 소스(보내는 쪽의 명령이 없다), --rotation을 준 경우, 속도, 미리보기 끔
    trained = dataclasses.replace(model, metadata={**model.metadata, "model_version": "1", "trained_on": "uta-rldd"})
    file_plan = live.SourcePlan(None, {}, False, True)
    lines = live.startup_lines(trained, "clip.mp4", file_plan, 90, 60.0, "http://x/", False)
    assert len(lines) == 8 and lines[1] == f"model: {model_v0} (version 1, trained_on: uta-rldd)"
    assert lines[3] == "source: clip.mp4 (video file, played back at its own frame timing)"
    assert lines[4] == (
        "rotation: 90 (from --rotation; a live stream has no automatic rotation check: look at the preview)"
    )
    assert lines[5] == "speed: 60 km/h (constant)"
    assert lines[6] == "page: http://x/ (status only; the camera preview is off (--no-preview))"
    assert lines[7] == STOP_LINE.format(ends="wait for the source to end")
    assert live.run_lines(trained, "clip.mp4", file_plan, 90, 60.0, opened=True)[3] == lines[3]
    device = live.plan_source("/dev/video0")
    other = live.startup_lines(model, "/dev/video0", device, 0, None, "http://x/", True, 5.0)
    assert other[3] == "source: /dev/video0 (opening, up to 5 s)" and len(other) == 8
    assert live.run_lines(model, "/dev/video0", device, 0, None, opened=True)[3] == "source: /dev/video0 (opened)"
    # 끝이 없는 소스(카메라 장치, 이 프로그램이 접속하는 스트림): 끝을 기다리라고 하지 않고 소스를 멈추라고 한다
    assert other[7] == STOP_LINE.format(ends="stop the source")
    client = live.plan_source("tcp://127.0.0.1:6002")
    assert live.startup_lines(model, "tcp://127.0.0.1:6002", client, 0, None, "http://x/", True)[-1] == (
        "stop: press Ctrl+C in this terminal, or stop the source. The files of the session are written then; keep "
        'this window open until the "wrote" lines have appeared.'
    )
    # 듣는 소스의 다른 포트: 보내는 쪽의 명령이 그 포트를 가리킨다
    source = "tcp://127.0.0.1:6001?listen=1"
    assert live.startup_lines(model, source, live.plan_source(source), 0, None, "http://x/", True)[4] == (
        SENDER_LINE.format(port=6001)
    )


def test_startup_lines_warn_when_drowsy_events_come_from_a_random_model(
    model_v0: Path, feature_spec: FeatureSpec
) -> None:
    """졸음 판정원이 모델인데 모델이 무작위 가중치면 졸음 이벤트에도 뜻이 없다: 시작 줄이 그렇게 말한다."""
    model = demo.verify_model(model_v0, feature_spec)
    trained = dataclasses.replace(model, metadata={**model.metadata, "model_version": "1", "trained_on": "uta-rldd"})
    plan = live.SourcePlan(None, {}, False, True)

    def lines(which: demo.ModelInfo, source: str) -> list[str]:
        return live.run_lines(which, "clip.mp4", plan, 0, None, drowsiness_source=source)[1:3]

    assert lines(model, "model") == [
        f"model: {model_v0} (version 0, trained_on: none (random weights)): the model output carries no meaning, "
        "and neither do drowsy events (drowsiness.source model takes them from the model); only the other rule "
        "events do",
        "drowsiness.source: model (drowsy events come from the model output) WARNING: this model has random "
        "weights, so drowsy events carry no meaning",
    ]
    either = lines(model, "either")
    assert "drowsiness.source either takes them from the model" in either[0]
    assert either[1] == (
        "drowsiness.source: either (drowsy events come from the model output or from PERCLOS) WARNING: this model "
        "has random weights, so drowsy events carry no meaning"
    )
    # 학습한 모델이면 경고가 없다
    assert lines(trained, "model") == [
        f"model: {model_v0} (version 1, trained_on: uta-rldd)",
        "drowsiness.source: model (drowsy events come from the model output)",
    ]
    assert lines(trained, "perclos")[1] == (
        "drowsiness.source: perclos (drowsy events come from PERCLOS, not from the model)"
    )


def test_baseline_note_and_help_text(pipeline_spec: PipelineSpec) -> None:
    assert live.baseline_note(pipeline_spec) == (
        "baseline: the first 60 s of the session are the reference for this driver (extended up to 180 s while too "
        "few usable frames were seen). Look ahead normally. Until the baseline is complete, eye-closure events are "
        "provisional and looking away gives no event."
    )
    # --help: 보내는 쪽의 명령(기본 포트), 카메라 이름과 해상도 목록, 차례, 끝내는 방법
    text = live._build_parser().format_help()
    assert SENDER_LINE.format(port=5599).split(": ", 1)[1] in text
    assert "ffmpeg -hide_banner -list_devices true -f dshow -i dummy" in text
    assert 'ffmpeg -hide_banner -list_options true -f dshow -i video="<camera name>"' in text
    assert "waits up to 120 s" in text and "http://localhost:8765/" in text
    # 127.0.0.1로 바꿔 볼 때: 접속이 거부될 때만이 아니라, 보내는 쪽을 시작했는데도 기다린다는 줄만 나올 때도
    hint = [
        "     key frame. Do not add -re or an output -r (they change the frame timing). If the sender is refused",
        "     while this program waits, or the sender has started and this program still prints only",
        '     "still waiting" lines, try tcp://127.0.0.1:5599.',
    ]
    epilog = live._help_epilog().splitlines()
    start = epilog.index(hint[0])
    assert epilog[start : start + 3] == hint and "\n".join(hint) in text
    assert "stop with Ctrl+C in this terminal, or stop the sender" in text
    assert "Do not add -re or an output -r" in text and "(default: 0, not turned)" in text


# ---------------------------------------------------------------------------
# (1) 웹 페이지: 페이지, 상태 JSON, 미리보기(켬과 끔)
# ---------------------------------------------------------------------------


def read_preview_part(port: int, host: str | None = None) -> tuple[str, dict[str, str], bytes]:
    """`/preview.mjpg`에서 첫 그림 하나를 읽는다: (응답의 Content-Type, 그 부분의 머리말, 그림 바이트)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", "/preview.mjpg?run=1&t=2", headers={} if host is None else {"Host": host})
        answer = connection.getresponse()
        assert answer.status == 200
        boundary = answer.fp.readline()
        headers: dict[str, str] = {}
        while True:
            line = answer.fp.readline().decode("ascii").strip()
            if not line:
                break
            name, value = line.split(":", 1)
            headers[name.strip()] = value.strip()
        body = answer.fp.read(int(headers["Content-Length"]))
        assert boundary.strip() == b"--" + answer.getheader("Content-Type").split("boundary=")[1].encode("ascii")
        return answer.getheader("Content-Type"), headers, body
    finally:
        connection.close()


def test_status_server_serves_the_page_the_status_and_the_preview(status_server: live.StatusServer) -> None:
    port = status_server.port
    assert port != 0 and status_server.url == f"http://localhost:{port}/"
    assert status_server._httpd.server_address[0] == "127.0.0.1" == live.BIND_HOST  # 이 PC 안에서만 열린다

    code, content_type, body = http_get(port, "/")
    page = body.decode("utf-8")
    assert code == 200 and content_type == "text/html; charset=utf-8"
    assert "<title>drowsy live</title>" in page and "/status.json" in page and "/preview.mjpg" in page
    # 스크립트의 줄바꿈 문자는 JavaScript의 이스케이프로 남아 있다(문자열 안에 진짜 줄바꿈이 들어가면 스크립트가 깨진다.
    # 스크립트를 실제로 돌려 보는 시험은 node가 있을 때만 돈다)
    assert '(s.info || []).join("\\n");' in page and page.count("<script>") == 1
    # 외부 자원을 쓰지 않는다: 다른 곳을 가리키는 주소가 없다
    assert "http://" not in page and "https://" not in page and "//cdn" not in page and "@import" not in page
    assert http_get(port, "/index.html")[2] == body

    # 처음에는 기다리는 상태다. publish한 내용이 그대로 나온다. 서버는 언제나 `run`(이 실행의 표식)을 붙인다.
    code, content_type, body = http_get(port, "/status.json?x=1")
    assert code == 200 and content_type == "application/json"
    assert json.loads(body) == {"phase": "waiting", "preview": True, "run": status_server.run_id}
    payload = {"phase": "running", "preview": True, **live.status_fields(a_status()), "events": [], "alerts": []}
    status_server.publish(payload)
    answer = json.loads(http_get(port, "/status.json")[2])
    assert answer == {**json.loads(json.dumps(payload)), "run": status_server.run_id}
    other = live.StatusServer(0)
    try:
        assert other.run_id != status_server.run_id and status_server.run_id != ""  # 실행마다 다른 값이다
    finally:
        other.stop()

    # 미리보기: multipart JPEG. publish_preview로 준 바이트가 그대로 나온다.
    jpeg = live.encode_preview(picture(3))
    status_server.publish_preview(jpeg)
    content_type, headers, received = read_preview_part(port)
    assert content_type.startswith("multipart/x-mixed-replace; boundary=")
    assert headers == {"Content-Type": "image/jpeg", "Content-Length": str(len(jpeg))} and received == jpeg

    code, content_type, body = http_get(port, "/elsewhere")
    assert code == 404 and body == b"not found\n"


def test_preview_stream_follows_new_pictures_and_ends_when_the_server_stops() -> None:
    server = live.StatusServer(0)
    server.start()
    first, second = live.encode_preview(picture(0)), live.encode_preview(picture(9))
    assert first != second
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        server.publish_preview(first)
        connection.request("GET", "/preview.mjpg")
        answer = connection.getresponse()

        def next_picture() -> bytes:
            length = 0
            while True:
                line = answer.fp.readline()
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1])
                elif line == b"\r\n" and length:
                    return answer.fp.read(length)

        assert next_picture() == first
        server.publish_preview(second)
        assert next_picture() == second
        server.stop()
        assert answer.fp.read() == b"\r\n"  # 남은 것은 마지막 그림 뒤의 줄 끝뿐이고 연결이 닫힌다
    finally:
        connection.close()
        server.stop()


def test_preview_can_be_switched_off() -> None:
    server = live.StatusServer(0, preview=False)
    server.start()
    try:
        server.publish_preview(live.encode_preview(picture(1)))  # 끄면 받지 않는다
        code, _, body = http_get(server.port, "/preview.mjpg")
        assert code == 404 and b"--no-preview" in body
        assert server.wait_preview(-1) == (0, None)  # 그림을 들고 있지 않다
        assert status_json(http_get(server.port, "/status.json")[2]) == {"phase": "waiting", "preview": False}
        assert http_get(server.port, "/")[0] == 200
    finally:
        server.stop()


def test_status_server_answers_only_requests_for_localhost(status_server: live.StatusServer) -> None:
    """Host 머리말이 이 PC의 이름이 아니면 403이다(다른 이름을 127.0.0.1로 돌려 미리보기를 읽어 가는 길을 막는다)."""
    port = status_server.port
    status_server.publish_preview(live.encode_preview(picture(1)))
    others = ("attacker.example", f"attacker.example:{port}", "localhost", f"localhost:{port + 1}", f"10.0.0.5:{port}")
    for host in others:
        for path in ("/", "/status.json", "/preview.mjpg", "/elsewhere"):
            code, content_type, body = http_get(port, path, host)
            assert (code, body) == (403, b"forbidden: this page answers only on localhost\n"), (host, path)
            assert content_type == "text/plain; charset=utf-8"
    for host in (f"localhost:{port}", f"LOCALHOST:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"):
        assert http_get(port, "/status.json", host)[0] == 200
        assert http_get(port, "/", host)[0] == 200
        assert read_preview_part(port, host)[1]["Content-Type"] == "image/jpeg"
    assert live.ALLOWED_HOST_NAMES == ("localhost", "127.0.0.1", "[::1]")
    # Host 머리말이 없는 요청(HTTP/1.0)은 받는다
    with socket.create_connection(("127.0.0.1", port), timeout=10) as plain:
        plain.sendall(b"GET /status.json HTTP/1.0\r\n\r\n")
        assert plain.recv(64).startswith(b"HTTP/1.0 200 OK")


def test_clients_that_break_off_leave_no_traceback_in_the_terminal(
    status_server: live.StatusServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """페이지를 새로 고치거나 닫을 때처럼 응답 도중에 끊는 연결이 터미널에 traceback을 남기지 않는다."""
    port = status_server.port
    status_server.publish({"phase": "running", "preview": True, "info": ["x" * 150] * 20})  # 3 kB쯤의 상태
    reset = struct.pack("ii", 1, 0)  # 닫을 때 RST를 보낸다
    request = b"GET /status.json HTTP/1.1\r\nHost: localhost:%d\r\n\r\n" % port
    for kind in ("reset after the request", "close after the request", "reset before any request"):
        for _ in range(40):
            client = socket.create_connection(("127.0.0.1", port), timeout=10)
            if kind != "reset before any request":
                client.sendall(request)
            if kind != "close after the request":
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, reset)
            client.close()
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:  # 요청 스레드들이 끝나기를 기다린다
        if not any("process_request_thread" in thread.name for thread in threading.enumerate()):
            break
        time.sleep(0.02)
    assert http_get(port, "/status.json")[0] == 200  # 서버는 계속 답한다
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""


def test_a_preview_client_that_never_reads_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """미리보기를 받다가 읽지 않는 연결은 소켓 시간 제한(`HTTP_TIMEOUT_S`)에 걸려 끊긴다(요청 스레드가 남지 않는다)."""
    assert live.HTTP_TIMEOUT_S == 10.0 and live._Handler.timeout == live.HTTP_TIMEOUT_S
    monkeypatch.setattr(live._Handler, "timeout", 0.3)
    server = live.StatusServer(0)
    server.start()
    try:
        client = socket.create_connection(("127.0.0.1", server.port), timeout=10)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        client.sendall(b"GET /preview.mjpg HTTP/1.0\r\n\r\n")

        def handlers() -> list[threading.Thread]:
            return [thread for thread in threading.enumerate() if "process_request_thread" in thread.name]

        deadline = time.monotonic() + 30.0
        big = b"\xff\xd8" + bytes(4_000_000) + b"\xff\xd9"  # 소켓 버퍼를 채우고도 남는 그림
        seen_handler = False
        while time.monotonic() < deadline:
            server.publish_preview(big)  # 읽지 않는 쪽으로 계속 보낸다
            seen_handler = seen_handler or bool(handlers())
            if seen_handler and not handlers():
                break
            time.sleep(0.05)
        assert seen_handler and handlers() == []
        client.close()
    finally:
        server.stop()


def test_status_server_refuses_a_port_in_use(status_server: live.StatusServer) -> None:
    with pytest.raises(OSError) as caught:
        live.StatusServer(status_server.port)
    assert caught.value.errno == errno.EADDRINUSE


# 웹 페이지의 스크립트를 브라우저 없이 돌리는 틀(node). 문서와 fetch를 흉내 내고, 상태를 하나씩 먹인 뒤의 화면을 적는다.
PAGE_HARNESS = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const steps = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const ids = [...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
function node(tag) {
  return { tag, textContent: "", className: "", style: {}, children: [], src: null, parent: null,
    appendChild(c) { c.parent = this; this.children.push(c); },
    prepend(c) { c.parent = this; this.children.unshift(c); },
    replaceChildren() { this.children = []; },
    remove() { if (this.parent) { this.parent.children = this.parent.children.filter((c) => c !== this); } } };
}
const nodes = {};
for (const id of ids) { nodes[id] = node("#" + id); }
const made = [];
global.document = { getElementById: (id) => nodes[id],
  createElement: (tag) => { const n = node(tag); made.push(n); return n; } };
let answers = [];
global.fetch = async () => {
  const a = answers.shift();
  if (a === null || a === undefined) { throw new Error("no answer"); }
  return { json: async () => a };
};
let pollFn = null;
global.setInterval = (fn) => { pollFn = fn; };
function snapshot() {
  const text = (id) => nodes[id].textContent;
  return { phase: text("phase"), problem: nodes.phase.className === "problem", hint: text("hint"),
    session: text("session"), face: text("face"), closure: text("closure"), gaze: text("gaze"),
    baseline: text("baseline"), fps: text("fps"), dropped: text("dropped"), skipped: text("skipped"),
    line: text("line"), info: text("info"), note: text("preview-note"),
    eventRows: nodes.events.children.map((row) => row.children.map((cell) => cell.textContent)),
    alertRows: nodes.alerts.children.length,
    pictures: nodes["preview-box"].children.filter((c) => c.tag === "img").map((c) => c.src),
    picturesMade: made.filter((n) => n.tag === "img").length };
}
(async () => {
  const out = [];
  answers = [steps[0]];
  eval(script);  // 스크립트가 한 번 읽고 주기를 단다
  await new Promise((resolve) => setTimeout(resolve, 20));
  out.push(snapshot());
  for (const step of steps.slice(1)) {
    answers = [step];
    await pollFn();
    out.push(snapshot());
  }
  console.log(JSON.stringify(out));
})();
"""


def test_page_script_shows_the_state_and_makes_the_preview_again_after_a_restart(tmp_path: Path) -> None:
    """프로그램을 다시 시작해도 열어 둔 페이지의 미리보기가 다시 만들어지고, 앞 세션의 값이 남지 않는다."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the page script is not run without a JavaScript engine")
    event = DriverStateEvent(EVENT_MICROSLEEP, 3, 20_501, 1_000, 1.0, "closure", PHASE_ONSET, True)
    fields = live.status_fields(a_status())
    events = [live.event_fields(demo.EventRecord(21_501, event, True), 1)]
    first = {"preview": True, "run": "run-a", "info": ["line one", "line two"], "end_reason": None}
    second = {**first, "run": "run-b"}
    running = {"phase": "running", "source_open": True, "hint": "baseline: look ahead", **fields, "events": events,
               "alerts": []}
    steps: list[dict[str, Any] | None] = [
        {**first, "phase": "waiting", "source_open": False},  # 0 보내는 쪽을 기다린다
        {**first, "phase": "waiting", "source_open": True, "hint": "", "events": [], "alerts": []},  # 1 소스가 열렸다
        {**first, **running},  # 2 돌고 있다
        {**first, **running, "phase": "ended", "end_reason": "interrupted (Ctrl+C)", "hint": ""},  # 3 끝났다
        None,  # 4 프로그램이 사라졌다
        {**second, "phase": "waiting", "source_open": False},  # 5 다시 시작했다(같은 탭)
        {**second, **running},  # 6
        {**first, "run": "run-c", "phase": "waiting", "source_open": False},  # 7 답이 끊긴 적 없이 실행이 바뀌었다
        {**first, "run": "run-c", "phase": "running", **running, "preview": False},  # 8 미리보기가 꺼졌다
    ]
    harness, page, steps_file = tmp_path / "harness.js", tmp_path / "page.html", tmp_path / "steps.json"
    harness.write_text(PAGE_HARNESS, encoding="utf-8")
    page.write_text(live.PAGE_HTML, encoding="utf-8")
    steps_file.write_text(json.dumps(steps), encoding="utf-8")
    done = subprocess.run([node, str(harness), str(page), str(steps_file)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    shown = json.loads(done.stdout)
    assert len(shown) == len(steps)

    assert shown[0]["phase"] == "waiting for the camera stream" and shown[0]["session"] == "-"
    assert shown[1]["phase"] == "the source is open; waiting for the first picture"
    assert shown[0]["info"] == "line one\nline two"
    # 돌고 있는 동안: 상태 줄과 같은 값, 기준선 안내, 이벤트
    assert (shown[2]["phase"], shown[2]["hint"]) == ("running", "baseline: look ahead")
    assert (shown[2]["session"], shown[2]["face"], shown[2]["closure"]) == ("12.3 s", "found", "0.07 (provisional)")
    assert (shown[2]["gaze"], shown[2]["baseline"], shown[2]["fps"]) == ("unknown", "12 / 60 s", "30.0 / 29.9")
    assert (shown[2]["dropped"], shown[2]["skipped"], shown[2]["line"]) == ("2", "1", fields["line"])
    assert shown[2]["eventRows"] == [["21.5 s", "microsleep", "onset", "3", "1.0 s", "closure", "provisional alert"]]
    assert shown[3]["phase"].startswith("session ended: interrupted (Ctrl+C).") and shown[3]["hint"] == ""
    # 미리보기 그림은 실행마다 하나다. 첫 실행 동안에는 같은 그림이 그대로 있다.
    assert [len(step["pictures"]) for step in shown[:4]] == [1, 1, 1, 1] and shown[3]["picturesMade"] == 1
    assert shown[0]["pictures"][0].startswith("/preview.mjpg?run=run-a&t=")
    # 프로그램이 사라지면 그림을 치우고 그렇게 알린다
    assert shown[4]["problem"] and shown[4]["phase"].startswith("no answer from the program")
    assert shown[4]["pictures"] == []
    # 다시 시작하면: 앞 세션의 값과 이벤트가 지워지고, 그림이 새 주소로 다시 만들어진다
    assert not shown[5]["problem"] and shown[5]["phase"] == "waiting for the camera stream"
    assert (shown[5]["session"], shown[5]["face"], shown[5]["line"], shown[5]["eventRows"]) == ("-", "-", "", [])
    assert len(shown[5]["pictures"]) == 1 and shown[5]["pictures"][0].startswith("/preview.mjpg?run=run-b&t=")
    assert shown[6]["session"] == "12.3 s" and shown[6]["pictures"] == shown[5]["pictures"]
    # 답이 끊긴 적이 없어도(숨은 탭은 드물게 읽는다) 실행이 바뀌면 같다
    assert shown[7]["session"] == "-" and shown[7]["eventRows"] == []
    assert len(shown[7]["pictures"]) == 1 and shown[7]["pictures"][0].startswith("/preview.mjpg?run=run-c&t=")
    # 미리보기가 꺼지면 그림이 없다
    assert shown[8]["pictures"] == [] and shown[8]["note"].startswith("the camera preview is off")
    assert shown[8]["picturesMade"] == 3


def test_page_script_comments_are_korean_like_the_other_comments() -> None:
    """웹 페이지 스크립트 안의 주석도 저장소의 다른 주석처럼 한국어로 적는다(페이지가 보여 주는 글은 영어 그대로다)."""
    script = re.search(r"<script>(.*?)</script>", live.PAGE_HTML, re.S)
    assert script is not None
    comments = re.findall(r"//(.*)$", script.group(1), re.M)
    assert len(comments) == 6, comments
    assert all(re.search(r"[가-힣]", comment) for comment in comments), comments
    assert not re.search(r"/\*|<!--", live.PAGE_HTML)  # 다른 형식의 주석은 없다


def test_preview_picture_is_a_small_jpeg_made_in_memory() -> None:
    big = np.zeros((720, 1280, 3), dtype=np.uint8)
    big[:, :640] = (255, 0, 0)  # 왼쪽 절반이 빨강(RGB)
    jpeg = live.encode_preview(big)
    assert jpeg[:3] == b"\xff\xd8\xff" and jpeg[-2:] == b"\xff\xd9"
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR
    assert decoded.shape == (270, 480, 3)  # 긴 변이 480
    assert decoded[100, 100, 2] > 200 and decoded[100, 100, 0] < 50  # 빨강은 빨강으로 남는다(RGB를 BGR로 바꿔 넣었다)
    assert decoded[100, 400].max() < 50
    # 세로로 긴 그림은 높이가 480이 된다. 작은 그림은 키우지 않는다.
    tall_jpeg = live.encode_preview(np.zeros((1280, 720, 3), np.uint8))
    tall = cv2.imdecode(np.frombuffer(tall_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert tall.shape == (480, 270, 3)
    small = cv2.imdecode(np.frombuffer(live.encode_preview(picture(0)), np.uint8), cv2.IMREAD_COLOR)
    assert small.shape == (H, W, 3)


# ---------------------------------------------------------------------------
# (2) 동등성: 같은 RawFrame이 실시간 경로와 데모의 run_pipeline에서 같은 결과를 낸다
# ---------------------------------------------------------------------------


def collect(session: live.LiveSession, frames: Sequence[RawFrame]) -> SimpleNamespace:
    """프레임을 `process_raw`에 차례로 넣고 상태가 알린 이벤트, 경고, 모델 출력을 모은다."""
    found = SimpleNamespace(events=[], alerts=[], probs=[], statuses=[])
    for frame in frames:
        status = session.process_raw(frame)
        found.events.extend(status.new_events)
        found.alerts.extend(status.new_alerts)
        found.probs.extend(status.new_probs)
        found.statuses.append(status)
    return found


@pytest.mark.parametrize(("speed_kmh", "fps"), [(None, 29.97), (30.0, 30.0)])
def test_live_session_gives_the_same_events_alerts_and_model_outputs_as_the_demo(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, model_fn: Any, speed_kmh: float | None, fps: float
) -> None:
    frames = five_minute_session(pipeline_spec, fps)
    frames.insert(4_000, frames[3_990])  # 시각이 되돌아간 프레임 하나: 두 길 모두 버린다
    run = demo.run_pipeline(frames, pipeline_spec, rules_config, model_fn, speed_kmh)

    session = live.LiveSession(pipeline_spec, rules_config, model_fn, speed_kmh)
    found = collect(session, frames)
    assert tuple(found.events) == run.events
    assert tuple(found.alerts) == run.alerts
    assert tuple(found.probs) == run.probs
    # 시험이 비어 있지 않다: 각본의 이벤트, 세 종류의 경고(속도가 없을 때), 모델 출력이 있다
    assert len(run.events) >= 10 and len(run.probs) == 61
    if fps != 30.0:
        # 29.97 fps에서는 스텝을 발행시키는 프레임의 시각이 그 스텝의 끝보다 뒤다: 모델 출력에 적는 두 시각이 구별된다
        # (초마다 프레임이 정확히 놓이는 30 fps 각본에서는 61개 모두 두 값이 같아서, 둘을 바꿔 적어도 드러나지 않는다)
        assert sum(point.t_ms != point.t_end_ms for point in run.probs) >= 50
        assert all(point.t_ms >= point.t_end_ms for point in run.probs)
    kinds = {record.kind for record in run.alerts}
    if speed_kmh is None:
        assert {demo.ALERT_IMMEDIATE, demo.ALERT_REPEAT} <= kinds
    else:  # 30 km/h: 졸음 경고(50 km/h 미만에서 억제)만 막힌다
        without_gate = demo.run_pipeline(frames, pipeline_spec, rules_config, model_fn, None)
        assert run.events == tuple(
            dataclasses.replace(record, alert=record.alert and record.event.type != "drowsy")
            for record in without_gate.events
        )
        assert len(run.alerts) < len(without_gate.alerts)

    # 마지막 상태는 데모가 모은 마지막 프레임의 값과 같다
    last = found.statuses[-1]
    assert (last.t_ms, last.frames) == (run.t_ms[-1], run.counters["frames_accepted"])
    assert last.session_ms == run.duration_ms
    assert (last.face, last.closure, last.closure_valid, last.provisional) == (
        run.face[-1],
        run.closure[-1],
        run.closure_valid[-1],
        run.provisional[-1],
    )
    assert last.gaze_zone == run.gaze_zone[-1]
    assert (last.baseline_status, last.baseline_ms) == (run.baseline.status, run.baseline.t_complete_ms)
    assert (last.events, last.alerts, last.model_outputs) == (len(run.events), len(run.alerts), len(run.probs))
    assert session.skipped == run.counters["dropped_frames"] == 1
    assert session.face_frames == sum(run.face)
    # 프레임별 값도 같다(버린 프레임의 상태는 직전 값 그대로이고 새 이벤트가 없다)
    dropped_at = 4_000
    assert found.statuses[dropped_at].new_events == () and found.statuses[dropped_at].frames == dropped_at
    kept = found.statuses[:dropped_at] + found.statuses[dropped_at + 1 :]
    assert [status.closure for status in kept] == list(run.closure)
    assert [status.gaze_zone for status in kept] == list(run.gaze_zone)
    assert [status.provisional for status in kept] == list(run.provisional)


def test_baseline_progress_follows_the_estimator(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    # 얼굴이 없는 세션: 마감이 60초에서 30초씩 180초까지 늘고, 180초에 전역 상수로 대체된다
    frames = [synth.no_face_frame(1 + 100 * index, 640, 480) for index in range(1_900)]
    session = live.LiveSession(pipeline_spec, rules_config, None)
    statuses = [session.process_raw(frame) for frame in frames]
    by_session_ms = {status.session_ms: status for status in statuses}
    assert session.status is statuses[-1]
    assert (by_session_ms[0].baseline_status, by_session_ms[0].baseline_ms) == (None, 60_000)
    assert by_session_ms[59_900].baseline_ms == 60_000
    assert by_session_ms[60_000].baseline_ms == 90_000 and by_session_ms[89_900].baseline_ms == 90_000
    assert by_session_ms[90_000].baseline_ms == 120_000 and by_session_ms[150_000].baseline_ms == 180_000
    assert (by_session_ms[179_900].baseline_status, by_session_ms[179_900].baseline_ms) == (None, 180_000)
    done = by_session_ms[180_000]
    assert (done.baseline_status, done.baseline_ms, done.provisional) == (BASELINE_FALLBACK_GLOBAL, 180_000, False)
    assert all(status.provisional for status in statuses if status.session_ms < 180_000)
    # 완성 전에는 마감이 언제나 세션 시간보다 뒤다
    assert all(status.baseline_ms > status.session_ms for status in statuses if status.baseline_status is None)

    # 얼굴이 있는 세션은 60초에 완성된다
    session = live.LiveSession(pipeline_spec, rules_config, None)
    statuses = [session.process_raw(frame) for frame in scripted_frames(pipeline_spec, 61_000)]
    assert statuses[0].new_events[0].event.type == EVENT_CALIBRATING
    assert [status.baseline_status for status in statuses if status.session_ms < 60_000] == [None] * 1800
    assert (statuses[-1].baseline_status, statuses[-1].baseline_ms) == (BASELINE_OK, 60_000)


def test_live_session_drops_a_frame_that_does_not_increase_before_mediapipe(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, scripted_landmarker: SimpleNamespace
) -> None:
    landmarker = extract.make_landmarker(Path("unused"), pipeline_spec)
    session = live.LiveSession(pipeline_spec, rules_config, None, None, landmarker)
    assert session.status.frames == 0 and session.status.baseline_ms == 60_000
    first = session.process(live_frame(1))
    assert (first.frames, first.t_ms, first.face) == (1, 1, True) and len(first.new_events) == 1
    assert session.process(live_frame(34)).frames == 2
    # 같은 시각, 되돌아간 시각, 0 이하의 시각: 가짜 랜드마커에 가지 않는다(가면 그 안의 단언이 실패한다)
    for t_ms in (34, 20, 0, -5):
        again = session.process(live_frame(t_ms))
        assert (again.frames, again.t_ms, again.new_events) == (2, 34, ())
    assert session.skipped == 4 and scripted_landmarker.seen == [1, 34]
    assert session.detect(live_frame(34)) is None and session.skipped == 5
    assert session.process(live_frame(68)).frames == 3 and scripted_landmarker.seen == [1, 34, 68]

    session.close()
    assert landmarker.closed
    with pytest.raises(live.LiveError, match="without a landmarker"):
        live.LiveSession(pipeline_spec, rules_config, None).process(live_frame(1))


# ---------------------------------------------------------------------------
# (2) 처리 루프와 프레임 표: 저장한 표를 다시 돌리면 실시간 세션이 알린 이벤트가 그대로 나온다
# ---------------------------------------------------------------------------


def run_loop(
    tmp_path: Path,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    model_fn: Any,
    server: live.StatusServer,
    reader: Any,
    stop: threading.Event | None = None,
    speed_kmh: float | None = None,
    **loop_args: Any,
) -> SimpleNamespace:
    """`main`이 하는 대로 처리 루프를 `extract.write_raw_frames`에 넘겨 돌린다. 랜드마커는 `extract`의 것(가짜)을 쓴다.

    loop_args는 `LiveLoop`에 그대로 넘긴다(`counts`, `clock`, `stop_reason`, `say`).
    """
    said: list[str] = []
    session = live.LiveSession(
        pipeline_spec, rules_config, model_fn, speed_kmh, extract.make_landmarker(Path("unused"), pipeline_spec)
    )
    metadata = live.table_metadata("test://frames", 0, pipeline_spec)
    counts = loop_args.pop("counts", live.StreamCounts())
    say = loop_args.pop("say", said.append)
    stop = stop or threading.Event()
    loop = live.LiveLoop(reader, session, server, counts, stop, metadata, ["info line"], say=say, **loop_args)
    parquet = tmp_path / "out" / "session.parquet"
    try:
        rows = extract.write_raw_frames(loop.frames(), parquet, pipeline_spec, metadata)
    finally:
        session.close()
    return SimpleNamespace(loop=loop, session=session, said=said, parquet=parquet, rows=rows, metadata=metadata)


def detect_from(frames: Sequence[RawFrame]) -> Any:
    """`extract.detect_frame` 자리에 넣는 함수: 호출 순서대로 준 RawFrame을 돌려준다(t_ms가 같은지 확인한다)."""
    pending = iter(frames)

    def detect_frame(landmarker: Any, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        raw = next(pending)
        assert raw.t_ms == t_ms
        return raw

    return detect_frame


def record_published(server: live.StatusServer, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """서버의 `publish`가 받은 상태를 차례로 모으는 목록(서버는 그대로 상태를 낸다)."""
    published: list[dict[str, Any]] = []
    real_publish = server.publish

    def publish(payload: Any) -> None:
        published.append(json.loads(json.dumps(payload)))
        real_publish(payload)

    monkeypatch.setattr(server, "publish", publish)
    return published


def stepping_clock(step_s: float = 0.02) -> Any:
    """읽을 때마다 step_s씩 가는 가짜 시계."""
    reads = [0]

    def clock() -> float:
        reads[0] += 1
        return 100.0 + step_s * reads[0]

    clock.reads = reads  # type: ignore[attr-defined]
    return clock


def test_replaying_the_saved_frame_table_gives_the_events_the_live_session_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    model_fn: Any,
    model_v0: Path,
    experiment_config: Any,
    repo_root: Path,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    raw_frames = five_minute_session(pipeline_spec)
    # 얼굴이 검출되지 않은 프레임도 섞는다(200초부터 2초). 표에는 얼굴 없는 프레임도 빠짐없이 들어가야 한다.
    no_face = [index for index, raw in enumerate(raw_frames) if 200_000 <= raw.t_ms < 202_000]
    for index in no_face:
        raw_frames[index] = synth.no_face_frame(raw_frames[index].t_ms, 640, 480)
    assert len(no_face) == 60
    monkeypatch.setattr(extract, "detect_frame", detect_from(raw_frames))
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    reader = ListReader([live.LiveFrame(raw.t_ms, blank) for raw in raw_frames])
    counts = live.StreamCounts(decoded=len(raw_frames))  # 읽기 스레드가 디코딩한 수(가짜 읽기 스레드 대신 시험이 정한다)
    done = run_loop(tmp_path, pipeline_spec, rules_config, model_fn, status_server, reader, counts=counts)
    assert done.rows == done.loop.processed == len(raw_frames) and done.loop.failure is None
    assert done.loop.reason == live.END_OF_STREAM
    assert reader.stopped  # 계수를 읽기 전에 읽기 스레드를 멈췄다

    # 표에는 파이프라인이 받은 프레임이 그대로, 빠짐없이 들어 있다
    saved = list(extract.read_raw_frames(done.parquet))
    assert saved == raw_frames and sum(not raw.face for raw in saved) == 60

    # 실시간 세션이 터미널에 알린 이벤트와 경고
    live_events = printed(done.said, "event ")
    live_alerts = printed(done.said, "alert ")
    # 저장한 표를 데모의 길로 다시 돌린다
    run = demo.run_pipeline(saved, pipeline_spec, rules_config, model_fn, None)
    origin = run.t_ms[0]
    assert live_events == [live.format_event_line(record, origin) for record in run.events]
    assert live_alerts == [live.format_alert_line(record, origin) for record in run.alerts]
    assert len(live_events) >= 10 and len(live_alerts) >= 5
    # 줄의 순서도 프레임 순서다: 한 프레임의 이벤트가 그 프레임의 경고보다 먼저 나온다
    first_alert = done.said.index(live_alerts[0])
    assert done.said[first_alert - 1].split(":")[0] == live_alerts[0].split(":")[0].replace("alert", "event")

    # `python -m drowsy.demo --raw`가 그 표로 쓰는 이벤트 JSON의 이벤트와 경고가, 실시간 세션이 상태 JSON에 낸 것과 값까지
    # 같다(이 세션의 이벤트 13개와 경고는 최근 목록의 상한 20개 안에 모두 들어 있다)
    paths = dataclasses.replace(
        experiment_config.paths,
        feature_spec=repo_root / "config" / "feature_spec.json",
        rules=repo_root / "config" / "rules.yaml",
        export_dir=repo_root / "export",
    )
    cfg = dataclasses.replace(experiment_config, paths=paths)
    _, json_path = demo.run_demo_raw(done.parquet, model_v0, tmp_path / "replay", cfg=cfg)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(report["events"]) == len(live_events) == 13 <= live.RECENT_ITEMS
    assert len(report["alerts"]) == len(live_alerts) <= live.RECENT_ITEMS
    payload = status_json(status_server)
    reported_events = [{k: v for k, v in item.items() if k not in ("session_s", "line")} for item in payload["events"]]
    reported_alerts = [{k: v for k, v in item.items() if k not in ("session_s", "line")} for item in payload["alerts"]]
    assert reported_events == report["events"]
    assert reported_alerts == report["alerts"]
    assert [item["line"] for item in payload["events"]] == live_events
    assert [item["line"] for item in payload["alerts"]] == live_alerts
    assert [item["session_s"] for item in payload["events"]] == [
        (item["t_ms"] - report["session"]["first_t_ms"]) / 1000.0 for item in report["events"]
    ]
    assert report["counters"]["frames_accepted"] == len(raw_frames)
    assert report["probabilities"]["n_outputs"] == done.session.status.model_outputs == 61
    assert report["source"]["extract_source"] == "test://frames"
    assert report["source"]["rotation_note"] == (
        "rotation: 0 (manual value recorded by the extraction; the automatic check was not run)"
    )

    # 끝난 뒤의 상태 JSON: 끝난 이유, 마지막 상태 줄, 시작 안내. 기준선이 완성됐으므로 기준선 안내는 없다.
    assert (payload["phase"], payload["end_reason"], payload["info"]) == ("ended", live.END_OF_STREAM, ["info line"])
    assert (payload["source_open"], payload["preview"], payload["hint"]) == (True, True, "")
    assert payload["line"] == live.format_status_line(done.loop.status()) and payload["frames"] == len(raw_frames)
    assert payload["baseline"] == {"done": True, "status": "ok", "elapsed_s": 299, "target_s": 60}

    # 표의 메타데이터: 시작할 때 아는 값과, 세션이 끝나야 아는 값(write_raw_frames가 프레임을 다 받은 뒤에 읽는다)
    metadata = extract.read_extract_metadata(done.parquet)
    assert metadata["source"] == "test://frames" and metadata["live"] is True and metadata["drop_sim"] is None
    assert metadata["model_sha256"] == pipeline_spec.mediapipe.model_sha256
    assert metadata["mediapipe_version"] == "0.0-fake"
    assert (metadata["rotation"], metadata["rotation_override"], metadata["rotation_given"]) == (0, 0, True)
    assert (metadata["n_decoded"], metadata["n_skipped"]) == (len(raw_frames), 0)
    assert (metadata["n_dropped"], metadata["n_left"], metadata["n_failed"], metadata["n_decode_errors"]) == (0, 0, 0, 0)
    assert (metadata["n_processed"], metadata["n_face"]) == (len(raw_frames), len(raw_frames) - 60)
    assert (metadata["width"], metadata["height"], metadata["arrival_clock"]) == (640, 480, False)
    assert metadata["end_reason"] == live.END_OF_STREAM
    # `--rotation`을 주지 않은 세션은 그렇게 적는다(각도는 기본값 0)
    assert live.table_metadata("x", 0, pipeline_spec, rotation_given=False)["rotation_given"] is False

    # 처음의 두 줄과 끝의 두 줄
    assert done.said[0] == "first frame: 640x480; the session clock starts here"
    assert done.said[1] == live.baseline_note(pipeline_spec)
    assert done.said[2].startswith("event 0.0 s: calibrating onset")
    assert done.said[-2] == f"session ended: {live.END_OF_STREAM}"
    assert done.said[-1] == (
        f"frames: decoded {len(raw_frames)}, skipped 0 (timestamp), dropped 0 (processing was behind), left 0 "
        f"(unprocessed at the end), failed 0 (processing error), processed {len(raw_frames)}, with a face "
        f"{len(raw_frames) - 60}"
    )


def short_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: PipelineSpec, n_frames: int
) -> tuple[list[RawFrame], ListReader]:
    """각본 프레임 n_frames장과, 그것을 차례로 돌려주는 가짜 추론과 읽기 스레드."""
    raw_frames = scripted_frames(spec, 4_000)[:n_frames]
    monkeypatch.setattr(extract, "detect_frame", detect_from(raw_frames))
    return raw_frames, ListReader([live_frame(raw.t_ms) for raw in raw_frames])


def test_status_lines_come_once_per_second_of_wall_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 90)
    # 가짜 시계: 루프가 시계를 읽을 때마다 20 ms가 간다. 루프는 프레임마다 시계를 두 번이나 세 번 읽는다
    # (미리보기를 낼 때 한 번 더). 그래서 프레임 하나에 40~60 ms, 초당 16.7~25프레임이다.
    clock = stepping_clock(0.02)
    said: list[str] = []
    landmarker = extract.make_landmarker(Path("unused"), pipeline_spec)
    session = live.LiveSession(pipeline_spec, rules_config, None, None, landmarker)
    loop = live.LiveLoop(
        reader, session, status_server, live.StreamCounts(), threading.Event(), {}, say=said.append, clock=clock
    )
    assert len(list(loop.frames())) == 90
    elapsed_s = 0.02 * clock.reads[0]
    assert 90 * 0.04 <= elapsed_s <= 90 * 0.06 + 0.1
    lines = [line for line in printed(said, "session ") if not line.startswith("session ended")]
    # 1초(가짜 시계)에 한 줄이다. 첫 줄은 첫 프레임에서 1초 뒤에 나온다.
    assert int(elapsed_s) - 1 <= len(lines) <= int(elapsed_s)
    sessions = [float(line.split(", ")[0].split()[1]) for line in lines]
    assert sessions == sorted(sessions) and len(set(sessions)) == len(sessions)
    for line in lines:
        words = line.split(", ")
        assert words[5].startswith("received ") and words[6].startswith("processed ")
        received, processed = float(words[5].split()[1]), float(words[6].split()[1])
        # 받은 프레임과 처리한 프레임이 같다(하나도 버리지 않았다). 값은 그 1초의 프레임 수 / 지난 시간이다.
        assert received == processed and 16.0 <= received <= 26.0
        assert words[7:] == ["dropped 0", "skipped 0"]
    assert 16.0 <= loop.status().fps_processed <= 26.0 and loop.status().fps_received == loop.status().fps_processed
    # 첫 프레임 뒤에 기준선 안내가 한 번 나온다(첫 60초에 무엇을 해야 하는지)
    assert said[0].startswith("first frame:") and said[1] == live.baseline_note(pipeline_spec)
    assert said[2].startswith("event 0.0 s: calibrating onset") and said.count(said[1]) == 1


def test_loop_reports_dropped_skipped_and_left_frames_in_every_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    """버린 프레임의 수가 상태 줄, 상태 JSON, 끝의 줄, 표의 메타데이터에 같은 값으로 나온다."""
    raw_frames = scripted_frames(pipeline_spec, 4_000)[:30]
    monkeypatch.setattr(extract, "detect_frame", detect_from(raw_frames))
    frames = [live_frame(raw.t_ms) for raw in raw_frames]
    frames.insert(10, frames[9])  # 같은 시각의 프레임 하나: 세션이 MediaPipe에 넣기 전에 버린다
    reader = ListReader(frames, dropped=7)  # 읽기 스레드가 밀려서 7장을 버렸다고 알린다
    counts = live.StreamCounts(decoded=45, skipped=3)  # 스트림에서 45장을 디코딩했고 3장은 타임스탬프 때문에 버렸다
    published = record_published(status_server, monkeypatch)
    done = run_loop(
        tmp_path, pipeline_spec, rules_config, None, status_server, reader, counts=counts, clock=stepping_clock(0.02)
    )
    assert done.rows == 30
    status = done.loop.status()
    assert (status.dropped, status.skipped) == (7, 4)  # 스트림의 3 + 세션의 1
    status_lines = [line for line in printed(done.said, "session ") if not line.startswith("session ended")]
    assert status_lines and all(line.endswith(", dropped 7, skipped 4") for line in status_lines)
    running = [payload for payload in published if payload["phase"] == "running"]
    assert (running[-1]["dropped"], running[-1]["skipped"]) == (7, 4)
    assert (running[0]["dropped"], running[0]["skipped"]) == (7, 3)  # 같은 시각의 프레임이 오기 전
    ended = status_json(status_server)
    assert (ended["phase"], ended["dropped"], ended["skipped"]) == ("ended", 7, 4)
    # 끝의 줄: 디코딩한 45장 = 타임스탬프 4 + 밀려서 7 + 남은 4 + 처리 오류 0 + 처리한 30
    assert done.said[-1] == (
        "frames: decoded 45, skipped 4 (timestamp), dropped 7 (processing was behind), left 4 (unprocessed at the "
        "end), failed 0 (processing error), processed 30, with a face 30"
    )
    assert_frame_metadata(done.parquet, decoded=45, skipped=4, dropped=7, left=4, failed=0, processed=30, face=30)


class FastSourceReader:
    """`LatestFrameReader` 자리에 넣는 것: 처리 루프가 하나를 가져갈 때마다 그 앞의 하나는 밀려서 버렸다고 알린다.

    그래서 받은 프레임 수가 처리한 프레임 수의 두 배다(처리가 스트림의 절반 속도밖에 못 따라가는 경우).
    """

    def __init__(self, frames: Sequence[live.LiveFrame]) -> None:
        self._inner = ListReader(frames)
        self.error: BaseException | None = None

    @property
    def received(self) -> int:
        return 2 * self._inner.received

    @property
    def dropped(self) -> int:
        return self._inner.received

    @property
    def finished(self) -> bool:
        return self._inner.finished

    def get(self, timeout_s: float) -> live.LiveFrame | None:
        return self._inner.get(timeout_s)

    def stop(self) -> bool:
        return self._inner.stop()


def test_status_tells_the_received_and_the_processed_frame_rates_apart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    """받은 프레임과 처리한 프레임의 초당 수는 따로 센다: 받은 것이 처리한 것의 두 배인 스트림에서 두 값이 그만큼 다르다."""
    raw_frames = scripted_frames(pipeline_spec, 4_000)[:60]
    monkeypatch.setattr(extract, "detect_frame", detect_from(raw_frames))
    reader = FastSourceReader([live_frame(raw.t_ms) for raw in raw_frames])
    published = record_published(status_server, monkeypatch)
    counts = live.StreamCounts(decoded=120)
    done = run_loop(
        tmp_path, pipeline_spec, rules_config, None, status_server, reader, counts=counts, clock=stepping_clock(0.02)
    )
    assert done.rows == 60
    status = done.loop.status()
    # 첫 프레임 뒤로 받은 수는 처리한 수의 꼭 두 배로 늘었다: 초당 수도 두 배다(같은 시간으로 나눈다)
    assert status.fps_processed > 5.0 and status.fps_received == pytest.approx(2.0 * status.fps_processed)
    lines = [line for line in printed(done.said, "session ") if not line.startswith("session ended")]
    assert len(lines) >= 2
    for line in lines:
        words = line.split(", ")
        received, processed = float(words[5].split()[1]), float(words[6].split()[1])
        assert received == pytest.approx(2.0 * processed, abs=0.16) and processed > 5.0, line
    # 끝의 상태가 든 두 값은 마지막 상태 줄의 값이다(초당 수는 상태 줄을 낼 때만 다시 구한다)
    assert f", received {status.fps_received:.1f} fps, processed {status.fps_processed:.1f} fps, " in lines[-1]
    # 상태 JSON(웹 페이지)도 같은 두 값이다
    last = [payload for payload in published if payload["phase"] == "running"][-1]
    assert (last["fps_received"], last["fps_processed"]) == (status.fps_received, status.fps_processed)
    # 끝의 계수: 120 = 밀려서 60 + 처리한 60
    assert done.said[-1] == frames_line(120, 0, 60, 0, 0, 60, 60)


def test_loop_sends_the_running_state_and_the_picture_to_the_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    """세션이 도는 동안 웹 페이지가 받는 것: 프레임마다 새 상태, 그리고 방금 처리한 프레임의 미리보기."""
    raw_frames = scripted_frames(pipeline_spec, 4_000)[:30]
    monkeypatch.setattr(extract, "detect_frame", detect_from(raw_frames))
    shown = picture(7)  # 모든 프레임이 같은 그림이다(어느 프레임의 미리보기든 이 그림이어야 한다)
    reader = ListReader([live.LiveFrame(raw.t_ms, shown) for raw in raw_frames])
    published = record_published(status_server, monkeypatch)
    encoded: list[np.ndarray] = []
    real_encode = live.encode_preview
    monkeypatch.setattr(live, "encode_preview", lambda rgb: (encoded.append(rgb), real_encode(rgb))[1])
    done = run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, reader, clock=stepping_clock(0.02))
    assert done.rows == 30

    # 상태: 처음에는 기다림, 프레임마다 도는 중(프레임 수가 하나씩 는다), 끝에 끝남
    # (서른 프레임 뒤에 스트림의 끝을 본 한 바퀴가 더 있다)
    phases = [payload["phase"] for payload in published]
    assert phases == ["waiting"] + ["running"] * 31 + ["ended"]
    assert [payload["frames"] for payload in published[1:32]] == [*range(1, 31), 30]
    assert all(payload["source_open"] is True and payload["preview"] is True for payload in published)
    assert all(payload["info"] == ["info line"] for payload in published)
    assert "frames" not in published[0] and published[0]["hint"] == ""
    # 기준선이 완성되기 전에는 도는 동안 기준선 안내가 함께 간다
    assert all(payload["hint"] == live.baseline_note(pipeline_spec) for payload in published[1:32])
    assert published[-1]["hint"] == "" and published[-1]["end_reason"] == live.END_OF_STREAM
    assert published[5]["line"].startswith("session 0.") and published[5]["face"] is True

    # 미리보기: 0.1초(가짜 시계)에 한 번까지, 방금 처리한 프레임 그대로
    assert 5 <= len(encoded) <= 16 and all(rgb is shown for rgb in encoded)
    sequence, jpeg = status_server.wait_preview(0)
    assert sequence == len(encoded) and jpeg is not None
    decoded = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert decoded.shape == shown.shape
    assert float(np.abs(decoded.astype(np.int16) - shown.astype(np.int16)).mean()) < 6.0  # JPEG의 손실만큼만 다르다


def test_loop_makes_no_preview_when_it_is_off_and_goes_on_when_the_preview_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    calls = [0]
    real_encode = live.encode_preview

    def encode(rgb: np.ndarray) -> bytes:
        calls[0] += 1
        if calls[0] == 2:
            raise cv2.error("Insufficient memory")  # cv2.error는 Exception의 바로 아래 종류다
        return real_encode(rgb)

    monkeypatch.setattr(live, "encode_preview", encode)

    # 미리보기를 끄면(--no-preview) 그림을 만들지도 않는다
    quiet = live.StatusServer(0, preview=False)
    try:
        raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 30)
        done = run_loop(tmp_path / "off", pipeline_spec, rules_config, None, quiet, reader, clock=stepping_clock(0.02))
        assert done.rows == 30 and calls[0] == 0
        assert status_json(quiet)["preview"] is False
    finally:
        quiet.stop()

    # 미리보기를 만들다 오류가 나면 미리보기만 끄고 세션은 끝까지 간다
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 30)
    published = record_published(status_server, monkeypatch)
    clock = stepping_clock(0.02)
    done = run_loop(tmp_path / "broken", pipeline_spec, rules_config, None, status_server, reader, clock=clock)
    assert done.rows == 30 and done.loop.failure is None and done.loop.reason == live.END_OF_STREAM
    assert calls[0] == 2  # 그 뒤로는 다시 만들지 않는다
    assert printed(done.said, "preview: ") == [
        "preview: stopped (error: Insufficient memory); the session continues without the camera preview"
    ]
    assert published[1]["preview"] is True and published[-1]["preview"] is False
    assert list(extract.read_raw_frames(done.parquet)) == raw_frames


def test_loop_goes_on_when_the_page_cannot_be_updated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 30)
    calls = [0]

    def publish(payload: Any) -> None:
        calls[0] += 1
        if calls[0] >= 5:
            raise ValueError("Out of range float values are not JSON compliant")

    monkeypatch.setattr(status_server, "publish", publish)
    done = run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, reader)
    assert done.rows == 30 and done.loop.failure is None
    assert calls[0] == 33  # 처음 한 번, 서른 프레임과 스트림의 끝을 본 한 바퀴, 끝에 한 번: 실패한 뒤에도 계속 냈다
    assert printed(done.said, "page: ") == [
        "page: the status could not be updated (Out of range float values are not JSON compliant); the session "
        "continues"
    ]


def test_a_frame_the_pipeline_accepted_is_in_the_table_even_when_its_lines_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    """파이프라인이 받아들인 프레임의 줄을 내다 오류가 나도 그 프레임은 표에 들어간다(표와 파이프라인이 같은 프레임을 갖는다)."""
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 30)
    said: list[str] = []

    def say(text: str) -> None:
        if text.startswith("event ") and len(said) >= 3:
            raise UnicodeEncodeError("ascii", text, 0, 1, "a terminal that cannot show the line")
        said.append(text)

    # 스무째 프레임이 이벤트를 하나 내게 한다(이 각본의 앞부분에는 첫 프레임의 calibrating 말고 이벤트가 없다).
    # 그 줄을 내는 `say`가 실패한다(첫 프레임의 세 줄은 지나간 뒤다).
    real_process = live.LiveSession.process_raw
    event = DriverStateEvent(EVENT_MICROSLEEP, 3, 1, 1_000, 1.0, "closure", PHASE_ONSET, True)
    extra = demo.EventRecord(raw_frames[19].t_ms, event, False)

    def process_raw(self: live.LiveSession, raw: RawFrame) -> live.LiveStatus:
        status = real_process(self, raw)
        return dataclasses.replace(status, new_events=(extra,)) if raw is raw_frames[19] else status

    monkeypatch.setattr(live.LiveSession, "process_raw", process_raw)
    counts = live.StreamCounts(decoded=30)
    done = run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, reader, say=say, counts=counts)
    assert isinstance(done.loop.failure, UnicodeEncodeError)
    assert done.loop.reason.startswith("processing failed (") and "cannot show the line" in done.loop.reason
    # 스무째 프레임까지 파이프라인이 받았고, 표에도 스무 프레임이 있다. 그 프레임은 처리 오류가 아니라 처리한 프레임이다.
    assert done.session.status.frames == 20 and done.rows == done.loop.processed == 20 and done.loop.failed == 0
    assert list(extract.read_raw_frames(done.parquet)) == raw_frames[:20]
    assert said[-1] == frames_line(30, 0, 0, 10, 0, 20, 20)
    assert_frame_metadata(done.parquet, decoded=30, left=10, failed=0, processed=20)


def test_loop_says_what_to_check_when_no_frame_comes_after_the_source_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    """키프레임 간격이 긴 보내는 쪽: 접속은 됐는데 프레임이 없다. 3초 뒤에 한 번, 무엇을 볼지 알린다."""
    assert live.FIRST_FRAME_HINT_S == 3.0
    raw_frames, _ = short_run(tmp_path, monkeypatch, pipeline_spec, 5)

    class LateReader(ListReader):
        """처음 12번의 `get`은 프레임 없이 돌아온다(가짜 시계로 6초 동안 프레임이 없다)."""

        quiet = 12

        @property
        def finished(self) -> bool:
            return self.quiet <= 0 and self._next >= len(self._frames)

        def get(self, timeout_s: float) -> live.LiveFrame | None:
            if self.quiet > 0:
                self.quiet -= 1
                return None
            return super().get(timeout_s)

    reader = LateReader([live_frame(raw.t_ms) for raw in raw_frames])
    published = record_published(status_server, monkeypatch)
    done = run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, reader, clock=stepping_clock(0.5))
    assert done.rows == 5
    hint = (
        "no frame yet, 3 s after the source opened. A stream starts at a key frame: if this lasts, give the sender "
        "a short key-frame interval (ffmpeg: -g 30)."
    )
    assert done.said[0] == hint and done.said.count(hint) == 1
    assert done.said[1] == "first frame: 640x480; the session clock starts here"
    assert [payload["phase"] for payload in published[:13]] == ["waiting"] * 13
    assert all(payload["source_open"] is True for payload in published)


def test_loop_finishes_the_frame_in_hand_and_starts_no_new_frame_after_a_stop_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    # (1) 스물다섯째 프레임을 처리하는 도중에 멈추라는 요청(Ctrl+C)이 온다: 그 프레임은 끝까지 처리해 표에 넣는다
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 60)
    stop = threading.Event()
    pending = iter(raw_frames)

    def detect_frame(landmarker: Any, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        raw = next(pending)
        if raw is raw_frames[24]:
            stop.set()
        return raw

    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    counts = live.StreamCounts(decoded=60)
    done = run_loop(tmp_path / "a", pipeline_spec, rules_config, None, status_server, reader, stop, counts=counts)
    assert done.loop.reason == live.END_INTERRUPTED == "interrupted (Ctrl+C)"
    assert done.rows == done.loop.processed == 25 and done.loop.failure is None
    assert list(extract.read_raw_frames(done.parquet)) == raw_frames[:25]
    assert extract.read_extract_metadata(done.parquet)["end_reason"] == "interrupted (Ctrl+C)"
    assert done.said[-3:-1] == ["stop requested: ending the session", "session ended: interrupted (Ctrl+C)"]
    # 끝의 계수: 디코딩한 60 = 남은 35(읽기 스레드가 아직 넘기지 않은 것) + 처리한 25
    assert done.said[-1] == frames_line(60, 0, 0, 35, 0, 25, 25)
    assert_frame_metadata(done.parquet, decoded=60, left=35, failed=0, processed=25)

    # (2) 다음 프레임을 기다리는 동안 요청이 오고 바로 뒤에 프레임이 도착한다: 그 프레임은 시작하지 않는다.
    #     끝난 이유는 `stop_reason`이 정한다(Ctrl+C가 아닌 신호).
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 60)
    stop = threading.Event()
    real_get = reader.get

    def get(timeout_s: float) -> live.LiveFrame | None:
        frame = real_get(timeout_s)
        if frame is not None and frame.t_ms == raw_frames[24].t_ms:
            stop.set()  # 스물다섯째 프레임을 넘기는 순간
        return frame

    reader.get = get  # type: ignore[method-assign]
    reason = live.END_SIGNAL.format(name="SIGTERM")
    counts = live.StreamCounts(decoded=60)
    done = run_loop(
        tmp_path / "b",
        pipeline_spec,
        rules_config,
        None,
        status_server,
        reader,
        stop,
        stop_reason=lambda: reason,
        counts=counts,
    )
    assert done.loop.reason == reason == "stopped by a signal (SIGTERM)"
    assert done.rows == done.loop.processed == 24
    assert list(extract.read_raw_frames(done.parquet)) == raw_frames[:24]
    assert done.said[-2] == "session ended: stopped by a signal (SIGTERM)"
    # 받았지만 시작하지 않은 스물다섯째 프레임은 남은 프레임이다(처리 오류가 아니다): 60 = 남은 36 + 처리한 24
    assert done.said[-1] == frames_line(60, 0, 0, 36, 0, 24, 24)
    assert_frame_metadata(done.parquet, decoded=60, left=36, failed=0, processed=24)

    # (3) 첫 프레임이 오기 전의 요청: 바로 뒤에 프레임이 도착해도 아무것도 쓰지 않는다
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 60)
    stop = threading.Event()
    real_get = reader.get

    def first_get(timeout_s: float) -> live.LiveFrame | None:
        stop.set()
        return real_get(timeout_s)

    reader.get = first_get  # type: ignore[method-assign]
    with pytest.raises(live.LiveError, match=r"before any frame was processed \(interrupted \(Ctrl\+C\)\)"):
        run_loop(tmp_path / "c", pipeline_spec, rules_config, None, status_server, reader, stop)
    assert not (tmp_path / "c").exists()


def test_status_json_keeps_only_the_most_recent_events_and_alerts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    assert live.RECENT_ITEMS == 20
    monkeypatch.setattr(live, "RECENT_ITEMS", 3)
    raw_frames = five_minute_session(pipeline_spec)
    monkeypatch.setattr(extract, "detect_frame", detect_from(raw_frames))
    reader = ListReader([live.LiveFrame(raw.t_ms, np.zeros((H, W, 3), dtype=np.uint8)) for raw in raw_frames])
    done = run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, reader)
    payload = status_json(status_server)
    # 터미널에는 모두 나오고, 상태 JSON에는 끝의 셋만 남는다(세션이 길어져도 목록이 늘지 않는다)
    assert len(printed(done.said, "event ")) == 13 and len(printed(done.said, "alert ")) > 3
    assert [item["line"] for item in payload["events"]] == printed(done.said, "event ")[-3:]
    assert [item["line"] for item in payload["alerts"]] == printed(done.said, "alert ")[-3:]


def test_loop_ends_at_the_frame_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    assert live.MAX_SESSION_FRAMES == 216_000  # 30 fps로 2시간
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 60)
    monkeypatch.setattr(live, "MAX_SESSION_FRAMES", 7)
    counts = live.StreamCounts(decoded=60)
    done = run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, reader, counts=counts)
    assert done.loop.reason == live.END_FRAME_LIMIT and "frame limit (216000 frames)" in live.END_FRAME_LIMIT
    assert done.rows == 7 and list(extract.read_raw_frames(done.parquet)) == raw_frames[:7]
    assert done.said[-2:] == [f"session ended: {live.END_FRAME_LIMIT}", frames_line(60, 0, 0, 53, 0, 7, 7)]
    assert_frame_metadata(done.parquet, decoded=60, left=53, failed=0, processed=7)


def test_a_session_without_any_frame_writes_nothing(
    tmp_path: Path,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
) -> None:
    with pytest.raises(live.LiveError, match="before any frame was processed") as caught:
        run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, ListReader([]))
    assert live.END_OF_STREAM in str(caught.value) and "nothing was written" in str(caught.value)
    assert not (tmp_path / "out").exists()
    payload = status_json(status_server)
    assert (payload["phase"], payload["end_reason"]) == ("ended", live.END_OF_STREAM) and "line" not in payload

    # 멈춘 스트림(읽기 시간 초과)과 그 밖의 읽기 오류는 끝난 이유에 적힌다
    stalled = av.error.ExitError(1414092869, "Immediate exit requested")
    with pytest.raises(live.LiveError, match="no data from the source for 10 s"):
        run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, ListReader([], stalled))
    with pytest.raises(live.LiveError, match=r"the source failed \(.*Connection reset"):
        broken = ListReader([], ConnectionResetError(104, "Connection reset by peer"))
        run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, broken)


@pytest.mark.parametrize(
    ("error", "text"),
    [
        (extract.ExtractError("the landmarker broke"), "the landmarker broke"),
        (TypeError("bad argument"), "TypeError: bad argument"),  # 아래 셋은 RuntimeError, ValueError, OSError가 아니다
        (KeyError("lm"), "KeyError: 'lm'"),
        (cv2.error("boom"), "error: boom"),
        (AssertionError(), "AssertionError"),
    ],
)
def test_a_processing_error_of_any_kind_ends_the_session_and_keeps_the_frames_so_far(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    scripted_landmarker: SimpleNamespace,
    status_server: live.StatusServer,
    error: Exception,
    text: str,
) -> None:
    raw_frames, reader = short_run(tmp_path, monkeypatch, pipeline_spec, 30)
    pending = iter(raw_frames[:10])

    def detect_frame(landmarker: Any, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        try:
            return next(pending)
        except StopIteration:
            raise error from None

    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    counts = live.StreamCounts(decoded=30)
    done = run_loop(tmp_path, pipeline_spec, rules_config, None, status_server, reader, counts=counts)
    assert done.loop.failure is error
    assert done.loop.reason == f"processing failed ({text})"
    assert done.rows == 10 and list(extract.read_raw_frames(done.parquet)) == raw_frames[:10]
    assert done.said[-2] == f"session ended: processing failed ({text})"
    assert status_json(status_server)["end_reason"] == f"processing failed ({text})"
    # 오류가 난 열한째 프레임은 "남은 프레임"이 아니라 처리 오류로 따로 센다: 30 = 남은 19 + 오류 1 + 처리한 10
    assert done.loop.failed == 1
    assert done.said[-1] == frames_line(30, 0, 0, 19, 1, 10, 10)
    assert_frame_metadata(done.parquet, decoded=30, left=19, failed=1, processed=10)


# ---------------------------------------------------------------------------
# (3)(4) 명령: 파일 소스와 TCP 왕복(가짜 랜드마커). 쓰는 것은 표, 그림, JSON 셋뿐이다
# ---------------------------------------------------------------------------


def check_session_files(out_dir: Path, lines: Sequence[str], pipeline_spec: PipelineSpec) -> dict[str, Any]:
    """명령이 끝난 뒤의 출력 폴더와 터미널 줄을 함께 확인하고 이벤트 JSON을 돌려준다."""
    parquet, png, events_json = out_files(out_dir)
    for path in (parquet, png, events_json):
        assert any(line.startswith(f"wrote {path} (") and line.endswith(" bytes)") for line in lines), path
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    counts = frame_counts(lines)

    # 표: 추출 결과와 같은 열만 있다(시각, 얼굴 여부, 크기, 랜드마크 44개의 x와 y, 블렌드셰이프 52개, 행렬 16개, 얼굴 폭).
    # 화소를 담을 열이 없다.
    schema = pq.read_schema(parquet)
    reference = out_dir.parent / "reference_schema.parquet"
    extract.write_raw_frames([synth.no_face_frame(1, W, H)], reference, pipeline_spec)
    assert schema.names == pq.read_schema(reference).names and len(schema.names) == 4 + 88 + 52 + 16 + 1
    assert {str(field.type) for field in schema} == {"int64", "bool", "int32", "double"}
    reference.unlink()
    table = pq.read_table(parquet)
    assert table.num_rows == counts["processed"] > 0
    t_ms = table.column("t_ms").to_pylist()
    assert t_ms[0] == 1 and all(later > earlier for earlier, later in zip(t_ms, t_ms[1:]))
    metadata = assert_frame_metadata(parquet, **{name: value for name, value in counts.items()})
    assert metadata["live"] is True
    assert metadata["n_face"] == counts["face"] == sum(table.column("face").to_pylist())
    # 디코딩한 프레임은 처리했거나, 타임스탬프 때문에 또는 밀려서 버렸거나, 끝날 때 남았거나(멈추라는 요청 뒤에 받은 하나,
    # 칸에 남은 하나, 읽기 스레드가 멈추라는 말을 듣기 직전에 디코딩한 하나까지), 처리하다 오류가 난 것이다(0이나 1)
    assert counts["decoded"] == sum(counts[name] for name in ("skipped", "dropped", "left", "failed", "processed"))
    assert 0 <= counts["left"] <= 3 and counts["failed"] in (0, 1)

    report = json.loads(events_json.read_text(encoding="utf-8"))
    assert report["note"] == demo.NOTE_RECORDING
    assert report["source"]["kind"] == demo.SOURCE_RAW and report["source"]["path"] == str(parquet)
    assert report["counters"]["frames_accepted"] == counts["processed"]
    # 터미널에 나온 이벤트와 경고가 이벤트 JSON에 그대로 있다
    origin = report["session"]["first_t_ms"]
    as_events = [
        demo.EventRecord(item["t_ms"], DriverStateEvent(**{k: item[k] for k in EVENT_KEYS}), item["alert"])
        for item in report["events"]
    ]
    as_alerts = [
        demo.AlertRecord(item["t_ms"], DriverStateEvent(**{k: item[k] for k in EVENT_KEYS}), item["kind"])
        for item in report["alerts"]
    ]
    assert printed(lines, "event ") == [live.format_event_line(record, origin) for record in as_events]
    assert printed(lines, "alert ") == [live.format_alert_line(record, origin) for record in as_alerts]
    # 표를 쓴 줄 다음에, 이어지는 데모 출력의 회전 줄이 이 세션에서 무엇을 뜻하는지 알린다. 데모의 요약 줄도 같은 수를 말한다
    wrote = next(index for index, line in enumerate(lines) if line.startswith(f"wrote {parquet} ("))
    given = "from --rotation" if metadata["rotation_given"] else "the default; --rotation was not given"
    assert lines[wrote + 1] == (
        "replay: the lines below are the output of python -m drowsy.demo --raw on that table. Its \"manual value "
        f"recorded by the extraction\" is the rotation of this session: {metadata['rotation']} ({given}; not checked)."
    )
    assert lines[wrote + 2] == demo.NOTE_RECORDING
    assert (
        f"frames {counts['processed']}, session {report['session']['duration_ms'] / 1000.0:.1f} s, "
        f"events {len(as_events)}, alerts {len(as_alerts)}, model outputs 0, drowsiness.source perclos"
    ) in lines
    return report


EVENT_KEYS = tuple(field.name for field in dataclasses.fields(DriverStateEvent))


def spy_on_sessions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, float | None]]:
    """`main`이 만든 `LiveSession`의 (모델 함수, 속도)를 모으는 목록."""
    made: list[tuple[Any, float | None]] = []

    class Spy(live.LiveSession):
        def __init__(self, spec: Any, rules: Any, model: Any, speed_kmh: float | None = None, landmarker: Any = None):
            made.append((model, speed_kmh))
            super().__init__(spec, rules, model, speed_kmh, landmarker)

    monkeypatch.setattr(live, "LiveSession", Spy)
    return made


def test_command_on_a_file_source_writes_only_the_table_the_timeline_and_the_events_json(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    repo_root: Path,
    pipeline_spec: PipelineSpec,
    scripted_landmarker: SimpleNamespace,
    write_watch: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    clip = clips / "plain.mp4"
    write_clip(clip, frame_times(150))  # 5초, 30 fps
    # 설정이 가리키는 규칙 파일의 값 하나를 바꾼다(마이크로슬립 1초 → 2초). 세션과, 끝의 다시 돌리기가 모두 이 설정을
    # 써야 터미널의 이벤트와 이벤트 JSON이 같다.
    rules_text = (repo_root / "config" / "rules.yaml").read_text(encoding="utf-8")
    assert rules_text.count("microsleep_ms: 1000") == 1
    rules_copy = tmp_path / "rules.yaml"
    rules_copy.write_text(rules_text.replace("microsleep_ms: 1000", "microsleep_ms: 2000"), encoding="utf-8")
    data = yaml.safe_load(cli_config.read_text(encoding="utf-8"))
    data["paths"]["rules"] = str(rules_copy)
    cli_config.write_text(yaml.safe_dump(data), encoding="utf-8")
    sessions = spy_on_sessions(monkeypatch)
    collector: list[str] = []  # 세션 동안 가비지 수거기에 한 일
    real_freeze, real_unfreeze = live.gc.freeze, live.gc.unfreeze
    monkeypatch.setattr(live.gc, "freeze", lambda: (collector.append("freeze"), real_freeze())[1])
    monkeypatch.setattr(live.gc, "unfreeze", lambda: (collector.append("unfreeze"), real_unfreeze())[1])
    out_dir = tmp_path / "out"
    port = free_port()
    page = PageWatcher(port)
    page.start()
    started = time.monotonic()
    argv = ["--config", str(cli_config), "--source", str(clip), "--model", str(model_v0), "--out", str(out_dir)]
    with write_watch.watching():
        code = live.main([*argv, "--port", str(port), "--speed-kmh", "60"])
    elapsed = time.monotonic() - started
    page.done.set()
    page.join(30.0)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert code == 0 and captured.err == ""

    # 처음: 무엇을 돌리는지
    assert lines[0] == live.NOTE_TOOL
    assert lines[1].startswith(f"model: {model_v0} (version 0, ") and lines[1].endswith(live.NOTE_RANDOM_MODEL)
    assert lines[2] == "drowsiness.source: perclos (drowsy events come from PERCLOS, not from the model)"
    assert lines[3] == f"source: {clip} (video file, played back at its own frame timing)"
    assert lines[4].startswith("rotation: 0 (the default; --rotation was not given; ")
    assert lines[5] == "speed: 60 km/h (constant)"
    assert lines[6] == (
        f"page: http://localhost:{port}/ (status and camera preview; the preview is kept in memory only and served "
        "on 127.0.0.1 only)"
    )
    assert lines[7] == STOP_LINE.format(ends="wait for the source to end")
    assert lines[8] == "source opened: waiting for the first frame"
    assert lines[9] == "first frame: 640x480; the session clock starts here"  # 가짜 랜드마커가 돌려준 크기다
    assert lines[10] == live.baseline_note(pipeline_spec)

    # 파일은 실제 속도로 재생된다: 5초짜리가 4.5초보다 빨리 끝나지 않는다. 1초에 한 번 상태 줄이 나온다.
    assert elapsed >= 4.5
    status_lines = [line for line in printed(lines, "session ") if not line.startswith("session ended")]
    assert len(status_lines) >= 3 and all(", face yes, " in line and "baseline " in line for line in status_lines)
    assert f"session ended: {live.END_OF_STREAM}" in lines

    report = check_session_files(out_dir, lines, pipeline_spec)
    assert report["speed_kmh"] == 60.0 and report["model"]["file"] == str(model_v0)
    # `main`은 세션에 모델 함수(데모와 같은 onnxruntime 함수)와 속도를 넘긴다
    assert len(sessions) == 1 and sessions[0][1] == 60.0
    assert sessions[0][0].__qualname__ == "make_model_fn.<locals>.model"
    # 세션이 도는 동안에는 시작할 때의 객체를 전체 수거에서 빼 두고(프레임 하나가 수거만큼 늦지 않게) 끝나면 되돌린다
    assert collector == ["freeze", "unfreeze"] and live.gc.get_freeze_count() == 0
    # 각본의 감김(0.8초부터 2.5초 머묾)이 기준선 완성 전의 임시 microsleep으로 터미널과 JSON에 나온다. 바꾼 규칙대로
    # 감긴 지 2초에 시작한다.
    event_lines = printed(lines, "event ")
    assert event_lines[0].startswith("event 0.0 s: calibrating onset ")
    (microsleep,) = [line for line in event_lines if " microsleep onset " in line]
    assert ", duration 2.0 s, " in microsleep and microsleep.endswith(", provisional)")
    assert any(line.startswith("alert ") and ": microsleep (immediate, " in line for line in lines)
    # 회전을 주지 않았다: 그림을 돌리지 않고 넣었고, 표에 그렇게 적힌다
    assert set(scripted_landmarker.shapes) == {(H, W, 3)}
    metadata = extract.read_extract_metadata(out_files(out_dir)[0])
    assert (metadata["rotation"], metadata["rotation_override"], metadata["rotation_given"]) == (0, 0, False)

    # 웹 페이지가 세션이 도는 동안 받은 것: 도는 중이라는 상태(프레임 수가 는다), 기준선 안내, 열린 소스의 안내 줄
    assert page.error is None and "running" in page.phases()
    running = [status for status in page.statuses if status["phase"] == "running"]
    frames_seen = [status["frames"] for status in running]
    assert len(running) >= 10 and frames_seen == sorted(frames_seen) and frames_seen[-1] - frames_seen[0] >= 30
    assert all(status["source_open"] and status["preview"] for status in running)
    assert running[0]["hint"] == live.baseline_note(pipeline_spec)
    assert running[0]["info"][3] == lines[3] and running[0]["info"][:3] == lines[:3] and len(running[0]["info"]) == 6
    assert running[-1]["line"].startswith("session ") and ", face yes, " in running[-1]["line"]
    # 미리보기: 영상의 그림 그대로다(회색 바탕에 붉은 사각형. 검은 그림이나 다른 크기가 아니다)
    assert page.preview_code == 200 and page.jpeg is not None
    preview = cv2.cvtColor(cv2.imdecode(np.frombuffer(page.jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert preview.shape == (H, W, 3)
    assert np.abs(preview[100:118, 5:155].astype(np.int16) - 110).max() <= 16  # 사각형 아래의 회색 바탕
    red = (preview[50, :, 0] > 160) & (preview[50, :, 1] < 110)
    assert 32 <= int(red.sum()) <= 46  # 너비 40의 사각형

    # 쓴 것은 출력 폴더의 세 파일뿐이다. 영상 폴더에도, 시스템 임시 폴더에도, 작업 폴더에도, 임시 폴더의 다른 곳에도
    # 아무것도 남기지 않았고, Python이 쓰기로 연 경로는 모두 출력 폴더 아래다(D13).
    write_watch.check(out_dir)
    assert sorted(path.name for path in clips.iterdir()) == ["plain.mp4"]
    left = sorted(path.name for path in tmp_path.iterdir())
    assert left == ["clips", "config.yaml", "cwd", "out", "rules.yaml", "systmp"]
    assert len(scripted_landmarker.created) == 1 and scripted_landmarker.created[0].closed
    # 랜드마커에 넣은 시각은 엄격히 증가하고 1에서 시작한다
    assert scripted_landmarker.seen[0] == 1 and len(scripted_landmarker.seen) == frame_counts(lines)["processed"]


def test_command_receives_a_tcp_stream_from_a_sender_on_this_pc(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    pipeline_spec: PipelineSpec,
    scripted_landmarker: SimpleNamespace,
    write_watch: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """이 PC 안의 왕복: PyAV가 MPEG-TS를 TCP로 보내고, 명령이 듣고 받아 세션을 돌린 뒤 보내는 쪽이 멈추면 끝난다."""
    encodes: list[int] = []
    monkeypatch.setattr(live, "encode_preview", lambda rgb: encodes.append(1) or b"")
    stream_port = free_port()
    sender = Sender(stream_port, n_frames=150)  # 5초
    out_dir = tmp_path / "out"
    port = free_port()
    page = PageWatcher(port, preview=False)
    page.start()

    def start_the_sender_once_the_page_shows_the_wait() -> None:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if any(status.get("source_open") is False for status in list(page.statuses)):
                break  # 프로그램이 듣고 있고, 웹 페이지가 "보내는 쪽을 기다린다"를 받았다
            time.sleep(0.02)
        sender.start()

    threading.Thread(target=start_the_sender_once_the_page_shows_the_wait, daemon=True).start()
    source = f"tcp://127.0.0.1:{stream_port}?listen=1"
    argv = ["--config", str(cli_config), "--source", source, "--model", str(model_v0), "--out", str(out_dir)]
    with write_watch.watching():
        code = live.main([*argv, "--port", str(port), "--no-preview", "--speed-kmh", "10"])
    sender.join(60.0)
    page.done.set()
    page.join(30.0)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert code == 0, captured.err
    assert sender.error is None and sender.sent == 150 and not sender.is_alive()

    assert lines[3] == f"source: {source} (listening; waiting up to 120 s for the sender to connect)"
    assert lines[4] == SENDER_LINE.format(port=stream_port) and lines[5] == CAMERA_LINE
    assert lines[7] == "speed: 10 km/h (constant)"
    assert lines[8].endswith("(status only; the camera preview is off (--no-preview))")
    assert lines[9] == STOP_LINE.format(ends="stop the sender")
    assert lines[10] == "source opened: the sender is connected and sends video; waiting for the first key frame"
    assert f"session ended: {live.END_OF_STREAM}" in lines
    counts = frame_counts(lines)
    # 프레임을 받았다. 수는 정하지 않는다: 받는 쪽은 스트림 분석이 끝난 뒤의 첫 키프레임부터 본다.
    assert 0 < counts["processed"] <= counts["decoded"] <= 150 and counts["skipped"] == 0
    report = check_session_files(out_dir, lines, pipeline_spec)
    assert report["source"]["extract_source"] == source
    assert printed(lines, "event ")[0].startswith("event 0.0 s: calibrating onset ")
    # 10 km/h: 각본의 microsleep은 이벤트로 나오지만 경고는 속도에 막힌다(20 km/h 미만). 터미널에도, 이벤트 JSON에도 없다.
    assert any(" microsleep onset " in line for line in printed(lines, "event "))
    assert not any(": microsleep (" in line for line in printed(lines, "alert "))
    assert report["speed_kmh"] == 10.0 and not any(item["type"] == "microsleep" for item in report["alerts"])
    # 미리보기를 껐다: 그림을 만들지 않았고 페이지에도 없다. 상태는 간다. 안내 줄의 소스는 "접속됐다"로 바뀐다.
    assert encodes == [] and page.error is None and page.preview_code == 404
    running = [status for status in page.statuses if status["phase"] == "running"]
    assert running and all(status["preview"] is False for status in page.statuses)
    assert running[0]["info"][3] == f"source: {source} (listening; the sender is connected)"
    # 보내는 쪽이 접속하기 전: 소스가 아직 열리지 않았다고, 안내 줄은 터미널의 소스 줄 그대로("기다리는 중")
    before = [status for status in page.statuses if status.get("source_open") is False]
    assert before and all(status["phase"] == "waiting" and status["info"][3] == lines[3] for status in before)
    assert all(status["source_open"] is True for status in running)
    write_watch.check(out_dir)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.yaml", "cwd", "out", "systmp"]


@contextlib.contextmanager
def guarded_signals() -> Iterator[SimpleNamespace]:
    """SIGINT, SIGTERM, SIGHUP에 "받았다고 적기만 하는" 처리기를 달아 둔다. 시험하는 코드가 자기 처리기를 달지 않았을 때
    신호가 pytest를 끝내지 않게 하려는 것이다. 끝나면 원래 처리기로 되돌린다."""
    leaked: list[int] = []

    def guard(signum: int, _frame: Any) -> None:
        leaked.append(signum)

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    before = {signum: signal.signal(signum, guard) for signum in signals}
    try:
        yield SimpleNamespace(leaked=leaked, guard=guard)
    finally:
        for signum, handler in before.items():
            signal.signal(signum, handler)


@pytest.mark.parametrize(
    ("signum", "reason"),
    [
        (signal.SIGINT, "interrupted (Ctrl+C)"),
        (signal.SIGTERM, "stopped by a signal (SIGTERM)"),  # kill
        (signal.SIGHUP, "stopped by a signal (SIGHUP)"),  # 터미널 창을 닫았다
    ],
)
def test_command_ends_the_session_and_writes_the_files_on_a_stop_signal(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    pipeline_spec: PipelineSpec,
    scripted_landmarker: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    signum: int,
    reason: str,
) -> None:
    """세션 도중의 Ctrl+C, kill, 터미널 닫기: 처리 중인 프레임을 마치고 세 파일을 쓴 뒤 0으로 끝난다."""
    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(240))  # 8초: 그 전에 신호를 보낸다
    out_dir = tmp_path / "out"

    def send_when_running() -> None:
        deadline = time.monotonic() + 60.0
        while len(scripted_landmarker.seen) < 30 and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signum)

    argv = ["--config", str(cli_config), "--source", str(clip), "--model", str(model_v0), "--out", str(out_dir)]
    with guarded_signals() as guards:
        threading.Thread(target=send_when_running, daemon=True).start()
        started = time.monotonic()
        code = live.main([*argv, "--port", "0"])
        elapsed = time.monotonic() - started
        # 세션이 끝나면 처리기를 되돌려 놓는다. 신호는 명령의 처리기가 받았다(밖의 처리기까지 가지 않았다).
        assert all(signal.getsignal(each) is guards.guard for each in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP))
        assert guards.leaked == []
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert code == 0, captured.err
    assert elapsed < 7.0  # 영상의 끝(8초)까지 가지 않았다
    ended = lines.index(f"session ended: {reason}")
    assert lines[ended - 1] == "stop requested: ending the session"
    counts = frame_counts(lines)
    assert 30 <= counts["processed"] < 240
    check_session_files(out_dir, lines, pipeline_spec)
    assert extract.read_extract_metadata(out_files(out_dir)[0])["end_reason"] == reason
    assert scripted_landmarker.created[0].closed


def test_command_run_with_python_m_leaves_ctrl_c_ignored_after_the_session(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    repo_root: Path,
    scripted_landmarker: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m drowsy.live`의 길(`main(ignore_sigint_at_exit=True)`): 파일을 쓴 뒤의 Ctrl+C가 종료 코드를 바꾸지 않는다."""
    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(15))
    argv = ["--config", str(cli_config), "--source", str(clip), "--model", str(model_v0), "--port", "0"]
    with guarded_signals() as guards:
        assert live.main([*argv, "--out", str(tmp_path / "a")], ignore_sigint_at_exit=True) == 0
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is guards.guard and signal.getsignal(signal.SIGHUP) is guards.guard
        os.kill(os.getpid(), signal.SIGINT)  # 아무 일도 일어나지 않는다
        time.sleep(0.05)
        assert guards.leaked == []
        # 세션을 돌리지 못하고 끝난 실행은 처리기를 건드리지 않는다
        signal.signal(signal.SIGINT, guards.guard)
        missing = ["--config", str(cli_config), "--source", str(tmp_path / "missing.mp4"), "--model", str(model_v0)]
        assert live.main([*missing, "--port", "0", "--out", str(tmp_path / "b")], ignore_sigint_at_exit=True) == 2
        assert signal.getsignal(signal.SIGINT) is guards.guard
    capsys.readouterr()
    # 명령으로 실행하는 길이 그 인자를 준다
    source = (repo_root / "drowsy" / "live.py").read_text(encoding="utf-8")
    assert source.rstrip().endswith('if __name__ == "__main__":\n    sys.exit(main(ignore_sigint_at_exit=True))')


def test_command_checks_the_output_folder_before_the_session_starts(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    scripted_landmarker: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """쓸 수 없는 `--out`은 세션을 시작하기 전에 알린다(끝에 가서 알면 세션 전체를 잃는다)."""
    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(30))
    base = ["--config", str(cli_config), "--source", str(clip), "--model", str(model_v0), "--port", "0"]

    # 파일 아래의 경로
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    assert live.main([*base, "--out", str(blocker / "sub")]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith(f"error: the output folder {blocker / 'sub'} cannot be used ([Errno 20] ")
    assert captured.err.endswith("Fix: pass --out with a folder that can be created and written.\n")
    assert captured.out == "" and scripted_landmarker.created == [] and scripted_landmarker.seen == []  # 시작도 안 했다
    assert blocker.read_text(encoding="utf-8") == "x"
    # 폴더가 아니라 파일인 경로
    assert live.main([*base, "--out", str(blocker)]) == 2
    assert f"error: the output folder {blocker} cannot be used (" in capsys.readouterr().err

    # 쓸 수 없는 폴더와 그 아래에 만들어야 하는 폴더(root는 권한에 걸리지 않으므로 건너뛴다)
    if os.geteuid() != 0:
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o555)
        try:
            for out in (locked, locked / "new" / "deeper"):
                assert live.main([*base, "--out", str(out)]) == 2
                assert f"error: the output folder {out} cannot be used ([Errno 13] " in capsys.readouterr().err
            assert list(locked.iterdir()) == []
        finally:
            locked.chmod(0o755)

    # 새로 만든 폴더는 아무것도 쓰지 않고 끝나면 되돌린다. 있던 폴더는 그대로 둔다(확인용 파일도 남지 않는다).
    kept = tmp_path / "kept"
    kept.mkdir()
    missing = ["--config", str(cli_config), "--source", str(tmp_path / "missing.mp4"), "--model", str(model_v0)]
    for out in (tmp_path / "x" / "y" / "out", kept):
        assert live.main([*missing, "--port", "0", "--out", str(out)]) == 2
        assert "missing.mp4" in capsys.readouterr().err
    assert not (tmp_path / "x").exists() and kept.is_dir() and list(kept.iterdir()) == []

    # 좋은 실행: 여러 겹의 새 폴더를 만들어 쓰고, 그 안에는 세 파일뿐이다
    nested = tmp_path / "p" / "q" / "out"
    assert live.main([*base, "--out", str(nested)]) == 0
    capsys.readouterr()
    out_files(nested)


def test_command_turns_the_picture_by_rotation_counter_clockwise(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    scripted_landmarker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """명령의 `--rotation`: 랜드마커가 받는 그림이 영상의 그림을 그 각도만큼 반시계로 돌린 것이다(`np.rot90`).

    90과 270은 서로 반대 방향이다. 영상의 그림은 `video_io`로 같은 파일을 돌리지 않고 읽은 것이다.
    """
    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(15))
    upright = {t_ms: rgb for t_ms, rgb in video_io.iter_video_frames(clip, rotation=0)}
    assert len(upright) == 15
    given: list[tuple[int, np.ndarray]] = []
    scripted_detect = extract.detect_frame

    def detect_frame(landmarker: Any, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        given.append((t_ms, rgb.copy()))
        return scripted_detect(landmarker, t_ms, rgb, spec)

    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    base = ["--config", str(cli_config), "--source", str(clip), "--model", str(model_v0), "--port", "0"]
    for rotation, quarter_turns in ((90, 1), (270, 3)):
        given.clear()
        scripted_landmarker.seen.clear()  # 새 세션이다
        assert live.main([*base, "--rotation", str(rotation), "--out", str(tmp_path / f"r{rotation}")]) == 0
        capsys.readouterr()
        assert len(given) >= 10
        for t_ms, rgb in given:
            expected = np.rot90(upright[t_ms], quarter_turns)
            assert rgb.shape == (W, H, 3) and np.array_equal(rgb, expected), (rotation, t_ms)
            # 반대 방향(90 ↔ 270)으로 돌린 그림과는 다르다: 방향을 바꾸면 이 시험이 알아챈다
            assert not np.array_equal(rgb, np.rot90(upright[t_ms], 4 - quarter_turns))
        metadata = extract.read_extract_metadata(out_files(tmp_path / f"r{rotation}")[0])
        assert (metadata["rotation"], metadata["rotation_given"]) == (rotation, True)


def test_command_returns_2_and_names_the_error_when_processing_fails_in_mid_session(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    pipeline_spec: PipelineSpec,
    scripted_landmarker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """세션 도중에 프레임 하나의 처리가 실패한다: 그때까지의 프레임으로 세 파일을 쓰고, 오류를 알리고 2로 끝난다."""
    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(60))  # 2초
    calls = [0]
    scripted_detect = extract.detect_frame

    def detect_frame(landmarker: Any, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        calls[0] += 1
        if calls[0] == 20:
            raise extract.ExtractError("the landmarker broke")
        return scripted_detect(landmarker, t_ms, rgb, spec)

    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    out_dir = tmp_path / "out"
    argv = ["--config", str(cli_config), "--source", str(clip), "--model", str(model_v0), "--port", "0"]
    code = live.main([*argv, "--out", str(out_dir)])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert code == 2
    assert captured.err == "error: the session was ended by a processing error: the landmarker broke\n"
    assert "session ended: processing failed (the landmarker broke)" in lines
    # 열아홉 프레임을 처리했고 스무째가 실패했다. 실패한 프레임은 따로 센다(남은 프레임이 아니다).
    counts = frame_counts(lines)
    assert (counts["processed"], counts["failed"]) == (19, 1)
    # 그래도 세 파일은 썼고, 데모의 다시 돌리기가 터미널의 이벤트를 그대로 낸다
    report = check_session_files(out_dir, lines, pipeline_spec)
    assert report["counters"]["frames_accepted"] == 19
    metadata = extract.read_extract_metadata(out_files(out_dir)[0])
    assert metadata["end_reason"] == "processing failed (the landmarker broke)" and metadata["n_failed"] == 1
    assert scripted_landmarker.created[0].closed


# ---------------------------------------------------------------------------
# (3) 실제 MediaPipe로 도는 왕복 (모델 파일이나 라이브러리가 없으면 건너뛴다)
# ---------------------------------------------------------------------------


@pytest.mark.mediapipe
def test_loopback_with_the_real_landmarker_until_the_sender_stops(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    pipeline_spec: PipelineSpec,
    real_landmarker_model: Path,
    write_watch: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stream_port = free_port()
    sender = Sender(stream_port, n_frames=90)  # 3초
    sender.start()
    out_dir = tmp_path / "out"
    source = f"tcp://127.0.0.1:{stream_port}?listen=1"
    argv = ["--config", str(cli_config), "--source", source, "--model", str(model_v0), "--out", str(out_dir)]
    with write_watch.watching():
        code = live.main([*argv, "--port", "0"])
    sender.join(60.0)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert code == 0, captured.err
    assert sender.error is None and sender.sent == 90

    assert "source opened: the sender is connected and sends video; waiting for the first key frame" in lines
    assert f"first frame: {W}x{H}; the session clock starts here" in lines
    assert f"session ended: {live.END_OF_STREAM}" in lines
    counts = frame_counts(lines)
    assert 0 < counts["processed"] <= counts["decoded"] <= 90 and counts["face"] == 0  # 얼굴 없는 그림이다
    report = check_session_files(out_dir, lines, pipeline_spec)
    assert report["baseline"] is None and report["counters"]["face_frames"] == 0
    assert [f"{event['type']}:{event['phase']}" for event in report["events"]] == ["calibrating:onset"]
    assert "No model output: the baseline was not completed" in captured.out
    metadata = extract.read_extract_metadata(out_files(out_dir)[0])
    assert metadata["mediapipe_version"] == extract.mediapipe_version()
    assert (metadata["width"], metadata["height"], metadata["arrival_clock"]) == (W, H, False)
    write_watch.check(out_dir)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.yaml", "cwd", "out", "systmp"]


class LiveProcess:
    """`python -m drowsy.live`를 따로 띄우고 표준 출력의 줄을 모은다. 표준 오류는 파일로 간다."""

    def __init__(self, command: Sequence[str], cwd: Path, log: Path, **popen_args: Any) -> None:
        self.lines: list[str] = []
        self._log = log
        self._stderr = log.open("w", encoding="utf-8")
        popen_args.setdefault("stdout", subprocess.PIPE)
        self.process = subprocess.Popen(command, cwd=cwd, stderr=self._stderr, text=True, **popen_args)
        self._pump: threading.Thread | None = None
        if self.process.stdout is not None:
            self._pump = threading.Thread(target=self._read, daemon=True)
            self._pump.start()

    def _read(self) -> None:
        for line in self.process.stdout:  # type: ignore[union-attr]
            self.lines.append(line.rstrip("\n"))

    def wait_for_status_line(self) -> None:
        """첫 상태 줄(세션이 돌고 있다)이 나올 때까지 기다린다."""
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline and self.process.poll() is None:
            if any(line.startswith("session ") and ", face " in line for line in self.lines):
                return
            time.sleep(0.05)
        raise AssertionError((self.lines, self.stderr_text()))

    def stderr_text(self) -> str:
        self._stderr.flush()
        return self._log.read_text(encoding="utf-8")

    def finish(self, timeout_s: float = 120.0) -> int:
        try:
            code = self.process.wait(timeout=timeout_s)
        finally:
            if self.process.poll() is None:
                self.process.kill()
            self._stderr.close()
        if self._pump is not None:
            self._pump.join(30.0)
        return code


@pytest.mark.mediapipe
def test_python_m_drowsy_live_ends_cleanly_on_ctrl_c(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    repo_root: Path,
    pipeline_spec: PipelineSpec,
    real_landmarker_model: Path,
) -> None:
    """`python -m drowsy.live`를 따로 띄우고, 스트림이 오는 도중에 SIGINT를 보낸다: 0으로 끝나고 세 파일을 쓴다.

    Ctrl+C를 여러 번 누른 것처럼 프로세스가 끝날 때까지 0.1초마다 다시 보낸다: 파일을 쓰는 동안에도, 인터프리터가 끝나는
    동안에도 종료 코드는 0이다.
    """
    stream_port = free_port()
    out_dir = tmp_path / "out"
    source = f"tcp://127.0.0.1:{stream_port}?listen=1"
    command = [sys.executable, "-m", "drowsy.live", "--config", str(cli_config), "--source", source]
    command += ["--model", str(model_v0), "--out", str(out_dir), "--port", "0"]
    running = LiveProcess(command, repo_root, tmp_path / "stderr.log")
    sender = Sender(stream_port, n_frames=30 * 120)  # 2분: 그 전에 끊는다
    sender.start()
    try:
        running.wait_for_status_line()
        presses = 0
        while running.process.poll() is None and presses < 600:
            running.process.send_signal(signal.SIGINT)
            presses += 1
            time.sleep(0.1)
    finally:
        code = running.finish()
    sender.join(30.0)
    lines, stderr = running.lines, (tmp_path / "stderr.log").read_text(encoding="utf-8")
    assert code == 0 and presses >= 2, (code, presses, lines, stderr)
    assert not sender.is_alive() and sender.error is None and 0 < sender.sent < 30 * 120  # 받는 쪽이 끊자 멈췄다

    assert "session ended: interrupted (Ctrl+C)" in lines
    assert lines[0] == live.NOTE_TOOL and f"first frame: {W}x{H}; the session clock starts here" in lines
    counts = frame_counts(lines)
    assert counts["processed"] > 0 and counts["face"] == 0
    check_session_files(out_dir, lines, pipeline_spec)
    assert extract.read_extract_metadata(out_files(out_dir)[0])["end_reason"] == "interrupted (Ctrl+C)"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.yaml", "out", "stderr.log"]
    assert "Traceback" not in stderr and "KeyboardInterrupt" not in stderr


@pytest.mark.mediapipe
def test_closing_the_terminal_window_in_mid_session_still_writes_the_files(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    repo_root: Path,
    real_landmarker_model: Path,
) -> None:
    """터미널 창을 닫은 것처럼: 프로그램의 제어 터미널(가짜 터미널)을 세션 도중에 닫는다.

    커널이 SIGHUP을 보내고, 그 뒤로는 터미널에 줄을 낼 수 없다(쓰면 오류다). 그래도 세션의 세 파일은 쓴다.
    """
    import pty  # 가짜 터미널은 이 시험에서만 쓴다
    import select

    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(30 * 60))  # 1분: 그 전에 창을 닫는다
    out_dir = tmp_path / "out"
    command = [sys.executable, "-m", "drowsy.live", "--config", str(cli_config), "--source", str(clip)]
    command += ["--model", str(model_v0), "--out", str(out_dir), "--port", "0"]
    master, slave = pty.openpty()
    # 새 세션의 첫 프로세스가 터미널을 열면 그 터미널이 제어 터미널이 된다(창을 닫으면 SIGHUP을 받는 자리). 그렇게 연 뒤
    # 표준 입출력을 그 터미널로 돌리고 명령으로 바뀐다(프로세스는 그대로다).
    take_the_terminal = (
        "import os, sys; fd = os.open(sys.argv[1], os.O_RDWR); [os.dup2(fd, n) for n in (0, 1, 2)]; "
        "os.execv(sys.executable, [sys.executable, *sys.argv[2:]])"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", take_the_terminal, os.ttyname(slave), *command[1:]],
        cwd=repo_root,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    seen = b""
    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline and b"\nsession 1." not in seen and process.poll() is None:
            if select.select([master], [], [], 0.2)[0]:
                try:
                    seen += os.read(master, 65536)
                except OSError:
                    break
        assert b"\nsession 1." in seen, seen.decode("utf-8", "replace")  # 세션이 돌고 있다
    finally:
        os.close(slave)
        os.close(master)  # 창을 닫는다
        try:
            process.wait(timeout=120.0)
        finally:
            if process.poll() is None:
                process.kill()
    # 종료 코드는 정하지 않는다(터미널이 없어 인터프리터가 끝날 때 표준 출력을 비우지 못한다). 파일은 온전하다.
    parquet, png, events_json = out_files(out_dir)
    metadata = extract.read_extract_metadata(parquet)
    assert metadata["end_reason"] == "stopped by a signal (SIGHUP)" and 30 <= metadata["n_processed"] < 30 * 60
    assert pq.read_table(parquet).num_rows == metadata["n_processed"]
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    report = json.loads(events_json.read_text(encoding="utf-8"))
    assert report["counters"]["frames_accepted"] == metadata["n_processed"]


# ---------------------------------------------------------------------------
# 명령: 실행할 수 없으면 2
# ---------------------------------------------------------------------------


def copy_model(model: Path, out_dir: Path, **metadata: str) -> Path:
    """모델과 feature_spec 사본을 out_dir에 복사한다. metadata에 준 키는 모델 메타데이터에서 그 값으로 바꾼다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = onnx.load(str(model))
    for prop in proto.metadata_props:
        if prop.key in metadata:
            prop.value = metadata[prop.key]
    copied = out_dir / model.name
    onnx.save(proto, str(copied))
    shutil.copy(exported_spec_path(model), out_dir)
    return copied


def test_command_reports_what_stops_it_with_exit_code_2(
    tmp_path: Path,
    cli_config: Path,
    model_v0: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted_landmarker: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "out"
    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(30))
    base = ["--config", str(cli_config), "--out", str(out_dir), "--port", "0"]
    good = [*base, "--source", str(clip), "--model", str(model_v0)]

    # 모델 메타데이터가 spec과 어긋난다(데모와 같은 대조)
    wrong = copy_model(model_v0, tmp_path / "wrong", feature_spec_hash="0" * 64)
    assert live.main([*base, "--source", str(clip), "--model", str(wrong)]) == 2
    assert "error: feature_spec_hash mismatch" in capsys.readouterr().err
    assert live.main([*base, "--source", str(clip), "--model", str(tmp_path / "missing.onnx")]) == 2
    assert "error: ONNX model not found" in capsys.readouterr().err

    # 설정 파일이 없다, 속도가 음수다
    assert live.main(["--config", str(tmp_path / "missing.yaml"), "--source", str(clip)]) == 2
    assert capsys.readouterr().err.startswith("error:")
    assert live.main([*good, "--speed-kmh", "-1"]) == 2
    assert "error: speed must be a finite value >= 0 km/h" in capsys.readouterr().err

    # 열 수 없는 소스
    assert live.main([*base, "--source", str(tmp_path / "missing.mp4"), "--model", str(model_v0)]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error:") and "missing.mp4" in captured.err
    assert captured.out.splitlines()[0] == live.NOTE_TOOL  # 소스를 열기 전에 무엇을 돌리는지 알렸다

    # 웹 페이지의 포트가 쓰이고 있다: 소스를 열기 전에 끝난다
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        argv = ["--config", str(cli_config), "--out", str(out_dir), "--source", str(clip), "--model", str(model_v0)]
        assert live.main([*argv, "--port", str(port)]) == 2
        captured = capsys.readouterr()
        assert f"error: port {port} on 127.0.0.1 is already in use" in captured.err and "--port" in captured.err
        assert captured.out == ""

    # 보내는 쪽이 접속하지 않는다(기다리는 시간을 줄여서 시험한다)
    monkeypatch.setattr(live, "CONNECT_TIMEOUT_S", 0.5)
    source = f"tcp://127.0.0.1:{free_port()}?listen=1"
    assert live.main([*base, "--source", source, "--model", str(model_v0)]) == 2
    captured = capsys.readouterr()
    assert captured.err == f"error: {no_video_error(source, '0.5')}\n"
    assert f"source: {source} (listening; waiting up to 0.5 s for the sender to connect)" in captured.out
    assert "source opened" not in captured.out

    # 어느 경우에도 아무것도 쓰지 않았고(출력 폴더도 남기지 않았다) 랜드마커는 닫혔다
    assert not out_dir.exists()
    assert all(landmarker.closed for landmarker in scripted_landmarker.created)

    with pytest.raises(SystemExit):  # --rotation은 0, 90, 180, 270만 받는다
        live.main([*good, "--rotation", "45"])
    capsys.readouterr()


def test_command_names_the_fix_when_the_landmarker_model_is_absent(
    tmp_path: Path, cli_config: Path, model_v0: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = yaml.safe_load(cli_config.read_text(encoding="utf-8"))
    data["paths"]["model_path"] = str(tmp_path / "face_landmarker.task")
    cli_config.write_text(yaml.safe_dump(data), encoding="utf-8")
    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(6))
    argv = ["--config", str(cli_config), "--source", str(clip), "--model", str(model_v0), "--port", "0"]
    assert live.main(argv) == 2
    captured = capsys.readouterr()
    assert "error: MediaPipe model file not found" in captured.err and captured.out == ""
    assert not (tmp_path / "configured_out").exists()


def test_command_defaults(
    tmp_path: Path,
    cli_config: Path,
    repo_root: Path,
    scripted_landmarker: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = live._build_parser().parse_args([])
    assert (args.source, args.port, args.no_preview) == ("tcp://127.0.0.1:5599?listen=1", 8765, False)
    assert args.rotation is None  # 주지 않으면 0을 쓰되, 주지 않았다는 것을 구별한다
    assert (args.config, args.model, args.speed_kmh, args.out) == ("config/default.yaml", None, None, None)

    # --model과 --out을 주지 않으면: 설정의 demo.model_path가 없을 때 무작위 가중치 모델을 쓰고 알린다. 출력은 <demo.out_dir>/live
    clip = tmp_path / "plain.mp4"
    write_clip(clip, frame_times(15))
    assert live.main(["--config", str(cli_config), "--source", str(clip), "--port", "0", "--rotation", "90"]) == 0
    captured = capsys.readouterr()
    assert "does not exist yet; using the random-weight contract model" in captured.err
    assert captured.err.count("\n") == 1  # 끝에 부르는 데모는 같은 알림을 되풀이하지 않는다
    assert "rotation: 90 (from --rotation; " in captured.out
    # --rotation 90: 랜드마커에 넣는 그림이 돌아가 있다(160x120 영상이 높이 160, 너비 120이 된다)
    assert set(scripted_landmarker.shapes) == {(W, H, 3)}
    parquet, _, events_json = out_files(tmp_path / "configured_out" / "live")
    assert parquet.name.startswith("live_2") and len(parquet.stem) == len("live_20261114_183520")
    metadata = extract.read_extract_metadata(parquet)
    assert (metadata["rotation"], metadata["rotation_override"], metadata["rotation_given"]) == (90, 90, True)
    report = json.loads(events_json.read_text(encoding="utf-8"))
    assert report["model"]["file"].endswith("drowsiness_v0.onnx") and report["source"]["rotation"] == 90
    assert "rotation: 90 (manual value recorded by the extraction; the automatic check was not run)" in captured.out
    assert 'is the rotation of this session: 90 (from --rotation; not checked).' in captured.out
    assert "drowsiness.source: perclos (drowsy events come from PERCLOS, not from the model)" in captured.out

    # 규칙 설정의 졸음 판정원이 모델을 쓰는데 모델이 무작위 가중치면, 시작 줄이 졸음 이벤트에도 뜻이 없다고 알린다
    rules_text = (repo_root / "config" / "rules.yaml").read_text(encoding="utf-8")
    assert rules_text.count("  source: perclos\n") == 1
    rules_copy = tmp_path / "rules.yaml"
    rules_copy.write_text(rules_text.replace("  source: perclos\n", "  source: either\n"), encoding="utf-8")
    data = yaml.safe_load(cli_config.read_text(encoding="utf-8"))
    data["paths"]["rules"] = str(rules_copy)
    cli_config.write_text(yaml.safe_dump(data), encoding="utf-8")
    scripted_landmarker.seen.clear()  # 새 세션이다: 가짜 랜드마커의 시각 검사를 처음으로 돌린다
    argv = ["--config", str(cli_config), "--source", str(clip), "--port", "0", "--out", str(tmp_path / "e")]
    assert live.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].endswith(
        ": the model output carries no meaning, and neither do drowsy events (drowsiness.source either takes them "
        "from the model); only the other rule events do"
    )
    assert lines[2] == (
        "drowsiness.source: either (drowsy events come from the model output or from PERCLOS) WARNING: this model "
        "has random weights, so drowsy events carry no meaning"
    )
    assert lines[-2].endswith(", model outputs 0, drowsiness.source either")


def test_session_stem_is_the_local_start_time() -> None:
    from datetime import datetime

    assert live.session_stem(datetime(2026, 11, 14, 9, 5, 7)) == "live_20261114_090507"
    assert live.session_stem().startswith("live_") and len(live.session_stem()) == 20


# ---------------------------------------------------------------------------
# 의존성과 화소: 선언하지 않은 패키지를 쓰지 않고, 그림을 파일로 쓰는 함수를 부르지 않는다
# ---------------------------------------------------------------------------

_IMPORT_NAMES = {"opencv-python-headless": "cv2", "scikit-learn": "sklearn", "pyyaml": "yaml"}


def test_live_uses_only_declared_dependencies_and_never_writes_pictures(repo_root: Path) -> None:
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = [requirement.split("==")[0].strip().lower() for requirement in project["dependencies"]]
    declared = {_IMPORT_NAMES.get(name, name) for name in names}
    source = (repo_root / "drowsy" / "live.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert sorted(imported - set(sys.stdlib_module_names) - declared - {"drowsy", "__future__"}) == []
    assert {"av", "cv2", "numpy"} <= imported

    # 그림이나 영상을 파일로 쓰는 호출이 없다(미리보기는 cv2.imencode로 메모리에서만 만든다. D13).
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert "imencode" in called
    writers = {"imwrite", "imwritemulti", "VideoWriter", "savefig", "save", "savez", "savez_compressed", "savetxt"}
    writers |= {"tofile", "write_bytes", "write_text", "dump", "fdopen", "mkstemp", "mkdtemp", "NamedTemporaryFile"}
    writers |= {"TemporaryFile", "SpooledTemporaryFile", "TemporaryDirectory", "memmap", "open_memmap"}
    assert called.isdisjoint(writers)
    # `write`를 부르는 곳은 HTTP 응답(`self.wfile.write`)뿐이다. 파일 기술자에 직접 쓰지 않는다.
    writes = {ast.unparse(node.func) for node in calls if getattr(node.func, "attr", "") == "write"}
    assert writes == {"self.wfile.write"}
    # 파일을 여는 곳은 PyAV의 입력(읽기)뿐이다
    opens = [node for node in calls if getattr(node.func, "attr", "") == "open"]
    assert len(opens) == 1 and ast.unparse(opens[0].func) == "av.open"
    assert all(not (isinstance(arg, ast.Constant) and arg.value == "w") for arg in opens[0].args)
    assert not any(getattr(node.func, "id", "") == "open" for node in calls)
    # 파일을 만드는 곳은 출력 폴더를 확인하는 빈 파일 하나뿐이고(`_prepare_out_dir`), 만든 자리에서 지운다
    touches = [ast.unparse(node) for node in calls if getattr(node.func, "attr", "") == "touch"]
    assert touches == ["probe.touch(exist_ok=False)"] and "probe.unlink()" in source
