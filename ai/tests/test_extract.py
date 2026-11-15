"""extract: 결과 → RawFrame, 흐름 제한기, parquet 왕복, 영상 추출, 회전 점검, 추출 리포트, 명령 (PLAN 2.3, 4.1, 5.2절, 11장 M3).

얼굴 이미지는 저장소에 없다. MediaPipe 결과는 손으로 만든 값이나 가짜 랜드마커로 대신하고,
`mediapipe` 표시가 붙은 테스트만 실제 랜드마커를 얼굴 없는 합성 영상에 돌린다. 실제 얼굴로 도는 테스트는 둘이고
환경 변수 `DROWSY_TEST_FACE_IMAGE`가 쓸 수 있는 얼굴 사진을 가리킬 때만 돈다(없으면 건너뛴다).
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import itertools
import math
import os
import shutil
import struct
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from drowsy import extract
from drowsy.config import DropSimConfig
from drowsy.extract import (
    ExtractError,
    FlowLimiter,
    drop_sim_for_fps,
    drop_sim_latencies,
    extract_video,
    find_videos,
    flow_limiter_select,
    output_is_current,
    raw_parquet_path,
    read_extract_metadata,
    read_raw_frames,
    result_to_raw_frame,
    simulate_frame_drops,
    verify_model_file,
    write_raw_frames,
)
from drowsy.spec import (
    PipelineSpec,
    compute_pipeline_hash,
    file_sha256,
    read_feature_spec_dict,
    write_feature_spec,
)
from drowsy.types import RawFrame
from drowsy.video_io import VideoError, iter_video_frames

W, H = 160, 120
N_LANDMARKS = 478

# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def f32(value: float) -> float:
    """MediaPipe가 내는 값처럼 float32로 한 번 내린 Python float."""
    return float(np.float32(value))


def fake_result(spec: PipelineSpec, seed: int, fortran_matrix: bool = True) -> SimpleNamespace:
    """FaceLandmarkerResult와 같은 모양의 손으로 만든 결과(랜드마크 478, 블렌드셰이프 52, 4x4 float32 행렬)."""
    rng = np.random.default_rng(seed)
    landmarks = [
        SimpleNamespace(x=f32(rng.random()), y=f32(rng.random()), z=f32(rng.random())) for _ in range(N_LANDMARKS)
    ]
    blendshapes = [
        SimpleNamespace(index=index, score=f32(rng.random()), category_name=name, display_name=None)
        for index, name in enumerate(spec.blendshapes.order)
    ]
    matrix = np.array(
        [
            [0.9, 0.1, 0.2, 1.5 + seed],
            [-0.1, 0.8, 0.3, -2.5],
            [-0.2, -0.3, 0.7, -30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if fortran_matrix:  # 실제 MediaPipe 행렬은 F 연속 float32다
        matrix = np.asfortranarray(matrix)
    return SimpleNamespace(
        face_landmarks=[landmarks], face_blendshapes=[blendshapes], facial_transformation_matrixes=[matrix]
    )


def empty_result() -> SimpleNamespace:
    return SimpleNamespace(face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[])


def hand_frame(spec: PipelineSpec, t_ms: int, w: int = 640, h: int = 480) -> RawFrame:
    """필수 항목만 가진 손으로 만든 RawFrame(랜드마크 44, 블렌드셰이프 11, 임의의 배정밀도 값)."""
    indices = spec.landmarks.required_indices()
    lm = {index: (index / 1000.0 + t_ms * 1e-7, index / 2000.0 + 1.0 / 3.0) for index in indices}
    bs = {name: 0.01 * (k + 1) + t_ms * 1e-9 for k, name in enumerate(spec.blendshapes.required_names())}
    m = tuple(float(v) for v in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1 * t_ms, -2.0, -30.0, 1))
    return RawFrame(t_ms=t_ms, face=True, w=w, h=h, lm=lm, bs=bs, m=m)


def no_face_frame(t_ms: int, w: int = 640, h: int = 480) -> RawFrame:
    return RawFrame(t_ms=t_ms, face=False, w=w, h=h, lm={}, bs={}, m=())


def plain_frame(index: int) -> np.ndarray:
    """얼굴이 없는 합성 그림: 검은 화면, 잡음, 움직이는 사각형을 번갈아 낸다."""
    kind = index % 3
    if kind == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)
    if kind == 1:
        return np.random.default_rng(index).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    image = np.full((H, W, 3), 128, dtype=np.uint8)
    image[30:60, 20 + 4 * index : 60 + 4 * index] = (200, 60, 60)
    return image


def write_plain_video(path: Path, pts_ms: Sequence[int], quarter_turns_cw: int = 0) -> None:
    """얼굴 없는 그림을 주어진 표시 시각(ms)으로 H.264 영상에 쓴다."""
    time_base = Fraction(1, 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30)
    first = np.rot90(plain_frame(0), -quarter_turns_cw)
    stream.width, stream.height, stream.pix_fmt = first.shape[1], first.shape[0], "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "20"}
    for index, value in enumerate(pts_ms):
        image = np.ascontiguousarray(np.rot90(plain_frame(index), -quarter_turns_cw))
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = value
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


PTS_30FPS = [0, 33, 67, 100, 133, 167, 200, 233, 267, 300, 333, 367]  # 12 프레임
T_30FPS = [1 + p for p in PTS_30FPS]


class FakeLandmarker:
    def __init__(self) -> None:
        self.closed = False
        self.seen: list[tuple[int, int, int]] = []  # (t_ms, w, h)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_mediapipe(monkeypatch: pytest.MonkeyPatch, pipeline_spec: PipelineSpec) -> list[FakeLandmarker]:
    """랜드마커와 추론을 가짜로 바꾼다. 세 프레임 중 셋째는 얼굴 없음, 나머지는 `hand_frame`을 돌려준다."""
    created: list[FakeLandmarker] = []

    def make_landmarker(model_path: Path, spec: PipelineSpec | None = None) -> FakeLandmarker:
        landmarker = FakeLandmarker()
        created.append(landmarker)
        return landmarker

    def detect_frame(landmarker: FakeLandmarker, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        assert not landmarker.closed
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        assert all(t_ms > seen_t for seen_t, _, _ in landmarker.seen)  # VIDEO 모드는 엄격히 증가하는 값만 받는다
        height, width = rgb.shape[0], rgb.shape[1]
        landmarker.seen.append((t_ms, width, height))
        if len(landmarker.seen) % 3 == 0:
            return no_face_frame(t_ms, width, height)
        return hand_frame(spec, t_ms, width, height)

    monkeypatch.setattr(extract, "make_landmarker", make_landmarker)
    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    monkeypatch.setattr(extract, "mediapipe_version", lambda: "0.0-fake")
    return created


def expected_fake_rows(spec: PipelineSpec, times: Sequence[int], w: int = W, h: int = H) -> list[RawFrame]:
    return [
        no_face_frame(t, w, h) if (k + 1) % 3 == 0 else hand_frame(spec, t, w, h) for k, t in enumerate(times)
    ]


def spec_for_model(spec: PipelineSpec, model_path: Path) -> PipelineSpec:
    """모델 sha256만 주어진 파일의 값으로 바꾼 spec."""
    return dataclasses.replace(
        spec, mediapipe=dataclasses.replace(spec.mediapipe, model_sha256=file_sha256(model_path))
    )


def write_cli_config(
    tmp_path: Path,
    config_path: Path,
    feature_spec_path: Path,
    model_path: Path,
    rotation_overrides: dict[str, int] | None = None,
    name: str = "config.yaml",
) -> Path:
    """기본 설정에서 경로만 임시 폴더로 바꾼 설정 파일을 쓴다."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"]["raw_dir"] = str(tmp_path / "raw")
    data["paths"]["reports_dir"] = str(tmp_path / "reports")  # 추출 리포트가 저장소의 reports/에 쓰이지 않게 한다
    data["paths"]["model_path"] = str(model_path)
    data["paths"]["feature_spec"] = str(feature_spec_path)
    data["extraction"]["workers"] = 2
    if rotation_overrides is not None:
        data["extraction"]["rotation_overrides"] = rotation_overrides
    out = tmp_path / name
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


@pytest.fixture
def fake_model_setup(tmp_path: Path, config_path: Path, feature_spec_path: Path) -> SimpleNamespace:
    """가짜 모델 파일과, 그 sha256을 적은 feature_spec 사본, 둘을 가리키는 설정 파일."""
    model_path = tmp_path / "fake_model.task"
    model_path.write_bytes(b"not a real model")
    spec_dict = read_feature_spec_dict(feature_spec_path)
    spec_dict["pipeline"]["mediapipe"]["model_sha256"] = file_sha256(model_path)
    spec_dict["pipeline_hash"] = compute_pipeline_hash(spec_dict["pipeline"])
    spec_path = tmp_path / "feature_spec.json"
    write_feature_spec(spec_dict, spec_path)
    config = write_cli_config(tmp_path, config_path, spec_path, model_path)
    return SimpleNamespace(
        model_path=model_path,
        spec_path=spec_path,
        config=config,
        raw_dir=tmp_path / "raw",
        reports_dir=tmp_path / "reports",
    )


# ---------------------------------------------------------------------------
# result_to_raw_frame
# ---------------------------------------------------------------------------


def test_result_to_raw_frame_copies_subset_blendshapes_and_size(pipeline_spec: PipelineSpec) -> None:
    result = fake_result(pipeline_spec, seed=3)
    frame = result_to_raw_frame(1234, result, 640, 480, pipeline_spec)
    assert (frame.t_ms, frame.face, frame.w, frame.h) == (1234, True, 640, 480)

    required = pipeline_spec.landmarks.required_indices()
    assert len(required) == 44
    assert tuple(frame.lm) == required  # 44개만, 인덱스 오름차순
    for index in required:
        assert frame.lm[index] == (result.face_landmarks[0][index].x, result.face_landmarks[0][index].y)

    assert tuple(frame.bs) == pipeline_spec.blendshapes.order  # 52개 전부, category_name으로
    assert len(frame.bs) == 52
    for category in result.face_blendshapes[0]:
        assert frame.bs[category.category_name] == category.score
    assert set(pipeline_spec.blendshapes.required_names()) <= set(frame.bs)
    assert frame.bs["eyeBlinkLeft"] == result.face_blendshapes[0][9].score
    assert frame.bs["eyeBlinkRight"] == result.face_blendshapes[0][10].score
    assert frame.bs["jawOpen"] == result.face_blendshapes[0][25].score

    assert all(type(v) is float for xy in frame.lm.values() for v in xy)
    assert all(type(v) is float for v in frame.bs.values())
    assert all(type(v) is float for v in frame.m)


@pytest.mark.parametrize("fortran_matrix", [True, False])
def test_result_to_raw_frame_matrix_is_column_major(pipeline_spec: PipelineSpec, fortran_matrix: bool) -> None:
    result = fake_result(pipeline_spec, seed=0, fortran_matrix=fortran_matrix)
    frame = result_to_raw_frame(1, result, 640, 480, pipeline_spec)
    matrix = result.facial_transformation_matrixes[0]
    assert len(frame.m) == 16
    for i in range(4):
        for j in range(4):
            assert frame.m[4 * j + i] == float(matrix[i, j])  # M[i][j] = m[4*j + i]
    # 손으로 편 값: 첫 열, 둘째 열, 셋째 열, 평행이동 열
    expected = [0.9, -0.1, -0.2, 0.0, 0.1, 0.8, -0.3, 0.0, 0.2, 0.3, 0.7, 0.0, 1.5, -2.5, -30.0, 1.0]
    assert list(frame.m) == [f32(v) for v in expected]
    assert frame.m[12:15] == (1.5, -2.5, -30.0)  # 평행이동은 12~14번(행 우선으로 잘못 읽으면 3, 7, 11번에 온다)
    assert frame.m[3] == 0.0 and frame.m[7] == 0.0 and frame.m[11] == 0.0
    # PLAN 2.2절의 식을 열 우선 인덱스로 쓴 것: yaw = atan2(M[0][2], M[2][2]) = atan2(m[8], m[10])
    assert math.atan2(frame.m[8], frame.m[10]) == pytest.approx(math.atan2(0.2, 0.7), abs=1e-6)
    assert math.atan2(frame.m[1], frame.m[5]) == pytest.approx(math.atan2(-0.1, 0.8), abs=1e-6)  # roll


def test_result_to_raw_frame_without_face(pipeline_spec: PipelineSpec) -> None:
    frame = result_to_raw_frame(77, empty_result(), 320, 240, pipeline_spec)
    assert frame == RawFrame(t_ms=77, face=False, w=320, h=240, lm={}, bs={}, m=())


def test_result_to_raw_frame_rejects_incomplete_results(pipeline_spec: PipelineSpec) -> None:
    result = fake_result(pipeline_spec, seed=0)
    result.face_blendshapes = []
    with pytest.raises(ExtractError, match="no blendshapes or no transformation matrix"):
        result_to_raw_frame(1, result, 640, 480, pipeline_spec)
    result = fake_result(pipeline_spec, seed=0)
    result.facial_transformation_matrixes = [np.eye(3, dtype=np.float32)]
    with pytest.raises(ExtractError, match="4x4"):
        result_to_raw_frame(1, result, 640, 480, pipeline_spec)
    result = fake_result(pipeline_spec, seed=0)
    result.face_landmarks = [result.face_landmarks[0][:468]]  # 홍채가 없는 구형 468점 결과
    with pytest.raises(ExtractError, match="468 landmarks"):
        result_to_raw_frame(1, result, 640, 480, pipeline_spec)


# ---------------------------------------------------------------------------
# 흐름 제한기
# ---------------------------------------------------------------------------


def test_flow_limiter_hand_worked_30fps_into_70ms() -> None:
    """30 fps 입력, 프레임당 처리 70 ms.

    f0@0   한가함 -> 처리 0~70
    f1@33  처리 중 -> 대기
    f2@67  처리 중 -> f1을 밀어내고 대기 (f1 버림)
    f3@100 70에 f0 끝, f2가 70~140 처리 -> f3 대기
    f4@133 처리 중 -> f3을 밀어내고 대기 (f3 버림)
    f5@167 140에 f2 끝, f4가 140~210 처리 -> f5 대기
    f6@200 처리 중 -> f5를 밀어내고 대기 (f5 버림)
    끝     대기 중인 f6 처리
    """
    assert flow_limiter_select([0, 33, 67, 100, 133, 167, 200], [70.0] * 7) == [0, 2, 4, 6]


def test_flow_limiter_fast_processing_keeps_every_frame() -> None:
    assert flow_limiter_select([1, 34, 67, 100], [20.0, 20.0, 20.0, 20.0]) == [0, 1, 2, 3]
    assert flow_limiter_select([1, 34, 67, 100], [0.0, 0.0, 0.0, 0.0]) == [0, 1, 2, 3]


def test_flow_limiter_variable_latency_hand_worked() -> None:
    """f0@0 처리 0~100. f1@30, f2@60, f3@90이 차례로 대기 칸을 차지(f1, f2 버림).
    f4@120: 100에 f0이 끝나 f3이 100~110 처리(지연 10), 110에 끝나 한가함 -> f4 바로 처리 120~170.
    f5@150 대기. f6@180: 170에 f4 끝, f5가 170~175 처리(지연 5), 끝 -> f6 바로 처리.
    """
    arrivals = [0, 30, 60, 90, 120, 150, 180]
    latencies = [100.0, 999.0, 999.0, 10.0, 50.0, 5.0, 40.0]
    assert flow_limiter_select(arrivals, latencies) == [0, 3, 4, 5, 6]


def test_flow_limiter_queued_frame_uses_its_own_latency() -> None:
    # f1은 30에 시작해 250에 끝난다(자기 지연 220). f2@100, f3@200은 대기 칸에서 바뀌고, f4@300 도착 때 f3이 250~260 처리.
    assert flow_limiter_select([0, 10, 100, 200, 300], [30.0, 220.0, 10.0, 10.0, 10.0]) == [0, 1, 3, 4]


def test_flow_limiter_finish_at_arrival_instant_counts_as_free() -> None:
    assert flow_limiter_select([0, 50, 100], [50.0, 50.0, 50.0]) == [0, 1, 2]
    # 1 ms라도 남아 있으면 대기했다가 끝나는 즉시 처리된다(51에 시작, 101에 끝남). f2@100은 그때 대기하다 마지막에 처리.
    assert flow_limiter_select([0, 50, 100], [51.0, 50.0, 50.0]) == [0, 1, 2]
    # 대기 중에 다음 프레임이 오면 밀려난다
    assert flow_limiter_select([0, 50, 100], [101.0, 50.0, 50.0]) == [0, 2]


def test_flow_limiter_streaming_interface() -> None:
    limiter: FlowLimiter[str] = FlowLimiter()
    assert limiter.offer(0, 70.0, "a") == ["a"]
    assert limiter.offer(33, 70.0, "b") == []
    assert limiter.offer(67, 70.0, "c") == []  # b가 버려진다
    assert limiter.offer(100, 70.0, "d") == ["c"]
    assert limiter.offer(400, 70.0, "e") == ["d", "e"]  # d는 140~210에 처리됐고 e는 바로 처리된다
    assert limiter.flush() == []
    assert limiter.offer(410, 70.0, "f") == []
    assert limiter.flush() == ["f"]
    assert limiter.flush() == []


def test_flow_limiter_edge_inputs() -> None:
    assert flow_limiter_select([], []) == []
    assert flow_limiter_select([5], [1000.0]) == [0]
    with pytest.raises(ValueError, match="same length"):
        flow_limiter_select([0, 1], [1.0])
    with pytest.raises(ValueError, match="latency_ms"):
        flow_limiter_select([0], [-1.0])


def test_drop_sim_latencies_are_seeded_and_clipped() -> None:
    cfg = DropSimConfig(subdir="fps15", latency_ms_mean=66.7, latency_ms_std=8.0, seed=0)
    first = list(itertools.islice(drop_sim_latencies(cfg), 50))
    assert first == list(itertools.islice(drop_sim_latencies(cfg), 50))
    assert first != list(itertools.islice(drop_sim_latencies(dataclasses.replace(cfg, seed=1)), 50))
    assert 60.0 < sum(first) / 50 < 73.0
    assert list(itertools.islice(drop_sim_latencies(dataclasses.replace(cfg, latency_ms_std=0.0)), 3)) == [66.7] * 3
    wide = list(itertools.islice(drop_sim_latencies(DropSimConfig("x", 1.0, 50.0, 0)), 200))
    assert min(wide) == 0.0 and all(value >= 0.0 for value in wide)  # 음수는 0으로 자른다


def test_simulate_frame_drops_hand_worked_constant_latency() -> None:
    cfg = DropSimConfig(subdir="x", latency_ms_mean=70.0, latency_ms_std=0.0, seed=0)
    frames = [(t, np.full((1, 1, 3), k, dtype=np.uint8)) for k, t in enumerate([1, 34, 68, 101, 134, 168, 201])]
    kept = list(simulate_frame_drops(iter(frames), cfg))
    assert [t for t, _ in kept] == [1, 68, 134, 201]
    assert [int(rgb[0, 0, 0]) for _, rgb in kept] == [0, 2, 4, 6]  # 시각과 그림이 짝을 유지한다


def test_simulate_frame_drops_matches_flow_limiter_select() -> None:
    cfg = DropSimConfig(subdir="fps15", latency_ms_mean=66.7, latency_ms_std=8.0, seed=0)
    times = [1 + math.floor(k * 1000 / 30 + 0.5) for k in range(300)]  # 30 fps, 10초
    frames = [(t, np.full((1, 1, 3), k % 256, dtype=np.uint8)) for k, t in enumerate(times)]
    kept = [t for t, _ in simulate_frame_drops(iter(frames), cfg)]
    latencies = list(itertools.islice(drop_sim_latencies(cfg), len(times)))
    assert kept == [times[i] for i in flow_limiter_select(times, latencies)]
    assert all(later > earlier for earlier, later in zip(kept, kept[1:]))
    # 처리 한 번에 평균 66.7 ms이므로 10초에 약 150장(15 fps)이 남는다
    assert 135 <= len(kept) <= 155


def test_drop_sim_for_fps() -> None:
    base = DropSimConfig(subdir="fps15", latency_ms_mean=66.7, latency_ms_std=8.0, seed=3)
    assert drop_sim_for_fps(base, 15) is base  # 설정이 이미 15 fps를 뜻한다
    ten = drop_sim_for_fps(base, 10)
    assert ten == DropSimConfig(subdir="fps10", latency_ms_mean=100.0, latency_ms_std=8.0, seed=3)
    with pytest.raises(ExtractError, match="positive"):
        drop_sim_for_fps(base, 0)


# ---------------------------------------------------------------------------
# parquet 왕복
# ---------------------------------------------------------------------------


def test_parquet_round_trip_of_hand_built_results(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    frames = [
        result_to_raw_frame(1, fake_result(pipeline_spec, seed=1), 640, 480, pipeline_spec),
        result_to_raw_frame(34, empty_result(), 640, 480, pipeline_spec),
        result_to_raw_frame(68, fake_result(pipeline_spec, seed=2), 640, 480, pipeline_spec),
        hand_frame(pipeline_spec, 101, w=1080, h=1920),  # 필수 블렌드셰이프 11개만, float32로 표현되지 않는 값
        no_face_frame(9_000_000_000_000, w=1080, h=1920),  # int32를 넘는 t_ms
    ]
    out = tmp_path / "nested" / "frames.parquet"
    assert write_raw_frames(frames, out, pipeline_spec, {"source": "synthetic", "note": "한글"}) == 5
    assert not out.with_name(out.name + ".tmp").exists()

    restored = list(read_raw_frames(out))
    assert restored == frames  # 값이 비트 단위로 같다
    assert [type(frame.t_ms) for frame in restored] == [int] * 5
    assert tuple(restored[0].lm) == pipeline_spec.landmarks.required_indices()
    assert tuple(restored[0].bs) == pipeline_spec.blendshapes.order
    assert tuple(restored[3].bs) == tuple(
        name for name in pipeline_spec.blendshapes.order if name in pipeline_spec.blendshapes.required_names()
    )

    metadata = read_extract_metadata(out)
    assert metadata["source"] == "synthetic" and metadata["note"] == "한글"
    assert metadata["schema_version"] == extract.SCHEMA_VERSION
    assert metadata["landmark_indices"] == list(pipeline_spec.landmarks.required_indices())
    assert metadata["blendshape_names"] == list(pipeline_spec.blendshapes.order)


def test_parquet_schema_and_null_rows(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    frames = [hand_frame(pipeline_spec, 1), no_face_frame(34), hand_frame(pipeline_spec, 68)]
    out = tmp_path / "frames.parquet"
    write_raw_frames(frames, out, pipeline_spec)
    table = pq.read_table(out)

    indices = pipeline_spec.landmarks.required_indices()
    expected_columns = ["t_ms", "face", "w", "h"]
    for index in indices:
        expected_columns += [f"lm{index}_x", f"lm{index}_y"]
    expected_columns += [f"bs_{name}" for name in pipeline_spec.blendshapes.order]
    expected_columns += [f"m{j}" for j in range(16)]
    expected_columns.append("face_w_px")
    assert table.column_names == expected_columns
    assert len(expected_columns) == 4 + 88 + 52 + 16 + 1

    assert table.schema.field("t_ms").type == pa.int64()
    assert table.schema.field("face").type == pa.bool_()
    assert table.schema.field("w").type == pa.int32()
    assert all(table.schema.field(name).type == pa.float64() for name in expected_columns[4:])
    assert table.column("t_ms").to_pylist() == [1, 34, 68]
    assert table.column("face").to_pylist() == [True, False, True]

    # 얼굴 없는 행의 실수 열은 모두 null, 얼굴 있는 행은 필수 항목이 모두 차 있다
    required_bs = {f"bs_{name}" for name in pipeline_spec.blendshapes.required_names()}
    for name in expected_columns[4:]:
        values = table.column(name).to_pylist()
        assert values[1] is None
        if not name.startswith("bs_") or name in required_bs:
            assert values[0] is not None and values[2] is not None
        else:
            assert values[0] is None  # hand_frame에 없는 블렌드셰이프
    assert table.column("m12").to_pylist() == [0.1, None, 0.1 * 68]


def test_parquet_face_width_in_pixels(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    left, right = pipeline_spec.landmarks.face_width
    assert (left, right) == (234, 454)
    square = hand_frame(pipeline_spec, 1, w=640, h=640)
    square.lm[left] = (0.25, 0.5)
    square.lm[right] = (0.75, 0.5)
    wide = hand_frame(pipeline_spec, 2, w=100, h=50)
    wide.lm[left] = (0.2, 0.4)
    wide.lm[right] = (0.5, 0.8)
    out = tmp_path / "frames.parquet"
    write_raw_frames([square, wide, no_face_frame(3)], out, pipeline_spec)
    widths = pq.read_table(out, columns=["face_w_px"]).column("face_w_px").to_pylist()
    assert widths[0] == pytest.approx(320.0)  # 0.5 · 640
    assert widths[1] == pytest.approx(math.sqrt(30.0**2 + 20.0**2))  # 픽셀 좌표 (x·w, y·h)에서의 거리
    assert widths[2] is None


def test_write_raw_frames_rejects_incomplete_face_frames(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    out = tmp_path / "bad.parquet"
    missing_landmark = hand_frame(pipeline_spec, 1)
    del missing_landmark.lm[468]
    with pytest.raises(ExtractError, match=r"landmark indices \[468\]"):
        write_raw_frames([missing_landmark], out, pipeline_spec)

    unknown_blendshape = hand_frame(pipeline_spec, 1)
    unknown_blendshape.bs["notABlendshape"] = 0.5
    with pytest.raises(ExtractError, match="notABlendshape"):
        write_raw_frames([unknown_blendshape], out, pipeline_spec)

    short_matrix = dataclasses.replace(hand_frame(pipeline_spec, 1), m=(1.0, 0.0, 0.0))
    with pytest.raises(ExtractError, match="16 matrix values"):
        write_raw_frames([short_matrix], out, pipeline_spec)
    assert not out.exists()


def test_read_raw_frames_crosses_batch_boundaries(
    tmp_path: Path, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extract, "READ_BATCH_ROWS", 4)
    frames = [no_face_frame(1 + 33 * k) if k % 5 == 4 else hand_frame(pipeline_spec, 1 + 33 * k) for k in range(11)]
    out = tmp_path / "frames.parquet"
    write_raw_frames(frames, out, pipeline_spec)
    assert list(read_raw_frames(out)) == frames


def test_empty_frame_list_round_trips(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    out = tmp_path / "empty.parquet"
    assert write_raw_frames([], out, pipeline_spec) == 0
    assert list(read_raw_frames(out)) == []


def test_read_rejects_foreign_parquet(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.parquet"
    pq.write_table(pa.table({"t_ms": [1, 2]}), foreign)
    with pytest.raises(ExtractError, match="not an extraction parquet"):
        read_extract_metadata(foreign)
    with pytest.raises(ExtractError):
        list(read_raw_frames(foreign))


# ---------------------------------------------------------------------------
# 모델 파일 검사, 랜드마커 생성 오류
# ---------------------------------------------------------------------------


def test_verify_model_file(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    model = tmp_path / "model.task"
    with pytest.raises(ExtractError, match="not found") as excinfo:
        verify_model_file(model, pipeline_spec)
    assert "Fix:" in str(excinfo.value)

    model.write_bytes(b"some other model")
    actual = file_sha256(model)
    with pytest.raises(ExtractError, match="sha256 mismatch") as excinfo:
        verify_model_file(model, pipeline_spec)
    message = str(excinfo.value)
    assert pipeline_spec.mediapipe.model_sha256 in message and actual in message
    assert "Fix:" in message and "update-hash" in message

    assert verify_model_file(model, spec_for_model(pipeline_spec, model)) == actual


def test_make_landmarker_checks_model_before_loading_mediapipe(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    model = tmp_path / "model.task"
    model.write_bytes(b"wrong bytes")
    with pytest.raises(ExtractError, match="sha256 mismatch"):
        extract.make_landmarker(model, pipeline_spec)


def test_make_landmarker_names_the_fix_when_mediapipe_cannot_load(
    tmp_path: Path, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.task"
    model.write_bytes(b"placeholder")
    monkeypatch.setitem(sys.modules, "mediapipe", None)  # import mediapipe가 ImportError를 내게 한다
    with pytest.raises(ExtractError, match="could not be imported") as excinfo:
        extract.make_landmarker(model, spec_for_model(pipeline_spec, model))
    message = str(excinfo.value)
    assert "Fix:" in message and "mediapipe==0.10.32" in message and "libgles2" in message


def test_make_landmarker_reports_native_load_failure(
    tmp_path: Path, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.task"
    model.write_bytes(b"placeholder")

    def create_from_options(options: Any) -> Any:
        raise OSError("libGLESv2.so.2: cannot open shared object file")

    fake_mp = SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            vision=SimpleNamespace(
                FaceLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                RunningMode=SimpleNamespace(VIDEO="VIDEO"),
                FaceLandmarker=SimpleNamespace(create_from_options=create_from_options),
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    with pytest.raises(ExtractError, match="native library could not be loaded") as excinfo:
        extract.make_landmarker(model, spec_for_model(pipeline_spec, model))
    assert "libGLESv2.so.2" in str(excinfo.value) and "sudo apt install libgles2" in str(excinfo.value)


def test_make_landmarker_options(tmp_path: Path, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    """VIDEO 모드, 얼굴 1개, 블렌드셰이프와 변환 행렬 출력으로 만든다."""
    model = tmp_path / "model.task"
    model.write_bytes(b"placeholder")
    fake_mp = SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            vision=SimpleNamespace(
                FaceLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                RunningMode=SimpleNamespace(VIDEO="VIDEO", IMAGE="IMAGE", LIVE_STREAM="LIVE_STREAM"),
                FaceLandmarker=SimpleNamespace(create_from_options=lambda options: options),
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    options = extract.make_landmarker(model, spec_for_model(pipeline_spec, model))
    assert options.running_mode == "VIDEO"
    assert options.num_faces == 1
    assert options.output_face_blendshapes is True
    assert options.output_facial_transformation_matrixes is True
    assert options.base_options.model_asset_path == str(model)


# ---------------------------------------------------------------------------
# extract_video (가짜 랜드마커)
# ---------------------------------------------------------------------------


def test_extract_video_writes_one_row_per_frame(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]
) -> None:
    video = tmp_path / "clip.mp4"
    write_plain_video(video, PTS_30FPS)
    out = tmp_path / "out" / "clip.parquet"
    stats = extract_video(video, tmp_path / "unused.task", out, spec=pipeline_spec)

    assert [t for t, _ in iter_video_frames(video)] == T_30FPS
    rows = list(read_raw_frames(out))
    assert rows == expected_fake_rows(pipeline_spec, T_30FPS)
    assert [row.t_ms for row in rows] == T_30FPS
    assert all(later.t_ms > earlier.t_ms for earlier, later in zip(rows, rows[1:]))

    assert len(fake_mediapipe) == 1 and fake_mediapipe[0].closed  # 영상마다 랜드마커 하나, 끝나면 닫는다
    assert fake_mediapipe[0].seen == [(t, W, H) for t in T_30FPS]

    assert (stats.source, stats.out_parquet) == (str(video), str(out))
    assert (stats.width, stats.height, stats.rotation) == (W, H, 0)
    assert (stats.n_decoded, stats.n_skipped, stats.n_dropped, stats.n_processed, stats.n_face) == (12, 0, 0, 12, 8)
    assert (stats.first_t_ms, stats.last_t_ms) == (1, 368)
    assert stats.elapsed_s > 0.0

    metadata = read_extract_metadata(out)
    assert metadata["source"] == str(video)
    assert metadata["model_sha256"] == pipeline_spec.mediapipe.model_sha256
    assert metadata["mediapipe_version"] == "0.0-fake"
    assert metadata["drop_sim"] is None
    assert (metadata["rotation"], metadata["rotation_override"]) == (0, None)
    assert (metadata["n_decoded"], metadata["n_skipped"], metadata["n_dropped"]) == (12, 0, 0)
    assert (metadata["n_processed"], metadata["n_face"]) == (12, 8)
    # 녹화 요약: 프레임 시각 1..368, 길이 367 ms, 11·1000/367 fps, 간격 33~34 ms(1 ms 흔들림은 일정으로 본다)
    assert (metadata["extract_version"], metadata["width"], metadata["height"]) == (2, W, H)
    assert (metadata["duration_ms"], metadata["mean_fps"]) == (367, pytest.approx(11 * 1000 / 367))
    assert (metadata["min_interval_ms"], metadata["max_interval_ms"], metadata["variable_intervals"]) == (33, 34, False)
    assert metadata["display_flip"] is False
    weights = np.array([0.299, 0.587, 0.114])
    lumas = [float((rgb.astype(np.float64) @ weights).mean()) for _, rgb in iter_video_frames(video)]
    assert metadata["brightness_mean"] == pytest.approx(float(np.mean(lumas)), rel=1e-12)
    assert metadata["brightness_median"] == pytest.approx(float(np.median(lumas)), rel=1e-12)


def test_extract_video_with_drop_simulation(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]
) -> None:
    video = tmp_path / "clip.mp4"
    write_plain_video(video, PTS_30FPS)
    out = tmp_path / "clip.parquet"

    # 지연이 정확히 70 ms: 도착 1, 34, 68, 101, ... 에서 0, 2, 4, ... 번째와 마지막 대기 프레임이 남는다
    constant = DropSimConfig(subdir="x", latency_ms_mean=70.0, latency_ms_std=0.0, seed=0)
    stats = extract_video(video, tmp_path / "unused.task", out, constant, spec=pipeline_spec)
    assert flow_limiter_select(T_30FPS, [70.0] * 12) == [0, 2, 4, 6, 8, 10, 11]
    kept = [T_30FPS[i] for i in (0, 2, 4, 6, 8, 10, 11)]
    assert [row.t_ms for row in read_raw_frames(out)] == kept  # 고른 프레임은 원래 t_ms를 그대로 쓴다
    assert (stats.n_decoded, stats.n_dropped, stats.n_processed) == (12, 5, 7)
    assert (stats.first_t_ms, stats.last_t_ms) == (1, 368)
    assert read_extract_metadata(out)["drop_sim"] == {"latency_ms_mean": 70.0, "latency_ms_std": 0.0, "seed": 0}
    assert read_extract_metadata(out)["n_dropped"] == 5

    # 설정의 기본 모사(평균 66.7, 표준편차 8): 같은 지연 수열로 고른 것과 같다
    noisy = DropSimConfig(subdir="fps15", latency_ms_mean=66.7, latency_ms_std=8.0, seed=0)
    extract_video(video, tmp_path / "unused.task", out, noisy, spec=pipeline_spec)
    latencies = list(itertools.islice(drop_sim_latencies(noisy), 12))
    expected = [T_30FPS[i] for i in flow_limiter_select(T_30FPS, latencies)]
    assert [row.t_ms for row in read_raw_frames(out)] == expected
    assert 5 <= len(expected) <= 8


def test_extract_video_rotation(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]
) -> None:
    lying = tmp_path / "lying.mp4"  # 메타데이터 없이 시계 방향 90도로 누운 영상
    write_plain_video(lying, PTS_30FPS[:3], quarter_turns_cw=1)
    out = tmp_path / "lying.parquet"

    stats = extract_video(lying, tmp_path / "unused.task", out, spec=pipeline_spec)
    assert (stats.width, stats.height, stats.rotation) == (H, W, 0)
    assert read_extract_metadata(out)["rotation_override"] is None

    stats = extract_video(lying, tmp_path / "unused.task", out, spec=pipeline_spec, rotation=90)
    assert (stats.width, stats.height, stats.rotation) == (W, H, 90)
    assert [(row.w, row.h) for row in read_raw_frames(out)] == [(W, H)] * 3
    metadata = read_extract_metadata(out)
    assert (metadata["rotation"], metadata["rotation_override"]) == (90, 90)


def test_extract_video_closes_landmarker_and_writes_nothing_on_failure(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker], monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    write_plain_video(video, PTS_30FPS[:4])
    out = tmp_path / "clip.parquet"

    def failing_detect(landmarker: FakeLandmarker, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        if t_ms > 40:
            raise ValueError("Input timestamp must be monotonically increasing.")
        return no_face_frame(t_ms, rgb.shape[1], rgb.shape[0])

    monkeypatch.setattr(extract, "detect_frame", failing_detect)
    with pytest.raises(ValueError, match="monotonically"):
        extract_video(video, tmp_path / "unused.task", out, spec=pipeline_spec)
    assert fake_mediapipe[-1].closed
    assert list(tmp_path.glob("*.parquet*")) == []

    monkeypatch.setattr(extract, "iter_video_frames", lambda path, rotation=None, counts=None: iter(()))
    with pytest.raises(ExtractError, match="no decodable video frame"):
        extract_video(video, tmp_path / "unused.task", out, spec=pipeline_spec)
    assert fake_mediapipe[-1].closed
    assert not out.exists()


def test_output_is_current(tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]) -> None:
    video = tmp_path / "clip.mp4"
    write_plain_video(video, PTS_30FPS[:4])
    out = tmp_path / "clip.parquet"
    drop_sim = DropSimConfig(subdir="fps15", latency_ms_mean=66.7, latency_ms_std=8.0, seed=0)

    assert output_is_current(out, pipeline_spec, None, None) is False  # 파일 없음
    extract_video(video, tmp_path / "unused.task", out, spec=pipeline_spec)
    assert output_is_current(out, pipeline_spec, None, None) is True
    assert output_is_current(out, pipeline_spec, drop_sim, None) is False  # 누락 모사 설정이 다르다
    assert output_is_current(out, pipeline_spec, None, 90) is False  # 수동 회전 보정이 생겼다
    other_model = dataclasses.replace(
        pipeline_spec, mediapipe=dataclasses.replace(pipeline_spec.mediapipe, model_sha256="0" * 64)
    )
    assert output_is_current(out, other_model, None, None) is False

    extract_video(video, tmp_path / "unused.task", out, drop_sim, spec=pipeline_spec, rotation=180)
    assert output_is_current(out, pipeline_spec, drop_sim, 180) is True
    assert output_is_current(out, pipeline_spec, drop_sim, -180) is True  # 같은 각도
    assert output_is_current(out, pipeline_spec, dataclasses.replace(drop_sim, subdir="elsewhere"), 180) is True
    assert output_is_current(out, pipeline_spec, dataclasses.replace(drop_sim, seed=1), 180) is False
    assert output_is_current(out, pipeline_spec, drop_sim, None) is False  # 보정표에서 빠졌다

    out.write_bytes(b"truncated")  # 깨진 파일은 다시 추출한다
    assert output_is_current(out, pipeline_spec, drop_sim, 180) is False
    foreign = tmp_path / "foreign.parquet"
    pq.write_table(pa.table({"t_ms": [1]}), foreign)
    assert output_is_current(foreign, pipeline_spec, None, None) is False


# ---------------------------------------------------------------------------
# 명령 (가짜 랜드마커, --video는 이 프로세스에서 돈다)
# ---------------------------------------------------------------------------


def test_find_videos_and_output_paths(tmp_path: Path) -> None:
    for name in ("b/2.MOV", "a/1.mp4", "a/notes.txt", "c.m4v", "a/deep/3.Mp4"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    found = [p.relative_to(tmp_path).as_posix() for p in find_videos(tmp_path)]
    assert found == ["a/1.mp4", "a/deep/3.Mp4", "b/2.MOV", "c.m4v"]
    assert raw_parquet_path(Path("data/raw/native"), Path("b/2.MOV")) == Path("data/raw/native/b/2.parquet")


def test_cli_video_is_resumable(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    fake_mediapipe: list[FakeLandmarker],
    pipeline_spec: PipelineSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "videos" / "clip.mp4"
    write_plain_video(video, PTS_30FPS)
    argv = ["--config", str(fake_model_setup.config), "--video", str(video)]

    assert extract.main(argv) == 0
    out = fake_model_setup.raw_dir / "native" / "clip.parquet"
    assert [row.t_ms for row in read_raw_frames(out)] == T_30FPS
    captured = capsys.readouterr()
    assert "done clip.mp4" in captured.out and "extracted 1, up to date 0, failed 0" in captured.out
    # --video도 회전을 자동으로 점검한다: 후보 각도마다 랜드마커 하나(4개), 그리고 추출용 하나.
    # 이 가짜 랜드마커는 방향을 구분하지 않으므로(roll이 늘 0) 점검은 회전 메타데이터 0을 유지하고 표시한다.
    assert len(fake_mediapipe) == 5 and all(landmarker.closed for landmarker in fake_mediapipe)
    assert "rotation: auto check chose 0 (metadata 0; ambiguous_kept_metadata; face rate over 1 frames" in captured.out
    assert "WARNING rotation ambiguous: look at the video and pass --rotation" in captured.out
    metadata = read_extract_metadata(out)
    assert (metadata["rotation"], metadata["rotation_override"]) == (0, 0)
    assert metadata["rotation_check"]["reason"] == "ambiguous_kept_metadata"

    modified = out.stat().st_mtime_ns
    assert extract.main(argv) == 0  # 다시 실행하면 건너뛴다(점검도 다시 하지 않는다)
    captured = capsys.readouterr()
    assert "skip clip.mp4" in captured.out and "extracted 0, up to date 1, failed 0" in captured.out
    assert out.stat().st_mtime_ns == modified
    assert len(fake_mediapipe) == 5

    # 누락 모사는 다른 폴더에 쓴다: 설정의 fps15, 설정과 다른 프레임률은 fps{N}
    assert extract.main(argv + ["--fps-sim", "15"]) == 0
    sim = fake_model_setup.raw_dir / "fps15" / "clip.parquet"
    assert read_extract_metadata(sim)["drop_sim"] == {"latency_ms_mean": 66.7, "latency_ms_std": 8.0, "seed": 0}
    assert 5 <= len(list(read_raw_frames(sim))) <= 8
    assert extract.main(argv + ["--fps-sim", "10"]) == 0
    slow = fake_model_setup.raw_dir / "fps10" / "clip.parquet"
    assert read_extract_metadata(slow)["drop_sim"]["latency_ms_mean"] == 100.0
    assert out.stat().st_mtime_ns == modified  # 원본 프레임률 결과는 그대로다
    assert not fake_model_setup.reports_dir.exists()  # --video는 추출 리포트를 쓰지 않는다


def test_cli_applies_rotation_overrides(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    fake_mediapipe: list[FakeLandmarker],
    config_path: Path,
) -> None:
    video = tmp_path / "videos" / "lying.mp4"
    write_plain_video(video, PTS_30FPS[:3], quarter_turns_cw=1)
    config = write_cli_config(
        tmp_path,
        config_path,
        fake_model_setup.spec_path,
        fake_model_setup.model_path,
        {"lying.mp4": 90},
        name="rotated.yaml",
    )
    assert extract.main(["--config", str(config), "--video", str(video)]) == 0
    out = fake_model_setup.raw_dir / "native" / "lying.parquet"
    assert [(row.w, row.h) for row in read_raw_frames(out)] == [(W, H)] * 3
    assert read_extract_metadata(out)["rotation_override"] == 90


def test_cli_reports_failed_video(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    fake_mediapipe: list[FakeLandmarker],
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = tmp_path / "videos" / "broken.mp4"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"this is not a video")
    assert extract.main(["--config", str(fake_model_setup.config), "--video", str(broken)]) == 1
    captured = capsys.readouterr()
    assert "FAILED broken.mp4" in captured.err
    assert "extracted 0, up to date 0, failed 1" in captured.out
    assert not (fake_model_setup.raw_dir / "native" / "broken.parquet").exists()
    assert all(landmarker.closed for landmarker in fake_mediapipe)  # 열어 둔 랜드마커가 없다


def test_cli_cannot_start(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    config_path: Path,
    feature_spec_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert extract.main(["--config", str(fake_model_setup.config), "--input-dir", str(empty_dir)]) == 2
    assert "no video files" in capsys.readouterr().err

    # 모델 파일의 sha256이 spec과 다르면 시작하지 않고 고치는 방법을 알려 준다
    config = write_cli_config(
        tmp_path, config_path, feature_spec_path, fake_model_setup.model_path, name="mismatch.yaml"
    )
    assert extract.main(["--config", str(config), "--input-dir", str(empty_dir)]) == 2
    error = capsys.readouterr().err
    assert "sha256 mismatch" in error and "Fix:" in error

    assert extract.main(["--config", str(fake_model_setup.config), "--video", "x.mp4", "--workers", "0"]) == 2
    assert "--workers" in capsys.readouterr().err
    assert extract.main(["--config", str(tmp_path / "missing.yaml"), "--video", "x.mp4"]) == 2
    with pytest.raises(SystemExit):
        extract.main(["--config", str(fake_model_setup.config)])  # --video와 --input-dir 중 하나는 필요하다
    capsys.readouterr()


# ---------------------------------------------------------------------------
# 실제 MediaPipe (모델 파일이나 라이브러리가 없으면 건너뛴다)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_model(repo_root: Path, pipeline_spec: PipelineSpec) -> Path:
    model = repo_root / "models" / "face_landmarker.task"
    if not model.is_file():
        pytest.skip("models/face_landmarker.task is absent")
    try:
        extract.make_landmarker(model, pipeline_spec).close()
    except ExtractError as exc:
        pytest.skip(f"MediaPipe cannot be used here: {exc}")
    return model


@pytest.mark.mediapipe
def test_real_landmarker_on_non_face_frames(tmp_path: Path, pipeline_spec: PipelineSpec, real_model: Path) -> None:
    landmarker = extract.make_landmarker(real_model, pipeline_spec)
    try:
        frames = [extract.detect_frame(landmarker, 1 + 33 * k, plain_frame(k), pipeline_spec) for k in range(6)]
        with pytest.raises(ValueError):  # VIDEO 모드는 같거나 작은 타임스탬프를 거부한다
            extract.detect_frame(landmarker, 1 + 33 * 5, plain_frame(0), pipeline_spec)
    finally:
        landmarker.close()
    assert frames == [no_face_frame(1 + 33 * k, W, H) for k in range(6)]


@pytest.mark.mediapipe
def test_real_extract_video_on_non_face_video(tmp_path: Path, pipeline_spec: PipelineSpec, real_model: Path) -> None:
    video = tmp_path / "plain.mp4"
    write_plain_video(video, PTS_30FPS)
    out = tmp_path / "plain.parquet"
    stats = extract_video(video, real_model, out, spec=pipeline_spec)

    rows = list(read_raw_frames(out))
    assert rows == [no_face_frame(t, W, H) for t in T_30FPS]  # 모든 행이 face=False, 빈 lm·bs·m
    assert (stats.n_decoded, stats.n_processed, stats.n_face, stats.n_dropped) == (12, 12, 0, 0)
    assert (stats.width, stats.height, stats.first_t_ms, stats.last_t_ms) == (W, H, 1, 368)

    metadata = read_extract_metadata(out)
    assert metadata["model_sha256"] == file_sha256(real_model) == pipeline_spec.mediapipe.model_sha256
    assert metadata["mediapipe_version"] == extract.mediapipe_version()
    assert output_is_current(out, pipeline_spec, None, None) is True

    drop_sim = DropSimConfig(subdir="fps15", latency_ms_mean=66.7, latency_ms_std=8.0, seed=0)
    sim_out = tmp_path / "plain_fps15.parquet"
    sim_stats = extract_video(video, real_model, sim_out, drop_sim, spec=pipeline_spec)
    latencies = list(itertools.islice(drop_sim_latencies(drop_sim), 12))
    expected = [T_30FPS[i] for i in flow_limiter_select(T_30FPS, latencies)]
    assert list(read_raw_frames(sim_out)) == [no_face_frame(t, W, H) for t in expected]
    assert sim_stats.n_dropped == 12 - len(expected)


@pytest.mark.mediapipe
def test_real_cli_input_dir_uses_one_process_per_video(
    tmp_path: Path,
    config_path: Path,
    feature_spec_path: Path,
    real_model: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    videos = tmp_path / "videos"
    write_plain_video(videos / "01" / "0.mp4", PTS_30FPS)
    write_plain_video(videos / "02" / "10.mov", PTS_30FPS[:6])
    (videos / "02" / "readme.txt").write_text("not a video", encoding="utf-8")
    (videos / "03").mkdir()
    (videos / "03" / "5.mp4").write_bytes(b"corrupt")
    config = write_cli_config(tmp_path, config_path, feature_spec_path, real_model)
    argv = ["--config", str(config), "--input-dir", str(videos), "--workers", "2"]

    assert extract.main(argv) == 1  # 깨진 영상 하나가 실패로 보고된다
    captured = capsys.readouterr()
    assert "done 01/0.mp4" in captured.out and "done 02/10.mov" in captured.out
    assert "FAILED 03/5.mp4" in captured.err
    assert "extracted 2, up to date 0, failed 1" in captured.out
    native = tmp_path / "raw" / "native"
    assert [row.t_ms for row in read_raw_frames(native / "01" / "0.parquet")] == T_30FPS
    assert [row.face for row in read_raw_frames(native / "02" / "10.parquet")] == [False] * 6
    assert not (native / "03" / "5.parquet").exists()

    assert extract.main(argv) == 1  # 끝난 영상은 건너뛰고 실패한 것만 다시 시도한다
    captured = capsys.readouterr()
    assert "skip 01/0.mp4" in captured.out and "skip 02/10.mov" in captured.out
    assert "extracted 0, up to date 2, failed 1" in captured.out


# ---------------------------------------------------------------------------
# 회전 점검과 데이터셋 추출 (PLAN 12장, 11장 M3). 검출기는 가짜다.
# ---------------------------------------------------------------------------


def marker_frame(index: int) -> np.ndarray:
    """똑바로 섰을 때 왼쪽 위에 흰 사각형이 있는 그림. 가짜 검출기가 방향을 알아보는 표시다."""
    image = np.full((H, W, 3), 40, dtype=np.uint8)
    image[:40, :40] = 255
    shift = 2 * (index % 20)
    image[60:80, 60 + shift : 80 + shift] = (90, 160, 90)
    return image


def marker_turns_cw(rgb: np.ndarray) -> int:
    """흰 사각형이 있는 모서리로, 그림이 똑바로 선 상태에서 시계 방향으로 몇 번(90도 단위) 돌아가 있는지 알아낸다."""
    corners = [rgb[:20, :20], rgb[:20, -20:], rgb[-20:, -20:], rgb[-20:, :20]]  # 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래
    return int(np.argmax([corner.mean() for corner in corners]))


def write_marker_video(
    path: Path, pts_ms: Sequence[int], quarter_turns_cw: int = 0, display_rotation: int | None = None
) -> None:
    """표시가 있는 그림을 시계 방향으로 quarter_turns_cw번 돌려서 저장한다. display_rotation을 주면 그 회전 메타데이터를 단다."""
    time_base = Fraction(1, 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w", format="mp4")
    stream = container.add_stream("libx264", rate=30)
    first = np.rot90(marker_frame(0), -quarter_turns_cw)
    stream.width, stream.height, stream.pix_fmt = first.shape[1], first.shape[0], "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "20"}
    if display_rotation is not None:
        stream.set_display_rotation(display_rotation)
    for index, value in enumerate(pts_ms):
        image = np.ascontiguousarray(np.rot90(marker_frame(index), -quarter_turns_cw))
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = value
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# 실제 랜드마커가 정면 얼굴 한 장(MediaPipe 공개 시험 이미지)에서 낸 값. 얼굴은 네 방향 모두에서 검출되고, roll은
# 똑바로 서면 −0.6, 시계 방향 90도로 누우면 −90.6, 거꾸로 서면 −1.9(똑바로 선 얼굴로 잘못 맞춰진다),
# 반시계 90도로 누우면 89.4도였다. 키는 그림이 시계 방향으로 돌아간 횟수다.
FAKE_ROLL = {0: -0.6, 1: -90.6, 2: -1.9, 3: 89.4}


def roll_frame(spec: PipelineSpec, t_ms: int, w: int, h: int, roll_deg: float) -> RawFrame:
    """머리 roll만 roll_deg인 얼굴 프레임. 변환 행렬은 z축 회전이다(열 우선: m[0]=cos, m[1]=sin, m[4]=−sin, m[5]=cos)."""
    cos, sin = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    m = (cos, sin, 0.0, 0.0, -sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.0, -30.0, 1.0)
    return dataclasses.replace(hand_frame(spec, t_ms, w, h), m=m)


def frame_roll(frame: RawFrame) -> float:
    return math.degrees(math.atan2(frame.m[1], frame.m[5]))


@pytest.fixture
def marker_mediapipe(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """실제 랜드마커처럼 어느 방향에서나 얼굴을 찾고 방향에 따라 다른 roll을 내는 가짜 랜드마커.

    `upside_down_found`를 False로 바꾸면 거꾸로 선 그림에서는 얼굴을 못 찾는다(실제 랜드마커가 다른 시험 이미지에서
    보인 동작).
    """
    state = SimpleNamespace(created=[], upside_down_found=True)

    def make_landmarker(model_path: Path, spec: PipelineSpec | None = None) -> FakeLandmarker:
        landmarker = FakeLandmarker()
        state.created.append(landmarker)
        return landmarker

    def detect_frame(landmarker: FakeLandmarker, t_ms: int, rgb: np.ndarray, spec: PipelineSpec) -> RawFrame:
        assert not landmarker.closed
        assert all(t_ms > seen_t for seen_t, _, _ in landmarker.seen)
        height, width = rgb.shape[0], rgb.shape[1]
        landmarker.seen.append((t_ms, width, height))
        turns = marker_turns_cw(rgb)
        if turns == 2 and not state.upside_down_found:
            return no_face_frame(t_ms, width, height)
        return roll_frame(spec, t_ms, width, height, FAKE_ROLL[turns])

    monkeypatch.setattr(extract, "make_landmarker", make_landmarker)
    monkeypatch.setattr(extract, "detect_frame", detect_frame)
    monkeypatch.setattr(extract, "mediapipe_version", lambda: "0.0-fake")
    return state


class InlinePool:
    """`ProcessPoolExecutor` 자리에 넣어 작업을 이 프로세스에서 바로 돌린다(가짜 랜드마커는 작업자 프로세스로 넘어가지 않는다)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> InlinePool:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def submit(self, function: Any, *args: Any) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.fixture
def inline_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract, "ProcessPoolExecutor", InlinePool)


def edit_config(config: Path, name: str, **changes: Any) -> Path:
    """설정 파일의 값을 바꾼 사본을 쓴다. 키는 `구역__이름` 꼴이다(예: extraction__rotation_roll_margin_deg=20.0)."""
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    for key, value in changes.items():
        section, field = key.split("__")
        data[section][field] = value
    out = config.with_name(name)
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def write_dataset_config(
    tmp_path: Path,
    config_path: Path,
    setup: SimpleNamespace,
    rldd_root: Path | None,
    rotation_overrides: dict[str, int] | None = None,
    name: str = "dataset.yaml",
) -> Path:
    """가짜 모델 설정에 데이터셋 루트와 짧은 회전 점검 설정을 넣은 설정 파일."""
    config = write_cli_config(tmp_path, config_path, setup.spec_path, setup.model_path, rotation_overrides, name)
    return edit_config(
        config,
        name,
        paths__rldd_root=None if rldd_root is None else str(rldd_root),
        extraction__rotation_check_frames=4,
        extraction__rotation_check_interval_ms=300,
    )


def read_report(reports_dir: Path, tag: str, run: str = "extraction") -> tuple[list[dict[str, str]], str]:
    """추출 리포트의 (CSV 행들, 요약 Markdown)."""
    folder = reports_dir / run / "extraction"
    with (folder / f"{tag}_videos.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, (folder / f"{tag}_summary.md").read_text(encoding="utf-8")


def luma_mean(rgb: np.ndarray) -> float:
    """밝기를 구현과 다른 길로 계산한다: 화소마다 0.299 R + 0.587 G + 0.114 B를 구해 평균한다."""
    return float((rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])).mean())


def rewrite_parquet(
    path: Path, t_ms: Sequence[int] | None = None, metadata_changes: dict[str, Any] | None = None
) -> None:
    """추출 결과 파일의 t_ms 열이나 메타데이터를 손으로 바꿔 다시 쓴다(리포트와 이어 하기 검사용)."""
    import json

    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[extract.METADATA_KEY].decode("utf-8"))
    metadata.update(metadata_changes or {})
    if t_ms is not None:
        table = table.set_column(0, "t_ms", pa.array(np.array(t_ms, dtype=np.int64)))
    table = table.replace_schema_metadata({extract.METADATA_KEY: json.dumps(metadata, sort_keys=True).encode("utf-8")})
    pq.write_table(table, path, compression="zstd", use_dictionary=False)


IDENTITY_MATRIX = struct.pack(">9i", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


def tag_display_matrix(path: Path, a: int, b: int, c: int, d: int) -> None:
    """mp4의 트랙 머리(tkhd)에 있는 표시 행렬 [[a, b], [c, d]]를 바꿔 쓴다(값은 −1, 0, 1). ffmpeg 명령 없이 뒤집기를 단다."""
    data = bytearray(path.read_bytes())
    tkhd = data.find(b"tkhd")
    at = data.find(IDENTITY_MATRIX, tkhd)
    assert tkhd > 0 and 0 < at - tkhd < 80, "the file does not hold an identity display matrix"
    data[at : at + 36] = struct.pack(">9i", a << 16, b << 16, 0, c << 16, d << 16, 0, 0, 0, 0x40000000)
    path.write_bytes(bytes(data))


PTS_10FPS = [100 * k for k in range(20)]  # 20 프레임, 2초


def test_sample_frames_takes_spaced_frames_and_stops_reading() -> None:
    consumed: list[int] = []

    def frames() -> Any:
        for t_ms in range(1, 3000, 33):
            consumed.append(t_ms)
            yield t_ms, np.zeros((2, 2, 3), dtype=np.uint8)

    samples = extract.sample_frames(frames(), 4, 200)
    assert [t_ms for t_ms, _ in samples] == [1, 232, 463, 694]
    assert consumed[-1] == 694  # 필요한 만큼 고르면 더 읽지 않는다
    short = extract.sample_frames(((t, np.zeros((2, 2, 3), dtype=np.uint8)) for t in (1, 50, 400)), 5, 300)
    assert [t_ms for t_ms, _ in short] == [1, 400]
    assert extract.sample_frames(iter(()), 3, 100) == []
    with pytest.raises(ValueError):
        extract.sample_frames(iter(()), 0, 100)


def test_median_angle_handles_the_wrap_at_180_degrees() -> None:
    assert extract.median_angle([10.0, 40.0, 20.0]) == pytest.approx(20.0)
    assert extract.median_angle([-2.0, 4.0]) == pytest.approx(1.0)  # 짝수 개: 가운데 두 값의 평균
    assert extract.median_angle([5.0]) == pytest.approx(5.0)
    assert extract.median_angle([-90.6, -88.9, -90.6, -90.2]) == pytest.approx(-90.4)
    # ±180 근처: 보통의 중앙값은 0(179와 −179의 평균)이나 −170이 되지만 실제 방향은 180 근처다
    assert extract.median_angle([179.0, -179.0]) == pytest.approx(180.0)
    assert extract.median_angle([-170.0, 170.0, 175.0]) == pytest.approx(175.0)
    assert extract.median_angle([-175.0, 170.0, -170.0]) == pytest.approx(-175.0)
    with pytest.raises(ValueError):
        extract.median_angle([])


def test_rotation_rates_and_rolls_with_stub_detector() -> None:
    # 시계 방향으로 한 번 돌려 저장된 그림: 반시계로 90도 돌려야 선다.
    samples = [(1 + 100 * k, np.ascontiguousarray(np.rot90(marker_frame(k), -1))) for k in range(5)]
    calls: list[tuple[int, int, tuple[int, int]]] = []

    def stub(rotation: int, t_ms: int, rgb: np.ndarray) -> float | None:
        calls.append((rotation, t_ms, (rgb.shape[0], rgb.shape[1])))
        return FAKE_ROLL[marker_turns_cw(rgb)]

    rates, rolls = extract.rotation_rates_and_rolls(samples, stub)
    assert rates == {0: 1.0, 90: 1.0, 180: 1.0, 270: 1.0}  # 실제 랜드마커처럼 어느 방향에서나 검출된다
    # 후보 0은 그림이 시계 방향 90도로 누운 채, 90은 똑바로, 180은 반시계 90도로 누운 채, 270은 거꾸로다.
    expected = ((0, -90.6), (90, -0.6), (180, 89.4), (270, -1.9))
    assert rolls == {rotation: pytest.approx(roll) for rotation, roll in expected}
    # 후보마다 표본 전체를 시각순으로, 그 각도로 돌린 그림으로 넣는다.
    assert [(rotation, t_ms) for rotation, t_ms, _ in calls] == [
        (rotation, 1 + 100 * k) for rotation in (0, 90, 180, 270) for k in range(5)
    ]
    assert {shape for rotation, _, shape in calls if rotation in (0, 180)} == {(W, H)}
    assert {shape for rotation, _, shape in calls if rotation in (90, 270)} == {(H, W)}

    # 일부 프레임에서만 검출되는 경우: 비율은 검출된 표본의 비율, roll은 검출된 표본만의 중앙값
    def partial(rotation: int, t_ms: int, rgb: np.ndarray) -> float | None:
        if rotation == 180:
            return None
        if rotation == 90:
            return {1: 3.0, 101: -1.0, 201: 8.0}.get(t_ms)
        if rotation == 270:  # ±180 근처에 걸친 roll: 보통의 중앙값은 0이지만 실제 방향은 180이다
            return {1: 179.0, 101: -179.0, 201: 178.0, 301: -178.0}.get(t_ms)
        return 50.0 if t_ms < 150 else None

    rates, rolls = extract.rotation_rates_and_rolls(samples, partial)
    assert rates == {0: 0.4, 90: 0.6, 180: 0.0, 270: 0.8}
    assert rolls == {0: pytest.approx(50.0), 90: pytest.approx(3.0), 180: None, 270: pytest.approx(180.0)}
    assert extract.rotation_rates_and_rolls([], stub) == (
        {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0},
        {0: None, 90: None, 180: None, 270: None},
    )


ALL_FOUND = {0: 1.0, 90: 1.0, 180: 1.0, 270: 1.0}
NONE_FOUND = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
NO_ROLLS = {0: None, 90: None, 180: None, 270: None}
FOUND_0_90 = {0: 1.0, 90: 1.0, 180: 0.0, 270: 0.0}
ROLLS_0_90 = {0: 2.0, 90: 88.0, 180: None, 270: None}


@pytest.mark.parametrize(
    ("rates", "rolls", "metadata", "n_frames", "expected"),
    [
        # 시계 방향 90도로 누워 저장되고 메타데이터가 없는 영상(실측 roll). 90과 270을 가르지 못하고 메타데이터 0은 그 둘이 아니다.
        (ALL_FOUND, {0: -90.6, 90: -0.6, 180: 89.4, 270: -1.9}, 0, 12, (90, True, "ambiguous_smallest_roll")),
        # 같은 영상에 메타데이터가 270이나 90이면 가르지 못한 둘 안에 있으므로 메타데이터를 유지한다.
        (ALL_FOUND, {0: -90.6, 90: -0.6, 180: 89.4, 270: -1.9}, 270, 12, (270, True, "ambiguous_kept_metadata")),
        (ALL_FOUND, {0: -90.6, 90: -0.6, 180: 89.4, 270: -1.9}, 90, 12, (90, True, "ambiguous_kept_metadata")),
        # 똑바로 저장된 영상에 틀린 메타데이터 90: 0과 180을 가르지 못하고 90은 그 둘이 아니다.
        (ALL_FOUND, {0: -0.6, 90: 89.4, 180: -1.9, 270: -90.6}, 90, 12, (0, True, "ambiguous_smallest_roll")),
        # 거꾸로 선 얼굴이 검출되지 않는 경우(다른 시험 이미지의 실측): 180도로 저장된 영상을 roll로 가른다.
        (
            {0: 0.0, 90: 1.0, 180: 1.0, 270: 10 / 12},
            {0: None, 90: -97.1, 180: -3.0, 270: 82.7},
            0,
            12,
            (180, False, "smallest_roll"),
        ),
        # 표본 한 장 차이는 동률이다: 검출 비율은 90이 더 높지만 roll이 0을 가리킨다.
        ({0: 11 / 12, 90: 1.0, 180: 0.0, 270: 0.0}, ROLLS_0_90, 0, 12, (0, False, "smallest_roll")),
        # 두 장 차이는 동률이 아니다: 검출 비율이 정한다.
        ({0: 10 / 12, 90: 1.0, 180: 0.0, 270: 0.0}, ROLLS_0_90, 0, 12, (90, False, "best_rate")),
        (
            {0: 0.5, 90: 1.0, 180: 0.2, 270: 0.0},
            {0: 1.0, 90: 80.0, 180: 5.0, 270: None},
            0,
            10,
            (90, False, "best_rate"),
        ),
        # 어느 각도에서도 얼굴이 없으면 메타데이터를 유지하고 표시한다.
        (NONE_FOUND, NO_ROLLS, 270, 12, (270, True, "no_face")),
        (NONE_FOUND, NO_ROLLS, 0, 0, (0, True, "no_face")),
        # 머리를 30도 기울인 얼굴(실측): 옆으로 누운 후보(60.7)는 31도 떨어져 갈리고 0과 180만 남는다.
        (
            {0: 1.0, 90: 0.0, 180: 1.0, 270: 1.0},
            {0: 29.3, 90: None, 180: 29.9, 270: -60.7},
            0,
            12,
            (0, True, "ambiguous_kept_metadata"),
        ),
        # 여유(15도)의 경계: 차이가 정확히 15이면 갈린 것이고 14.9이면 가르지 못한 것이다.
        (FOUND_0_90, {0: 10.0, 90: 25.0, 180: None, 270: None}, 90, 8, (0, False, "smallest_roll")),
        (FOUND_0_90, {0: 10.0, 90: 24.9, 180: None, 270: None}, 90, 8, (90, True, "ambiguous_kept_metadata")),
        (FOUND_0_90, {0: 10.0, 90: 24.9, 180: None, 270: None}, 180, 8, (0, True, "ambiguous_smallest_roll")),
        # 방향을 제대로 알아보는 검출기라면 거꾸로 선 얼굴의 roll은 ±180 근처다: 그때는 갈린다.
        (
            {0: 1.0, 90: 0.0, 180: 1.0, 270: 0.0},
            {0: 179.0, 90: None, 180: -1.0, 270: None},
            0,
            8,
            (180, False, "smallest_roll"),
        ),
        # |roll|까지 같고 메타데이터가 그 안에 없으면 작은 각도
        (
            {0: 0.0, 90: 1.0, 180: 0.0, 270: 1.0},
            {0: None, 90: 5.0, 180: None, 270: -5.0},
            0,
            8,
            (90, True, "ambiguous_smallest_roll"),
        ),
    ],
)
def test_choose_rotation_hand_worked_cases(
    rates: dict[int, float],
    rolls: dict[int, float | None],
    metadata: int,
    n_frames: int,
    expected: tuple[int, bool, str],
) -> None:
    choice = extract.choose_rotation(rates, rolls, metadata, n_frames, roll_margin_deg=15.0)
    assert (choice.rotation, choice.ambiguous, choice.reason) == expected


def test_choose_rotation_margin_changes_the_flag() -> None:
    rolls = {0: 12.0, 90: None, 180: 30.0, 270: None}
    rates = {0: 1.0, 90: 0.0, 180: 1.0, 270: 0.0}
    assert extract.choose_rotation(rates, rolls, 180, 6, roll_margin_deg=15.0).rotation == 0  # 18도 차이는 갈린다
    wide = extract.choose_rotation(rates, rolls, 180, 6, roll_margin_deg=20.0)
    assert (wide.rotation, wide.ambiguous) == (180, True)


# (저장할 때 시계 방향으로 돌린 횟수, 회전 메타데이터, 거꾸로 선 얼굴이 검출되는지) → (고른 회전, 가르지 못함, 사유)
STORED_VIDEO_CASES = [
    ((0, None, True), (0, True, "ambiguous_kept_metadata")),
    ((1, None, True), (90, True, "ambiguous_smallest_roll")),
    ((2, None, True), (0, True, "ambiguous_kept_metadata")),  # 틀린 각도지만 추측하지 않고 표시한다
    ((3, None, True), (270, True, "ambiguous_smallest_roll")),
    ((0, 90, True), (0, True, "ambiguous_smallest_roll")),  # 틀린 메타데이터 90
    ((1, 270, True), (270, True, "ambiguous_kept_metadata")),  # 정반대로 틀린 메타데이터는 표시만 된다
    ((1, 90, True), (90, True, "ambiguous_kept_metadata")),  # 맞는 메타데이터
    ((0, None, False), (0, False, "smallest_roll")),
    ((1, None, False), (90, False, "smallest_roll")),
    ((2, None, False), (180, False, "smallest_roll")),
    ((3, None, False), (270, False, "smallest_roll")),
    ((0, 90, False), (0, False, "smallest_roll")),
    ((1, 270, False), (90, False, "smallest_roll")),
]


@pytest.mark.parametrize(("case", "expected"), STORED_VIDEO_CASES)
def test_check_rotation_on_a_stored_video(
    tmp_path: Path,
    pipeline_spec: PipelineSpec,
    marker_mediapipe: SimpleNamespace,
    case: tuple[int, int | None, bool],
    expected: tuple[int, bool, str],
) -> None:
    quarter_turns_cw, tag, upside_down_found = case
    marker_mediapipe.upside_down_found = upside_down_found
    video = tmp_path / "clip.mp4"
    write_marker_video(video, PTS_10FPS, quarter_turns_cw, display_rotation=tag)
    check = extract.check_rotation(
        video, tmp_path / "unused.task", pipeline_spec, n_frames=5, interval_ms=400, roll_margin_deg=15.0
    )

    assert (check.rotation, check.ambiguous, check.reason) == expected
    assert (check.metadata_rotation, check.n_frames) == (tag or 0, 5)
    # 후보 각도로 돌린 뒤에 그림이 시계 방향으로 남아 있는 횟수가 가짜 검출기의 roll을 정한다.
    for rotation in (0, 90, 180, 270):
        left = (quarter_turns_cw - rotation // 90) % 4
        if left == 2 and not upside_down_found:
            assert (check.rates[rotation], check.rolls[rotation]) == (0.0, None)
        else:
            assert check.rates[rotation] == 1.0
            assert check.rolls[rotation] == pytest.approx(FAKE_ROLL[left], abs=1e-9)
    # 후보마다 랜드마커를 하나씩 만들어 표본 5장을 시각순으로 넣고 닫는다.
    created = marker_mediapipe.created
    assert len(created) == 4 and all(landmarker.closed for landmarker in created)
    assert all([t for t, _, _ in landmarker.seen] == [1, 401, 801, 1201, 1601] for landmarker in created)


def test_check_rotation_errors(tmp_path: Path, pipeline_spec: PipelineSpec, marker_mediapipe: SimpleNamespace) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(Exception):  # 디코딩 오류는 PyAV의 예외 그대로 올라온다
        extract.check_rotation(broken, tmp_path / "unused.task", pipeline_spec, 5, 400, 15.0)

    # 표시 행렬에 뒤집기가 있으면 회전 메타데이터를 믿을 수 없으므로 점검하지 않고 오류로 알린다.
    flipped = tmp_path / "flipped.mp4"
    write_marker_video(flipped, PTS_10FPS[:4])
    tag_display_matrix(flipped, -1, 0, 0, 1)
    with pytest.raises(VideoError, match="display matrix contains a flip"):
        extract.check_rotation(flipped, tmp_path / "unused.task", pipeline_spec, 5, 400, 15.0)
    assert marker_mediapipe.created == []


def test_extract_video_extra_metadata(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]
) -> None:
    video = tmp_path / "clip.mp4"
    write_plain_video(video, PTS_30FPS[:4])
    out = tmp_path / "clip.parquet"
    extract_video(video, tmp_path / "m.task", out, spec=pipeline_spec, extra_metadata={"note": {"a": 1}})
    assert read_extract_metadata(out)["note"] == {"a": 1}
    with pytest.raises(ValueError, match="must not redefine"):
        extract_video(video, tmp_path / "m.task", out, spec=pipeline_spec, extra_metadata={"rotation": 5})
    with pytest.raises(ValueError, match="must not redefine"):
        extract_video(video, tmp_path / "m.task", out, spec=pipeline_spec, extra_metadata={"brightness_mean": 1.0})


# ---------------------------------------------------------------------------
# 밝기와 녹화 요약 (PLAN 6장 교란 점검 (1)의 입력)
# ---------------------------------------------------------------------------


def test_frame_brightness_is_the_mean_bt601_luma() -> None:
    constant = np.empty((6, 8, 3), dtype=np.uint8)
    constant[:] = (200, 100, 50)
    # 0.299·200 + 0.587·100 + 0.114·50 = 59.8 + 58.7 + 5.7
    assert extract.frame_brightness(constant) == pytest.approx(124.2, abs=1e-9)
    assert extract.frame_brightness(np.zeros((4, 4, 3), dtype=np.uint8)) == 0.0
    assert extract.frame_brightness(np.full((4, 4, 3), 255, dtype=np.uint8)) == pytest.approx(255.0, abs=1e-9)

    halves = np.zeros((4, 10, 3), dtype=np.uint8)
    halves[:, :5] = (255, 0, 0)  # 왼쪽 절반 빨강: 0.299·255 = 76.245
    halves[:, 5:] = (0, 0, 255)  # 오른쪽 절반 파랑: 0.114·255 = 29.07
    assert extract.frame_brightness(halves) == pytest.approx((76.245 + 29.07) / 2, abs=1e-9)
    # 채널 순서가 RGB다: 같은 값이 초록 자리에 있으면 0.587배다
    green = np.zeros((2, 2, 3), dtype=np.uint8)
    green[..., 1] = 100
    assert extract.frame_brightness(green) == pytest.approx(58.7, abs=1e-9)
    # 큰 프레임에서도 합이 넘치지 않는다(uint8 합으로 계산하면 틀린다)
    big = np.full((1080, 1920, 3), 250, dtype=np.uint8)
    assert extract.frame_brightness(big) == pytest.approx(250.0, abs=1e-9)
    # 세운 프레임을 돌려도 평균은 같다
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    assert extract.frame_brightness(noise) == pytest.approx(luma_mean(noise), abs=1e-9)
    assert extract.frame_brightness(np.ascontiguousarray(np.rot90(noise))) == pytest.approx(luma_mean(noise), abs=1e-9)

    with pytest.raises(ValueError, match="RGB image"):
        extract.frame_brightness(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB image"):
        extract.frame_brightness(np.zeros((0, 4, 3), dtype=np.uint8))


def write_gray_video(path: Path, pts_ms: Sequence[int], levels: Sequence[int]) -> None:
    """프레임마다 한 가지 회색(levels의 값)으로 채운 영상. 이 회색들은 H.264 부호화를 거쳐도 값이 그대로다."""
    time_base = Fraction(1, 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "20"}
    for value, level in zip(pts_ms, levels, strict=True):
        frame = av.VideoFrame.from_ndarray(np.full((H, W, 3), level, dtype=np.uint8), format="rgb24")
        frame.pts = value
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_extract_video_records_brightness_and_source_timing(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]
) -> None:
    video = tmp_path / "gray.mp4"
    levels = [255, 255, 0, 128, 64, 16, 235]
    pts = [0, 33, 67, 100, 133, 167, 400]
    write_gray_video(video, pts, levels)
    decoded = [int(rgb[0, 0, 0]) for _, rgb in iter_video_frames(video)]
    assert decoded == levels and all(np.all(rgb == rgb[0, 0, 0]) for _, rgb in iter_video_frames(video))

    out = tmp_path / "gray.parquet"
    extract_video(video, tmp_path / "unused.task", out, spec=pipeline_spec)
    metadata = read_extract_metadata(out)
    # 한 가지 회색 프레임의 luma 평균은 그 회색 값이다. 평균 (255+255+0+128+64+16+235)/7 = 953/7, 중앙값 128
    assert metadata["brightness_mean"] == pytest.approx(953 / 7, abs=1e-9)
    assert metadata["brightness_median"] == pytest.approx(128.0, abs=1e-9)
    # 프레임 시각 1, 34, 68, 101, 134, 168, 401: 길이 400 ms, 평균 6·1000/400 = 15 fps, 간격 33~233 ms
    assert (metadata["duration_ms"], metadata["mean_fps"]) == (400, pytest.approx(15.0))
    assert (metadata["min_interval_ms"], metadata["max_interval_ms"], metadata["variable_intervals"]) == (33, 233, True)
    assert (metadata["width"], metadata["height"], metadata["display_flip"]) == (W, H, False)
    assert metadata["extract_version"] == extract.EXTRACT_VERSION == 2
    assert metadata["schema_version"] == extract.SCHEMA_VERSION == 1  # 열은 그대로다

    # 누락 모사(지연 70 ms). f0@1은 1~71에 처리, f1@34는 f2@68에 밀려 버려진다. f2는 71~141, f3@101은 f4@134에 밀린다.
    # f4는 141~211, f5@168은 대기하다 211~281에 처리되고, f6@401은 바로 처리된다: 0, 2, 4, 5, 6번이 남는다.
    # 밝기는 처리한 프레임(255, 0, 64, 16, 235)만 보고, 시각 요약은 원본 영상의 것 그대로다.
    constant = DropSimConfig(subdir="x", latency_ms_mean=70.0, latency_ms_std=0.0, seed=0)
    sim_out = tmp_path / "gray_sim.parquet"
    extract_video(video, tmp_path / "unused.task", sim_out, constant, spec=pipeline_spec)
    sim = read_extract_metadata(sim_out)
    assert [row.t_ms for row in read_raw_frames(sim_out)] == [1, 68, 134, 168, 401]
    assert (sim["n_processed"], sim["n_dropped"]) == (5, 2)
    assert sim["brightness_mean"] == pytest.approx((255 + 0 + 64 + 16 + 235) / 5, abs=1e-9)
    assert sim["brightness_median"] == pytest.approx(64.0, abs=1e-9)
    assert (sim["duration_ms"], sim["mean_fps"], sim["min_interval_ms"], sim["max_interval_ms"]) == (
        400,
        pytest.approx(15.0),
        33,
        233,
    )

    # 일정한 간격(1 ms 흔들림)은 변하는 것으로 보지 않는다. 프레임이 하나면 길이와 프레임률이 0이다.
    steady = tmp_path / "steady.mp4"
    write_gray_video(steady, PTS_30FPS, [30] * 12)
    extract_video(steady, tmp_path / "unused.task", out, spec=pipeline_spec)
    metadata = read_extract_metadata(out)
    assert (metadata["min_interval_ms"], metadata["max_interval_ms"], metadata["variable_intervals"]) == (33, 34, False)
    assert metadata["mean_fps"] == pytest.approx(11 * 1000 / 367)
    assert (metadata["brightness_mean"], metadata["brightness_median"]) == (pytest.approx(30.0), pytest.approx(30.0))
    single = tmp_path / "single.mp4"
    write_gray_video(single, [0], [128])
    extract_video(single, tmp_path / "unused.task", out, spec=pipeline_spec)
    metadata = read_extract_metadata(out)
    assert (metadata["duration_ms"], metadata["mean_fps"], metadata["min_interval_ms"]) == (0, 0.0, 0)


def test_extraction_is_byte_identical_when_repeated(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]
) -> None:
    """처리 시간처럼 실행마다 달라지는 값은 결과 파일에 적지 않는다."""
    video = tmp_path / "clip.mp4"
    write_plain_video(video, PTS_30FPS)
    first, second = tmp_path / "a" / "clip.parquet", tmp_path / "b" / "clip.parquet"
    extract_video(video, tmp_path / "unused.task", first, spec=pipeline_spec)
    extract_video(video, tmp_path / "unused.task", second, spec=pipeline_spec)
    assert first.read_bytes() == second.read_bytes()


def test_results_without_the_recording_summary_are_not_current(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]
) -> None:
    video = tmp_path / "clip.mp4"
    write_plain_video(video, PTS_30FPS[:4])
    out = tmp_path / "clip.parquet"
    extract_video(video, tmp_path / "unused.task", out, spec=pipeline_spec)
    assert output_is_current(out, pipeline_spec, None, None) is True
    rewrite_parquet(out, metadata_changes={"extract_version": 1})  # 밝기를 적기 전의 판으로 만든 결과
    assert output_is_current(out, pipeline_spec, None, None) is False
    assert [row.t_ms for row in read_raw_frames(out)] == T_30FPS[:4]  # 읽기는 그대로 된다


# ---------------------------------------------------------------------------
# 데이터셋 추출과 추출 리포트 (가짜 랜드마커, 작업자 풀은 이 프로세스에서 돈다)
# ---------------------------------------------------------------------------

# 과제가 요구한 리포트의 열. 구현의 `REPORT_COLUMNS`와 따로 손으로 적었다.
REQUIRED_REPORT_COLUMNS = [
    "video",
    "status",
    "reason",
    "frames_decoded",
    "frames_kept",
    "duration_ms",
    "mean_fps",
    "min_interval_ms",
    "max_interval_ms",
    "variable_intervals",
    "width",
    "height",
    "rotation",
    "rate_0",
    "rate_90",
    "rate_180",
    "rate_270",
    "roll_0",
    "roll_90",
    "roll_180",
    "roll_270",
    "rotation_ambiguous",
    "display_flip",
    "face_rate",
    "ms_per_frame",
    "brightness_mean",
    "brightness_median",
    "model_sha256",
    "t_ms_ok",
]


def test_cli_dataset_pilot_checks_rotation_and_mirrors_the_manifest_layout(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "rldd"
    write_marker_video(root / "Fold1_part1" / "01" / "0.mp4", PTS_10FPS, quarter_turns_cw=1)
    write_marker_video(root / "Fold1_part1" / "01" / "5.MOV", PTS_10FPS)
    write_marker_video(root / "Fold1_part1" / "01" / "10.mp4", PTS_10FPS, quarter_turns_cw=2)
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root)
    native = fake_model_setup.raw_dir / "native"

    # 시험 추출: 목록의 앞 2개만(0, 5), 이 프로세스에서 차례로.
    assert extract.main(["--config", str(config), "--dataset", "--pilot", "2"]) == 0
    out = capsys.readouterr().out
    written = sorted(path.relative_to(native).as_posix() for path in native.rglob("*.parquet"))
    assert written == ["Fold1_part1/01/0.parquet", "Fold1_part1/01/5.parquet"]  # 목록의 배치 그대로
    assert "done Fold1_part1/01/0.mp4" in out
    assert "auto check chose 90 (metadata 0; ambiguous_smallest_roll; face rate over 4 frames" in out
    assert "median roll 0:-90.6 90:-0.6 180:89.4 270:-1.9" in out
    assert "done Fold1_part1/01/5.MOV" in out and "auto check chose 0 (metadata 0; ambiguous_kept_metadata;" in out
    assert out.count("WARNING rotation ambiguous") == 2
    assert out.count("ms/frame") == 3 and "face=1.000" in out
    assert "pilot: 2 videos, 40 frames," in out and "rotations applied [0, 90]" in out
    assert "extracted 2, up to date 0, failed 0" in out

    lying = native / "Fold1_part1" / "01" / "0.parquet"
    metadata = read_extract_metadata(lying)
    assert (metadata["rotation"], metadata["rotation_override"]) == (90, 90)
    recorded = metadata["rotation_check"]
    assert recorded["rates"] == {"0": 1.0, "90": 1.0, "180": 1.0, "270": 1.0}
    expected_rolls = {"0": -90.6, "90": -0.6, "180": 89.4, "270": -1.9}
    assert recorded["rolls"] == {key: pytest.approx(value) for key, value in expected_rolls.items()}
    assert {key: recorded[key] for key in ("version", "metadata_rotation", "n_frames", "ambiguous", "reason")} == {
        "version": 2,
        "metadata_rotation": 0,
        "n_frames": 4,
        "ambiguous": True,
        "reason": "ambiguous_smallest_roll",
    }
    assert recorded["settings"] == {"frames": 4, "interval_ms": 300, "roll_margin_deg": 15.0}
    rows = list(read_raw_frames(lying))
    assert [(row.w, row.h, row.face) for row in rows] == [(W, H, True)] * 20  # 세워서 추출했다
    assert all(frame_roll(row) == pytest.approx(-0.6) for row in rows)

    # 시험 추출의 리포트: 앞 2개만 적는다.
    report, summary = read_report(fake_model_setup.reports_dir, "pilot_native")
    assert [row["video"] for row in report] == ["Fold1_part1/01/0.mp4", "Fold1_part1/01/5.MOV"]
    assert [row["status"] for row in report] == ["done", "done"]
    assert summary.splitlines()[0].startswith(f"# Data source: labelled video dataset at {root} ")
    assert summary.splitlines()[0].endswith("native frame rate extraction; pilot extraction of the first 2 file(s)")
    assert "시험 추출이다: 목록의 파일 3개 중 앞 2개만" in summary
    assert "- 이번 실행에서 잰 것: 영상 2개, 프레임 40개, 프레임당 " in summary
    assert "이번 실행은 영상을 한 번에 하나씩 처리했다" in summary and "- 전 리포트에서 이어 온 것: 없음." in summary

    # N을 생략하면 설정의 pilot_videos(6)이고, 이미 있는 결과도 다시 만든다.
    before = lying.stat().st_mtime_ns
    assert extract.main(["--config", str(config), "--dataset", "--pilot"]) == 0
    out = capsys.readouterr().out
    assert "pilot: 3 videos, 60 frames," in out and "rotations applied [0, 90]" in out
    assert "skip Fold1" not in out and lying.stat().st_mtime_ns != before
    # 거꾸로 저장된 영상(10.mp4)은 가짜 검출기가 실제 랜드마커처럼 0도와 180도를 가르지 못한다:
    # 추측하지 않고 메타데이터(0)를 유지한 채 표시한다.
    upside_down = read_extract_metadata(native / "Fold1_part1" / "01" / "10.parquet")
    assert upside_down["rotation_override"] == 0
    assert (upside_down["rotation_check"]["ambiguous"], upside_down["rotation_check"]["reason"]) == (
        True,
        "ambiguous_kept_metadata",
    )

    # 전체 추출: 자동 점검으로 만든 결과가 있으면 건너뛴다(랜드마커를 만들지 않는다).
    landmarkers = len(marker_mediapipe.created)
    assert extract.main(["--config", str(config), "--dataset"]) == 0
    out = capsys.readouterr().out
    assert out.count("skip Fold1_part1/01/") == 3 and "extracted 0, up to date 3, failed 0" in out
    assert len(marker_mediapipe.created) == landmarkers
    report, summary = read_report(fake_model_setup.reports_dir, "dataset_native")  # 모두 건너뛰어도 리포트는 쓴다
    assert [row["status"] for row in report] == ["up_to_date"] * 3
    assert [row["rotation_ambiguous"] for row in report] == ["true"] * 3
    # 처리 시간은 시험 추출의 리포트에서 잇는다(결과 파일에는 시간이 없다).
    pilot_report, _ = read_report(fake_model_setup.reports_dir, "pilot_native")
    assert [row["ms_per_frame"] for row in report] == [row["ms_per_frame"] for row in pilot_report]
    assert all(float(row["ms_per_frame"]) > 0.0 for row in report)
    assert "- 이번 실행에서 잰 것: 없음." in summary
    # 어디서 잰 값인지 적는다: 시험 추출 리포트의 값이다.
    assert "- 전 리포트에서 이어 온 것(시험 추출 리포트 `pilot_native_videos.csv`): 영상 3개, 프레임 60개, 프레임당 " in summary
    assert "이 리포트의 전 판 `dataset_native_videos.csv`" not in summary
    assert "이번 실행은" not in summary  # 이번 실행에서 잰 값이 없으면 처리 방식을 말하지 않는다

    # 점검 설정이 바뀌면 전에 만든 결과를 최신으로 보지 않는다.
    wider = edit_config(config, "wider.yaml", extraction__rotation_roll_margin_deg=20.0)
    assert extract.main(["--config", str(wider), "--dataset"]) == 0
    assert "extracted 3, up to date 0, failed 0" in capsys.readouterr().out
    assert read_extract_metadata(lying)["rotation_check"]["settings"]["roll_margin_deg"] == 20.0
    # 점검 규칙의 판이 다른(전 규칙으로 만든) 결과도 다시 만든다. 설정은 같고 판만 다르다.
    older = dict(read_extract_metadata(lying)["rotation_check"], version=1)
    rewrite_parquet(lying, metadata_changes={"rotation_check": older})
    assert extract.main(["--config", str(wider), "--dataset"]) == 0
    assert "extracted 1, up to date 2, failed 0" in capsys.readouterr().out

    # 누락 모사 추출은 다른 폴더에 같은 배치로 쓰고 리포트 이름도 다르다.
    assert extract.main(["--config", str(config), "--dataset", "--pilot", "1", "--fps-sim", "15"]) == 0
    capsys.readouterr()
    sim = fake_model_setup.raw_dir / "fps15" / "Fold1_part1" / "01" / "0.parquet"
    assert read_extract_metadata(sim)["drop_sim"]["latency_ms_mean"] == 66.7
    assert read_extract_metadata(sim)["rotation_override"] == 90
    _, sim_summary = read_report(fake_model_setup.reports_dir, "pilot_fps15")
    assert "device frame-drop simulation 'fps15' (latency mean 66.7 ms)" in sim_summary.splitlines()[0]
    names = sorted(path.name for path in (fake_model_setup.reports_dir / "extraction" / "extraction").iterdir())
    assert names == sorted(
        f"{tag}_{part}"
        for tag in ("dataset_native", "pilot_native", "pilot_fps15")
        for part in ("config.yaml", "summary.md", "videos.csv")
    )


def test_cli_dataset_rotation_resolved_when_upside_down_faces_are_not_found(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker_mediapipe.upside_down_found = False
    root = tmp_path / "rldd"
    write_marker_video(root / "02" / "0.mp4", PTS_10FPS, quarter_turns_cw=2)
    write_marker_video(root / "02" / "5.mp4", PTS_10FPS, quarter_turns_cw=3)
    write_marker_video(root / "02" / "10.mp4", PTS_10FPS)
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root)
    assert extract.main(["--config", str(config), "--dataset"]) == 0
    out = capsys.readouterr().out
    assert "auto check chose 180 (metadata 0; smallest_roll;" in out
    assert out.count("WARNING rotation differs from the video metadata") == 2 and "ambiguous" not in out

    report, summary = read_report(fake_model_setup.reports_dir, "dataset_native")
    by_video = {row["video"]: row for row in report}
    assert [by_video[name]["rotation"] for name in ("02/0.mp4", "02/5.mp4", "02/10.mp4")] == ["180", "270", "0"]
    assert [row["rotation_ambiguous"] for row in report] == ["false"] * 3
    assert by_video["02/0.mp4"]["warnings"] == "rotation_differs_from_metadata"
    assert by_video["02/10.mp4"]["warnings"] == ""
    # 거꾸로 저장된 영상에서 후보 0은 거꾸로인 채라 검출이 없다(비율 0, roll 없음).
    assert (by_video["02/0.mp4"]["rate_0"], by_video["02/0.mp4"]["roll_0"]) == ("0", "")
    assert float(by_video["02/0.mp4"]["roll_180"]) == pytest.approx(-0.6)
    assert "### 점검이 회전 메타데이터와 다른 각도를 골랐음 (가르지 못한 경우는 위에 있다)\n\n2개." in summary
    upright_rows = read_raw_frames(fake_model_setup.raw_dir / "native" / "02" / "0.parquet")
    assert all(frame_roll(row) == pytest.approx(-0.6) for row in upright_rows)


def test_cli_dataset_report_rows_failures_and_resume(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "rldd"
    folder = root / "Fold1_part1" / "01"
    write_marker_video(folder / "0.mp4", PTS_10FPS, quarter_turns_cw=1)
    write_marker_video(folder / "5.MOV", PTS_10FPS)
    write_marker_video(folder / "10_1.mp4", [0, 100, 200, 400, 500])
    (folder / "10_2.mp4").write_bytes(b"this is not a video")
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root)
    argv = ["--config", str(config), "--dataset", "--run-name", "fix_report"]

    assert extract.main(argv) == 1  # 깨진 파일 하나가 실패한다. 그래도 리포트는 쓴다
    captured = capsys.readouterr()
    assert "FAILED Fold1_part1/01/10_2.mp4" in captured.err and "extracted 3, up to date 0, failed 1" in captured.out
    report_dir = fake_model_setup.reports_dir / "fix_report" / "extraction"
    assert f"report: {report_dir / 'dataset_native_summary.md'} (table: dataset_native_videos.csv)" in captured.out
    # 숫자와 글만 쓴다. 그림 파일은 없다(얼굴 화소를 저장하지 않는다는 PLAN 1.2절의 결정 D13).
    assert sorted(path.name for path in report_dir.iterdir()) == [
        "dataset_native_config.yaml",
        "dataset_native_summary.md",
        "dataset_native_videos.csv",
    ]
    assert yaml.safe_load((report_dir / "dataset_native_config.yaml").read_text(encoding="utf-8")) == yaml.safe_load(
        config.read_text(encoding="utf-8")
    )

    report, summary = read_report(fake_model_setup.reports_dir, "dataset_native", run="fix_report")
    assert [name for name in REQUIRED_REPORT_COLUMNS if name not in report[0]] == []
    assert list(report[0]) == list(extract.REPORT_COLUMNS)
    # 목록 순서(피험자, 라벨, 부분)대로 영상마다 한 행
    names = ("0.mp4", "5.MOV", "10_1.mp4", "10_2.mp4")
    assert [row["video"] for row in report] == [f"Fold1_part1/01/{name}" for name in names]
    assert [row["status"] for row in report] == ["done", "done", "done", "failed"]
    assert [(row["video_id"], row["session_id"], row["subject"], row["label"], row["part"]) for row in report] == [
        ("S01_0", "S01_0", "1", "0", "0"),
        ("S01_5", "S01_5", "1", "5", "0"),
        ("S01_10", "S01_10_1", "1", "10", "1"),
        ("S01_10", "S01_10_2", "1", "10", "2"),
    ]

    lying = report[0]  # 시계 방향 90도로 누운 20프레임, 10 fps
    native = fake_model_setup.raw_dir / "native" / "Fold1_part1" / "01"
    assert lying["out_parquet"] == str(native / "0.parquet") and lying["reason"] == ""
    assert (lying["frames_decoded"], lying["frames_skipped"], lying["frames_dropped"]) == ("20", "0", "0")
    assert (lying["frames_kept"], lying["frames_face"], lying["face_rate"]) == ("20", "20", "1")
    # 프레임 시각 1, 101, ..., 1901: 길이 1900 ms, 19·1000/1900 = 10 fps, 간격 100 ms로 일정
    assert (lying["duration_ms"], lying["mean_fps"]) == ("1900", "10")
    assert (lying["min_interval_ms"], lying["max_interval_ms"], lying["variable_intervals"]) == ("100", "100", "false")
    assert (lying["t_ms_ok"], lying["first_t_ms"], lying["last_t_ms"]) == ("true", "1", "1901")
    assert (lying["width"], lying["height"], lying["rotation"]) == (str(W), str(H), "90")
    assert (lying["rotation_source"], lying["rotation_reason"]) == ("auto_check", "ambiguous_smallest_roll")
    assert (lying["metadata_rotation"], lying["rotation_ambiguous"], lying["check_frames"]) == ("0", "true", "4")
    assert [lying[f"rate_{rotation}"] for rotation in (0, 90, 180, 270)] == ["1", "1", "1", "1"]
    lying_rolls = [float(lying[f"roll_{rotation}"]) for rotation in (0, 90, 180, 270)]
    assert lying_rolls == pytest.approx([-90.6, -0.6, 89.4, -1.9])
    assert (lying["display_flip"], lying["warnings"]) == ("false", "rotation_ambiguous")
    assert lying["model_sha256"] == file_sha256(fake_model_setup.model_path)
    assert lying["mediapipe_version"] == "0.0-fake"
    # 밝기: 세운 프레임들의 luma 평균을 다른 계산 길로 구해 비교한다.
    lumas = [luma_mean(rgb) for _, rgb in iter_video_frames(folder / "0.mp4", rotation=90)]
    assert float(lying["brightness_mean"]) == pytest.approx(float(np.mean(lumas)), rel=1e-5)
    assert float(lying["brightness_median"]) == pytest.approx(float(np.median(lumas)), rel=1e-5)
    assert float(lying["elapsed_s"]) > 0.0
    assert float(lying["ms_per_frame"]) == pytest.approx(1000.0 * float(lying["elapsed_s"]) / 20, rel=1e-4)

    varied = report[2]  # 프레임 시각 1, 101, 201, 401, 501: 길이 500 ms, 4·1000/500 = 8 fps, 간격 100~200 ms
    assert (varied["duration_ms"], varied["mean_fps"]) == ("500", "8")
    assert (varied["min_interval_ms"], varied["max_interval_ms"]) == ("100", "200")
    assert (varied["variable_intervals"], varied["frames_kept"], varied["check_frames"]) == ("true", "5", "2")
    assert (varied["rotation"], varied["rotation_reason"]) == ("0", "ambiguous_kept_metadata")

    failed = report[3]
    assert failed["reason"].startswith("InvalidDataError: ") and failed["out_parquet"] == str(native / "10_2.parquet")
    blank = ("frames_kept", "face_rate", "duration_ms", "rotation", "brightness_mean", "t_ms_ok")
    assert all(failed[name] == "" for name in blank)
    assert failed["model_sha256"] == file_sha256(fake_model_setup.model_path)  # 이번 실행이 쓰려던 모델
    assert not (native / "10_2.parquet").exists()

    lines = summary.splitlines()
    assert lines[0] == (
        f"# Data source: labelled video dataset at {root} (paths.rldd_root, RLDD folder layout): "
        "4 files, 3 logical videos, 1 subjects; native frame rate extraction"
    )
    assert f"- 생성 명령: `python -m drowsy.extract --config {config} --dataset --run-name fix_report`" in lines
    for expected in (
        "| 이 리포트의 입력 영상(파일) 수 | 4 |",
        "| 이번 실행에서 추출 (`done`) | 3 |",
        "| 이미 있던 결과를 그대로 씀 (`up_to_date`) | 0 |",
        "| 실패 (`failed`) | 1 |",
        "| 데이터셋 목록 전체의 파일 수 | 4 (UTA-RLDD 전체 사본의 기준값 182) |",
        "| 데이터셋 목록 전체의 논리 영상 수 | 3 (기준값 180) |",
        "| 기준값과 비교 | 다르다 |",
        "| 이 리포트의 논리 영상 수 | 3 |",
        "| 모든 부분이 추출된 논리 영상 | 2 |",
        "| 실패한 부분이 있는 논리 영상 | 1 (S01_10) |",
        "| 얼굴 검출 비율(프레임 기준, 전체) | 1.000 |",
        "| 처리한 프레임(= parquet 행) 합계 | 45 |",
        "| 프레임 간격이 변하는 영상 (최대 − 최소 > 1 ms) | 1 / 3 |",
        f"| 해상도(세운 뒤)별 영상 수 | {W}x{H} 3개 |",
    ):
        assert expected in lines, expected
    assert "통과 3개, 위반 0개." in summary
    assert "| `Fold1_part1/01/10_2.mp4` | failed | InvalidDataError: " in summary
    assert "자동 점검 3개, 수동 보정 0개, 회전 메타데이터 그대로 0개. 적용한 회전별 영상 수: 0도 2개, 90도 1개." in summary
    assert "### 가르지 못했고 회전 메타데이터가 후보에 없음" in summary
    assert "- 이번 실행에서 잰 것: 영상 3개, 프레임 45개, 프레임당 " in summary
    assert "이번 실행은 영상을 최대 2개씩 동시에 처리했다(CPU를 나눠 쓴 값이다)" in summary
    assert (
        "| `Fold1_part1/01/0.mp4` | 90 | 0 | ambiguous_smallest_roll | 1.00 / 1.00 / 1.00 / 1.00 "
        "| -90.6 / -0.6 / 89.4 / -1.9 |"
    ) in lines

    # 다시 실행: 끝난 영상은 건너뛰고 실패한 것만 다시 시도한다. 리포트는 전체를 다시 쓰고 처리 시간은 전 리포트에서 잇는다.
    assert extract.main(argv) == 1
    assert "extracted 0, up to date 3, failed 1" in capsys.readouterr().out
    again, summary = read_report(fake_model_setup.reports_dir, "dataset_native", run="fix_report")
    assert [row["status"] for row in again] == ["up_to_date", "up_to_date", "up_to_date", "failed"]
    unchanged = [name for name in extract.REPORT_COLUMNS if name != "status"]
    assert [[row[name] for name in unchanged] for row in again] == [[row[name] for name in unchanged] for row in report]
    assert "- 이번 실행에서 잰 것: 없음." in summary and "이번 실행은" not in summary
    assert (
        "- 전 리포트에서 이어 온 것(이 리포트의 전 판 `dataset_native_videos.csv`): 영상 3개, 프레임 45개, 프레임당 "
    ) in summary
    assert "시험 추출 리포트 `pilot_native_videos.csv`): 영상" not in summary  # 시험 추출을 한 적이 없다

    # 깨진 파일을 고치면 그 영상만 추출되고 실패가 없어진다.
    write_marker_video(folder / "10_2.mp4", PTS_10FPS[:6])
    assert extract.main(argv) == 0
    assert "extracted 1, up to date 3, failed 0" in capsys.readouterr().out
    fixed, summary = read_report(fake_model_setup.reports_dir, "dataset_native", run="fix_report")
    assert [row["status"] for row in fixed] == ["up_to_date", "up_to_date", "up_to_date", "done"]
    assert (fixed[3]["reason"], fixed[3]["frames_kept"], fixed[3]["t_ms_ok"]) == ("", "6", "true")
    assert "| 실패 (`failed`) | 0 |" in summary and "| 모든 부분이 추출된 논리 영상 | 3 |" in summary
    assert "## 3. 실패한 영상과 사유\n\n없음.\n" in summary and "통과 4개, 위반 0개." in summary

    # 저장된 t_ms가 엄격히 증가하지 않는 결과 파일은 검사에 걸린다(메타데이터는 최신이라 다시 추출되지는 않는다).
    rewrite_parquet(native / "5.parquet", t_ms=[1, 101, 101] + [201 + 100 * k for k in range(17)])
    assert extract.main(argv) == 0
    capsys.readouterr()
    checked, summary = read_report(fake_model_setup.reports_dir, "dataset_native", run="fix_report")
    assert [row["t_ms_ok"] for row in checked] == ["true", "false", "true", "true"]
    assert checked[1]["warnings"] == "rotation_ambiguous;timestamps_not_increasing"
    assert "통과 3개, 위반 1개." in summary and "위반: `Fold1_part1/01/5.MOV`" in summary
    rewrite_parquet(native / "5.parquet", t_ms=[0] + [100 * k for k in range(1, 20)])  # 0에서 시작하는 것도 위반이다
    assert extract.main(argv) == 0
    capsys.readouterr()
    assert read_report(fake_model_setup.reports_dir, "dataset_native", run="fix_report")[0][1]["t_ms_ok"] == "false"


def test_cli_dataset_manual_rotation_override_wins(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "rldd"
    for name in ("0.mp4", "5.mp4", "10.mp4"):
        write_marker_video(root / "07" / name, PTS_10FPS, quarter_turns_cw=1)
    # 자동 점검은 90을 고를 영상에 일부러 다른 값(270)을 적는다.
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root, {"07/5.mp4": 270})
    assert extract.main(["--config", str(config), "--dataset", "--pilot", "3"]) == 0
    out = capsys.readouterr().out
    assert "done 07/5.mp4" in out and "rotation: manual override" in out
    native = fake_model_setup.raw_dir / "native"

    manual = read_extract_metadata(native / "07" / "5.parquet")
    assert manual["rotation_override"] == 270 and "rotation_check" not in manual
    # 표의 값대로 돌렸다: 시계 방향 90도로 누운 그림을 반시계 270도 돌리면 거꾸로 선다(가짜 검출기의 roll −1.9).
    assert all(frame_roll(row) == pytest.approx(-1.9) for row in read_raw_frames(native / "07" / "5.parquet"))
    automatic = read_extract_metadata(native / "07" / "0.parquet")
    assert automatic["rotation_override"] == 90 and "rotation_check" in automatic

    # 다시 실행하면 모두 최신이다. 리포트에서 수동 보정 영상은 가르지 못함 표시가 없다.
    assert extract.main(["--config", str(config), "--dataset"]) == 0
    assert "extracted 0, up to date 3, failed 0" in capsys.readouterr().out
    report, summary = read_report(fake_model_setup.reports_dir, "dataset_native")
    by_video = {row["video"]: row for row in report}
    manual_row, auto_row = by_video["07/5.mp4"], by_video["07/0.mp4"]
    assert (manual_row["rotation_source"], manual_row["rotation_ambiguous"]) == ("manual_override", "false")
    assert (manual_row["rotation"], manual_row["warnings"], manual_row["rate_0"]) == ("270", "", "")
    assert (auto_row["rotation_source"], auto_row["rotation_ambiguous"]) == ("auto_check", "true")
    assert "자동 점검 2개, 수동 보정 1개, 회전 메타데이터 그대로 0개." in summary

    # 보정표의 값을 바꾸면 그 영상만 다시 추출 대상이 된다.
    changed = write_dataset_config(tmp_path, config_path, fake_model_setup, root, {"07/5.mp4": 90}, name="changed.yaml")
    assert extract.main(["--config", str(changed), "--dataset"]) == 0
    assert "extracted 1, up to date 2, failed 0" in capsys.readouterr().out
    assert read_extract_metadata(native / "07" / "5.parquet")["rotation_override"] == 90
    assert all(frame_roll(row) == pytest.approx(-0.6) for row in read_raw_frames(native / "07" / "5.parquet"))

    # 자동 점검이 가르지 못한 영상을 사람이 보고 같은 각도로 확인해 적으면: 다시 추출하지 않고 표시만 없어진다.
    confirmed = write_dataset_config(
        tmp_path, config_path, fake_model_setup, root, {"07/5.mp4": 90, "07/0.mp4": 90}, name="confirmed.yaml"
    )
    assert extract.main(["--config", str(confirmed), "--dataset"]) == 0
    assert "extracted 0, up to date 3, failed 0" in capsys.readouterr().out
    report, _ = read_report(fake_model_setup.reports_dir, "dataset_native")
    first = {row["video"]: row for row in report}["07/0.mp4"]
    assert (first["rotation_source"], first["rotation_ambiguous"]) == ("manual_override", "false")
    assert first["warnings"] == ""
    assert float(first["roll_90"]) == pytest.approx(-0.6)  # 점검 때 잰 수치는 남는다


def test_cli_writes_the_report_when_the_run_is_interrupted(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "rldd"
    for name in ("0.mp4", "5.mp4", "10.mp4"):
        write_marker_video(root / "04" / name, PTS_10FPS[:6])
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root)
    run_job = extract._run_job

    def interrupted(job: Any) -> Any:
        if job.label == "04/5.mp4":
            raise KeyboardInterrupt  # 둘째 영상을 처리하던 중에 사용자가 실행을 끊었다
        return run_job(job)

    monkeypatch.setattr(extract, "_run_job", interrupted)
    with pytest.raises(KeyboardInterrupt):
        extract.main(["--config", str(config), "--dataset"])
    capsys.readouterr()
    report, summary = read_report(fake_model_setup.reports_dir, "dataset_native")
    assert [(row["video"], row["status"]) for row in report] == [
        ("04/0.mp4", "done"),  # 결과 파일은 써졌다. 보고를 받기 전에 끊겨 처리 시간은 모른다
        ("04/5.mp4", "pending"),
        ("04/10.mp4", "pending"),
    ]
    assert (report[0]["frames_kept"], report[0]["t_ms_ok"], report[0]["ms_per_frame"]) == ("6", "true", "")
    assert (report[1]["frames_kept"], report[1]["reason"]) == ("", "")
    assert "| 처리되지 않음 (`pending`, 실행이 끊김) | 2 |" in summary
    assert "| 처리되지 않은 부분이 있는 논리 영상 | 2 |" in summary

    monkeypatch.setattr(extract, "_run_job", run_job)
    assert extract.main(["--config", str(config), "--dataset"]) == 0  # 다시 실행하면 이어서 한다
    assert "extracted 2, up to date 1, failed 0" in capsys.readouterr().out
    report, summary = read_report(fake_model_setup.reports_dir, "dataset_native")
    assert [row["status"] for row in report] == ["up_to_date", "done", "done"]
    assert "| 처리되지 않음 (`pending`, 실행이 끊김) | 0 |" in summary

    # 끊긴 실행이 다시 추출한 영상에는 전 리포트의 처리 시간을 잇지 않는다: 그 값은 지금의 결과 파일을 만든 시간이 아니고,
    # "이번 실행이 건너뛴 영상"이라는 요약의 말도 맞지 않게 된다.
    native = fake_model_setup.raw_dir / "native" / "04"
    (native / "0.parquet").unlink()
    assert extract.main(["--config", str(config), "--dataset"]) == 0  # 04/0을 다시 추출한다: 이제 셋 다 처리 시간이 있다
    capsys.readouterr()
    assert all(row["ms_per_frame"] != "" for row in read_report(fake_model_setup.reports_dir, "dataset_native")[0])
    (native / "0.parquet").unlink()
    (native / "5.parquet").unlink()
    monkeypatch.setattr(extract, "_run_job", interrupted)
    with pytest.raises(KeyboardInterrupt):
        extract.main(["--config", str(config), "--dataset"])  # 04/0을 추출한 뒤 04/5에서 끊긴다
    capsys.readouterr()
    report, summary = read_report(fake_model_setup.reports_dir, "dataset_native")
    assert [(row["status"], row["frames_kept"], row["ms_per_frame"] != "") for row in report] == [
        ("done", "6", False),  # 이번 실행이 추출했고 처리 시간은 받지 못했다. 전 리포트에 프레임 수가 같은 값이 있어도 잇지 않는다
        ("pending", "", False),
        ("up_to_date", "6", True),  # 건너뛴 영상은 전 판의 값을 잇는다
    ]
    section = timing_section(summary)
    assert section[1] == "- 이번 실행에서 잰 것: 없음."
    assert section[2].startswith(
        "- 전 리포트에서 이어 온 것(이 리포트의 전 판 `dataset_native_videos.csv`): 영상 1개, 프레임 6개, 프레임당 "
    )
    assert section[3].startswith("- 처리 시간을 모르는 것: 영상 1개.") and len(section) == 4


def test_cli_rejects_override_keys_that_match_no_video(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "rldd"
    for name in ("0.mp4", "5.MOV", "10.mp4"):
        write_marker_video(root / "Fold1_part1" / "01" / name, PTS_10FPS[:4])

    # 확장자의 대소문자와 폴더 깊이가 틀린 키: 시작하지 않고 가장 비슷한 영상을 알려 준다.
    wrong = {"Fold1_part1/01/0.MP4": 90, "01/5.MOV": 180, "Fold1_part1/01/10.mp4": 90, "Fold9/77/zzz.avi": 0}
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root, wrong)
    assert extract.main(["--config", str(config), "--dataset", "--pilot", "3"]) == 2
    captured = capsys.readouterr()
    assert "extraction.rotation_overrides has 3 key(s) that match no video of this run" in captured.err
    assert "'Fold1_part1/01/0.MP4' (closest: 'Fold1_part1/01/0.mp4')" in captured.err
    assert "'01/5.MOV' (closest: 'Fold1_part1/01/5.MOV')" in captured.err
    assert "'Fold9/77/zzz.avi' (no similar video)" in captured.err
    assert "'Fold1_part1/01/10.mp4' (" not in captured.err  # 맞는 키는 적지 않는다
    assert not fake_model_setup.raw_dir.exists() and marker_mediapipe.created == []  # 아무것도 추출하지 않았다

    # 시험 추출 범위 밖의 영상에 대한 맞는 키는 받아들인다(목록 전체와 비교한다).
    valid = {"Fold1_part1/01/10.mp4": 180}
    later = write_dataset_config(tmp_path, config_path, fake_model_setup, root, valid, "later.yaml")
    assert extract.main(["--config", str(later), "--dataset", "--pilot", "1"]) == 0
    capsys.readouterr()

    # --video에서는 키가 파일 이름이다. 데이터셋 경로를 적은 표를 그대로 쓰면 시작하지 않는다.
    video = root / "Fold1_part1" / "01" / "0.mp4"
    assert extract.main(["--config", str(later), "--video", str(video)]) == 2
    assert "'Fold1_part1/01/10.mp4'" in capsys.readouterr().err
    # --input-dir에서는 그 폴더 기준 상대 경로다.
    assert extract.main(["--config", str(later), "--input-dir", str(root / "Fold1_part1")]) == 2
    assert "(closest: '01/10.mp4')" in capsys.readouterr().err


def test_check_override_keys_suggestions() -> None:
    labels = ["Fold1_part1/01/0.mp4", "Fold1_part1/01/5.MOV", "Fold1_part1/02/0.mp4", "Fold2_part1/13/10_1.mp4"]
    extract.check_override_keys({}, labels, "table")
    extract.check_override_keys({"Fold1_part1/01/5.MOV": 90, "Fold2_part1/13/10_1.mp4": 0}, labels, "table")
    with pytest.raises(ExtractError) as excinfo:
        wrong = {"fold1_part1/01/5.mov": 90, "Fold1_part1/Fold1_part1/02/0.mp4": 90, "Fold2_part1/13/10_2.mp4": 0}
        extract.check_override_keys(wrong, labels, "table")
    message = str(excinfo.value)
    assert message.startswith("table has 3 key(s) that match no video of this run: ")
    assert "'fold1_part1/01/5.mov' (closest: 'Fold1_part1/01/5.MOV'" in message  # 대소문자만 다르다
    assert "'Fold1_part1/Fold1_part1/02/0.mp4' (closest: 'Fold1_part1/02/0.mp4'" in message  # 폴더가 한 겹 더 있다
    assert "'Fold2_part1/13/10_2.mp4' (closest: 'Fold2_part1/13/10_1.mp4'" in message  # 철자가 비슷하다
    assert "Fix:" in message


def test_cli_display_flip_needs_a_manual_rotation(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "rldd"
    for name in ("0.mp4", "5.mp4", "10.mp4"):
        write_marker_video(root / "03" / name, PTS_10FPS)
    tag_display_matrix(root / "03" / "0.mp4", -1, 0, 0, 1)  # 좌우 뒤집기: libav는 회전 −180으로 읽는다
    with av.open(str(root / "03" / "0.mp4")) as container:
        assert next(container.decode(video=0)).rotation == -180

    # --video: 보정표에 없으면 실패하고 결과를 쓰지 않는다(전에는 거꾸로 돌려 추출했다).
    assert extract.main(["--config", str(fake_model_setup.config), "--video", str(root / "03" / "0.mp4")]) == 1
    captured = capsys.readouterr()
    assert "FAILED 0.mp4: VideoError" in captured.err and "display matrix contains a flip" in captured.err
    assert not (fake_model_setup.raw_dir / "native" / "0.parquet").exists()

    # --dataset: 실패 행에 사유가 남고, 파일에서 읽을 수 있는 길이와 프레임률은 채워진다.
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root)
    assert extract.main(["--config", str(config), "--dataset"]) == 1
    capsys.readouterr()
    report, summary = read_report(fake_model_setup.reports_dir, "dataset_native")
    flipped = report[0]
    assert (flipped["video"], flipped["status"]) == ("03/0.mp4", "failed")
    assert flipped["reason"].startswith("VideoError: ") and "display matrix contains a flip" in flipped["reason"]
    assert (flipped["display_flip"], flipped["warnings"]) == ("true", "display_flip")
    assert (flipped["duration_ms"], flipped["mean_fps"], flipped["variable_intervals"]) == ("1900", "10", "false")
    assert (flipped["width"], flipped["rotation"], flipped["frames_kept"]) == ("", "", "")
    assert "표시 행렬에 뒤집기(거울상)가 든 영상: 1개." in summary and "`03/0.mp4` (failed)" in summary
    assert [row["status"] for row in report[1:]] == ["done", "done"]

    # 보정표에 각도를 적으면 그 각도로만 돌리고(거울상은 그대로) 표시를 남긴다.
    manual = write_dataset_config(tmp_path, config_path, fake_model_setup, root, {"03/0.mp4": 0}, name="manual.yaml")
    assert extract.main(["--config", str(manual), "--dataset"]) == 0
    capsys.readouterr()
    metadata = read_extract_metadata(fake_model_setup.raw_dir / "native" / "03" / "0.parquet")
    assert (metadata["display_flip"], metadata["rotation"], metadata["rotation_override"]) == (True, 0, 0)
    rows = list(read_raw_frames(fake_model_setup.raw_dir / "native" / "03" / "0.parquet"))
    assert all(frame_roll(row) == pytest.approx(-0.6) for row in rows)  # 저장된 그대로(똑바로) 넘겼다
    report, _ = read_report(fake_model_setup.reports_dir, "dataset_native")
    assert (report[0]["status"], report[0]["display_flip"]) == ("done", "true")
    assert report[0]["rotation_source"] == "manual_override"
    assert report[0]["warnings"] == "display_flip"


def test_cli_input_dir_checks_rotation_and_writes_a_report(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--input-dir`도 `--dataset`, `--unlabeled`와 같은 회전 점검을 한다: 메타데이터 없이 누워 저장된 영상이 세워져 추출된다."""
    videos = tmp_path / "videos"
    write_marker_video(videos / "a" / "one.mp4", PTS_10FPS[:6], quarter_turns_cw=1)
    write_marker_video(videos / "two.MOV", PTS_10FPS[:4])
    (videos / "b").mkdir()
    (videos / "b" / "broken.mp4").write_bytes(b"")
    marker_mediapipe.upside_down_found = False  # 거꾸로 선 얼굴을 못 찾는 얼굴: 점검이 끝까지 가른다
    argv = ["--config", str(fake_model_setup.config), "--input-dir", str(videos)]
    assert extract.main(argv) == 1
    out = capsys.readouterr().out
    # 읽을 수 있는 영상마다 점검용 랜드마커 4개(후보 각도마다 하나)와 추출용 1개. 깨진 파일은 디코딩에서 실패한다.
    assert len(marker_mediapipe.created) == 10 and all(landmarker.closed for landmarker in marker_mediapipe.created)
    assert "done a/one.mp4" in out and "rotation: auto check chose 90 (metadata 0; smallest_roll;" in out
    assert out.count("WARNING rotation differs from the video metadata") == 1 and "ambiguous" not in out

    native = fake_model_setup.raw_dir / "native"
    rows = list(read_raw_frames(native / "a" / "one.parquet"))
    assert [(row.w, row.h, row.face) for row in rows] == [(W, H, True)] * 6  # 세워서 추출했다
    assert all(frame_roll(row) == pytest.approx(-0.6) for row in rows)  # 똑바로 선 그림에서 가짜 검출기가 내는 roll
    metadata = read_extract_metadata(native / "a" / "one.parquet")
    assert (metadata["rotation"], metadata["rotation_override"]) == (90, 90)
    # 결과 파일의 점검 기록은 --dataset이 적는 것과 같은 이름과 꼴이다(다른 명령이 이 기록을 읽는다).
    assert sorted(metadata["rotation_check"]) == [
        "ambiguous",
        "metadata_rotation",
        "n_frames",
        "rates",
        "reason",
        "rolls",
        "settings",
        "version",
    ]
    recorded = metadata["rotation_check"]
    assert (recorded["ambiguous"], recorded["reason"], recorded["metadata_rotation"]) == (False, "smallest_roll", 0)
    # 기본 설정(12장, 2000 ms 간격)에서 0.5초짜리 영상의 표본은 첫 프레임 한 장이다.
    assert recorded["n_frames"] == 1
    assert recorded["settings"] == {"frames": 12, "interval_ms": 2000, "roll_margin_deg": 15.0}
    # 후보 0은 시계 방향 90도로 누운 채(roll −90.6), 90은 똑바로(−0.6), 180은 반시계로 누운 채(89.4), 270은 거꾸로(검출 없음)
    assert recorded["rates"] == {"0": 1.0, "90": 1.0, "180": 1.0, "270": 0.0}
    expected_rolls = {"0": pytest.approx(-90.6), "90": pytest.approx(-0.6), "180": pytest.approx(89.4), "270": None}
    assert recorded["rolls"] == expected_rolls

    report, summary = read_report(fake_model_setup.reports_dir, "input_dir_native")
    assert [(row["video"], row["status"]) for row in report] == [
        ("a/one.mp4", "done"),
        ("b/broken.mp4", "failed"),
        ("two.MOV", "done"),
    ]
    one, two = report[0], report[2]
    assert (one["rotation"], one["rotation_source"], one["rotation_ambiguous"]) == ("90", "auto_check", "false")
    assert (one["width"], one["height"], one["warnings"]) == (str(W), str(H), "rotation_differs_from_metadata")
    assert (one["metadata_rotation"], one["check_frames"], one["rate_270"], one["roll_270"]) == ("0", "1", "0", "")
    assert float(one["roll_90"]) == pytest.approx(-0.6)
    assert (two["rotation"], two["rotation_source"], two["warnings"]) == ("0", "auto_check", "")
    assert [one[name] for name in ("video_id", "subject")] == ["", ""]
    assert summary.splitlines()[0] == (
        f"# Data source: videos below {videos} (--input-dir): 3 files; native frame rate extraction"
    )
    assert "- 회전: 영상별 자동 점검(표본 12장, 간격 2000 ms, roll 여유 15도)." in summary
    assert "자동 점검 2개, 수동 보정 0개, 회전 메타데이터 그대로 0개. 적용한 회전별 영상 수: 0도 1개, 90도 1개." in summary
    assert "데이터셋 목록 전체" not in summary  # 라벨 있는 데이터셋이 아니면 180개 기준 합계를 적지 않는다
    assert "이번 실행은 영상을 최대 2개씩 동시에 처리했다" in summary

    # 다시 실행하면 건너뛴다. 점검도 다시 하지 않는다.
    assert extract.main(argv) == 1
    assert "extracted 0, up to date 2, failed 1" in capsys.readouterr().out
    assert len(marker_mediapipe.created) == 10

    # 수동 회전 보정표의 값이 자동 점검을 이긴다: 그 영상은 점검 없이 표의 각도로 다시 추출된다.
    manual = write_cli_config(
        tmp_path,
        config_path,
        fake_model_setup.spec_path,
        fake_model_setup.model_path,
        {"a/one.mp4": 180},
        name="manual.yaml",
    )
    assert extract.main(["--config", str(manual), "--input-dir", str(videos)]) == 1
    out = capsys.readouterr().out
    assert "done a/one.mp4" in out and "rotation: manual override" in out and "skip two.MOV" in out
    assert len(marker_mediapipe.created) == 11
    metadata = read_extract_metadata(native / "a" / "one.parquet")
    assert metadata["rotation_override"] == 180 and "rotation_check" not in metadata
    # 시계 방향 90도로 누운 그림을 180도 돌리면 반시계 90도로 누운 채다(가짜 검출기의 roll 89.4).
    assert all(frame_roll(row) == pytest.approx(89.4) for row in read_raw_frames(native / "a" / "one.parquet"))
    report, _ = read_report(fake_model_setup.reports_dir, "input_dir_native")
    assert (report[0]["rotation"], report[0]["rotation_source"]) == ("180", "manual_override")

    # --video는 리포트를 쓰지 않는다.
    report_dir = fake_model_setup.reports_dir / "extraction" / "extraction"
    before = sorted(path.name for path in report_dir.iterdir())
    assert extract.main(["--config", str(fake_model_setup.config), "--video", str(videos / "two.MOV")]) == 0
    assert "report:" not in capsys.readouterr().out
    assert sorted(path.name for path in report_dir.iterdir()) == before


def test_results_made_without_the_rotation_check_are_extracted_again(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    pipeline_spec: PipelineSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """회전 메타데이터만 써서 만든 전의 `--input-dir` 결과(누운 채 추출된 것)는 최신으로 보지 않는다."""
    videos = tmp_path / "videos"
    write_marker_video(videos / "lying.mp4", PTS_10FPS[:6], quarter_turns_cw=1)
    out = fake_model_setup.raw_dir / "native" / "lying.parquet"
    # 전의 명령이 하던 대로: 점검 없이 회전 메타데이터(0)로 추출한다. 설정과 모델은 명령과 같다.
    spec = spec_for_model(pipeline_spec, fake_model_setup.model_path)
    extract_video(videos / "lying.mp4", fake_model_setup.model_path, out, spec=spec)
    old = read_extract_metadata(out)
    assert (old["rotation"], old["rotation_override"]) == (0, None) and "rotation_check" not in old
    assert [(row.w, row.h) for row in read_raw_frames(out)] == [(H, W)] * 6  # 누운 채다
    assert output_is_current(out, spec, None, None) is True  # 회전을 빼면 최신인 결과다

    assert extract.main(["--config", str(fake_model_setup.config), "--input-dir", str(videos)]) == 0
    assert "extracted 1, up to date 0, failed 0" in capsys.readouterr().out
    assert [(row.w, row.h) for row in read_raw_frames(out)] == [(W, H)] * 6
    assert read_extract_metadata(out)["rotation_override"] == 90


def test_cli_video_rotation_option(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "videos" / "lying.mp4"
    write_marker_video(video, PTS_10FPS[:6], quarter_turns_cw=1)  # 메타데이터 없이 시계 방향 90도로 누운 영상
    argv = ["--config", str(fake_model_setup.config), "--video", str(video)]
    out_parquet = fake_model_setup.raw_dir / "native" / "lying.parquet"

    def rolls() -> list[float]:
        return [frame_roll(row) for row in read_raw_frames(out_parquet)]

    # --rotation 없이: 자동 점검. 이 가짜 검출기는 실제 랜드마커처럼 거꾸로 선 얼굴도 찾으므로 90과 270을 가르지 못하고
    # |roll|이 더 작은 90을 고른 뒤 표시한다. --video에서는 고치는 방법으로 --rotation을 알려 준다.
    assert extract.main(argv) == 0
    out = capsys.readouterr().out
    assert "rotation: auto check chose 90 (metadata 0; ambiguous_smallest_roll;" in out
    assert "WARNING rotation ambiguous: look at the video and pass --rotation" in out
    assert len(marker_mediapipe.created) == 5
    assert rolls() == [pytest.approx(-0.6)] * 6
    assert read_extract_metadata(out_parquet)["rotation_check"]["ambiguous"] is True

    # 사람이 보고 같은 각도를 --rotation으로 준다: 점검 없이 다시 추출해, 결과 파일에 점검 기록이 남지 않게 한다
    # (그 기록이 없고 rotation_override가 있는 파일이 수동 값으로 만든 결과다).
    assert extract.main([*argv, "--rotation", "90"]) == 0
    out = capsys.readouterr().out
    assert "done lying.mp4" in out and "rotation: manual override (--rotation)" in out and "WARNING" not in out
    assert len(marker_mediapipe.created) == 6  # 추출용 하나만 더 만들었다
    metadata = read_extract_metadata(out_parquet)
    assert (metadata["rotation"], metadata["rotation_override"]) == (90, 90) and "rotation_check" not in metadata
    assert rolls() == [pytest.approx(-0.6)] * 6
    assert extract.main([*argv, "--rotation", "90"]) == 0  # 같은 명령을 다시 주면 건너뛴다
    assert "skip lying.mp4" in capsys.readouterr().out and len(marker_mediapipe.created) == 6

    # 다른 각도를 주면 그 각도로 돌린다: 270이면 거꾸로 선다(가짜 검출기의 roll −1.9).
    assert extract.main([*argv, "--rotation", "270"]) == 0
    capsys.readouterr()
    assert read_extract_metadata(out_parquet)["rotation_override"] == 270 and rolls() == [pytest.approx(-1.9)] * 6

    # --rotation은 그 실행에만 적용된다: 빼고 실행하면 자동 점검으로 다시 추출한다.
    assert extract.main(argv) == 0
    assert "auto check chose 90" in capsys.readouterr().out
    assert "rotation_check" in read_extract_metadata(out_parquet) and rolls() == [pytest.approx(-0.6)] * 6

    # 보정표에 그 영상이 있어도 명령에 준 --rotation이 이긴다. 표만 있으면 표의 값을 쓴다.
    table = write_cli_config(
        tmp_path, config_path, fake_model_setup.spec_path, fake_model_setup.model_path, {"lying.mp4": 270}, "table.yaml"
    )
    assert extract.main(["--config", str(table), "--video", str(video), "--rotation", "180"]) == 0
    capsys.readouterr()
    assert read_extract_metadata(out_parquet)["rotation_override"] == 180
    assert extract.main(["--config", str(table), "--video", str(video)]) == 0
    out = capsys.readouterr().out
    assert "rotation: manual override" in out and "(--rotation)" not in out
    assert read_extract_metadata(out_parquet)["rotation_override"] == 270

    # 표시 행렬에 뒤집기가 든 영상: --rotation 없이는 실패하고 오류 문장이 --rotation을 알려 준다. 주면 추출된다.
    flipped = tmp_path / "videos" / "flipped.mp4"
    write_marker_video(flipped, PTS_10FPS[:4])
    tag_display_matrix(flipped, -1, 0, 0, 1)
    flipped_argv = ["--config", str(fake_model_setup.config), "--video", str(flipped)]
    assert extract.main(flipped_argv) == 1
    error = capsys.readouterr().err
    assert "display matrix contains a flip" in error and "--rotation" in error
    assert extract.main([*flipped_argv, "--rotation", "0"]) == 0
    capsys.readouterr()
    flipped_metadata = read_extract_metadata(fake_model_setup.raw_dir / "native" / "flipped.parquet")
    assert (flipped_metadata["display_flip"], flipped_metadata["rotation_override"]) == (True, 0)

    # --rotation은 --video에서만 받는다. 값은 0, 90, 180, 270뿐이다.
    created = len(marker_mediapipe.created)
    for source in (["--input-dir", str(video.parent)], ["--dataset"], ["--unlabeled"]):
        assert extract.main(["--config", str(fake_model_setup.config), *source, "--rotation", "90"]) == 2
        assert "--rotation needs --video" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        extract.main([*argv, "--rotation", "45"])
    capsys.readouterr()
    assert len(marker_mediapipe.created) == created and not fake_model_setup.reports_dir.exists()


@pytest.mark.parametrize("run_name", ["a/b", "../escape", "..", ".", "", "a\\b"])
def test_cli_rejects_a_run_name_that_is_not_a_plain_folder_name(
    tmp_path: Path, run_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--run-name`은 다른 일을 하기 전에 확인한다(`drowsy.config.check_run_name`): 설정 파일이 없어도 이 오류가 먼저다."""
    from drowsy.config import check_run_name

    with pytest.raises(ValueError):
        check_run_name(run_name)
    argv = ["--config", str(tmp_path / "missing.yaml"), "--input-dir", str(tmp_path), "--run-name", run_name]
    assert extract.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err == f"error: --run-name must be a plain folder name, got {run_name!r}\n"
    assert captured.out == "" and list(tmp_path.iterdir()) == []
    # 평범한 이름이면 이 검사를 지나 다음 오류(설정 파일 없음)에 닿는다.
    assert extract.main(argv[:-1] + ["fix_ok"]) == 2
    assert "--run-name" not in capsys.readouterr().err


def test_cli_unlabeled_source(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from drowsy.config import load_config

    root = tmp_path / "team_videos"
    write_marker_video(root / "drive_a" / "clip1.mp4", PTS_10FPS[:8])
    write_marker_video(root / "drive_b" / "night" / "clip2.MOV", PTS_10FPS[:8], quarter_turns_cw=1)
    (root / "drive_b" / "notes.txt").write_text("not a video", encoding="utf-8")
    base = write_dataset_config(tmp_path, config_path, fake_model_setup, None)

    assert extract.main(["--config", str(base), "--unlabeled"]) == 2
    assert "paths.unlabeled_root is null" in capsys.readouterr().err

    # 데이터셋용 보정표(rotation_overrides)의 키는 라벨 없는 영상과 맞지 않아도 된다. 표가 따로다.
    config = edit_config(
        base,
        "unlabeled.yaml",
        paths__unlabeled_root=str(root),
        extraction__rotation_overrides={"Fold1_part1/01/0.mp4": 90},
        extraction__unlabeled_rotation_overrides={"drive_b/night/clip2.MOV": 90},
    )
    cfg = load_config(config)
    assert extract.unlabeled_raw_dir(cfg) == tmp_path / "raw" / "unlabeled_native"
    assert extract.unlabeled_raw_dir(cfg, fps_sim=True) == tmp_path / "raw" / "unlabeled_fps15"

    assert extract.main(["--config", str(config), "--unlabeled"]) == 0
    out = capsys.readouterr().out
    assert "done drive_a/clip1.mp4" in out and "auto check chose 0 (metadata 0; ambiguous_kept_metadata;" in out
    assert "done drive_b/night/clip2.MOV" in out and "rotation: manual override" in out
    written = sorted(path.relative_to(tmp_path / "raw").as_posix() for path in (tmp_path / "raw").rglob("*.parquet"))
    assert written == ["unlabeled_native/drive_a/clip1.parquet", "unlabeled_native/drive_b/night/clip2.parquet"]
    clip2 = extract.raw_parquet_path(extract.unlabeled_raw_dir(cfg), Path("drive_b/night/clip2.MOV"))
    assert read_extract_metadata(clip2)["rotation_override"] == 90
    assert [(row.w, row.h) for row in read_raw_frames(clip2)] == [(W, H)] * 8

    report, summary = read_report(fake_model_setup.reports_dir, "unlabeled_native")
    assert [(row["video"], row["status"], row["rotation_source"]) for row in report] == [
        ("drive_a/clip1.mp4", "done", "auto_check"),
        ("drive_b/night/clip2.MOV", "done", "manual_override"),
    ]
    assert [row["video_id"] for row in report] == ["", ""]
    assert summary.splitlines()[0] == (
        f"# Data source: unlabeled videos at {root} (paths.unlabeled_root, no labels): 2 files; "
        "native frame rate extraction"
    )
    assert "수동 회전 보정표 `extraction.unlabeled_rotation_overrides`" in summary

    assert extract.main(["--config", str(config), "--unlabeled"]) == 0  # 이어 하기
    assert "extracted 0, up to date 2, failed 0" in capsys.readouterr().out

    # 누락 모사 추출은 unlabeled_<모사 폴더 이름>에 쓴다.
    assert extract.main(["--config", str(config), "--unlabeled", "--fps-sim", "15"]) == 0
    capsys.readouterr()
    sim = extract.raw_parquet_path(extract.unlabeled_raw_dir(cfg, fps_sim=True), Path("drive_a/clip1.mp4"))
    assert read_extract_metadata(sim)["drop_sim"] == {"latency_ms_mean": 66.7, "latency_ms_std": 8.0, "seed": 0}
    assert "device frame-drop simulation 'fps15'" in read_report(fake_model_setup.reports_dir, "unlabeled_fps15")[1]

    # 라벨 없는 영상의 보정표에 맞지 않는 키가 있으면 시작하지 않는다.
    bad = edit_config(config, "bad.yaml", extraction__unlabeled_rotation_overrides={"drive_a/CLIP1.mp4": 90})
    assert extract.main(["--config", str(bad), "--unlabeled"]) == 2
    error = capsys.readouterr().err
    assert "extraction.unlabeled_rotation_overrides has 1 key(s)" in error and "closest: 'drive_a/clip1.mp4'" in error
    assert extract.main(["--config", str(config), "--unlabeled", "--pilot", "1"]) == 2
    assert "--pilot needs --dataset" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        extract.main(["--config", str(config), "--unlabeled", "--dataset"])  # 입력은 하나만 고른다
    capsys.readouterr()

    # 추출 결과(parquet) 폴더를 가리키면 추출할 것이 없다고 알린다.
    extracted_tree = str(tmp_path / "raw" / "unlabeled_native")
    parquet_root = edit_config(config, "parquet_root.yaml", paths__unlabeled_root=extracted_tree)
    assert extract.main(["--config", str(parquet_root), "--unlabeled"]) == 2
    assert "holds raw parquet files, not videos" in capsys.readouterr().err


def report_row(video: str, status: str, **values: Any) -> dict[str, Any]:
    """손으로 만든 리포트 행(구현의 열 목록에 값만 채운다)."""
    row: dict[str, Any] = dict.fromkeys(extract.REPORT_COLUMNS)
    row.update(video=video, status=status, reason="", warnings="")
    row.update(values)
    return row


def test_write_extraction_report_totals_for_a_full_dataset(tmp_path: Path) -> None:
    """파일 182개, 논리 영상 180개인 목록의 합계와, 실패·미처리·가르지 못한 영상의 목록을 손으로 만든 행으로 확인한다."""
    done = dict(
        frames_decoded=100,
        frames_skipped=0,
        frames_dropped=0,
        frames_kept=100,
        frames_face=90,
        face_rate=0.9,
        duration_ms=6000,
        mean_fps=16.5,
        min_interval_ms=40,
        max_interval_ms=90,
        variable_intervals=True,
        width=720,
        height=1280,
        rotation=0,
        rotation_source="auto_check",
        rotation_reason="smallest_roll",
        metadata_rotation=0,
        rotation_ambiguous=False,
        display_flip=False,
        brightness_mean=100.0,
        brightness_median=101.0,
        t_ms_ok=True,
        first_t_ms=1,
        last_t_ms=6001,
        elapsed_s=0.5,
        ms_per_frame=5.0,
        **{f"rate_{rotation}": 1.0 for rotation in (0, 90, 180, 270)},
        **{f"roll_{rotation}": roll for rotation, roll in ((0, 1.0), (90, 91.0), (180, 170.0), (270, -89.0))},
    )
    rows = []
    for subject in range(1, 61):
        for label in (0, 5, 10):
            parts = (1, 2) if label == 10 and subject in (32, 49) else (0,)
            for part in parts:
                suffix = f"_{part}" if part else ""
                ids = {
                    "video_id": f"S{subject:02d}_{label}",
                    "session_id": f"S{subject:02d}_{label}{suffix}",
                    "subject": subject,
                    "label": label,
                    "part": part,
                }
                rows.append(report_row(f"{subject:02d}/{label}{suffix}.mp4", "done", **done) | ids)
    assert len(rows) == 182
    by_session = {row["session_id"]: row for row in rows}
    # 실패한 행과 처리되지 않은 행은 식별 열만 남기고 나머지를 비운다.
    for session, status, reason in (("S32_10_2", "failed", "InvalidDataError: bad file"), ("S60_10", "pending", "")):
        row = by_session[session]
        ids = {name: row[name] for name in ("video_id", "session_id", "subject", "label", "part")}
        row.update(report_row(row["video"], status, reason=reason) | ids)
    by_session["S05_0"].update(
        rotation=180,
        rotation_reason="ambiguous_smallest_roll",
        rotation_ambiguous=True,
        metadata_rotation=90,
        warnings="rotation_ambiguous",
    )
    by_session["S06_5"].update(face_rate=0.5, frames_face=50, warnings="low_face_rate", brightness_mean=40.0)
    by_session["S07_0"].update(status="up_to_date", elapsed_s=None, ms_per_frame=None, width=1920, height=1080)

    context = extract.ReportContext(
        description="Data source: labelled video dataset at /data/rldd (hand-made rows)",
        command="python -m drowsy.extract --config config/local.yaml --dataset",
        config_path="config/local.yaml",
        out_root="data/raw/native",
        drop_sim=None,
        model_sha256="ab" * 32,
        mediapipe_version="0.10.32",
        rotation_check=extract.RotationCheckSettings(12, 2000, 15.0),
        overrides_key="extraction.rotation_overrides",
        min_face_ratio=0.7,
        workers=4,
        manifest_totals=(182, 180, 60),
    )
    report_dir = tmp_path / "reports" / "run" / "extraction"
    csv_path, md_path = extract.write_extraction_report(report_dir, "dataset_native", rows, context)
    assert (csv_path.name, md_path.name) == ("dataset_native_videos.csv", "dataset_native_summary.md")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 182 and list(table[0]) == list(extract.REPORT_COLUMNS)
    assert [row["status"] for row in table].count("failed") == 1
    failed = next(row for row in table if row["status"] == "failed")
    assert (failed["video"], failed["reason"]) == ("32/10_2.mp4", "InvalidDataError: bad file")
    assert failed["frames_kept"] == ""
    assert (table[0]["variable_intervals"], table[0]["mean_fps"], table[0]["roll_270"]) == ("true", "16.5", "-89")

    lines = md_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Data source: labelled video dataset at /data/rldd (hand-made rows)"
    for expected in (
        "| 이 리포트의 입력 영상(파일) 수 | 182 |",
        "| 이번 실행에서 추출 (`done`) | 179 |",
        "| 이미 있던 결과를 그대로 씀 (`up_to_date`) | 1 |",
        "| 실패 (`failed`) | 1 |",
        "| 처리되지 않음 (`pending`, 실행이 끊김) | 1 |",
        "| 데이터셋 목록 전체의 파일 수 | 182 (UTA-RLDD 전체 사본의 기준값 182) |",
        "| 데이터셋 목록 전체의 논리 영상 수 | 180 (기준값 180) |",
        "| 데이터셋 목록 전체의 피험자 수 | 60 |",
        "| 기준값과 비교 | 같다 |",
        "| 이 리포트의 논리 영상 수 | 180 |",
        "| 모든 부분이 추출된 논리 영상 | 178 |",  # S32_10은 한 부분이 실패, S60_10은 미처리
        "| 실패한 부분이 있는 논리 영상 | 1 (S32_10) |",
        "| 처리되지 않은 부분이 있는 논리 영상 | 1 |",
        "| 처리한 프레임(= parquet 행) 합계 | 18000 |",  # 추출된 180개 x 100
        "| 얼굴 검출 비율(프레임 기준, 전체) | 0.898 |",  # (179·90 + 50) / 18000 = 16160 / 18000
        "| 얼굴 검출 비율이 가장 낮은 영상 | `06/5.mp4` 0.500 |",
        "| 얼굴 검출 비율이 0.7 미만인 영상 (`low_face_rate`) | 1 (`06/5.mp4`) |",
        "| 원본 영상 길이 합계 | 0.300시간 (1080.0초) |",  # 180 x 6초
        "| 프레임 간격이 변하는 영상 (최대 − 최소 > 1 ms) | 180 / 180 |",
        "| 해상도(세운 뒤)별 영상 수 | 1920x1080 1개, 720x1280 179개 |",
        "| 밝기(영상별 평균, 0~255) | 최소 40.0, 중앙값 100.0, 최대 100.0 |",
        "| `05/0.mp4` | 180 | 90 | ambiguous_smallest_roll | 1.00 / 1.00 / 1.00 / 1.00 | 1.0 / 91.0 / 170.0 / -89.0 |",
        "| `32/10_2.mp4` | failed | InvalidDataError: bad file |",
    ):
        assert expected in lines, expected
    text = "\n".join(lines)
    assert "통과 180개, 위반 0개." in text
    assert "자동 점검 180개, 수동 보정 0개, 회전 메타데이터 그대로 0개. 적용한 회전별 영상 수: 0도 179개, 180도 1개." in text
    # 프레임당 시간: 시간이 있는 179개 x 0.5초 / (179 x 100 프레임) = 5 ms
    assert (
        "- 이번 실행에서 잰 것: 영상 179개, 프레임 17900개, 프레임당 5.00 ms (영상별 최소 5.00, 중앙값 5.00, 최대 5.00)."
        " 이번 실행은 영상을 최대 4개씩 동시에 처리했다(CPU를 나눠 쓴 값이다)."
    ) in lines
    assert "- 전 리포트에서 이어 온 것: 없음." in lines
    assert any(line.startswith("- 처리 시간을 모르는 것: 영상 1개.") for line in lines)
    assert "얼굴 화소는 어디에도 저장하지 않는다(계획서 1.2절의 결정 D13)" in text


def timed_row(video: str, status: str, elapsed_s: float | None, carried: str | None = None) -> dict[str, Any]:
    """프레임 100개짜리 추출된 영상의 손으로 만든 행. elapsed_s가 None이면 처리 시간을 모르는 행이다.

    carried는 처리 시간을 이어 온 리포트다: "pilot_report"(시험 추출 리포트), "own_report"(같은 이름의 리포트의 전 판).
    """
    row = report_row(
        video,
        status,
        frames_decoded=100,
        frames_skipped=0,
        frames_dropped=0,
        frames_kept=100,
        frames_face=100,
        face_rate=1.0,
        duration_ms=3300,
        mean_fps=30.0,
        variable_intervals=False,
        width=640,
        height=480,
        rotation=0,
        rotation_source="auto_check",
        rotation_reason="smallest_roll",
        metadata_rotation=0,
        rotation_ambiguous=False,
        display_flip=False,
        brightness_mean=100.0,
        t_ms_ok=True,
        elapsed_s=elapsed_s,
        ms_per_frame=None if elapsed_s is None else 1000.0 * elapsed_s / 100,
    )
    if carried is not None:
        row["_timing_carried"] = carried
    return row


def timing_section(summary: str) -> list[str]:
    """요약의 6절(처리 시간)에서 빈 줄을 뺀 줄들."""
    lines = summary.splitlines()
    start, end = lines.index("## 6. 처리 시간"), lines.index("## 7. 영상별 표")
    return [line for line in lines[start + 1 : end] if line]


def test_write_extraction_report_separates_measured_and_carried_timings(tmp_path: Path) -> None:
    """이번 실행에서 잰 처리 시간과 전 리포트에서 이어 온 처리 시간을 따로 적고, 처리 방식은 이번에 잰 값에만 붙인다."""

    def context(workers: int) -> extract.ReportContext:
        return extract.ReportContext(
            description="Data source: hand-made rows",
            command="python -m drowsy.extract --config config/local.yaml --dataset",
            config_path="config/local.yaml",
            out_root="data/raw/native",
            drop_sim=None,
            model_sha256="ab" * 32,
            mediapipe_version="0.10.32",
            rotation_check=extract.RotationCheckSettings(12, 2000, 15.0),
            overrides_key="extraction.rotation_overrides",
            min_face_ratio=0.7,
            workers=workers,
        )

    def section(tag: str, rows: list[dict[str, Any]], workers: int) -> list[str]:
        _, md_path = extract.write_extraction_report(tmp_path / tag, tag, rows, context(workers))
        return timing_section(md_path.read_text(encoding="utf-8"))

    # 이번에 잰 영상 둘: 0.5초와 1.5초, 100프레임씩 → (500 + 1500) ms / 200프레임 = 10 ms. 영상별 5와 15.
    # 시험 추출 리포트에서 이어 온 영상 둘: 0.8초씩 → 1600 ms / 200프레임 = 8 ms. 영상별 8과 8.
    # 이 리포트의 전 판에서 이어 온 영상 둘: 0.9초와 1.2초 → 2100 ms / 200프레임 = 10.5 ms. 영상별 9와 12.
    rows = [
        timed_row("a.mp4", "done", 0.5),
        timed_row("b.mp4", "done", 1.5),
        timed_row("c.mp4", "up_to_date", 0.8, carried="pilot_report"),
        timed_row("d.mp4", "up_to_date", 0.8, carried="pilot_report"),
        timed_row("e.mp4", "up_to_date", 0.9, carried="own_report"),
        timed_row("h.mp4", "up_to_date", 1.2, carried="own_report"),
        timed_row("f.mp4", "up_to_date", None),
        report_row("g.mp4", "failed", reason="InvalidDataError: bad file"),
    ]
    assert (extract.CARRIED_FROM_PILOT, extract.CARRIED_FROM_OWN) == ("pilot_report", "own_report")
    mixed = section("dataset_native", rows, workers=3)
    assert len(mixed) == 5
    assert mixed[0].startswith("영상 하나의 랜드마커 생성, 디코딩, 추론, 저장을 합친 시간이며 회전 점검은 뺀 값이다.")
    # 전체 추출의 리포트는 두 리포트의 값이 다 있을 때 어느 쪽을 쓰는지를 머리말에 적는다.
    assert "`pilot_native_videos.csv`)와 이 리포트의 전 판에 다 있으면 나중에 쓴 리포트의 값을 쓰고" in mixed[0]
    assert mixed[1] == (
        "- 이번 실행에서 잰 것: 영상 2개, 프레임 200개, 프레임당 10.00 ms (영상별 최소 5.00, 중앙값 10.00, 최대 15.00)."
        " 이번 실행은 영상을 최대 3개씩 동시에 처리했다(CPU를 나눠 쓴 값이다)."
    )
    # 이어 온 값은 그 값이 적혀 있던 리포트별로 한 줄씩이고, 어디서 어떻게 잰 값인지가 붙는다.
    assert mixed[2] == (
        "- 전 리포트에서 이어 온 것(시험 추출 리포트 `pilot_native_videos.csv`): 영상 2개, 프레임 200개, 프레임당 8.00 ms"
        " (영상별 최소 8.00, 중앙값 8.00, 최대 8.00). 이번 실행이 건너뛴 영상이고, 값은 시험 추출(`--dataset --pilot`)이"
        " 영상을 한 번에 하나씩 처리하며 잰 것이다."
    )
    assert mixed[3] == (
        "- 전 리포트에서 이어 온 것(이 리포트의 전 판 `dataset_native_videos.csv`): 영상 2개, 프레임 200개, 프레임당"
        " 10.50 ms (영상별 최소 9.00, 중앙값 10.50, 최대 12.00). 이번 실행이 건너뛴 영상이고, 값은 그 영상을 추출한 전의"
        " 실행에서 잰 것이다. 그 실행이 영상을 몇 개씩 동시에 처리했는지는 이 리포트에 남아 있지 않다."
    )
    # 이어 온 값에는 이번 실행의 처리 방식을 붙이지 않는다.
    assert not any("이번 실행은" in line or "동시에 처리했다" in line for line in mixed[2:])
    assert mixed[4].startswith("- 처리 시간을 모르는 것: 영상 1개.")
    # 묶음을 합친 값은 어디에도 적지 않는다: 이어 온 넷 3700 ms / 400프레임 = 9.25 ms, 여섯 전부 5700 / 600 = 9.50 ms.
    assert not any("9.25 ms" in line or "9.50 ms" in line for line in mixed)
    # 누락 모사 추출의 리포트는 같은 폴더 이름의 시험 추출 리포트를 가리킨다.
    assert section("dataset_fps15", rows, workers=3)[2].startswith(
        "- 전 리포트에서 이어 온 것(시험 추출 리포트 `pilot_fps15_videos.csv`): 영상 2개, "
    )

    # 차례로 처리한 실행. 시험 추출의 리포트에는 "나중에 쓴 리포트" 문장이 없다(이어 올 리포트가 자기 전 판 하나다).
    sequential = section("pilot_native", rows[:2], workers=1)
    assert sequential[1].endswith("이번 실행은 영상을 한 번에 하나씩 처리했다.")
    assert sequential[2:] == ["- 전 리포트에서 이어 온 것: 없음."]
    assert "나중에 쓴 리포트" not in sequential[0]
    # 끊긴 시험 추출이 자기 전 판에서 이어 온 값: 시험 추출은 언제나 한 번에 하나씩 처리한다.
    pilot_resumed = section("pilot_native", rows[4:6], workers=1)
    assert pilot_resumed[2] == (
        "- 전 리포트에서 이어 온 것(이 리포트의 전 판 `pilot_native_videos.csv`): 영상 2개, 프레임 200개, 프레임당"
        " 10.50 ms (영상별 최소 9.00, 중앙값 10.50, 최대 12.00). 이번 실행이 건너뛴 영상이고, 값은 전의 시험 추출이 영상을"
        " 한 번에 하나씩 처리하며 잰 것이다."
    )

    # 모두 건너뛴 실행(이어 하기): 이번 실행의 처리 방식을 말하는 문장이 없다. 설정의 작업자 수와 무관하다.
    resumed = section("input_dir_native", rows[4:6], workers=8)
    assert resumed[1] == "- 이번 실행에서 잰 것: 없음."
    assert resumed[2].startswith(
        "- 전 리포트에서 이어 온 것(이 리포트의 전 판 `input_dir_native_videos.csv`): 영상 2개, 프레임 200개, 프레임당 10.50 ms"
    )
    assert len(resumed) == 3 and not any("pilot" in line for line in resumed)  # 데이터셋 추출이 아니면 시험 추출 리포트가 없다
    assert not any("이번 실행은" in line or "8개" in line for line in resumed)

    # 추출된 영상이 없는 실행
    nothing = section("unlabeled_native", rows[7:], workers=2)
    assert nothing[1:] == ["- 이번 실행에서 잰 것: 없음.", "- 전 리포트에서 이어 온 것: 없음."]


def test_previous_timings_take_the_newer_report_and_tell_where_the_value_was_measured(tmp_path: Path) -> None:
    """건너뛴 영상의 처리 시간: 시험 추출 리포트와 같은 이름의 리포트의 전 판 중 나중에 쓴 쪽의 값, 그리고 그 출처."""
    # (frames_kept, elapsed_s, ms_per_frame). CSV에 적힌 글자 그대로다.
    own = {
        "both_differ.mp4": ("100", "0.5", "5"),
        "both_equal.mp4": ("100", "0.7", "7"),
        "own_only.mp4": ("100", "0.9", "9"),
        "own_stale.mp4": ("90", "0.3", "3.33333"),
    }
    pilot = {
        "both_differ.mp4": ("100", "0.2", "2"),
        "both_equal.mp4": ("100", "0.7", "7"),
        "pilot_only.mp4": ("100", "0.4", "4"),
        "own_stale.mp4": ("100", "0.6", "6"),
    }
    own_newer = extract.PreviousTimings(own, pilot, pilot_is_newer=False)
    pilot_newer = extract.PreviousTimings(own, pilot, pilot_is_newer=True)
    # 두 리포트의 값이 다르다: 나중에 쓴 리포트의 값이다.
    assert own_newer.lookup("both_differ.mp4", 100) == (0.5, 5.0, "own_report")
    assert pilot_newer.lookup("both_differ.mp4", 100) == (0.2, 2.0, "pilot_report")
    # 두 리포트의 값이 같다: 전체 추출의 리포트가 시험 추출의 값을 옮겨 적은 것이므로 어느 쪽이 나중이든 시험 추출의 값이다
    # (이어 하기가 전체 추출의 리포트를 다시 써서 더 나중 파일이 돼도 출처가 바뀌지 않는다).
    assert own_newer.lookup("both_equal.mp4", 100) == (0.7, 7.0, "pilot_report")
    assert pilot_newer.lookup("both_equal.mp4", 100) == (0.7, 7.0, "pilot_report")
    # 한쪽에만 있다.
    assert pilot_newer.lookup("own_only.mp4", 100) == (0.9, 9.0, "own_report")
    assert own_newer.lookup("pilot_only.mp4", 100) == (0.4, 4.0, "pilot_report")
    # 프레임 수가 지금의 결과 파일과 다른 행은 다른 추출의 값이라 쓰지 않는다. 나중에 쓴 리포트의 행이어도 그렇다.
    assert own_newer.lookup("own_stale.mp4", 100) == (0.6, 6.0, "pilot_report")
    assert pilot_newer.lookup("own_stale.mp4", 90) == (0.3, 3.33333, "own_report")
    assert pilot_newer.lookup("own_stale.mp4", 80) is None
    assert pilot_newer.lookup("unknown.mp4", 100) is None

    # 파일에서 읽기: "나중"은 두 CSV 파일의 수정 시각으로 정한다. 처리 시간이 빈 행은 읽지 않는다.
    def write_csv(name: str, table: list[tuple[str, str, str, str]]) -> Path:
        path = tmp_path / name
        lines = ["video,frames_kept,elapsed_s,ms_per_frame"] + [",".join(row) for row in table]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    own_csv = write_csv("dataset_native_videos.csv", [("a.mp4", "100", "0.5", "5"), ("b.mp4", "100", "", "")])
    pilot_csv = write_csv("pilot_native_videos.csv", [("a.mp4", "100", "0.2", "2")])
    os.utime(own_csv, ns=(1_000_000_000, 1_000_000_000))
    os.utime(pilot_csv, ns=(2_000_000_000, 2_000_000_000))
    read = extract.read_previous_timings(own_csv, pilot_csv)
    assert read.own == {"a.mp4": ("100", "0.5", "5")} and read.pilot == {"a.mp4": ("100", "0.2", "2")}
    assert read.pilot_is_newer is True and read.lookup("a.mp4", 100) == (0.2, 2.0, "pilot_report")
    os.utime(own_csv, ns=(3_000_000_000, 3_000_000_000))
    assert extract.read_previous_timings(own_csv, pilot_csv).pilot_is_newer is False
    assert extract.read_previous_timings(own_csv, pilot_csv).lookup("a.mp4", 100) == (0.5, 5.0, "own_report")
    os.utime(pilot_csv, ns=(3_000_000_000, 3_000_000_000))  # 수정 시각이 같으면 시험 추출 리포트를 더 나중으로 보지 않는다
    assert extract.read_previous_timings(own_csv, pilot_csv).pilot_is_newer is False
    # 같은 이름의 리포트가 아직 없으면(시험 추출 뒤의 첫 전체 추출) 시험 추출의 값을 쓴다.
    first = extract.read_previous_timings(tmp_path / "missing.csv", pilot_csv)
    assert first.own == {} and first.lookup("a.mp4", 100) == (0.2, 2.0, "pilot_report")
    # 시험 추출 리포트를 주지 않으면(--dataset이 아닌 입력) 같은 이름의 리포트만 본다.
    alone = extract.read_previous_timings(own_csv)
    assert alone.pilot == {} and alone.pilot_is_newer is False
    assert alone.lookup("a.mp4", 100) == (0.5, 5.0, "own_report")
    nothing = extract.read_previous_timings(tmp_path / "missing.csv", tmp_path / "also_missing.csv")
    assert (nothing.own, nothing.pilot, nothing.pilot_is_newer) == ({}, {}, False)


class StepClock:
    """`extract` 모듈의 `time` 자리에 넣는 가짜 시계. `perf_counter()`를 부를 때마다 step초씩 간다.

    `extract_video`는 시작과 끝에 한 번씩 부르므로 영상 하나의 처리 시간이 정확히 step초가 된다. step에는 2의 거듭제곱
    분수(0.5, 0.25, 0.125)만 써서 더하고 빼도 오차가 없게 한다. 그래서 리포트의 값을 손으로 계산해 글자까지 비교할 수 있다.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.step = 0.5

    def perf_counter(self) -> float:
        self.now += self.step
        return self.now


@pytest.fixture
def step_clock(monkeypatch: pytest.MonkeyPatch) -> StepClock:
    clock = StepClock()
    monkeypatch.setattr(extract, "time", clock)
    return clock


def stamp(path: Path, seconds: int) -> None:
    """파일의 수정 시각을 정해진 값으로 바꾼다. 두 리포트 중 어느 쪽이 나중에 쓰였는지를 시험이 정하게 한다."""
    os.utime(path, ns=(seconds * 10**9, seconds * 10**9))


def six_video_dataset(tmp_path: Path) -> Path:
    """피험자 01, 02의 영상 여섯 개. 라벨 0, 5, 10의 영상이 각각 4, 5, 6프레임이다. 목록 순서는 01/0, 01/5, 01/10, 02/0, ..."""
    root = tmp_path / "rldd"
    for subject in ("01", "02"):
        for label in ("0", "5", "10"):
            write_marker_video(root / subject / f"{label}.mp4", PTS_10FPS[: 4 + int(label) // 5])
    return root


def timing_report(reports_dir: Path, tag: str) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """리포트의 (영상, 상태, elapsed_s, ms_per_frame) 행들과 요약 6절의 줄들. CSV의 열이 47개 그대로인지도 본다."""
    rows, summary = read_report(reports_dir, tag)
    assert len(rows[0]) == 47 and list(rows[0]) == list(extract.REPORT_COLUMNS)
    assert "_timing_carried" not in rows[0]  # 출처 표시는 CSV에 쓰지 않는다
    return [(row["video"], row["status"], row["elapsed_s"], row["ms_per_frame"]) for row in rows], timing_section(summary)


def pilot_line(figures: str) -> str:
    """요약 6절에서 시험 추출 리포트의 값을 이어 온 줄(손으로 적은 기대값)."""
    return (
        f"- 전 리포트에서 이어 온 것(시험 추출 리포트 `pilot_native_videos.csv`): {figures}. 이번 실행이 건너뛴 영상이고,"
        " 값은 시험 추출(`--dataset --pilot`)이 영상을 한 번에 하나씩 처리하며 잰 것이다."
    )


def own_line(figures: str) -> str:
    """요약 6절에서 같은 이름의 리포트의 전 판에 있던 값을 이어 온 줄(손으로 적은 기대값)."""
    return (
        f"- 전 리포트에서 이어 온 것(이 리포트의 전 판 `dataset_native_videos.csv`): {figures}. 이번 실행이 건너뛴 영상이고,"
        " 값은 그 영상을 추출한 전의 실행에서 잰 것이다. 그 실행이 영상을 몇 개씩 동시에 처리했는지는 이 리포트에 남아 있지 않다."
    )


# 영상 셋(4, 5, 6프레임)을 0.125초씩에 처리: 375 ms / 15프레임 = 25 ms. 영상별 31.25, 25, 20.8333 ms.
EIGHTH_3 = "영상 3개, 프레임 15개, 프레임당 25.00 ms (영상별 최소 20.83, 중앙값 25.00, 최대 31.25)"
# 영상 셋을 0.5초씩에 처리: 1500 ms / 15프레임 = 100 ms. 영상별 125, 100, 83.3333 ms. 여섯이면 3000 ms / 30프레임.
HALF_3 = "영상 3개, 프레임 15개, 프레임당 100.00 ms (영상별 최소 83.33, 중앙값 100.00, 최대 125.00)"
HALF_6 = "영상 6개, 프레임 30개, 프레임당 100.00 ms (영상별 최소 83.33, 중앙값 100.00, 최대 125.00)"
EIGHTH_ROWS = [("01/0.mp4", "0.125", "31.25"), ("01/5.mp4", "0.125", "25"), ("01/10.mp4", "0.125", "20.8333")]
HALF_ROWS_01 = [("01/0.mp4", "0.5", "125"), ("01/5.mp4", "0.5", "100"), ("01/10.mp4", "0.5", "83.3333")]
HALF_ROWS_02 = [("02/0.mp4", "0.5", "125"), ("02/5.mp4", "0.5", "100"), ("02/10.mp4", "0.5", "83.3333")]


def with_status(rows: list[tuple[str, str, str]], status: str) -> list[tuple[str, str, str, str]]:
    return [(video, status, elapsed, per_frame) for video, elapsed, per_frame in rows]


def test_cli_timings_of_a_pilot_a_parallel_run_and_a_resume(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    step_clock: StepClock,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """시험 추출(차례로) → 전체 추출(동시에) → 이어 하기: 요약의 처리 시간이 어느 실행에서, 어느 리포트에서 온 값인지 구분된다."""
    root = six_video_dataset(tmp_path)
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root)
    reports = fake_model_setup.reports_dir
    dataset_csv = reports / "extraction" / "extraction" / "dataset_native_videos.csv"
    pilot_csv = dataset_csv.with_name("pilot_native_videos.csv")

    # 1. 시험 추출: 앞 3개(피험자 01)를 차례로, 영상마다 0.125초.
    step_clock.step = 0.125
    assert extract.main(["--config", str(config), "--dataset", "--pilot", "3"]) == 0
    capsys.readouterr()
    stamp(pilot_csv, 1000)
    rows, section = timing_report(reports, "pilot_native")
    assert rows == with_status(EIGHTH_ROWS, "done")
    assert section[1:] == [
        f"- 이번 실행에서 잰 것: {EIGHTH_3}. 이번 실행은 영상을 한 번에 하나씩 처리했다.",
        "- 전 리포트에서 이어 온 것: 없음.",
    ]

    # 2. 전체 추출(작업자 2), 영상마다 0.5초: 시험 추출의 3개는 건너뛰고 나머지 3개를 추출한다.
    step_clock.step = 0.5
    assert extract.main(["--config", str(config), "--dataset", "--workers", "2"]) == 0
    assert "extracted 3, up to date 3, failed 0" in capsys.readouterr().out
    stamp(dataset_csv, 2000)
    rows, section = timing_report(reports, "dataset_native")
    assert rows == with_status(EIGHTH_ROWS, "up_to_date") + with_status(HALF_ROWS_02, "done")
    # 시험 추출에서 차례로 잰 3개는 따로 적히고 "동시에 처리했다"는 말이 붙지 않는다.
    assert section[1:] == [
        f"- 이번 실행에서 잰 것: {HALF_3}. 이번 실행은 영상을 최대 2개씩 동시에 처리했다(CPU를 나눠 쓴 값이다).",
        pilot_line(EIGHTH_3),
    ]

    # 3. 이어 하기: 추출할 것이 없다. 이어 온 값은 적혀 있던 리포트별로 두 줄이다(전에는 한 줄에 합쳐 62.50 ms라고 적었다:
    #    1875 ms / 30프레임).
    assert extract.main(["--config", str(config), "--dataset", "--workers", "2"]) == 0
    assert "extracted 0, up to date 6, failed 0" in capsys.readouterr().out
    stamp(dataset_csv, 3000)
    rows, section = timing_report(reports, "dataset_native")
    resumed = with_status(EIGHTH_ROWS, "up_to_date") + with_status(HALF_ROWS_02, "up_to_date")
    assert rows == resumed
    assert section[1:] == ["- 이번 실행에서 잰 것: 없음.", pilot_line(EIGHTH_3), own_line(HALF_3)]
    assert not any("62.50" in line or "이번 실행은" in line for line in section)

    # 4. 이어 하기를 거듭해도 값과 출처가 그대로다. 이제 전체 추출의 리포트가 더 나중 파일이지만(3000 > 1000), 앞 3개의 값은
    #    시험 추출 리포트의 값과 같으므로 시험 추출에서 잰 것으로 적힌다.
    assert extract.main(["--config", str(config), "--dataset", "--workers", "2"]) == 0
    capsys.readouterr()
    stamp(dataset_csv, 4000)
    assert timing_report(reports, "dataset_native") == (resumed, section)

    # 5. 영상 하나만 다시 추출하는 실행(결과 파일을 지웠다), 0.25초: 작업자 설정이 2여도 한 번에 하나만 처리했다.
    #    250 ms / 6프레임 = 41.67 ms. 전 판에서 이어 온 둘(02/0, 02/5): 1000 ms / 9프레임 = 111.11 ms, 영상별 125와 100.
    (fake_model_setup.raw_dir / "native" / "02" / "10.parquet").unlink()
    step_clock.step = 0.25
    assert extract.main(["--config", str(config), "--dataset", "--workers", "2"]) == 0
    capsys.readouterr()
    rows, section = timing_report(reports, "dataset_native")
    assert rows == resumed[:5] + [("02/10.mp4", "done", "0.25", "41.6667")]
    assert section[1:] == [
        "- 이번 실행에서 잰 것: 영상 1개, 프레임 6개, 프레임당 41.67 ms (영상별 최소 41.67, 중앙값 41.67, 최대 41.67)."
        " 이번 실행은 영상을 한 번에 하나씩 처리했다.",
        pilot_line(EIGHTH_3),
        own_line("영상 2개, 프레임 9개, 프레임당 111.11 ms (영상별 최소 100.00, 중앙값 112.50, 최대 125.00)"),
    ]


@pytest.mark.parametrize("pilot_report_is_newer", [True, False])
def test_cli_timings_after_a_full_run_then_a_pilot_then_resumes(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    step_clock: StepClock,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    pilot_report_is_newer: bool,
) -> None:
    """전체 추출(동시에) → 시험 추출(앞 3개를 다시 추출) → 이어 하기: 건너뛴 영상의 처리 시간은 나중에 쓴 리포트의 값이다.

    시험 추출은 결과가 있어도 다시 추출하므로, 그 뒤에 디스크에 있는 앞 3개의 결과는 시험 추출이 만든 것이다. 이어 하기의
    리포트는 그 3개에 시험 추출에서 잰 값을 적어야 한다(전에는 전체 추출 때의 옛 값을 적었다).
    pilot_report_is_newer가 거짓인 경우는 대조군이다: 시험 추출 리포트의 수정 시각을 전체 추출 리포트보다 앞으로 돌려놓으면
    전체 추출 리포트의 값을 쓴다(언제나 시험 추출 리포트가 이기는 것이 아니라 나중에 쓴 쪽이 이긴다).
    """
    root = six_video_dataset(tmp_path)
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, root)
    reports = fake_model_setup.reports_dir
    dataset_csv = reports / "extraction" / "extraction" / "dataset_native_videos.csv"
    pilot_csv = dataset_csv.with_name("pilot_native_videos.csv")
    resume = ["--config", str(config), "--dataset", "--workers", "2"]

    # 1. 전체 추출(작업자 2), 영상마다 0.5초.
    step_clock.step = 0.5
    assert extract.main(resume) == 0
    assert "extracted 6, up to date 0, failed 0" in capsys.readouterr().out
    stamp(dataset_csv, 2000)
    rows, section = timing_report(reports, "dataset_native")
    assert rows == with_status(HALF_ROWS_01 + HALF_ROWS_02, "done")
    assert section[1:] == [
        f"- 이번 실행에서 잰 것: {HALF_6}. 이번 실행은 영상을 최대 2개씩 동시에 처리했다(CPU를 나눠 쓴 값이다).",
        "- 전 리포트에서 이어 온 것: 없음.",
    ]

    # 2. 시험 추출: 앞 3개를 차례로 다시 추출한다, 영상마다 0.125초. 전체 추출의 리포트는 건드리지 않는다.
    step_clock.step = 0.125
    assert extract.main(["--config", str(config), "--dataset", "--pilot", "3"]) == 0
    assert "extracted 3, up to date 0, failed 0" in capsys.readouterr().out
    stamp(pilot_csv, 3000 if pilot_report_is_newer else 1000)
    assert timing_report(reports, "pilot_native")[0] == with_status(EIGHTH_ROWS, "done")
    assert timing_report(reports, "dataset_native")[0] == with_status(HALF_ROWS_01 + HALF_ROWS_02, "done")

    # 3. 이어 하기.
    assert extract.main(resume) == 0
    assert "extracted 0, up to date 6, failed 0" in capsys.readouterr().out
    stamp(dataset_csv, 4000)
    rows, section = timing_report(reports, "dataset_native")
    if pilot_report_is_newer:
        expected_rows = with_status(EIGHTH_ROWS + HALF_ROWS_02, "up_to_date")
        expected_lines = ["- 이번 실행에서 잰 것: 없음.", pilot_line(EIGHTH_3), own_line(HALF_3)]
    else:
        expected_rows = with_status(HALF_ROWS_01 + HALF_ROWS_02, "up_to_date")
        expected_lines = ["- 이번 실행에서 잰 것: 없음.", own_line(HALF_6)]
    assert rows == expected_rows
    assert section[1:] == expected_lines

    # 4. 이어 하기를 한 번 더: 방금 다시 쓴 전체 추출의 리포트가 어느 경우에나 더 나중 파일이다(4000). 값과 출처가 그대로다.
    assert extract.main(resume) == 0
    capsys.readouterr()
    stamp(dataset_csv, 5000)
    rows, section = timing_report(reports, "dataset_native")
    assert rows == expected_rows and section[1:] == expected_lines
    if not pilot_report_is_newer:
        return

    # 5. 그 뒤 전체 추출이 01/5를 다시 추출했다(결과 파일을 지웠다), 0.25초: 250 ms / 5프레임 = 50 ms.
    #    시험 추출 리포트에서 이어 온 것은 01/0과 01/10: 250 ms / 10프레임 = 25 ms, 영상별 31.25와 20.8333(중앙값 26.04).
    (fake_model_setup.raw_dir / "native" / "01" / "5.parquet").unlink()
    step_clock.step = 0.25
    assert extract.main(resume) == 0
    assert "extracted 1, up to date 5, failed 0" in capsys.readouterr().out
    stamp(dataset_csv, 6000)
    rows, section = timing_report(reports, "dataset_native")
    assert rows[1] == ("01/5.mp4", "done", "0.25", "50")
    pilot_two = "영상 2개, 프레임 10개, 프레임당 25.00 ms (영상별 최소 20.83, 중앙값 26.04, 최대 31.25)"
    assert section[1:] == [
        "- 이번 실행에서 잰 것: 영상 1개, 프레임 5개, 프레임당 50.00 ms (영상별 최소 50.00, 중앙값 50.00, 최대 50.00)."
        " 이번 실행은 영상을 한 번에 하나씩 처리했다.",
        pilot_line(pilot_two),
        own_line(HALF_3),
    ]

    # 6. 이어 하기: 01/5의 값은 두 리포트에서 다르고(0.25초와 0.125초) 전체 추출의 리포트가 나중에 쓰였으므로 0.25초다.
    #    전 판에서 이어 온 넷(01/5, 02/0, 02/5, 02/10): 1750 ms / 20프레임 = 87.5 ms, 영상별 50, 125, 100, 83.3333.
    assert extract.main(resume) == 0
    capsys.readouterr()
    rows, section = timing_report(reports, "dataset_native")
    assert rows == with_status(
        [EIGHTH_ROWS[0], ("01/5.mp4", "0.25", "50"), EIGHTH_ROWS[2]] + HALF_ROWS_02, "up_to_date"
    )
    assert section[1:] == [
        "- 이번 실행에서 잰 것: 없음.",
        pilot_line(pilot_two),
        own_line("영상 4개, 프레임 20개, 프레임당 87.50 ms (영상별 최소 50.00, 중앙값 91.67, 최대 125.00)"),
    ]


# ---------------------------------------------------------------------------
# 이어 하기와 원본 영상: 결과에 적힌 원본 파일 크기, 이름이 같은 다른 영상, 옮긴 데이터 폴더
# ---------------------------------------------------------------------------


def strip_metadata_keys(path: Path, *keys: str) -> None:
    """추출 결과 파일의 메타데이터에서 키를 지워 다시 쓴다(그 키를 적기 전의 추출 결과를 흉내 낸다)."""
    import json

    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[extract.METADATA_KEY].decode("utf-8"))
    for key in keys:
        del metadata[key]
    table = table.replace_schema_metadata({extract.METADATA_KEY: json.dumps(metadata, sort_keys=True).encode("utf-8")})
    pq.write_table(table, path, compression="zstd", use_dictionary=False)


def replace_line(label: str, out: Path, recorded: Path, recorded_size: int, size: int) -> str:
    """다른 영상의 결과를 다시 추출한다고 알리는 줄(손으로 적은 기대값)."""
    return (
        f"replace {label}: {out} was extracted from {recorded} ({recorded_size} bytes); this video has {size} bytes, "
        "so it is extracted again and that result is overwritten"
    )


def moved_skip_line(label: str, out: Path, recorded: Path, size: int) -> str:
    """건너뛴 결과가 다른 경로의 (크기가 같은) 영상에서 나온 것일 때의 줄(손으로 적은 기대값)."""
    return (
        f"skip {label}: {out} is up to date (extracted from {recorded}, a file of the same size as this video: "
        f"{size} bytes)"
    )


def test_output_is_current_compares_the_recorded_source_size(
    tmp_path: Path, pipeline_spec: PipelineSpec, fake_mediapipe: list[FakeLandmarker]
) -> None:
    first = tmp_path / "a" / "clip.mp4"
    write_plain_video(first, PTS_30FPS)
    other = tmp_path / "b" / "clip.mp4"  # 이름만 같은 다른 영상(5프레임)
    write_plain_video(other, PTS_30FPS[:5])
    size = os.path.getsize(first)
    assert size != os.path.getsize(other)
    out = tmp_path / "clip.parquet"
    extract_video(first, tmp_path / "unused.task", out, spec=pipeline_spec)
    metadata = read_extract_metadata(out)
    # 새 키 둘. 전부터 있던 `source`는 받은 경로 그대로다.
    assert (metadata["source"], metadata["source_name"], metadata["source_size_bytes"]) == (str(first), "clip.mp4", size)

    assert output_is_current(out, pipeline_spec, None, None, first) is True
    assert output_is_current(out, pipeline_spec, None, None, other) is False  # 크기가 다르다
    assert output_is_current(out, pipeline_spec, None, None) is True  # 영상을 주지 않으면 비교하지 않는다
    moved = tmp_path / "moved" / "renamed.mp4"  # 같은 파일을 다른 경로, 다른 이름으로 옮겼다
    moved.parent.mkdir()
    shutil.copyfile(first, moved)
    assert output_is_current(out, pipeline_spec, None, None, moved) is True
    link = tmp_path / "link.mp4"  # 심볼릭 링크는 가리키는 파일의 크기를 본다
    link.symlink_to(other)
    assert output_is_current(out, pipeline_spec, None, None, link) is False
    # 크기가 적혀 있는데 지금의 영상을 읽을 수 없으면 최신이 아니다(추출을 시도해 오류가 드러나게 한다).
    assert output_is_current(out, pipeline_spec, None, None, tmp_path / "missing.mp4") is False
    # 알려진 한계: 크기가 바이트 단위로 같은 다른 파일은 가려내지 못한다(내용을 읽지 않는다).
    same_size = tmp_path / "same_size.mp4"
    same_size.write_bytes(bytes(size))
    assert output_is_current(out, pipeline_spec, None, None, same_size) is True
    assert output_is_current(out, pipeline_spec, None, 90, first) is False  # 나머지 조건은 그대로 본다

    # 원본 파일 크기를 적기 전의 결과(키가 없다): 전과 같이 크기를 비교하지 않고 최신으로 본다.
    strip_metadata_keys(out, "source_size_bytes", "source_name")
    assert "source_size_bytes" not in read_extract_metadata(out)
    assert output_is_current(out, pipeline_spec, None, None, other) is True
    assert output_is_current(out, pipeline_spec, None, None, tmp_path / "missing.mp4") is True
    assert [row.t_ms for row in read_raw_frames(out)] == T_30FPS  # 읽기도 그대로 된다


def test_cli_video_with_the_same_name_in_another_folder_is_extracted_again(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--video`: 다른 폴더의 같은 이름의 영상은 건너뛰지 않고 추출한다(전에는 첫 영상의 결과를 최신이라며 건너뛰었다)."""
    first = tmp_path / "trip_a" / "clip.mp4"  # 똑바로 선 6프레임
    write_marker_video(first, PTS_10FPS[:6])
    second = tmp_path / "trip_b" / "clip.mp4"  # 시계 방향 90도로 누운 9프레임
    write_marker_video(second, PTS_10FPS[:9], quarter_turns_cw=1)
    size_a, size_b = os.path.getsize(first), os.path.getsize(second)
    assert size_a != size_b
    out = fake_model_setup.raw_dir / "native" / "clip.parquet"
    base = ["--config", str(fake_model_setup.config), "--video"]

    assert extract.main([*base, str(first)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("done clip.mp4: ") and len(lines) == 2  # 건너뛰거나 덮어쓴다는 줄이 없다
    metadata = read_extract_metadata(out)
    assert (metadata["source"], metadata["source_name"], metadata["source_size_bytes"]) == (str(first), "clip.mp4", size_a)
    assert len(list(read_raw_frames(out))) == 6

    # 다른 폴더의 같은 이름: 추출하고, 결과는 둘째 영상을 적는다. 회전 점검도 둘째 영상으로 다시 한다.
    created = len(marker_mediapipe.created)
    assert extract.main([*base, str(second)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == replace_line("clip.mp4", out, first, size_a, size_b)
    assert lines[1].startswith("done clip.mp4: ") and "rotation: auto check chose 90 " in lines[1]
    assert lines[2].startswith("extracted 1, up to date 0, failed 0")
    assert len(marker_mediapipe.created) == created + 5  # 점검용 넷과 추출용 하나
    metadata = read_extract_metadata(out)
    assert (metadata["source"], metadata["source_name"], metadata["source_size_bytes"]) == (str(second), "clip.mp4", size_b)
    assert (metadata["rotation"], metadata["rotation_check"]["reason"]) == (90, "ambiguous_smallest_roll")
    assert [(row.w, row.h) for row in read_raw_frames(out)] == [(W, H)] * 9  # 둘째 영상의 9프레임을 세워서 추출했다

    # 같은 영상을 다시 주면 건너뛴다. 적힌 원본 경로가 같으므로 덧붙는 말이 없다.
    modified = out.stat().st_mtime_ns
    assert extract.main([*base, str(second)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == f"skip clip.mp4: {out} is up to date"

    # 데이터 폴더를 옮겼다(같은 파일, 다른 경로): 다시 추출하지 않고, 건너뛴 줄이 결과에 적힌 원본 경로를 알려 준다.
    moved = tmp_path / "moved"
    second.parent.rename(moved)
    created = len(marker_mediapipe.created)
    assert extract.main([*base, str(moved / "clip.mp4")]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == moved_skip_line("clip.mp4", out, second, size_b)
    assert lines[1].startswith("extracted 0, up to date 1, failed 0")
    assert len(marker_mediapipe.created) == created and out.stat().st_mtime_ns == modified
    assert read_extract_metadata(out)["source"] == str(second)  # 결과 파일은 그대로다

    # `--rotation`으로 만든 결과(수동 값의 갈래)도 같다: 같은 영상이면 건너뛰고, 크기가 다른 영상이면 다시 추출한다.
    by_hand = [*base, str(moved / "clip.mp4"), "--rotation", "90"]
    assert extract.main(by_hand) == 0  # 자동 점검으로 만든 결과이므로 수동 값으로 다시 추출한다
    capsys.readouterr()
    assert extract.main(by_hand) == 0
    assert capsys.readouterr().out.splitlines()[0] == f"skip clip.mp4: {out} is up to date"
    assert extract.main([*base, str(first), "--rotation", "90"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == replace_line("clip.mp4", out, moved / "clip.mp4", size_b, size_a)
    assert lines[1].startswith("done clip.mp4: ")
    assert read_extract_metadata(out)["source"] == str(first) and len(list(read_raw_frames(out))) == 6

    # 원본 파일 크기를 적기 전의 결과(키가 없다)는 전과 같이 건너뛴다. 건너뛴 줄이 적힌 원본과, 비교하지 못했다는 것을 알린다.
    strip_metadata_keys(out, "source_size_bytes", "source_name")
    assert extract.main(by_hand) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        f"skip clip.mp4: {out} is up to date (extracted from {first}; the result records no source size, "
        "so the two files were not compared)"
    )
    assert len(list(read_raw_frames(out))) == 6  # 첫 영상의 결과 그대로다


def test_cli_input_dir_extracts_a_same_name_video_again_and_skips_a_moved_folder(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--input-dir`: 두 폴더를 차례로 주면 이름만 같은 다른 영상은 추출하고 같은 파일은 건너뛴다. 옮긴 폴더는 다시 추출하지 않는다."""
    trip_a, trip_b = tmp_path / "trip_a", tmp_path / "trip_b"
    write_marker_video(trip_a / "clip.mp4", PTS_10FPS[:6])
    write_marker_video(trip_a / "cam" / "front.mp4", PTS_10FPS[:5])
    write_marker_video(trip_b / "clip.mp4", PTS_10FPS[:9], quarter_turns_cw=1)  # 이름만 같은 다른 영상
    (trip_b / "cam").mkdir()
    shutil.copyfile(trip_a / "cam" / "front.mp4", trip_b / "cam" / "front.mp4")  # 같은 파일의 사본
    size_a, size_b = os.path.getsize(trip_a / "clip.mp4"), os.path.getsize(trip_b / "clip.mp4")
    front_size = os.path.getsize(trip_a / "cam" / "front.mp4")
    assert size_a != size_b
    native = fake_model_setup.raw_dir / "native"
    base = ["--config", str(fake_model_setup.config), "--input-dir"]

    assert extract.main([*base, str(trip_a)]) == 0
    assert "extracted 2, up to date 0, failed 0" in capsys.readouterr().out
    assert read_extract_metadata(native / "clip.parquet")["source"] == str(trip_a / "clip.mp4")

    assert extract.main([*base, str(trip_b)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == moved_skip_line("cam/front.mp4", native / "cam" / "front.parquet", trip_a / "cam" / "front.mp4", front_size)
    assert lines[1] == replace_line("clip.mp4", native / "clip.parquet", trip_a / "clip.mp4", size_a, size_b)
    assert lines[2].startswith("done clip.mp4: ") and lines[3].startswith("extracted 1, up to date 1, failed 0")
    metadata = read_extract_metadata(native / "clip.parquet")
    assert (metadata["source"], metadata["source_name"], metadata["source_size_bytes"]) == (
        str(trip_b / "clip.mp4"),
        "clip.mp4",
        size_b,
    )
    assert [(row.w, row.h) for row in read_raw_frames(native / "clip.parquet")] == [(W, H)] * 9
    report, _ = read_report(fake_model_setup.reports_dir, "input_dir_native")
    assert [(row["video"], row["status"], row["frames_kept"]) for row in report] == [
        ("cam/front.mp4", "up_to_date", "5"),
        ("clip.mp4", "done", "9"),
    ]
    assert len(report[0]) == 47  # 리포트의 열은 그대로다(원본 크기는 결과 파일의 메타데이터에만 적는다)

    # 데이터 폴더를 옮겼다(같은 파일들, 다른 경로): 하나도 다시 추출하지 않는다.
    moved = tmp_path / "moved"
    trip_b.rename(moved)
    created = len(marker_mediapipe.created)
    modified = {path: path.stat().st_mtime_ns for path in native.rglob("*.parquet")}
    assert extract.main([*base, str(moved)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        moved_skip_line("cam/front.mp4", native / "cam" / "front.parquet", trip_a / "cam" / "front.mp4", front_size),
        moved_skip_line("clip.mp4", native / "clip.parquet", trip_b / "clip.mp4", size_b),
        f"extracted 0, up to date 2, failed 0 -> {native}",
    ]
    assert len(marker_mediapipe.created) == created
    assert {path: path.stat().st_mtime_ns for path in native.rglob("*.parquet")} == modified


def test_cli_dataset_and_unlabeled_extract_a_replaced_video_again(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    marker_mediapipe: SimpleNamespace,
    inline_pool: None,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--dataset`, `--unlabeled`: 영상 파일을 다른 녹화로 바꾸면(이름은 그대로) 그 영상만 다시 추출한다."""
    root = tmp_path / "rldd"
    for label in ("0", "5", "10"):
        write_marker_video(root / "05" / f"{label}.mp4", PTS_10FPS[:6])
    team = tmp_path / "team"
    write_marker_video(team / "drive" / "clip.mp4", PTS_10FPS[:6])
    write_marker_video(team / "drive" / "keep.mp4", PTS_10FPS[:5])
    base = write_dataset_config(tmp_path, config_path, fake_model_setup, root)
    config = edit_config(base, "both.yaml", paths__unlabeled_root=str(team))
    raw = fake_model_setup.raw_dir
    for source in ("--dataset", "--unlabeled"):
        assert extract.main(["--config", str(config), source]) == 0
    capsys.readouterr()
    old_sizes = os.path.getsize(root / "05" / "5.mp4"), os.path.getsize(team / "drive" / "clip.mp4")

    write_marker_video(root / "05" / "5.mp4", PTS_10FPS[:9])
    write_marker_video(team / "drive" / "clip.mp4", PTS_10FPS[:8])
    new_sizes = os.path.getsize(root / "05" / "5.mp4"), os.path.getsize(team / "drive" / "clip.mp4")
    assert old_sizes[0] != new_sizes[0] and old_sizes[1] != new_sizes[1]

    assert extract.main(["--config", str(config), "--dataset"]) == 0
    out = capsys.readouterr().out
    labelled = raw / "native" / "05" / "5.parquet"
    assert replace_line("05/5.mp4", labelled, root / "05" / "5.mp4", old_sizes[0], new_sizes[0]) in out.splitlines()
    assert out.count("skip 05/") == 2 and "extracted 1, up to date 2, failed 0" in out
    assert len(list(read_raw_frames(labelled))) == 9
    assert read_extract_metadata(labelled)["source_size_bytes"] == new_sizes[0]

    assert extract.main(["--config", str(config), "--unlabeled"]) == 0
    out = capsys.readouterr().out
    unlabeled = raw / "unlabeled_native" / "drive" / "clip.parquet"
    assert replace_line("drive/clip.mp4", unlabeled, team / "drive" / "clip.mp4", old_sizes[1], new_sizes[1]) in out.splitlines()
    assert "skip drive/keep.mp4" in out and "extracted 1, up to date 1, failed 0" in out
    assert len(list(read_raw_frames(unlabeled))) == 8


def test_cli_dataset_cannot_start(
    tmp_path: Path,
    fake_model_setup: SimpleNamespace,
    config_path: Path,
    pipeline_spec: PipelineSpec,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 기본 설정처럼 rldd_root가 null이면 시작하지 않고 고칠 곳을 알려 준다.
    null_root = write_dataset_config(tmp_path, config_path, fake_model_setup, None, name="null.yaml")
    assert extract.main(["--config", str(null_root), "--dataset"]) == 2
    assert "paths.rldd_root is null" in capsys.readouterr().err

    assert extract.main(["--config", str(null_root), "--video", "x.mp4", "--pilot", "2"]) == 2
    assert "--pilot needs --dataset" in capsys.readouterr().err

    incomplete = tmp_path / "incomplete"
    write_marker_video(incomplete / "01" / "0.mp4", PTS_10FPS[:3])
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, incomplete, name="incomplete.yaml")
    assert extract.main(["--config", str(config), "--dataset"]) == 2
    assert "lacks label(s) [5, 10]" in capsys.readouterr().err
    assert extract.main(["--config", str(config), "--dataset", "--pilot", "0"]) == 2
    assert "--pilot must be >= 1" in capsys.readouterr().err

    raw_tree = tmp_path / "raw_tree"
    for label in (0, 5, 10):
        write_raw_frames([hand_frame(pipeline_spec, 1)], raw_tree / "01" / f"{label}.parquet", pipeline_spec)
    config = write_dataset_config(tmp_path, config_path, fake_model_setup, raw_tree, name="raw_tree.yaml")
    assert extract.main(["--config", str(config), "--dataset"]) == 2
    assert "holds raw parquet files, not videos" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        extract.main(["--config", str(config), "--dataset", "--video", "x.mp4"])  # 입력은 하나만 고른다
    capsys.readouterr()
    assert not fake_model_setup.reports_dir.exists()  # 시작하지 못한 실행은 리포트를 쓰지 않는다


@pytest.mark.mediapipe
def test_real_rotation_check_and_dataset_extraction(
    tmp_path: Path,
    config_path: Path,
    feature_spec_path: Path,
    pipeline_spec: PipelineSpec,
    real_model: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """실제 랜드마커로 회전 점검과 `--dataset` 추출(작업자 프로세스 포함)을 돌린다. 얼굴이 없는 합성 영상만 쓴다."""
    root = tmp_path / "rldd"
    write_plain_video(root / "Fold1_part1" / "01" / "0.mp4", PTS_30FPS)
    write_plain_video(root / "Fold1_part1" / "01" / "5.MOV", PTS_30FPS, quarter_turns_cw=1)
    write_plain_video(root / "Fold1_part1" / "01" / "10_1.mp4", PTS_30FPS[:6])
    write_plain_video(root / "Fold1_part1" / "01" / "10_2.mp4", PTS_30FPS[:6])

    # 어느 방향에서도 얼굴이 없으면 회전 메타데이터(0)를 그대로 쓰고 가르지 못했다고 표시한다.
    check = extract.check_rotation(root / "Fold1_part1" / "01" / "5.MOV", real_model, pipeline_spec, 4, 100, 15.0)
    assert (check.rotation, check.metadata_rotation, check.n_frames) == (0, 0, 4)
    assert check.rates == {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    assert check.rolls == {0: None, 90: None, 180: None, 270: None}
    assert (check.ambiguous, check.reason) == (True, "no_face")

    setup = SimpleNamespace(spec_path=feature_spec_path, model_path=real_model)
    config = write_dataset_config(tmp_path, config_path, setup, root)
    assert extract.main(["--config", str(config), "--dataset", "--pilot", "1"]) == 0
    out = capsys.readouterr().out
    assert "done Fold1_part1/01/0.mp4" in out and "pilot: 1 videos, 12 frames," in out

    assert extract.main(["--config", str(config), "--dataset", "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "skip Fold1_part1/01/0.mp4" in out and "extracted 3, up to date 1, failed 0" in out
    assert "done Fold1_part1/01/10_2.mp4" in out and "auto check chose 0 (metadata 0; no_face;" in out
    native = tmp_path / "raw" / "native" / "Fold1_part1" / "01"
    assert sorted(path.name for path in native.iterdir()) == ["0.parquet", "10_1.parquet", "10_2.parquet", "5.parquet"]
    assert [(row.w, row.h) for row in read_raw_frames(native / "5.parquet")] == [(H, W)] * 12
    # 설정의 표본 간격(300 ms)보다 짧은 영상(167 ms)에서는 첫 프레임 한 장만 본다.
    assert read_extract_metadata(native / "10_1.parquet")["rotation_check"]["n_frames"] == 1

    # 작업자 프로세스에서 추출한 결과로 쓴 리포트: 얼굴이 없어 네 영상 모두 가르지 못함으로 표시된다.
    report, summary = read_report(tmp_path / "reports", "dataset_native")
    assert [row["status"] for row in report] == ["up_to_date", "done", "done", "done"]
    assert [row["rotation_reason"] for row in report] == ["no_face"] * 4
    assert [row["warnings"] for row in report] == ["rotation_ambiguous;low_face_rate"] * 4
    assert [row["t_ms_ok"] for row in report] == ["true"] * 4
    assert report[1]["model_sha256"] == pipeline_spec.mediapipe.model_sha256
    lumas = [luma_mean(rgb) for _, rgb in iter_video_frames(root / "Fold1_part1" / "01" / "5.MOV")]
    assert float(report[1]["brightness_mean"]) == pytest.approx(float(np.mean(lumas)), rel=1e-5)
    assert "| 실패 (`failed`) | 0 |" in summary and "### 어느 각도에서도 얼굴을 못 찾음 (`no_face`): 회전 메타데이터를 유지했다\n\n4개." in summary


# ---------------------------------------------------------------------------
# 실제 얼굴 (환경 변수로 준 사진이 있을 때만. 저장소에는 얼굴 사진이 없다)
# ---------------------------------------------------------------------------

FACE_IMAGE_ENV = "DROWSY_TEST_FACE_IMAGE"


def write_image_video(
    path: Path, frames: Sequence[np.ndarray], step_ms: int, quarter_turns_cw: int, tag: int | None
) -> None:
    """세운 프레임들을 시계 방향으로 quarter_turns_cw번 돌려 저장한다. tag는 회전 메타데이터(없으면 달지 않는다)."""
    time_base = Fraction(1, 1000)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30)
    first = np.rot90(frames[0], -quarter_turns_cw)
    stream.width, stream.height, stream.pix_fmt = first.shape[1], first.shape[0], "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    stream.options = {"bf": "0", "crf": "16"}
    if tag is not None:
        stream.set_display_rotation(tag)
    for index, image in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(np.rot90(image, -quarter_turns_cw)), format="rgb24")
        frame.pts = index * step_ms
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture(scope="module")
def face_frames() -> list[np.ndarray]:
    """환경 변수가 가리키는 정면 얼굴 사진을 640x480 화면 가운데에 놓은 프레임 8장(조금씩 좌우로 움직인다)."""
    image_path = os.environ.get(FACE_IMAGE_ENV)
    if not image_path:
        pytest.skip(f"{FACE_IMAGE_ENV} is not set (path of an upright frontal face image you may use)")
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        pytest.fail(f"{FACE_IMAGE_ENV}={image_path} cannot be read as an image")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = min(640 / rgb.shape[1], 480 / rgb.shape[0])
    size = (int(rgb.shape[1] * scale) // 2 * 2, int(rgb.shape[0] * scale) // 2 * 2)
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    frames = []
    for index in range(8):
        canvas = np.full((480, 640, 3), 70, dtype=np.uint8)
        x0 = min(max((640 - small.shape[1]) // 2 + (index % 3) - 1, 0), 640 - small.shape[1])
        y0 = (480 - small.shape[0]) // 2
        canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
        frames.append(canvas)
    return frames


@pytest.mark.mediapipe
def test_real_landmarker_on_a_face_image(
    tmp_path: Path, pipeline_spec: PipelineSpec, real_model: Path, face_frames: list[np.ndarray]
) -> None:
    """실제 랜드마커와 실제 얼굴: 추출 결과가 MediaPipe를 직접 부른 값과 같은지, 그리고 회전 점검의 규칙.

    `DROWSY_TEST_FACE_IMAGE=/path/to/face.png`를 주고 돌린다. 사진은 똑바로 선 정면 얼굴 하나여야 한다.
    """
    import mediapipe as mp

    upright = tmp_path / "upright.mp4"
    write_image_video(upright, face_frames, 100, 0, None)
    out = tmp_path / "upright.parquet"
    stats = extract_video(upright, real_model, out, spec=pipeline_spec)
    assert (stats.n_processed, stats.n_face, stats.width, stats.height) == (8, 8, 640, 480)  # 얼굴 검출률 1.0
    rows = list(read_raw_frames(out))
    metadata = read_extract_metadata(out)
    assert metadata["blendshape_names"] == list(pipeline_spec.blendshapes.order)
    assert len(metadata["blendshape_names"]) == 52

    # MediaPipe를 직접 불러 얻은 결과와 값이 정확히 같다(랜드마크 44개, 블렌드셰이프 52개와 그 순서, 행렬 16개 값).
    required = pipeline_spec.landmarks.required_indices()
    decoded = list(iter_video_frames(upright))
    landmarker = extract.make_landmarker(real_model, pipeline_spec)
    try:
        for (t_ms, rgb), row in zip(decoded, rows, strict=True):
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
            assert row.face and row.t_ms == t_ms and len(result.face_landmarks) == 1
            landmarks = result.face_landmarks[0]
            assert row.lm == {index: (float(landmarks[index].x), float(landmarks[index].y)) for index in required}
            categories = result.face_blendshapes[0]
            assert [category.category_name for category in categories] == list(pipeline_spec.blendshapes.order)
            assert list(row.bs.items()) == [(category.category_name, float(category.score)) for category in categories]
            matrix = np.asarray(result.facial_transformation_matrixes[0])
            assert all(row.m[4 * j + i] == float(matrix[i, j]) for i in range(4) for j in range(4))
            # 좌우를 뒤집지 않았다: 피험자의 오른쪽 눈 바깥 꼬리(33)가 영상의 왼쪽에 있다.
            assert row.lm[33][0] < row.lm[263][0]
    finally:
        landmarker.close()
    lumas = [luma_mean(rgb) for _, rgb in decoded]
    assert metadata["brightness_mean"] == pytest.approx(float(np.mean(lumas)), rel=1e-9)

    # 회전 점검: 시계 방향으로 k번 돌려 저장한 영상의 맞는 각도는 90k다.
    for quarter_turns_cw, tag in [(0, None), (1, None), (2, None), (3, None), (0, 90), (1, 270), (1, 90), (2, 180)]:
        truth = 90 * quarter_turns_cw
        opposite = (truth + 180) % 360
        clip = tmp_path / f"stored{quarter_turns_cw}_tag{tag}.mp4"
        write_image_video(clip, face_frames, 100, quarter_turns_cw, tag)
        check = extract.check_rotation(clip, real_model, pipeline_spec, 4, 200, roll_margin_deg=15.0)
        note = f"stored {truth} CW, tag {tag}: {check}"
        assert check.metadata_rotation == (tag or 0) and check.n_frames == 4, note
        assert check.rates[truth] == 1.0 and abs(check.rolls[truth]) < 45.0, note
        # 옆으로 누운 방향(맞는 각도 ±90)은 고르지 않는다. 맞는 각도이거나, 가르지 못해 표시한 정반대 각도다.
        assert check.rotation in (truth, opposite), note
        if check.rotation != truth:
            assert check.ambiguous and check.reason == "ambiguous_kept_metadata", note
            assert check.metadata_rotation == opposite, note
        if not check.ambiguous or check.metadata_rotation == truth:
            assert check.rotation == truth, note
        if check.rolls[opposite] is None:  # 거꾸로 선 얼굴이 검출되지 않으면 점검이 끝까지 가른다
            assert not check.ambiguous, note


@pytest.mark.mediapipe
def test_real_cli_turns_a_sideways_face_clip_upright(
    tmp_path: Path,
    config_path: Path,
    feature_spec_path: Path,
    real_model: Path,
    face_frames: list[np.ndarray],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """실제 랜드마커와 실제 얼굴: 회전 메타데이터 없이 누워 저장된 영상을 `--input-dir`과 `--video`가 세워서 추출한다.

    `DROWSY_TEST_FACE_IMAGE=/path/to/face.png`를 주고 돌린다. 전에는 두 명령이 회전 메타데이터(없으면 0)만 믿어서 얼굴이
    옆으로 누운 채(얼굴 검출 비율 1.0, roll 중앙값 약 ±90도) 추출됐다.
    """
    from drowsy import signals

    videos = tmp_path / "videos"
    videos.mkdir()
    write_image_video(videos / "lying.mp4", face_frames, 100, 1, None)  # 시계 방향 90도로 누움: 맞는 각도는 90
    config = edit_config(
        write_cli_config(tmp_path, config_path, feature_spec_path, real_model),
        "real.yaml",
        extraction__rotation_check_frames=4,
        extraction__rotation_check_interval_ms=200,
    )
    out = tmp_path / "raw" / "native" / "lying.parquet"
    for source in (["--input-dir", str(videos)], ["--video", str(videos / "lying.mp4")]):
        out.unlink(missing_ok=True)
        assert extract.main(["--config", str(config), *source]) == 0
        printed = capsys.readouterr().out
        assert "done lying.mp4" in printed and "rotation: auto check chose" in printed, source
        metadata = read_extract_metadata(out)
        recorded = metadata["rotation_check"]
        rows = list(read_raw_frames(out))
        rolls = [signals.euler_from_colmajor(row.m)[2] for row in rows if row.face]
        note = f"{source[0]}: rotation {metadata['rotation']}, check {recorded}"
        assert len(rolls) == len(rows) == 8, note  # 얼굴 검출 비율 1.0
        # 옆으로 누운 채로는 추출하지 않는다: 가로로 긴 프레임이고 roll이 0 근처다.
        assert {(row.w, row.h) for row in rows} == {(640, 480)}, note
        assert abs(extract.median_angle(rolls)) < 45.0, note
        # 맞는 각도(90)이거나, 거꾸로 선 얼굴도 검출되는 얼굴이라 가르지 못해 표시한 정반대 각도(270)다.
        assert metadata["rotation"] in (90, 270) and metadata["rotation_override"] == metadata["rotation"], note
        if metadata["rotation"] != 90 or recorded["rolls"]["270"] is not None:
            assert recorded["ambiguous"] is True and "WARNING rotation ambiguous" in printed, note
        else:
            assert recorded["ambiguous"] is False and recorded["reason"] == "smallest_roll", note

    # 사람이 확인한 각도를 주면 점검 없이 그 각도로 추출한다.
    assert extract.main(["--config", str(config), "--video", str(videos / "lying.mp4"), "--rotation", "90"]) == 0
    assert "rotation: manual override (--rotation)" in capsys.readouterr().out
    metadata = read_extract_metadata(out)
    assert metadata["rotation"] == 90 and "rotation_check" not in metadata
    rolls = [signals.euler_from_colmajor(row.m)[2] for row in read_raw_frames(out) if row.face]
    assert len(rolls) == 8 and abs(extract.median_angle(rolls)) < 45.0


def test_rotation_check_settings_in_the_default_config(config_path: Path, tmp_path: Path) -> None:
    from drowsy.config import load_config
    from drowsy.spec import SpecError

    extraction = load_config(config_path).extraction
    assert (extraction.rotation_check_frames, extraction.rotation_check_interval_ms) == (12, 2000)
    assert extraction.rotation_roll_margin_deg == 15.0
    assert extraction.unlabeled_rotation_overrides == {}
    assert extraction.pilot_videos == 6
    text = config_path.read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("rotation_check_frames: 12", "rotation_check_frames: 0"), encoding="utf-8")
    with pytest.raises(SpecError, match="rotation_check_frames"):
        load_config(bad)
    bad.write_text(text.replace("rotation_roll_margin_deg: 15.0", "rotation_roll_margin_deg: 90.0"), encoding="utf-8")
    with pytest.raises(SpecError, match="rotation_roll_margin_deg"):
        load_config(bad)
    bad.write_text(
        text.replace("unlabeled_rotation_overrides: {}", "unlabeled_rotation_overrides: {a.mp4: 45}"), encoding="utf-8"
    )
    with pytest.raises(SpecError, match="unlabeled_rotation_overrides"):
        load_config(bad)
