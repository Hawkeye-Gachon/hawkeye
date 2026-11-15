"""video_io: 타임스탬프, 회전, 표시 행렬의 뒤집기, 프레임 건너뛰기, probe (PLAN 2.3절).

영상은 테스트 안에서 PyAV로 합성한다(얼굴 없는 표식 그림). 기대 시각은 써 넣은 표시 시각에서 손으로 계산한 값이다.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from drowsy.video_io import (
    DecodeCounts,
    VideoError,
    index_to_ms,
    iter_video_frames,
    normalize_rotation,
    probe_video,
    pts_to_ms,
    rotate_upright,
)

W, H = 96, 64
MS = Fraction(1, 1000)


def marker_frame(index: int = 0) -> np.ndarray:
    """세운 상태의 표식 그림 [H, W, 3]: 왼쪽 위 흰 사각형, 왼쪽 아래 빨간 사각형, 오른쪽 초록 띠, 움직이는 회색 칸."""
    image = np.full((H, W, 3), 30, dtype=np.uint8)
    image[0:16, 0:16] = 255
    image[H - 16 : H, 0:16] = (255, 0, 0)
    image[:, W - 8 :] = (0, 255, 0)
    left = 24 + (index % 6) * 8
    image[24:40, left : left + 8] = 200
    return image


def write_video(
    path: Path,
    pts: Sequence[int],
    time_base: Fraction = MS,
    *,
    b_frames: int = 0,
    display_rotation: int | None = None,
    stored_quarter_turns_cw: int = 0,
    container_format: str | None = None,
) -> None:
    """표식 그림을 주어진 pts(time_base 단위)로 H.264 영상에 쓴다.

    stored_quarter_turns_cw: 세운 그림을 시계 방향으로 이만큼(90도 단위) 돌린 채 저장한다(세로로 찍은 휴대전화 영상 흉내).
    display_rotation: 컨테이너에 적는 회전 메타데이터(표시하려면 반시계로 돌려야 하는 각도).
    """
    container = av.open(str(path), "w", format=container_format)
    stream = container.add_stream("libx264", rate=30)
    first = np.rot90(marker_frame(0), -stored_quarter_turns_cw)
    stream.width, stream.height, stream.pix_fmt = first.shape[1], first.shape[0], "yuv420p"
    stream.codec_context.time_base = time_base
    stream.time_base = time_base
    # b_strategy 0: 적응 판단을 끄고 B 프레임을 항상 넣는다(그림이 단순해도 디코딩 순서가 표시 순서와 달라지게)
    stream.options = {"bf": str(b_frames), "b_strategy": "0", "crf": "12"}
    if display_rotation is not None:
        stream.set_display_rotation(display_rotation)
    for index, value in enumerate(pts):
        image = np.ascontiguousarray(np.rot90(marker_frame(index), -stored_quarter_turns_cw))
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = value
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def read_all(path: Path, rotation: int | None = None) -> tuple[list[int], list[np.ndarray], DecodeCounts]:
    counts = DecodeCounts()
    frames = list(iter_video_frames(path, rotation=rotation, counts=counts))
    return [t for t, _ in frames], [rgb for _, rgb in frames], counts


def assert_positive_increasing(times: Sequence[int]) -> None:
    assert all(isinstance(t, int) for t in times)
    assert times[0] >= 1
    assert all(later > earlier for earlier, later in zip(times, times[1:]))


def assert_upright_marker(rgb: np.ndarray) -> None:
    """세운 표식 그림인지: 모양, 흰 사각형(왼쪽 위), 빨간 사각형(왼쪽 아래), 초록 띠(오른쪽)."""
    assert rgb.shape == (H, W, 3)
    assert rgb.dtype == np.uint8
    assert rgb.flags.c_contiguous
    assert rgb[3:13, 3:13].mean() > 200
    red = rgb[H - 13 : H - 3, 3:13].reshape(-1, 3).mean(axis=0)
    assert red[0] > 170 and red[1] < 80 and red[2] < 80  # 채널 순서가 RGB인지도 여기서 드러난다
    green = rgb[20:44, W - 6 : W - 2].reshape(-1, 3).mean(axis=0)
    assert green[1] > 170 and green[0] < 80 and green[2] < 80


# ---------------------------------------------------------------------------
# 타임스탬프 식
# ---------------------------------------------------------------------------


def test_pts_to_ms_starts_at_one_and_uses_time_base() -> None:
    assert pts_to_ms(0, 0, MS) == 1
    assert pts_to_ms(40, 0, MS) == 41
    assert pts_to_ms(8528, 8000, Fraction(1, 16000)) == 34  # 528 / 16 = 33 ms
    assert pts_to_ms(123_456_789, 123_456_789, Fraction(1, 90000)) == 1  # 시작 오프셋은 첫 pts로 없어진다


def test_pts_to_ms_rounds_half_up_not_half_even() -> None:
    half_ms = Fraction(1, 2000)
    # 0.5 ms -> 1, 1.5 ms -> 2, 2.5 ms -> 3. Python의 round()는 0, 2, 2를 낸다.
    assert [pts_to_ms(p, 0, half_ms) for p in (1, 3, 5)] == [2, 3, 4]
    assert [1 + round(p / 2) for p in (1, 3, 5)] == [1, 3, 3]
    # 0.4999… ms는 내려가고 0.5 ms는 올라간다 (1/90000 단위: 44.99 틱이 0.5 ms 미만, 45 틱이 정확히 0.5 ms)
    tick = Fraction(1, 90000)
    assert pts_to_ms(44, 0, tick) == 1
    assert pts_to_ms(45, 0, tick) == 2


def test_pts_to_ms_ntsc_time_base_is_exact() -> None:
    ntsc = Fraction(1001, 30000)  # pts가 프레임 번호인 29.97 fps
    # 33.3667, 66.7333, 100.1, 133.4667, 166.8333 ms
    assert [pts_to_ms(i, 0, ntsc) for i in range(6)] == [1, 34, 68, 101, 134, 168]
    # 부동소수로는 표현할 수 없는 큰 pts에서도 정확하다: 10^15 틱 · 1001/30000 · 1000 = 33366666666666666.67 ms
    assert pts_to_ms(10**15, 0, ntsc) == 1 + 33_366_666_666_666_667


def test_pts_to_ms_before_first_frame_is_not_positive() -> None:
    assert pts_to_ms(-1, 0, MS) == 0
    assert pts_to_ms(90, 100, MS) == -9


def test_index_to_ms() -> None:
    assert [index_to_ms(i, Fraction(25)) for i in range(4)] == [1, 41, 81, 121]
    assert [index_to_ms(i, Fraction(30000, 1001)) for i in range(4)] == [1, 34, 68, 101]
    assert index_to_ms(1, Fraction(2000)) == 2  # 0.5 ms는 올림


# ---------------------------------------------------------------------------
# 회전 도우미
# ---------------------------------------------------------------------------


def test_normalize_rotation() -> None:
    assert [normalize_rotation(r) for r in (0, 90, 180, 270, 360, -90, -180, 450)] == [0, 90, 180, 270, 0, 270, 180, 90]
    with pytest.raises(VideoError, match="multiple of 90"):
        normalize_rotation(45)


def test_rotate_upright_is_counter_clockwise() -> None:
    # 값이 위치를 나타내는 2x3 그림:  1 2 3
    #                                4 5 6
    gray = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    rgb = np.stack([gray, gray + 10, gray + 20], axis=-1)

    def channel0(rotation: int) -> list[list[int]]:
        out = rotate_upright(rgb, rotation)
        assert out.flags.c_contiguous
        assert np.array_equal(out[..., 1], out[..., 0] + 10)  # 채널은 섞이지 않는다
        return out[..., 0].tolist()

    assert channel0(0) == [[1, 2, 3], [4, 5, 6]]
    assert channel0(90) == [[3, 6], [2, 5], [1, 4]]  # 반시계 90도: 오른쪽 변이 위로
    assert channel0(180) == [[6, 5, 4], [3, 2, 1]]
    assert channel0(270) == [[4, 1], [5, 2], [6, 3]]  # 시계 90도: 왼쪽 변이 위로
    assert channel0(-90) == channel0(270)
    assert channel0(360) == channel0(0)
    with pytest.raises(VideoError):
        rotate_upright(rgb, 30)


# ---------------------------------------------------------------------------
# 합성 영상의 타임스탬프
# ---------------------------------------------------------------------------

CONSTANT_MS = [40 * i for i in range(10)]
VARIABLE_MS = [0, 17, 50, 83, 150, 283, 300, 366, 400, 433]
OFFSET_MS = [500, 533, 567, 600, 633, 667]


def test_constant_interval_video(tmp_path: Path) -> None:
    path = tmp_path / "constant.mp4"
    write_video(path, CONSTANT_MS)
    times, frames, counts = read_all(path)
    assert times == [1 + t for t in CONSTANT_MS]
    assert_positive_increasing(times)
    assert counts == DecodeCounts(decoded=10, skipped=0, rotation=0, index_clock=False)
    for rgb in frames:
        assert_upright_marker(rgb)

    info = probe_video(path)
    assert (info.n_frames, info.n_skipped, info.duration_ms) == (10, 0, 360)
    assert info.mean_fps == pytest.approx(25.0)
    assert (info.min_interval_ms, info.max_interval_ms, info.variable_intervals) == (40, 40, False)
    assert (info.rotation, info.width, info.height) == (0, W, H)


def test_variable_interval_video(tmp_path: Path) -> None:
    path = tmp_path / "variable.mp4"
    write_video(path, VARIABLE_MS)
    times, _, counts = read_all(path)
    assert times == [1, 18, 51, 84, 151, 284, 301, 367, 401, 434]
    assert_positive_increasing(times)
    assert counts.skipped == 0

    info = probe_video(path)
    assert (info.n_frames, info.duration_ms) == (10, 433)
    assert info.mean_fps == pytest.approx(9 * 1000.0 / 433)
    assert (info.min_interval_ms, info.max_interval_ms, info.variable_intervals) == (17, 133, True)


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".mkv"])
def test_nonzero_start_offset_is_removed(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"offset{suffix}"
    write_video(path, OFFSET_MS)
    times, _, _ = read_all(path)
    assert times == [1, 34, 68, 101, 134, 168]  # 첫 프레임이 500 ms여도 1에서 시작한다
    assert_positive_increasing(times)

    info = probe_video(path)
    assert (info.n_frames, info.duration_ms) == (6, 167)
    assert (info.min_interval_ms, info.max_interval_ms, info.variable_intervals) == (33, 34, False)  # 1 ms 흔들림은 일정으로 본다


def test_b_frames_come_out_in_presentation_order(tmp_path: Path) -> None:
    path = tmp_path / "bframes.mp4"
    write_video(path, VARIABLE_MS, b_frames=2)
    with av.open(str(path)) as container:
        packets = [p for p in container.demux(container.streams.video[0]) if p.size > 0]
    packet_pts = [p.pts for p in packets]
    assert packet_pts != sorted(packet_pts)  # 디코딩 순서와 표시 순서가 실제로 다른 파일이다

    times, frames, counts = read_all(path)
    assert times == [1 + t for t in VARIABLE_MS]
    assert counts.skipped == 0
    # 그림도 표시 순서대로 나온다: 움직이는 회색 칸의 위치가 프레임 번호와 맞는다
    for index, rgb in enumerate(frames):
        left = 24 + (index % 6) * 8
        assert rgb[28:36, left + 2 : left + 6].mean() > 150

    info = probe_video(path)  # 패킷은 디코딩 순서로 오지만 결과는 같아야 한다
    assert (info.n_frames, info.n_skipped, info.duration_ms) == (10, 0, 433)
    assert (info.min_interval_ms, info.max_interval_ms) == (17, 133)


def test_ntsc_time_base_video(tmp_path: Path) -> None:
    path = tmp_path / "ntsc.mp4"
    write_video(path, [1001 * i for i in range(8)], Fraction(1, 30000))
    times, _, _ = read_all(path)
    # 1001/30000 s = 33.3667 ms 간격: 0, 33.37, 66.73, 100.1, 133.47, 166.83, 200.2, 233.57
    assert times == [1, 34, 68, 101, 134, 168, 201, 235]
    info = probe_video(path)
    assert info.variable_intervals is False
    assert info.mean_fps == pytest.approx(7 * 1000.0 / 234)


def test_half_millisecond_pts_round_up(tmp_path: Path) -> None:
    path = tmp_path / "half.mp4"
    # 0, 0.5, 1.5, 50, 100.5 ms. 짝수 반올림이면 0, 0, 2, 50, 100이 되어 둘째 프레임이 중복으로 버려진다.
    write_video(path, [0, 1, 3, 100, 201], Fraction(1, 2000))
    times, _, counts = read_all(path)
    assert times == [1, 2, 3, 51, 102]
    assert counts.skipped == 0


def test_non_increasing_timestamps_are_skipped_and_counted(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.mp4"
    # 1/90000 s 단위: 0, 0.33, 0.8, 33.33, 33.67, 66.67, 67.17, 100 ms
    # 반올림: 0, 0(중복), 1, 33, 34, 67, 67(중복), 100
    write_video(path, [0, 30, 72, 3000, 3030, 6000, 6045, 9000], Fraction(1, 90000))
    times, frames, counts = read_all(path)
    assert times == [1, 2, 34, 35, 68, 101]
    assert_positive_increasing(times)
    assert (counts.decoded, counts.skipped) == (8, 2)
    assert len(frames) == 6

    info = probe_video(path)
    assert (info.n_frames, info.n_skipped, info.duration_ms) == (6, 2, 100)


def test_missing_pts_falls_back_to_frame_index_clock(tmp_path: Path) -> None:
    path = tmp_path / "raw.h264"  # 컨테이너 없는 원시 스트림: 프레임에 pts가 없다
    write_video(path, [0, 1, 2, 3, 4], Fraction(1, 30), container_format="h264")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        assert all(frame.pts is None for frame in container.decode(stream))
    assert rate is not None and rate > 0

    times, frames, counts = read_all(path)
    expected = [1 + math.floor(Fraction(1000 * i) / rate + Fraction(1, 2)) for i in range(5)]
    assert times == expected
    assert_positive_increasing(times)
    assert counts == DecodeCounts(decoded=5, skipped=0, rotation=0, index_clock=True)
    assert_upright_marker(frames[0])

    info = probe_video(path)
    assert (info.n_frames, info.n_skipped, info.duration_ms) == (5, 0, expected[-1] - 1)


def test_times_are_independent_of_decode_restart(tmp_path: Path) -> None:
    """같은 파일을 두 번 읽으면 같은 시각과 같은 그림이 나온다(추출을 다시 해도 결과가 같아야 한다)."""
    path = tmp_path / "again.mp4"
    write_video(path, VARIABLE_MS, b_frames=2)
    first_times, first_frames, _ = read_all(path)
    second_times, second_frames, _ = read_all(path)
    assert first_times == second_times
    assert all(np.array_equal(a, b) for a, b in zip(first_frames, second_frames))


# ---------------------------------------------------------------------------
# 회전 메타데이터
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", [90, 180, 270, -90])
def test_rotation_metadata_is_applied(tmp_path: Path, rotation: int) -> None:
    """세운 그림을 시계 방향으로 rotation도 돌려 저장하고 메타데이터에 rotation을 적으면 다시 세워져 나온다.

    같은 파일을 ffmpeg 명령행(자동 회전)으로 디코딩한 그림과 화소 단위로 같다는 것을 스크래치에서 확인했다.
    """
    path = tmp_path / f"rot{rotation}.mp4"
    quarter_turns = (rotation // 90) % 4
    write_video(path, [0, 40, 80], display_rotation=rotation, stored_quarter_turns_cw=quarter_turns)

    with av.open(str(path)) as container:
        stored = next(container.decode(video=0))
        assert stored.rotation % 360 == rotation % 360  # PyAV는 메타데이터를 알려 주기만 하고 적용하지 않는다
        expected_stored_shape = (W, H) if quarter_turns % 2 == 1 else (H, W)
        assert (stored.height, stored.width) == expected_stored_shape

    times, frames, counts = read_all(path)
    assert times == [1, 41, 81]
    assert counts.rotation == rotation % 360
    for rgb in frames:
        assert_upright_marker(rgb)

    info = probe_video(path)
    assert (info.rotation, info.width, info.height) == (rotation % 360, W, H)


def test_rotation_override_replaces_metadata(tmp_path: Path) -> None:
    # 메타데이터 없이 시계 방향 90도로 누운 영상: 수동 보정 90(반시계)으로 세운다
    lying = tmp_path / "lying.mp4"
    write_video(lying, [0, 40], stored_quarter_turns_cw=1)
    _, frames, counts = read_all(lying)
    assert frames[0].shape == (W, H, 3) and counts.rotation == 0
    _, frames, counts = read_all(lying, rotation=90)
    assert counts.rotation == 90
    assert_upright_marker(frames[0])
    _, frames, _ = read_all(lying, rotation=-270)
    assert_upright_marker(frames[0])

    # 메타데이터가 틀린 영상: 보정 0이면 저장된 그대로 나온다
    wrong = tmp_path / "wrong.mp4"
    write_video(wrong, [0, 40], display_rotation=90)
    _, frames, counts = read_all(wrong)
    assert frames[0].shape == (W, H, 3) and counts.rotation == 90
    _, frames, counts = read_all(wrong, rotation=0)
    assert counts.rotation == 0
    assert_upright_marker(frames[0])

    with pytest.raises(VideoError, match="multiple of 90"):
        list(iter_video_frames(lying, rotation=45))


# ---------------------------------------------------------------------------
# 표시 행렬의 뒤집기(거울상)
# ---------------------------------------------------------------------------

IDENTITY_MATRIX = struct.pack(">9i", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
# 표시 행렬의 왼쪽 위 2x2 블록 (a, b, c, d). 행렬식 a·d − b·c가 −1이면 뒤집기가 들어 있다.
FLIP_MATRICES = {
    "hflip": (-1, 0, 0, 1),  # 좌우 뒤집기
    "vflip": (1, 0, 0, -1),  # 상하 뒤집기
    "transpose": (0, 1, 1, 0),  # 대각선 뒤집기
    "rot90_hflip": (0, -1, -1, 0),  # ffmpeg의 -display_rotation 90 -display_hflip이 쓰는 값
}
ROTATION_MATRICES = {"rot180": (-1, 0, 0, -1), "quarter_a": (0, 1, -1, 0), "quarter_b": (0, -1, 1, 0)}  # 행렬식 +1


def tag_display_matrix(path: Path, a: int, b: int, c: int, d: int) -> None:
    """mp4/mov의 트랙 머리(tkhd)에 있는 표시 행렬의 2x2 블록을 [[a, b], [c, d]](값은 −1, 0, 1)로 바꿔 쓴다."""
    data = bytearray(path.read_bytes())
    tkhd = data.find(b"tkhd")
    at = data.find(IDENTITY_MATRIX, tkhd)
    assert tkhd > 0 and 0 < at - tkhd < 80, "the file does not hold an identity display matrix"
    data[at : at + 36] = struct.pack(">9i", a << 16, b << 16, 0, c << 16, d << 16, 0, 0, 0, 0x40000000)
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("suffix", [".mp4", ".mov"])
@pytest.mark.parametrize("name", sorted(FLIP_MATRICES))
def test_display_matrix_with_a_flip_needs_an_explicit_rotation(tmp_path: Path, name: str, suffix: str) -> None:
    """뒤집기가 든 표시 행렬: libav가 알려 주는 회전을 믿을 수 없으므로 회전을 주지 않으면 거부한다.

    전에는 좌우 뒤집기만 있는(똑바로 선) 영상이 회전 −180으로 읽혀 거꾸로 세워졌다.
    """
    path = tmp_path / f"{name}{suffix}"
    write_video(path, [0, 40, 80])  # 똑바로 선 채로 저장한다
    a, b, c, d = FLIP_MATRICES[name]
    assert a * d - b * c == -1
    tag_display_matrix(path, a, b, c, d)

    with pytest.raises(VideoError, match="display matrix contains a flip") as excinfo:
        list(iter_video_frames(path))
    assert str(path) in str(excinfo.value) and "Fix:" in str(excinfo.value)
    # 고치는 방법을 명령의 옵션 이름으로 알려 준다(데모와 `extract --video`는 --rotation, 나머지 추출은 설정의 보정표).
    assert "--rotation" in str(excinfo.value) and "manual rotation table" in str(excinfo.value)

    # 회전을 주면 저장된 그림을 그 각도로만 돌린다(거울상을 되돌리지 않는다)
    times, frames, counts = read_all(path, rotation=0)
    assert times == [1, 41, 81]
    assert (counts.rotation, counts.display_flip, counts.decoded) == (0, True, 3)
    for rgb in frames:
        assert_upright_marker(rgb)
    _, turned, counts = read_all(path, rotation=180)
    assert (counts.rotation, counts.display_flip) == (180, True)
    assert np.array_equal(turned[0], np.rot90(frames[0], 2))

    info = probe_video(path)  # probe는 오류를 내지 않고 알려 주기만 한다
    assert info.display_flip is True
    assert (info.n_frames, info.duration_ms, info.mean_fps) == (3, 80, pytest.approx(25.0))


@pytest.mark.parametrize("name", sorted(ROTATION_MATRICES))
def test_display_matrix_with_a_pure_rotation_is_not_a_flip(tmp_path: Path, name: str) -> None:
    path = tmp_path / f"{name}.mp4"
    write_video(path, [0, 40, 80])
    a, b, c, d = ROTATION_MATRICES[name]
    assert a * d - b * c == 1
    tag_display_matrix(path, a, b, c, d)
    times, frames, counts = read_all(path)
    assert times == [1, 41, 81] and counts.display_flip is False
    if name == "rot180":
        assert counts.rotation == 180 and frames[0].shape == (H, W, 3)
    else:  # 90도 회전: 어느 쪽이든 가로와 세로가 바뀐다
        assert counts.rotation in (90, 270) and frames[0].shape == (W, H, 3)
    assert probe_video(path).display_flip is False


def test_files_without_a_display_matrix_have_no_flip(tmp_path: Path) -> None:
    plain = tmp_path / "plain.mp4"
    write_video(plain, [0, 40])
    tagged = tmp_path / "tagged.mp4"
    write_video(tagged, [0, 40], display_rotation=90, stored_quarter_turns_cw=1)
    raw = tmp_path / "raw.h264"  # 컨테이너가 없어 표시 행렬도 없다
    write_video(raw, [0, 1], Fraction(1, 30), container_format="h264")
    for path in (plain, tagged, raw):
        _, _, counts = read_all(path)
        assert counts.display_flip is False
        assert probe_video(path).display_flip is False


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="the ffmpeg command line tool is not installed")
@pytest.mark.parametrize(
    ("flags", "flip"),
    [
        (["-display_hflip"], True),
        (["-display_vflip"], True),
        (["-display_rotation", "90", "-display_hflip"], True),
        (["-display_rotation", "180"], False),
    ],
)
def test_flip_tags_written_by_the_ffmpeg_command(tmp_path: Path, flags: list[str], flip: bool) -> None:
    """ffmpeg 명령이 실제로 써 넣은 표시 행렬로도 같은 판정이 나온다(손으로 고친 tkhd만 믿지 않는다)."""
    source = tmp_path / "plain.mp4"
    write_video(source, [0, 40, 80])
    tagged = tmp_path / "tagged.mp4"
    command = ["ffmpeg", "-v", "error", "-y", *flags, "-i", str(source), "-c", "copy", str(tagged)]
    subprocess.run(command, check=True, capture_output=True)
    assert probe_video(tagged).display_flip is flip
    if flip:
        with pytest.raises(VideoError, match="display matrix contains a flip"):
            list(iter_video_frames(tagged))
        _, frames, counts = read_all(tagged, rotation=0)
        assert counts.display_flip is True
        assert_upright_marker(frames[0])  # 저장된 그림 그대로다
    else:
        _, frames, counts = read_all(tagged)
        assert (counts.rotation, counts.display_flip) == (180, False)
        assert frames[0].shape == (H, W, 3)


# ---------------------------------------------------------------------------
# 오류
# ---------------------------------------------------------------------------


def test_file_without_video_stream(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    container = av.open(str(path), "w")
    stream = container.add_stream("pcm_s16le", rate=8000)
    frame = av.AudioFrame.from_ndarray(np.zeros((1, 800), dtype=np.int16), format="s16", layout="mono")
    frame.sample_rate = 8000
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    with pytest.raises(VideoError, match="no video stream"):
        list(iter_video_frames(path))
    with pytest.raises(VideoError, match="no video stream"):
        probe_video(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        list(iter_video_frames(tmp_path / "missing.mp4"))
    with pytest.raises(OSError):
        probe_video(tmp_path / "missing.mp4")
