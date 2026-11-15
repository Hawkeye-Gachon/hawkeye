"""PyAV 디코딩, 타임스탬프, 회전 보정 (PLAN 2.3절). 이식 대상이 아니다.

- 타임스탬프: `t_ms = 1 + floor((pts − 첫 프레임 pts)·time_base·1000 + 0.5)`. 정수 연산으로만 구한다.
  1을 더하는 이유: MediaPipe의 평활 필터는 첫 타임스탬프가 정확히 0이면 두 번째 프레임을 기본 30 Hz로
  처리하고 음수면 평활을 끈다. 앱은 0을 넘기지 않으므로 오프라인 추출도 양수에서 시작한다.
- `t_ms`가 이전 값보다 크지 않은 프레임은 버리고 센다(`detect_for_video`는 엄격히 증가하는 값만 받는다).
- 회전: PyAV는 회전 메타데이터를 적용하지 않는다. `frame.rotation`(표시하려면 반시계 방향으로 돌려야 하는
  각도)만큼 numpy로 돌려 세운다. OpenCV의 자동 회전은 버전마다 기본값이 달라서 쓰지 않는다.
- 표시 행렬에 뒤집기(거울상)가 들어 있으면 `frame.rotation`을 믿을 수 없다(좌우 뒤집기만 있는 파일이 −180으로
  읽힌다). 그런 파일은 회전을 직접 주지 않으면 오류로 알린다(`display_matrix_has_flip`). 오류 문장은 회전을 주는
  방법(데모와 `extract --video`의 `--rotation`, 그 밖의 추출 명령은 설정의 수동 회전 보정표)을 알려 준다.
- pts가 없는 파일(컨테이너 없는 원시 스트림 등)만 프레임 번호 / 평균 프레임률로 시각을 만든다.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from av.sidedata.sidedata import Type as SideDataType

# 정수 ms로 반올림하면 일정한 프레임 간격도 1 ms씩 흔들린다(예: 29.97 fps는 33과 34가 섞인다).
# 간격의 최대와 최소가 이 값보다 더 차이 나야 "간격이 변한다"고 본다.
CONSTANT_INTERVAL_TOLERANCE_MS = 1


class VideoError(ValueError):
    """영상을 읽을 수 없거나 시각을 정할 수 없을 때."""


@dataclass(slots=True)
class DecodeCounts:
    """`iter_video_frames`가 진행하면서 채우는 계수. 순회가 끝난 뒤에 읽는다."""

    decoded: int = 0  # 디코더가 내준 프레임 수
    skipped: int = 0  # 타임스탬프가 이전 값 이하이거나 pts가 없어서 버린 프레임 수
    rotation: int = 0  # 마지막으로 낸 프레임에 적용한 회전(반시계, 0·90·180·270)
    index_clock: bool = False  # pts가 없어 프레임 번호 / 평균 프레임률로 시각을 만들었는지
    display_flip: bool = False  # 표시 행렬에 뒤집기가 들어 있는 파일인지(회전을 직접 줬을 때만 True가 될 수 있다)


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """`probe_video`의 결과. 시각과 간격은 `iter_video_frames`가 내는 t_ms 기준이다."""

    n_frames: int  # 받아들일 프레임 수(타임스탬프가 엄격히 증가하는 것만)
    n_skipped: int  # 타임스탬프 때문에 버려질 프레임 수
    duration_ms: int  # 마지막 프레임 t_ms − 첫 프레임 t_ms
    mean_fps: float  # (n_frames − 1)·1000 / duration_ms. 계산할 수 없으면 0.0
    min_interval_ms: int  # 프레임 간격의 최소. 프레임이 2개 미만이면 0
    max_interval_ms: int
    variable_intervals: bool  # max − min > CONSTANT_INTERVAL_TOLERANCE_MS
    rotation: int  # 세우려고 적용하는 회전(반시계, 0·90·180·270)
    width: int  # 세운 뒤의 너비(픽셀)
    height: int
    display_flip: bool = False  # 표시 행렬에 뒤집기가 들어 있는지. True이면 rotation, width, height를 믿을 수 없다


def _round_half_up_div(num: int, den: int) -> int:
    """floor(num / den + 0.5)를 정수 연산으로 구한다(den > 0). 음수도 같은 식이다."""
    return (2 * num + den) // (2 * den)


def pts_to_ms(pts: int, first_pts: int, time_base: Fraction) -> int:
    """`t_ms = 1 + floor((pts − first_pts)·time_base·1000 + 0.5)`.

    Python의 round()는 짝수 반올림이라 쓰지 않는다. 부동소수 오차가 끼지 않도록 정수 연산만 쓴다.
    """
    num = (pts - first_pts) * time_base.numerator * 1000
    return 1 + _round_half_up_div(num, time_base.denominator)


def index_to_ms(index: int, rate: Fraction) -> int:
    """pts가 없는 파일용: `t_ms = 1 + floor(index·1000 / rate + 0.5)` (index는 0부터, rate는 평균 프레임률)."""
    return 1 + _round_half_up_div(index * 1000 * rate.denominator, rate.numerator)


def normalize_rotation(rotation: int) -> int:
    """회전 각도(반시계, 도)를 0·90·180·270 중 하나로 바꾼다. 90의 배수가 아니면 VideoError."""
    if rotation % 90 != 0:
        raise VideoError(f"rotation must be a multiple of 90 degrees, got {rotation}")
    return rotation % 360


def rotate_upright(rgb: np.ndarray, rotation: int) -> np.ndarray:
    """영상 [H, W, 3]을 반시계 방향으로 `rotation`도 돌린다(`frame.rotation`과 같은 규약). C 연속 배열을 돌려준다.

    90이면 원래의 오른쪽 변이 위로, 270(= −90)이면 왼쪽 변이 위로 온다.
    """
    quarter_turns = normalize_rotation(rotation) // 90
    if quarter_turns == 0:
        return np.ascontiguousarray(rgb)
    return np.ascontiguousarray(np.rot90(rgb, quarter_turns))


_DISPLAY_MATRIX_BYTES = 36  # int32 9개(16.16 고정소수점). 순서는 a, b, u, c, d, v, x, y, w


def display_matrix_has_flip(frame: Any) -> bool:
    """디코딩한 프레임의 표시 행렬(DISPLAYMATRIX 부가 정보)에 뒤집기(거울상)가 들어 있는지.

    행렬의 왼쪽 위 2x2 블록 [[a, b], [c, d]]의 행렬식 a·d − b·c가 음수이면 뒤집기다(순수한 회전은 양수).
    부가 정보가 없으면(회전 메타데이터가 없는 파일, 컨테이너 없는 원시 스트림) False다.
    가정: 부가 정보의 정수는 libav가 메모리에 둔 그대로(기계의 바이트 순서)다.
    """
    side_data = frame.side_data.get(SideDataType.DISPLAYMATRIX)
    if side_data is None:
        return False
    raw = bytes(side_data)
    if len(raw) != _DISPLAY_MATRIX_BYTES:
        return False
    a, b, _, c, d = struct.unpack("=5i", raw[:20])
    return a * d - b * c < 0


def _flip_error(path: Path) -> VideoError:
    return VideoError(
        f"{path}: the display matrix contains a flip (mirrored display), so the rotation metadata cannot be "
        "trusted (a horizontal flip alone reads as a 180 degree rotation). Fix: look at the video and give its "
        "rotation explicitly (0, 90, 180 or 270 degrees counter-clockwise): pass --rotation to "
        "`python -m drowsy.demo --video` or `python -m drowsy.extract --video`, and for the other extract inputs "
        "add the video to the manual rotation table of the config. The picture is then turned by that angle and "
        "not un-mirrored"
    )


def iter_video_frames(
    path: Path,
    rotation: int | None = None,
    counts: DecodeCounts | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """영상의 프레임을 (양수 t_ms, 똑바로 세운 RGB uint8 [H, W, 3])로 하나씩 낸다.

    - t_ms는 1에서 시작하고 엄격히 증가한다. 증가하지 않는 프레임은 버리고 `counts.skipped`에 센다.
    - rotation: None이면 프레임의 회전 메타데이터(`frame.rotation`)를 쓴다. 값을 주면 메타데이터 대신
      그 각도(반시계, 90의 배수)만큼 돌린다(설정의 수동 회전 보정표, 명령의 `--rotation`, 회전 자동 점검의 결과).
    - counts: 주면 디코딩 수, 버린 수, 적용한 회전, 표시 행렬의 뒤집기 여부를 여기에 적는다.

    가정: 표시 행렬에 좌우·상하 뒤집기가 들어 있는 파일(첫 프레임으로 판정한다)은 rotation을 주지 않으면
    VideoError다. libav가 알려 주는 회전은 뒤집기를 무시한 값이라, 좌우 뒤집기만 있는 똑바로 선 영상을 180도
    돌려 거꾸로 세우기 때문이다. rotation을 주면 그 각도로만 돌리고 거울상은 되돌리지 않으며
    `counts.display_flip`에 표시한다(모델 특징은 좌우 대칭이어서 거울상은 견디지만 거꾸로 선 얼굴은 못 견딘다).
    가정: 첫 프레임에 pts가 없으면 파일 전체를 프레임 번호 / 평균 프레임률 시계로 읽는다.
    첫 프레임에는 pts가 있는데 뒤 프레임에 없으면 그 프레임은 버리고 센다(두 시계를 섞지 않는다).
    """
    if counts is None:
        counts = DecodeCounts()
    forced = None if rotation is None else normalize_rotation(rotation)
    with av.open(str(path)) as container:
        if len(container.streams.video) == 0:
            raise VideoError(f"no video stream in {path}")
        stream = container.streams.video[0]
        time_base = stream.time_base
        rate = stream.average_rate
        first_pts = 0
        prev_ms = 0  # t_ms는 양수여야 하므로 0 이하인 값도 이 비교에서 걸러진다
        index = -1
        for frame in container.decode(stream):
            index += 1
            counts.decoded += 1
            if index == 0:
                counts.display_flip = display_matrix_has_flip(frame)
                if counts.display_flip and forced is None:
                    raise _flip_error(Path(path))
                counts.index_clock = frame.pts is None
                if counts.index_clock:
                    if rate is None or rate <= 0:
                        raise VideoError(f"{path}: frames carry no pts and the stream has no average rate")
                elif time_base is None:
                    raise VideoError(f"{path}: stream has no time base")
                else:
                    first_pts = frame.pts
            if counts.index_clock:
                t_ms = index_to_ms(index, rate)
            elif frame.pts is None:
                counts.skipped += 1
                continue
            else:
                t_ms = pts_to_ms(frame.pts, first_pts, time_base)
            if t_ms <= prev_ms:
                counts.skipped += 1
                continue
            prev_ms = t_ms
            counts.rotation = normalize_rotation(frame.rotation) if forced is None else forced
            yield t_ms, rotate_upright(frame.to_ndarray(format="rgb24"), counts.rotation)


def probe_video(path: Path) -> VideoInfo:
    """영상의 길이, 프레임 수, 평균 프레임률, 프레임 간격이 변하는지, 회전, 세운 뒤의 크기를 구한다.

    첫 프레임만 디코딩하고(회전과 크기) 나머지는 패킷의 pts만 읽으므로 빠르다.
    표시 행렬에 뒤집기가 들어 있어도 오류를 내지 않고 `display_flip`으로 알린다(그때 rotation과 크기는 libav가
    알려 준 회전을 그대로 따른 값이라 믿을 수 없다).
    가정: 패킷의 pts 집합이 디코딩된 프레임의 pts 집합과 같다고 본다(정상 파일에서 성립한다. 디코더가
    버리도록 표시된 패킷은 뺀다). 추출이 실제로 본 프레임 수는 추출 결과의 통계에 따로 남는다.
    """
    with av.open(str(path)) as container:
        if len(container.streams.video) == 0:
            raise VideoError(f"no video stream in {path}")
        stream = container.streams.video[0]
        first = next(container.decode(stream), None)
        if first is None:
            raise VideoError(f"no decodable video frame in {path}")
        rotation = normalize_rotation(first.rotation)
        display_flip = display_matrix_has_flip(first)
        width, height = first.width, first.height
        if rotation in (90, 270):
            width, height = height, width
        first_pts = first.pts
        time_base = stream.time_base
        rate = stream.average_rate

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # 끝을 알리는 빈 패킷(size 0)과 디코더가 버리는 패킷은 프레임이 되지 않는다.
        packet_pts = [p.pts for p in container.demux(stream) if p.size > 0 and not p.is_discard]

    n_skipped = 0
    if first_pts is None:
        if rate is None or rate <= 0:
            raise VideoError(f"{path}: frames carry no pts and the stream has no average rate")
        candidates = [index_to_ms(i, rate) for i in range(len(packet_pts))]
    else:
        if time_base is None:
            raise VideoError(f"{path}: stream has no time base")
        with_pts = [pts for pts in packet_pts if pts is not None]
        n_skipped = len(packet_pts) - len(with_pts)
        # 패킷은 디코딩 순서로 온다(B 프레임이 있으면 표시 순서와 다르다). 표시 순서로 정렬한다.
        candidates = [pts_to_ms(pts, first_pts, time_base) for pts in sorted(with_pts)]

    times: list[int] = []
    prev_ms = 0
    for t_ms in candidates:
        if t_ms <= prev_ms:
            n_skipped += 1
            continue
        prev_ms = t_ms
        times.append(t_ms)

    intervals = [later - earlier for earlier, later in zip(times, times[1:])]
    duration_ms = times[-1] - times[0] if times else 0
    min_interval = min(intervals) if intervals else 0
    max_interval = max(intervals) if intervals else 0
    return VideoInfo(
        n_frames=len(times),
        n_skipped=n_skipped,
        duration_ms=duration_ms,
        mean_fps=(len(times) - 1) * 1000.0 / duration_ms if duration_ms > 0 else 0.0,
        min_interval_ms=min_interval,
        max_interval_ms=max_interval,
        variable_intervals=max_interval - min_interval > CONSTANT_INTERVAL_TOLERANCE_MS,
        rotation=rotation,
        width=width,
        height=height,
        display_flip=display_flip,
    )
