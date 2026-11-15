"""원시 신호 계산 (이식 대상, 표준 라이브러리만 사용): EAR, 머리 자세, 홍채 비율, 눈동자 합성값.

`RawFrame` 하나를 받아 `FrameSignals` 하나를 돌려준다. 상태를 갖지 않는다.
PLAN 2.1절(랜드마크와 블렌드셰이프), 2.2절(행렬과 각도 규약), 4.1절(원시 신호).

규약
- 좌표: 랜드마크는 x가 너비, y가 높이로 따로 정규화돼 있다. 거리와 비율은 모두 픽셀 좌표
  (x·w, y·h)에서 구한다. 정규화 좌표로 EAR을 구하면 비정사각 프레임에서 값이 틀린다.
- 행렬: 16개 값은 열 우선이다. `M[i][j] = m[4·j + i]`. 행 우선으로 읽으면 전치가 되어 부호가 뒤집힌다.
- 회전: `R = Ry(yaw)·Rx(pitch)·Rz(roll)`. 3x3은 행 우선 9개 값으로 주고받는다. `R[i][j] = r[3·i + j]`.
- 각도: 도. yaw 양수 = 피험자가 자기 왼쪽으로 돌림, pitch 양수 = 턱이 내려감, roll 양수 = 영상에서 반시계 방향.
- 좌우: `_l`은 MediaPipe Left(L 집합, `*Left` 블렌드셰이프), `_r`은 Right다.
- 수학 함수는 atan2, hypot, sin, cos만 쓴다. 도와 라디안은 상수 곱으로 바꾼다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from drowsy.spec import PipelineSpec
from drowsy.types import (
    EL_DOWN_L,
    EL_DOWN_R,
    EL_IN_L,
    EL_IN_R,
    EL_OUT_L,
    EL_OUT_R,
    EL_UP_L,
    EL_UP_R,
    FrameSignals,
    RawFrame,
)

RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0

_NO_EYE_LOOK: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 눈: EAR, 홍채 비율
# ---------------------------------------------------------------------------


def eye_aspect_ratio(pts_px: Sequence[tuple[float, float]]) -> float:
    """EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|). 입력은 p1..p6 순서의 픽셀 좌표 6개다.

    가정: 눈 폭 |p1−p4|가 0이면 0.0을 돌려준다(PLAN 3.3절: 0으로 나누게 되는 비율은 0).
    `compute`는 폭이 6 px 미만인 눈에서 이 함수를 부르지 않는다.
    """
    if len(pts_px) != 6:
        raise ValueError("eye_aspect_ratio needs 6 points (p1..p6)")
    width = _distance(pts_px[0], pts_px[3])
    if width <= 0.0:
        return 0.0
    return (_distance(pts_px[1], pts_px[5]) + _distance(pts_px[2], pts_px[4])) / (2.0 * width)


def iris_ratio(
    outer: tuple[float, float], inner: tuple[float, float], iris: tuple[float, float]
) -> tuple[float, float]:
    """홍채 중심의 눈 안 위치 (h, v). 입력은 바깥 눈 꼬리, 안쪽 눈 꼬리, 홍채 중심의 픽셀 좌표다.

    a = inner − outer, d = iris − outer일 때 h = dot(d, a)/|a|², v = cross(a, d)/|a|².
    h는 바깥 꼬리에서 0, 안쪽 꼬리에서 1이다. cross(a, d) = a.x·d.y − a.y·d.x.

    두 눈은 a의 방향이 반대라서, 같은 시선 이동에 h와 v가 서로 반대 부호로 움직인다.
    두 눈 값을 합칠 때는 수평과 수직 모두 한쪽 부호를 뒤집어야 한다(PLAN 4.1절).
    이 모듈은 눈별 값만 내고 합치지 않는다(합친 값을 쓰는 곳이 없다).

    가정: 두 눈 꼬리가 같은 점이면 (0.0, 0.0)을 돌려준다(PLAN 3.3절).
    """
    ax = inner[0] - outer[0]
    ay = inner[1] - outer[1]
    dx = iris[0] - outer[0]
    dy = iris[1] - outer[1]
    norm_sq = ax * ax + ay * ay
    if norm_sq <= 0.0:
        return (0.0, 0.0)
    h = (dx * ax + dy * ay) / norm_sq
    v = (ax * dy - ay * dx) / norm_sq
    return (h, v)


# ---------------------------------------------------------------------------
# 눈동자 합성값 (시선 규칙 보조)
# ---------------------------------------------------------------------------


def gaze_h(eye_look: Sequence[float]) -> float:
    """수평 눈동자 합성값. 입력은 `FrameSignals.eye_look`(EYE_LOOK_ORDER 순서 8개).

    gaze_h = ((eyeLookInRight + eyeLookOutLeft) − (eyeLookOutRight + eyeLookInLeft)) / 2.
    양수는 피험자가 자기 왼쪽을 보는 방향이다(yaw와 같은 부호). 좌우 반전 입력에서는 부호가 뒤집힌다.
    정면에서도 0이 아니므로 기준선 대비 변화만 쓴다.
    """
    toward_left = eye_look[EL_IN_R] + eye_look[EL_OUT_L]
    toward_right = eye_look[EL_OUT_R] + eye_look[EL_IN_L]
    return (toward_left - toward_right) / 2.0


def gaze_v(eye_look: Sequence[float]) -> float:
    """수직 눈동자 합성값 = (eyeLookUp 두 눈 평균) − (eyeLookDown 두 눈 평균). 양수는 위쪽이다."""
    up = (eye_look[EL_UP_L] + eye_look[EL_UP_R]) / 2.0
    down = (eye_look[EL_DOWN_L] + eye_look[EL_DOWN_R]) / 2.0
    return up - down


# ---------------------------------------------------------------------------
# 머리 자세
# ---------------------------------------------------------------------------


def euler_from_colmajor(m: Sequence[float]) -> tuple[float, float, float]:
    """열 우선 4x4 변환 행렬(16개 값)에서 (yaw, pitch, roll)을 도 단위로 구한다 (PLAN 2.2절).

    yaw = atan2(M[0][2], M[2][2]), pitch = atan2(−M[1][2], hypot(M[1][0], M[1][1])),
    roll = atan2(M[1][0], M[1][1]). 열 우선이므로 M[0][2] = m[8], M[2][2] = m[10], M[1][2] = m[9],
    M[1][0] = m[1], M[1][1] = m[5]다.
    """
    if len(m) != 16:
        raise ValueError("transform matrix must hold 16 values (column-major 4x4)")
    yaw = math.atan2(m[8], m[10])
    pitch = math.atan2(-m[9], math.hypot(m[1], m[5]))
    roll = math.atan2(m[1], m[5])
    return (yaw * RAD_TO_DEG, pitch * RAD_TO_DEG, roll * RAD_TO_DEG)


def euler_from_rotation(r: Sequence[float]) -> tuple[float, float, float]:
    """행 우선 3x3 회전(9개 값)에서 (yaw, pitch, roll)을 도 단위로 구한다. 식은 `euler_from_colmajor`와 같다."""
    if len(r) != 9:
        raise ValueError("rotation must hold 9 values (row-major 3x3)")
    yaw = math.atan2(r[2], r[8])
    pitch = math.atan2(-r[5], math.hypot(r[3], r[4]))
    roll = math.atan2(r[3], r[4])
    return (yaw * RAD_TO_DEG, pitch * RAD_TO_DEG, roll * RAD_TO_DEG)


def rotation_from_euler(yaw: float, pitch: float, roll: float) -> tuple[float, ...]:
    """도 단위 각도로 R = Ry(yaw)·Rx(pitch)·Rz(roll)을 만든다. 행 우선 9개 값.

    Ry = [[cy, 0, sy], [0, 1, 0], [−sy, 0, cy]]
    Rx = [[1, 0, 0], [0, cp, −sp], [0, sp, cp]]
    Rz = [[cr, −sr, 0], [sr, cr, 0], [0, 0, 1]]
    """
    cy = math.cos(yaw * DEG_TO_RAD)
    sy = math.sin(yaw * DEG_TO_RAD)
    cp = math.cos(pitch * DEG_TO_RAD)
    sp = math.sin(pitch * DEG_TO_RAD)
    cr = math.cos(roll * DEG_TO_RAD)
    sr = math.sin(roll * DEG_TO_RAD)
    return (
        cy * cr + sy * sp * sr,
        sy * sp * cr - cy * sr,
        sy * cp,
        cp * sr,
        cp * cr,
        -sp,
        cy * sp * sr - sy * cr,
        sy * sr + cy * sp * cr,
        cy * cp,
    )


def rotation_from_colmajor(m: Sequence[float]) -> tuple[float, ...]:
    """열 우선 4x4 변환 행렬에서 회전 블록을 행 우선 9개 값으로 꺼낸다. R[i][j] = m[4·j + i]."""
    if len(m) != 16:
        raise ValueError("transform matrix must hold 16 values (column-major 4x4)")
    return (m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10])


def relative_rotation(r: Sequence[float], r0: Sequence[float]) -> tuple[float, ...]:
    """기준 회전 대비 회전 R0ᵀ·R. 입력과 출력 모두 행 우선 9개 값이다.

    (R0ᵀ·R)[i][j] = Σ_k R0[k][i]·R[k][j].
    """
    if len(r) != 9 or len(r0) != 9:
        raise ValueError("rotation must hold 9 values (row-major 3x3)")
    out: list[float] = []
    for i in range(3):
        for j in range(3):
            out.append(r0[i] * r[j] + r0[3 + i] * r[3 + j] + r0[6 + i] * r[6 + j])
    return tuple(out)


def relative_euler(m: Sequence[float], r0: Sequence[float]) -> tuple[float, float, float]:
    """기준 자세 대비 (yaw, pitch, roll) = Euler(R0ᵀ·R). m은 열 우선 16개 값, r0은 행 우선 9개 값.

    오일러 각의 단순 차가 아니다. 기준선이 (30, 15, 0)도일 때 단순 차는 pitch와 roll에 수 도의 오차를 낸다.
    """
    return euler_from_rotation(relative_rotation(rotation_from_colmajor(m), r0))


# ---------------------------------------------------------------------------
# 프레임 하나의 원시 신호
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EyeSignals:
    """눈 하나의 기하 신호. 쓸 수 없는 눈(폭이 하한 미만)은 ear가 0.0, 홍채 비율이 None이고 폭은 그대로 적는다."""

    ear: float
    width_px: float
    iris_h: float | None
    iris_v: float | None


def compute(frame: RawFrame, spec: PipelineSpec) -> FrameSignals:
    """프레임 하나의 원시 신호를 계산한다 (PLAN 4.1절).

    - 얼굴이 없으면 모든 실수 필드가 0.0, eye_look이 0.0 여덟 개, 홍채 비율이 None인 값을 돌려준다.
    - 눈 폭 |p1−p4|가 `spec.signals.min_eye_width_px`(6 px) 미만인 눈은 쓸 수 없는 눈이다.
      EAR과 홍채 비율을 계산하지 않고 ear = 0.0, 홍채 비율 = None으로 두며, 눈 폭은 측정값을 그대로 적는다.
      호출자는 `eye_w_* < min_eye_width_px`로 쓸 수 없는 눈을 가려낸다.
    - 블렌드셰이프와 각도는 눈 폭과 무관하게 채운다.

    가정: 홍채 비율은 쓸 수 있는 눈마다 항상 계산한다. `RawFrame.lm`이 홍채 랜드마크를 담는 것이 계약이다.
    face=True인데 필요한 랜드마크나 블렌드셰이프가 없으면 KeyError, 행렬 길이가 16이 아니면 ValueError가 난다.
    """
    if not frame.face:
        return FrameSignals(
            t_ms=frame.t_ms,
            face=False,
            ear_l=0.0,
            ear_r=0.0,
            eye_w_l=0.0,
            eye_w_r=0.0,
            blink_l=0.0,
            blink_r=0.0,
            eye_look=_NO_EYE_LOOK,
            jaw_open=0.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
        )

    lms = spec.landmarks
    min_width_px = spec.signals.min_eye_width_px
    eye_l = _eye_signals(frame, lms.ear_l, lms.corner_outer_l, lms.corner_inner_l, lms.iris_l[0], min_width_px)
    eye_r = _eye_signals(frame, lms.ear_r, lms.corner_outer_r, lms.corner_inner_r, lms.iris_r[0], min_width_px)

    names = spec.blendshapes
    eye_look: list[float] = []
    for name in names.eye_look:
        eye_look.append(frame.bs[name])

    yaw, pitch, roll = euler_from_colmajor(frame.m)

    return FrameSignals(
        t_ms=frame.t_ms,
        face=True,
        ear_l=eye_l.ear,
        ear_r=eye_r.ear,
        eye_w_l=eye_l.width_px,
        eye_w_r=eye_r.width_px,
        blink_l=frame.bs[names.eye_blink_l],
        blink_r=frame.bs[names.eye_blink_r],
        eye_look=tuple(eye_look),
        jaw_open=frame.bs[names.jaw_open],
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        iris_h_l=eye_l.iris_h,
        iris_h_r=eye_r.iris_h,
        iris_v_l=eye_l.iris_v,
        iris_v_r=eye_r.iris_v,
    )


def _eye_signals(
    frame: RawFrame,
    ear_indices: Sequence[int],
    outer_index: int,
    inner_index: int,
    iris_centre_index: int,
    min_width_px: float,
) -> EyeSignals:
    pts: list[tuple[float, float]] = []
    for index in ear_indices:
        pts.append(_pixel(frame, index))
    width = _distance(pts[0], pts[3])
    if width < min_width_px:
        return EyeSignals(ear=0.0, width_px=width, iris_h=None, iris_v=None)
    h, v = iris_ratio(_pixel(frame, outer_index), _pixel(frame, inner_index), _pixel(frame, iris_centre_index))
    return EyeSignals(ear=eye_aspect_ratio(pts), width_px=width, iris_h=h, iris_v=v)


def _pixel(frame: RawFrame, index: int) -> tuple[float, float]:
    x, y = frame.lm[index]
    return (x * frame.w, y * frame.h)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
