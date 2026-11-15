"""합성 세션 생성기 (PLAN 9장, D14). 이식 대상이 아니다. 테스트, 테스트 벡터, 합성 데이터 전 과정(M2)이 함께 쓴다.

실제 얼굴 이미지는 쓰지 않는다. 만드는 것은 앱이 MediaPipe에서 받는 것과 같은 모양의 `RawFrame`이고,
신호·closure·감김 검출·스텝 특징은 여기서 계산하지 않는다(그것은 `DriverPipeline`이 한다. PLAN 3.1절).

네 부분으로 되어 있다.
 (a) 기하: `frame_from_signals`가 요청한 원시 신호(`FrameSignals`)에서 `RawFrame`을 만든다.
     `signals.compute(frame_from_signals(s, w, h, spec), spec)`는 s를 되돌려준다. 블렌드셰이프 값은 비트 단위로 같고,
     EAR·눈 폭·홍채 비율·각도는 좌표와 행렬을 거치므로 부동소수 반올림 오차(절대 1e-9 미만) 안에서 같다.
     `mirror_frame`은 좌우 반전 영상에서 MediaPipe가 냈을 프레임을, `no_face_frame`은 얼굴 없는 프레임을 만든다.
 (b) 각본 시나리오: `Scenario`에 구간(`Blink`, `Closure`, `Linger`, `FaceMissing`, `HeadTurn`, `LookAside`,
     `Yawn`, `Nod`, `Open`)을 시간표로 적고 `render`로 프레임 목록을 얻는다. 시드가 같으면 결과가 같다.
 (c) 통계 생성기: `make_subject`와 `generate_session`이 피험자별 특성과 세 상태(alert, low_vigilant, drowsy)의
     차이를 가진 긴 세션을 만든다.
 (d) 명령: 합성 데이터셋을 추출 결과와 같은 형식의 parquet으로 쓴다(저장소 루트에서 실행).
         python -m drowsy.synth --config config/default.yaml --out data/synth --subjects 60 [--minutes 10] [--seed 0]
     배치는 RLDD 미러와 같다: `Fold{k}_part{p}/{NN}/{0|5|10}.parquet`. 두 피험자의 drowsy 세션은 `10_1`, `10_2`로 나뉜다.
     `--fps-sim 15 --out data/synth_fps15`를 주면 같은 세션에서 기기의 흐름 제한기가 버렸을 프레임을 뺀 것을 쓴다
     (실제 데이터의 `python -m drowsy.extract --dataset --fps-sim 15`에 해당하는 합성 데이터).

합성 데이터의 점수는 파이프라인 점검용이고 성능으로 보고하지 않는다(PLAN 3.4절).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import multiprocessing
import random
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from drowsy import signals
from drowsy.config import DropSimConfig
from drowsy.spec import LandmarkSpec, PipelineSpec, SpecError
from drowsy.types import (
    CLASS_DROWSY,
    CLASS_ORDER,
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

# ---------------------------------------------------------------------------
# (a) 기하
# ---------------------------------------------------------------------------

# 합성 얼굴의 비율(눈 폭 대비). 신호 계산에는 영향이 없고 얼굴 폭 통계(face_w_px)와 좌표 배치만 정한다.
EYE_GAP_RATIO = 2.1  # 두 눈 중심 사이의 거리
FACE_WIDTH_RATIO = 4.6  # 얼굴 폭(랜드마크 234와 454 사이)
FACE_EDGE_DROP_RATIO = 0.3  # 얼굴 폭 측정점이 눈 높이보다 아래에 있는 정도
IRIS_RADIUS_RATIO = 0.15
DEFAULT_EYE_WIDTH_PX = 40.0
DEFAULT_CENTRE: tuple[float, float] = (0.5, 0.45)  # 두 눈 사이 중점의 정규화 좌표
DEFAULT_TRANSLATION: tuple[float, float, float] = (0.0, 0.0, -50.0)  # 행렬의 평행이동(cm). 파이프라인은 쓰지 않는다

# 눈 윤곽 고리(16점)에서 위치 i(1..7)의 눈꺼풀 높이 비율. 위치 3과 5(EAR의 p2, p3가 놓이는 곳)에서 1이다.
# g(f) = min(1, f·(1 − f)·64/15), f = i/8
_LID_PROFILE: tuple[float, ...] = (0.0, 7.0 / 15.0, 0.8, 1.0, 1.0, 1.0, 0.8, 7.0 / 15.0)
_RING_POINTS = 16
_RING_INNER = 8  # 고리에서 안쪽 눈 꼬리의 위치
_EAR_RING_POSITIONS = frozenset({0, 3, 5, 8, 11, 13})


@functools.lru_cache(maxsize=8)
def _check_landmark_layout(landmarks: LandmarkSpec) -> None:
    """이 모듈의 좌표 배치가 전제하는 눈 윤곽 순서를 spec이 지키는지 검사한다.

    전제: 윤곽은 바깥 꼬리(위치 0)에서 윗눈꺼풀(1..7)을 따라 안쪽 꼬리(8)까지, 아랫눈꺼풀(9..15)로 돌아오는
    16점 고리이고, EAR의 여섯 점은 위치 0, 3, 5, 8, 11, 13에 있다(위치 3과 13, 5와 11이 위아래 짝).
    """
    for contour, ear, outer, inner, iris in (
        (landmarks.contour_r, landmarks.ear_r, landmarks.corner_outer_r, landmarks.corner_inner_r, landmarks.iris_r),
        (landmarks.contour_l, landmarks.ear_l, landmarks.corner_outer_l, landmarks.corner_inner_l, landmarks.iris_l),
    ):
        ok = len(contour) == _RING_POINTS and len(ear) == 6 and len(iris) == 5
        ok = ok and contour[0] == outer and contour[_RING_INNER] == inner
        if ok:
            positions = [contour.index(index) if index in contour else -1 for index in ear]
            pairs = {frozenset((positions[1], positions[5])), frozenset((positions[2], positions[4]))}
            ok = (
                set(positions) == _EAR_RING_POSITIONS
                and {positions[0], positions[3]} == {0, _RING_INNER}
                and pairs == {frozenset((3, 13)), frozenset((5, 11))}
            )
        if not ok:
            raise SpecError(
                "drowsy.synth supports only the eye contour layout of spec v0 "
                "(16-point ring, EAR points at ring positions 0, 3, 5, 8, 11, 13)"
            )
    if len(landmarks.face_width) != 2:
        raise SpecError("drowsy.synth needs exactly two face-width landmarks")


def pose_matrix(
    yaw: float,
    pitch: float,
    roll: float,
    translation: tuple[float, float, float] = DEFAULT_TRANSLATION,
) -> tuple[float, ...]:
    """각도(도)에서 열 우선 4x4 변환 행렬 16개 값을 만든다. `M[i][j] = m[4·j + i]`, R = Ry(yaw)·Rx(pitch)·Rz(roll).

    회전은 `signals.rotation_from_euler`(이식 코드와 같은 함수)로 만든다.
    `signals.euler_from_colmajor`가 같은 각도를 되돌려주려면 |pitch| < 90이어야 한다.
    """
    r = signals.rotation_from_euler(yaw, pitch, roll)
    return (
        r[0], r[3], r[6], 0.0,
        r[1], r[4], r[7], 0.0,
        r[2], r[5], r[8], 0.0,
        translation[0], translation[1], translation[2], 1.0,
    )  # fmt: skip


def no_face_frame(t_ms: int, w: int, h: int) -> RawFrame:
    """얼굴이 검출되지 않은 프레임(lm, bs는 빈 dict, m은 빈 튜플)."""
    return RawFrame(t_ms=t_ms, face=False, w=w, h=h, lm={}, bs={}, m=())


def frame_from_signals(
    s: FrameSignals,
    w: int,
    h: int,
    spec: PipelineSpec,
    centre: tuple[float, float] = DEFAULT_CENTRE,
) -> RawFrame:
    """요청한 원시 신호 s를 내는 `RawFrame`을 만든다. `signals.compute(결과, spec)`은 s를 되돌려준다.

    - 랜드마크 44개(`spec.landmarks.required_indices()`), 블렌드셰이프 11개(`required_names()`), 행렬 16개 값을 담는다.
    - 눈은 축에 나란한 2D 도형으로 그린다. EAR의 위아래 짝은 같은 x에, 두 눈 꼬리는 같은 y에 두므로
      EAR = 눈꺼풀 간격 / 눈 폭이 된다. 프레임이 정사각형이 아니어도 성립한다(좌표를 픽셀에서 정한 뒤 정규화한다).
    - 머리 자세는 행렬에만 넣는다. 랜드마크는 돌리지 않는다(요청한 EAR이 그대로 나오게 하기 위해서다).
    - 눈 폭은 `s.eye_w_l`, `s.eye_w_r`(픽셀)이다. `spec.signals.min_eye_width_px`보다 작으면 쓸 수 없는 눈이 되고,
      그 눈의 `ear_*`와 홍채 비율은 무시된다(`signals.compute`가 0.0과 None을 낸다).
    - 홍채 비율이 None이면 (h, v) = (0.5, 0.0)에 둔다.
    - `centre`는 두 눈 사이 중점의 정규화 좌표다. MediaPipe Right 눈(R 집합)이 영상 왼쪽에 온다(비반전 영상).
    - `s.face`가 False이면 얼굴 없는 프레임이다.

    정확도: 블렌드셰이프는 넣은 값 그대로다. EAR, 눈 폭, 홍채 비율, 각도는 반올림 오차만큼 다를 수 있다
    (정규화 좌표와 행렬을 거치기 때문이다. 절대 1e-9 미만이고, 무작위 요청 2만 개에서 잰 최대 오차는 1e-12 미만이었다).
    가정: 요청값의 범위는 검사하지 않는다(EAR ≥ 0, 눈 폭 > 0, |pitch| < 90을 호출자가 지킨다).
    """
    if not s.face:
        return no_face_frame(s.t_ms, w, h)
    lms = spec.landmarks
    _check_landmark_layout(lms)
    if len(s.eye_look) != len(spec.blendshapes.eye_look):
        raise ValueError(f"eye_look must hold {len(spec.blendshapes.eye_look)} values, got {len(s.eye_look)}")

    mean_eye_w = 0.5 * (s.eye_w_l + s.eye_w_r)
    cx = centre[0] * w
    cy = centre[1] * h
    half_gap = 0.5 * EYE_GAP_RATIO * mean_eye_w
    lm: dict[int, tuple[float, float]] = {}
    _place_eye(lm, lms.contour_r, lms.iris_r, cx - half_gap, cy, -1.0, s.eye_w_r, s.ear_r, s.iris_h_r, s.iris_v_r, w, h)
    _place_eye(lm, lms.contour_l, lms.iris_l, cx + half_gap, cy, 1.0, s.eye_w_l, s.ear_l, s.iris_h_l, s.iris_v_l, w, h)
    edge_y = (cy + FACE_EDGE_DROP_RATIO * mean_eye_w) / h
    half_face = 0.5 * FACE_WIDTH_RATIO * mean_eye_w
    lm[lms.face_width[0]] = ((cx - half_face) / w, edge_y)  # 234: 피험자의 오른쪽 = 영상 왼쪽
    lm[lms.face_width[1]] = ((cx + half_face) / w, edge_y)

    names = spec.blendshapes
    bs = {names.eye_blink_l: s.blink_l, names.eye_blink_r: s.blink_r, names.jaw_open: s.jaw_open}
    for name, value in zip(names.eye_look, s.eye_look):
        bs[name] = value
    return RawFrame(t_ms=s.t_ms, face=True, w=w, h=h, lm=lm, bs=bs, m=pose_matrix(s.yaw, s.pitch, s.roll))


def _place_eye(
    lm: dict[int, tuple[float, float]],
    contour: Sequence[int],
    iris: Sequence[int],
    eye_x: float,
    eye_y: float,
    outer_side: float,
    eye_w: float,
    ear: float,
    iris_h: float | None,
    iris_v: float | None,
    w: int,
    h: int,
) -> None:
    """눈 하나의 윤곽 16점과 홍채 5점을 놓는다. 좌표는 픽셀에서 정하고 w, h로 나눠 적는다.

    outer_side는 바깥 눈 꼬리가 눈 중심의 어느 쪽에 있는지다(영상 왼쪽이면 −1, 오른쪽이면 +1).
    """
    x_outer = eye_x + outer_side * 0.5 * eye_w
    x_inner = eye_x - outer_side * 0.5 * eye_w
    span = x_inner - x_outer
    y_mid = eye_y / h
    lm[contour[0]] = (x_outer / w, y_mid)
    lm[contour[_RING_INNER]] = (x_inner / w, y_mid)
    half_open = 0.5 * ear * eye_w
    for i in range(1, _RING_INNER):
        x = (x_outer + span * (i / 8.0)) / w  # 위아래 짝이 같은 x를 쓴다
        dy = half_open * _LID_PROFILE[i]
        lm[contour[i]] = (x, (eye_y - dy) / h)
        lm[contour[_RING_POINTS - i]] = (x, (eye_y + dy) / h)

    # 홍채: a = 안쪽 − 바깥 = (span, 0)이므로 h = dx/span, v = cross(a, d)/|a|² = dy/span
    if iris_h is None:
        iris_h = 0.5
    if iris_v is None:
        iris_v = 0.0
    ix = x_outer + iris_h * span
    iy = eye_y + iris_v * span
    radius = IRIS_RADIUS_RATIO * eye_w
    lm[iris[0]] = (ix / w, iy / h)
    lm[iris[1]] = ((ix + radius) / w, iy / h)  # 순서: 중심, +x, 위, −x, 아래
    lm[iris[2]] = (ix / w, (iy - radius) / h)
    lm[iris[3]] = ((ix - radius) / w, iy / h)
    lm[iris[4]] = (ix / w, (iy + radius) / h)


def mirror_frame(frame: RawFrame, spec: PipelineSpec) -> RawFrame:
    """좌우 반전한 영상에서 MediaPipe가 냈을 프레임 (PLAN 2.1, 2.2절).

    - 좌표: x → 1 − x. 인덱스는 좌우 짝으로 바꾼다(R 윤곽 ↔ L 윤곽의 같은 고리 위치, 홍채의 +x점 ↔ −x점,
      얼굴 폭의 두 점).
    - 블렌드셰이프: 이름 끝의 Left ↔ Right를 바꾼다.
    - 행렬: M' = S·M·S, S = diag(−1, 1, 1, 1). yaw와 roll의 부호가 뒤집히고 pitch는 그대로다.
    그래서 `signals.compute`의 결과는 `_l`과 `_r`이 서로 바뀌고 yaw, roll의 부호가 뒤집힌다.
    가정: `frame.lm`은 `spec.landmarks.required_indices()`의 인덱스만 담고 있어야 한다(다른 인덱스는 짝을 모른다).
    """
    if not frame.face:
        return frame
    partner = _mirror_partners(spec.landmarks)
    lm: dict[int, tuple[float, float]] = {}
    for index, (x, y) in frame.lm.items():
        if index not in partner:
            raise ValueError(f"mirror_frame does not know the mirror partner of landmark {index}")
        lm[partner[index]] = (1.0 - x, y)
    bs: dict[str, float] = {}
    for name, value in frame.bs.items():
        if name.endswith("Left"):
            bs[name[: -len("Left")] + "Right"] = value
        elif name.endswith("Right"):
            bs[name[: -len("Right")] + "Left"] = value
        else:
            bs[name] = value
    m = list(frame.m)
    for i in range(4):
        for j in range(4):
            if (i == 0) != (j == 0):  # 첫 행과 첫 열 중 한쪽에만 속한 값의 부호가 바뀐다
                m[4 * j + i] = -m[4 * j + i]
    return RawFrame(t_ms=frame.t_ms, face=True, w=frame.w, h=frame.h, lm=lm, bs=bs, m=tuple(m))


@functools.lru_cache(maxsize=8)
def _mirror_partners(landmarks: LandmarkSpec) -> dict[int, int]:
    _check_landmark_layout(landmarks)
    partner: dict[int, int] = {}
    for a, b in zip(landmarks.contour_r, landmarks.contour_l):
        partner[a] = b
        partner[b] = a
    for i, j in ((0, 0), (1, 3), (2, 2), (3, 1), (4, 4)):  # 중심, +x ↔ −x, 위, 아래
        partner[landmarks.iris_r[i]] = landmarks.iris_l[j]
        partner[landmarks.iris_l[j]] = landmarks.iris_r[i]
    partner[landmarks.face_width[0]] = landmarks.face_width[1]
    partner[landmarks.face_width[1]] = landmarks.face_width[0]
    return partner


# ---------------------------------------------------------------------------
# (b) 각본 시나리오
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FaceProfile:
    """각본 시나리오의 얼굴: 눈을 뜨고 정면을 볼 때의 신호 수준과 영상 안의 위치.

    `default_profile(spec)`은 전역 상수(E, B, jaw0)와 같은 값을 쓴다. 그러면 보정 전의 임시 closure(D4)와
    보정 후의 closure가 모두 구간에 적은 수준(`depth`, `level`)과 같아진다.
    가정: 구간에 적는 수준은 "기준선이 이 얼굴의 뜬 눈 값일 때의 closure"로 정의한다(`lid_signals`).
    쉬는 자세가 0이 아니면 `HeadTurn`의 각도는 기준선 대비 각도(Euler(R0ᵀ·R))와 조금 다를 수 있다.
    """

    e_open_l: float
    e_open_r: float
    b_open_l: float
    b_open_r: float
    jaw_rest: float
    yaw: float = 0.0  # 쉬는 자세(카메라 거치 각도). 도
    pitch: float = 0.0
    roll: float = 0.0
    eye_look_rest: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # EYE_LOOK_ORDER
    eye_w_px: float = DEFAULT_EYE_WIDTH_PX
    centre: tuple[float, float] = DEFAULT_CENTRE


def default_profile(spec: PipelineSpec) -> FaceProfile:
    """뜬 눈의 EAR과 eyeBlink, 쉬는 턱 값이 spec의 전역 상수와 같은 얼굴."""
    constants = spec.global_constants
    return FaceProfile(
        e_open_l=constants.e,
        e_open_r=constants.e,
        b_open_l=constants.b,
        b_open_r=constants.b,
        jaw_rest=constants.jaw0,
    )


def lid_signals(level: float, e_open: float, b_open: float, spec: PipelineSpec) -> tuple[float, float]:
    """눈꺼풀 수준 level(0 = 뜸, 1 = 감김)에 해당하는 눈 하나의 (EAR, eyeBlink).

    기준선이 E = e_open, B = b_open일 때 `calibrate.eye_closure`가 level을 되돌려주도록 PLAN 4.3절의 식을 뒤집은 것이다.
        ear   = e_open·(1 − (1 − rho)·level)
        blink = b_open + max(b_closed − b_open, b_span_min)·level
    0 ≤ level ≤ 1에서 EAR 쪽 a_e와 블렌드셰이프 쪽 s_e가 둘 다 level이므로 W가 무엇이든 c_e = level이다.
    """
    c = spec.closure
    span = c.b_closed - b_open
    if span < c.b_span_min:
        span = c.b_span_min
    return (e_open * (1.0 - (1.0 - c.rho) * level), b_open + span * level)


@dataclass(frozen=True, slots=True)
class Open:
    """눈을 뜨고 있는 구간. 아무 신호도 바꾸지 않고 `sequence`에서 시간만 차지한다."""

    duration_ms: int
    start_ms: int = 0


@dataclass(frozen=True, slots=True)
class Blink:
    """깜빡임: 눈꺼풀 수준이 0에서 depth까지 올라갔다가 내려온다(감는 데 40%, 뜨는 데 60%, 머무는 구간 없음).

    가정: 모양은 삼각형으로 정했다. 감김 검출기가 재는 길이(0.5를 넘는 시간)는 depth가 1일 때 duration_ms의 절반이다.
    """

    duration_ms: int = 200
    depth: float = 1.0
    start_ms: int = 0


@dataclass(frozen=True, slots=True)
class Closure:
    """감김: close_ms 동안 감고, depth에서 hold_ms 동안 머물고, open_ms 동안 뜬다."""

    hold_ms: int
    depth: float = 1.0
    close_ms: int = 100
    open_ms: int = 150
    start_ms: int = 0

    @property
    def duration_ms(self) -> int:
        return self.close_ms + self.hold_ms + self.open_ms


@dataclass(frozen=True, slots=True)
class Linger:
    """눈꺼풀이 level에 머무는 구간(반쯤 감긴 눈). duration_ms는 양 끝의 ramp_ms를 포함한다."""

    duration_ms: int
    level: float
    ramp_ms: int = 150
    start_ms: int = 0


@dataclass(frozen=True, slots=True)
class FaceMissing:
    """얼굴이 검출되지 않는 구간. 프레임은 오지만 face=False다."""

    duration_ms: int
    start_ms: int = 0


@dataclass(frozen=True, slots=True)
class HeadTurn:
    """머리를 돌린 구간. 쉬는 자세에 (yaw, pitch, roll)도를 더한다. duration_ms는 양 끝의 ramp_ms를 포함한다."""

    duration_ms: int
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    ramp_ms: int = 300
    start_ms: int = 0


@dataclass(frozen=True, slots=True)
class LookAside:
    """머리는 두고 눈만 돌린 구간. 눈동자 합성값(`signals.gaze_h`, `gaze_v`)이 쉬는 값에서 그만큼 변한다.

    gaze_h > 0은 피험자의 왼쪽(eyeLookInRight, eyeLookOutLeft가 커진다), gaze_v > 0은 위쪽이다.
    """

    duration_ms: int
    gaze_h: float = 0.0
    gaze_v: float = 0.0
    ramp_ms: int = 100
    start_ms: int = 0


@dataclass(frozen=True, slots=True)
class Yawn:
    """하품: jawOpen이 쉬는 값에서 jaw만큼 커진다. duration_ms는 양 끝의 ramp_ms를 포함한다."""

    duration_ms: int = 4000
    jaw: float = 0.6
    ramp_ms: int = 500
    start_ms: int = 0


@dataclass(frozen=True, slots=True)
class Nod:
    """끄덕임: pitch가 pitch도만큼 내려갔다가(턱이 내려가는 쪽이 양수) 돌아온다. 내려가는 데 절반, 올라오는 데 절반."""

    duration_ms: int = 1000
    pitch: float = 20.0
    start_ms: int = 0


Segment = Open | Blink | Closure | Linger | FaceMissing | HeadTurn | LookAside | Yawn | Nod


def sequence(*segments: Segment, start_ms: int = 0) -> tuple[Segment, ...]:
    """구간을 차례로 이어 붙인다. 각 구간의 `start_ms`를 앞 구간이 끝나는 시각으로 바꾼 사본을 돌려준다.

    겹치는 구간(예: 감긴 채로 고개를 돌림)은 `start_ms`를 직접 적은 구간을 결과에 더하면 된다.
    """
    placed: list[Segment] = []
    cursor = start_ms
    for segment in segments:
        placed.append(replace(segment, start_ms=cursor))
        cursor += segment.duration_ms
    return tuple(placed)


@dataclass(frozen=True, slots=True)
class Scenario:
    """각본 세션 하나. 시각은 모두 세션 시간(첫 프레임이 0)이고 구간은 `start_ms ≤ t < start_ms + duration_ms`에서 작용한다.

    - segments: 구간 목록. 구간이 없는 시간은 눈을 뜨고 쉬는 자세로 있는 상태다. 겹쳐도 된다.
      눈꺼풀 수준은 겹친 구간의 최댓값, 자세·시선·턱의 변화는 합이다.
    - duration_ms: 세션 길이. 프레임 시각은 0 이상 duration_ms 미만이다.
    - fps: 명목 프레임률. 프레임 k의 명목 시각은 floor(k·1000/fps + 0.5)다.
    - jitter_ms: 프레임 시각을 [−jitter_ms, +jitter_ms]의 정수만큼 무작위로 민다(첫 프레임은 밀지 않는다).
    - drop_prob: 첫 프레임을 뺀 각 프레임을 이 확률로 버린다.
    - gaps: (시작, 끝) 세션 시간 구간. 이 안의 프레임은 없다(시작 ≤ t < 끝).
    - first_t_ms: 첫 프레임의 t_ms. 추출 결과처럼 1에서 시작하는 것이 기본이다.
    - profile: None이면 `default_profile(spec)`.
    - seed: jitter와 drop의 난수 시드. 같은 시나리오는 언제나 같은 프레임을 낸다.

    가정: 각본 시나리오의 신호에는 잡음을 넣지 않는다(기대값을 손으로 계산할 수 있게). 잡음 있는 세션은 통계 생성기가 만든다.
    """

    segments: tuple[Segment, ...]
    duration_ms: int
    fps: float = 30.0
    w: int = 640
    h: int = 480
    jitter_ms: int = 0
    drop_prob: float = 0.0
    gaps: tuple[tuple[int, int], ...] = ()
    first_t_ms: int = 1
    profile: FaceProfile | None = None
    seed: int = 0


def scenario_times(scenario: Scenario) -> list[int]:
    """시나리오의 프레임 시각(세션 시간 ms, 엄격히 증가). 첫 값은 항상 0이다."""
    if scenario.fps <= 0.0 or scenario.duration_ms <= 0:
        raise ValueError("scenario needs fps > 0 and duration_ms > 0")
    if scenario.jitter_ms < 0 or not 0.0 <= scenario.drop_prob < 1.0:
        raise ValueError("scenario needs jitter_ms >= 0 and 0 <= drop_prob < 1")
    rng = random.Random(scenario.seed)
    times: list[int] = [0]
    k = 1
    while True:
        nominal = math.floor(k * 1000.0 / scenario.fps + 0.5)
        if nominal >= scenario.duration_ms:
            break
        k += 1
        # 난수는 버려지는 프레임에서도 뽑는다. 프레임 하나가 빠져도 뒤 프레임들의 난수가 밀리지 않는다.
        shift = rng.randint(-scenario.jitter_ms, scenario.jitter_ms) if scenario.jitter_ms > 0 else 0
        dropped = rng.random() < scenario.drop_prob if scenario.drop_prob > 0.0 else False
        t = nominal + shift
        if dropped or t <= times[-1] or t >= scenario.duration_ms:
            continue
        if any(start <= t < end for start, end in scenario.gaps):
            continue
        times.append(t)
    return times


def render(scenario: Scenario, spec: PipelineSpec) -> list[RawFrame]:
    """시나리오를 `RawFrame` 목록으로 만든다. `t_ms = first_t_ms + 세션 시간`."""
    profile = scenario.profile if scenario.profile is not None else default_profile(spec)
    frames: list[RawFrame] = []
    for t in scenario_times(scenario):
        s = _scenario_signals(scenario.segments, profile, t, scenario.first_t_ms + t, spec)
        frames.append(frame_from_signals(s, scenario.w, scenario.h, spec, profile.centre))
    return frames


def _trapezoid(x: float, rise: float, hold: float, fall: float) -> float:
    """x = 0에서 오르기 시작해 rise 동안 1까지 오르고, hold 동안 머물고, fall 동안 0으로 내려오는 모양."""
    if x < 0.0 or x >= rise + hold + fall:
        return 0.0
    if x < rise:
        return x / rise
    if x < rise + hold:
        return 1.0
    return (rise + hold + fall - x) / fall


def _scenario_signals(
    segments: Sequence[Segment], profile: FaceProfile, t: int, t_ms: int, spec: PipelineSpec
) -> FrameSignals:
    """세션 시간 t에서의 원시 신호."""
    level = 0.0
    face = True
    yaw = profile.yaw
    pitch = profile.pitch
    roll = profile.roll
    gaze_h = 0.0
    gaze_v = 0.0
    jaw = profile.jaw_rest
    for seg in segments:
        x = float(t - seg.start_ms)
        if x < 0.0 or x >= seg.duration_ms:
            continue
        if isinstance(seg, Blink):
            value = seg.depth * _trapezoid(x, 0.4 * seg.duration_ms, 0.0, 0.6 * seg.duration_ms)
            level = max(level, value)
        elif isinstance(seg, Closure):
            level = max(level, seg.depth * _trapezoid(x, seg.close_ms, seg.hold_ms, seg.open_ms))
        elif isinstance(seg, Linger):
            hold = seg.duration_ms - 2 * seg.ramp_ms
            level = max(level, seg.level * _trapezoid(x, seg.ramp_ms, hold, seg.ramp_ms))
        elif isinstance(seg, FaceMissing):
            face = False
        elif isinstance(seg, HeadTurn):
            shape = _trapezoid(x, seg.ramp_ms, seg.duration_ms - 2 * seg.ramp_ms, seg.ramp_ms)
            yaw += seg.yaw * shape
            pitch += seg.pitch * shape
            roll += seg.roll * shape
        elif isinstance(seg, LookAside):
            shape = _trapezoid(x, seg.ramp_ms, seg.duration_ms - 2 * seg.ramp_ms, seg.ramp_ms)
            gaze_h += seg.gaze_h * shape
            gaze_v += seg.gaze_v * shape
        elif isinstance(seg, Yawn):
            jaw += seg.jaw * _trapezoid(x, seg.ramp_ms, seg.duration_ms - 2 * seg.ramp_ms, seg.ramp_ms)
        elif isinstance(seg, Nod):
            pitch += seg.pitch * _trapezoid(x, 0.5 * seg.duration_ms, 0.0, 0.5 * seg.duration_ms)
    if not face:
        return _no_face_signals(t_ms)
    ear_l, blink_l = lid_signals(level, profile.e_open_l, profile.b_open_l, spec)
    ear_r, blink_r = lid_signals(level, profile.e_open_r, profile.b_open_r, spec)
    return FrameSignals(
        t_ms=t_ms,
        face=True,
        ear_l=ear_l,
        ear_r=ear_r,
        eye_w_l=profile.eye_w_px,
        eye_w_r=profile.eye_w_px,
        blink_l=blink_l,
        blink_r=blink_r,
        eye_look=_eye_look(profile.eye_look_rest, gaze_h, gaze_v),
        jaw_open=jaw,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def _no_face_signals(t_ms: int) -> FrameSignals:
    return FrameSignals(
        t_ms=t_ms,
        face=False,
        ear_l=0.0,
        ear_r=0.0,
        eye_w_l=0.0,
        eye_w_r=0.0,
        blink_l=0.0,
        blink_r=0.0,
        eye_look=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        jaw_open=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )


def _eye_look(rest: Sequence[float], gaze_h: float, gaze_v: float) -> tuple[float, ...]:
    """쉬는 값에서 눈동자 합성값이 (gaze_h, gaze_v)만큼 변한 eyeLook 8개(EYE_LOOK_ORDER)."""
    values = list(rest)
    if gaze_h > 0.0:  # 피험자의 왼쪽: 오른눈은 코 쪽(In), 왼눈은 관자놀이 쪽(Out)
        values[EL_IN_R] += gaze_h
        values[EL_OUT_L] += gaze_h
    elif gaze_h < 0.0:
        values[EL_OUT_R] -= gaze_h
        values[EL_IN_L] -= gaze_h
    if gaze_v > 0.0:
        values[EL_UP_L] += gaze_v
        values[EL_UP_R] += gaze_v
    elif gaze_v < 0.0:
        values[EL_DOWN_L] -= gaze_v
        values[EL_DOWN_R] -= gaze_v
    return tuple(values)


# ---------------------------------------------------------------------------
# (c) 통계 생성기
# ---------------------------------------------------------------------------

LABELS: tuple[int, ...] = (0, 5, 10)  # 클래스 순서(alert, low_vigilant, drowsy)의 RLDD 라벨
FPS_MIN = 12.0
FPS_MAX = 30.0
# 녹화 해상도 후보(너비, 높이). RLDD의 범위(568x320 ~ 1920x1080)와 세로 영상을 흉내 낸다.
RESOLUTIONS: tuple[tuple[int, int], ...] = ((640, 480), (1280, 720), (1920, 1080), (720, 1280), (480, 640), (568, 320))

# 상태별 졸림 수준 d(0~1)의 세션 평균. 세션마다 표준편차 _LEVEL_SD로 흩어지므로 이웃 상태가 겹친다.
_LEVEL_MEAN: tuple[float, ...] = (0.12, 0.45, 0.78)
_LEVEL_SD = 0.11
_LEVEL_DRIFT_SD = 0.10  # 세션 안에서 d가 천천히 변하는 폭
_LEVEL_DRIFT_TAU_S = 60.0


@dataclass(frozen=True, slots=True)
class SubjectProfile:
    """합성 피험자 한 명의 고정 특성. 세 세션(상태)이 함께 쓴다.

    녹화 조건(fps, 해상도, 얼굴 크기, 프레임 누락률)은 피험자에게 묶여 있고 상태와 무관하다. 그래서 녹화 조건만으로는
    상태를 맞힐 수 없다(PLAN 4.6절: 녹화 조건이 라벨을 드러내면 안 된다).
    """

    subject: int
    e_open_l: float  # 뜬 눈의 EAR
    e_open_r: float
    b_open_l: float  # 뜬 눈의 eyeBlink
    b_open_r: float
    ear_closed_ratio: float  # 완전히 감았을 때의 EAR / 뜬 눈의 EAR
    b_closed: float  # 완전히 감았을 때의 eyeBlink
    jaw_rest: float
    eye_look_rest: tuple[float, ...]
    yaw0: float  # 카메라 거치 각도(도)
    pitch0: float
    roll0: float
    blink_rate_per_min: float  # 각성 상태의 깜빡임 빈도
    blink_duration_ms: float  # 각성 상태의 깜빡임 길이(중앙값)
    fps: float  # 명목 프레임률. 구간마다 이 값과 그 70% 사이에서 변한다(12 fps 아래로는 내려가지 않는다)
    drop_prob: float
    w: int
    h: int
    eye_w_px: float
    centre: tuple[float, float]


def make_subject(subject: int, seed: int) -> SubjectProfile:
    """피험자 번호와 시드로 정해지는 피험자 특성. 같은 (subject, seed)는 언제나 같은 값을 낸다."""
    rng = np.random.default_rng([seed, subject])
    e_open = rng.uniform(0.22, 0.34)
    b_open = rng.uniform(0.08, 0.28)
    w, h = RESOLUTIONS[int(rng.integers(len(RESOLUTIONS)))]
    face_w = rng.uniform(0.25, 0.45) * min(w, h)
    return SubjectProfile(
        subject=subject,
        e_open_l=float(e_open * rng.uniform(0.96, 1.04)),
        e_open_r=float(e_open * rng.uniform(0.96, 1.04)),
        b_open_l=float(b_open + rng.uniform(-0.02, 0.02)),
        b_open_r=float(b_open + rng.uniform(-0.02, 0.02)),
        ear_closed_ratio=float(rng.uniform(0.15, 0.35)),
        b_closed=float(rng.uniform(0.60, 0.85)),
        jaw_rest=float(rng.uniform(0.01, 0.10)),
        eye_look_rest=tuple(float(v) for v in rng.uniform(0.0, 0.12, size=8)),
        yaw0=float(rng.uniform(-12.0, 12.0)),
        pitch0=float(rng.uniform(-10.0, 10.0)),
        roll0=float(rng.uniform(-4.0, 4.0)),
        blink_rate_per_min=float(rng.uniform(10.0, 22.0)),
        blink_duration_ms=float(rng.uniform(200.0, 300.0)),
        fps=float(rng.uniform(17.0, FPS_MAX)),
        drop_prob=float(rng.uniform(0.005, 0.03)),
        w=w,
        h=h,
        eye_w_px=float(face_w / FACE_WIDTH_RATIO),
        centre=(float(rng.uniform(0.42, 0.58)), float(rng.uniform(0.38, 0.52))),
    )


def generate_session(
    profile: SubjectProfile,
    class_index: int,
    duration_ms: int,
    seed: int,
    spec: PipelineSpec,
    part: int = 0,
) -> Iterator[RawFrame]:
    """피험자 한 명의 한 상태 세션을 프레임 하나씩 낸다. 첫 프레임의 t_ms는 1이다(추출 결과와 같다).

    세션은 (seed, 피험자, 상태, part)로 정해진다. part는 나뉜 영상의 부분 번호이고 부분마다 독립된 세션이다.

    상태가 바꾸는 것: 깜빡임 빈도, 깜빡임 길이, 긴 감김의 비율, 눈꺼풀 처짐, 하품, 끄덕임, 머리 움직임의 크기.
    세션의 졸림 수준 d는 상태별 평균(0.12, 0.45, 0.78) 주위에서 뽑고 세션 안에서도 천천히 변한다. 위의 양은 모두
    d의 함수이므로, 상태 차이는 평균적으로는 뚜렷하지만 세션 하나만 보면 이웃 상태와 겹친다.
    상태가 바꾸지 않는 것: 프레임 간격(12~30 fps에서 구간마다 변함), 프레임 누락, 해상도, 얼굴 크기,
    얼굴이 검출되지 않는 구간. 세션마다 달라지는 거치 각도와 뜬 눈 수준의 작은 차이도 상태와 무관하게 뽑는다.

    가정: 상태별 수치는 문헌의 방향(졸리면 깜빡임이 길어지고 긴 감김, 하품, 끄덕임이 늘어난다)만 따른 설계값이다.
    실제 사람의 분포를 재현하지 않는다. 이 데이터의 점수는 파이프라인 점검용이다.
    """
    if class_index not in range(len(CLASS_ORDER)):
        raise ValueError(f"class_index must be 0, 1 or 2, got {class_index}")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0")
    rng = np.random.default_rng([seed, profile.subject, class_index, part])
    duration_s = duration_ms / 1000.0

    # 녹화: 상태와 무관하다.
    times = _session_times(rng, duration_ms, profile.fps, profile.drop_prob)
    n = len(times)
    seconds = times / 1000.0
    face = _face_present(rng, times, duration_s)
    session_e = rng.normal(1.0, 0.03)  # 세션마다 조금 다른 뜬 눈 수준(조명, 거리)
    session_yaw = profile.yaw0 + rng.normal(0.0, 3.0)  # 세션마다 다시 놓은 카메라
    session_pitch = profile.pitch0 + rng.normal(0.0, 3.0)
    session_roll = profile.roll0 + rng.normal(0.0, 1.0)

    # 졸림 수준 d(t)
    d0 = float(np.clip(rng.normal(_LEVEL_MEAN[class_index], _LEVEL_SD), 0.02, 0.98))
    grid_s = np.arange(0.0, duration_s + 1.0, 1.0)
    drift = _ou(rng, len(grid_s), 1.0, _LEVEL_DRIFT_TAU_S)
    d_grid = np.clip(d0 + _LEVEL_DRIFT_SD * drift, 0.0, 1.0)
    d = np.interp(seconds, grid_s, d_grid)

    # 눈꺼풀 수준: 처짐 위에 깜빡임과 긴 감김을 얹는다.
    droop_noise = np.interp(seconds, grid_s, _ou(rng, len(grid_s), 1.0, 20.0))
    level = np.clip(0.30 * d**1.5 + 0.02 * droop_noise, 0.0, 0.45)
    for start_ms, close_ms, hold_ms, open_ms, depth in _lid_events(rng, duration_ms, profile, grid_s, d_grid):
        lo, hi, shape = _pulse(times, start_ms, close_ms, hold_ms, open_ms)
        np.maximum(level[lo:hi], depth * shape, out=level[lo:hi])

    # 머리: 느린 흔들림(크기가 d에 따라 커진다) + 끄덕임 + 가끔 옆을 봄
    head_grid = np.arange(0.0, duration_s + 0.2, 0.2)
    yaw = session_yaw + (1.5 + 2.5 * d) * np.interp(seconds, head_grid, _ou(rng, len(head_grid), 0.2, 4.0))
    pitch = session_pitch + (1.0 + 3.0 * d) * np.interp(seconds, head_grid, _ou(rng, len(head_grid), 0.2, 4.0))
    roll = session_roll + (0.5 + 1.0 * d) * np.interp(seconds, head_grid, _ou(rng, len(head_grid), 0.2, 4.0))
    for start_ms in _poisson_times(rng, duration_ms, lambda dd: 0.02 + 0.6 * dd * dd, grid_s, d_grid):
        length_ms = rng.uniform(500.0, 1500.0)
        lo, hi, shape = _pulse(times, start_ms, 0.6 * length_ms, 0.0, 0.4 * length_ms)
        pitch[lo:hi] += rng.uniform(13.0, 25.0) * shape
    for start_ms in _poisson_times(rng, duration_ms, lambda dd: 0.3, grid_s, d_grid):
        lo, hi, shape = _pulse(times, start_ms, 300.0, rng.uniform(500.0, 2000.0), 300.0)
        yaw[lo:hi] += rng.choice((-1.0, 1.0)) * rng.uniform(25.0, 50.0) * shape

    # 턱: 쉬는 값 + 하품
    jaw = profile.jaw_rest + rng.normal(0.0, 0.01, size=n)
    for start_ms in _poisson_times(rng, duration_ms, lambda dd: 0.02 + 0.45 * dd, grid_s, d_grid):
        lo, hi, shape = _pulse(times, start_ms, 800.0, rng.uniform(1500.0, 4500.0), 800.0)
        jaw[lo:hi] += rng.uniform(0.50, 0.85) * shape
    jaw = np.clip(jaw, 0.0, 1.0)

    # 눈 신호: 눈마다 따로 잡음을 넣는다.
    ear_span = 1.0 - profile.ear_closed_ratio
    ear_l = profile.e_open_l * session_e * (1.0 - ear_span * level) * (1.0 + rng.normal(0.0, 0.025, size=n))
    ear_r = profile.e_open_r * session_e * (1.0 - ear_span * level) * (1.0 + rng.normal(0.0, 0.025, size=n))
    blink_l = profile.b_open_l + (profile.b_closed - profile.b_open_l) * level + rng.normal(0.0, 0.015, size=n)
    blink_r = profile.b_open_r + (profile.b_closed - profile.b_open_r) * level + rng.normal(0.0, 0.015, size=n)
    ear_l = np.clip(ear_l, 0.01, None)
    ear_r = np.clip(ear_r, 0.01, None)
    blink_l = np.clip(blink_l, 0.0, 1.0)
    blink_r = np.clip(blink_r, 0.0, 1.0)
    eye_look = np.clip(np.asarray(profile.eye_look_rest) + rng.normal(0.0, 0.01, size=(n, 8)), 0.0, 1.0)

    t_list = (times + 1).tolist()
    face_list = face.tolist()
    columns = [a.tolist() for a in (ear_l, ear_r, blink_l, blink_r, jaw, yaw, pitch, roll)]
    eye_look_rows = eye_look.tolist()
    for i in range(n):
        if not face_list[i]:
            yield no_face_frame(t_list[i], profile.w, profile.h)
            continue
        s = FrameSignals(
            t_ms=t_list[i],
            face=True,
            ear_l=columns[0][i],
            ear_r=columns[1][i],
            eye_w_l=profile.eye_w_px,
            eye_w_r=profile.eye_w_px,
            blink_l=columns[2][i],
            blink_r=columns[3][i],
            eye_look=tuple(eye_look_rows[i]),
            jaw_open=columns[4][i],
            yaw=columns[5][i],
            pitch=columns[6][i],
            roll=columns[7][i],
        )
        yield frame_from_signals(s, profile.w, profile.h, spec, profile.centre)


def _session_times(rng: np.random.Generator, duration_ms: int, fps: float, drop_prob: float) -> np.ndarray:
    """프레임 시각(세션 시간 ms, int64, 엄격히 증가, 첫 값 0).

    가변 프레임률: 10~60초 구간마다 프레임률을 [max(12, 0.7·fps), fps]에서 새로 뽑고, 프레임 간격마다 ±8%쯤 흔든다.
    간격은 12~30 fps의 범위(33.3~83.3 ms)로 자른다. 그 뒤 첫 프레임을 뺀 각 프레임을 drop_prob로 버린다.
    """
    lowest = max(FPS_MIN, 0.7 * fps)
    intervals: list[np.ndarray] = []
    covered_ms = 0.0
    while covered_ms < duration_ms:
        rate = rng.uniform(lowest, max(fps, lowest))
        count = max(1, int(rng.uniform(10.0, 60.0) * rate))
        chunk = (1000.0 / rate) * (1.0 + 0.08 * np.clip(rng.standard_normal(count), -2.0, 2.0))
        chunk = np.clip(chunk, 1000.0 / FPS_MAX, 1000.0 / FPS_MIN)
        intervals.append(chunk)
        covered_ms += float(chunk.sum())
    stamps = np.concatenate([[0.0], np.cumsum(np.concatenate(intervals))])
    times = np.floor(stamps + 0.5).astype(np.int64)
    times = times[times < duration_ms]
    keep = rng.random(len(times)) >= drop_prob
    keep[0] = True
    return times[keep]


def _face_present(rng: np.random.Generator, times: np.ndarray, duration_s: float) -> np.ndarray:
    """프레임마다 얼굴이 검출됐는지. 10분에 평균 두 번쯤 0.3~12초 동안 얼굴이 사라지고, 낱개 프레임도 가끔 빠진다."""
    face = rng.random(len(times)) >= 0.002
    for _ in range(int(rng.poisson(0.2 * duration_s / 60.0))):
        start_ms = rng.uniform(0.0, duration_s) * 1000.0
        length_ms = float(np.clip(rng.lognormal(math.log(1500.0), 0.8), 300.0, 12000.0))
        lo = int(np.searchsorted(times, start_ms))
        hi = int(np.searchsorted(times, start_ms + length_ms))
        face[lo:hi] = False
    return face


def _ou(rng: np.random.Generator, count: int, dt_s: float, tau_s: float) -> np.ndarray:
    """평균 0, 분산 1인 Ornstein-Uhlenbeck 과정을 dt_s 간격으로 count개 뽑는다(상관 시간 tau_s)."""
    keep = math.exp(-dt_s / tau_s)
    scale = math.sqrt(1.0 - keep * keep)
    noise = rng.standard_normal(count).tolist()
    values = [noise[0]]
    for i in range(1, count):
        values.append(keep * values[-1] + scale * noise[i])
    return np.asarray(values)


def _pulse(
    times: np.ndarray, start_ms: float, rise_ms: float, hold_ms: float, fall_ms: float
) -> tuple[int, int, np.ndarray]:
    """사다리꼴 모양(0 → 1 → 0)이 걸치는 프레임 범위 [lo, hi)와 그 프레임들에서의 값."""
    lo = int(np.searchsorted(times, start_ms))
    hi = int(np.searchsorted(times, start_ms + rise_ms + hold_ms + fall_ms))
    x = times[lo:hi] - start_ms
    shape = np.minimum(x / rise_ms, (rise_ms + hold_ms + fall_ms - x) / fall_ms)
    return lo, hi, np.clip(shape, 0.0, 1.0)


def _poisson_times(
    rng: np.random.Generator,
    duration_ms: int,
    rate_per_min: Callable[[float], float],
    grid_s: np.ndarray,
    d_grid: np.ndarray,
) -> list[float]:
    """분당 발생률이 rate_per_min(d(t))인 포아송 과정의 발생 시각(ms). 솎아내기로 뽑는다."""
    peak = max(float(rate_per_min(0.0)), float(rate_per_min(1.0)))  # 쓰는 발생률은 모두 d에 대해 단조 증가다
    count = int(rng.poisson(peak * duration_ms / 60000.0))
    candidates = np.sort(rng.uniform(0.0, float(duration_ms), size=count))
    accept = rng.random(count)
    out: list[float] = []
    for t_ms, u in zip(candidates.tolist(), accept.tolist()):
        level = float(np.interp(t_ms / 1000.0, grid_s, d_grid))
        if u * peak < rate_per_min(level):
            out.append(t_ms)
    return out


def _lid_events(
    rng: np.random.Generator,
    duration_ms: int,
    profile: SubjectProfile,
    grid_s: np.ndarray,
    d_grid: np.ndarray,
) -> list[tuple[float, float, float, float, float]]:
    """깜빡임과 긴 감김의 목록: (시작 ms, 감는 ms, 머무는 ms, 뜨는 ms, 깊이). 서로 겹치지 않는다.

    졸림 수준 d에서: 빈도 = 피험자 빈도·(1 + 0.6d), 길이 중앙값 = 피험자 길이·(1 + 0.9d),
    긴 감김(0.35~3.5초 머묾, 중앙값 0.9초)일 확률 = 0.004 + 0.22d².
    """
    base = profile.blink_rate_per_min
    starts = _poisson_times(rng, duration_ms, lambda dd: base * (1.0 + 0.6 * dd), grid_s, d_grid)
    events: list[tuple[float, float, float, float, float]] = []
    busy_until = 0.0
    for start_ms in starts:
        level = float(np.interp(start_ms / 1000.0, grid_s, d_grid))
        # 난수는 버려지는 후보에서도 같은 수만큼 뽑는다(앞의 후보가 뒤 후보의 난수를 밀지 않게).
        is_long = rng.random() < 0.004 + 0.22 * level * level
        median_ms = profile.blink_duration_ms * (1.0 + 0.9 * level)
        length_ms = float(np.clip(median_ms * rng.lognormal(0.0, 0.25), 90.0, 700.0))
        hold_ms = float(np.clip(rng.lognormal(math.log(900.0), 0.6), 350.0, 3500.0))
        depth = 1.0 if rng.random() < 0.85 else float(rng.uniform(0.55, 0.9))
        if start_ms < busy_until:
            continue
        if is_long:
            event = (start_ms, 120.0, hold_ms, 200.0, 1.0)
        else:
            event = (start_ms, 0.30 * length_ms, 0.25 * length_ms, 0.45 * length_ms, depth)
        events.append(event)
        busy_until = start_ms + event[1] + event[2] + event[3] + 150.0
    return events


# ---------------------------------------------------------------------------
# (d) 합성 데이터셋 쓰기와 명령
# ---------------------------------------------------------------------------

SPLIT_SUBJECTS_RLDD: tuple[int, ...] = (32, 49)  # RLDD에서 drowsy 영상이 10_1, 10_2로 나뉜 피험자
INFO_FILE = "synth_info.json"


def split_subjects(n_subjects: int) -> tuple[int, ...]:
    """drowsy 세션을 `10_1`, `10_2`로 나눠 쓰는 피험자.

    가정: 피험자가 49명 이상이면 RLDD와 같은 32번과 49번이다. 그보다 적으면 가운데 번호와 마지막 번호를 쓴다
    (피험자가 한 명이면 그 한 명).
    """
    if n_subjects >= max(SPLIT_SUBJECTS_RLDD):
        return SPLIT_SUBJECTS_RLDD
    return tuple(sorted({max(1, n_subjects // 2), n_subjects}))


@dataclass(frozen=True, slots=True)
class SessionJob:
    """합성 세션 하나(= parquet 파일 하나)를 만드는 데 필요한 것."""

    subject: int
    class_index: int
    part: int  # 0이면 나뉘지 않은 영상, 1 이상이면 나뉜 영상의 부분 번호
    rel_path: str  # 출력 폴더 기준 상대 경로(슬래시 구분)
    duration_ms: int
    seed: int
    drop_sim: DropSimConfig | None = None  # 주면 흐름 제한기가 버렸을 프레임을 빼고 쓴다


def session_jobs(
    n_subjects: int, minutes: float, seed: int, drop_sim: DropSimConfig | None = None
) -> list[SessionJob]:
    """합성 데이터셋의 세션 목록. 배치는 `Fold{k}_part{p}/{NN}/{라벨}.parquet`이다.

    - fold k는 공식 대응(`dataset.official_fold`), part p는 fold 안의 앞 6명이 1, 뒤 6명이 2다.
    - `split_subjects`의 피험자는 drowsy 세션이 `10_1`, `10_2` 두 파일이다.
      가정: 나뉜 부분은 각각 다른 세션과 같은 길이의 독립된 녹화다(부분마다 창이 나올 수 있는 길이가 되게).
    """
    from drowsy.dataset import SUBJECTS_PER_FOLD, official_fold

    if n_subjects < 1:
        raise ValueError(f"--subjects must be >= 1, got {n_subjects}")
    if not minutes > 0.0:
        raise ValueError(f"--minutes must be > 0, got {minutes}")
    duration_ms = int(math.floor(minutes * 60000.0 + 0.5))
    split = split_subjects(n_subjects)
    jobs: list[SessionJob] = []
    for subject in range(1, n_subjects + 1):
        part_folder = 1 if (subject - 1) % SUBJECTS_PER_FOLD < SUBJECTS_PER_FOLD // 2 else 2
        folder = f"Fold{official_fold(subject)}_part{part_folder}/{subject:02d}"
        for class_index, label in enumerate(LABELS):
            if class_index == CLASS_DROWSY and subject in split:
                for part in (1, 2):
                    rel_path = f"{folder}/{label}_{part}.parquet"
                    jobs.append(SessionJob(subject, class_index, part, rel_path, duration_ms, seed, drop_sim))
            else:
                rel_path = f"{folder}/{label}.parquet"
                jobs.append(SessionJob(subject, class_index, 0, rel_path, duration_ms, seed, drop_sim))
    return jobs


def simulate_device_drops(frames: Iterator[RawFrame], drop_sim: DropSimConfig) -> Iterator[RawFrame]:
    """기기의 흐름 제한기가 처리했을 프레임만 낸다. 추출의 누락 모사와 같은 제한기와 같은 지연 수열을 쓴다.

    가정: 합성 프레임에는 MediaPipe의 시간 평활이 없으므로 프레임을 빼는 것이 누락 모사의 전부다
    (실제 추출에서는 빠진 프레임 때문에 평활 결과도 달라진다).
    """
    from drowsy.extract import FlowLimiter, drop_sim_latencies

    limiter: FlowLimiter[RawFrame] = FlowLimiter()
    latencies = drop_sim_latencies(drop_sim)
    for frame in frames:
        yield from limiter.offer(frame.t_ms, next(latencies), frame)
    yield from limiter.flush()


def write_session(job: SessionJob, out_dir: Path, spec: PipelineSpec) -> int:
    """세션 하나를 추출 결과와 같은 형식의 parquet으로 쓰고 프레임 수를 돌려준다(`extract.write_raw_frames`를 쓴다)."""
    from drowsy.extract import write_raw_frames

    profile = make_subject(job.subject, job.seed)
    metadata = {
        "source": "synthetic (drowsy.synth)",
        "synthetic": True,
        "seed": job.seed,
        "subject": job.subject,
        "label": LABELS[job.class_index],
        "part": job.part,
        "duration_ms": job.duration_ms,
        "nominal_fps": profile.fps,
        "drop_sim": None,
    }
    frames = generate_session(profile, job.class_index, job.duration_ms, job.seed, spec, job.part)
    if job.drop_sim is not None:
        # 추출 결과의 메타데이터와 같은 모양으로 적는다(저장 폴더 이름은 결과를 바꾸지 않으므로 뺀다).
        metadata["drop_sim"] = {
            "latency_ms_mean": job.drop_sim.latency_ms_mean,
            "latency_ms_std": job.drop_sim.latency_ms_std,
            "seed": job.drop_sim.seed,
        }
        frames = simulate_device_drops(frames, job.drop_sim)
    return write_raw_frames(frames, Path(out_dir) / job.rel_path, spec, metadata)


def _write_session_job(args: tuple[SessionJob, Path, PipelineSpec]) -> tuple[str, int]:
    job, out_dir, spec = args
    return job.rel_path, write_session(job, out_dir, spec)


def write_dataset(
    out_dir: Path,
    n_subjects: int,
    minutes: float,
    seed: int,
    spec: PipelineSpec,
    workers: int = 1,
    drop_sim: DropSimConfig | None = None,
) -> dict[str, int]:
    """합성 데이터셋 전체를 쓴다. 상대 경로 → 프레임 수를 돌려준다.

    세션마다 독립된 난수열을 쓰므로 workers 수와 처리 순서는 결과를 바꾸지 않는다.
    drop_sim을 주면 같은 세션에서 흐름 제한기가 버렸을 프레임을 뺀 것을 쓴다(`simulate_device_drops`).
    출력 폴더에 parquet이 이미 있으면 ValueError다(이전 실행의 파일이 섞이면 목록이 틀어진다).
    """
    out_dir = Path(out_dir)
    if workers < 1:
        raise ValueError(f"--workers must be >= 1, got {workers}")
    jobs = session_jobs(n_subjects, minutes, seed, drop_sim)
    if out_dir.exists() and any(out_dir.rglob("*.parquet")):
        raise ValueError(f"{out_dir} already holds parquet files; remove the directory or choose another --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(job, out_dir, spec) for job in jobs]
    if workers == 1:
        results = [_write_session_job(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = list(pool.map(_write_session_job, tasks))
    info = {
        "note": "SYNTHETIC data from drowsy.synth. Pipeline check only; scores on it are not performance results.",
        "subjects": n_subjects,
        "minutes": minutes,
        "seed": seed,
        "split_subjects": list(split_subjects(n_subjects)),
        "files": len(results),
        "drop_sim_latency_ms_mean": None if drop_sim is None else drop_sim.latency_ms_mean,
    }
    (out_dir / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return dict(results)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m drowsy.synth`: 합성 데이터셋을 쓴다. 종료 코드: 0 성공, 2 시작할 수 없음."""
    from drowsy.config import load_config
    from drowsy.spec import load_feature_spec

    parser = argparse.ArgumentParser(
        prog="python -m drowsy.synth",
        description="Write a synthetic dataset as raw parquet files in the RLDD layout (pipeline check only).",
    )
    parser.add_argument("--config", default="config/default.yaml", help="experiment config YAML")
    parser.add_argument("--out", type=Path, required=True, help="output directory, e.g. data/synth")
    parser.add_argument("--subjects", type=int, required=True, help="number of synthetic subjects")
    parser.add_argument("--minutes", type=float, default=10.0, help="session length in minutes (default 10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None, help="parallel sessions (default: extraction.workers)")
    parser.add_argument(
        "--fps-sim",
        type=int,
        default=None,
        metavar="FPS",
        help="write the same sessions without the frames a device flow limiter would drop at about this frame rate",
    )
    args = parser.parse_args(argv)
    try:
        from drowsy.extract import ExtractError, drop_sim_for_fps

        cfg = load_config(args.config)
        spec = load_feature_spec(cfg.paths.feature_spec).pipeline
        workers = cfg.extraction.workers if args.workers is None else args.workers
        drop_sim = None if args.fps_sim is None else drop_sim_for_fps(cfg.extraction.drop_sim, args.fps_sim)
        started = time.perf_counter()
        written = write_dataset(args.out, args.subjects, args.minutes, args.seed, spec, workers, drop_sim)
    except (ValueError, OSError, ExtractError) as exc:  # SpecError도 ValueError다
        print(f"error: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - started
    mode = "" if args.fps_sim is None else f", device frame drops simulated at about {args.fps_sim} fps"
    print(
        f"SYNTHETIC data (pipeline check only): wrote {len(written)} sessions, {sum(written.values())} frames, "
        f"{args.subjects} subjects x {args.minutes:g} min, seed {args.seed}{mode} -> {args.out} in {elapsed:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
