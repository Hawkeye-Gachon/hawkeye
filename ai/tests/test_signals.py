"""`drowsy.signals` 단위 테스트 (PLAN 2.1절, 2.2절, 4.1절, 10.1절 "신호").

합성 랜드마크만 쓴다. 실제 얼굴 이미지는 없다. 표준 얼굴 모델 좌표는 MediaPipe 공개 파일
`canonical_face_model.obj`의 정점 12개(EAR에 쓰는 점)의 x, y 값이다.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from drowsy import calibrate, signals
from drowsy.spec import PipelineSpec
from drowsy.types import (
    BASELINE_OK,
    EL_DOWN_L,
    EL_DOWN_R,
    EL_IN_L,
    EL_IN_R,
    EL_OUT_L,
    EL_OUT_R,
    EL_UP_L,
    EL_UP_R,
    EYE_LOOK_ORDER,
    Baseline,
    RawFrame,
)

ZERO8: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# 공식 기대 출력 파일의 변환 행렬(열 우선 packed_data). 기대 각도는 yaw 2.24, pitch 6.32, roll 0.99도.
OFFICIAL_MATRIX: tuple[float, ...] = (
    0.9995292,
    0.01715696,
    -0.03715533,
    0.0,
    -0.01294756,
    0.9937692,
    0.11070588,
    0.0,
    0.038823195,
    -0.1101321,
    0.99315894,
    0.0,
    -0.3691378,
    22.75809,
    -65.765925,
    1.0,
)

# canonical_face_model.obj의 정점 (x, y), 단위 cm. +x는 피험자 왼쪽, +y는 위.
CANONICAL_XY: dict[int, tuple[float, float]] = {
    33: (-4.445859, 2.663991),
    160: (-3.670075, 2.927714),
    158: (-2.724032, 2.961810),
    133: (-1.856432, 2.585245),
    153: (-2.724032, 2.315802),
    144: (-3.670075, 2.360153),
    362: (1.856432, 2.585245),
    385: (2.724032, 2.961810),
    387: (3.670075, 2.927714),
    263: (4.445859, 2.663991),
    373: (3.670075, 2.360153),
    380: (2.724032, 2.315802),
}


# ---------------------------------------------------------------------------
# 합성 프레임 도우미
# ---------------------------------------------------------------------------


def colmajor_from_rotation(
    r: tuple[float, ...], translation: tuple[float, float, float] = (0.0, 0.0, -60.0)
) -> tuple[float, ...]:
    """행 우선 3x3 회전과 평행이동으로 열 우선 4x4(16개 값)를 만든다. m[4*j + i] = M[i][j]."""
    m = [0.0] * 16
    for i in range(3):
        for j in range(3):
            m[4 * j + i] = r[3 * i + j]
    m[12] = translation[0]
    m[13] = translation[1]
    m[14] = translation[2]
    m[15] = 1.0
    return tuple(m)


def eye_points_px(cx: float, cy: float, width_px: float, ear: float) -> list[tuple[float, float]]:
    """EAR이 `ear`, 폭이 `width_px`인 눈의 여섯 점(픽셀). 순서: 왼쪽 꼬리, 위 왼쪽, 위 오른쪽, 오른쪽 꼬리,
    아래 오른쪽, 아래 왼쪽. 위아래 점이 같은 x에 있으므로 EAR = (ear*W + ear*W) / (2W) = ear다.

    영상에서 두 눈 모두 p1이 왼쪽(작은 x), p4가 오른쪽이다(R 집합 p1 = 33은 바깥 꼬리, L 집합 p1 = 362는 안쪽 꼬리).
    """
    half_w = width_px / 2.0
    half_h = ear * width_px / 2.0
    return [
        (cx - half_w, cy),
        (cx - width_px / 6.0, cy - half_h),
        (cx + width_px / 6.0, cy - half_h),
        (cx + half_w, cy),
        (cx + width_px / 6.0, cy + half_h),
        (cx - width_px / 6.0, cy + half_h),
    ]


def make_frame(
    spec: PipelineSpec,
    *,
    t_ms: int = 1,
    w: int = 640,
    h: int = 480,
    ear_l: float = 0.30,
    ear_r: float = 0.30,
    eye_w_l: float = 40.0,
    eye_w_r: float = 40.0,
    blink_l: float = 0.2,
    blink_r: float = 0.2,
    eye_look: tuple[float, ...] = ZERO8,
    jaw_open: float = 0.05,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    iris_shift_px: tuple[float, float] = (0.0, 0.0),
) -> RawFrame:
    """원하는 눈별 EAR·눈 폭, 블렌드셰이프, 오일러 각으로 얼굴 프레임을 만든다.

    R 눈(피험자 오른쪽)은 영상 왼쪽(x = 0.4w), L 눈은 영상 오른쪽(x = 0.6w)에 둔다(비반전 영상).
    홍채 중심은 눈 중심에서 `iris_shift_px`만큼 옮긴 곳에 둔다. EAR에 쓰지 않는 윤곽 점은 눈 중심에 둔다.
    """
    lms = spec.landmarks
    centre_r = (0.4 * w, 0.45 * h)
    centre_l = (0.6 * w, 0.45 * h)
    px: dict[int, tuple[float, float]] = {}
    for index in lms.contour_r:
        px[index] = centre_r
    for index in lms.contour_l:
        px[index] = centre_l
    for index in lms.iris_r:
        px[index] = (centre_r[0] + iris_shift_px[0], centre_r[1] + iris_shift_px[1])
    for index in lms.iris_l:
        px[index] = (centre_l[0] + iris_shift_px[0], centre_l[1] + iris_shift_px[1])
    px[lms.face_width[0]] = (0.3 * w, 0.5 * h)
    px[lms.face_width[1]] = (0.7 * w, 0.5 * h)
    for index, point in zip(lms.ear_r, eye_points_px(centre_r[0], centre_r[1], eye_w_r, ear_r)):
        px[index] = point
    for index, point in zip(lms.ear_l, eye_points_px(centre_l[0], centre_l[1], eye_w_l, ear_l)):
        px[index] = point

    lm = {index: (point[0] / w, point[1] / h) for index, point in px.items()}
    names = spec.blendshapes
    bs = {names.eye_blink_l: blink_l, names.eye_blink_r: blink_r, names.jaw_open: jaw_open}
    for name, value in zip(names.eye_look, eye_look):
        bs[name] = value
    m = colmajor_from_rotation(signals.rotation_from_euler(yaw, pitch, roll))
    return RawFrame(t_ms=t_ms, face=True, w=w, h=h, lm=lm, bs=bs, m=m)


def mirror_frame(spec: PipelineSpec, frame: RawFrame) -> RawFrame:
    """좌우 반전한 영상에서 MediaPipe가 낼 프레임. x -> 1 − x, L/R 집합과 Left/Right 블렌드셰이프를 맞바꾸고,
    행렬은 S·M·S (S = diag(−1, 1, 1, 1))로 바꾼다."""
    lms = spec.landmarks
    pairs: list[tuple[int, int]] = list(zip(lms.contour_r, lms.contour_l))
    # 홍채 순서는 중심, +x, 위, −x, 아래. 반전하면 +x와 −x가 바뀐다.
    for i_r, i_l in ((0, 0), (1, 3), (2, 2), (3, 1), (4, 4)):
        pairs.append((lms.iris_r[i_r], lms.iris_l[i_l]))
    pairs.append((lms.face_width[0], lms.face_width[1]))
    lm: dict[int, tuple[float, float]] = {}
    for a, b in pairs:
        lm[a] = (1.0 - frame.lm[b][0], frame.lm[b][1])
        lm[b] = (1.0 - frame.lm[a][0], frame.lm[a][1])

    names = spec.blendshapes
    bs = {
        names.eye_blink_l: frame.bs[names.eye_blink_r],
        names.eye_blink_r: frame.bs[names.eye_blink_l],
        names.jaw_open: frame.bs[names.jaw_open],
    }
    for i in range(4):
        left_name = names.eye_look[i]
        right_name = names.eye_look[i + 4]
        bs[left_name] = frame.bs[right_name]
        bs[right_name] = frame.bs[left_name]

    m = list(frame.m)
    for i in range(4):
        for j in range(4):
            if (i == 0) != (j == 0):
                m[4 * j + i] = -m[4 * j + i]
    return RawFrame(t_ms=frame.t_ms, face=True, w=frame.w, h=frame.h, lm=lm, bs=bs, m=tuple(m))


def matmul3(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    out: list[float] = []
    for i in range(3):
        for j in range(3):
            out.append(sum(a[3 * i + k] * b[3 * k + j] for k in range(3)))
    return tuple(out)


# ---------------------------------------------------------------------------
# EAR
# ---------------------------------------------------------------------------


def test_ear_hand_computed() -> None:
    # |p2−p6| = |p3−p5| = 2, |p1−p4| = 4 -> (2 + 2) / (2*4) = 0.5
    pts = [(0.0, 0.0), (1.0, -1.0), (3.0, -1.0), (4.0, 0.0), (3.0, 1.0), (1.0, 1.0)]
    assert signals.eye_aspect_ratio(pts) == 0.5
    # 3-4-5 삼각형: |p2−p6| = 5, |p3−p5| = 5, |p1−p4| = 10 -> 10 / 20 = 0.5
    pts = [(0.0, 0.0), (2.0, -2.0), (7.0, -2.0), (10.0, 0.0), (10.0, 2.0), (5.0, 2.0)]
    assert signals.eye_aspect_ratio(pts) == 0.5


def test_ear_is_rotation_and_scale_invariant() -> None:
    pts = eye_points_px(100.0, 50.0, 40.0, 0.28)
    assert signals.eye_aspect_ratio(pts) == pytest.approx(0.28, abs=1e-12)
    cos_a = math.cos(math.radians(25.0))
    sin_a = math.sin(math.radians(25.0))
    rotated = [(3.0 * (x * cos_a - y * sin_a), 3.0 * (x * sin_a + y * cos_a)) for x, y in pts]
    assert signals.eye_aspect_ratio(rotated) == pytest.approx(0.28, abs=1e-12)


def test_ear_needs_six_points() -> None:
    with pytest.raises(ValueError):
        signals.eye_aspect_ratio([(0.0, 0.0), (1.0, 1.0)])


def test_ear_on_canonical_face_model(pipeline_spec: PipelineSpec) -> None:
    # 손 계산(R 집합): |160−144| = 2.927714 − 2.360153 = 0.567561, |158−153| = 2.961810 − 2.315802 = 0.646008,
    # |33−133| = hypot(2.589427, 0.078746) = 2.590624 -> (0.567561 + 0.646008) / (2*2.590624) = 0.234223
    expected = 0.234223
    lms = pipeline_spec.landmarks
    assert signals.eye_aspect_ratio([CANONICAL_XY[i] for i in lms.ear_r]) == pytest.approx(expected, abs=1e-6)
    assert signals.eye_aspect_ratio([CANONICAL_XY[i] for i in lms.ear_l]) == pytest.approx(expected, abs=1e-6)

    # 같은 기하를 640x480 프레임에 정사영(1 cm = 20 px)해서 compute로 계산해도 같다. 눈 폭은 2.590624*20 px.
    w, h = 640, 480
    frame = make_frame(pipeline_spec, w=w, h=h)
    lm = dict(frame.lm)
    for index, (x_cm, y_cm) in CANONICAL_XY.items():
        lm[index] = ((320.0 + 20.0 * x_cm) / w, (240.0 - 20.0 * y_cm) / h)
    s = signals.compute(RawFrame(t_ms=1, face=True, w=w, h=h, lm=lm, bs=frame.bs, m=frame.m), pipeline_spec)
    assert s.ear_r == pytest.approx(expected, abs=1e-6)
    assert s.ear_l == pytest.approx(expected, abs=1e-6)
    assert s.eye_w_r == pytest.approx(2.590624 * 20.0, abs=1e-4)
    assert s.eye_w_l == pytest.approx(2.590624 * 20.0, abs=1e-4)


def test_ear_on_non_square_frame(pipeline_spec: PipelineSpec) -> None:
    frame = make_frame(pipeline_spec, w=1280, h=720, ear_l=0.30, ear_r=0.24, eye_w_l=48.0, eye_w_r=44.0)
    s = signals.compute(frame, pipeline_spec)
    assert s.ear_l == pytest.approx(0.30, abs=1e-9)
    assert s.ear_r == pytest.approx(0.24, abs=1e-9)
    assert s.eye_w_l == pytest.approx(48.0, abs=1e-9)
    assert s.eye_w_r == pytest.approx(44.0, abs=1e-9)

    # 같은 정규화 좌표를 720x720으로 읽으면(= 종횡비를 무시하면) 가로만 720/1280로 줄어 EAR이 1280/720배가 된다.
    # 픽셀 좌표를 써야 하는 이유다.
    squashed = RawFrame(t_ms=1, face=True, w=720, h=720, lm=frame.lm, bs=frame.bs, m=frame.m)
    s_wrong = signals.compute(squashed, pipeline_spec)
    assert s_wrong.ear_l == pytest.approx(0.30 * 1280.0 / 720.0, abs=1e-9)
    assert s_wrong.eye_w_l == pytest.approx(48.0 * 720.0 / 1280.0, abs=1e-9)


def test_zero_width_eye(pipeline_spec: PipelineSpec) -> None:
    point = (5.0, 5.0)
    assert signals.eye_aspect_ratio([point, (5.0, 4.0), (5.0, 4.0), point, (5.0, 6.0), (5.0, 6.0)]) == 0.0

    frame = make_frame(pipeline_spec, eye_w_r=0.0, ear_r=0.3, ear_l=0.27)
    s = signals.compute(frame, pipeline_spec)
    # 쓸 수 없는 눈: EAR은 0.0, 홍채 비율은 None, 눈 폭은 측정값 그대로 -> 호출자가 폭으로 가려낸다.
    assert s.eye_w_r == 0.0
    assert s.ear_r == 0.0
    assert s.iris_h_r is None
    assert s.iris_v_r is None
    # 다른 눈과 나머지 신호는 그대로 계산한다.
    assert s.ear_l == pytest.approx(0.27, abs=1e-9)
    assert s.eye_w_l == pytest.approx(40.0, abs=1e-9)
    assert s.iris_h_l == pytest.approx(0.5, abs=1e-9)
    assert s.blink_r == 0.2


def test_eye_narrower_than_six_pixels_is_unusable(pipeline_spec: PipelineSpec) -> None:
    assert pipeline_spec.signals.min_eye_width_px == 6.0
    narrow = signals.compute(make_frame(pipeline_spec, eye_w_l=5.5, ear_l=0.3), pipeline_spec)
    assert narrow.eye_w_l == pytest.approx(5.5, abs=1e-9)
    assert narrow.ear_l == 0.0
    assert narrow.iris_h_l is None
    wide = signals.compute(make_frame(pipeline_spec, eye_w_l=6.5, ear_l=0.3), pipeline_spec)
    assert wide.eye_w_l == pytest.approx(6.5, abs=1e-9)
    assert wide.ear_l == pytest.approx(0.3, abs=1e-9)
    assert wide.iris_h_l is not None


def test_eye_exactly_six_pixels_wide_is_usable(pipeline_spec: PipelineSpec) -> None:
    # 경계: 폭이 정확히 6.0 px인 눈은 쓸 수 있다("미만"만 버린다).
    # 1024x512 프레임에서는 정규화 좌표와 픽셀 사이의 변환이 2의 거듭제곱 곱셈·나눗셈이라 오차가 없고,
    # 두 눈꼬리 x가 중심 ± 3.0이므로 폭이 정확히 6.0으로 계산된다.
    assert pipeline_spec.signals.min_eye_width_px == 6.0
    edge = signals.compute(make_frame(pipeline_spec, w=1024, h=512, eye_w_l=6.0, ear_l=0.3), pipeline_spec)
    assert edge.eye_w_l == 6.0
    assert edge.ear_l == pytest.approx(0.3, abs=1e-9)
    assert edge.iris_h_l == pytest.approx(0.5, abs=1e-9)
    assert edge.iris_v_l == pytest.approx(0.0, abs=1e-9)
    # 반대쪽 눈도 같은 규칙이다(L과 R을 따로 판정한다).
    edge_r = signals.compute(make_frame(pipeline_spec, w=1024, h=512, eye_w_r=6.0, ear_r=0.3), pipeline_spec)
    assert edge_r.eye_w_r == 6.0
    assert edge_r.ear_r == pytest.approx(0.3, abs=1e-9)
    assert edge_r.iris_h_r is not None

    # 6.0에서 백만분의 1 px만 모자라도 쓸 수 없는 눈이다. 폭은 측정값 그대로 적는다.
    below = signals.compute(make_frame(pipeline_spec, w=1024, h=512, eye_w_l=5.999999, ear_l=0.3), pipeline_spec)
    assert below.eye_w_l < 6.0
    assert below.eye_w_l == pytest.approx(5.999999, abs=1e-9)
    assert below.ear_l == 0.0
    assert below.iris_h_l is None
    assert below.iris_v_l is None
    # 다른 눈(폭 40 px)은 그대로 계산한다.
    assert below.ear_r == pytest.approx(0.3, abs=1e-9)


def test_one_closed_eye_lands_on_the_right_side(pipeline_spec: PipelineSpec) -> None:
    # L(피험자 왼쪽, 비반전 영상의 오른쪽) 눈만 감았다.
    frame = make_frame(pipeline_spec, ear_l=0.05, ear_r=0.31, blink_l=0.8, blink_r=0.15)
    assert frame.lm[pipeline_spec.landmarks.ear_l[0]][0] > 0.5
    assert frame.lm[pipeline_spec.landmarks.ear_r[0]][0] < 0.5
    s = signals.compute(frame, pipeline_spec)
    assert s.ear_l == pytest.approx(0.05, abs=1e-9)
    assert s.ear_r == pytest.approx(0.31, abs=1e-9)
    assert s.blink_l == 0.8
    assert s.blink_r == 0.15


# ---------------------------------------------------------------------------
# 홍채 비율, 눈동자 합성값
# ---------------------------------------------------------------------------


def test_iris_ratio_hand_computed() -> None:
    # a = (10, 0), d = (4, 1): h = 40/100, v = (10*1 − 0*4)/100
    assert signals.iris_ratio((0.0, 0.0), (10.0, 0.0), (4.0, 1.0)) == (0.4, 0.1)
    # 반대쪽 눈(안쪽 꼬리가 왼쪽): a = (−10, 0), d = (−6, 1): h = 60/100, v = (−10*1 − 0)/100
    assert signals.iris_ratio((10.0, 0.0), (0.0, 0.0), (4.0, 1.0)) == (0.6, -0.1)
    # 두 눈 꼬리가 같은 점이면 0으로 나누지 않고 (0, 0)
    assert signals.iris_ratio((3.0, 3.0), (3.0, 3.0), (4.0, 1.0)) == (0.0, 0.0)


def test_iris_ratio_on_a_tilted_eye_axis() -> None:
    # 눈 축이 기울어진 경우(3:4:5 삼각형). a = (30, 40), |a|² = 2500. 홍채는 바깥 꼬리에서 축을 따라 0.4·a,
    # 축에 수직인 (−40, 30) 방향으로 0.1만큼: d = (12 − 4, 16 + 3) = (8, 19).
    # h = (8·30 + 19·40)/2500 = 1000/2500 = 0.4, v = (30·19 − 40·8)/2500 = 250/2500 = 0.1.
    assert signals.iris_ratio((100.0, 100.0), (130.0, 140.0), (108.0, 119.0)) == pytest.approx((0.4, 0.1), abs=1e-12)
    # a의 세로 성분이 든 항의 부호가 틀리면: |a|² = 900 − 1600 < 0이라 (0, 0), h = (240 − 760)/2500 = −0.208,
    # v = (570 + 320)/2500 = 0.356. 축이 수평(a의 세로 성분 0)인 위의 경우에서는 이 차이가 드러나지 않는다.
    # 반대쪽 눈(a = (−30, −40)): d = iris − outer = (−22, −21). h = (660 + 840)/2500 = 0.6, v = (630 − 880)/2500 = −0.1.
    assert signals.iris_ratio((130.0, 140.0), (100.0, 100.0), (108.0, 119.0)) == pytest.approx((0.6, -0.1), abs=1e-12)


def test_iris_ratio_from_frame(pipeline_spec: PipelineSpec) -> None:
    centred = signals.compute(make_frame(pipeline_spec), pipeline_spec)
    assert centred.iris_h_l == pytest.approx(0.5, abs=1e-9)
    assert centred.iris_h_r == pytest.approx(0.5, abs=1e-9)
    assert centred.iris_v_l == pytest.approx(0.0, abs=1e-9)
    assert centred.iris_v_r == pytest.approx(0.0, abs=1e-9)

    # 홍채를 영상 오른쪽으로 4 px, 아래로 2 px 옮긴다(눈 폭 40 px).
    # R 눈은 바깥 꼬리가 왼쪽이라 h가 0.1 커지고, L 눈은 바깥 꼬리가 오른쪽이라 h가 0.1 작아진다.
    # v = cross(a, d)/|a|²도 두 눈에서 부호가 반대다.
    shifted = signals.compute(make_frame(pipeline_spec, iris_shift_px=(4.0, 2.0)), pipeline_spec)
    assert shifted.iris_h_r == pytest.approx(0.6, abs=1e-9)
    assert shifted.iris_h_l == pytest.approx(0.4, abs=1e-9)
    assert shifted.iris_v_r == pytest.approx(0.05, abs=1e-9)
    assert shifted.iris_v_l == pytest.approx(-0.05, abs=1e-9)


def test_gaze_composites_hand_computed() -> None:
    eye_look = [0.0] * 8
    eye_look[EL_IN_L] = 0.10
    eye_look[EL_OUT_L] = 0.50
    eye_look[EL_UP_L] = 0.20
    eye_look[EL_DOWN_L] = 0.40
    eye_look[EL_IN_R] = 0.60
    eye_look[EL_OUT_R] = 0.05
    eye_look[EL_UP_R] = 0.30
    eye_look[EL_DOWN_R] = 0.20
    # gaze_h = ((in_r + out_l) − (out_r + in_l)) / 2 = ((0.60 + 0.50) − (0.05 + 0.10)) / 2 = 0.475
    assert signals.gaze_h(eye_look) == pytest.approx(0.475, abs=1e-12)
    # gaze_v = (0.20 + 0.30)/2 − (0.40 + 0.20)/2 = −0.05
    assert signals.gaze_v(eye_look) == pytest.approx(-0.05, abs=1e-12)
    assert signals.gaze_h(ZERO8) == 0.0
    assert signals.gaze_v(ZERO8) == 0.0


# ---------------------------------------------------------------------------
# 머리 자세
# ---------------------------------------------------------------------------


def test_official_matrix_angles() -> None:
    yaw, pitch, roll = signals.euler_from_colmajor(OFFICIAL_MATRIX)
    assert yaw == pytest.approx(2.24, abs=0.01)
    assert pitch == pytest.approx(6.32, abs=0.01)
    assert roll == pytest.approx(0.99, abs=0.01)
    # 식을 직접 적용한 값: M[0][2] = m[8], M[2][2] = m[10], M[1][2] = m[9], M[1][0] = m[1], M[1][1] = m[5]
    assert yaw == pytest.approx(math.degrees(math.atan2(0.038823195, 0.99315894)), abs=1e-12)
    assert pitch == pytest.approx(math.degrees(math.atan2(0.1101321, math.hypot(0.01715696, 0.9937692))), abs=1e-12)
    assert roll == pytest.approx(math.degrees(math.atan2(0.01715696, 0.9937692)), abs=1e-12)


def test_euler_round_trip_over_a_grid() -> None:
    for yaw in range(-75, 76, 15):
        for pitch in range(-60, 61, 15):
            for roll in range(-45, 46, 15):
                expected = pytest.approx((yaw, pitch, roll), abs=1e-9)
                r = signals.rotation_from_euler(float(yaw), float(pitch), float(roll))
                assert signals.euler_from_rotation(r) == expected
                assert signals.euler_from_colmajor(colmajor_from_rotation(r)) == expected


def test_rotation_from_euler_is_orthonormal() -> None:
    r = signals.rotation_from_euler(33.0, -21.0, 12.0)
    product = matmul3(r, (r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]))
    assert product == pytest.approx((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), abs=1e-12)
    assert signals.rotation_from_euler(0.0, 0.0, 0.0) == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_sign_conventions() -> None:
    # 얼굴 모델: +X 피험자 왼쪽(비반전 영상의 오른쪽), +Y 위, +Z 얼굴 앞쪽. 카메라: +X 영상 오른쪽, +Y 위.
    # yaw 양수: 얼굴 앞쪽(+Z)이 영상 오른쪽(+X) = 피험자 자기 왼쪽으로 돈다.
    r = signals.rotation_from_euler(30.0, 0.0, 0.0)
    assert r[2] == pytest.approx(0.5, abs=1e-12)  # R[0][2]
    # pitch 양수: 얼굴 앞쪽이 아래(−Y)로 = 턱이 내려간다.
    r = signals.rotation_from_euler(0.0, 30.0, 0.0)
    assert r[5] == pytest.approx(-0.5, abs=1e-12)  # R[1][2]
    # roll 양수: 모델의 +X 축이 위(+Y)로 = 영상에서 반시계 방향.
    r = signals.rotation_from_euler(0.0, 0.0, 30.0)
    assert r[3] == pytest.approx(0.5, abs=1e-12)  # R[1][0]


def test_colmajor_array_matches_4x4() -> None:
    r = signals.rotation_from_euler(20.0, -10.0, 5.0)
    matrix = [
        [r[0], r[1], r[2], 1.5],
        [r[3], r[4], r[5], -2.5],
        [r[6], r[7], r[8], -60.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # 열 우선: m[4*j + i] = M[i][j]
    colmajor = [0.0] * 16
    for i in range(4):
        for j in range(4):
            colmajor[4 * j + i] = matrix[i][j]
    yaw, pitch, roll = signals.euler_from_colmajor(colmajor)
    expected_yaw = math.atan2(matrix[0][2], matrix[2][2])
    expected_pitch = math.atan2(-matrix[1][2], math.hypot(matrix[1][0], matrix[1][1]))
    expected_roll = math.atan2(matrix[1][0], matrix[1][1])
    assert yaw == pytest.approx(math.degrees(expected_yaw), abs=1e-12)
    assert pitch == pytest.approx(math.degrees(expected_pitch), abs=1e-12)
    assert roll == pytest.approx(math.degrees(expected_roll), abs=1e-12)
    assert (yaw, pitch, roll) == pytest.approx((20.0, -10.0, 5.0), abs=1e-9)
    assert signals.rotation_from_colmajor(colmajor) == r
    # 평행이동은 각도에 영향을 주지 않는다.
    moved = colmajor_from_rotation(r, (9.0, 9.0, 9.0))
    assert signals.euler_from_colmajor(moved) == pytest.approx((20.0, -10.0, 5.0), abs=1e-9)

    # 행 우선으로 잘못 넣으면 전치 행렬이 되어 부호가 뒤집힌다(순수 yaw에서 정확히 반대).
    p = signals.rotation_from_euler(20.0, 0.0, 0.0)
    rowmajor = [p[0], p[1], p[2], 0.0, p[3], p[4], p[5], 0.0, p[6], p[7], p[8], 0.0, 0.0, 0.0, 0.0, 1.0]
    assert signals.euler_from_colmajor(rowmajor)[0] == pytest.approx(-20.0, abs=1e-9)


def test_matrix_length_is_checked() -> None:
    with pytest.raises(ValueError):
        signals.euler_from_colmajor((1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        signals.euler_from_rotation((1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        signals.relative_euler(OFFICIAL_MATRIX, (1.0, 0.0))
    # 회전 블록을 꺼내는 함수도 길이를 본다. 12개 값(3x4)을 주면 색인은 닿지만 4x4 행렬이 아니다.
    with pytest.raises(ValueError):
        signals.rotation_from_colmajor(OFFICIAL_MATRIX[:12])
    with pytest.raises(ValueError):
        signals.relative_euler(OFFICIAL_MATRIX[:12], signals.rotation_from_euler(0.0, 0.0, 0.0))


def test_relative_euler_is_not_the_angle_difference() -> None:
    r0 = signals.rotation_from_euler(30.0, 15.0, 0.0)
    delta = signals.rotation_from_euler(10.0, 5.0, 2.0)
    m = colmajor_from_rotation(matmul3(r0, delta))  # R = R0·Δ -> R0ᵀ·R = Δ
    assert signals.relative_euler(m, r0) == pytest.approx((10.0, 5.0, 2.0), abs=1e-9)

    # 오일러 각의 단순 차는 pitch나 roll에서 1도 넘게 틀린다.
    absolute = signals.euler_from_colmajor(m)
    naive = (absolute[0] - 30.0, absolute[1] - 15.0, absolute[2] - 0.0)
    assert max(abs(naive[1] - 5.0), abs(naive[2] - 2.0)) > 1.0

    # 기준이 단위 행렬이면 절대 각도와 같고, 자기 자신을 기준으로 하면 0이다.
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert signals.relative_euler(m, identity) == pytest.approx(absolute, abs=1e-12)
    assert signals.relative_euler(colmajor_from_rotation(r0), r0) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


def test_compute_copies_blendshapes_in_fixed_order(pipeline_spec: PipelineSpec) -> None:
    eye_look = (0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18)
    frame = make_frame(
        pipeline_spec,
        t_ms=123456,
        blink_l=0.31,
        blink_r=0.42,
        jaw_open=0.07,
        eye_look=eye_look,
        yaw=12.0,
        pitch=-7.0,
        roll=5.0,
    )
    s = signals.compute(frame, pipeline_spec)
    assert s.t_ms == 123456
    assert s.face is True
    assert s.blink_l == 0.31
    assert s.blink_r == 0.42
    assert s.jaw_open == 0.07
    assert s.eye_look == eye_look
    assert (s.yaw, s.pitch, s.roll) == pytest.approx((12.0, -7.0, 5.0), abs=1e-9)
    # eye_look의 순서는 EYE_LOOK_ORDER이고 이름과 이렇게 짝이다.
    assert EYE_LOOK_ORDER == ("in_l", "out_l", "up_l", "down_l", "in_r", "out_r", "up_r", "down_r")
    assert frame.bs["eyeLookInLeft"] == s.eye_look[EL_IN_L]
    assert frame.bs["eyeLookOutLeft"] == s.eye_look[EL_OUT_L]
    assert frame.bs["eyeLookUpLeft"] == s.eye_look[EL_UP_L]
    assert frame.bs["eyeLookDownLeft"] == s.eye_look[EL_DOWN_L]
    assert frame.bs["eyeLookInRight"] == s.eye_look[EL_IN_R]
    assert frame.bs["eyeLookOutRight"] == s.eye_look[EL_OUT_R]
    assert frame.bs["eyeLookUpRight"] == s.eye_look[EL_UP_R]
    assert frame.bs["eyeLookDownRight"] == s.eye_look[EL_DOWN_R]
    assert frame.bs["eyeBlinkLeft"] == s.blink_l
    assert frame.bs["eyeBlinkRight"] == s.blink_r
    assert frame.bs["jawOpen"] == s.jaw_open


def test_compute_without_face(pipeline_spec: PipelineSpec) -> None:
    frame = RawFrame(t_ms=77, face=False, w=640, h=480, lm={}, bs={}, m=())
    s = signals.compute(frame, pipeline_spec)
    assert s.t_ms == 77
    assert s.face is False
    assert (s.ear_l, s.ear_r, s.eye_w_l, s.eye_w_r, s.blink_l, s.blink_r) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert s.eye_look == ZERO8
    assert (s.jaw_open, s.yaw, s.pitch, s.roll) == (0.0, 0.0, 0.0, 0.0)
    assert (s.iris_h_l, s.iris_h_r, s.iris_v_l, s.iris_v_r) == (None, None, None, None)


def test_compute_rejects_incomplete_face_frame(pipeline_spec: PipelineSpec) -> None:
    frame = make_frame(pipeline_spec)
    lm = dict(frame.lm)
    del lm[pipeline_spec.landmarks.ear_l[2]]
    with pytest.raises(KeyError):
        signals.compute(RawFrame(t_ms=1, face=True, w=640, h=480, lm=lm, bs=frame.bs, m=frame.m), pipeline_spec)
    bs = dict(frame.bs)
    del bs[pipeline_spec.blendshapes.jaw_open]
    with pytest.raises(KeyError):
        signals.compute(RawFrame(t_ms=1, face=True, w=640, h=480, lm=frame.lm, bs=bs, m=frame.m), pipeline_spec)
    with pytest.raises(ValueError):
        signals.compute(RawFrame(t_ms=1, face=True, w=640, h=480, lm=frame.lm, bs=frame.bs, m=()), pipeline_spec)


def test_helper_frame_holds_every_required_input(pipeline_spec: PipelineSpec) -> None:
    frame = make_frame(pipeline_spec)
    assert sorted(frame.lm) == list(pipeline_spec.landmarks.required_indices())
    assert len(frame.lm) == 44
    assert sorted(frame.bs) == sorted(pipeline_spec.blendshapes.required_names())
    assert len(frame.m) == 16


# ---------------------------------------------------------------------------
# 반전 항등식
# ---------------------------------------------------------------------------


def _asymmetric_frame(spec: PipelineSpec, yaw: float = 12.0) -> RawFrame:
    return make_frame(
        spec,
        w=1280,
        h=720,
        ear_l=0.31,
        ear_r=0.22,
        eye_w_l=40.0,
        eye_w_r=36.0,
        blink_l=0.10,
        blink_r=0.60,
        eye_look=(0.11, 0.42, 0.13, 0.24, 0.35, 0.06, 0.17, 0.08),
        jaw_open=0.09,
        yaw=yaw,
        pitch=-7.0,
        roll=5.0,
        iris_shift_px=(3.0, -1.5),
    )


def test_mirror_identity_for_signals(pipeline_spec: PipelineSpec) -> None:
    frame = _asymmetric_frame(pipeline_spec)
    s = signals.compute(frame, pipeline_spec)
    mirrored = signals.compute(mirror_frame(pipeline_spec, frame), pipeline_spec)

    # 반전은 yaw와 roll의 부호를 뒤집고 pitch는 그대로 둔다.
    assert mirrored.yaw == pytest.approx(-s.yaw, abs=1e-12)
    assert mirrored.pitch == pytest.approx(s.pitch, abs=1e-12)
    assert mirrored.roll == pytest.approx(-s.roll, abs=1e-12)
    assert (s.yaw, s.pitch, s.roll) == pytest.approx((12.0, -7.0, 5.0), abs=1e-9)

    # 눈별 신호는 좌우가 바뀐다.
    assert mirrored.ear_l == pytest.approx(s.ear_r, abs=1e-9)
    assert mirrored.ear_r == pytest.approx(s.ear_l, abs=1e-9)
    assert mirrored.eye_w_l == pytest.approx(s.eye_w_r, abs=1e-9)
    assert mirrored.eye_w_r == pytest.approx(s.eye_w_l, abs=1e-9)
    assert mirrored.blink_l == s.blink_r
    assert mirrored.blink_r == s.blink_l
    assert mirrored.eye_look == s.eye_look[4:] + s.eye_look[:4]
    assert mirrored.jaw_open == s.jaw_open

    # 눈동자 합성값: 수평은 부호가 뒤집히고 수직은 그대로다.
    assert signals.gaze_h(mirrored.eye_look) == pytest.approx(-signals.gaze_h(s.eye_look), abs=1e-12)
    assert signals.gaze_v(mirrored.eye_look) == pytest.approx(signals.gaze_v(s.eye_look), abs=1e-12)

    # 홍채 비율: h는 바깥 꼬리 기준이라 그대로 옮겨 가고, v는 부호가 뒤집힌다.
    assert s.iris_h_l is not None and s.iris_h_r is not None and s.iris_v_l is not None and s.iris_v_r is not None
    assert mirrored.iris_h_l == pytest.approx(s.iris_h_r, abs=1e-9)
    assert mirrored.iris_h_r == pytest.approx(s.iris_h_l, abs=1e-9)
    assert mirrored.iris_v_l == pytest.approx(-s.iris_v_r, abs=1e-9)
    assert mirrored.iris_v_r == pytest.approx(-s.iris_v_l, abs=1e-9)


def test_mirror_identity_keeps_closure(pipeline_spec: PipelineSpec) -> None:
    baseline = Baseline(
        e_l=0.33,
        e_r=0.29,
        b_l=0.08,
        b_r=0.21,
        r0=signals.rotation_from_euler(4.0, -3.0, 2.0),
        jaw0=0.05,
        gaze_h0=0.03,
        gaze_v0=-0.02,
        t_complete_ms=60000,
        status=BASELINE_OK,
    )
    # 반전한 세션의 기준선: 눈별 값이 바뀌고 yaw0, roll0, gaze_h0의 부호가 뒤집힌다.
    mirrored_baseline = Baseline(
        e_l=0.29,
        e_r=0.33,
        b_l=0.21,
        b_r=0.08,
        r0=signals.rotation_from_euler(-4.0, -3.0, -2.0),
        jaw0=0.05,
        gaze_h0=-0.03,
        gaze_v0=-0.02,
        t_complete_ms=60000,
        status=BASELINE_OK,
    )
    # 먼 쪽 눈 가중이 1, 중간, 0인 yaw를 모두 지난다.
    for yaw in (-34.0, -24.0, -12.0, 0.0, 4.0, 12.0, 24.0, 34.0):
        frame = _asymmetric_frame(pipeline_spec, yaw=yaw)
        state = calibrate.frame_state(signals.compute(frame, pipeline_spec), baseline, pipeline_spec, False)
        mirrored_state = calibrate.frame_state(
            signals.compute(mirror_frame(pipeline_spec, frame), pipeline_spec), mirrored_baseline, pipeline_spec, False
        )
        assert state.closure_valid is True
        assert mirrored_state.closure_valid is True
        assert 0.0 < state.closure < 1.0
        assert mirrored_state.closure == pytest.approx(state.closure, abs=1e-9)
        assert mirrored_state.yaw_rel == pytest.approx(-state.yaw_rel, abs=1e-9)
        assert mirrored_state.pitch_rel == pytest.approx(state.pitch_rel, abs=1e-9)
        assert mirrored_state.gaze_h_rel == pytest.approx(-state.gaze_h_rel, abs=1e-9)
        assert mirrored_state.gaze_v_rel == pytest.approx(state.gaze_v_rel, abs=1e-9)
        assert mirrored_state.jaw_rel == state.jaw_rel


# ---------------------------------------------------------------------------
# 이식 제약
# ---------------------------------------------------------------------------


def test_module_imports_standard_library_only(repo_root: Path) -> None:
    code = (
        "import sys\n"
        "import drowsy.signals\n"
        "banned = [name for name in ('numpy', 'pandas', 'yaml', 'torch') if name in sys.modules]\n"
        "assert not banned, banned\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
