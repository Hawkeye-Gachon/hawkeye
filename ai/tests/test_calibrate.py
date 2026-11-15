"""`drowsy.calibrate` 단위 테스트 (PLAN 4.2절, 4.3절, 10.1절 "기준선", "closure").

`FrameSignals`를 직접 만들어 넣는다. 기댓값은 주석의 손 계산이다.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from drowsy import calibrate, signals
from drowsy.calibrate import BaselineEstimator
from drowsy.spec import PipelineSpec
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OFF_AXIS,
    BASELINE_OK,
    IDENTITY_R0,
    Baseline,
    FrameSignals,
)

ZERO8: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def sig(
    t_ms: int,
    *,
    ear_l: float = 0.30,
    ear_r: float = 0.30,
    eye_w_l: float = 40.0,
    eye_w_r: float = 40.0,
    blink_l: float = 0.20,
    blink_r: float = 0.20,
    eye_look: tuple[float, ...] = ZERO8,
    jaw_open: float = 0.05,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> FrameSignals:
    """얼굴이 있는 프레임의 신호."""
    return FrameSignals(
        t_ms=t_ms,
        face=True,
        ear_l=ear_l,
        ear_r=ear_r,
        eye_w_l=eye_w_l,
        eye_w_r=eye_w_r,
        blink_l=blink_l,
        blink_r=blink_r,
        eye_look=eye_look,
        jaw_open=jaw_open,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def no_face(t_ms: int) -> FrameSignals:
    """얼굴이 없는 프레임의 신호(`signals.compute`가 내는 것과 같은 모양)."""
    return FrameSignals(
        t_ms=t_ms,
        face=False,
        ear_l=0.0,
        ear_r=0.0,
        eye_w_l=0.0,
        eye_w_r=0.0,
        blink_l=0.0,
        blink_r=0.0,
        eye_look=ZERO8,
        jaw_open=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )


def mirror_signals(s: FrameSignals) -> FrameSignals:
    """좌우 반전 입력의 신호: 눈별 값이 바뀌고 yaw와 roll의 부호가 뒤집힌다."""
    return FrameSignals(
        t_ms=s.t_ms,
        face=s.face,
        ear_l=s.ear_r,
        ear_r=s.ear_l,
        eye_w_l=s.eye_w_r,
        eye_w_r=s.eye_w_l,
        blink_l=s.blink_r,
        blink_r=s.blink_l,
        eye_look=s.eye_look[4:] + s.eye_look[:4],
        jaw_open=s.jaw_open,
        yaw=-s.yaw,
        pitch=s.pitch,
        roll=-s.roll,
    )


def mirror_baseline(b: Baseline) -> Baseline:
    r = b.r0
    # S·R0·S, S = diag(−1, 1, 1): 첫 행과 첫 열 중 하나에만 속한 원소의 부호가 뒤집힌다.
    mirrored_r0 = (r[0], -r[1], -r[2], -r[3], r[4], r[5], -r[6], r[7], r[8])
    return Baseline(
        e_l=b.e_r,
        e_r=b.e_l,
        b_l=b.b_r,
        b_r=b.b_l,
        r0=mirrored_r0,
        jaw0=b.jaw0,
        gaze_h0=-b.gaze_h0,
        gaze_v0=b.gaze_v0,
        t_complete_ms=b.t_complete_ms,
        status=b.status,
    )


def make_baseline(
    *,
    e_l: float = 0.30,
    e_r: float = 0.30,
    b_l: float = 0.20,
    b_r: float = 0.20,
    yaw0: float = 0.0,
    pitch0: float = 0.0,
    roll0: float = 0.0,
    jaw0: float = 0.05,
    gaze_h0: float = 0.0,
    gaze_v0: float = 0.0,
    status: str = BASELINE_OK,
) -> Baseline:
    return Baseline(
        e_l=e_l,
        e_r=e_r,
        b_l=b_l,
        b_r=b_r,
        r0=signals.rotation_from_euler(yaw0, pitch0, roll0),
        jaw0=jaw0,
        gaze_h0=gaze_h0,
        gaze_v0=gaze_v0,
        t_complete_ms=60000,
        status=status,
    )


def run_estimator(est: BaselineEstimator, frames: list[FrameSignals]) -> list[tuple[int, Baseline]]:
    """프레임을 차례로 넣고 (기준선을 돌려준 프레임의 t_ms, 기준선) 목록을 돌려준다."""
    out: list[tuple[int, Baseline]] = []
    for s in frames:
        baseline = est.update(s)
        if baseline is not None:
            out.append((s.t_ms, baseline))
    return out


# ---------------------------------------------------------------------------
# 분위수와 중앙값
# ---------------------------------------------------------------------------


def test_nearest_rank_quantile_definition() -> None:
    values = [7.0, 3.0, 10.0, 1.0, 5.0, 9.0, 2.0, 8.0, 4.0, 6.0]  # 정렬하면 1..10
    # sorted[ceil(q*n) − 1]
    assert calibrate.nearest_rank_quantile(values, 0.9) == 9.0  # ceil(9.0) = 9
    assert calibrate.nearest_rank_quantile(values, 0.1) == 1.0  # ceil(1.0) = 1
    assert calibrate.nearest_rank_quantile(values, 0.5) == 5.0  # ceil(5.0) = 5
    assert calibrate.nearest_rank_quantile(values, 0.25) == 3.0  # ceil(2.5) = 3
    assert calibrate.nearest_rank_quantile(values, 0.91) == 10.0  # ceil(9.1) = 10
    assert calibrate.nearest_rank_quantile(values, 1.0) == 10.0
    assert calibrate.nearest_rank_quantile(values, 0.0) == 1.0  # 순위는 1 이상으로 자른다
    assert calibrate.nearest_rank_quantile([4.2], 0.9) == 4.2
    # 입력을 바꾸지 않는다.
    assert values[0] == 7.0


def test_nearest_rank_quantile_at_integer_ranks(pipeline_spec: PipelineSpec) -> None:
    # spec의 분위수(0.1, 0.5, 0.9)는 q*n이 정수일 때 배정밀도 곱도 정확히 그 정수여서 순위가 밀리지 않는다.
    cfg = pipeline_spec.baseline
    assert (cfg.ear_quantile, cfg.blink_quantile, cfg.blink_record_quantiles) == (0.9, 0.1, (0.5, 0.9))
    for n in range(10, 5410, 10):  # 180초 * 30 fps = 5400 프레임까지
        values = [float(i) for i in range(1, n + 1)]
        assert calibrate.nearest_rank_quantile(values, 0.1) == n / 10
        assert calibrate.nearest_rank_quantile(values, 0.5) == n / 2
        assert calibrate.nearest_rank_quantile(values, 0.9) == 9 * n / 10
    # q*n이 정수가 아니면 올림한다: ceil(0.9 * 1499) = ceil(1349.1) = 1350
    values = [float(i) for i in range(1, 1500)]
    assert calibrate.nearest_rank_quantile(values, 0.9) == 1350.0
    assert calibrate.nearest_rank_quantile(values, 0.1) == 150.0  # ceil(149.9)


def test_median_definition() -> None:
    assert calibrate.median([3.0, 1.0, 2.0]) == 2.0
    # 짝수 개면 가운데 두 값의 평균(최근접 순위라면 2.0이 된다)
    assert calibrate.median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert calibrate.median([5.0]) == 5.0
    assert calibrate.median([1.0, 2.0]) == 1.5
    # 부호를 뒤집은 표본의 중앙값은 정확히 부호만 뒤집힌다(반전 입력에서 R0이 대칭이 되는 조건).
    values = [0.3, -1.7, 2.9, 0.11, -0.05, 4.4, 1.3, -2.2]
    assert calibrate.median([-v for v in values]) == -calibrate.median(values)


def test_quantile_and_median_reject_empty_input() -> None:
    with pytest.raises(ValueError):
        calibrate.nearest_rank_quantile([], 0.5)
    with pytest.raises(ValueError):
        calibrate.median([])


# ---------------------------------------------------------------------------
# 기준선: 정상 완성
# ---------------------------------------------------------------------------

T0 = 987_654_321  # 첫 t_ms는 임의의 큰 값이다. 추정기는 세션 시간으로 계산한다.
EYE_LOOK = (0.10, 0.50, 0.20, 0.40, 0.60, 0.05, 0.30, 0.20)  # gaze_h = 0.475, gaze_v = −0.05


def _normal_session() -> list[FrameSignals]:
    """25 fps, 세션 시간 0~59960 ms의 얼굴 프레임 1500개. 값은 10개 주기로 돈다."""
    frames: list[FrameSignals] = []
    for i in range(1500):
        k = i % 10
        frames.append(
            sig(
                T0 + 40 * i,
                ear_l=0.21 + 0.01 * k,
                ear_r=0.26 + 0.01 * k,
                blink_l=0.10 + 0.02 * k,
                blink_r=0.15 + 0.02 * k,
                eye_look=EYE_LOOK,
                jaw_open=0.05 + 0.02 * (i % 2),
                yaw=5.0,
                pitch=-3.0,
                roll=2.0,
            )
        )
    return frames


def test_baseline_statistics_hand_computed(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    assert est.diagnostics is None
    assert run_estimator(est, _normal_session()) == []
    assert est.diagnostics is None

    # 세션 시간 60000 ms에 온 프레임에서 기준선이 한 번 나온다. 그 프레임은 기준선에 들어가지 않는다.
    completing = sig(T0 + 60000, ear_l=0.99, ear_r=0.99, blink_l=0.99, blink_r=0.99, yaw=80.0)
    baseline = est.update(completing)
    assert baseline is not None
    assert baseline.status == BASELINE_OK
    assert baseline.t_complete_ms == 60000

    # 값마다 150번씩 나온다. 90 분위수: sorted[ceil(0.9*1500) − 1] = sorted[1349] -> 9번째 값(k = 8)
    assert baseline.e_l == 0.21 + 0.01 * 8
    assert baseline.e_r == 0.26 + 0.01 * 8
    # 10 분위수: sorted[ceil(0.1*1500) − 1] = sorted[149] -> 첫 번째 값(k = 0)
    assert baseline.b_l == 0.10
    assert baseline.b_r == 0.15
    # 중앙값(표본 1500개, 짝수): jaw는 0.05와 0.07이 750개씩 -> 가운데 두 값의 평균 0.06
    assert baseline.jaw0 == pytest.approx(0.06, abs=1e-12)
    assert baseline.gaze_h0 == pytest.approx(0.475, abs=1e-12)
    assert baseline.gaze_v0 == pytest.approx(-0.05, abs=1e-12)
    assert baseline.r0 == signals.rotation_from_euler(5.0, -3.0, 2.0)

    diagnostics = est.diagnostics
    assert diagnostics is not None
    assert diagnostics.n_face_frames == 1500
    # 프레임마다 40 ms를 대표하고 마지막 프레임(59960)은 마감까지 40 ms -> 60000 / 60000
    assert diagnostics.face_ratio == 1.0
    # 기록용 50, 90 분위수: sorted[749] -> k = 4, sorted[1349] -> k = 8
    assert diagnostics.blink_l_quantiles == (0.10 + 0.02 * 4, 0.10 + 0.02 * 8)
    assert diagnostics.blink_r_quantiles == (0.15 + 0.02 * 4, 0.15 + 0.02 * 8)

    # 기준선은 한 번만 돌려준다.
    assert est.update(sig(T0 + 60040)) is None
    assert est.update(sig(T0 + 200000)) is None


def test_t_complete_is_the_deadline_not_the_frame_time(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    run_estimator(est, _normal_session())
    baseline = est.update(sig(T0 + 60013))
    assert baseline is not None
    assert baseline.t_complete_ms == 60000


def test_frame_just_before_deadline_does_not_complete(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    run_estimator(est, _normal_session())
    assert est.update(sig(T0 + 59999)) is None
    assert est.update(sig(T0 + 60000)) is not None


def test_non_increasing_timestamp_is_rejected(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    est.update(sig(1000))
    est.update(sig(1040))
    with pytest.raises(ValueError):
        est.update(sig(1040))
    with pytest.raises(ValueError):
        est.update(sig(1000))


def test_off_axis_status(pipeline_spec: PipelineSpec) -> None:
    def status_for(yaw: float) -> str:
        est = BaselineEstimator(pipeline_spec)
        frames = [sig(1 + 100 * i, yaw=yaw) for i in range(601)]  # 10 fps, 마지막 프레임이 60000 ms
        emitted = run_estimator(est, frames)
        assert len(emitted) == 1
        return emitted[0][1].status

    assert status_for(20.0) == BASELINE_OFF_AXIS
    assert status_for(-16.0) == BASELINE_OFF_AXIS
    # |yaw0|이 15도를 "넘으면" off_axis다.
    assert status_for(15.0) == BASELINE_OK
    assert status_for(0.0) == BASELINE_OK


def test_unusable_eye_samples_are_left_out_of_e(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    frames: list[FrameSignals] = []
    for i in range(1500):
        if i % 20 == 0:
            frames.append(sig(1 + 40 * i, ear_l=0.30, ear_r=0.28))
        else:
            # L 눈 폭이 6 px 미만: signals.compute는 이런 눈의 ear를 0.0으로 둔다.
            frames.append(sig(1 + 40 * i, ear_l=0.0, eye_w_l=3.0, ear_r=0.28))
    frames.append(sig(1 + 60000))
    emitted = run_estimator(est, frames)
    assert len(emitted) == 1
    baseline = emitted[0][1]
    # 0.0을 표본에 넣었다면 90 분위수가 0.0이 되어 범위를 벗어났을 것이다.
    assert baseline.status == BASELINE_OK
    assert baseline.e_l == 0.30
    assert baseline.e_r == 0.28


def test_mirrored_session_gives_mirrored_baseline(pipeline_spec: PipelineSpec) -> None:
    frames: list[FrameSignals] = []
    for i in range(1500):
        frames.append(
            sig(
                1 + 40 * i,
                ear_l=0.25 + 0.0001 * (i % 97),
                ear_r=0.31 - 0.0002 * (i % 53),
                blink_l=0.10 + 0.001 * (i % 31),
                blink_r=0.22 - 0.001 * (i % 17),
                eye_look=(0.1 + 0.001 * (i % 7), 0.5, 0.2, 0.4, 0.6, 0.05 + 0.002 * (i % 11), 0.3, 0.2),
                jaw_open=0.04 + 0.0005 * (i % 13),
                yaw=7.0 + 0.01 * (i % 41),
                pitch=-4.0 + 0.02 * (i % 29),
                roll=3.0 - 0.01 * (i % 37),
            )
        )
    frames.append(sig(1 + 60000))
    original = run_estimator(BaselineEstimator(pipeline_spec), frames)
    mirrored = run_estimator(BaselineEstimator(pipeline_spec), [mirror_signals(s) for s in frames])
    assert len(original) == 1 and len(mirrored) == 1
    # 정확히 대칭이다(근사 비교가 아니다).
    assert mirrored[0][1] == mirror_baseline(original[0][1])
    assert original[0][1].e_l != original[0][1].e_r


# ---------------------------------------------------------------------------
# 기준선: 연장과 대체
# ---------------------------------------------------------------------------


def test_extension_when_face_frames_are_too_few(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    # 4 fps: 60초에 얼굴 프레임 240개(< 300), 90초에 360개.
    frames = [sig(1 + 250 * i) for i in range(361)]  # 세션 시간 0, 250, ..., 90000
    emitted = run_estimator(est, frames)
    assert [t for t, _ in emitted] == [1 + 90000]
    baseline = emitted[0][1]
    assert baseline.status == BASELINE_OK
    assert baseline.t_complete_ms == 90000
    diagnostics = est.diagnostics
    assert diagnostics is not None
    assert diagnostics.n_face_frames == 360
    # 프레임 하나가 대표하는 시간은 최대 200 ms: 360 * 200 / 90000 = 0.8
    assert diagnostics.face_ratio == 0.8


def test_extension_when_face_time_ratio_is_too_low(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    frames: list[FrameSignals] = []
    for i in range(2251):  # 25 fps, 세션 시간 0 ~ 90000
        t = 40 * i
        if t < 25000 or t >= 60000:
            frames.append(sig(1 + t))
        else:
            frames.append(no_face(1 + t))
    emitted = run_estimator(est, frames)
    # 60초 시점: 얼굴 프레임 625개(>= 300)지만 얼굴 시간 25000 / 60000 = 0.417 < 0.5 -> 연장
    # 90초 시점: (25000 + 30000) / 90000 = 0.611 -> 완성
    assert [t for t, _ in emitted] == [1 + 90000]
    assert emitted[0][1].t_complete_ms == 90000
    assert emitted[0][1].status == BASELINE_OK
    diagnostics = est.diagnostics
    assert diagnostics is not None
    assert diagnostics.n_face_frames == 625 + 750
    assert diagnostics.face_ratio == pytest.approx(55000.0 / 90000.0, abs=1e-12)


def test_hold_cap_in_face_time_ratio(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    # 400 ms 간격의 얼굴 프레임: 프레임마다 200 ms만 인정 -> 비율 0.5(통과), 프레임 수는 120초에 300개가 된다.
    frames = [sig(1 + 400 * i) for i in range(451)]  # 세션 시간 0 ~ 180000
    emitted = run_estimator(est, frames)
    assert [t for t, _ in emitted] == [1 + 120000]
    assert emitted[0][1].t_complete_ms == 120000
    diagnostics = est.diagnostics
    assert diagnostics is not None
    assert diagnostics.n_face_frames == 300
    assert diagnostics.face_ratio == 0.5


def test_fallback_to_global_constants_without_any_face(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    frames = [no_face(5000 + 100 * i) for i in range(1801)]  # 세션 시간 0 ~ 180000
    emitted = run_estimator(est, frames)
    # 60, 90, 120, 150초에서는 연장만 하고 180초에서 전역 상수로 대체한다.
    assert [t for t, _ in emitted] == [5000 + 180000]
    baseline = emitted[0][1]
    constants = pipeline_spec.global_constants
    assert baseline.status == BASELINE_FALLBACK_GLOBAL
    assert baseline.t_complete_ms == 180000
    assert (baseline.e_l, baseline.e_r) == (constants.e, constants.e)
    assert (baseline.b_l, baseline.b_r) == (constants.b, constants.b)
    assert baseline.r0 == IDENTITY_R0
    assert (baseline.jaw0, baseline.gaze_h0, baseline.gaze_v0) == (constants.jaw0, 0.0, 0.0)
    diagnostics = est.diagnostics
    assert diagnostics is not None
    assert diagnostics.n_face_frames == 0
    assert diagnostics.face_ratio == 0.0
    assert diagnostics.blink_l_quantiles == (0.0, 0.0)
    assert diagnostics.blink_r_quantiles == (0.0, 0.0)
    assert est.update(no_face(5000 + 180100)) is None


@pytest.mark.parametrize(
    ("ear_l", "ear_r", "eye_w_r"),
    [
        (0.50, 0.50, 40.0),  # E가 0.45보다 크다
        (0.30, 0.10, 40.0),  # 한쪽 E가 0.12보다 작다
        (0.30, 0.30, 3.0),  # R 눈을 한 번도 쓸 수 없다
    ],
)
def test_fallback_when_e_is_out_of_range(
    pipeline_spec: PipelineSpec, ear_l: float, ear_r: float, eye_w_r: float
) -> None:
    est = BaselineEstimator(pipeline_spec)
    frames = [
        sig(1 + 100 * i, ear_l=ear_l, ear_r=ear_r, eye_w_r=eye_w_r, blink_l=0.3, blink_r=0.4) for i in range(1801)
    ]
    emitted = run_estimator(est, frames)
    assert [t for t, _ in emitted] == [1 + 180000]
    assert emitted[0][1] == dataclasses.replace(calibrate.global_baseline(pipeline_spec), t_complete_ms=180000)
    # 대체해도 분포 기록은 남긴다.
    diagnostics = est.diagnostics
    assert diagnostics is not None
    assert diagnostics.n_face_frames == 1800
    assert diagnostics.face_ratio == 1.0
    assert diagnostics.blink_l_quantiles == (0.3, 0.3)
    assert diagnostics.blink_r_quantiles == (0.4, 0.4)


def test_e_range_bounds_are_inclusive(pipeline_spec: PipelineSpec) -> None:
    cfg = pipeline_spec.baseline
    est = BaselineEstimator(pipeline_spec)
    frames = [sig(1 + 100 * i, ear_l=cfg.e_min, ear_r=cfg.e_max) for i in range(601)]
    emitted = run_estimator(est, frames)
    assert len(emitted) == 1
    assert emitted[0][1].status == BASELINE_OK
    assert (emitted[0][1].e_l, emitted[0][1].e_r) == (0.12, 0.45)


def test_e_range_bounds_are_inclusive_for_the_other_eye_too(pipeline_spec: PipelineSpec) -> None:
    # 위의 테스트와 좌우를 바꾼 경우: Left 눈의 E가 정확히 e_max, Right 눈의 E가 정확히 e_min이어도 범위 안이다.
    cfg = pipeline_spec.baseline
    est = BaselineEstimator(pipeline_spec)
    frames = [sig(1 + 100 * i, ear_l=cfg.e_max, ear_r=cfg.e_min) for i in range(601)]
    emitted = run_estimator(est, frames)
    assert len(emitted) == 1
    assert emitted[0][1].status == BASELINE_OK
    assert (emitted[0][1].e_l, emitted[0][1].e_r) == (0.45, 0.12)


def test_fallback_when_the_left_eye_is_never_usable(pipeline_spec: PipelineSpec) -> None:
    # Left 눈의 표본이 하나도 없으면 E를 정할 수 없다: 연장을 거쳐 180초에 전역 상수로 대체한다(빈 목록의 분위수를
    # 구하려다 오류를 내면 안 된다). Right 눈이 없는 경우는 test_fallback_when_e_is_out_of_range의 셋째 경우다.
    est = BaselineEstimator(pipeline_spec)
    frames = [sig(1 + 100 * i, eye_w_l=3.0) for i in range(1801)]
    emitted = run_estimator(est, frames)
    assert [t for t, _ in emitted] == [1 + 180000]
    assert emitted[0][1] == dataclasses.replace(calibrate.global_baseline(pipeline_spec), t_complete_ms=180000)


@pytest.mark.parametrize("side", ["l", "r"])
def test_eye_exactly_six_pixels_wide_gives_samples_for_e(pipeline_spec: PipelineSpec, side: str) -> None:
    # 눈 폭이 정확히 min_eye_width_px(6 px)인 눈은 쓸 수 있는 눈이고 그 EAR이 E의 표본이 된다('이상').
    # '초과'로 구현하면 그 눈의 표본이 하나도 없어 60초에 완성되지 않는다.
    assert pipeline_spec.signals.min_eye_width_px == 6.0
    est = BaselineEstimator(pipeline_spec)
    width = {"eye_w_l": 6.0} if side == "l" else {"eye_w_r": 6.0}
    frames = [sig(1 + 100 * i, ear_l=0.31, ear_r=0.29, **width) for i in range(601)]
    emitted = run_estimator(est, frames)
    assert [t for t, _ in emitted] == [1 + 60000]
    assert (emitted[0][1].e_l, emitted[0][1].e_r, emitted[0][1].status) == (0.31, 0.29, BASELINE_OK)


def test_deadlines_are_checked_in_order_after_a_long_frame_gap(pipeline_spec: PipelineSpec) -> None:
    # 59초까지 정상, 그 뒤 프레임이 세션 시간 125000 ms에 온다. 조건은 60초 마감에서 이미 만족한다.
    est = BaselineEstimator(pipeline_spec)
    frames = [sig(1 + 100 * i) for i in range(590)]  # 세션 시간 0 ~ 58900
    assert run_estimator(est, frames) == []
    baseline = est.update(sig(1 + 125000))
    assert baseline is not None
    assert baseline.t_complete_ms == 60000
    diagnostics = est.diagnostics
    assert diagnostics is not None
    # 마지막 프레임(58900)은 200 ms만 대표한다: (589 * 100 + 200) / 60000
    assert diagnostics.face_ratio == pytest.approx(59100.0 / 60000.0, abs=1e-12)


def test_completion_at_the_last_deadline_is_not_a_fallback(pipeline_spec: PipelineSpec) -> None:
    est = BaselineEstimator(pipeline_spec)
    # 4 fps로 60초(얼굴 프레임 240개), 그 뒤 125초까지 프레임이 없다.
    frames = [sig(1 + 250 * i) for i in range(240)]  # 세션 시간 0 ~ 59750
    assert run_estimator(est, frames) == []
    # 125000 ms의 프레임: 60, 90, 120초 마감을 차례로 검사하지만 모두 프레임 수 미달이다.
    assert est.update(sig(1 + 125000)) is None
    # 150초 마감: 프레임 340개, 얼굴 시간 340 * 200 = 68000 / 150000 = 0.453 < 0.5 -> 연장
    frames = [sig(1 + 125000 + 250 * i) for i in range(1, 100)]  # ~ 149750
    assert run_estimator(est, frames) == []
    assert est.update(sig(1 + 150000)) is None
    # 180초 마감: 프레임 460개, 얼굴 시간 460 * 200 = 92000 / 180000 = 0.511 -> 대체가 아니라 완성
    frames = [sig(1 + 150000 + 250 * i) for i in range(1, 120)]  # ~ 179750
    assert run_estimator(est, frames) == []
    baseline = est.update(sig(1 + 180000))
    assert baseline is not None
    assert baseline.status == BASELINE_OK
    assert baseline.t_complete_ms == 180000
    diagnostics = est.diagnostics
    assert diagnostics is not None
    assert diagnostics.n_face_frames == 460
    assert diagnostics.face_ratio == pytest.approx(92000.0 / 180000.0, abs=1e-12)


def test_global_baseline(pipeline_spec: PipelineSpec) -> None:
    baseline = calibrate.global_baseline(pipeline_spec)
    assert baseline == Baseline(
        e_l=0.27,
        e_r=0.27,
        b_l=0.22,
        b_r=0.22,
        r0=IDENTITY_R0,
        jaw0=0.08,
        gaze_h0=0.0,
        gaze_v0=0.0,
        t_complete_ms=0,
        status=BASELINE_FALLBACK_GLOBAL,
    )


# ---------------------------------------------------------------------------
# closure
# ---------------------------------------------------------------------------


def test_eye_closure_hand_computed(pipeline_spec: PipelineSpec) -> None:
    cfg = pipeline_spec.closure
    assert (cfg.rho, cfg.b_closed, cfg.b_span_min, cfg.w) == (0.35, 0.6, 0.2, 0.5)
    # a = (0.30 − 0.15) / (0.30 * 0.65) = 0.769231, s = (0.5 − 0.2) / max(0.6 − 0.2, 0.2) = 0.75
    # c = 0.5 * 0.769231 + 0.5 * 0.75 = 0.759615
    assert calibrate.eye_closure(0.15, 0.5, 0.30, 0.20, cfg) == pytest.approx(0.759615, abs=1e-6)
    # 뜬 눈 기준: a = 0, s = 0
    assert calibrate.eye_closure(0.30, 0.20, 0.30, 0.20, cfg) == 0.0
    # 기준보다 더 뜬 눈은 0으로 자른다.
    assert calibrate.eye_closure(0.40, 0.05, 0.30, 0.20, cfg) == 0.0
    # 완전히 감긴 눈: ear <= E * rho = 0.105이면 a = 1, blink >= 0.6이면 s = 1
    assert calibrate.eye_closure(0.105, 0.60, 0.30, 0.20, cfg) == pytest.approx(1.0, abs=1e-12)
    assert calibrate.eye_closure(0.0, 0.95, 0.30, 0.20, cfg) == 1.0
    # EAR 쪽만 감김: 0.5 * 1 + 0.5 * 0
    assert calibrate.eye_closure(0.0, 0.20, 0.30, 0.20, cfg) == 0.5
    # 블렌드셰이프 분모의 하한: B = 0.5이면 max(0.1, 0.2) = 0.2 -> s = (0.6 − 0.5) / 0.2 = 0.5 -> c = 0.25
    assert calibrate.eye_closure(0.30, 0.60, 0.30, 0.50, cfg) == pytest.approx(0.25, abs=1e-12)


def test_eye_closure_with_a_zero_ear_baseline_does_not_divide(pipeline_spec: PipelineSpec) -> None:
    # E·(1 − rho)가 0이면 EAR 쪽 감김은 0이다(0으로 나누지 않는다). c = 0.5·0 + 0.5·s, s = (0.5 − 0.2) / 0.4 = 0.75.
    # 파이프라인이 만든 기준선에서는 E가 e_min 이상이라 이 자리에 오지 않는다. 넣어 준 기준선의 E가 0일 때만 온다.
    cfg = pipeline_spec.closure
    assert calibrate.eye_closure(0.15, 0.5, 0.0, 0.20, cfg) == pytest.approx(0.375, abs=1e-12)
    assert calibrate.eye_closure(0.0, 0.2, 0.0, 0.20, cfg) == 0.0


def test_eye_closure_weight_w(pipeline_spec: PipelineSpec) -> None:
    # a = 0.769231, s = 0.75. W = 1이면 EAR 쪽만, W = 0이면 블렌드셰이프 쪽만 쓴다(E1의 비교 조건).
    ear_only = dataclasses.replace(pipeline_spec.closure, w=1.0)
    blink_only = dataclasses.replace(pipeline_spec.closure, w=0.0)
    assert calibrate.eye_closure(0.15, 0.5, 0.30, 0.20, ear_only) == pytest.approx(0.15 / 0.195, abs=1e-12)
    assert calibrate.eye_closure(0.15, 0.5, 0.30, 0.20, blink_only) == pytest.approx(0.75, abs=1e-12)


def test_frame_state_both_eyes_frontal(pipeline_spec: PipelineSpec) -> None:
    baseline = make_baseline(e_l=0.30, e_r=0.28, b_l=0.20, b_r=0.10)
    s = sig(4242, ear_l=0.15, blink_l=0.5, ear_r=0.28, blink_r=0.10)
    state = calibrate.frame_state(s, baseline, pipeline_spec, False)
    assert state.t_ms == 4242
    assert state.face is True
    assert state.closure_valid is True
    assert state.provisional is False
    # c_l = 0.759615, c_r = 0, yaw_rel = 0 -> w_far = 1 -> 두 눈 평균 0.379808
    assert state.closure == pytest.approx(0.759615 / 2.0, abs=1e-6)
    assert state.yaw_rel == pytest.approx(0.0, abs=1e-12)
    assert state.pitch_rel == pytest.approx(0.0, abs=1e-12)


def test_far_eye_weighting_at_0_20_30_degrees(pipeline_spec: PipelineSpec) -> None:
    baseline = make_baseline()
    # L 눈은 완전히 감김(c_l = 1), R 눈은 뜸(c_r = 0)
    def closure_at(yaw: float) -> float:
        s = sig(1, ear_l=0.0, blink_l=0.9, ear_r=0.30, blink_r=0.20, yaw=yaw)
        state = calibrate.frame_state(s, baseline, pipeline_spec, False)
        assert state.closure_valid is True
        assert state.yaw_rel == pytest.approx(yaw, abs=1e-9)
        return state.closure

    # 0도: w_far = 1 -> (0 + 1) / 2
    assert closure_at(0.0) == pytest.approx(0.5, abs=1e-12)
    # 15도까지는 두 눈 평균
    assert closure_at(15.0) == pytest.approx(0.5, abs=1e-9)
    assert closure_at(-15.0) == pytest.approx(0.5, abs=1e-9)
    # +20도(피험자가 자기 왼쪽으로 돌림): 먼 쪽은 L, w_far = 0.5 -> (c_r + 0.5*c_l) / 1.5 = 1/3
    assert closure_at(20.0) == pytest.approx(1.0 / 3.0, abs=1e-9)
    # −20도: 먼 쪽은 R, w_far = 0.5 -> (c_l + 0.5*c_r) / 1.5 = 2/3
    assert closure_at(-20.0) == pytest.approx(2.0 / 3.0, abs=1e-9)
    # +30도: w_far = 0 -> 가까운 눈(R)만 = 0, −30도: 가까운 눈(L)만 = 1
    assert closure_at(30.0) == pytest.approx(0.0, abs=1e-12)
    assert closure_at(-30.0) == pytest.approx(1.0, abs=1e-12)
    # 25도부터는 가까운 눈만 쓴다.
    assert closure_at(25.5) == pytest.approx(0.0, abs=1e-12)


def test_far_eye_weighting_uses_yaw_relative_to_baseline(pipeline_spec: PipelineSpec) -> None:
    # 기준 yaw 10도, 현재 yaw 30도 -> yaw_rel = 20 -> w_far = 0.5 (절대 각도 30도였다면 가까운 눈만 썼을 것이다)
    baseline = make_baseline(yaw0=10.0)
    s = sig(1, ear_l=0.0, blink_l=0.9, ear_r=0.30, blink_r=0.20, yaw=30.0)
    state = calibrate.frame_state(s, baseline, pipeline_spec, False)
    assert state.yaw_rel == pytest.approx(20.0, abs=1e-9)
    assert state.closure == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_relative_pose_is_not_the_angle_difference(pipeline_spec: PipelineSpec) -> None:
    baseline = make_baseline(yaw0=30.0, pitch0=15.0, roll0=0.0)
    # 기준 자세에서 pitch만 10도 더 숙였다: R = R0 · Rx(10)
    r0 = signals.rotation_from_euler(30.0, 15.0, 0.0)
    delta = signals.rotation_from_euler(0.0, 10.0, 0.0)
    product = tuple(sum(r0[3 * i + k] * delta[3 * k + j] for k in range(3)) for i in range(3) for j in range(3))
    yaw, pitch, roll = signals.euler_from_rotation(product)
    state = calibrate.frame_state(sig(1, yaw=yaw, pitch=pitch, roll=roll), baseline, pipeline_spec, False)
    assert state.yaw_rel == pytest.approx(0.0, abs=1e-9)
    assert state.pitch_rel == pytest.approx(10.0, abs=1e-9)
    # 기준 자세 그대로면 0
    same = calibrate.frame_state(sig(1, yaw=30.0, pitch=15.0, roll=0.0), baseline, pipeline_spec, False)
    assert same.yaw_rel == pytest.approx(0.0, abs=1e-9)
    assert same.pitch_rel == pytest.approx(0.0, abs=1e-9)


def test_one_usable_eye_is_used_as_is(pipeline_spec: PipelineSpec) -> None:
    baseline = make_baseline()
    # R 눈을 쓸 수 없다(폭 3 px). yaw가 +20도여서 L이 먼 쪽이지만 가중 없이 c_l을 그대로 쓴다.
    s = sig(1, ear_l=0.15, blink_l=0.5, ear_r=0.0, eye_w_r=3.0, blink_r=0.9, yaw=20.0)
    state = calibrate.frame_state(s, baseline, pipeline_spec, False)
    assert state.closure_valid is True
    assert state.closure == pytest.approx(0.759615, abs=1e-6)
    # L 눈을 쓸 수 없으면 c_r
    s = sig(1, ear_l=0.0, eye_w_l=5.9, blink_l=0.9, ear_r=0.30, blink_r=0.20, yaw=-20.0)
    state = calibrate.frame_state(s, baseline, pipeline_spec, False)
    assert state.closure_valid is True
    assert state.closure == 0.0
    # 폭이 정확히 6 px이면 쓸 수 있는 눈이다.
    s = sig(1, ear_l=0.0, eye_w_l=6.0, blink_l=0.9, ear_r=0.30, blink_r=0.20)
    assert calibrate.frame_state(s, baseline, pipeline_spec, False).closure == pytest.approx(0.5, abs=1e-12)
    # Right 눈도 같다: 감긴 Right 눈(c = 1)과 뜬 Left 눈(c = 0)의 평균 0.5. 쓸 수 없는 눈으로 치면 Left의 0.0이 된다.
    s = sig(1, ear_l=0.30, blink_l=0.20, ear_r=0.0, eye_w_r=6.0, blink_r=0.9)
    assert calibrate.frame_state(s, baseline, pipeline_spec, False).closure == pytest.approx(0.5, abs=1e-12)


def test_validity_conditions(pipeline_spec: PipelineSpec) -> None:
    baseline = make_baseline(jaw0=0.05, gaze_h0=0.1, gaze_v0=-0.1)
    closed = dict(ear_l=0.0, blink_l=0.9, ear_r=0.0, blink_r=0.9)

    # 쓸 수 있는 눈이 없다 -> 무효. closure는 0.0이고 자세와 시선 값은 그대로 채운다.
    s = sig(1, eye_w_l=2.0, eye_w_r=2.0, yaw=10.0, pitch=4.0, jaw_open=0.25, eye_look=EYE_LOOK, **closed)
    state = calibrate.frame_state(s, baseline, pipeline_spec, False)
    assert state.face is True
    assert state.closure_valid is False
    assert state.closure == 0.0
    assert state.pose_valid is True
    assert state.yaw_rel == pytest.approx(10.0, abs=1e-9)
    assert state.pitch_rel == pytest.approx(4.0, abs=1e-9)
    assert state.jaw_rel == pytest.approx(0.20, abs=1e-12)
    assert state.gaze_h_rel == pytest.approx(0.375, abs=1e-12)
    assert state.gaze_v_rel == pytest.approx(0.05, abs=1e-12)

    # |yaw_rel| <= 40도까지 유효
    assert calibrate.frame_state(sig(1, yaw=39.9, **closed), baseline, pipeline_spec, False).closure_valid is True
    assert calibrate.frame_state(sig(1, yaw=-39.9, **closed), baseline, pipeline_spec, False).closure_valid is True
    # 정확히 40도도 유효하다('이하'). 기준 회전이 단위 행렬이면 yaw 40.0의 yaw_rel은 정확히 40.0으로 되돌아온다.
    for yaw in (40.0, -40.0):
        state = calibrate.frame_state(sig(1, yaw=yaw, **closed), baseline, pipeline_spec, False)
        assert abs(state.yaw_rel) == pipeline_spec.closure.valid_max_abs_yaw_deg == 40.0
        assert state.closure_valid is True and state.closure == 1.0
    for yaw in (40.1, -40.1, 75.0):
        state = calibrate.frame_state(sig(1, yaw=yaw, **closed), baseline, pipeline_spec, False)
        assert state.closure_valid is False
        assert state.closure == 0.0
        assert state.yaw_rel == pytest.approx(yaw, abs=1e-9)

    # 얼굴 없음 -> 모든 실수 0.0, 두 유효 플래그 False
    state = calibrate.frame_state(no_face(999), baseline, pipeline_spec, True)
    assert state.t_ms == 999
    assert state.face is False
    assert state.closure_valid is False
    assert state.pose_valid is False
    assert state.provisional is True
    assert (state.closure, state.yaw_rel, state.pitch_rel) == (0.0, 0.0, 0.0)
    assert (state.gaze_h_rel, state.gaze_v_rel, state.jaw_rel) == (0.0, 0.0, 0.0)


def test_pose_valid_and_provisional_flags(pipeline_spec: PipelineSpec) -> None:
    s = sig(1, ear_l=0.0945, ear_r=0.0945, blink_l=0.60, blink_r=0.60, yaw=8.0, pitch=-2.0, jaw_open=0.30)

    # 기준선 완성 전(D4): 전역 상수로 계산한다. closure는 나오지만 시선 규칙용 자세는 쓸 수 없다.
    provisional = calibrate.frame_state(s, calibrate.global_baseline(pipeline_spec), pipeline_spec, True)
    assert provisional.provisional is True
    assert provisional.pose_valid is False
    assert provisional.closure_valid is True
    # E = 0.27: ear <= 0.27 * 0.35 = 0.0945 -> a = 1, B = 0.22: (0.60 − 0.22) / 0.38 -> s = 1
    assert provisional.closure == pytest.approx(1.0, abs=1e-9)
    # R0이 단위 행렬이므로 상대 각도는 절대 각도다.
    assert provisional.yaw_rel == pytest.approx(8.0, abs=1e-9)
    assert provisional.pitch_rel == pytest.approx(-2.0, abs=1e-9)
    assert provisional.jaw_rel == pytest.approx(0.30 - 0.08, abs=1e-12)

    # 보정이 전역 상수로 대체된 뒤: provisional은 False지만 pose_valid는 계속 False다.
    fallback = calibrate.frame_state(s, calibrate.global_baseline(pipeline_spec), pipeline_spec, False)
    assert fallback.provisional is False
    assert fallback.pose_valid is False

    # 개인 기준선(ok, off_axis)에서는 pose_valid가 True다.
    assert calibrate.frame_state(s, make_baseline(status=BASELINE_OK), pipeline_spec, False).pose_valid is True
    assert calibrate.frame_state(s, make_baseline(status=BASELINE_OFF_AXIS), pipeline_spec, False).pose_valid is True


def test_closure_is_mirror_invariant(pipeline_spec: PipelineSpec) -> None:
    baseline = make_baseline(
        e_l=0.33, e_r=0.29, b_l=0.08, b_r=0.21, yaw0=4.0, pitch0=-3.0, roll0=2.0, gaze_h0=0.03, gaze_v0=-0.02
    )
    mirrored_baseline = mirror_baseline(baseline)
    for yaw in (-36.0, -30.0, -22.0, -15.0, -4.0, 0.0, 4.0, 9.0, 18.0, 27.0, 33.0, 43.0, 50.0):
        s = sig(1, ear_l=0.31, ear_r=0.22, blink_l=0.10, blink_r=0.60, eye_look=EYE_LOOK, yaw=yaw, pitch=-7.0, roll=5.0)
        state = calibrate.frame_state(s, baseline, pipeline_spec, False)
        mirrored = calibrate.frame_state(mirror_signals(s), mirrored_baseline, pipeline_spec, False)
        # yaw 부호와 눈 집합이 함께 바뀌므로 closure가 비트 단위로 같다.
        assert mirrored.closure == state.closure
        assert mirrored.closure_valid == state.closure_valid
        assert mirrored.yaw_rel == -state.yaw_rel
        assert mirrored.pitch_rel == state.pitch_rel
        assert mirrored.gaze_h_rel == pytest.approx(-state.gaze_h_rel, abs=1e-12)
        assert mirrored.gaze_v_rel == state.gaze_v_rel
        assert mirrored.jaw_rel == state.jaw_rel


def test_closure_stays_finite_and_within_unit_range(pipeline_spec: PipelineSpec) -> None:
    baseline = make_baseline()
    for ear in (0.0, 0.1, 0.3, 0.9):
        for blink in (0.0, 0.2, 0.6, 1.0):
            for yaw in (-39.0, -20.0, 0.0, 20.0, 39.0):
                s = sig(1, ear_l=ear, ear_r=0.3 - ear / 3.0, blink_l=blink, blink_r=1.0 - blink, yaw=yaw)
                state = calibrate.frame_state(s, baseline, pipeline_spec, False)
                assert state.closure_valid is True
                assert 0.0 <= state.closure <= 1.0


# ---------------------------------------------------------------------------
# 이식 제약
# ---------------------------------------------------------------------------


def test_module_imports_standard_library_only(repo_root: Path) -> None:
    code = (
        "import sys\n"
        "import drowsy.calibrate\n"
        "banned = [name for name in ('numpy', 'pandas', 'yaml', 'torch') if name in sys.modules]\n"
        "assert not banned, banned\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
