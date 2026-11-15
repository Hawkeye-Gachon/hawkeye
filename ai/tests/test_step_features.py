"""`drowsy.step_features` 단위 테스트 (PLAN 3.3절, 4.4절, 4.6절, 10.1절 "스텝 특징").

`FrameState` 수열을 직접 만들어 넣는다. 감김 구간은 실제 특징용 `ClosureRunDetector`(진입 0.5 / 이탈 0.3)가
만든 것을 `StepAggregator`에 넘긴다(파이프라인과 같은 순서). 기댓값은 주석의 손 계산이다.
"""

from __future__ import annotations

import dataclasses
import math
import random

import pytest

from drowsy import step_features
from drowsy.blink import ClosureRunDetector
from drowsy.spec import PipelineSpec
from drowsy.step_features import NodDetector, StepAggregator, ThresholdEventCounter, yawn_counter
from drowsy.types import (
    F_BLINK_DUR_MEAN_60,
    F_BLINK_DUR_STD_60,
    F_BLINK_RATE_60,
    F_CLOSING_VEL_MEAN_60,
    F_CLOSURE_MEAN_10,
    F_CLOSURE_MEAN_60,
    F_FACE_MISSING_RATIO_10,
    F_FACE_MISSING_RATIO_60,
    F_LONG_CLOSURE_COUNT_60,
    F_MAX_CLOSURE_MS_10,
    F_MAX_CLOSURE_MS_60,
    F_NOD_COUNT_60,
    F_OPENING_VEL_MEAN_60,
    F_PERCLOS_10,
    F_PERCLOS_60,
    F_PITCH_STD_60,
    F_YAWN_COUNT_60,
    FEATURE_NAMES,
    N_FEATURES,
    ClosureRun,
    FrameState,
    StepFeatures,
)

# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def st(
    t_ms: int,
    closure: float = 0.1,
    *,
    valid: bool = True,
    face: bool = True,
    pitch: float = 0.0,
    jaw: float = 0.0,
) -> FrameState:
    """프레임 상태 하나. face=False이면 계약대로 모든 값이 0이고 두 유효 플래그가 False다."""
    if not face:
        return FrameState(
            t_ms=t_ms,
            face=False,
            closure=0.0,
            closure_valid=False,
            provisional=False,
            yaw_rel=0.0,
            pitch_rel=0.0,
            pose_valid=False,
            gaze_h_rel=0.0,
            gaze_v_rel=0.0,
            jaw_rel=0.0,
        )
    return FrameState(
        t_ms=t_ms,
        face=True,
        closure=closure if valid else 0.0,
        closure_valid=valid,
        provisional=False,
        yaw_rel=0.0,
        pitch_rel=pitch,
        pose_valid=True,
        gaze_h_rel=0.0,
        gaze_v_rel=0.0,
        jaw_rel=jaw,
    )


class Session:
    """특징용 검출기와 누적기를 파이프라인과 같은 순서로 돌린다."""

    def __init__(self, spec: PipelineSpec, start_ms: int = 0, origin_ms: int = 0) -> None:
        self.spec = spec
        self.detector = ClosureRunDetector(spec.run_detector)
        self.aggregator = StepAggregator(spec, start_ms, origin_ms)
        self.steps: list[StepFeatures] = []
        self.runs: list[ClosureRun] = []
        self._has_prev = False
        self._prev_t_ms = 0

    def feed(self, state: FrameState) -> list[StepFeatures]:
        if self._has_prev and state.t_ms - self._prev_t_ms > self.spec.clock.reset_gap_ms:
            self.detector.reset()
        self._has_prev = True
        self._prev_t_ms = state.t_ms
        run = self.detector.update(state.t_ms, state.closure, state.closure_valid)
        if run is not None:
            self.runs.append(run)
        steps = self.aggregator.update(state, run, self.detector.ongoing_ms())
        self.steps.extend(steps)
        return steps

    def feed_all(self, states: list[FrameState]) -> list[StepFeatures]:
        for state in states:
            self.feed(state)
        return self.steps

    def step(self, k: int) -> StepFeatures:
        for item in self.steps:
            if item.k == k:
                return item
        raise KeyError(k)


def open_frames(t_from: int, t_to: int, period: int = 100, closure: float = 0.1) -> list[FrameState]:
    """[t_from, t_to) 구간의 눈 뜬 프레임."""
    return [st(t, closure) for t in range(t_from, t_to, period)]


def approx(value: float) -> object:
    return pytest.approx(value, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# 특징 순서
# ---------------------------------------------------------------------------


def test_feature_order_is_fixed(pipeline_spec: PipelineSpec) -> None:
    assert pipeline_spec.step_features.names == FEATURE_NAMES
    assert N_FEATURES == 17
    expected = {
        "perclos_60": F_PERCLOS_60,
        "perclos_10": F_PERCLOS_10,
        "closure_mean_60": F_CLOSURE_MEAN_60,
        "closure_mean_10": F_CLOSURE_MEAN_10,
        "blink_rate_60": F_BLINK_RATE_60,
        "blink_dur_mean_60": F_BLINK_DUR_MEAN_60,
        "blink_dur_std_60": F_BLINK_DUR_STD_60,
        "closing_vel_mean_60": F_CLOSING_VEL_MEAN_60,
        "opening_vel_mean_60": F_OPENING_VEL_MEAN_60,
        "long_closure_count_60": F_LONG_CLOSURE_COUNT_60,
        "max_closure_ms_60": F_MAX_CLOSURE_MS_60,
        "max_closure_ms_10": F_MAX_CLOSURE_MS_10,
        "yawn_count_60": F_YAWN_COUNT_60,
        "pitch_std_60": F_PITCH_STD_60,
        "nod_count_60": F_NOD_COUNT_60,
        "face_missing_ratio_60": F_FACE_MISSING_RATIO_60,
        "face_missing_ratio_10": F_FACE_MISSING_RATIO_10,
    }
    for name, index in expected.items():
        assert FEATURE_NAMES[index] == name


# ---------------------------------------------------------------------------
# ThresholdEventCounter: 하품
# ---------------------------------------------------------------------------


def test_yawn_counts_once_when_duration_reaches_2000(pipeline_spec: PipelineSpec) -> None:
    counter = yawn_counter(pipeline_spec.yawn)
    counted_at: list[int] = []
    # 1000부터 0.5(≥ 0.40)로 시작. 3000에서 길이 2000에 도달한다. 4000까지 이어져도 한 번만 센다.
    for t in range(0, 5000, 100):
        value = 0.5 if 1000 <= t <= 4000 else 0.0
        if counter.update(t, value, True):
            counted_at.append(t)
    assert counted_at == [3000]
    assert counter.active is False


def test_yawn_shorter_than_2000_is_not_counted(pipeline_spec: PipelineSpec) -> None:
    counter = yawn_counter(pipeline_spec.yawn)
    counted = False
    # 1000~2900에서 0.5, 3000에서 0.0. 3000 프레임의 길이는 2000이지만 그 프레임에서 끝났으므로 세지 않는다.
    for t in range(0, 4000, 100):
        value = 0.5 if 1000 <= t <= 2900 else 0.0
        counted = counter.update(t, value, True) or counted
    assert counted is False


def test_yawn_hysteresis_and_thresholds(pipeline_spec: PipelineSpec) -> None:
    counter = yawn_counter(pipeline_spec.yawn)
    # 0.39는 시작 임계(0.40) 미만이다.
    assert counter.update(0, 0.39, True) is False
    assert counter.active is False
    # 정확히 0.40이면 시작한다.
    counter.update(100, 0.40, True)
    assert counter.active is True
    # 0.25 이상이면 이어진다(정확히 0.25는 "0.25 미만"이 아니다).
    counter.update(200, 0.25, True)
    assert counter.active is True
    counter.update(300, 0.249, True)
    assert counter.active is False
    # 0.25~0.40 사이에서는 새로 시작하지 않는다.
    counter.update(400, 0.3, True)
    assert counter.active is False


def test_yawn_survives_dip_between_thresholds(pipeline_spec: PipelineSpec) -> None:
    counter = yawn_counter(pipeline_spec.yawn)
    counted_at: list[int] = []
    for t in range(0, 3000, 100):
        value = 0.3 if 500 <= t < 1500 else 0.6  # 중간에 0.3으로 내려가도(≥ 0.25) 같은 이벤트다
        if counter.update(t, value, True):
            counted_at.append(t)
    assert counted_at == [2000]


def test_invalid_gap_over_500_cancels(pipeline_spec: PipelineSpec) -> None:
    counter = yawn_counter(pipeline_spec.yawn)
    counter.update(0, 0.5, True)
    counter.update(100, 0.5, True)
    # 무효 프레임이 이어져도 마지막 유효 프레임(100) 뒤 500 ms까지는 유지한다.
    assert counter.update(300, 0.0, False) is False
    assert counter.update(600, 0.0, False) is False
    assert counter.active is True
    # 601 ms째에 취소된다.
    assert counter.update(701, 0.0, False) is False
    assert counter.active is False


def test_gap_of_exactly_500_keeps_event_and_restart_after_cancel(pipeline_spec: PipelineSpec) -> None:
    counter = yawn_counter(pipeline_spec.yawn)
    counter.update(0, 0.5, True)
    # 무효 프레임 없이 프레임이 끊겨도 같은 규칙이다. 정확히 500 ms는 취소하지 않는다.
    assert counter.update(500, 0.5, True) is False
    assert counter.active is True
    # 501 ms 끊긴 뒤 온 유효 프레임: 먼저 취소하고, 그 프레임에서 새 이벤트를 시작한다(길이는 1001부터 다시 잰다).
    assert counter.update(1001, 0.5, True) is False
    assert counter.active is True
    assert counter.update(2900, 0.5, True) is False  # 1899 ms 끊김 -> 다시 취소하고 2900에서 새로 시작
    counted_at = [t for t in range(3000, 5100, 100) if counter.update(t, 0.5, True)]
    assert counted_at == [4900]


def test_reset_drops_event_in_progress(pipeline_spec: PipelineSpec) -> None:
    counter = yawn_counter(pipeline_spec.yawn)
    counter.update(0, 0.5, True)
    counter.reset()
    assert counter.active is False
    assert counter.update(100, 0.3, True) is False
    assert counter.active is False


# ---------------------------------------------------------------------------
# ThresholdEventCounter: 끝날 때 세는 방식(끄덕임)
# ---------------------------------------------------------------------------


def _end_counter() -> ThresholdEventCounter:
    return ThresholdEventCounter(
        enter_thr=10.0, exit_thr=5.0, min_duration_ms=200, max_duration_ms=3000, count_at_end=True, cancel_gap_ms=500
    )


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [(100, False), (199, False), (200, True), (1000, True), (3000, True), (3001, False)],
)
def test_count_at_end_duration_window(duration_ms: int, expected: bool) -> None:
    counter = _end_counter()
    assert counter.update(1000, 12.0, True) is False
    t = 1000
    # 500 ms 넘게 끊기지 않게 중간 프레임을 넣는다.
    while t + 400 < 1000 + duration_ms:
        t += 400
        assert counter.update(t, 12.0, True) is False
    # 길이는 (끝난 프레임 시각) − (시작한 프레임 시각)이다.
    assert counter.update(1000 + duration_ms, 0.0, True) is expected
    assert counter.active is False


def test_count_at_end_does_not_count_while_ongoing() -> None:
    counter = _end_counter()
    results = [counter.update(t, 12.0, True) for t in range(0, 2000, 100)]
    assert results == [False] * 20
    assert counter.active is True


def test_count_at_end_exit_threshold_is_strict() -> None:
    counter = _end_counter()
    counter.update(0, 10.0, True)  # 정확히 10이면 시작
    assert counter.active is True
    counter.update(100, 5.0, True)  # 정확히 5는 "5 미만"이 아니다
    assert counter.active is True
    assert counter.update(300, 4.99, True) is True


def test_count_at_end_cancelled_event_is_not_counted() -> None:
    counter = _end_counter()
    counter.update(0, 12.0, True)
    counter.update(100, 12.0, True)
    assert counter.update(700, 0.0, True) is False  # 600 ms 끊김 -> 취소. 이 프레임은 새 이벤트도 아니다
    assert counter.active is False


# ---------------------------------------------------------------------------
# NodDetector: 느린 평균
# ---------------------------------------------------------------------------


def test_nod_slow_mean_hand_computed(pipeline_spec: PipelineSpec) -> None:
    nod = NodDetector(pipeline_spec.nod)
    # 첫 유효 표본이 초기값이다.
    assert nod.update(0, 2.0, True) is False
    assert nod.mean == 2.0
    # dt = 100 -> α = 100 / 10100. m = 2 + α·(12.5 − 2)
    nod.update(100, 12.5, True)
    m1 = 2.0 + (100.0 / 10100.0) * 10.5
    assert nod.mean == approx(m1)
    # dt = 1000은 200으로 자른다 -> α = 200 / 10200
    nod.update(1100, 12.5, True)
    m2 = m1 + (200.0 / 10200.0) * (12.5 - m1)
    assert nod.mean == approx(m2)
    # 무효 프레임에서는 갱신하지 않는다.
    nod.update(1200, 99.0, False)
    assert nod.mean == approx(m2)
    # 다음 유효 표본의 dt는 직전 유효 표본(1100)부터 잰다: 150 ms
    nod.update(1250, 0.0, True)
    m3 = m2 + (150.0 / 10150.0) * (0.0 - m2)
    assert nod.mean == approx(m3)


def test_nod_d_is_computed_before_mean_update(pipeline_spec: PipelineSpec) -> None:
    nod = NodDetector(pipeline_spec.nod)
    nod.update(0, 0.0, True)
    # 10.05도로 뛴다. 갱신 전의 m(0)으로는 d = 10.05 ≥ 10이라 시작한다.
    # 갱신 뒤의 m(10.05·100/10100 ≈ 0.0995)으로 계산하면 d ≈ 9.95라서 시작하지 못한다.
    results = [nod.update(t, 10.05, True) for t in (100, 200, 300, 400)]
    assert results == [False, False, False, False]
    # 500에서 0도로 돌아오면 d = 0 − m < 5로 끝난다. 길이 400 ms(200~3000 안)이므로 센다.
    assert nod.update(500, 0.0, True) is True


def test_nod_counts_only_durations_between_200_and_3000(pipeline_spec: PipelineSpec) -> None:
    def run(duration_ms: int) -> bool:
        nod = NodDetector(pipeline_spec.nod)
        for t in range(0, 1000, 100):
            nod.update(t, 0.0, True)
        counted = False
        for t in range(1000, 1000 + duration_ms, 100):
            counted = nod.update(t, 15.0, True) or counted
        return nod.update(1000 + duration_ms, 0.0, True) or counted

    assert run(100) is False
    assert run(200) is True
    assert run(3000) is True
    # 3100 ms 동안 15도를 유지해도 느린 평균은 4.5도 못 미치게 오르므로 d는 계속 10 이상이다.
    assert run(3100) is False


def test_nod_follows_slow_posture_change(pipeline_spec: PipelineSpec) -> None:
    nod = NodDetector(pipeline_spec.nod)
    counted = False
    # 60초에 걸쳐 0도에서 12도로 천천히 숙이면 느린 평균이 따라가서 끄덕임으로 세지 않는다.
    for i in range(600):
        counted = nod.update(i * 100, 12.0 * i / 600.0, True) or counted
    assert counted is False
    assert 8.0 < nod.mean < 12.0


def test_nod_reset_forgets_mean(pipeline_spec: PipelineSpec) -> None:
    nod = NodDetector(pipeline_spec.nod)
    nod.update(0, 20.0, True)
    nod.reset()
    assert nod.mean == 0.0
    # 초기화 뒤 첫 유효 표본이 다시 초기값이 된다(d = 0이라 시작하지 않는다).
    nod.update(100, 15.0, True)
    assert nod.mean == 15.0
    assert nod.update(200, 15.0, True) is False


# ---------------------------------------------------------------------------
# StepAggregator: 손으로 계산한 짧은 세션
# ---------------------------------------------------------------------------


def scripted_states(origin_ms: int) -> list[FrameState]:
    """6.1초, 100 ms 간격. 세션 시간 기준으로
    - [200, 500): closure 0.9 (깜빡임 A)
    - [2000, 2500): closure 0.6 (깜빡임 B. 0.8 미만이라 PERCLOS에는 들어가지 않는다)
    - [3000, 3500): 얼굴 없음
    - [4000, 5000): pitch_rel 4도
    - 나머지: closure 0.1, pitch_rel 0
    """
    states: list[FrameState] = []
    for t in range(0, 6100, 100):
        if 3000 <= t < 3500:
            states.append(st(origin_ms + t, face=False))
            continue
        closure = 0.1
        if 200 <= t < 500:
            closure = 0.9
        elif 2000 <= t < 2500:
            closure = 0.6
        pitch = 4.0 if 4000 <= t < 5000 else 0.0
        states.append(st(origin_ms + t, closure, pitch=pitch))
    return states


def test_scripted_session_runs(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all(scripted_states(0))
    assert len(session.runs) == 2
    a, b = session.runs
    # A: 100(0.1)->200(0.9)에서 0.5를 위로: 100 + (0.4/0.8)·100 = 150. 400(0.9)->500(0.1)에서 아래로: 450.
    assert (a.start_ms, a.end_ms, a.duration_ms) == (150, 450, 300)
    assert a.peak_closing_velocity == approx(8.0)  # 0.8 / 0.1 s
    assert a.peak_opening_velocity == approx(8.0)
    # B: 1900(0.1)->2000(0.6): 1900 + (0.4/0.5)·100 = 1980. 2400(0.6)->2500(0.1): 2400 + (0.1/0.5)·100 = 2420.
    assert (b.start_ms, b.end_ms, b.duration_ms) == (1980, 2420, 440)
    assert b.peak_closing_velocity == approx(5.0)
    assert b.peak_opening_velocity == approx(5.0)
    assert a.velocity_valid and b.velocity_valid


def test_scripted_session_step_1(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec, origin_ms=7000)
    session.feed_all(scripted_states(7000))
    assert [s.k for s in session.steps] == [1, 2, 3, 4, 5, 6]
    step = session.step(1)
    assert step.t_end_ms == 8000  # 원점 7000 + 1000·k
    assert step.full_lookback is False
    v = step.values
    assert len(v) == N_FEATURES
    # 스텝 1 = [0, 1000). 유효 시간 1000 ms는 하한(5000, 2000) 미만이라 PERCLOS와 깜빡임 빈도는 0이다.
    assert v[F_PERCLOS_60] == 0.0
    assert v[F_PERCLOS_10] == 0.0
    assert v[F_BLINK_RATE_60] == 0.0
    # closure 평균: (0.9·300 + 0.1·700) / 1000 = 0.34
    assert v[F_CLOSURE_MEAN_60] == approx(0.34)
    assert v[F_CLOSURE_MEAN_10] == approx(0.34)
    # 깜빡임 A(300 ms)는 프레임 500이 발행했으므로 스텝 1에 귀속된다.
    assert v[F_BLINK_DUR_MEAN_60] == 300.0
    assert v[F_BLINK_DUR_STD_60] == 0.0  # 이벤트가 2개 미만
    assert v[F_CLOSING_VEL_MEAN_60] == approx(8.0)
    assert v[F_OPENING_VEL_MEAN_60] == approx(8.0)
    assert v[F_LONG_CLOSURE_COUNT_60] == 0.0
    assert v[F_MAX_CLOSURE_MS_60] == 300.0
    assert v[F_MAX_CLOSURE_MS_10] == 300.0
    assert v[F_YAWN_COUNT_60] == 0.0
    assert v[F_PITCH_STD_60] == 0.0  # 얼굴 시간 1000 ms < 5000
    assert v[F_NOD_COUNT_60] == 0.0
    # 누적 시작 전의 시간은 데이터 없는 시간이다: (60000 − 1000) / 60000, (10000 − 1000) / 10000
    assert v[F_FACE_MISSING_RATIO_60] == approx(59000.0 / 60000.0)
    assert v[F_FACE_MISSING_RATIO_10] == approx(0.9)


def test_scripted_session_step_6(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all(scripted_states(0))
    v = session.step(6).values
    # 유효 시간: 6000 − 500(얼굴 없음) = 5500. closure ≥ 0.8인 시간: 300.
    assert v[F_PERCLOS_60] == approx(300.0 / 5500.0)
    assert v[F_PERCLOS_10] == approx(300.0 / 5500.0)
    # Σ closure·ms: 340 + 100 + (0.6·500 + 0.1·500) + 0.1·500 + 100 + 100 = 1040
    assert v[F_CLOSURE_MEAN_60] == approx(1040.0 / 5500.0)
    assert v[F_CLOSURE_MEAN_10] == approx(1040.0 / 5500.0)
    # 이벤트 2개 / 유효 5500 ms -> 2·60000/5500 회/분
    assert v[F_BLINK_RATE_60] == approx(2.0 * 60000.0 / 5500.0)
    assert v[F_BLINK_DUR_MEAN_60] == approx(370.0)  # (300 + 440) / 2
    assert v[F_BLINK_DUR_STD_60] == approx(70.0)  # 모표준편차
    assert v[F_CLOSING_VEL_MEAN_60] == approx(6.5)  # (8 + 5) / 2
    assert v[F_OPENING_VEL_MEAN_60] == approx(6.5)
    assert v[F_LONG_CLOSURE_COUNT_60] == 0.0
    assert v[F_MAX_CLOSURE_MS_60] == 440.0
    assert v[F_MAX_CLOSURE_MS_10] == 440.0
    assert v[F_YAWN_COUNT_60] == 0.0
    # pitch: 얼굴 시간 5500 ms 중 1000 ms가 4도. 평균 4000/5500, 제곱 평균 16000/5500.
    mean = 4000.0 / 5500.0
    assert v[F_PITCH_STD_60] == approx(math.sqrt(16000.0 / 5500.0 - mean * mean))
    assert v[F_NOD_COUNT_60] == 0.0
    assert v[F_FACE_MISSING_RATIO_60] == approx(54500.0 / 60000.0)
    assert v[F_FACE_MISSING_RATIO_10] == approx(0.45)


def test_scripted_session_intermediate_steps(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all(scripted_states(0))
    # 스텝 3: 유효 3000 ms(하한 5000 미만) -> perclos_60 = 0, perclos_10 = 300/3000 (10초 구간 하한은 2000)
    v3 = session.step(3).values
    assert v3[F_PERCLOS_60] == 0.0
    assert v3[F_PERCLOS_10] == approx(0.1)
    assert v3[F_BLINK_RATE_60] == 0.0
    assert v3[F_BLINK_DUR_MEAN_60] == approx(370.0)
    # 스텝 4: 얼굴 없는 500 ms는 유효 시간에도 얼굴 시간에도 들어가지 않는다.
    v4 = session.step(4).values
    assert v4[F_CLOSURE_MEAN_60] == approx((340.0 + 100.0 + 350.0 + 50.0) / 3500.0)
    assert v4[F_FACE_MISSING_RATIO_10] == approx(0.65)
    # 스텝 5: 얼굴 시간 4500 ms < 5000 -> pitch_std는 아직 0
    assert session.step(5).values[F_PITCH_STD_60] == 0.0


# ---------------------------------------------------------------------------
# 시간 규칙: 경계, 유지 상한, 정확한 분할
# ---------------------------------------------------------------------------


def test_frame_exactly_on_boundary_emits_step_and_belongs_to_next(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    assert session.feed(st(0)) == []
    assert session.feed(st(999)) == []
    emitted = session.feed(st(1000, 1.0))
    assert [s.k for s in emitted] == [1]
    # 1000에 온 프레임(closure 1.0)은 스텝 1에 들어가지 않는다.
    assert emitted[0].values[F_CLOSURE_MEAN_60] == approx(0.1)
    assert session.feed(st(1100, 1.0)) == []
    assert session.feed(st(1999, 1.0)) == []
    emitted = session.feed(st(2000, 1.0))
    assert [s.k for s in emitted] == [2]
    # 스텝 2: 1000~1100 100 ms, 1100~1300 200 ms(유지 상한), 1999~2000 1 ms 모두 closure 1.0
    assert emitted[0].values[F_CLOSURE_MEAN_60] == approx((0.1 * 201.0 + 1.0 * 301.0) / 502.0)


def test_hold_is_capped_and_split_exactly_at_boundary(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    for state in (st(0, 0.0), st(900, 1.0), st(1050, 0.0), st(2000, 0.0)):
        session.feed(state)
    v1 = session.step(1).values
    v2 = session.step(2).values
    # 스텝 1: 프레임 0은 200 ms만 인정(0~200), 프레임 900은 900~1050 중 경계 앞 100 ms.
    assert v1[F_CLOSURE_MEAN_60] == approx(100.0 / 300.0)
    assert v1[F_FACE_MISSING_RATIO_10] == approx((10000.0 - 300.0) / 10000.0)
    # 스텝 2: 프레임 900의 나머지 50 ms(closure 1.0)와 프레임 1050의 200 ms(closure 0.0).
    assert v2[F_CLOSURE_MEAN_10] == approx((100.0 + 50.0) / 550.0)
    assert v2[F_FACE_MISSING_RATIO_10] == approx((10000.0 - 550.0) / 10000.0)


def test_sparse_frames_count_missing_time(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    # 500 ms 간격: 프레임마다 200 ms만 인정되고 300 ms는 데이터 없음이다.
    session.feed_all([st(t) for t in range(0, 12001, 500)])
    v = session.step(12).values
    assert v[F_FACE_MISSING_RATIO_10] == approx(0.6)
    # 60초 구간에는 12초 치 통만 있다: 얼굴 시간 12·400 = 4800 ms
    assert v[F_FACE_MISSING_RATIO_60] == approx((60000.0 - 4800.0) / 60000.0)
    assert v[F_CLOSURE_MEAN_10] == approx(0.1)


def test_non_increasing_timestamp_is_rejected(pipeline_spec: PipelineSpec) -> None:
    aggregator = StepAggregator(pipeline_spec, 0, 0)
    aggregator.update(st(100), None, 0)
    with pytest.raises(ValueError):
        aggregator.update(st(100), None, 0)
    with pytest.raises(ValueError):
        aggregator.update(st(50), None, 0)


def test_start_must_be_on_step_boundary(pipeline_spec: PipelineSpec) -> None:
    with pytest.raises(ValueError):
        StepAggregator(pipeline_spec, 60500, 0)
    with pytest.raises(ValueError):
        StepAggregator(pipeline_spec, -1000, 0)


# ---------------------------------------------------------------------------
# 초기 구간과 full_lookback
# ---------------------------------------------------------------------------


def test_steps_start_at_t_b_and_full_lookback_flag(pipeline_spec: PipelineSpec) -> None:
    origin = 5000
    session = Session(pipeline_spec, start_ms=60000, origin_ms=origin)
    # T_b 이전의 프레임은 누적하지 않는다.
    assert session.aggregator.update(st(origin + 59000), None, 0) == []
    assert session.aggregator.update(st(origin + 59900), None, 0) == []
    session.feed_all(open_frames(origin + 60000, origin + 122001))
    ks = [s.k for s in session.steps]
    assert ks == list(range(61, 123))
    first = session.steps[0]
    assert first.t_end_ms == origin + 61000
    assert first.full_lookback is False
    # 스텝 끝 ≥ T_b + 60초인 첫 스텝은 k = 120이다.
    assert [s.k for s in session.steps if s.full_lookback][0] == 120
    assert session.step(119).full_lookback is False
    # 조회 구간이 덜 찬 동안: 얼굴 없는 비율은 통이 없는 시간만큼 크다.
    assert first.values[F_FACE_MISSING_RATIO_60] == approx(59000.0 / 60000.0)
    assert session.step(70).values[F_FACE_MISSING_RATIO_10] == approx(0.0)
    assert session.step(70).values[F_FACE_MISSING_RATIO_60] == approx(50000.0 / 60000.0)
    assert session.step(120).values[F_FACE_MISSING_RATIO_60] == approx(0.0)


def test_perclos_minimum_valid_time(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    # closure 0.9가 계속된다(감김 구간은 끝나지 않으므로 이벤트는 없다).
    session.feed_all([st(t, 0.9) for t in range(0, 6001, 100)])
    # 10초 구간: 유효 2000 ms부터 정의된다.
    assert session.step(1).values[F_PERCLOS_10] == 0.0
    assert session.step(2).values[F_PERCLOS_10] == 1.0
    # 60초 구간: 유효 5000 ms부터 정의된다.
    assert session.step(4).values[F_PERCLOS_60] == 0.0
    assert session.step(5).values[F_PERCLOS_60] == 1.0
    # closure 평균은 유효 시간이 조금이라도 있으면 정의된다.
    assert session.step(1).values[F_CLOSURE_MEAN_60] == approx(0.9)


def test_perclos_threshold_is_inclusive(pipeline_spec: PipelineSpec) -> None:
    thr = pipeline_spec.step_features.perclos_thr
    session = Session(pipeline_spec)
    session.feed_all([st(t, thr if t < 3000 else 0.79) for t in range(0, 6001, 100)])
    # 정확히 0.8인 3000 ms는 세고 0.79인 3000 ms는 세지 않는다.
    assert session.step(6).values[F_PERCLOS_60] == approx(0.5)


def test_first_frame_later_than_start_emits_empty_steps(pipeline_spec: PipelineSpec) -> None:
    aggregator = StepAggregator(pipeline_spec, 60000, 0)
    # 첫 프레임이 T_b보다 3.5초 늦게 왔다: 그 사이에 끝난 스텝 61, 62, 63을 빈 스텝으로 발행한다.
    steps = aggregator.update(st(63500), None, 0)
    assert [s.k for s in steps] == [61, 62, 63]
    for step in steps:
        assert step.values[F_FACE_MISSING_RATIO_10] == 1.0
        assert step.values[F_CLOSURE_MEAN_60] == 0.0


def test_first_frame_more_than_10s_after_start_restarts(pipeline_spec: PipelineSpec) -> None:
    aggregator = StepAggregator(pipeline_spec, 60000, 0)
    assert aggregator.update(st(70001), None, 0) == []
    steps = aggregator.update(st(71000), None, 0)
    assert [s.k for s in steps] == [71]


# ---------------------------------------------------------------------------
# 깜빡임 없는 구간, 창에서 빠져나가는 이벤트
# ---------------------------------------------------------------------------


def test_no_blink_interval(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all(open_frames(0, 70001))
    for k in (10, 60, 70):
        v = session.step(k).values
        assert v[F_PERCLOS_60] == 0.0
        assert v[F_PERCLOS_10] == 0.0
        assert v[F_CLOSURE_MEAN_60] == approx(0.1)
        assert v[F_CLOSURE_MEAN_10] == approx(0.1)
        assert v[F_BLINK_RATE_60] == 0.0
        assert v[F_BLINK_DUR_MEAN_60] == 0.0
        assert v[F_BLINK_DUR_STD_60] == 0.0
        assert v[F_CLOSING_VEL_MEAN_60] == 0.0
        assert v[F_OPENING_VEL_MEAN_60] == 0.0
        assert v[F_LONG_CLOSURE_COUNT_60] == 0.0
        assert v[F_MAX_CLOSURE_MS_60] == 0.0
        assert v[F_MAX_CLOSURE_MS_10] == 0.0
        assert v[F_FACE_MISSING_RATIO_10] == approx(0.0)
    assert session.step(60).full_lookback is True
    assert session.step(60).values[F_FACE_MISSING_RATIO_60] == approx(0.0)


def test_event_leaves_the_windows(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    states = [st(t, 0.9 if 200 <= t < 500 else 0.1) for t in range(0, 62001, 100)]
    session.feed_all(states)
    # 스텝 1에 귀속된 깜빡임(300 ms)은 10초 구간에는 스텝 10까지, 60초 구간에는 스텝 60까지 들어 있다.
    assert session.step(10).values[F_MAX_CLOSURE_MS_10] == 300.0
    assert session.step(11).values[F_MAX_CLOSURE_MS_10] == 0.0
    assert session.step(60).values[F_MAX_CLOSURE_MS_60] == 300.0
    assert session.step(60).values[F_BLINK_RATE_60] == approx(1.0)  # 1회 / 60초
    assert session.step(60).values[F_PERCLOS_60] == approx(300.0 / 60000.0)
    assert session.step(61).values[F_MAX_CLOSURE_MS_60] == 0.0
    assert session.step(61).values[F_BLINK_RATE_60] == 0.0
    assert session.step(61).values[F_BLINK_DUR_MEAN_60] == 0.0
    assert session.step(61).values[F_PERCLOS_60] == 0.0


# ---------------------------------------------------------------------------
# 감김 구간의 귀속, 진행 중인 감김
# ---------------------------------------------------------------------------


def test_run_is_attributed_to_the_step_of_the_publishing_frame(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    # 700~900에서 감고 프레임 1000(closure 0.1)이 구간을 발행한다. 1000은 스텝 2에 속한다.
    session.feed_all([st(t, 0.9 if 700 <= t < 1000 else 0.1) for t in range(0, 3001, 100)])
    assert len(session.runs) == 1
    assert session.runs[0].duration_ms == 300  # 650 ~ 950
    v1 = session.step(1).values
    v2 = session.step(2).values
    # 스텝 1: 귀속된 이벤트는 없고, 진행 중인 감김(프레임 900 기준 900 − 650 = 250 ms)만 보인다.
    assert v1[F_BLINK_DUR_MEAN_60] == 0.0
    assert v1[F_MAX_CLOSURE_MS_60] == 250.0
    # 스텝 2: 발행된 구간이 귀속된다.
    assert v2[F_BLINK_DUR_MEAN_60] == 300.0
    assert v2[F_MAX_CLOSURE_MS_60] == 300.0
    assert v2[F_CLOSING_VEL_MEAN_60] == approx(8.0)


def test_long_closure_across_steps_has_no_gap_and_no_double_count(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    # 300~2700에서 감는다. start = 250(200과 300 사이 보간), end = 2750, 길이 2500. 프레임 2800이 발행한다.
    session.feed_all([st(t, 0.9 if 300 <= t < 2800 else 0.1) for t in range(0, 5001, 100)])
    assert [r.duration_ms for r in session.runs] == [2500]
    maxes = [session.step(k).values[F_MAX_CLOSURE_MS_60] for k in (1, 2, 3, 4)]
    # 스텝 1, 2는 그 스텝의 마지막 프레임(900, 1900)에서의 진행 길이, 스텝 3부터는 발행된 구간의 전체 길이.
    assert maxes == [650.0, 1650.0, 2500.0, 2500.0]
    longs = [session.step(k).values[F_LONG_CLOSURE_COUNT_60] for k in (1, 2, 3, 4)]
    assert longs == [1.0, 1.0, 1.0, 1.0]
    # 지속 시간 평균은 상한 2000을 적용한다. 최대 감김 길이에는 적용하지 않는다.
    assert session.step(3).values[F_BLINK_DUR_MEAN_60] == 2000.0


def test_run_ending_on_the_emitting_frame_is_still_seen_as_ongoing(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    # 300~900에서 감고, 스텝 1을 발행시키는 프레임 1000에서 구간이 끝난다(길이 700, 스텝 2에 귀속).
    session.feed_all([st(t, 0.9 if 300 <= t < 1000 else 0.1) for t in range(0, 2001, 100)])
    assert [r.duration_ms for r in session.runs] == [700]
    # 스텝 1은 직전 프레임(900)의 진행 길이 650을 본다. 발행시킨 프레임의 값(0)을 쓰면 여기서 사라진다.
    assert session.step(1).values[F_MAX_CLOSURE_MS_60] == 650.0
    assert session.step(1).values[F_LONG_CLOSURE_COUNT_60] == 1.0
    assert session.step(2).values[F_MAX_CLOSURE_MS_60] == 700.0
    assert session.step(2).values[F_LONG_CLOSURE_COUNT_60] == 1.0


def test_truncated_run_counts_for_length_features_only(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    states: list[FrameState] = []
    for t in range(0, 4001, 100):
        if 300 <= t < 1100:
            states.append(st(t, 0.9))
        elif 1100 <= t < 2000:
            states.append(st(t, face=False))  # 감은 채로 얼굴이 사라진다
        else:
            states.append(st(t, 0.1))
    session.feed_all(states)
    assert len(session.runs) == 1
    run = session.runs[0]
    # start = 250, 잘린 구간의 end는 마지막 감김 프레임 1000. 프레임 1600(끊긴 시간 600 > 500)이 발행한다.
    assert (run.start_ms, run.end_ms, run.truncated) == (250, 1000, True)
    v = session.step(3).values
    assert v[F_LONG_CLOSURE_COUNT_60] == 1.0
    assert v[F_MAX_CLOSURE_MS_60] == 750.0
    # 잘린 구간은 "이벤트"가 아니다: 빈도, 지속 시간, 속도에 들어가지 않는다.
    assert v[F_BLINK_DUR_MEAN_60] == 0.0
    assert v[F_CLOSING_VEL_MEAN_60] == 0.0
    assert v[F_BLINK_DUR_STD_60] == 0.0


def test_max_closure_is_capped_at_10000(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all([st(t, 0.9) for t in range(0, 13001, 100)])
    # 진행 중인 감김: 스텝 10의 마지막 프레임 9900에서 9900 ms, 스텝 11부터는 상한 10000.
    assert session.step(10).values[F_MAX_CLOSURE_MS_60] == 9900.0
    assert session.step(11).values[F_MAX_CLOSURE_MS_60] == 10000.0
    assert session.step(13).values[F_MAX_CLOSURE_MS_10] == 10000.0
    assert session.step(13).values[F_LONG_CLOSURE_COUNT_60] == 1.0


def test_velocity_mean_uses_only_velocity_valid_events(pipeline_spec: PipelineSpec) -> None:
    aggregator = StepAggregator(pipeline_spec, 0, 0)
    with_velocity = ClosureRun(100, 300, 200, 0.9, 6.0, 4.0, True, False)
    without_velocity = ClosureRun(500, 620, 120, 0.9, 0.0, 0.0, False, False)
    aggregator.update(st(400), with_velocity, 0)
    aggregator.update(st(700), without_velocity, 0)
    v = aggregator.update(st(1000), None, 0)[0].values
    assert v[F_BLINK_DUR_MEAN_60] == 160.0  # 두 이벤트 모두 지속 시간에는 들어간다
    assert v[F_BLINK_DUR_STD_60] == 40.0
    assert v[F_CLOSING_VEL_MEAN_60] == 6.0  # 속도는 유효한 이벤트 하나의 값
    assert v[F_OPENING_VEL_MEAN_60] == 4.0


# ---------------------------------------------------------------------------
# 반쯤 뜬 채로 머문 시간은 감김 길이에 들어가지 않는다 (2026-11-13 결정)
# ---------------------------------------------------------------------------


def _lingering_lids(closed_frames: int) -> list[FrameState]:
    """깜빡임(0.9, closed_frames 프레임), 0.4에 머묾(6300 직전까지), 깜빡임(0.9, closed_frames 프레임), 뜬 눈. 100 ms 간격."""
    first = range(1000, 1000 + 100 * closed_frames, 100)
    second = range(6300, 6300 + 100 * closed_frames, 100)
    states: list[FrameState] = []
    for t in range(0, 18001, 100):
        if t in first or t in second:
            states.append(st(t, 0.9))
        elif first[-1] < t < 6300:
            states.append(st(t, 0.4))  # 이탈 0.3과 진입 0.5 사이: 구간은 끝나지 않는다
        else:
            states.append(st(t, 0.1))
    return states


def test_lingering_lids_between_two_blinks_add_no_length(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all(_lingering_lids(2))
    # 두 깜빡임은 여전히 한 구간이다(눈꺼풀이 0.3 아래로 내려간 적이 없다). 길이는 0.5 이상이었던 두 토막의 합이다.
    # 첫 토막: 950(= 900 + (0.4/0.8)*100) ~ 1180(= 1100 + (0.4/0.5)*100), 230 ms
    # 둘째 토막: 6220(= 6200 + (0.1/0.5)*100) ~ 6450(= 6400 + (0.4/0.8)*100), 230 ms
    # 2026-11-13 결정 전에는 6450 − 950 = 5500 ms였다.
    assert [(r.start_ms, r.end_ms, r.duration_ms, r.truncated) for r in session.runs] == [(950, 6450, 460, False)]

    # 머무는 동안(스텝 2 ~ 6): 진행 중인 길이는 0.5 이상이었던 마지막 프레임(1100)까지인 150 ms에서 멈춰 있다.
    for k in (2, 6):
        v = session.step(k).values
        assert v[F_MAX_CLOSURE_MS_60] == 150.0
        assert v[F_MAX_CLOSURE_MS_10] == 150.0
        assert v[F_LONG_CLOSURE_COUNT_60] == 0.0
        assert v[F_BLINK_RATE_60] == 0.0  # 구간이 아직 끝나지 않았다

    # 스텝 7(프레임 6500이 구간을 발행한다): 이벤트는 하나, 길이는 460 ms다.
    v = session.step(7).values
    assert v[F_BLINK_RATE_60] == approx(60000.0 / 7000.0)  # 1회 / 유효 7초 (결정 전과 같다: 구간의 경계는 그대로다)
    assert v[F_BLINK_DUR_MEAN_60] == 460.0  # 결정 전: 상한 2000
    assert v[F_BLINK_DUR_STD_60] == 0.0
    assert v[F_LONG_CLOSURE_COUNT_60] == 0.0  # 결정 전: 1 (5500 ms ≥ 500 ms)
    assert v[F_MAX_CLOSURE_MS_60] == 460.0  # 결정 전: 5500
    assert v[F_MAX_CLOSURE_MS_10] == 460.0  # 결정 전: 5500
    # 시간 누적 특징은 감김 구간과 무관하다: 0.8 이상이던 시간은 프레임 4개 × 100 ms.
    assert v[F_PERCLOS_60] == approx(400.0 / 7000.0)

    # 스텝 16까지는 10초 구간에 그 구간이 들어 있고, 스텝 17부터는 60초 구간에만 남는다.
    assert session.step(16).values[F_MAX_CLOSURE_MS_10] == 460.0
    assert session.step(17).values[F_MAX_CLOSURE_MS_10] == 0.0
    assert session.step(17).values[F_MAX_CLOSURE_MS_60] == 460.0
    assert session.step(17).values[F_LONG_CLOSURE_COUNT_60] == 0.0


def test_lingering_lids_long_closure_uses_the_summed_length(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all(_lingering_lids(3))
    # 깜빡임이 프레임 3개씩이면 토막은 950 ~ 1280, 6220 ~ 6550으로 330 ms씩이고 합은 660 ms다.
    assert [(r.start_ms, r.end_ms, r.duration_ms) for r in session.runs] == [(950, 6550, 660)]
    # 머무는 동안에는 250 ms(950 ~ 1200)라 긴 감김이 아니다.
    assert session.step(6).values[F_LONG_CLOSURE_COUNT_60] == 0.0
    assert session.step(6).values[F_MAX_CLOSURE_MS_60] == 250.0
    # 스텝 7의 마지막 프레임 전에 구간이 끝났다(프레임 6600). 합친 길이 660 ms가 500 ms 이상이므로 긴 감김 1회다.
    v = session.step(7).values
    assert v[F_LONG_CLOSURE_COUNT_60] == 1.0
    assert v[F_MAX_CLOSURE_MS_60] == 660.0
    assert v[F_BLINK_DUR_MEAN_60] == 660.0


@pytest.mark.parametrize(("duration_ms", "expected_long"), [(499, 0.0), (500, 1.0)])
def test_run_features_read_duration_ms_not_end_minus_start(
    pipeline_spec: PipelineSpec, duration_ms: int, expected_long: float
) -> None:
    # 구간의 처음부터 끝까지는 4900 ms지만 길이(duration_ms)는 그보다 짧다. 특징은 duration_ms만 읽는다.
    aggregator = StepAggregator(pipeline_spec, 0, 0)
    aggregator.update(st(0), None, 0)
    aggregator.update(st(100), ClosureRun(100, 5000, duration_ms, 0.9, 6.0, 4.0, True, False), 0)
    v = aggregator.update(st(1000), None, 0)[0].values
    assert v[F_BLINK_DUR_MEAN_60] == float(duration_ms)
    assert v[F_MAX_CLOSURE_MS_60] == float(duration_ms)
    assert v[F_MAX_CLOSURE_MS_10] == float(duration_ms)
    assert v[F_LONG_CLOSURE_COUNT_60] == expected_long


# ---------------------------------------------------------------------------
# 얼굴 소실, 무효 closure
# ---------------------------------------------------------------------------


def test_face_loss(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    states = [st(t, face=not (5000 <= t < 8000)) for t in range(0, 20001, 100)]
    session.feed_all(states)
    v10 = session.step(10).values
    # 최근 10초 중 3초 동안 얼굴이 없다.
    assert v10[F_FACE_MISSING_RATIO_10] == approx(0.3)
    assert v10[F_FACE_MISSING_RATIO_60] == approx((60000.0 - 7000.0) / 60000.0)
    # 얼굴 없는 시간은 평균의 분모에 들어가지 않는다.
    assert v10[F_CLOSURE_MEAN_10] == approx(0.1)
    # 스텝 18의 10초 구간 [8000, 18000)에는 얼굴 없는 시간이 없다.
    assert session.step(18).values[F_FACE_MISSING_RATIO_10] == approx(0.0)
    assert session.step(17).values[F_FACE_MISSING_RATIO_10] == approx(0.1)


def test_face_with_invalid_closure_counts_as_face_time(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    # 얼굴은 있지만 closure가 무효(|yaw_rel| > 40도)인 프레임: 얼굴 시간에는 들어가고 유효 시간에는 안 들어간다.
    states = [st(t, 0.9, valid=t < 3000, pitch=2.0) for t in range(0, 6001, 100)]
    session.feed_all(states)
    v = session.step(6).values
    assert v[F_FACE_MISSING_RATIO_10] == approx(0.4)
    assert v[F_PERCLOS_60] == 0.0  # 유효 3000 ms < 5000
    assert v[F_PERCLOS_10] == 1.0
    assert v[F_CLOSURE_MEAN_60] == approx(0.9)
    assert v[F_PITCH_STD_60] == 0.0  # 얼굴 시간 6000 ms, pitch는 계속 2도


# ---------------------------------------------------------------------------
# 긴 끊김: 여러 스텝 발행, 10초 넘으면 초기화
# ---------------------------------------------------------------------------


def test_gap_of_5s_emits_all_pending_steps(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all(open_frames(0, 20501))  # 마지막 프레임 20500
    assert session.steps[-1].k == 20
    emitted = session.feed(st(25500))
    assert [s.k for s in emitted] == [21, 22, 23, 24, 25]
    assert [s.t_end_ms for s in emitted] == [21000, 22000, 23000, 24000, 25000]
    # 스텝 21: 20000~20500의 프레임 + 프레임 20500이 유지된 200 ms = 700 ms. 나머지는 데이터 없음.
    ratios = [s.values[F_FACE_MISSING_RATIO_10] for s in emitted]
    assert ratios == [approx(0.03), approx(0.13), approx(0.23), approx(0.33), approx(0.43)]
    # 끊긴 동안에도 창은 흘러간다: 스텝 25의 60초 구간 얼굴 시간은 20700 ms
    assert emitted[-1].values[F_FACE_MISSING_RATIO_60] == approx((60000.0 - 20700.0) / 60000.0)
    assert emitted[-1].values[F_CLOSURE_MEAN_10] == approx(0.1)
    # 그 뒤로는 평소대로 이어진다.
    assert [s.k for s in session.feed(st(26000))] == [26]


def test_ongoing_closure_is_kept_through_catch_up_steps(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    states = [st(t, 0.9 if t >= 1300 else 0.1) for t in range(0, 2101, 100)]
    session.feed_all(states)  # 1250부터 감긴 채로 프레임 2100에서 끊긴다(진행 길이 850)
    emitted = session.feed(st(6100, 0.1))
    assert [s.k for s in emitted] == [3, 4, 5, 6]
    # 끊긴 동안의 스텝은 끊기기 직전의 진행 길이를 본다.
    assert [s.values[F_MAX_CLOSURE_MS_10] for s in emitted] == [850.0, 850.0, 850.0, 850.0]
    # 프레임 6100이 잘린 구간(1250~2100)을 발행하고 스텝 7에 귀속된다.
    assert [(r.duration_ms, r.truncated) for r in session.runs] == [(850, True)]
    session.feed(st(7000, 0.1))
    assert session.step(7).values[F_MAX_CLOSURE_MS_10] == 850.0
    assert session.step(7).values[F_LONG_CLOSURE_COUNT_60] == 1.0


def test_gap_of_exactly_10s_does_not_reset(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all(open_frames(0, 20501))
    emitted = session.feed(st(30500))
    assert [s.k for s in emitted] == list(range(21, 31))


def test_gap_over_10s_resets_accumulators(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec, start_ms=0, origin_ms=0)
    states = [st(t, 0.9 if 200 <= t < 500 else 0.1) for t in range(0, 20501, 100)]
    session.feed_all(states)
    assert session.step(20).values[F_MAX_CLOSURE_MS_60] == 300.0
    n_before = len(session.steps)

    # 12초 끊김: 끊긴 동안의 스텝(21~32)은 발행하지 않는다.
    assert session.feed(st(32500)) == []
    session.feed_all(open_frames(32600, 93001))
    after = session.steps[n_before:]
    assert [s.k for s in after] == list(range(33, 94))
    first = after[0]
    assert first.t_end_ms == 33000
    assert first.full_lookback is False
    # 이전 누적은 지워졌다: 스텝 33의 얼굴 시간은 32500~33000의 500 ms뿐이다.
    assert first.values[F_FACE_MISSING_RATIO_10] == approx(0.95)
    assert first.values[F_FACE_MISSING_RATIO_60] == approx(59500.0 / 60000.0)
    assert first.values[F_MAX_CLOSURE_MS_60] == 0.0
    assert first.values[F_BLINK_DUR_MEAN_60] == 0.0
    # 새 누적 시작 시각은 새 프레임이 속한 스텝의 시작(32000)이다. full_lookback은 92000부터다.
    assert session.step(91).full_lookback is False
    assert session.step(92).full_lookback is True
    assert session.step(93).full_lookback is True


def test_reset_cancels_yawn_and_nod_in_progress(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all([st(t, jaw=0.6, pitch=3.0) for t in range(0, 1501, 100)])  # 하품 진행 중(1.5초)
    session.feed(st(12600, jaw=0.6, pitch=20.0))  # 11.1초 끊김 -> 초기화. 느린 평균도 20도에서 다시 시작한다
    session.feed_all([st(t, jaw=0.6, pitch=20.0) for t in range(12700, 14001, 100)])
    session.feed_all([st(t, jaw=0.0, pitch=20.0) for t in range(14100, 16001, 100)])
    for step in session.steps:
        # 하품은 12600에서 새로 시작해 1500 ms 만에 끝났으므로 세지 않는다. 끄덕임도 없다.
        assert step.values[F_YAWN_COUNT_60] == 0.0
        assert step.values[F_NOD_COUNT_60] == 0.0


# ---------------------------------------------------------------------------
# 하품, 끄덕임 특징
# ---------------------------------------------------------------------------


def test_yawn_and_nod_counts(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    states: list[FrameState] = []
    for t in range(0, 12001, 100):
        jaw = 0.5 if 1000 <= t < 3500 else 0.0  # 2.5초 하품: 프레임 3000에서 센다(스텝 4)
        pitch = 15.0 if 6000 <= t < 6500 else 0.0  # 0.5초 끄덕임: 프레임 6500에서 센다(스텝 7)
        states.append(st(t, jaw=jaw, pitch=pitch))
    session.feed_all(states)
    yawns = [session.step(k).values[F_YAWN_COUNT_60] for k in (3, 4, 5, 12)]
    assert yawns == [0.0, 1.0, 1.0, 1.0]
    nods = [session.step(k).values[F_NOD_COUNT_60] for k in (6, 7, 8, 12)]
    assert nods == [0.0, 1.0, 1.0, 1.0]


def test_yawn_uses_face_as_validity(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    states: list[FrameState] = []
    for t in range(0, 6001, 100):
        if 2000 <= t < 2700:
            states.append(st(t, face=False))  # 하품 중 0.7초(마지막 유효 프레임 1900부터 800 ms) 얼굴 소실 -> 취소
        else:
            states.append(st(t, jaw=0.5 if 1000 <= t < 4500 else 0.0))
    session.feed_all(states)
    # 2700에서 다시 시작해 4400까지 1700 ms -> 2000에 못 미친다.
    assert session.step(6).values[F_YAWN_COUNT_60] == 0.0


# ---------------------------------------------------------------------------
# 네 번째 검증에서 어느 테스트도 잡지 못하던 다섯 동작 (손으로 계산한 값)
# ---------------------------------------------------------------------------
#
# 참조 구현의 한 줄을 틀리게 바꿔도 단위 테스트와 벡터가 모두 통과하던 자리다: 끄덕임 도중의 얼굴 소실, 느린 평균을
# 빼는 방향, 10초 넘는 끊김 뒤의 끄덕임 검출기 초기화, 한 스텝에서 끝난 두 감김 가운데 긴 것, 얼굴 시간이 정확히
# 5000 ms일 때의 pitch_std_60. 아래 다섯 테스트는 그 한 줄이 틀리면 떨어진다.


def test_face_loss_in_the_middle_of_a_nod_cancels_it(pipeline_spec: PipelineSpec) -> None:
    # 0도로 1초(m = 0), 1000~1300에 15도(d = 15로 시작), 1400~2300은 얼굴 없음, 2400부터 0도.
    # 얼굴 없는 프레임은 값을 내지 않는다. 마지막 유효 프레임 1300에서 500 ms를 넘긴 1900의 프레임에서 취소된다.
    # 얼굴 없는 프레임을 값 0의 유효 표본으로 넘기면 1400에서 길이 400 ms(200~3000 ms)로 끝나 하나를 센다.
    nod = NodDetector(pipeline_spec.nod)
    counted: list[bool] = []
    session = Session(pipeline_spec)
    for t in range(0, 4001, 100):
        if 1400 <= t < 2400:
            state = st(t, face=False)
        else:
            state = st(t, pitch=15.0 if 1000 <= t < 1400 else 0.0)
        counted.append(nod.update(t, state.pitch_rel, state.face))
        session.feed(state)
    assert counted == [False] * 41
    assert [step.values[F_NOD_COUNT_60] for step in session.steps] == [0.0, 0.0, 0.0, 0.0]


def test_nod_deviation_is_pitch_minus_the_slow_mean(pipeline_spec: PipelineSpec) -> None:
    # 첫 표본 4도가 m의 초기값이다. 100~600의 여섯 표본이 7도: d = 7 − m은 3에서 줄어든다(시작 임계 10 미만).
    # m = 7 − 3·(100/101)^6 = 4.174. 700에 0도: d = −4.174. 끄덕임이 없다.
    # d = pitch_rel + m으로 구하면 100에서 d = 11로 시작하고 700에서 d = 0 + 4.174 < 5로 끝난다(600 ms): 하나를 센다.
    nod = NodDetector(pipeline_spec.nod)
    session = Session(pipeline_spec)
    counted = []
    for t in range(0, 2001, 100):
        pitch = 4.0 if t == 0 else (7.0 if t <= 600 else 0.0)
        counted.append(nod.update(t, pitch, True))
        if t == 600:
            assert nod.mean == approx(7.0 - 3.0 * (100.0 / 101.0) ** 6)
            assert 7.0 - 3.0 * (100.0 / 101.0) ** 6 == pytest.approx(4.174, abs=1e-3)
        session.feed(st(t, pitch=pitch))
    assert counted == [False] * 21
    assert [step.values[F_NOD_COUNT_60] for step in session.steps] == [0.0, 0.0]


def test_gap_over_10s_restarts_the_slow_mean_of_the_nod_detector(pipeline_spec: PipelineSpec) -> None:
    # 0~1000은 0도(m = 0). 11초 끊긴 뒤 12000~12900이 12도, 13000부터 2도.
    # 끊김 뒤 첫 표본이 m의 새 초기값(12)이다: d는 0, 13000에서 2 − 12 = −10. 끄덕임이 없다.
    # 검출기를 초기화하지 않으면 m = 0이 남아 12000에서 d = 12로 시작하고, 13000에서 d = 2 − m < 5로 끝난다
    # (m = 12·(1 − (50/51)·(100/101)^9) = 1.25, 길이 1000 ms): 스텝 14에서 하나를 센다.
    assert 12.0 * (1.0 - (50.0 / 51.0) * (100.0 / 101.0) ** 9) == pytest.approx(1.25, abs=0.01)
    session = Session(pipeline_spec)
    session.feed_all([st(t) for t in range(0, 1001, 100)])
    session.feed_all([st(t, pitch=12.0) for t in range(12000, 13000, 100)])
    session.feed_all([st(t, pitch=2.0) for t in range(13000, 15001, 100)])
    assert [step.k for step in session.steps] == [1, 13, 14, 15]
    assert [step.values[F_NOD_COUNT_60] for step in session.steps] == [0.0, 0.0, 0.0, 0.0]


def test_step_keeps_the_longest_of_two_closures_that_end_in_it(pipeline_spec: PipelineSpec) -> None:
    # 0.1 <-> 0.9의 100 ms 쌍은 가운데(50 ms)에서 0.5를 지난다. 감김 150~550(400 ms)이 600의 프레임에서, 감김
    # 750~850(100 ms)이 900의 프레임에서 끝난다. 둘 다 스텝 1에 귀속되고 그 스텝의 값은 긴 쪽이다.
    # 나중에 끝난 감김으로 덮어쓰면 100이 된다.
    session = Session(pipeline_spec)
    session.feed_all([st(t, 0.9 if 200 <= t < 600 or t == 800 else 0.1) for t in range(0, 2001, 100)])
    assert [(run.start_ms, run.end_ms, run.duration_ms) for run in session.runs] == [(150, 550, 400), (750, 850, 100)]
    for k in (1, 2):
        assert session.step(k).values[F_MAX_CLOSURE_MS_60] == 400.0
        assert session.step(k).values[F_MAX_CLOSURE_MS_10] == 400.0
    assert session.step(1).values[F_BLINK_DUR_MEAN_60] == 250.0


def test_pitch_std_is_computed_at_exactly_5000_ms_of_face_time(pipeline_spec: PipelineSpec) -> None:
    # 100 ms 간격의 얼굴 프레임: 스텝 k가 끝날 때 얼굴 시간은 정확히 1000k ms다. pitch는 0~2000에 0도, 그 뒤 5도.
    # 스텝 4(4000 ms): 기준 5000 ms에 못 미쳐 0. 스텝 5(정확히 5000 ms, '이상'): 평균 15000/5000 = 3, 제곱 평균
    # 75000/5000 = 15, 표준편차 sqrt(15 − 9) = sqrt(6). '초과'로 구현하면 스텝 5가 0이다.
    assert pipeline_spec.step_features.pitch_std_min_face_ms == 5000
    session = Session(pipeline_spec)
    session.feed_all([st(t, pitch=0.0 if t < 2000 else 5.0) for t in range(0, 6001, 100)])
    assert [session.step(k).values[F_FACE_MISSING_RATIO_60] for k in (4, 5)] == [56000.0 / 60000.0, 55000.0 / 60000.0]
    assert [session.step(k).values[F_PITCH_STD_60] for k in (1, 2, 3, 4)] == [0.0, 0.0, 0.0, 0.0]
    assert session.step(5).values[F_PITCH_STD_60] == approx(math.sqrt(6.0))
    assert session.step(6).values[F_PITCH_STD_60] == approx(math.sqrt(50.0 / 9.0))  # 평균 10/3, 제곱 평균 50/3


# ---------------------------------------------------------------------------
# 분산이 0인 구간, 유한성
# ---------------------------------------------------------------------------


def test_zero_variance_is_exactly_zero_for_exact_inputs(pipeline_spec: PipelineSpec) -> None:
    aggregator = StepAggregator(pipeline_spec, 0, 0)
    same = ClosureRun(0, 200, 200, 0.9, 5.0, 5.0, True, False)
    steps: list[StepFeatures] = []
    for t in range(0, 8001, 100):
        run = same if t in (300, 2300, 4300) else None
        steps.extend(aggregator.update(st(t, pitch=8.0), run, 0))
    v = steps[-1].values
    assert v[F_BLINK_DUR_MEAN_60] == 200.0
    assert v[F_BLINK_DUR_STD_60] == 0.0
    assert v[F_PITCH_STD_60] == 0.0


def test_zero_variance_with_rounding_noise_is_clamped(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    # 7.3은 이진수로 정확히 적을 수 없어 Σp²/T − (Σp/T)²에 반올림 오차가 남는다. 음수가 되면 0으로 자른다.
    session.feed_all([st(t, pitch=7.3) for t in range(0, 60001, 33)])
    for step in session.steps:
        value = step.values[F_PITCH_STD_60]
        assert value >= 0.0
        assert value < 1e-5


def test_std_helper_clamps_negative_variance() -> None:
    assert step_features._std_from_moments(3.0, 8.9999999) == 0.0
    assert step_features._std_from_moments(3.0, 13.0) == approx(2.0)


def test_all_outputs_are_finite_and_in_range(pipeline_spec: PipelineSpec) -> None:
    rng = random.Random(20261113)
    session = Session(pipeline_spec)
    t = 0
    closure = 0.1
    for _ in range(6000):
        # 프레임 간격은 대부분 33~66 ms, 가끔 수 초(초기화 포함)
        roll = rng.random()
        if roll < 0.002:
            t += rng.randint(10001, 15000)
        elif roll < 0.01:
            t += rng.randint(300, 6000)
        else:
            t += rng.randint(20, 70)
        closure = min(1.0, max(0.0, closure + rng.uniform(-0.3, 0.3)))
        face = rng.random() > 0.1
        session.feed(
            st(
                t,
                closure,
                valid=rng.random() > 0.1,
                face=face,
                pitch=rng.uniform(-30.0, 30.0),
                jaw=rng.uniform(0.0, 0.8),
            )
        )
    assert len(session.steps) > 100
    ratio_features = (F_PERCLOS_60, F_PERCLOS_10, F_CLOSURE_MEAN_60, F_CLOSURE_MEAN_10)
    for step in session.steps:
        assert len(step.values) == N_FEATURES
        for value in step.values:
            assert isinstance(value, float)
            assert math.isfinite(value)
            assert value >= 0.0
        for index in ratio_features + (F_FACE_MISSING_RATIO_60, F_FACE_MISSING_RATIO_10):
            assert step.values[index] <= 1.0
        assert step.values[F_MAX_CLOSURE_MS_60] <= 10000.0
        assert step.values[F_MAX_CLOSURE_MS_10] <= step.values[F_MAX_CLOSURE_MS_60]


def test_empty_windows_give_zero_not_nan(pipeline_spec: PipelineSpec) -> None:
    session = Session(pipeline_spec)
    session.feed_all([st(t, face=False) for t in range(0, 12001, 100)])
    for step in session.steps:
        assert step.values[:15] == (0.0,) * 15
    assert session.step(12).values[F_FACE_MISSING_RATIO_10] == 1.0
    assert session.step(12).values[F_FACE_MISSING_RATIO_60] == 1.0


# ---------------------------------------------------------------------------
# 시간 누적을 1 ms씩 세는 기준 구현과 비교
# ---------------------------------------------------------------------------


def _reference_time_features(states: list[FrameState], n_steps: int, spec: PipelineSpec) -> list[dict[str, float]]:
    """세션 시간의 1 ms마다 "그 시각의 상태"를 정해서 스텝 특징의 시간 누적 부분을 다시 계산한다.

    시각 x의 상태는 x 이전의 마지막 프레임 것이다. 단 그 프레임에서 200 ms 이상 지났으면 데이터 없음이다.
    """
    hold = spec.clock.max_hold_ms
    sf = spec.step_features
    total_ms = n_steps * 1000
    face = [False] * total_ms
    valid = [False] * total_ms
    closure = [0.0] * total_ms
    pitch = [0.0] * total_ms
    for i, state in enumerate(states):
        end = state.t_ms + hold
        if i + 1 < len(states) and states[i + 1].t_ms < end:
            end = states[i + 1].t_ms
        for x in range(state.t_ms, min(end, total_ms)):
            face[x] = state.face
            valid[x] = state.closure_valid
            closure[x] = state.closure
            pitch[x] = state.pitch_rel

    out: list[dict[str, float]] = []
    for k in range(1, n_steps + 1):
        row: dict[str, float] = {}
        for name, length in (("60", 60000), ("10", 10000)):
            lo = max(0, 1000 * k - length)
            xs = range(lo, 1000 * k)
            valid_ms = sum(1 for x in xs if valid[x])
            closed_ms = sum(1 for x in xs if valid[x] and closure[x] >= sf.perclos_thr)
            face_ms = sum(1 for x in xs if face[x])
            minimum = sf.perclos_min_valid_ms_60 if name == "60" else sf.perclos_min_valid_ms_10
            row["perclos_" + name] = closed_ms / valid_ms if valid_ms >= minimum else 0.0
            row["closure_mean_" + name] = sum(closure[x] for x in xs if valid[x]) / valid_ms if valid_ms > 0 else 0.0
            row["face_missing_" + name] = (length - face_ms) / length
            if name == "60":
                row["pitch_std"] = 0.0
                if face_ms >= sf.pitch_std_min_face_ms:
                    mean = sum(pitch[x] for x in xs if face[x]) / face_ms
                    var = sum((pitch[x] - mean) ** 2 for x in xs if face[x]) / face_ms
                    row["pitch_std"] = math.sqrt(var)
        out.append(row)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_time_accumulation_matches_millisecond_reference(pipeline_spec: PipelineSpec, seed: int) -> None:
    rng = random.Random(seed)
    n_steps = 75
    states: list[FrameState] = []
    t = 0
    closure = 0.2
    while t < n_steps * 1000:
        closure = min(1.0, max(0.0, closure + rng.uniform(-0.4, 0.4)))
        states.append(
            st(
                t,
                closure,
                valid=rng.random() > 0.15,
                face=rng.random() > 0.1,
                pitch=rng.uniform(-20.0, 20.0),
            )
        )
        roll = rng.random()
        if roll < 0.02:
            t += rng.randint(201, 9000)  # 유지 상한을 넘는 끊김(초기화는 아님)
        elif roll < 0.1:
            t += rng.choice((999, 1000, 1001, 200, 201))  # 경계와 상한 근처
        else:
            t += rng.randint(15, 120)
    states.append(st(n_steps * 1000))  # 마지막 스텝을 발행시키는 프레임

    aggregator = StepAggregator(pipeline_spec, 0, 0)
    steps: list[StepFeatures] = []
    for state in states:
        steps.extend(aggregator.update(state, None, 0))
    assert [s.k for s in steps] == list(range(1, n_steps + 1))

    reference = _reference_time_features(states[:-1], n_steps, pipeline_spec)
    for step, ref in zip(steps, reference):
        v = step.values
        assert v[F_PERCLOS_60] == pytest.approx(ref["perclos_60"], abs=1e-9)
        assert v[F_PERCLOS_10] == pytest.approx(ref["perclos_10"], abs=1e-9)
        assert v[F_CLOSURE_MEAN_60] == pytest.approx(ref["closure_mean_60"], abs=1e-9)
        assert v[F_CLOSURE_MEAN_10] == pytest.approx(ref["closure_mean_10"], abs=1e-9)
        assert v[F_FACE_MISSING_RATIO_60] == pytest.approx(ref["face_missing_60"], abs=1e-12)
        assert v[F_FACE_MISSING_RATIO_10] == pytest.approx(ref["face_missing_10"], abs=1e-12)
        assert v[F_PITCH_STD_60] == pytest.approx(ref["pitch_std"], abs=1e-7)


# ---------------------------------------------------------------------------
# 첫 t_ms가 큰 임의 값인 경우
# ---------------------------------------------------------------------------


def _blink_level(t_ms: int) -> float:
    """4초마다 사다리꼴 깜빡임(감기 80, 유지 170, 뜨기 120 ms). 그 밖에는 0.05.

    프레임이 경사 위에 놓이므로 진입·이탈 쌍의 closure 값과 보간 비율이 프레임마다 제각각이다.
    """
    x = t_ms % 4000
    if x < 80:
        return 0.05 + 0.9 * x / 80.0
    if x < 250:
        return 0.95
    if x < 370:
        return 0.95 - 0.9 * (x - 250) / 120.0
    return 0.05


def test_large_first_timestamp_gives_the_same_steps(pipeline_spec: PipelineSpec) -> None:
    big = 1_723_456_789_012  # 에포크 ms 수준. 이 크기에서 배정밀도 간격은 2^-12 ms(0.00024)다.
    rng = random.Random(7)
    times: list[int] = []
    t = 0
    while t < 80000:
        times.append(t)
        t += rng.choice((33, 34, 66, 67, 100))
    closures = [_blink_level(t) for t in times]

    # 깜빡임이 없는 30.5초 부근에 하나를 더 넣는다. 진입 쌍 (0.2 -> c1)에서 0.5에 닿는 오프셋이 n + 0.49995 ms가
    # 되도록 c1을 정한다(n = 간격의 절반). 반올림하면 n이다. 원점을 먼저 더하고 반올림하면 큰 원점에서는
    # x.49995가 x.5로 읽혀 n + 1이 되므로, 보간이 원점에 의존하면 이 구간의 길이와 그 스텝 특징이 달라진다.
    i = next(index for index, t in enumerate(times) if t >= 30500)
    dt = times[i + 1] - times[i]
    n = dt // 2
    closures[i] = 0.2
    closures[i + 1] = 0.2 + 0.3 * dt / (n + 0.49995)
    closures[i + 2] = 0.9
    closures[i + 3] = 0.9
    closures[i + 4] = 0.05
    assert 0.78 < closures[i + 1] < 0.81 and times[i + 4] < 32000
    expected_start = times[i] + n
    # 뜨는 쪽: (0.9 -> 0.05)에서 0.5에 닿는 오프셋 = (0.4/0.85)·간격. 반올림 경계에서 멀다.
    expected_end = times[i + 3] + math.floor(0.4 / 0.85 * (times[i + 4] - times[i + 3]) + 0.5)

    def run(origin_ms: int) -> Session:
        session = Session(pipeline_spec, start_ms=3000, origin_ms=origin_ms)
        states = [
            st(
                origin_ms + t,
                closure,
                face=not (20000 <= t < 23000),
                pitch=5.0 * math.sin(t / 3000.0),
                jaw=0.5 if 40000 <= t < 43000 else 0.0,
            )
            for t, closure in zip(times, closures)
        ]
        session.feed_all(states)
        return session

    small = run(1)
    large = run(big)

    # 감김 구간: 원점을 빼면 같다. 넣어 둔 깜빡임의 시작과 끝은 손 계산값이다.
    small_runs = [(r.start_ms - 1, r.end_ms - 1, r.duration_ms, r.truncated) for r in small.runs]
    large_runs = [(r.start_ms - big, r.end_ms - big, r.duration_ms, r.truncated) for r in large.runs]
    assert small_runs == large_runs
    assert len(small_runs) >= 19  # 4초마다 하나씩(얼굴이 없던 20초의 깜빡임은 빠진다) + 넣어 둔 하나
    crafted = (expected_start, expected_end, expected_end - expected_start, False)
    assert crafted in small_runs
    assert [(r.peak_closing_velocity, r.peak_opening_velocity, r.velocity_valid) for r in small.runs] == [
        (r.peak_closing_velocity, r.peak_opening_velocity, r.velocity_valid) for r in large.runs
    ]
    # 대부분의 구간은 시작 시각이 프레임 시각이 아니다(두 프레임 사이에서 보간한 값이다).
    frame_times = set(times)
    assert sum(1 for start, _, _, _ in small_runs if start not in frame_times) >= 15

    assert len(small.steps) == len(large.steps) > 70
    for a, b in zip(small.steps, large.steps):
        assert a.k == b.k
        assert a.full_lookback == b.full_lookback
        # 누적기는 세션 시간만 쓰고, 검출기의 보간은 쌍 안의 오프셋만 반올림한다. 그래서 원점이 달라도 비트 단위로 같다.
        assert a.values == b.values
        assert a.t_end_ms - 1 == b.t_end_ms - big == 1000 * a.k
    # 넣어 둔 깜빡임이 60초 창에 들어 있는 스텝에서 지속 시간 특징이 실제로 0이 아니다.
    assert small.step(40).values[F_BLINK_DUR_MEAN_60] > 0.0


# ---------------------------------------------------------------------------
# 같은 신호를 다른 프레임률로 넣었을 때의 근접성
# ---------------------------------------------------------------------------


def _signal(t_ms: float) -> float:
    """6초마다 사다리꼴 깜빡임(감기 80, 유지 160, 뜨기 160 ms), 20~21.5초에는 긴 감김. 그 밖에는 0.05."""
    if 20000.0 <= t_ms < 21500.0:
        return 0.95
    x = t_ms % 6000.0 - 1000.0
    low, high = 0.05, 0.95
    if 0.0 <= x < 80.0:
        return low + (high - low) * x / 80.0
    if 80.0 <= x < 240.0:
        return high
    if 240.0 <= x < 400.0:
        return high - (high - low) * (x - 240.0) / 160.0
    return low


def _run_at_fps(spec: PipelineSpec, fps: float, duration_ms: int) -> Session:
    session = Session(spec)
    period = 1000.0 / fps
    last = -1
    i = 0
    while True:
        t = int(math.floor(i * period + 0.5))
        i += 1
        if t > duration_ms:
            break
        if t <= last:
            continue
        last = t
        session.feed(st(t, _signal(float(t)), pitch=3.0 * math.sin(t / 2000.0)))
    return session


def test_features_are_close_across_frame_rates(pipeline_spec: PipelineSpec) -> None:
    fast = _run_at_fps(pipeline_spec, 30.0, 70000)
    slow = _run_at_fps(pipeline_spec, 15.0, 70000)
    assert [s.k for s in fast.steps] == [s.k for s in slow.steps] == list(range(1, 71))
    for a, b in zip(fast.steps, slow.steps):
        va, vb = a.values, b.values
        # 시간 가중 누적값: 표본 간격(33 대 67 ms)만큼의 양자화 차이만 난다.
        assert va[F_PERCLOS_60] == pytest.approx(vb[F_PERCLOS_60], abs=0.01)
        assert va[F_PERCLOS_10] == pytest.approx(vb[F_PERCLOS_10], abs=0.02)
        assert va[F_CLOSURE_MEAN_60] == pytest.approx(vb[F_CLOSURE_MEAN_60], abs=0.01)
        assert va[F_CLOSURE_MEAN_10] == pytest.approx(vb[F_CLOSURE_MEAN_10], abs=0.02)
        assert va[F_PITCH_STD_60] == pytest.approx(vb[F_PITCH_STD_60], abs=0.05)
        assert va[F_FACE_MISSING_RATIO_60] == pytest.approx(vb[F_FACE_MISSING_RATIO_60], abs=0.002)
        assert va[F_FACE_MISSING_RATIO_10] == pytest.approx(vb[F_FACE_MISSING_RATIO_10], abs=0.01)
        # 개수는 같아야 한다.
        assert va[F_LONG_CLOSURE_COUNT_60] == vb[F_LONG_CLOSURE_COUNT_60]
        # 길이는 보간 덕분에 프레임 간격보다 훨씬 작게 어긋난다.
        assert va[F_MAX_CLOSURE_MS_60] == pytest.approx(vb[F_MAX_CLOSURE_MS_60], abs=67.0)
    # 60초가 찬 뒤: 이벤트 수가 같으므로 빈도가 같고, 평균 지속 시간도 가깝다.
    for k in (60, 65, 70):
        va, vb = fast.step(k).values, slow.step(k).values
        assert va[F_BLINK_RATE_60] == pytest.approx(vb[F_BLINK_RATE_60], rel=0.01)
        assert va[F_BLINK_DUR_MEAN_60] == pytest.approx(vb[F_BLINK_DUR_MEAN_60], abs=40.0)
    # 속도는 프레임률에 민감하다(PLAN 2.6절): 15 fps에서 더 낮게 읽힌다. 근접성을 요구하지 않고 방향만 확인한다.
    assert slow.step(70).values[F_CLOSING_VEL_MEAN_60] < fast.step(70).values[F_CLOSING_VEL_MEAN_60]


# ---------------------------------------------------------------------------
# 설정을 읽는지
# ---------------------------------------------------------------------------


def test_constants_come_from_spec(pipeline_spec: PipelineSpec) -> None:
    custom_sf = dataclasses.replace(
        pipeline_spec.step_features, perclos_thr=0.5, long_closure_min_ms=200, max_closure_cap_ms=220
    )
    custom = dataclasses.replace(pipeline_spec, step_features=custom_sf)
    session = Session(custom)
    session.feed_all([st(t, 0.6 if 200 <= t < 500 else 0.1) for t in range(0, 6001, 100)])
    v = session.step(6).values
    assert v[F_PERCLOS_60] == approx(300.0 / 6000.0)  # 0.6 ≥ 0.5
    assert v[F_LONG_CLOSURE_COUNT_60] == 1.0  # 길이 240(180~420) ≥ 200
    assert v[F_MAX_CLOSURE_MS_60] == 220.0  # 상한 220
