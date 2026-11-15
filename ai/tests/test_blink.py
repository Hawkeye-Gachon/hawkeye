"""`drowsy.blink.ClosureRunDetector` 단위 테스트 (PLAN 4.5절, 10.1절 "감김 검출기").

합성 closure 수열만 쓴다. 기댓값은 주석의 손 계산이다. 특징용 설정은 진입 0.5 / 이탈 0.3,
규칙용 설정은 진입 0.7 / 이탈 0.5이고 나머지(min 50 ms, max_dt 200 ms, hold 500 ms, lookback 2)는 같다.
"""

from __future__ import annotations

import dataclasses
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from drowsy.blink import STATE_CLOSED, STATE_HOLD, STATE_OPEN, ClosureRunDetector
from drowsy.spec import PipelineSpec, RulesConfig, RunDetectorSpec, SpecError, apply_overrides
from drowsy.types import ClosureRun

Sample = tuple[int, float | None]  # (t_ms, closure). closure가 None이면 무효 프레임이다.


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_settings(pipeline_spec: PipelineSpec) -> RunDetectorSpec:
    return pipeline_spec.run_detector


@pytest.fixture
def rule_settings(rules_config: RulesConfig) -> RunDetectorSpec:
    return rules_config.closure.detector


@pytest.fixture
def det(feature_settings: RunDetectorSpec) -> ClosureRunDetector:
    return ClosureRunDetector(feature_settings)


def step(detector: ClosureRunDetector, t_ms: int, closure: float | None) -> ClosureRun | None:
    if closure is None:
        return detector.update(t_ms, 0.0, False)
    return detector.update(t_ms, closure, True)


def feed(detector: ClosureRunDetector, samples: list[Sample]) -> list[tuple[int, ClosureRun]]:
    """표본을 차례로 넣고 (발행한 프레임의 t_ms, 구간) 목록을 돌려준다."""
    emitted: list[tuple[int, ClosureRun]] = []
    for t_ms, closure in samples:
        run = step(detector, t_ms, closure)
        if run is not None:
            emitted.append((t_ms, run))
    return emitted


def trapezoid_blink(
    t_ms: float, t0: float, close_ms: float = 80.0, plateau_ms: float = 100.0, open_ms: float = 120.0
) -> float:
    """t0에서 감기 시작하는 사다리꼴 깜빡임. 0.5를 넘는 시간은 close/2 + plateau + open/2다."""
    x = t_ms - t0
    if x <= 0.0:
        return 0.0
    if x < close_ms:
        return x / close_ms
    if x < close_ms + plateau_ms:
        return 1.0
    if x < close_ms + plateau_ms + open_ms:
        return 1.0 - (x - close_ms - plateau_ms) / open_ms
    return 0.0


def frame_times(fps: float, duration_ms: int) -> list[int]:
    period = 1000.0 / fps
    return [int(i * period + 0.5) for i in range(int(duration_ms / period))]


# 손 계산 기준 수열 (40 ms 간격)
BASIC: list[Sample] = [
    (0, 0.1),
    (40, 0.1),
    (80, 0.3),
    (120, 0.7),
    (160, 0.9),
    (200, 0.6),
    (240, 0.2),
    (280, 0.1),
]


# ---------------------------------------------------------------------------
# 기본 깜빡임과 두 설정
# ---------------------------------------------------------------------------


def test_default_settings(feature_settings: RunDetectorSpec, rule_settings: RunDetectorSpec) -> None:
    assert (feature_settings.enter_thr, feature_settings.exit_thr) == (0.5, 0.3)
    assert (rule_settings.enter_thr, rule_settings.exit_thr) == (0.7, 0.5)
    for settings in (feature_settings, rule_settings):
        assert settings.min_duration_ms == 50
        assert settings.max_dt_ms == 200
        assert settings.hold_ms == 500
        assert settings.velocity_lookback_samples == 2


def test_basic_blink_hand_computed(det: ClosureRunDetector) -> None:
    emitted = feed(det, BASIC)
    assert len(emitted) == 1
    t_emit, run = emitted[0]
    # 이탈 0.3 아래로 내려간 프레임(240)에서 발행한다.
    assert t_emit == 240
    # start: 80(0.3)과 120(0.7) 사이에서 0.5 -> 80 + (0.2/0.4)*40 = 100
    assert run.start_ms == 100
    # end: 200(0.6)과 240(0.2) 사이에서 0.5 -> 200 + (0.1/0.4)*40 = 210
    assert run.end_ms == 210
    assert run.duration_ms == 110
    assert run.amplitude == 0.9
    # 감는 속도: 쌍 (40->80) 5/s, (80->120) 10/s, (120->160) 5/s 중 최대
    assert run.peak_closing_velocity == pytest.approx(10.0)
    # 뜨는 속도: 쌍 (160->200) 7.5/s, (200->240) 10/s 중 최대
    assert run.peak_opening_velocity == pytest.approx(10.0)
    assert run.velocity_valid is True
    assert run.truncated is False
    assert det.state == STATE_OPEN


def test_same_samples_with_rule_thresholds(rule_settings: RunDetectorSpec) -> None:
    det = ClosureRunDetector(rule_settings)
    emitted = feed(det, BASIC)
    assert len(emitted) == 1
    t_emit, run = emitted[0]
    # 200(0.6)은 0.5~0.7 사이라 구간이 유지되고, 0.5 아래로 내려간 240에서 발행한다.
    assert t_emit == 240
    # start: 80(0.3)과 120(0.7) 사이에서 0.7 -> 120
    assert run.start_ms == 120
    # end: 160(0.9)과 200(0.6) 사이에서 0.7 -> 160 + (0.2/0.3)*40 = 186.67 -> 187
    assert run.end_ms == 187
    assert run.duration_ms == 67
    assert run.amplitude == 0.9
    assert run.peak_closing_velocity == pytest.approx(10.0)
    assert run.peak_opening_velocity == pytest.approx(10.0)
    assert run.velocity_valid is True
    assert run.truncated is False


def test_thresholds_are_not_hard_coded(feature_settings: RunDetectorSpec) -> None:
    custom = dataclasses.replace(feature_settings, enter_thr=0.6, exit_thr=0.2)
    det = ClosureRunDetector(custom)
    emitted = feed(det, BASIC)
    # 240(0.2)은 이탈 0.2 미만이 아니므로 280(0.1)에서 발행한다.
    assert [t for t, _ in emitted] == [280]
    run = emitted[0][1]
    # start: 80 + (0.6-0.3)/(0.7-0.3)*40 = 110, end: 200(0.6)과 240(0.2) 사이에서 0.6 -> 200
    assert (run.start_ms, run.end_ms, run.duration_ms) == (110, 200, 90)


def test_two_instances_do_not_share_state(feature_settings: RunDetectorSpec, rule_settings: RunDetectorSpec) -> None:
    feature_det = ClosureRunDetector(feature_settings)
    rule_det = ClosureRunDetector(rule_settings)
    for t_ms, closure in [(0, 0.1), (40, 0.6)]:
        step(feature_det, t_ms, closure)
        step(rule_det, t_ms, closure)
    # 0.6은 특징용(0.5)에서는 감김이고 규칙용(0.7)에서는 아니다.
    assert feature_det.state == STATE_CLOSED
    assert rule_det.state == STATE_OPEN


# ---------------------------------------------------------------------------
# 전이표의 모든 칸
# ---------------------------------------------------------------------------


def test_open_row(det: ClosureRunDetector) -> None:
    # OPEN, 무효 -> 그대로 (closure 값은 읽지 않는다)
    assert det.update(0, 1.0, False) is None
    assert det.state == STATE_OPEN
    # OPEN, c < exit -> 그대로
    assert step(det, 40, 0.1) is None
    assert det.state == STATE_OPEN
    # OPEN, exit <= c < enter -> 그대로 (히스테리시스: 0.3~0.5에서는 시작하지 않는다)
    assert step(det, 80, 0.3) is None
    assert step(det, 120, 0.49) is None
    assert det.state == STATE_OPEN
    assert det.ongoing_ms() == 0
    # OPEN, c >= enter -> CLOSED, start_ms 기록. 120(0.49)과 160(0.5) 사이에서 0.5 -> 160
    assert step(det, 160, 0.5) is None
    assert det.state == STATE_CLOSED
    assert det.ongoing_ms() == 0
    assert step(det, 200, 0.5) is None
    assert det.ongoing_ms() == 40


def test_closed_row_stay_and_peak_update(det: ClosureRunDetector) -> None:
    # CLOSED, c >= enter -> 그대로, 정점 갱신
    emitted = feed(det, [(0, 0.1), (40, 0.6), (80, 0.55), (120, 0.9)])
    assert emitted == []
    assert det.state == STATE_CLOSED
    emitted = feed(det, [(160, 0.1)])
    run = emitted[0][1]
    assert run.amplitude == 0.9
    # start: 0 + (0.4/0.5)*40 = 32, end: 120 + (0.4/0.8)*40 = 140
    assert (run.start_ms, run.end_ms, run.duration_ms) == (32, 140, 108)
    # 정점이 120으로 옮겨졌다. 감는 속도는 진입 쌍 12.5/s와 (80->120) 8.75/s 중 최대,
    # 뜨는 속도는 정점 뒤의 쌍 (120->160) 20/s뿐이다. 정점 앞의 하강 (40->80) 1.25/s는 들어가지 않는다.
    assert run.peak_closing_velocity == pytest.approx(12.5)
    assert run.peak_opening_velocity == pytest.approx(20.0)


def test_closed_row_between_thresholds_records_last_down_once(det: ClosureRunDetector) -> None:
    # CLOSED, exit <= c < enter -> 그대로. 직전 표본이 enter 이상일 때만 last_down_ms를 적는다.
    emitted = feed(det, [(0, 0.1), (40, 0.9), (80, 0.4)])
    assert emitted == []
    assert det.state == STATE_CLOSED
    emitted = feed(det, [(120, 0.45), (160, 0.35)])
    assert emitted == []
    assert det.state == STATE_CLOSED
    # CLOSED, c < exit -> 발행, OPEN
    emitted = feed(det, [(200, 0.1)])
    assert det.state == STATE_OPEN
    t_emit, run = emitted[0]
    assert t_emit == 200
    # start: 0 + (0.4/0.8)*40 = 20. end: 40(0.9)과 80(0.4) 사이에서 0.5 -> 40 + (0.4/0.5)*40 = 72.
    # 120, 160, 200의 표본은 end를 바꾸지 않는다.
    assert (run.start_ms, run.end_ms, run.duration_ms) == (20, 72, 52)
    assert run.truncated is False
    assert run.peak_closing_velocity == pytest.approx(20.0)
    # 뜨는 속도: (40->80) 12.5/s, (120->160) 2.5/s, (160->200) 6.25/s 중 최대
    assert run.peak_opening_velocity == pytest.approx(12.5)


def test_closed_row_invalid_goes_to_hold(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9)])
    assert det.state == STATE_CLOSED
    assert step(det, 80, None) is None
    assert det.state == STATE_HOLD


def test_hold_row_returns_to_closed_when_closed_frame_comes_back(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, None)])
    assert det.state == STATE_HOLD
    # HOLD, c >= enter, 끊긴 시간 80 ms <= hold_ms -> CLOSED로 복귀
    assert step(det, 120, 0.8) is None
    assert det.state == STATE_CLOSED
    emitted = feed(det, [(160, 0.1)])
    run = emitted[0][1]
    # 같은 구간으로 이어진다. start 20, end: 120 + (0.3/0.7)*40 = 137.14 -> 137
    assert (run.start_ms, run.end_ms, run.truncated) == (20, 137, False)


def test_hold_row_returns_to_closed_between_thresholds(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, None)])
    # HOLD, exit <= c < enter, 끊긴 시간 <= hold_ms -> CLOSED로 복귀
    assert step(det, 120, 0.4) is None
    assert det.state == STATE_CLOSED
    emitted = feed(det, [(160, 0.1)])
    run = emitted[0][1]
    # 직전 유효 표본 40(0.9)과 120(0.4)의 간격 80 ms는 max_dt 이하이므로 보간한다: 40 + (0.4/0.5)*80 = 104
    assert (run.start_ms, run.end_ms, run.truncated) == (20, 104, False)


def test_hold_row_open_frame_emits_truncated(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.9), (120, None)])
    assert det.state == STATE_HOLD
    # HOLD, c < exit -> 발행(truncated=true), OPEN
    run = step(det, 160, 0.1)
    assert det.state == STATE_OPEN
    assert run is not None
    assert run.truncated is True
    # 잘린 구간의 end는 enter 이상이었던 마지막 유효 프레임의 시각(80)이다. 보간하지 않는다.
    assert (run.start_ms, run.end_ms, run.duration_ms) == (20, 80, 60)
    assert run.amplitude == 0.9
    assert run.velocity_valid is False
    assert run.peak_closing_velocity == 0.0
    assert run.peak_opening_velocity == 0.0


def test_hold_row_invalid_frames_until_hold_expires(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (100, 0.9)])
    # HOLD, 무효, 끊긴 시간 <= hold_ms -> 그대로. 600 - 100 = 500은 hold_ms를 넘지 않는다.
    assert feed(det, [(140, None), (400, None), (600, None)]) == []
    assert det.state == STATE_HOLD
    assert det.ongoing_ms() == 80
    # HOLD, 무효, 끊긴 시간 501 > hold_ms -> 발행(truncated=true), OPEN
    run = step(det, 601, None)
    assert det.state == STATE_OPEN
    assert run is not None
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 100, 80, True)
    assert det.ongoing_ms() == 0


def test_hold_expired_frame_is_reprocessed_in_open_state(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (100, 0.9), (140, None)])
    # 끊긴 시간 601 ms 뒤에 감긴 프레임이 왔다. 먼저 구간을 닫고, 그 프레임은 OPEN에서 처리해 새 구간을 연다.
    run = step(det, 701, 0.9)
    assert run is not None
    assert (run.start_ms, run.end_ms, run.truncated) == (20, 100, True)
    assert det.state == STATE_CLOSED
    assert det.ongoing_ms() == 0
    # 새 구간의 start는 보간하지 않은 프레임 시각 701이다(간격이 max_dt를 넘는다).
    emitted = feed(det, [(741, 0.9), (781, 0.1)])
    second = emitted[0][1]
    assert (second.start_ms, second.end_ms, second.truncated) == (701, 761, False)
    # 감는 쪽 쌍이 없으므로 속도는 유효하지 않다.
    assert second.velocity_valid is False


def test_hold_expired_open_frame_closes_and_stays_open(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (100, 0.9), (140, None)])
    run = step(det, 701, 0.1)
    assert run is not None
    assert run.truncated is True
    assert det.state == STATE_OPEN


# ---------------------------------------------------------------------------
# 시각 보간, 반올림, 최소 길이
# ---------------------------------------------------------------------------


def test_interpolation_needs_gap_within_max_dt(det: ClosureRunDetector) -> None:
    # 간격이 정확히 max_dt(200 ms)면 보간한다.
    emitted = feed(det, [(0, 0.0), (200, 1.0), (400, 0.0)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms) == (100, 300)
    assert run.velocity_valid is True
    assert run.peak_closing_velocity == pytest.approx(5.0)

    # 간격이 201 ms면 보간하지 않는다. start는 enter 이상을 처음 본 프레임(201),
    # end는 enter 이상이었던 마지막 유효 프레임(300)이다. 그 쌍은 속도에도 쓰지 않는다.
    det.reset()
    emitted = feed(det, [(0, 0.0), (201, 1.0), (300, 1.0), (501, 0.0)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms) == (201, 300, 99)
    assert run.truncated is False
    assert run.velocity_valid is False
    assert run.peak_closing_velocity == 0.0
    assert run.peak_opening_velocity == 0.0


def test_first_sample_already_closed_starts_at_frame_time(det: ClosureRunDetector) -> None:
    emitted = feed(det, [(1000, 0.9), (1040, 0.9), (1080, 0.9), (1120, 0.1)])
    run = emitted[0][1]
    # 직전 표본이 없으므로 start는 프레임 시각이다. end: 1080 + (0.4/0.8)*40 = 1100
    assert (run.start_ms, run.end_ms, run.duration_ms) == (1000, 1100, 100)
    assert run.velocity_valid is False


def test_rounding_is_floor_of_x_plus_half(det: ClosureRunDetector) -> None:
    # start = 0 + 0.5*25 = 12.5 -> 13 (짝수 반올림이면 12), end = 25 + 0.5*125 = 87.5 -> 88
    emitted = feed(det, [(0, 0.0), (25, 1.0), (150, 0.0)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms) == (13, 88, 75)


# 원점: 0, 기기 가동 시간 수준(1e9 ms), 에포크 ms 수준(1.7e12 ms). 1.7e12 근처에서 배정밀도 간격은 2^-12 ms(0.00024)다.
TIME_ORIGINS = (0, 1_000_000_000, 1_731_480_000_123)


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        # start: (0.5 − 0.30)/(0.78 − 0.30)·66. 정확한 산술로는 27.5지만 배정밀도 계산값은 27.499999999999996이라 27.
        # end: 66 + (0.28/0.63)·67 = 66 + 29.78 -> 96. (원점을 먼저 더하면 1e9에서 x.5로 뭉개져 start가 28이 된다.)
        ([(0, 0.30), (66, 0.78), (133, 0.15)], (27, 96, 69)),
        # start: (0.1/0.24)·66 = 27.499999999999993 -> 27. end: 66 + (0.14/0.42)·33 = 66 + 11 = 77. 길이 50 ms.
        # (start가 28이 되면 길이 49 ms라 min_duration_ms에 걸려 구간이 사라진다.)
        ([(0, 0.40), (66, 0.64), (99, 0.22)], (27, 77, 50)),
        # start: (0.3/0.6000009)·67 = 33.49995 -> 33 (동률이 아니다. 0.5에서 0.00005 모자란다).
        # end: 134 + (0.4/0.85)·67 = 134 + 31.53 -> 166. (원점 1.7e12를 먼저 더하면 33.49995가 33.5로 읽혀 34가 된다.)
        ([(0, 0.2), (67, 0.8000009), (134, 0.9), (201, 0.05)], (33, 166, 133)),
        # 끝 쪽도 같다. start: (0.4/0.8)·40 = 20. end: 80 + (0.4/0.8000012)·67 = 80 + 33.49995 -> 113.
        ([(0, 0.1), (40, 0.9), (80, 0.9), (147, 0.0999988)], (20, 113, 93)),
    ],
)
def test_interpolated_times_do_not_depend_on_the_time_origin(
    feature_settings: RunDetectorSpec, samples: list[Sample], expected: tuple[int, int, int]
) -> None:
    for origin_ms in TIME_ORIGINS:
        emitted = feed(ClosureRunDetector(feature_settings), [(origin_ms + t, c) for t, c in samples])
        relative = [(run.start_ms - origin_ms, run.end_ms - origin_ms, run.duration_ms) for _, run in emitted]
        assert relative == [expected], f"origin_ms={origin_ms}"


def test_interpolation_offsets_used_above_are_what_the_comments_say() -> None:
    # 위 시험이 가르는 값인지 확인한다. 이식 코드도 같은 순서(나눗셈, 그다음 곱셈)의 IEEE-754 배정밀도 연산이면 같은 값이다.
    assert (0.5 - 0.30) / (0.78 - 0.30) * 66 == 27.499999999999996
    assert (0.5 - 0.40) / (0.64 - 0.40) * 66 == 27.499999999999993
    assert (0.5 - 0.2) / (0.8000009 - 0.2) * 67 == pytest.approx(33.49995, abs=1e-6)
    # 원점을 먼저 더하면 달라지는 이유: 큰 수에 더하는 순간 x.5로 반올림된다.
    assert 1_000_000_000 + (0.5 - 0.30) / (0.78 - 0.30) * 66 == 1_000_000_027.5
    assert 1_731_480_000_123 + (0.5 - 0.2) / (0.8000009 - 0.2) * 67 == 1_731_480_000_156.5
    assert (0.9 - 0.5) / (0.9 - 0.0999988) * 67 == pytest.approx(33.49995, abs=1e-6)
    assert 1_731_480_000_203 + (0.9 - 0.5) / (0.9 - 0.0999988) * 67 == 1_731_480_000_236.5


def test_run_shorter_than_min_duration_is_dropped(det: ClosureRunDetector) -> None:
    # start = 33*(0.4/0.5) = 26.4 -> 26, end = 33 + 34*(0.1/0.5) = 39.8 -> 40, 길이 14 ms < 50 ms
    emitted = feed(det, [(0, 0.1), (33, 0.6), (67, 0.1)])
    assert emitted == []
    assert det.state == STATE_OPEN
    # 버린 뒤에도 다음 구간은 정상으로 검출한다.
    emitted = feed(det, [(100, 0.1), (133, 0.9), (167, 0.9), (200, 0.9), (233, 0.1)])
    assert len(emitted) == 1


def test_min_duration_boundary(det: ClosureRunDetector) -> None:
    # start = 20, end = 40 + 0.5*60 = 70 -> 길이 50 ms는 남긴다(50 미만만 버린다).
    emitted = feed(det, [(0, 0.0), (40, 1.0), (100, 0.0)])
    assert [run.duration_ms for _, run in emitted] == [50]
    # start = 20, end = 40 + 0.5*58 = 69 -> 길이 49 ms는 버린다.
    det.reset()
    assert feed(det, [(0, 0.0), (40, 1.0), (98, 0.0)]) == []


def test_min_duration_applies_to_truncated_runs(det: ClosureRunDetector) -> None:
    # start 20, 잘린 end 40 -> 길이 20 ms
    emitted = feed(det, [(0, 0.1), (40, 0.9), (80, None), (120, 0.1)])
    assert emitted == []
    assert det.state == STATE_OPEN


# ---------------------------------------------------------------------------
# 속도
# ---------------------------------------------------------------------------


def test_closing_velocity_looks_back_two_samples_before_entry(det: ClosureRunDetector) -> None:
    # 진입 표본은 80. 쌍 (0->40) 11.25/s가 진입 쌍 (40->80) 2.5/s보다 크다.
    emitted = feed(det, [(0, 0.0), (40, 0.45), (80, 0.55), (120, 0.55), (160, 0.0)])
    run = emitted[0][1]
    # start: 40 + (0.05/0.10)*40 = 60, end: 120 + (0.05/0.55)*40 = 123.6 -> 124
    assert (run.start_ms, run.end_ms) == (60, 124)
    assert run.peak_closing_velocity == pytest.approx(11.25)
    assert run.peak_opening_velocity == pytest.approx(13.75)
    assert run.velocity_valid is True


def test_closing_velocity_ignores_pairs_older_than_lookback(det: ClosureRunDetector) -> None:
    # 진입 표본은 160. 볼 수 있는 쌍은 (80->120) 0.5/s와 (120->160) 2.75/s뿐이고 (0->40) 10/s는 너무 오래됐다.
    emitted = feed(det, [(0, 0.0), (40, 0.4), (80, 0.42), (120, 0.44), (160, 0.55), (200, 0.55), (240, 0.0)])
    run = emitted[0][1]
    assert run.peak_closing_velocity == pytest.approx(2.75)


def test_lookback_of_three_samples_keeps_three_pairs_in_order(feature_settings: RunDetectorSpec) -> None:
    # velocity_lookback_samples = 3(기본은 2): 진입 표본 세 개 전부터의 쌍 셋을 본다.
    # 쌍의 증가율: (0->40) 0.25/s, (40->80) 10.0/s, (80->120) 0.25/s, 진입 쌍 (120->160) 2.5/s.
    samples: list[Sample] = [(0, 0.0), (40, 0.01), (80, 0.41), (120, 0.42), (160, 0.52), (200, 0.52), (240, 0.0)]
    three = ClosureRunDetector(dataclasses.replace(feature_settings, velocity_lookback_samples=3))
    run = feed(three, samples)[0][1]
    # 세 쌍은 10.0, 0.25, 2.5: 가장 오래된 쌍이 가장 크다.
    assert run.peak_closing_velocity == pytest.approx(10.0)
    # 기본 설정(2)에서는 (80->120)과 진입 쌍만 본다.
    two = ClosureRunDetector(feature_settings)
    assert feed(two, samples)[0][1].peak_closing_velocity == pytest.approx(2.5)
    # 네 표본 전의 쌍은 3으로도 보지 않는다: 10.0/s를 한 칸 더 앞으로 옮기면 2.5/s다.
    older: list[Sample] = [(0, 0.0), (40, 0.4), (80, 0.41), (120, 0.42), (160, 0.43), (200, 0.53), (240, 0.53)]
    older.append((280, 0.0))
    three = ClosureRunDetector(dataclasses.replace(feature_settings, velocity_lookback_samples=3))
    assert feed(three, older)[0][1].peak_closing_velocity == pytest.approx(2.5)


def test_rise_that_stays_below_the_peak_is_not_a_closing_velocity(det: ClosureRunDetector) -> None:
    # 정점(0.9) 뒤에 0.55 -> 0.86으로 급히 오르지만(33 ms, 9.39/s) 정점을 넘지 못한 채 눈을 뜬다.
    first: list[Sample] = [(0, 0.1), (100, 0.1), (200, 0.9), (300, 0.55), (333, 0.86), (433, 0.1)]
    # 다음 구간: 0.1 -> 0.55(200 ms, 2.25/s) -> 0.82(100 ms, 2.7/s: 정점이 옮겨진다) -> 0.1
    second: list[Sample] = [(1000, 0.1), (1200, 0.1), (1400, 0.55), (1500, 0.82), (1600, 0.1)]
    emitted = feed(det, first + second)
    assert [t for t, _ in emitted] == [433, 1600]
    before, after = emitted[0][1], emitted[1][1]
    # 앞 구간의 감는 속도는 진입 쌍의 8.0/s다. 뜨는 속도는 정점 뒤의 가장 큰 감소율 7.6/s(0.86 -> 0.1).
    assert before.peak_closing_velocity == pytest.approx(8.0)
    assert before.peak_opening_velocity == pytest.approx(7.6)
    # 앞 구간에 남은 9.39/s가 뒤 구간으로 넘어오지 않는다: 정점까지의 쌍은 2.25와 2.7뿐이다.
    assert after.peak_closing_velocity == pytest.approx(2.7)
    assert after.peak_opening_velocity == pytest.approx(7.2)


def test_peak_stays_at_the_first_sample_that_reaches_the_maximum(det: ClosureRunDetector) -> None:
    # 0.9, 0.6, 0.9, 0.1: 같은 최댓값이 다시 나온다. 정점은 처음 닿은 표본(200)이다.
    emitted = feed(det, [(0, 0.1), (100, 0.1), (200, 0.9), (233, 0.6), (333, 0.9), (533, 0.1)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms) == (150, 433, 283)
    assert run.peak_closing_velocity == pytest.approx(8.0)
    # 정점 뒤의 쌍: 0.9 -> 0.6(33 ms) 9.09/s, 0.9 -> 0.1(200 ms) 4.0/s. 정점이 333으로 옮겨졌다면 4.0/s가 된다.
    assert run.peak_opening_velocity == pytest.approx(0.3 / 0.033)


def test_truncated_run_with_one_closed_sample_ends_at_that_sample(det: ClosureRunDetector) -> None:
    # 0.42 -> 0.9(100 ms)의 17 ms에 0.5를 지난다. 감긴 표본은 1100 하나이고 다음 프레임은 600 ms 뒤다.
    emitted = feed(det, [(900, 0.05), (1000, 0.42), (1100, 0.9), (1700, 0.05)])
    assert [t for t, _ in emitted] == [1700]
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (1017, 1100, 83, True)
    assert (run.velocity_valid, run.peak_closing_velocity, run.peak_opening_velocity) == (False, 0.0, 0.0)
    # 진행 중인 길이도 진입 표본의 시각까지다: 같은 모양을 5초 뒤에 다시 넣고 진입 표본에서 읽는다.
    for t_ms, closure in ((5900, 0.05), (6000, 0.42), (6100, 0.9)):
        assert step(det, t_ms, closure) is None
    assert (det.state, det.run_start_ms(), det.ongoing_ms()) == (STATE_CLOSED, 6017, 83)


def test_later_rise_to_the_peak_counts_as_closing(det: ClosureRunDetector) -> None:
    # 진입 쌍 (0->40) 3.75/s, 잠깐 내려갔다가 (80->120) 12.5/s로 정점 1.0에 도달한다.
    emitted = feed(det, [(0, 0.4), (40, 0.55), (80, 0.5), (120, 1.0), (160, 0.0)])
    run = emitted[0][1]
    assert run.amplitude == 1.0
    assert run.peak_closing_velocity == pytest.approx(12.5)
    assert run.peak_opening_velocity == pytest.approx(25.0)


def test_single_sample_blink_at_15_fps(det: ClosureRunDetector) -> None:
    # 한 표본짜리 깜빡임. start = 0.5*67 = 33.5 -> 34, end = 67 + 0.5*66 = 100
    emitted = feed(det, [(0, 0.0), (67, 1.0), (133, 0.0)])
    assert len(emitted) == 1
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms) == (34, 100, 66)
    assert run.amplitude == 1.0
    assert run.peak_closing_velocity == pytest.approx(1000.0 / 67.0)
    assert run.peak_opening_velocity == pytest.approx(1000.0 / 66.0)
    assert run.velocity_valid is True


# ---------------------------------------------------------------------------
# 감는 속도의 거슬러 보기는 쓸 수 없는 쌍에서 끊긴다
# ---------------------------------------------------------------------------


def test_rise_before_a_long_invalid_stretch_is_not_the_closing_velocity(det: ClosureRunDetector) -> None:
    # (0 -> 33)에서 0.10 -> 0.45로 오른다(10.6/s). 그 뒤 9초 동안 무효 프레임만 오고, 9000에서는 이미 감겨 있다.
    samples: list[Sample] = [(0, 0.10), (33, 0.45)]
    samples += [(t, None) for t in range(66, 9000, 33)]
    samples += [(9000, 0.9), (9040, 0.9), (9080, 0.1)]
    emitted = feed(det, samples)
    assert [t for t, _ in emitted] == [9080]
    run = emitted[0][1]
    # 진입 쌍 (33 -> 9000)은 간격이 max_dt를 넘는다: start는 프레임 시각이고 9초 전의 증가율은 버린다.
    # end: 9040(0.9)과 9080(0.1) 사이에서 0.5 -> 9040 + (0.4/0.8)*40 = 9060
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (9000, 9060, 60, False)
    assert run.amplitude == 0.9
    # 이 구간에서 오른 쌍을 하나도 보지 못했으므로 감는 속도가 없고, 속도는 유효하지 않다.
    assert run.velocity_valid is False
    assert run.peak_closing_velocity == 0.0
    assert run.peak_opening_velocity == 0.0


def test_rise_before_a_frame_gap_is_not_the_closing_velocity(det: ClosureRunDetector) -> None:
    # 같은 상황인데 무효 프레임 없이 프레임 자체가 8967 ms 끊겼다(10초 이하라 호출자가 reset을 부르지 않는다).
    emitted = feed(det, [(0, 0.10), (33, 0.45), (9000, 0.9), (9040, 0.9), (9080, 0.1)])
    assert [t for t, _ in emitted] == [9080]
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (9000, 9060, 60, False)
    assert run.velocity_valid is False
    assert run.peak_closing_velocity == 0.0
    assert run.peak_opening_velocity == 0.0


def test_flat_sample_between_the_rise_and_the_gap_gives_no_velocity_either(det: ClosureRunDetector) -> None:
    # 대조: 오른 쌍 뒤에 유효 표본이 하나 더 있으면(3000) 표본 수만으로도 그 쌍은 거슬러 보는 범위 밖이다.
    emitted = feed(det, [(0, 0.10), (33, 0.45), (3000, 0.4), (6000, 0.9), (6040, 0.9), (6080, 0.1)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms) == (6000, 6060, 60)
    assert run.velocity_valid is False
    assert run.peak_closing_velocity == 0.0


def test_reentry_after_truncation_does_not_inherit_the_previous_rise(det: ClosureRunDetector) -> None:
    # 감는 중(40 -> 80 -> 120에서 5/s, 3.75/s로 오른다)에 프레임이 640 ms 끊기고, 돌아온 760에서도 감겨 있다.
    emitted = feed(det, [(0, 0.1), (40, 0.6), (80, 0.8), (120, 0.95), (760, 0.9), (800, 0.92), (840, 0.1)])
    assert [t for t, _ in emitted] == [760, 840]
    first, second = emitted[0][1], emitted[1][1]
    # 앞 구간: start = (0.4/0.5)*40 = 32, 잘린 end = enter 이상이었던 마지막 프레임 120.
    assert (first.start_ms, first.end_ms, first.duration_ms, first.truncated) == (32, 120, 88, True)
    assert first.velocity_valid is False
    # 뒤 구간: 끊김을 건너는 쌍 (120 -> 760)은 쓸 수 없으므로 start는 760이고, 앞 구간의 증가율(3.75/s)은 넘어오지 않는다.
    # end: 800(0.92)과 840(0.1) 사이에서 0.5 -> 800 + (0.42/0.82)*40 = 820.5 -> 820
    assert (second.start_ms, second.end_ms, second.duration_ms, second.truncated) == (760, 820, 60, False)
    # 감는 속도는 구간 안에서 정점까지 오른 쌍 (760 -> 800) 0.02/0.04 = 0.5/s뿐이다(진입 쌍을 못 봤어도 이 값은 쓴다).
    assert second.peak_closing_velocity == pytest.approx(0.5)
    assert second.peak_opening_velocity == pytest.approx(0.82 * 1000.0 / 40.0)
    assert second.velocity_valid is True

    # 돌아온 뒤 구간 안에서 오른 쌍이 없으면 감는 속도가 없다.
    det.reset()
    emitted = feed(det, [(0, 0.1), (40, 0.6), (80, 0.8), (120, 0.95), (760, 0.9), (800, 0.9), (840, 0.1)])
    second = emitted[1][1]
    assert (second.start_ms, second.end_ms, second.truncated) == (760, 820, False)
    assert second.velocity_valid is False
    assert second.peak_closing_velocity == 0.0
    assert second.peak_opening_velocity == 0.0


def test_invalid_frame_inside_a_usable_pair_does_not_cut_the_lookback(det: ClosureRunDetector) -> None:
    # 80의 무효 프레임을 사이에 둔 쌍 (40 -> 120)은 간격 80 ms라 쓸 수 있는 쌍이다. 거슬러 보기가 끊기지 않으므로
    # 그 앞의 쌍 (0 -> 40) 11.25/s가 감는 속도다(진입 쌍은 0.10/0.08 = 1.25/s).
    emitted = feed(det, [(0, 0.0), (40, 0.45), (80, None), (120, 0.55), (160, 0.55), (200, 0.0)])
    run = emitted[0][1]
    # start: 40 + (0.05/0.10)*80 = 80, end: 160 + (0.05/0.55)*40 = 163.6 -> 164
    assert (run.start_ms, run.end_ms, run.duration_ms) == (80, 164, 84)
    assert run.peak_closing_velocity == pytest.approx(11.25)
    assert run.peak_opening_velocity == pytest.approx(13.75)
    assert run.velocity_valid is True


def test_first_sample_after_reset_starts_a_fresh_lookback(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.10), (33, 0.45)])
    det.reset()
    # 초기화 뒤 34 ms 만에 감긴 표본이 와도(간격은 max_dt 이하) 직전 유효 표본이 없으므로 쌍이 아니다.
    emitted = feed(det, [(67, 0.9), (107, 0.9), (147, 0.1)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms) == (67, 127)
    assert run.velocity_valid is False
    assert run.peak_closing_velocity == 0.0


def test_spec_validation_requires_max_dt_within_hold(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    # hold_ms를 넘는 끊김으로 잘린 구간 뒤에는 그 끊김을 건너는 쌍이 반드시 쓸 수 없는 쌍이어야 한다.
    # 두 설정 모두 max_dt_ms ≤ hold_ms를 요구한다(같으면 통과).
    assert apply_overrides(rules_config, {"closure.detector.hold_ms": 200}).closure.detector.hold_ms == 200
    with pytest.raises(SpecError, match="max_dt_ms <= hold_ms"):
        apply_overrides(rules_config, {"closure.detector.hold_ms": 199})
    assert apply_overrides(pipeline_spec, {"run_detector.max_dt_ms": 500}).run_detector.max_dt_ms == 500
    with pytest.raises(SpecError, match="max_dt_ms <= hold_ms"):
        apply_overrides(pipeline_spec, {"run_detector.max_dt_ms": 501})


# ---------------------------------------------------------------------------
# 합성 깜빡임: 30 fps와 15 fps, 프레임 누락
# ---------------------------------------------------------------------------

TRUE_DURATION_MS = 200  # 사다리꼴(80 / 100 / 120 ms)이 0.5를 넘는 시간 = 40 + 100 + 60
TRUE_CLOSING_VELOCITY = 1000.0 / 80.0  # 12.5 /s
TRUE_OPENING_VELOCITY = 1000.0 / 120.0  # 8.33 /s


def _blink_runs_over_phases(settings: RunDetectorSpec, fps: float) -> list[tuple[int, ClosureRun]]:
    """깜빡임 시작 시각을 한 프레임 주기에 걸쳐 1 ms씩 옮겨 가며 (t0, 구간)을 모은다."""
    out: list[tuple[int, ClosureRun]] = []
    times = frame_times(fps, 3000)
    for t0 in range(1000, 1000 + int(1000.0 / fps) + 2):
        det = ClosureRunDetector(settings)
        emitted = feed(det, [(t, trapezoid_blink(t, t0)) for t in times])
        assert len(emitted) == 1, f"fps={fps} t0={t0}"
        out.append((t0, emitted[0][1]))
    return out


def test_synthetic_blink_at_30_fps(feature_settings: RunDetectorSpec) -> None:
    for t0, run in _blink_runs_over_phases(feature_settings, 30.0):
        # 임계값 앞뒤 표본이 모두 같은 경사 위에 있으므로 보간이 정확하다(반올림 1 ms).
        assert abs(run.start_ms - (t0 + 40)) <= 1
        assert abs(run.end_ms - (t0 + 240)) <= 1
        assert abs(run.duration_ms - TRUE_DURATION_MS) <= 1
        assert run.amplitude == 1.0
        assert run.velocity_valid is True
        assert run.truncated is False
        # 감는 경사(80 ms) 위에 표본이 둘 이상 놓이므로 속도가 참값으로 읽힌다.
        assert run.peak_closing_velocity == pytest.approx(TRUE_CLOSING_VELOCITY)
        assert run.peak_opening_velocity == pytest.approx(TRUE_OPENING_VELOCITY)


def test_synthetic_blink_at_15_fps(feature_settings: RunDetectorSpec) -> None:
    runs = _blink_runs_over_phases(feature_settings, 15.0)
    closing: list[float] = []
    for _, run in runs:
        # 표본 간격(67 ms)이 경사의 절반보다 길어 보간 오차가 생긴다. 이 파형에서 관찰한 최대 오차는 3 ms다.
        assert abs(run.duration_ms - TRUE_DURATION_MS) <= 5
        assert run.amplitude == 1.0
        assert run.velocity_valid is True
        assert run.truncated is False
        assert 0.0 < run.peak_closing_velocity <= TRUE_CLOSING_VELOCITY + 1e-9
        assert 0.0 < run.peak_opening_velocity <= TRUE_OPENING_VELOCITY + 1e-9
        closing.append(run.peak_closing_velocity)
    # 15 fps에서는 위상에 따라 감는 속도가 참값보다 낮게 읽힌다(PLAN 4.5절).
    assert min(closing) < 0.75 * TRUE_CLOSING_VELOCITY


def test_dropped_frame_inside_plateau_changes_nothing(feature_settings: RunDetectorSpec) -> None:
    times = frame_times(30.0, 3000)
    t0 = 1010
    full = feed(ClosureRunDetector(feature_settings), [(t, trapezoid_blink(t, t0)) for t in times])
    # 평탄 구간(1090~1190) 안의 프레임 둘을 뺀다. 간격 100 ms는 max_dt 이하다.
    dropped = [t for t in times if t not in (1133, 1167)]
    assert len(dropped) == len(times) - 2
    partial = feed(ClosureRunDetector(feature_settings), [(t, trapezoid_blink(t, t0)) for t in dropped])
    assert partial == full


def test_frames_dropped_across_the_closing_edge(det: ClosureRunDetector) -> None:
    # 67~233 ms의 프레임이 없다. 진입 쌍의 간격 234 ms는 max_dt를 넘으므로 start는 프레임 시각 267이다.
    emitted = feed(det, [(0, 0.0), (33, 0.0), (267, 0.8), (300, 1.0), (333, 1.0), (367, 0.4), (400, 0.0)])
    assert [t for t, _ in emitted] == [400]
    run = emitted[0][1]
    # end: 333(1.0)과 367(0.4) 사이에서 0.5 -> 333 + (0.5/0.6)*34 = 361.3 -> 361
    assert (run.start_ms, run.end_ms, run.duration_ms) == (267, 361, 94)
    # 감는 속도는 남은 쌍 (267->300) 0.2/0.033 = 6.06/s, 뜨는 속도는 (333->367) 0.6/0.034 = 17.65/s
    assert run.peak_closing_velocity == pytest.approx(0.2 * 1000.0 / 33.0)
    assert run.peak_opening_velocity == pytest.approx(0.6 * 1000.0 / 34.0)
    assert run.velocity_valid is True


# ---------------------------------------------------------------------------
# 중복 타임스탬프 (호출 계약)
# ---------------------------------------------------------------------------


def test_non_increasing_timestamp_is_rejected(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9)])
    with pytest.raises(ValueError):
        det.update(40, 0.9, True)
    with pytest.raises(ValueError):
        det.update(39, 0.1, True)
    with pytest.raises(ValueError):
        det.update(40, 0.0, False)
    # 거부된 호출은 상태를 바꾸지 않는다.
    assert det.state == STATE_CLOSED
    emitted = feed(det, [(80, 0.9), (120, 0.1)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms) == (20, 100)


# 중복되거나 되돌아간 타임스탬프를 버리는 것은 호출자(`DriverPipeline`)다. 그 동작은 실제 파이프라인으로
# tests/test_pipeline.py의 test_non_increasing_timestamps_are_dropped_and_counted와
# test_dropped_frames_leave_runs_and_steps_as_hand_computed가 시험한다.


# ---------------------------------------------------------------------------
# 긴 감김, 머무는 눈꺼풀
# ---------------------------------------------------------------------------


def test_long_closure_and_ongoing_ms(feature_settings: RunDetectorSpec, rule_settings: RunDetectorSpec) -> None:
    t0 = 1000
    times = frame_times(30.0, 5000)
    feature_det = ClosureRunDetector(feature_settings)
    rule_det = ClosureRunDetector(rule_settings)
    feature_runs: list[ClosureRun] = []
    rule_runs: list[ClosureRun] = []
    previous_ongoing = 0
    first_frame_at_1000 = -1
    for t in times:
        c = trapezoid_blink(t, t0, plateau_ms=2000.0)
        feature_run = feature_det.update(t, c, True)
        rule_run = rule_det.update(t, c, True)
        if feature_run is not None:
            feature_runs.append(feature_run)
        if rule_run is not None:
            rule_runs.append(rule_run)
        ongoing = rule_det.ongoing_ms()
        if rule_det.state == STATE_CLOSED:
            # 진행 중인 길이는 줄어들지 않는다.
            assert ongoing >= previous_ongoing
            previous_ongoing = ongoing
            if ongoing >= 1000 and first_frame_at_1000 < 0:
                first_frame_at_1000 = t

    # 특징용(0.5): 0.5를 넘는 시간 = 40 + 2000 + 60 = 2100
    assert len(feature_runs) == 1
    assert abs(feature_runs[0].start_ms - 1040) <= 1
    assert abs(feature_runs[0].duration_ms - 2100) <= 1
    assert feature_runs[0].truncated is False
    # 규칙용(0.7): 0.7을 넘는 시간 = 24 + 2000 + 36 = 2060, 시작은 1000 + 56
    assert len(rule_runs) == 1
    assert abs(rule_runs[0].start_ms - 1056) <= 1
    assert abs(rule_runs[0].duration_ms - 2060) <= 1
    # 마이크로슬립 임계(1000 ms)는 start + 1000 이후에 온 첫 프레임에서 넘는다.
    assert first_frame_at_1000 == min(t for t in times if t >= rule_runs[0].start_ms + 1000)
    assert feature_det.ongoing_ms() == 0
    assert rule_det.ongoing_ms() == 0


def test_lids_lingering_between_exit_and_enter(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.9)])
    assert det.ongoing_ms() == 60  # start 20, enter 이상이었던 마지막 프레임 80
    # 1초 동안 0.4에 머문다. 구간은 끝나지 않고 길이도 늘어나지 않는다.
    for t in range(120, 1160, 40):
        assert step(det, t, 0.4) is None
        assert det.state == STATE_CLOSED
        assert det.ongoing_ms() == 60
    run = step(det, 1160, 0.1)
    assert run is not None
    # end는 0.5를 아래로 지난 시각: 80 + (0.4/0.5)*40 = 112. 머문 시간은 길이에 들어가지 않는다.
    assert (run.start_ms, run.end_ms, run.duration_ms) == (20, 112, 92)
    assert run.truncated is False
    assert det.state == STATE_OPEN


def test_reclosing_after_lingering_is_one_run(det: ClosureRunDetector) -> None:
    emitted = feed(det, [(0, 0.1), (40, 0.9), (80, 0.4), (120, 0.9), (160, 0.1)])
    # 합치는 것이 아니라 끝나지 않은 한 구간이다. end는 마지막으로 0.5를 아래로 지난 시각 120 + 0.5*40 = 140
    assert len(emitted) == 1
    run = emitted[0][1]
    # 길이는 0.5 이상이었던 두 토막의 합이다(2026-11-13 결정. 그 전에는 end − start = 120이었다).
    # 첫 토막: 20(= 0 + (0.4/0.8)*40) ~ 72(= 40 + (0.4/0.5)*40), 52 ms
    # 둘째 토막: 88(= 80 + (0.1/0.5)*40) ~ 140, 52 ms. 사이의 72~88은 0.5 아래였으므로 넣지 않는다.
    assert (run.start_ms, run.end_ms, run.duration_ms) == (20, 140, 104)
    assert run.amplitude == 0.9
    # 정점은 처음 0.9에 도달한 40이다. 감는 속도는 진입 쌍 20/s, 뜨는 속도는 (40->80) 12.5/s와 (120->160) 20/s 중 최대.
    assert run.peak_closing_velocity == pytest.approx(20.0)
    assert run.peak_opening_velocity == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 길이 = enter 이상이었던 시간의 합 (2026-11-13 결정)
# ---------------------------------------------------------------------------


def test_ongoing_ms_and_run_start_ms_over_two_spans(det: ClosureRunDetector) -> None:
    # 진행 중인 구간이 없으면 둘 다 0이다.
    assert (det.ongoing_ms(), det.run_start_ms()) == (0, 0)
    step(det, 0, 0.1)
    assert (det.ongoing_ms(), det.run_start_ms()) == (0, 0)
    # 40: 진입. start 20, 이 프레임까지 20 ms
    step(det, 40, 0.9)
    assert (det.ongoing_ms(), det.run_start_ms()) == (20, 20)
    # 80: 0.5 아래(0.4). 길이는 0.5 이상이었던 마지막 프레임(40)까지 그대로다.
    step(det, 80, 0.4)
    assert (det.ongoing_ms(), det.run_start_ms()) == (20, 20)
    # 120: 다시 감김. 앞 토막을 72에서 닫아 52 ms를 더하고, 88에서 연 새 토막이 이 프레임까지 32 ms -> 84.
    # (고치기 전 규칙이면 120 − 20 = 100이다.) 구간의 시작은 그대로 20이다.
    step(det, 120, 0.9)
    assert (det.ongoing_ms(), det.run_start_ms()) == (84, 20)
    # 160: 계속 감김 -> 52 + (160 − 88) = 124
    step(det, 160, 0.9)
    assert (det.ongoing_ms(), det.run_start_ms()) == (124, 20)
    # 200: 구간 끝. end = 160 + (0.4/0.8)*40 = 180, 길이 52 + (180 − 88) = 144
    run = step(det, 200, 0.1)
    assert run is not None
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 180, 144, False)
    assert (det.ongoing_ms(), det.run_start_ms()) == (0, 0)


def test_dwell_between_thresholds_is_not_added_when_the_lid_closes_again(rule_settings: RunDetectorSpec) -> None:
    # 결정문의 확인 항목(규칙용 0.7 / 0.5): 0.9로 300 ms, 0.6으로 5초, 0.9로 300 ms, 그 뒤 뜬 눈. 40 ms 간격.
    det = ClosureRunDetector(rule_settings)
    samples: list[Sample] = [(0, 0.1)]
    samples += [(t, 0.9) for t in range(40, 321, 40)]  # 40 ~ 320
    samples += [(t, 0.6) for t in range(360, 5321, 40)]  # 360 ~ 5320
    samples += [(t, 0.9) for t in range(5360, 5641, 40)]  # 5360 ~ 5640
    ongoing: dict[int, int] = {}
    emitted: list[tuple[int, ClosureRun]] = []
    previous = 0
    for t_ms, closure in samples:
        run = step(det, t_ms, closure)
        assert run is None
        assert det.state == STATE_CLOSED or t_ms == 0
        ongoing[t_ms] = det.ongoing_ms()
        # 한 구간 안에서 길이는 줄지 않는다. 한 프레임에 느는 양은 프레임 간격의 두 배를 넘지 않는다(다시 감긴
        # 프레임에서는 앞 토막의 끝을 보간한 몫과 새 토막의 몫이 함께 더해진다). 머문 5초가 한꺼번에 더해지지 않는다.
        assert 0 <= ongoing[t_ms] - previous <= 80
        previous = ongoing[t_ms]
        if t_ms > 0:
            assert det.run_start_ms() == 30
    run = step(det, 5680, 0.1)
    if run is not None:
        emitted.append((5680, run))

    # start: 0 + (0.6/0.8)*40 = 30. 첫 토막의 끝: 320 + (0.2/0.3)*40 = 346.67 -> 347 (317 ms)
    assert ongoing[320] == 290
    # 0.6에 머무는 5초 동안 길이는 0.7 이상이었던 마지막 프레임(320)까지인 290에서 멈춘다.
    assert {ongoing[t] for t in range(360, 5321, 40)} == {290}
    # 다시 감긴 5360: 둘째 토막의 시작은 5320 + (0.1/0.3)*40 = 13.33 -> 5333. 317 + (5360 − 5333) = 344
    # (고치기 전 규칙이면 5360 − 30 = 5330으로 1초, 3초 임계를 한 프레임에 넘는다.)
    assert ongoing[5360] == 344
    assert ongoing[5640] == 624
    # 둘째 토막의 끝: 5640 + (0.2/0.8)*40 = 5650 (317 ms). 구간은 하나이고 길이는 317 + 317 = 634, 약 600 ms다.
    assert len(emitted) == 1
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (30, 5650, 634, False)
    assert run.duration_ms < run.end_ms - run.start_ms


def test_sample_exactly_at_enter_does_not_split_the_span(det: ClosureRunDetector) -> None:
    # 0.5(= enter)는 감긴 표본이다. 다시 감김이 아니므로 토막은 하나이고 길이는 end − start 그대로다.
    emitted = feed(det, [(0, 0.1), (40, 0.9), (80, 0.5), (120, 0.9), (160, 0.1)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms) == (20, 140, 120)

    # 0.45로 내려가면 토막이 둘이다. 앞 토막의 끝: 40 + (0.4/0.45)*40 = 75.56 -> 76,
    # 뒤 토막의 시작: 80 + (0.05/0.45)*40 = 84.44 -> 84. 길이 (76 − 20) + (140 − 84) = 112
    det.reset()
    emitted = feed(det, [(0, 0.1), (40, 0.9), (80, 0.45), (120, 0.9), (160, 0.1)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms) == (20, 140, 112)


def test_span_start_is_interpolated_only_within_max_dt(det: ClosureRunDetector) -> None:
    # 다시 감기는 쌍 (80 -> 280)의 간격이 정확히 max_dt(200 ms)면 보간한다: 80 + (0.1/0.5)*200 = 120.
    emitted = feed(det, [(0, 0.1), (40, 0.9), (80, 0.4), (280, 0.9), (320, 0.1)])
    run = emitted[0][1]
    # 앞 토막 20 ~ 72(52 ms), 뒤 토막 120 ~ 300(= 280 + 0.5*40). 길이 52 + 180 = 232
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 300, 232, False)

    # 간격이 201 ms면 보간하지 않는다. 뒤 토막은 enter 이상을 다시 본 프레임(281)에서 시작한다.
    det.reset()
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.4)])
    assert step(det, 281, 0.9) is None
    assert det.state == STATE_CLOSED
    assert det.ongoing_ms() == 52  # 52 + (281 − 281)
    run = step(det, 321, 0.1)
    assert run is not None
    # 뒤 토막 281 ~ 301(= 281 + 0.5*40). 길이 52 + 20 = 72
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 301, 72, False)


def test_reclosing_on_the_frame_that_returns_from_hold(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.4), (120, None)])
    assert det.state == STATE_HOLD
    assert det.ongoing_ms() == 20
    # HOLD에서 복귀하는 프레임이 다시 감김이다. 직전 유효 표본 80(0.4)과의 간격 80 ms는 max_dt 이하이므로
    # 뒤 토막의 시작을 보간한다: 80 + (0.1/0.5)*80 = 96. 길이 52 + (160 − 96) = 116
    assert step(det, 160, 0.9) is None
    assert det.state == STATE_CLOSED
    assert (det.ongoing_ms(), det.run_start_ms()) == (116, 20)
    run = step(det, 200, 0.1)
    assert run is not None
    # end: 160 + (0.4/0.8)*40 = 180. 길이 52 + (180 − 96) = 136 (고치기 전에는 180 − 20 = 160)
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 180, 136, False)


def test_truncated_run_with_two_spans(det: ClosureRunDetector) -> None:
    # 둘째 토막 도중에 유효 프레임이 끊긴다. 잘린 구간의 마지막 토막은 enter 이상이었던 마지막 프레임(160)에서 끝난다.
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.4), (120, 0.9), (160, 0.9), (200, None)])
    run = step(det, 661, None)  # 661 − 160 = 501 > hold_ms
    assert run is not None
    # 52 + (160 − 88) = 124 (고치기 전에는 160 − 20 = 140)
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 160, 124, True)

    # 둘째 토막 뒤에 다시 0.5 아래로 내려간 표본(160)을 본 뒤 끊긴 경우에도 끝은 enter 이상이었던 마지막 프레임(120)이다.
    # 그 뒤에 0.5를 아래로 지난 시각(120 + (0.4/0.5)*40 = 152)은 쓰지 않는다. 52 + (120 − 88) = 84
    det.reset()
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.4), (120, 0.9), (160, 0.4)])
    run = step(det, 661, 0.1)
    assert run is not None
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 120, 84, True)


def test_min_duration_applies_to_the_summed_length(det: ClosureRunDetector) -> None:
    # 26 ms 간격, 0.3 -> 0.7 -> 0.3 -> 0.7 -> 끝. start = 0 + 0.5*26 = 13, 첫 토막의 끝 = 26 + 0.5*26 = 39 (26 ms),
    # 둘째 토막의 시작 = 52 + 0.5*26 = 65.
    # 마지막 표본이 0.22면 둘째 토막의 끝 = 78 + (0.2/0.48)*26 = 78 + 10.83 -> 89 (24 ms). 합 50 ms: 남긴다.
    emitted = feed(det, [(0, 0.3), (26, 0.7), (52, 0.3), (78, 0.7), (104, 0.22)])
    assert [(run.start_ms, run.end_ms, run.duration_ms) for _, run in emitted] == [(13, 89, 50)]
    # 마지막 표본이 0.2면 끝 = 78 + (0.2/0.5)*26 = 78 + 10.4 -> 88 (23 ms). 합 49 ms: 버린다.
    # (end − start = 75 ms라서 고치기 전 규칙이면 남는다.)
    det.reset()
    assert feed(det, [(0, 0.3), (26, 0.7), (52, 0.3), (78, 0.7), (104, 0.2)]) == []
    assert det.state == STATE_OPEN


# ---------------------------------------------------------------------------
# 감김 중 얼굴 소실, 프레임 끊김
# ---------------------------------------------------------------------------


def test_face_lost_mid_closure_shorter_than_hold(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.9)])
    # 120~360 ms 동안 무효 프레임. 마지막 유효 프레임(80)에서 400까지 320 ms <= hold_ms
    for t in range(120, 400, 40):
        assert step(det, t, None) is None
        assert det.state == STATE_HOLD
        assert det.ongoing_ms() == 60
    assert step(det, 400, 0.9) is None
    assert det.state == STATE_CLOSED
    # 끊긴 구간도 길이에 들어간다.
    assert det.ongoing_ms() == 380
    emitted = feed(det, [(440, 0.9), (480, 0.1)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms) == (20, 460, 440)
    assert run.truncated is False
    # 끊긴 구간을 건너는 쌍(80->400, 320 ms)은 속도에 쓰지 않는다.
    assert run.peak_closing_velocity == pytest.approx(20.0)
    assert run.peak_opening_velocity == pytest.approx(20.0)
    assert run.velocity_valid is True


def test_face_lost_mid_closure_longer_than_hold(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.9)])
    emitted = feed(det, [(t, None) for t in range(120, 680, 40)])
    # 끊긴 시간이 처음 hold_ms를 넘는 프레임은 600(520 ms)이다. 거기서 잘린 구간을 발행한다.
    assert [t for t, _ in emitted] == [600]
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 80, 60, True)
    assert run.velocity_valid is False
    assert det.state == STATE_OPEN
    # 얼굴이 돌아왔을 때 아직 감겨 있으면 새 구간이다.
    emitted = feed(det, [(680, 0.9), (720, 0.9), (760, 0.1)])
    assert [t for t, _ in emitted] == [760]
    second = emitted[0][1]
    assert (second.start_ms, second.end_ms, second.truncated) == (680, 740, False)


def test_face_returns_with_open_eyes_after_long_loss(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.9)])
    emitted = feed(det, [(t, None) for t in range(120, 680, 40)])
    assert len(emitted) == 1
    assert feed(det, [(680, 0.1), (720, 0.1)]) == []
    assert det.state == STATE_OPEN


def test_frame_gap_without_invalid_frames(det: ClosureRunDetector) -> None:
    # 무효 프레임 없이 프레임이 끊긴 경우. 500 ms까지는 같은 구간으로 잇는다.
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.9)])
    assert step(det, 580, 0.9) is None
    assert det.state == STATE_CLOSED
    assert det.ongoing_ms() == 560

    # 501 ms 끊기면 잇지 않는다. 잘린 구간을 닫고 그 프레임에서 새 구간을 연다.
    det.reset()
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.9)])
    run = step(det, 581, 0.9)
    assert run is not None
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 80, 60, True)
    assert det.state == STATE_CLOSED
    assert det.ongoing_ms() == 0


def test_invalid_frame_after_long_gap_closes_the_run_at_once(det: ClosureRunDetector) -> None:
    feed(det, [(0, 0.1), (40, 0.9), (80, 0.9)])
    run = step(det, 700, None)
    assert run is not None
    assert (run.end_ms, run.truncated) == (80, True)
    assert det.state == STATE_OPEN


def test_gap_closed_run_with_open_frame_is_not_truncated(det: ClosureRunDetector) -> None:
    # CLOSED에서 300 ms 뒤(무효 프레임 없음)에 뜬 눈이 왔다. 전이표의 CLOSED 행 그대로 발행한다.
    # 간격이 max_dt를 넘으므로 end는 enter 이상이었던 마지막 프레임 80이다.
    emitted = feed(det, [(0, 0.1), (40, 0.9), (80, 0.9), (380, 0.1)])
    run = emitted[0][1]
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (20, 80, 60, False)
    assert run.velocity_valid is False


# ---------------------------------------------------------------------------
# 무작위 수열에서도 지켜지는 성질
# ---------------------------------------------------------------------------


def test_random_streams_keep_run_invariants(feature_settings: RunDetectorSpec, rule_settings: RunDetectorSpec) -> None:
    rng = random.Random(20261113)
    gaps_ms = (16, 33, 34, 66, 67, 100, 150, 200, 201, 350, 500, 501, 700)
    n_runs = 0
    n_shorter = 0
    for settings in (feature_settings, rule_settings):
        for _ in range(150):
            det = ClosureRunDetector(settings)
            t_ms = rng.randrange(1, 10**9)
            closure = rng.random()
            last_end_ms: int | None = None
            for _ in range(300):
                t_ms += rng.choice(gaps_ms)
                closure = min(1.0, max(0.0, closure + rng.uniform(-0.4, 0.4)))
                valid = rng.random() > 0.15
                run = det.update(t_ms, closure, valid)

                assert det.ongoing_ms() >= 0
                if det.state == STATE_OPEN:
                    assert det.ongoing_ms() == 0
                    assert det.run_start_ms() == 0
                else:
                    # 진행 중인 길이는 구간의 시작부터 이 프레임까지의 시간을 넘지 않는다.
                    assert det.run_start_ms() <= t_ms
                    assert det.ongoing_ms() <= t_ms - det.run_start_ms()
                if run is None:
                    continue
                n_runs += 1
                # 길이는 enter 이상이었던 시간의 합이므로 end − start를 넘지 않는다. 도중에 enter 아래로 내려갔다가
                # 다시 올라온 구간에서는 더 짧다.
                assert run.duration_ms <= run.end_ms - run.start_ms
                n_shorter += run.duration_ms < run.end_ms - run.start_ms
                assert run.duration_ms >= settings.min_duration_ms
                assert run.end_ms <= t_ms
                assert settings.enter_thr <= run.amplitude <= 1.0
                if run.velocity_valid:
                    assert run.truncated is False
                    assert run.peak_closing_velocity > 0.0
                    assert run.peak_opening_velocity > 0.0
                else:
                    assert run.peak_closing_velocity == 0.0
                    assert run.peak_opening_velocity == 0.0
                # 구간은 시간 순서대로 나오고 겹치지 않는다.
                if last_end_ms is not None:
                    assert run.start_ms >= last_end_ms
                last_end_ms = run.end_ms
                # 구간을 닫은 프레임이 감긴 프레임이면(끊긴 뒤 다시 처리) 바로 새 구간이 열려 있다.
                if valid and closure >= settings.enter_thr:
                    assert run.truncated is True
                    assert det.state == STATE_CLOSED
                else:
                    assert det.state == STATE_OPEN
    assert n_runs > 500
    # 이 수열들에는 토막이 여럿인 구간이 실제로 들어 있다.
    assert n_shorter > 100


# ---------------------------------------------------------------------------
# 오프라인 기준 구현과 비교
# ---------------------------------------------------------------------------

RefRun = tuple[int, int, int, float, float, float, bool, bool]  # ClosureRun의 필드 순서


def _reference_runs(
    samples: list[tuple[int, float, bool]],
    cfg: RunDetectorSpec,
    cut_at_unusable_pair: bool = True,
    closed_time_only: bool = True,
) -> tuple[dict[int, RefRun], list[int], list[int]]:
    """모듈 설명의 정의를 다른 방식으로 옮긴 기준 구현.

    (표본 번호 -> 구간, 표본마다의 ongoing_ms, 표본마다의 진행 중인 구간의 start_ms)를 돌려준다.

    검출기는 지나간 쌍의 증가율을 고정 길이 버퍼와 누적 최댓값으로 들고 가지만, 여기서는 유효 표본을 전부
    모아 두었다가 구간이 끝날 때 목록을 거슬러 보며 정점과 두 속도를 처음부터 다시 구한다.
    길이도 검출기처럼 토막을 더해 가지 않는다. 구간에 속한 유효 표본 목록에서 "enter 아래로 내려갔다가 다시
    올라오기까지의 틈"을 찾아서 (구간의 처음부터 끝까지의 시간) − (틈의 합)으로 구한다.
    `cut_at_unusable_pair=False`는 고치기 전 규칙(진입 전 쌍을 표본 수로만 센다), `closed_time_only=False`는
    2026-11-13 결정 전의 규칙(길이 = end − start, 틈을 빼지 않는다)이다. 비교가 실제로 그 차이를 가르는지
    확인하는 데만 쓴다.
    """
    valid: list[tuple[int, float]] = []  # 유효 표본 (t, c)
    runs: dict[int, RefRun] = {}
    ongoing: list[int] = []
    run_starts: list[int] = []
    state = "open"
    start = entry = last_closed = last_down = 0
    members: list[int] = []  # 이 구간에 속한 유효 표본의 번호(valid 안에서의 위치)

    def rate(v: int) -> float | None:
        """유효 표본 v−1과 v의 쌍의 변화율. 쓸 수 없는 쌍이면 None."""
        if v < 1 or valid[v][0] - valid[v - 1][0] > cfg.max_dt_ms:
            return None
        return (valid[v][1] - valid[v - 1][1]) * 1000.0 / (valid[v][0] - valid[v - 1][0])

    def crossing(v: int) -> int:
        """유효 표본 v−1과 v 사이에서 enter를 지난 시각. 보간할 수 없으면 둘 중 enter 이상인 표본의 시각."""
        (t0, c0), (t1, c1) = valid[v - 1], valid[v]
        if t1 - t0 > cfg.max_dt_ms:
            return t1 if c1 >= cfg.enter_thr else t0
        return t0 + int(math.floor((cfg.enter_thr - c0) / (c1 - c0) * (t1 - t0) + 0.5))

    def below_enter_ms() -> int:
        """지금 구간에서 enter 아래로 내려갔다가 다시 올라온 틈들의 합. 아직 다시 올라오지 않은 틈은 넣지 않는다."""
        if not closed_time_only:
            return 0
        total = 0
        down = 0
        for a, b in zip(members, members[1:], strict=False):
            was_closed = valid[a][1] >= cfg.enter_thr
            is_closed = valid[b][1] >= cfg.enter_thr
            if was_closed and not is_closed:
                down = crossing(b)
            elif is_closed and not was_closed:
                total += crossing(b) - down
        return total

    def close(truncated: bool) -> RefRun | None:
        end = last_closed if truncated else last_down
        length = end - start - below_enter_ms()
        if length < cfg.min_duration_ms:
            return None
        levels = [valid[v][1] for v in members]
        amplitude = max(levels)
        peak = members[levels.index(amplitude)]  # 최댓값에 처음 도달한 표본
        before_entry: list[int] = []
        for v in range(entry, entry - cfg.velocity_lookback_samples, -1):
            if cut_at_unusable_pair and rate(v) is None:
                break
            before_entry.append(v)
        closing_pairs = before_entry + list(range(entry + 1, peak + 1))
        opening_pairs = range(peak + 1, members[-1] + 1)
        ups = [r for r in (rate(v) for v in closing_pairs) if r is not None and r > 0.0]
        downs = [-r for r in (rate(v) for v in opening_pairs) if r is not None and r < 0.0]
        velocity_valid = (not truncated) and len(ups) > 0 and len(downs) > 0
        closing = max(ups) if velocity_valid else 0.0
        opening = max(downs) if velocity_valid else 0.0
        return (start, end, length, amplitude, closing, opening, velocity_valid, truncated)

    for i, (t, c, ok) in enumerate(samples):
        run: RefRun | None = None
        if state != "open" and t - valid[-1][0] > cfg.hold_ms:
            run = close(True)
            state = "open"
        if not ok:
            if state == "closed":
                state = "hold"
        else:
            near = len(valid) > 0 and t - valid[-1][0] <= cfg.max_dt_ms
            prev_t, prev_c = valid[-1] if valid else (0, 0.0)
            valid.append((t, c))
            v = len(valid) - 1
            if state == "open":
                if c >= cfg.enter_thr:
                    start = t
                    if near and prev_c < cfg.enter_thr:
                        offset = (cfg.enter_thr - prev_c) / (c - prev_c) * (t - prev_t)
                        start = prev_t + int(math.floor(offset + 0.5))
                    state, entry, last_closed, last_down, members = "closed", v, t, t, [v]
            else:
                resumed_from_hold = state == "hold"
                state = "closed"
                members.append(v)
                if c >= cfg.enter_thr:
                    last_closed = t
                else:
                    if prev_c >= cfg.enter_thr:
                        last_down = prev_t
                        if near:
                            offset = (prev_c - cfg.enter_thr) / (prev_c - c) * (t - prev_t)
                            last_down = prev_t + int(math.floor(offset + 0.5))
                    if c < cfg.exit_thr:
                        run = close(resumed_from_hold)
                        state = "open"
        if run is not None:
            runs[i] = run
        ongoing.append(0 if state == "open" else last_closed - start - below_enter_ms())
        run_starts.append(0 if state == "open" else start)
    return runs, ongoing, run_starts


def _as_tuple(run: ClosureRun) -> RefRun:
    return (
        run.start_ms,
        run.end_ms,
        run.duration_ms,
        run.amplitude,
        run.peak_closing_velocity,
        run.peak_opening_velocity,
        run.velocity_valid,
        run.truncated,
    )


def test_random_streams_match_offline_reference(
    feature_settings: RunDetectorSpec, rule_settings: RunDetectorSpec
) -> None:
    rng = random.Random(20261114)
    usual_ms = (16, 33, 34, 66, 67, 100)
    rare_ms = (150, 200, 201, 350, 500, 501, 700, 3000, 9000)
    n_runs = 0
    n_velocity_valid = 0
    n_old_rule_differs = 0
    n_old_length_differs = 0
    n_old_ongoing_differs = 0
    for settings in (feature_settings, rule_settings):
        for _ in range(150):
            # 큰 임의 원점. 기준 구현은 같은 절대 시각을 받는다(보간이 원점과 무관해야 일치한다).
            t_ms = rng.choice((0, 1, 10**9 + 7, 1_731_480_000_123))
            closure = rng.random()
            samples: list[tuple[int, float, bool]] = []
            for _ in range(400):
                t_ms += rng.choice(rare_ms) if rng.random() < 0.08 else rng.choice(usual_ms)
                roll = rng.random()
                if roll < 0.05:
                    closure = rng.choice((0.0, 1.0, settings.enter_thr, settings.exit_thr))
                elif roll < 0.15:
                    closure = round(rng.random(), 2)  # 두 자리 값: 보간이 x.5 근처에 놓이는 경우를 만든다
                else:
                    closure = min(1.0, max(0.0, closure + rng.uniform(-0.3, 0.3)))
                samples.append((t_ms, closure, rng.random() > 0.12))

            det = ClosureRunDetector(settings)
            reference, reference_ongoing, reference_start = _reference_runs(samples, settings)
            old_rule, _, _ = _reference_runs(samples, settings, cut_at_unusable_pair=False)
            old_length, old_ongoing, _ = _reference_runs(samples, settings, closed_time_only=False)
            for i, (t, c, ok) in enumerate(samples):
                run = det.update(t, c, ok)
                assert det.ongoing_ms() == reference_ongoing[i]
                assert det.run_start_ms() == reference_start[i]
                n_old_ongoing_differs += old_ongoing[i] != reference_ongoing[i]
                if run is None:
                    assert i not in reference
                    # 결정 전 규칙으로는 남았을 구간이 합친 길이로는 min_duration_ms에 못 미쳐 버려진 경우도 센다.
                    n_old_length_differs += i in old_length
                    continue
                expected = reference[i]
                got = _as_tuple(run)
                assert got[:4] == expected[:4]
                assert got[4] == pytest.approx(expected[4], rel=1e-12, abs=1e-12)
                assert got[5] == pytest.approx(expected[5], rel=1e-12, abs=1e-12)
                assert got[6:] == expected[6:]
                n_runs += 1
                n_velocity_valid += run.velocity_valid
                n_old_rule_differs += old_rule[i] != expected
                n_old_length_differs += old_length[i] != expected
    assert n_runs > 1500
    assert n_velocity_valid > 500
    # 이 수열들은 고치기 전 규칙(끊김 앞의 증가율이 넘어온다)과 실제로 갈린다.
    assert n_old_rule_differs > 20
    # 2026-11-13 결정 전의 길이 규칙(end − start)과도 실제로 갈린다(구간과 진행 중인 길이 둘 다).
    assert n_old_length_differs > 200
    assert n_old_ongoing_differs > 2000


# ---------------------------------------------------------------------------
# reset, 이식 제약
# ---------------------------------------------------------------------------


def test_reset_drops_the_ongoing_run(feature_settings: RunDetectorSpec) -> None:
    det = ClosureRunDetector(feature_settings)
    feed(det, [(5000, 0.1), (5040, 0.9), (5080, 0.9)])
    assert det.state == STATE_CLOSED
    det.reset()
    assert det.state == STATE_OPEN
    assert det.ongoing_ms() == 0
    # 초기화 뒤에는 새로 만든 것과 같다. 버린 구간은 발행하지 않고, 시간 순서 검사도 새로 시작한다.
    assert feed(det, BASIC) == feed(ClosureRunDetector(feature_settings), BASIC)


def test_module_imports_standard_library_only(repo_root: Path) -> None:
    code = (
        "import sys\n"
        "import drowsy.blink\n"
        "banned = [name for name in ('numpy', 'pandas', 'yaml', 'torch') if name in sys.modules]\n"
        "assert not banned, banned\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
