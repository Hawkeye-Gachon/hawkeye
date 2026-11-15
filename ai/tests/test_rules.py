"""`drowsy.rules` 단위 테스트 (PLAN 8장, 10.1절 "규칙").

`FrameState`와 `StepFeatures`를 직접 만들어 넣는다. 프레임은 100 ms 간격이고 기댓값은 주석의 손 계산이다.
기본 설정(config/rules.yaml)의 졸음 판정원은 perclos다(D11). 모델 판정은 `apply_overrides`로 바꿔서 시험한다.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable

import pytest

from drowsy import rules as rules_module
from drowsy.blink import ClosureRunDetector
from drowsy.rules import (
    ZONE_IN,
    ZONE_OUT,
    ZONE_UNKNOWN,
    AlertPolicy,
    ForwardZone,
    IntervalWindow,
    RuleEngine,
    StepHysteresis,
)
from drowsy.spec import RulesConfig, apply_overrides
from drowsy.types import (
    EVENT_CALIBRATING,
    EVENT_DROWSY,
    EVENT_GAZE_AWAY_LONG,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
    EVENT_TYPES,
    EVENT_UNABLE_TO_ANALYZE,
    EVENT_VATS,
    F_FACE_MISSING_RATIO_60,
    F_PERCLOS_60,
    N_FEATURES,
    PHASE_END,
    PHASE_ONSET,
    SOURCE_CLOSURE,
    SOURCE_EYE,
    SOURCE_FACE,
    SOURCE_HEAD_POSE,
    SOURCE_MODEL,
    SOURCE_PERCLOS,
    DriverStateEvent,
    FrameState,
    StepFeatures,
)

# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def st(
    t_ms: int,
    closure: float = 0.0,
    *,
    valid: bool = True,
    face: bool = True,
    yaw: float = 0.0,
    pitch: float = 0.0,
    gaze_h: float = 0.0,
    gaze_v: float = 0.0,
    pose_valid: bool = True,
    provisional: bool = False,
) -> FrameState:
    if not face:
        return FrameState(
            t_ms=t_ms,
            face=False,
            closure=0.0,
            closure_valid=False,
            provisional=provisional,
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
        provisional=provisional,
        yaw_rel=yaw,
        pitch_rel=pitch,
        pose_valid=pose_valid,
        gaze_h_rel=gaze_h,
        gaze_v_rel=gaze_v,
        jaw_rel=0.0,
    )


Emitted = list[tuple[int, DriverStateEvent]]


def feed(engine: RuleEngine, states: list[FrameState]) -> Emitted:
    """상태를 차례로 넣고 (발행한 프레임의 t_ms, 이벤트) 목록을 돌려준다."""
    out: Emitted = []
    for state in states:
        for event in engine.update(state):
            out.append((state.t_ms, event))
    return out


def brief(emitted: Emitted) -> list[tuple[int, str, str]]:
    return [(t, e.type, e.phase) for t, e in emitted]


def of_type(emitted: Emitted, *types: str) -> Emitted:
    return [(t, e) for t, e in emitted if e.type in types]


def closure_session(last_closed_ms: int, t_end_ms: int, **kwargs: bool) -> list[FrameState]:
    """눈 뜬 프레임 0~700, 감긴 프레임 1000~last_closed_ms(closure 1.0), 그 뒤 눈 뜬 프레임.

    700과 1000 사이가 300 ms(> max_dt 200)라서 시작 시각을 보간하지 않는다: start_ms = 1000.
    끝은 마지막 감긴 프레임(1.0)과 다음 프레임(0.0) 사이에서 0.7을 지난 시각 = last_closed_ms + 30.
    """
    states = [st(t, 0.0, **kwargs) for t in range(0, 701, 100)]
    states += [st(t, 1.0, **kwargs) for t in range(1000, last_closed_ms + 1, 100)]
    states += [st(t, 0.0, **kwargs) for t in range(last_closed_ms + 100, t_end_ms + 1, 100)]
    return states


def gaze_session(
    out_from: int, out_to: int, t_end_ms: int, *, yaw: float = 30.0, gaze_h: float = 0.0
) -> list[FrameState]:
    """[out_from, out_to)의 프레임만 영역 밖이다."""
    states: list[FrameState] = []
    for t in range(0, t_end_ms + 1, 100):
        if out_from <= t < out_to:
            states.append(st(t, yaw=yaw, gaze_h=gaze_h))
        else:
            states.append(st(t))
    return states


def step(k: int, perclos: float = 0.0, *, full: bool = True, missing: float = 0.0) -> StepFeatures:
    values = [0.0] * N_FEATURES
    values[F_PERCLOS_60] = perclos
    values[F_FACE_MISSING_RATIO_60] = missing
    return StepFeatures(k=k, t_end_ms=1000 * k, full_lookback=full, values=tuple(values))


def probs(p_drowsy: float) -> tuple[float, float, float]:
    rest = (1.0 - p_drowsy) / 2.0
    return (rest, rest, p_drowsy)


def event(
    event_type: str,
    t_start_ms: int = 0,
    phase: str = PHASE_ONSET,
    duration_ms: int = 0,
    severity: int = 2,
    source: str = SOURCE_FACE,
) -> DriverStateEvent:
    return DriverStateEvent(
        type=event_type,
        severity=severity,
        t_start_ms=t_start_ms,
        duration_ms=duration_ms,
        confidence=1.0,
        source=source,
        phase=phase,
        provisional=False,
    )


@pytest.fixture
def engine(rules_config: RulesConfig) -> RuleEngine:
    return RuleEngine(rules_config)


@pytest.fixture
def model_rules(rules_config: RulesConfig) -> RulesConfig:
    return apply_overrides(rules_config, {"drowsiness.source": "model"})


# ---------------------------------------------------------------------------
# 설정과 표 8.2
# ---------------------------------------------------------------------------


def test_default_rules_match_table(rules_config: RulesConfig) -> None:
    assert rules_config.drowsiness.source == "perclos"
    severities = {name: rules_config.events.get(name).severity for name in EVENT_TYPES}
    assert severities == {
        EVENT_CALIBRATING: 0,
        EVENT_UNABLE_TO_ANALYZE: 1,
        EVENT_GAZE_AWAY_LONG: 2,
        EVENT_VATS: 2,
        EVENT_DROWSY: 2,
        EVENT_MICROSLEEP: 3,
        EVENT_SLEEP: 3,
    }
    assert rules_config.alert_priority[:5] == (
        EVENT_SLEEP,
        EVENT_MICROSLEEP,
        EVENT_GAZE_AWAY_LONG,
        EVENT_VATS,
        EVENT_DROWSY,
    )


def test_non_increasing_timestamp_is_rejected(engine: RuleEngine) -> None:
    engine.update(st(100))
    with pytest.raises(ValueError):
        engine.update(st(100))
    with pytest.raises(ValueError):
        engine.update(st(99))


def test_a_rejected_frame_leaves_the_engine_unchanged(rules_config: RulesConfig) -> None:
    def calibrating_engine() -> RuleEngine:
        engine = RuleEngine(rules_config)
        emitted = feed(engine, [st(t, provisional=True) for t in range(0, 1001, 100)])
        assert brief(emitted) == [(0, EVENT_CALIBRATING, PHASE_ONSET)]
        return engine

    reference = calibrating_engine()
    engine = calibrating_engine()
    # 직전 프레임과 같은 t_ms(1000)와 더 이른 t_ms. 규칙 엔진이 아무 상태도 바꾸기 전에 거부해야 한다. 이 프레임을
    # 처리하기 시작했다면 provisional=False라서 calibrating이 여기서 끝난 것으로 적히고, 그 end는 예외와 함께 사라진다
    # (안쪽의 감김 검출기도 같은 검사를 하므로 예외만 봐서는 드러나지 않는다).
    with pytest.raises(ValueError):
        engine.update(st(1000, provisional=False))
    with pytest.raises(ValueError):
        engine.update(st(999, provisional=False))
    later = [st(1100, provisional=True), st(1200, provisional=False)]
    expected = feed(reference, later)
    assert brief(expected) == [(1200, EVENT_CALIBRATING, PHASE_END)]
    # [0, 1200] 동안 얼굴이 계속 있었다: 1200 / 1200
    assert expected[0][1].confidence == 1.0
    assert feed(engine, later) == expected


# ---------------------------------------------------------------------------
# IntervalWindow
# ---------------------------------------------------------------------------


def test_interval_window_merges_and_trims_exactly() -> None:
    window = IntervalWindow(1000)
    window.add(0, 100)
    window.add(100, 300)  # 이어지면 합친다
    window.add(500, 600)
    window.add(700, 700)  # 길이 0은 넣지 않는다
    assert window.total_ms() == 400
    assert window.first_start_ms() == 0
    assert window.since_ms(200) == 200  # [200, 300) + [500, 600)
    assert window.since_ms(550) == 50
    window.trim(1250)  # 250보다 오래된 부분을 자른다: [250, 300) + [500, 600)
    assert window.total_ms() == 150
    assert window.first_start_ms() == 250
    window.trim(1300)  # 첫 구간이 통째로 빠진다
    assert window.total_ms() == 100
    assert window.first_start_ms() == 500
    window.clear()
    assert window.total_ms() == 0


def test_interval_window_ignores_empty_intervals() -> None:
    window = IntervalWindow(1000)
    window.add(100, 100)  # 길이 0인 구간은 넣지 않는다
    window.add(300, 200)  # 끝이 시작보다 이른 구간도 넣지 않는다
    assert window.total_ms() == 0
    window.add(500, 600)
    # 빈 구간이 첫 구간으로 남으면 vats의 t_start_ms(창 안의 첫 영역 밖 구간의 시작)가 그 시각이 된다.
    assert window.first_start_ms() == 500
    window.add(700, 700)
    window.add(800, 900)
    assert window.total_ms() == 200
    assert window.since_ms(650) == 100
    window.trim(1650)  # 650보다 오래된 [500, 600)이 빠진다. 첫 구간은 다음의 실제 구간이다(700의 빈 구간이 아니다)
    assert window.first_start_ms() == 800


def test_ratio_with_a_zero_or_negative_length_is_zero() -> None:
    # PLAN 3.3절: 0으로 나누게 되는 비율은 0이다. 길이가 0인 구간의 confidence가 이 함수를 지난다.
    assert rules_module._ratio(5, 0) == 0.0
    assert rules_module._ratio(0, 0) == 0.0
    assert rules_module._ratio(5, -1) == 0.0
    assert rules_module._ratio(1, 4) == 0.25


# ---------------------------------------------------------------------------
# calibrating, provisional (D4)
# ---------------------------------------------------------------------------


def test_calibrating_onset_and_end(engine: RuleEngine) -> None:
    states = [st(5000 + t, provisional=True, pose_valid=False) for t in range(0, 60000, 100)]
    states += [st(5000 + t) for t in range(60000, 61000, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [(5000, EVENT_CALIBRATING, PHASE_ONSET), (65000, EVENT_CALIBRATING, PHASE_END)]
    onset, end = emitted[0][1], emitted[1][1]
    assert onset == DriverStateEvent(
        type=EVENT_CALIBRATING,
        severity=0,
        t_start_ms=5000,
        duration_ms=0,
        confidence=0.0,
        source=SOURCE_FACE,
        phase=PHASE_ONSET,
        provisional=True,
    )
    # 종료는 provisional이 False인 첫 프레임에서. 60초 내내 얼굴이 있었으므로 유효 시간 비율은 1이다.
    assert (end.t_start_ms, end.duration_ms, end.confidence, end.provisional) == (5000, 60000, 1.0, False)


def test_calibrating_confidence_is_face_time_ratio(engine: RuleEngine) -> None:
    states = [st(t, provisional=True, pose_valid=False, face=t >= 1000) for t in range(0, 4000, 100)]
    states.append(st(4000))
    end = feed(engine, states)[-1][1]
    assert (end.type, end.phase) == (EVENT_CALIBRATING, PHASE_END)
    # [0, 4000] 중 얼굴이 있던 시간은 1000~4000의 3000 ms
    assert end.confidence == pytest.approx(0.75)


def test_no_calibrating_when_first_frame_is_not_provisional(engine: RuleEngine) -> None:
    emitted = feed(engine, [st(t) for t in range(0, 3000, 100)])
    assert emitted == []


def test_gaze_rules_wait_for_baseline_but_closure_rules_run_provisionally(engine: RuleEngine) -> None:
    # 보정 중: 고개를 30도 돌린 채 5초, 그리고 눈을 1.5초 감는다. pose_valid는 False다.
    states = [st(t, yaw=30.0, provisional=True, pose_valid=False) for t in range(0, 5000, 100)]
    states += [st(t, 1.0, yaw=30.0, provisional=True, pose_valid=False) for t in range(5200, 6800, 100)]
    states += [st(t, yaw=30.0, provisional=True, pose_valid=False) for t in range(6800, 8000, 100)]
    emitted = feed(engine, states)
    assert of_type(emitted, EVENT_GAZE_AWAY_LONG, EVENT_VATS) == []
    closure_events = of_type(emitted, EVENT_MICROSLEEP, EVENT_SLEEP)
    assert brief(closure_events) == [(6200, EVENT_MICROSLEEP, PHASE_ONSET), (6800, EVENT_MICROSLEEP, PHASE_END)]
    # 임시 closure로 판정한 이벤트에는 provisional=true가 붙는다.
    assert all(e.provisional for _, e in closure_events)


# ---------------------------------------------------------------------------
# 눈 감김: 0.9, 1.1, 2.5, 3.0, 6초
# ---------------------------------------------------------------------------


def test_closure_0_9s_gives_no_event(engine: RuleEngine) -> None:
    # 감긴 프레임 1000~1900: 마지막 프레임에서 ongoing = 900 < 1000
    assert feed(engine, closure_session(1900, 3000)) == []


def test_closure_1_1s_is_microsleep(engine: RuleEngine) -> None:
    emitted = feed(engine, closure_session(2100, 3000))
    assert brief(emitted) == [(2000, EVENT_MICROSLEEP, PHASE_ONSET), (2200, EVENT_MICROSLEEP, PHASE_END)]
    onset, end = emitted[0][1], emitted[1][1]
    # 프레임 2000에서 ongoing = 1000(≥ 1000). t_start는 0.7을 위로 지난 시각 1000.
    assert onset == DriverStateEvent(
        type=EVENT_MICROSLEEP,
        severity=3,
        t_start_ms=1000,
        duration_ms=1000,
        confidence=1.0,
        source=SOURCE_CLOSURE,
        phase=PHASE_ONSET,
        provisional=False,
    )
    # 감김이 끝난 프레임 2200에서 end. duration은 발행 시점까지의 길이다.
    assert (end.t_start_ms, end.duration_ms, end.severity, end.source) == (1000, 1200, 3, SOURCE_CLOSURE)


def test_closure_2_5s_stays_microsleep(engine: RuleEngine) -> None:
    emitted = feed(engine, closure_session(3500, 5000))
    assert brief(emitted) == [(2000, EVENT_MICROSLEEP, PHASE_ONSET), (3600, EVENT_MICROSLEEP, PHASE_END)]


def test_closure_2_9s_is_not_sleep(engine: RuleEngine) -> None:
    emitted = feed(engine, closure_session(3900, 5000))
    assert brief(emitted) == [(2000, EVENT_MICROSLEEP, PHASE_ONSET), (4000, EVENT_MICROSLEEP, PHASE_END)]


def test_closure_3_0s_is_sleep(engine: RuleEngine) -> None:
    emitted = feed(engine, closure_session(4000, 5000))
    assert brief(emitted) == [
        (2000, EVENT_MICROSLEEP, PHASE_ONSET),
        (4000, EVENT_SLEEP, PHASE_ONSET),
        (4100, EVENT_SLEEP, PHASE_END),
    ]
    sleep_onset, end = emitted[1][1], emitted[2][1]
    # "3.0초 이상": 프레임 4000에서 ongoing이 정확히 3000이다. microsleep과 같은 t_start를 쓴다.
    assert (sleep_onset.t_start_ms, sleep_onset.duration_ms, sleep_onset.severity) == (1000, 3000, 3)
    # 끝날 때는 도달한 가장 높은 종류(sleep)의 end 하나만 낸다.
    assert (end.t_start_ms, end.duration_ms) == (1000, 3100)


def test_sleep_threshold_is_configurable(rules_config: RulesConfig) -> None:
    # 명세의 "3초 초과"는 sleep_ms = 3001로 표현한다: 정확히 3000에서는 sleep이 아니다.
    strict = RuleEngine(apply_overrides(rules_config, {"closure.sleep_ms": 3001}))
    emitted = feed(strict, closure_session(4000, 5000))
    assert brief(emitted) == [(2000, EVENT_MICROSLEEP, PHASE_ONSET), (4100, EVENT_MICROSLEEP, PHASE_END)]
    strict = RuleEngine(apply_overrides(rules_config, {"closure.sleep_ms": 3001}))
    emitted = feed(strict, closure_session(4100, 5000))
    assert brief(emitted)[1] == (4100, EVENT_SLEEP, PHASE_ONSET)


def test_closure_6s_sequence_and_alerts(rules_config: RulesConfig) -> None:
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    events: Emitted = []
    alerts: Emitted = []
    for state in closure_session(7000, 8000):
        frame_events = engine.update(state)
        events += [(state.t_ms, e) for e in frame_events]
        alerts += [(state.t_ms, a) for a in policy.decide(frame_events, 60.0, state.t_ms)]
    assert brief(events) == [
        (2000, EVENT_MICROSLEEP, PHASE_ONSET),
        (4000, EVENT_SLEEP, PHASE_ONSET),
        (7100, EVENT_SLEEP, PHASE_END),
    ]
    # 경고: microsleep(1.0초), sleep(3.0초), 그리고 sleep은 3초마다 되풀이(6.0초). 종료 뒤에는 없다.
    assert brief(alerts) == [
        (2000, EVENT_MICROSLEEP, PHASE_ONSET),
        (4000, EVENT_SLEEP, PHASE_ONSET),
        (7000, EVENT_SLEEP, PHASE_ONSET),
    ]
    repeat = alerts[2][1]
    # 되풀이 경고는 시작 이벤트의 사본이고 duration만 그 시각 기준이다.
    assert (repeat.t_start_ms, repeat.duration_ms, repeat.severity, repeat.source) == (1000, 6000, 3, SOURCE_CLOSURE)


def test_closure_start_is_interpolated(engine: RuleEngine) -> None:
    # 900(0.0) -> 1000(1.0): 0.7을 지난 시각은 970. 프레임 2000에서 ongoing = 1030.
    states = [st(t, 0.0) for t in range(0, 1000, 100)] + [st(t, 1.0) for t in range(1000, 2101, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [(2000, EVENT_MICROSLEEP, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms) == (970, 1030)
    assert onset.confidence == 1.0


def test_rule_detector_uses_0_7_and_0_5(engine: RuleEngine) -> None:
    # closure 0.69는 규칙용 진입 임계(0.7) 미만이다: 5초 이어져도 이벤트가 없다.
    assert feed(engine, [st(t, 0.69) for t in range(0, 5000, 100)]) == []
    # 0.7로 시작한 뒤 0.5~0.7에 머무는 동안에는 길이가 늘지 않는다(1초에 도달하지 못한다).
    states = [st(t, 0.0) for t in range(0, 701, 100)] + [st(1000, 0.7), st(1100, 0.7)]
    states += [st(t, 0.6) for t in range(1200, 4000, 100)]
    fresh = RuleEngine(engine._cfg)
    assert feed(fresh, states) == []


def test_closure_exactly_at_the_enter_threshold_counts_as_closed(engine: RuleEngine, rules_config: RulesConfig) -> None:
    # closure가 규칙용 진입 임계와 정확히 같은 값(0.7)으로 3.5초 이어진다. "이상"이므로 감긴 프레임이다.
    enter = rules_config.closure.detector.enter_thr
    assert enter == 0.7
    states = [st(t, 0.0) for t in range(0, 1000, 100)]
    states += [st(t, enter) for t in range(1000, 4500, 100)]
    states += [st(4500, 0.0)]
    emitted = feed(engine, states)
    # 900(0.0) -> 1000(0.7): 0.7에 닿는 자리는 쌍의 끝이라 감김의 시작은 1000이다. 2000에서 길이 1000, 4000에서 3000.
    # 4400(0.7) -> 4500(0.0): 0.7을 아래로 지난 자리는 쌍의 처음(4400)이고, 0.5 아래로 내려간 4500에서 끝난다.
    assert brief(emitted) == [
        (2000, EVENT_MICROSLEEP, PHASE_ONSET),
        (4000, EVENT_SLEEP, PHASE_ONSET),
        (4500, EVENT_SLEEP, PHASE_END),
    ]
    assert [(e.t_start_ms, e.duration_ms, e.confidence) for _, e in emitted] == [
        (1000, 1000, 1.0),
        (1000, 3000, 1.0),
        (1000, 3500, 1.0),
    ]


def test_closure_that_starts_at_time_zero_is_tracked_from_its_first_closed_frame(engine: RuleEngine) -> None:
    # 세션의 첫 프레임(t = 0)이 진입 임계 바로 아래(0.6999)이고 다음 프레임(100)이 1.0이다. 0.7을 지난 자리는
    # 100 ms 중 0.03 ms라 감김의 시작이 0으로 보간된다(start_ms가 0인 감김도 "새 감김"으로 기록돼야 한다).
    states = [st(0, 0.6999), st(100, 1.0), st(200, 1.0), st(300, valid=False), st(400, valid=False)]
    states += [st(t, 1.0) for t in range(500, 1101, 100)]
    states += [st(1200, 0.0)]
    emitted = feed(engine, states)
    # 끊긴 구간(200 -> 500, 300 ms)은 길이에 들어간다: 프레임 1000에서 1000 − 0 = 1000 ms.
    assert brief(emitted) == [(1000, EVENT_MICROSLEEP, PHASE_ONSET), (1200, EVENT_MICROSLEEP, PHASE_END)]
    onset, end = emitted[0][1], emitted[1][1]
    assert (onset.t_start_ms, onset.duration_ms) == (0, 1000)
    # confidence의 구간은 감김을 처음 본 프레임(100)부터다: [100, 1000]의 900 ms 중 프레임 300, 400이 무효 -> 700 / 900.
    # (세션의 처음부터 세면 800 / 1000이 된다.)
    assert onset.confidence == pytest.approx(700.0 / 900.0)
    # 1100(1.0) -> 1200(0.0): [100, 1200]의 1100 ms 중 900 ms
    assert (end.t_start_ms, end.duration_ms) == (0, 1200)
    assert end.confidence == pytest.approx(900.0 / 1100.0)


def test_no_onset_inside_hold_but_hold_counts_toward_length(engine: RuleEngine) -> None:
    states = [st(t, 0.0) for t in range(0, 701, 100)]
    states += [st(t, 1.0) for t in range(1000, 1501, 100)]
    states += [st(t, face=False) for t in range(1600, 2000, 100)]  # 끊긴 구간(1500 -> 2000, 500 ms)
    states += [st(t, 1.0) for t in range(2000, 2301, 100)]
    emitted = feed(engine, states)
    # 끊긴 구간 안(1600~1900)에서는 시작 이벤트를 내지 않고, 유효한 감김 프레임 2000에서 ongoing = 1000으로 낸다.
    assert brief(emitted) == [(2000, EVENT_MICROSLEEP, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms) == (1000, 1000)
    # [1000, 2000] 중 closure가 유효했던 시간: 프레임 1000~1500이 각각 100 ms씩 = 600 ms
    assert onset.confidence == pytest.approx(0.6)


def test_closure_truncated_by_face_loss_ends_event(engine: RuleEngine) -> None:
    states = [st(t, 0.0) for t in range(0, 701, 100)]
    states += [st(t, 1.0) for t in range(1000, 2501, 100)]
    states += [st(t, face=False) for t in range(2600, 4000, 100)]
    emitted = feed(engine, states)
    # 마지막 유효 프레임 2500 뒤 500 ms를 넘긴 첫 프레임(3100)에서 구간이 잘리고 end가 나간다.
    assert brief(emitted) == [(2000, EVENT_MICROSLEEP, PHASE_ONSET), (3100, EVENT_MICROSLEEP, PHASE_END)]
    end = emitted[1][1]
    assert (end.t_start_ms, end.duration_ms) == (1000, 2100)
    # [1000, 3100] 중 유효 시간: 프레임 1000~2500의 16개 x 100 ms
    assert end.confidence == pytest.approx(1600.0 / 2100.0)


def test_two_closures_give_two_event_pairs(engine: RuleEngine) -> None:
    states = closure_session(2200, 3000)
    states += [st(t, 1.0) for t in range(3300, 4501, 100)]  # 3000 -> 3300은 300 ms라 보간하지 않는다
    states += [st(t, 0.0) for t in range(4600, 5001, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [
        (2000, EVENT_MICROSLEEP, PHASE_ONSET),
        (2300, EVENT_MICROSLEEP, PHASE_END),
        (4300, EVENT_MICROSLEEP, PHASE_ONSET),
        (4600, EVENT_MICROSLEEP, PHASE_END),
    ]
    assert [e.t_start_ms for _, e in emitted] == [1000, 1000, 3300, 3300]


# ---------------------------------------------------------------------------
# 눈 감김: 반쯤 뜬 채로 머문 시간은 길이에 들어가지 않는다 (2026-11-13 결정)
# ---------------------------------------------------------------------------


def lingering_session(
    first: tuple[int, int],
    second: tuple[int, int],
    t_end_ms: int,
    *,
    dwell: float = 0.5,
    first_frame_shift_ms: int = 0,
) -> list[FrameState]:
    """감았다가(0.9) 반쯤 뜬 채로 머물고(dwell) 다시 감는(0.9) 세션. 100 ms 간격.

    눈 뜬 프레임 0~700, 감긴 프레임 first[0]~first[1], 머무는 프레임, 감긴 프레임 second[0]~second[1], 그 뒤 눈 뜬
    프레임(0.0)이다. first[0]은 1000이어야 한다(700과의 간격이 300 ms라 감김의 시작을 보간하지 않는다).
    `first_frame_shift_ms`는 첫 감긴 프레임의 시각만 그만큼 늦춘다(start_ms가 그만큼 늦어진다).

    규칙용 임계는 진입 0.7 / 이탈 0.5다. dwell이 0.5(이탈 임계 그대로, 감김이 끝나지 않는다)이면 0.7을 지나는 시각이
    두 프레임의 한가운데다: 앞 토막은 마지막 감긴 프레임 + 50 ms((0.2/0.4)·100)에서 끝나고 뒤 토막은 다시 감긴
    첫 프레임 − 50 ms에서 시작한다. dwell이 0.6이면 앞 토막은 마지막 감긴 프레임 + 67 ms((0.2/0.3)·100 = 66.7)에서
    끝나고, 뒤 토막은 머문 마지막 프레임 + 33 ms((0.1/0.3)·100 = 33.3), 곧 다시 감긴 첫 프레임 − 67 ms에서 시작한다.
    """
    assert first[0] == 1000
    states = [st(t, 0.0) for t in range(0, 701, 100)]
    for t in range(first[0], t_end_ms + 1, 100):
        if first[0] <= t <= first[1] or second[0] <= t <= second[1]:
            closure = 0.9
        elif first[1] < t < second[0]:
            closure = dwell
        else:
            closure = 0.0
        states.append(st(t + first_frame_shift_ms if t == first[0] else t, closure))
    return states


def test_half_open_dwell_between_two_blinks_gives_no_event(engine: RuleEngine, rules_config: RulesConfig) -> None:
    # 결정문의 확인 항목: 0.9로 300 ms(프레임 1000~1300), 0.6으로 5초(1400~6300), 0.9로 300 ms(6400~6700), 뜬 눈.
    states = lingering_session((1000, 1300), (6400, 6700), 8000, dwell=0.6)
    # 앞 토막 1000 ~ 1367(367 ms), 뒤 토막 6333 ~ . 마지막 감긴 프레임 6700에서 ongoing = 367 + 367 = 734 < 1000.
    # 2026-11-13 결정 전에는 프레임 6400에서 ongoing = 5400이 되어 microsleep과 sleep이 함께 났다.
    assert feed(engine, states) == []

    # 같은 수열로 규칙용 검출기를 따로 돌리면 구간은 하나이고 길이는 두 토막의 합이다.
    # 뒤 토막의 끝: 6700 + (0.2/0.9)·100 = 6722. 길이 367 + (6722 − 6333) = 756 ms
    detector = ClosureRunDetector(rules_config.closure.detector)
    runs = [detector.update(s.t_ms, s.closure, s.closure_valid) for s in states]
    assert [(r.start_ms, r.end_ms, r.duration_ms, r.truncated) for r in runs if r is not None] == [
        (1000, 6722, 756, False)
    ]


def test_microsleep_onset_when_accumulated_closed_time_reaches_1s(engine: RuleEngine) -> None:
    # 위와 같은데 둘째 감김이 800 ms(프레임 6400~7200)다.
    emitted = feed(engine, lingering_session((1000, 1300), (6400, 7200), 8000, dwell=0.6))
    # ongoing: 프레임 6400에서 367 + 67 = 434, 그 뒤 프레임마다 100씩. 6900에서 934, 7000에서 1034가 되어
    # 누적 감김 시간이 1000 ms에 이른 첫 감긴 프레임 7000에서 microsleep이 시작된다.
    # sleep은 없다(7200에서 1234). 감김이 끝난 프레임 7300에서 end.
    assert brief(emitted) == [(7000, EVENT_MICROSLEEP, PHASE_ONSET), (7300, EVENT_MICROSLEEP, PHASE_END)]
    onset, end = emitted[0][1], emitted[1][1]
    # t_start는 감김의 start_ms(0.7을 처음 위로 지난 시각), duration은 공통 규칙대로 발행 시각 − t_start다.
    # 그래서 duration(6000)은 시작 이벤트를 낸 누적 감김 시간(1000)보다 길다.
    assert onset == DriverStateEvent(
        type=EVENT_MICROSLEEP,
        severity=3,
        t_start_ms=1000,
        duration_ms=6000,
        confidence=1.0,
        source=SOURCE_CLOSURE,
        phase=PHASE_ONSET,
        provisional=False,
    )
    assert (end.t_start_ms, end.duration_ms, end.confidence) == (1000, 6300, 1.0)


@pytest.mark.parametrize(
    ("shift_ms", "microsleep_at", "sleep_at"),
    [
        # 앞 토막 1000 ~ 1450(450 ms), 뒤 토막은 2950에서 시작. 프레임 t(≥ 3000)에서 ongoing = 450 + (t − 2950).
        # 3500에서 정확히 1000, 5500에서 정확히 3000이다: "이상"이므로 그 프레임에서 시작한다.
        (0, 3500, 5500),
        # 첫 감긴 프레임이 1 ms 늦으면 앞 토막이 449 ms다. 3500에서 999, 5500에서 2999라 한 프레임씩 늦다.
        (1, 3600, 5600),
    ],
)
def test_accumulated_thresholds_are_inclusive(
    engine: RuleEngine, shift_ms: int, microsleep_at: int, sleep_at: int
) -> None:
    emitted = feed(engine, lingering_session((1000, 1400), (3000, 6000), 7000, first_frame_shift_ms=shift_ms))
    assert brief(emitted) == [
        (microsleep_at, EVENT_MICROSLEEP, PHASE_ONSET),
        (sleep_at, EVENT_SLEEP, PHASE_ONSET),
        (6100, EVENT_SLEEP, PHASE_END),
    ]
    start_ms = 1000 + shift_ms
    assert [(e.t_start_ms, e.duration_ms) for _, e in emitted] == [
        (start_ms, microsleep_at - start_ms),
        (start_ms, sleep_at - start_ms),
        (start_ms, 6100 - start_ms),
    ]


def test_sleep_after_microsleep_counts_only_closed_time(engine: RuleEngine) -> None:
    # 1.1초 감김(1000~2100)으로 microsleep이 시작된 뒤 반쯤 뜬 채로 1.9초 머물고(2200~4000) 다시 감는다(4100~5900).
    emitted = feed(engine, lingering_session((1000, 2100), (4100, 5900), 7000))
    # 앞 토막 1000 ~ 2150(1150 ms), 뒤 토막은 4050에서 시작. 프레임 4100에서 ongoing = 1200이고(결정 전에는 3100이라
    # 여기서 sleep이 났다) 5900에서 정확히 3000이 된다. 머무는 동안과 다시 감긴 프레임에서 microsleep이 다시 나지 않는다.
    assert brief(emitted) == [
        (2000, EVENT_MICROSLEEP, PHASE_ONSET),
        (5900, EVENT_SLEEP, PHASE_ONSET),
        (6000, EVENT_SLEEP, PHASE_END),
    ]
    assert [(e.t_start_ms, e.duration_ms) for _, e in emitted] == [(1000, 1000), (1000, 4900), (1000, 5000)]


def test_closure_confidence_with_dwell_covers_first_seen_frame_to_emit(engine: RuleEngine) -> None:
    states = lingering_session((1000, 1400), (3000, 3600), 5000)
    # 머무는 동안 프레임 2000~2300의 closure가 무효다(끊긴 시간 1900 -> 2400, 500 ms: 감김은 이어진다).
    states = [st(s.t_ms, valid=False) if 2000 <= s.t_ms <= 2300 else s for s in states]
    emitted = feed(engine, states)
    assert brief(emitted) == [(3500, EVENT_MICROSLEEP, PHASE_ONSET), (3700, EVENT_MICROSLEEP, PHASE_END)]
    onset, end = emitted[0][1], emitted[1][1]
    assert (onset.t_start_ms, onset.duration_ms) == (1000, 2500)
    # [감김을 처음 본 프레임 1000, 발행 시각 3500] 중 closure가 유효했던 시간: 25개 프레임 중 4개가 무효 -> 2100 / 2500
    assert onset.confidence == pytest.approx(2100.0 / 2500.0)
    assert (end.t_start_ms, end.duration_ms) == (1000, 2700)
    assert end.confidence == pytest.approx(2300.0 / 2700.0)


# ---------------------------------------------------------------------------
# 전방 주시와 장시간 시선 이탈: 2.9, 3.1초
# ---------------------------------------------------------------------------


def test_gaze_away_2_9s_gives_no_event(engine: RuleEngine) -> None:
    # 영역 밖 프레임 1000~3800, 3900부터 영역 안. 마지막 영역 밖 프레임에서 2800 ms.
    assert feed(engine, gaze_session(1000, 3900, 6000)) == []


def test_gaze_away_3_1s_onset_and_end(engine: RuleEngine) -> None:
    emitted = feed(engine, gaze_session(1000, 4100, 6000))
    assert brief(emitted) == [(4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET), (4400, EVENT_GAZE_AWAY_LONG, PHASE_END)]
    onset, end = emitted[0][1], emitted[1][1]
    # 시작 시각은 변화가 처음 나타난 프레임(1000)으로 소급한다. 프레임 4000에서 연속 3000 ms.
    assert onset == DriverStateEvent(
        type=EVENT_GAZE_AWAY_LONG,
        severity=2,
        t_start_ms=1000,
        duration_ms=3000,
        confidence=1.0,
        source=SOURCE_HEAD_POSE,
        phase=PHASE_ONSET,
        provisional=False,
    )
    # 영역 안 후보는 4100에서 시작하고 250 ms 뒤의 첫 프레임 4400에서 확정된다.
    assert (end.t_start_ms, end.duration_ms, end.source) == (1000, 3400, SOURCE_HEAD_POSE)


@pytest.mark.parametrize(
    ("kwargs", "expected_out"),
    [
        ({"yaw": 23.0}, False),
        ({"yaw": 23.01}, True),
        ({"yaw": -23.0}, False),
        ({"yaw": -23.01}, True),
        ({"pitch": -25.0}, False),  # 위쪽 25도까지
        ({"pitch": -25.01}, True),
        ({"pitch": 18.0}, False),  # 아래쪽 18도까지
        ({"pitch": 18.01}, True),
        ({"gaze_h": 0.25}, False),  # 눈동자 보조는 "넘으면" 영역 밖
        ({"gaze_h": 0.2501}, True),
        ({"gaze_h": -0.25}, False),  # 좌우와 위아래 모두 절댓값으로 비교한다
        ({"gaze_h": -0.2501}, True),
        ({"gaze_v": 0.35}, False),
        ({"gaze_v": 0.3501}, True),
        ({"gaze_v": -0.35}, False),
        ({"gaze_v": -0.3501}, True),
    ],
)
def test_forward_zone_boundaries(rules_config: RulesConfig, kwargs: dict[str, float], expected_out: bool) -> None:
    zone = ForwardZone(rules_config.forward_zone, rules_config.clock.max_hold_ms, rules_config.vats.window_ms)
    for t in range(0, 1000, 100):
        zone.update(st(t, **kwargs), True, 0)
    assert zone.label == (ZONE_OUT if expected_out else ZONE_IN)
    assert zone.zone == (ZONE_OUT if expected_out else ZONE_IN)


def test_eye_only_deviation_has_source_eye(engine: RuleEngine) -> None:
    emitted = feed(engine, gaze_session(1000, 4500, 5000, yaw=0.0, gaze_h=0.3))
    assert brief(emitted)[0] == (4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)
    assert emitted[0][1].source == SOURCE_EYE


def test_downward_eye_deviation_is_out_of_zone(engine: RuleEngine) -> None:
    # 눈동자 합성값의 세로 성분이 음수(아래)인 이탈: gaze_v_rel = −0.5가 3.5초. |−0.5| > 0.35라 영역 밖이다.
    states = [st(t) for t in range(0, 1000, 100)] + [st(t, gaze_v=-0.5) for t in range(1000, 4500, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [(4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms, onset.source, onset.confidence) == (1000, 3000, SOURCE_EYE, 1.0)


def test_head_condition_wins_over_eye(engine: RuleEngine) -> None:
    emitted = feed(engine, gaze_session(1000, 4500, 5000, yaw=30.0, gaze_h=0.3))
    assert emitted[0][1].source == SOURCE_HEAD_POSE


def test_eye_aux_can_be_disabled(rules_config: RulesConfig) -> None:
    engine = RuleEngine(apply_overrides(rules_config, {"forward_zone.eye_aux_enabled": False}))
    assert feed(engine, gaze_session(1000, 6000, 7000, yaw=0.0, gaze_h=0.9)) == []


def test_debounce_ignores_short_out_and_backdates(rules_config: RulesConfig) -> None:
    zone = ForwardZone(rules_config.forward_zone, rules_config.clock.max_hold_ms, rules_config.vats.window_ms)
    for t in (0, 100):
        zone.update(st(t), True, 0)
    # 영역 밖이 200 ms(프레임 200, 300, 400에서 보이고 500에서 돌아옴)면 확정되지 않는다.
    for t in (200, 300, 400):
        zone.update(st(t, yaw=30.0), True, 0)
    assert zone.zone == ZONE_IN  # 400 − 200 = 200 < 250
    zone.update(st(500), True, 0)
    assert zone.zone == ZONE_IN
    assert zone.away.total_ms() == 0
    # 다시 영역 밖: 600에서 후보, 900(300 ms 뒤)에서 확정. 시작 시각은 600으로 소급한다.
    for t in (600, 700, 800):
        zone.update(st(t, yaw=30.0), True, 0)
        assert zone.zone == ZONE_IN
    zone.update(st(900, yaw=30.0), True, 0)
    assert zone.zone == ZONE_OUT
    assert zone.out_start_ms == 600
    assert zone.away.total_ms() == 300


def test_debounce_boundary_is_exactly_250_ms(rules_config: RulesConfig) -> None:
    def zone_after(times: tuple[int, ...]) -> ForwardZone:
        zone = ForwardZone(rules_config.forward_zone, rules_config.clock.max_hold_ms, rules_config.vats.window_ms)
        for t in (0, 100):
            zone.update(st(t), True, 0)
        for t in times:
            zone.update(st(t, yaw=30.0), True, 0)
        return zone

    assert rules_config.forward_zone.debounce_ms == 250
    # 후보는 200에서 시작한다. 240 ms 뒤(440), 249 ms 뒤(449)의 프레임에서는 확정되지 않는다.
    assert zone_after((200, 440)).zone == ZONE_IN
    assert zone_after((200, 440, 449)).zone == ZONE_IN
    # 정확히 250 ms 뒤(450)의 프레임에서 확정된다("이상"). 시작 시각은 200으로 소급하고 영역 밖 누적은 250 ms다.
    for times in ((200, 450), (200, 440, 449, 450)):
        zone = zone_after(times)
        assert zone.zone == ZONE_OUT
        assert zone.out_start_ms == 200
        assert zone.away.total_ms() == 250
        assert zone.candidate_pending is False
    # 확정 전에는 후보가 걸려 있다.
    assert zone_after((200, 440)).candidate_pending is True


def test_candidate_survives_until_the_end_of_the_credit_inclusive(rules_config: RulesConfig) -> None:
    def zone_after(t_next: int) -> ForwardZone:
        zone = ForwardZone(rules_config.forward_zone, rules_config.clock.max_hold_ms, rules_config.vats.window_ms)
        for t in (0, 100):
            zone.update(st(t), True, 0)
        zone.update(st(200, yaw=30.0), True, 0)  # 영역 밖 후보. 이 관측은 400까지 유지되고 인정 구간은 3400까지다.
        zone.update(st(t_next, yaw=30.0), True, 0)
        return zone

    # 다음 프레임이 정확히 3400(= 200 + max_hold 200 + 인정 3000)에 오면 후보가 살아 있어 확정되고 200으로 소급한다.
    zone = zone_after(3400)
    assert (zone.zone, zone.out_start_ms, zone.away.total_ms()) == (ZONE_OUT, 200, 3200)
    # 1 ms 늦으면 후보를 버리고 그 프레임에서 새 후보를 시작한다(아직 확정 전).
    zone = zone_after(3401)
    assert (zone.zone, zone.away.total_ms(), zone.candidate_pending) == (ZONE_IN, 0, True)


def test_short_forward_glance_does_not_break_continuity(engine: RuleEngine) -> None:
    states: list[FrameState] = []
    for t in range(0, 6001, 100):
        out = 1000 <= t < 4500 and t not in (2600, 2700)  # 중간에 200 ms만 앞을 본다
        states.append(st(t, yaw=30.0 if out else 0.0))
    emitted = feed(engine, states)
    # 앞을 본 200 ms는 확정되지 않았으므로 영역 밖이 1000부터 이어진 것으로 본다.
    assert brief(emitted)[0] == (4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)
    assert emitted[0][1].t_start_ms == 1000
    # 영역 밖 누적에도 그 200 ms가 들어간다: [1000, 4500) 전체
    assert engine._zone.away.total_ms() == 3500


def test_no_onset_while_forward_candidate_is_pending(engine: RuleEngine) -> None:
    states = []
    for t in range(0, 5001, 100):
        states.append(st(t, yaw=30.0 if 1000 <= t < 3900 else 0.0))
    emitted = feed(engine, states)
    # 3900부터 앞을 본다. 4000, 4100에서 상태는 아직 영역 밖이지만(확정 전) 시작 이벤트를 내지 않는다.
    assert emitted == []
    assert engine._zone.zone == ZONE_IN
    assert engine._zone.away.total_ms() == 2900  # [1000, 3900)


def test_head_turn_with_face_loss_is_credited_for_3s(engine: RuleEngine) -> None:
    # 고개를 돌리자마자(1000, 1100) 얼굴이 사라진다(1200~). 영역 밖 후보는 이어받은 표지로 확정된다.
    states = [st(t) for t in range(0, 1000, 100)]
    states += [st(1000, yaw=30.0), st(1100, yaw=35.0)]
    states += [st(t, face=False) for t in range(1200, 9000, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [(4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms, onset.source) == (1000, 3000, SOURCE_HEAD_POSE)
    # [1000, 4000] 중 자세가 유효했던 시간은 프레임 1000, 1100의 200 ms
    assert onset.confidence == pytest.approx(200.0 / 3000.0)
    zone = engine._zone
    # 얼굴이 사라진 시각은 1200(마지막 유효 관측이 유지된 끝). 3초 뒤인 4200부터는 "알 수 없음"이라 누적이 멈춘다.
    assert zone.zone == ZONE_OUT
    assert zone.label == ZONE_UNKNOWN
    assert zone.away.total_ms() == 3200  # [1000, 4200)
    assert zone.unknown_ms == 8900 - 4200


def turn_then_face_lost(missing: Callable[[int], bool], t_end_ms: int = 9000) -> list[FrameState]:
    """앞을 보다가 1000, 1100, 1200 세 프레임에서 고개를 돌리고 1300부터 얼굴이 사라진다.

    마지막 관측(1200)은 1300까지 유지되므로 얼굴이 사라진 시각은 1300, 영역 밖으로 인정하는 구간은 4300 전까지다.
    연속 영역 밖 시간은 4000에서 3000 ms가 된다. `missing(t)`가 True인 프레임은 도착하지 않는다.
    """
    states = [st(t) for t in range(0, 1000, 100)]
    states += [st(t, yaw=30.0) for t in (1000, 1100, 1200)]
    states += [st(t, face=False) for t in range(1300, t_end_ms, 100) if not missing(t)]
    return states


def test_onset_due_inside_the_credit_window_is_emitted_on_the_next_frame(engine: RuleEngine) -> None:
    # 프레임 4000, 4100, 4200이 빠졌다: 임계 시간이 찬 뒤 인정 구간(4300 전) 안에 도착한 프레임이 하나도 없다.
    emitted = feed(engine, turn_then_face_lost(lambda t: 4000 <= t < 4300))
    # 구멍 뒤 첫 프레임 4300의 표지는 UNKNOWN이지만(인정 구간이 끝났다) 이미 찬 3300 ms로 시작 이벤트를 낸다.
    assert brief(emitted) == [(4300, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms, onset.source) == (1000, 3300, SOURCE_HEAD_POSE)
    # [1000, 4300] 중 자세가 유효했던 시간은 프레임 1000, 1100, 1200의 300 ms
    assert onset.confidence == pytest.approx(300.0 / 3300.0)
    zone = engine._zone
    assert (zone.zone, zone.label) == (ZONE_OUT, ZONE_UNKNOWN)
    assert zone.away.total_ms() == 3300  # [1000, 4300)
    assert zone.unknown_ms == 8900 - 4300


def test_onset_after_a_longer_hole_counts_only_the_credited_time(engine: RuleEngine) -> None:
    # 4000~5900의 프레임이 없다. 6000에서 연속 영역 밖 시간은 여전히 3300 ms(4300부터는 세지 않는다)다.
    emitted = feed(engine, turn_then_face_lost(lambda t: 4000 <= t < 6000))
    assert brief(emitted) == [(6000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms) == (1000, 5000)
    assert onset.confidence == pytest.approx(300.0 / 5000.0)
    assert engine._zone.away.total_ms() == 3300


def test_no_late_onset_before_the_threshold_was_reached(engine: RuleEngine) -> None:
    # 고개를 돌린 것이 1500부터라면(인정 구간은 4800 전까지) 4300에서는 2800 ms라 아직 내지 않고, 4500에서 낸다.
    states = [st(t) for t in range(0, 1500, 100)]
    states += [st(t, yaw=30.0) for t in (1500, 1600, 1700)]
    states += [st(t, face=False) for t in range(1800, 9000, 100) if not 4000 <= t < 4300]
    emitted = feed(engine, states)
    assert brief(emitted) == [(4500, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    assert emitted[0][1].duration_ms == 3000


def test_late_onset_ends_when_forward_is_confirmed(engine: RuleEngine) -> None:
    states = turn_then_face_lost(lambda t: 4000 <= t < 4300, t_end_ms=6000)
    states += [st(t) for t in range(6000, 7000, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [(4300, EVENT_GAZE_AWAY_LONG, PHASE_ONSET), (6300, EVENT_GAZE_AWAY_LONG, PHASE_END)]
    end = emitted[1][1]
    # [1000, 6300] 중 자세가 유효했던 시간: 1000~1200의 300 ms + 6000~6200의 300 ms
    assert (end.t_start_ms, end.duration_ms) == (1000, 5300)
    assert end.confidence == pytest.approx(600.0 / 5300.0)


def test_no_late_onset_when_the_driver_is_seen_looking_forward_after_the_hole(engine: RuleEngine) -> None:
    # 구멍 뒤 첫 프레임에서 얼굴이 보이고 앞을 보고 있다(표지 IN, 영역 안 후보). 시작 이벤트를 내지 않는다.
    states = turn_then_face_lost(lambda t: False, t_end_ms=4000)
    states += [st(t) for t in range(4300, 6000, 100)]
    assert feed(engine, states) == []
    assert engine._zone.zone == ZONE_IN


def test_no_gaze_onset_while_unable_to_analyze_after_a_stream_pause(engine: RuleEngine) -> None:
    # 20초 동안 앞을 보다가 0.5초 흘끗 옆을 본 순간(20000~20400) 프레임이 6초 끊기고, 돌아왔을 때는 앞을 보고 있다.
    states = [st(t) for t in range(0, 20000, 100)]
    states += [st(t, yaw=30.0) for t in range(20000, 20500, 100)]
    states += [st(t) for t in range(26400, 33000, 100)]
    emitted = feed(engine, states)
    # 끊긴 시간 중 3초가 영역 밖으로 인정되어 연속 시간은 3600 ms지만(20000~23600), 26400은 분석 불가가 시작된
    # 프레임이다(최근 10초의 얼굴 시간 4200 ms). 분석 불가 중에는 시선 이벤트를 내지 않고, 끝난 31400에서는
    # 앞을 보고 있으므로(표지 IN) 역시 내지 않는다.
    assert brief(emitted) == [
        (26400, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET),
        (31400, EVENT_UNABLE_TO_ANALYZE, PHASE_END),
    ]
    assert engine._zone.zone == ZONE_IN


def test_pending_forward_candidate_is_not_counted_as_away_at_the_credit_boundary(rules_config: RulesConfig) -> None:
    # 0.4초 옆을 보고(1000~1300, 1300에서 확정) 1400에서 앞을 본다(영역 안 후보). 그 뒤 프레임이 끊긴다.
    # 1400의 관측은 1600까지 유지되고 인정 구간은 4600 전까지다.
    for t_back in (4599, 4600, 4601):
        engine = RuleEngine(rules_config)
        states = [st(t) for t in range(0, 1000, 100)]
        states += [st(t, yaw=30.0) for t in range(1000, 1400, 100)]
        states += [st(1400)]
        states += [st(t, face=False) for t in range(t_back, t_back + 500, 100)]
        # 4599: 이어받은 IN 표지로 영역 안이 확정된다. 4601: 후보를 버리고 UNKNOWN(연속 영역 밖 시간 400 ms).
        # 4600: 표지는 UNKNOWN인데 후보가 아직 걸려 있어 t − out_start − unknown_ms가 3600으로 읽힌다.
        # 어느 경우에도 운전자는 0.4초만 옆을 봤으므로 이벤트가 없어야 한다.
        assert feed(engine, states) == [], f"t_back={t_back}"


def test_forward_candidate_dropped_at_a_hole_is_not_counted_as_away(engine: RuleEngine) -> None:
    # 2.0초 옆을 보고(1000~2900), 앞을 본 프레임 하나(3000, 영역 안 후보), 3.3초 동안 프레임이 없고, 다시 옆을 본다.
    states = [st(t) for t in range(0, 1000, 100)]
    states += [st(t, yaw=30.0) for t in range(1000, 3000, 100)]
    states += [st(3000)]
    states += [st(t, yaw=30.0) for t in range(6300, 8000, 100)]
    emitted = feed(engine, states)
    # 3000의 관측은 3200까지 유지되고 인정 구간은 6200까지다. 6300은 그 뒤라 후보를 버린다. 후보 구간(3000~6300)은
    # 영역 밖 시간이 아니므로 연속 시간은 6300에서 2000 ms로 이어지고, 7300에서 3000 ms가 된다.
    assert brief(emitted) == [(7300, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms, onset.source) == (1000, 6300, SOURCE_HEAD_POSE)
    # [1000, 7300]의 pose_valid 시간: 1000~2900의 2000 ms, 3000의 프레임이 유지된 200 ms, 6300~7200의 1000 ms
    assert onset.confidence == pytest.approx(3200.0 / 6300.0)
    zone = engine._zone
    assert zone.unknown_ms == 3300
    assert zone.away.total_ms() == 2000 + 1600  # [1000, 3000) + [6300, 7900)


def test_away_timer_pauses_during_unable_to_analyze_and_vats_starts_at_the_first_interval_in_its_window(
    engine: RuleEngine,
) -> None:
    """고개를 돌린 지 3초가 되기 전에 분석 불가가 시작되고, 분석 불가가 끝났을 때도 고개를 돌리고 있다.

    프레임은 100 ms 간격. 얼굴: 0~11900 정면, 12000~15900 없음, 16000~16400 옆(yaw 30), 16500~39900 없음, 40000~55900 옆,
    56000부터 정면.
    - 영역 밖 후보는 16000, 16300에서 확정된다. 16400의 관측은 16500까지 유지되고 그 뒤 3초는 영역 밖을 이어받는다.
    - 분석 불가: 프레임 t에서 최근 10초의 얼굴 시간은 (22000 − t) + 500. 17600에서 4900 ms라 시작된다.
      그때까지의 연속 영역 밖 시간은 1600 ms다. 분석 불가 중인 프레임의 표지는 UNKNOWN이라 시간이 멈춘다.
    - 얼굴이 40000에 돌아온다. 45000에서 얼굴 시간이 5000 ms가 되어 분석 불가가 끝난다(27400 ms 동안 얼굴 시간 5000).
    - 45000부터 다시 센다: 1600 + 1400 = 3000이 되는 46400에서 gaze_away_long이 시작된다.
    - vats: 영역 밖 구간은 [16000, 17600)과 [45000, ...)이다. 앞 구간은 47600에 30초 창을 벗어나고, 뒤 구간이
      10000 ms가 되는 55000에서 시작된다. t_start는 그때 창 안의 첫 구간의 시작인 45000이다(분석 불가 동안의
      프레임 시각이 아니다).
    """
    states = [st(t) for t in range(0, 12000, 100)]
    states += [st(t, face=False) for t in range(12000, 16000, 100)]
    states += [st(t, yaw=30.0) for t in range(16000, 16500, 100)]
    states += [st(t, face=False) for t in range(16500, 40000, 100)]
    states += [st(t, yaw=30.0) for t in range(40000, 56000, 100)]
    states += [st(t) for t in range(56000, 56500, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [
        (17600, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET),
        (45000, EVENT_UNABLE_TO_ANALYZE, PHASE_END),
        (46400, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (55000, EVENT_VATS, PHASE_ONSET),
        (56300, EVENT_GAZE_AWAY_LONG, PHASE_END),
    ]
    by_kind = {(e.type, e.phase): e for _, e in emitted}
    unable_end = by_kind[(EVENT_UNABLE_TO_ANALYZE, PHASE_END)]
    assert (unable_end.t_start_ms, unable_end.duration_ms) == (17600, 27400)
    assert unable_end.confidence == pytest.approx(5000.0 / 27400.0)
    gaze = by_kind[(EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    assert (gaze.t_start_ms, gaze.duration_ms, gaze.source) == (16000, 30400, SOURCE_HEAD_POSE)
    # [16000, 46400]의 pose_valid 시간: 16000~16400의 500 ms + 40000~46300의 6400 ms
    assert gaze.confidence == pytest.approx(6900.0 / 30400.0)
    vats = by_kind[(EVENT_VATS, PHASE_ONSET)]
    assert (vats.t_start_ms, vats.duration_ms, vats.confidence) == (45000, 10000, 1.0)
    gaze_end = by_kind[(EVENT_GAZE_AWAY_LONG, PHASE_END)]
    # [16000, 56300]: 500 + 40000~56200의 16300 ms
    assert (gaze_end.t_start_ms, gaze_end.duration_ms) == (16000, 40300)
    assert gaze_end.confidence == pytest.approx(16800.0 / 40300.0)


def test_gaze_zone_property(engine: RuleEngine) -> None:
    seen: dict[int, str] = {}
    states = [st(t, provisional=True, pose_valid=False) for t in range(0, 500, 100)]  # 기준선 완성 전
    states += [st(t) for t in range(500, 1000, 100)]
    states += [st(1000, yaw=30.0), st(1100, yaw=30.0)]
    states += [st(t, face=False) for t in range(1200, 5000, 100)]
    states += [st(t) for t in range(5000, 5500, 100)]
    for state in states:
        engine.update(state)
        seen[state.t_ms] = engine.gaze_zone
    assert seen[400] == ZONE_UNKNOWN  # 자세가 유효하지 않다
    assert seen[900] == ZONE_IN
    assert seen[1200] == ZONE_IN  # 영역 밖 후보가 걸려 있지만 아직 확정 전
    assert seen[1300] == ZONE_OUT  # 이어받은 표지로 확정
    assert seen[4100] == ZONE_OUT
    assert seen[4200] == ZONE_UNKNOWN  # 얼굴이 사라진 뒤 3초가 지났다
    assert seen[5000] == ZONE_OUT  # 영역 안 후보가 걸려 있지만 아직 확정 전
    assert seen[5300] == ZONE_IN


def test_face_returns_out_of_zone_after_unknown_resumes_same_event(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 1000, 100)]
    states += [st(t, yaw=30.0) for t in range(1000, 1500, 100)]
    states += [st(t, face=False) for t in range(1500, 7000, 100)]
    states += [st(t, yaw=30.0) for t in range(7000, 8000, 100)]
    states += [st(t) for t in range(8000, 9000, 100)]
    emitted = feed(engine, states)
    # 얼굴이 돌아와도 여전히 영역 밖이면 같은 이벤트가 이어지고, 영역 안이 확정될 때(8300) 끝난다.
    assert brief(emitted) == [(4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET), (8300, EVENT_GAZE_AWAY_LONG, PHASE_END)]
    end = emitted[1][1]
    assert (end.t_start_ms, end.duration_ms) == (1000, 7300)
    # 영역 밖 누적: [1000, 4500)(얼굴 소실 1500 + 3초) + [7000, 8000) = 4500 ms. "알 수 없음" 2500 ms는 빠진다.
    assert engine._zone.away.total_ms() == 4500


def test_short_face_loss_while_away_then_forward(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 1000, 100)]
    states += [st(1000, yaw=30.0), st(1100, yaw=30.0)]
    states += [st(t, face=False) for t in range(1200, 2500, 100)]
    states += [st(t) for t in range(2500, 4000, 100)]
    emitted = feed(engine, states)
    # 영역 밖 1.5초(1000~2500) 뒤 앞을 본다: 이벤트 없음. 영역 밖 누적은 [1000, 2500).
    assert emitted == []
    assert engine._zone.zone == ZONE_IN
    assert engine._zone.away.total_ms() == 1500


def test_face_loss_while_forward_never_counts_as_away(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 1000, 100)]
    states += [st(t, face=False) for t in range(1000, 9000, 100)]
    emitted = feed(engine, states)
    assert emitted == []
    assert engine._zone.zone == ZONE_IN
    assert engine._zone.away.total_ms() == 0


def test_data_gap_while_away_is_credited_like_face_loss(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 1000, 100)]
    states += [st(t, yaw=30.0) for t in range(1000, 2001, 100)]
    states.append(st(8000, yaw=30.0))  # 6초 동안 프레임이 없다(10초 이하라 초기화는 아니다)
    emitted = feed(engine, states)
    # 프레임 2000은 2200까지 유지되고 그 뒤 3초(5200까지)를 영역 밖으로 친다. 나머지 2800 ms는 알 수 없음.
    assert engine._zone.away.total_ms() == 4200
    assert engine._zone.unknown_ms == 2800
    # 프레임 8000에서 연속 영역 밖 시간 = 7000 − 2800 = 4200 ≥ 3000
    assert brief(emitted) == [(8000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    assert emitted[0][1].duration_ms == 7000


# ---------------------------------------------------------------------------
# VATS
# ---------------------------------------------------------------------------


def vats_cycles(n_cycles: int, t_end_ms: int, eye_cycles: tuple[int, ...] = ()) -> list[FrameState]:
    """5초 주기: [5000c + 1000, 5000c + 3900)에서 영역 밖(2.9초), 나머지는 앞을 본다(2.1초).

    `eye_cycles`에 든 주기에서는 고개는 정면이고 눈동자만 돌린다(gaze_h 0.3). 나머지 주기는 고개를 30도 돌린다.
    """
    states: list[FrameState] = []
    for t in range(0, t_end_ms + 1, 100):
        cycle, phase = divmod(t, 5000)
        out = cycle < n_cycles and 1000 <= phase < 3900
        if out and cycle in eye_cycles:
            states.append(st(t, gaze_h=0.3))
        else:
            states.append(st(t, yaw=30.0 if out else 0.0))
    return states


def test_vats_cumulative_10s_within_30s(engine: RuleEngine) -> None:
    emitted = feed(engine, vats_cycles(4, 40000))
    # 한 번에 3초를 넘지 않으므로 gaze_away_long은 없다.
    assert of_type(emitted, EVENT_GAZE_AWAY_LONG) == []
    assert brief(emitted) == [(17300, EVENT_VATS, PHASE_ONSET), (32700, EVENT_VATS, PHASE_END)]
    onset, end = emitted[0][1], emitted[1][1]
    # 누적: 2900·3 = 8700, 네 번째 구간(16000~)의 1300 ms째인 프레임 17300에서 10000.
    # t_start는 그 30초 안의 첫 영역 밖 구간 시작(1000).
    assert onset == DriverStateEvent(
        type=EVENT_VATS,
        severity=2,
        t_start_ms=1000,
        duration_ms=16300,
        confidence=1.0,
        source=SOURCE_HEAD_POSE,
        phase=PHASE_ONSET,
        provisional=False,
    )
    # 종료: 네 구간 합 11600에서 첫 구간이 창 밖으로 1700 ms 빠져나간 프레임 32700(합 9900).
    assert (end.t_start_ms, end.duration_ms) == (1000, 31700)
    # 자세는 내내 유효했다. 구간(31.7초)이 30초 창보다 길어도 비율은 [t_start, 발행 시각] 전체에서 잰다.
    assert end.confidence == 1.0
    assert end.source == SOURCE_HEAD_POSE


def test_vats_end_confidence_covers_the_whole_event(engine: RuleEngine) -> None:
    # 60초 동안 영역 밖(1000~61000), 자세는 내내 유효하다.
    states = [st(t, yaw=30.0 if 1000 <= t < 61000 else 0.0) for t in range(0, 95001, 100)]
    vats = of_type(feed(engine, states), EVENT_VATS)
    # 영역 밖 구간은 [1000, 61000). 최근 30초의 합 = 91000 − t가 10000 아래로 내려가는 첫 프레임은 81100이다.
    assert brief(vats) == [(11000, EVENT_VATS, PHASE_ONSET), (81100, EVENT_VATS, PHASE_END)]
    end = vats[1][1]
    assert (end.t_start_ms, end.duration_ms) == (1000, 80100)
    # 80.1초 구간 전체에서 자세가 유효했다(최근 30초만 세면 30000/80100 = 0.37이 된다).
    assert end.confidence == 1.0


def test_vats_end_confidence_with_a_pose_invalid_stretch(engine: RuleEngine) -> None:
    # 40초 동안 영역 밖(1000~41000). 그중 20000~25000의 프레임은 얼굴은 있지만 자세가 유효하지 않다.
    states = [
        st(t, yaw=30.0 if 1000 <= t < 41000 else 0.0, pose_valid=not (20000 <= t < 25000))
        for t in range(0, 70001, 100)
    ]
    emitted = feed(engine, states)
    assert brief(emitted) == [
        (4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (11000, EVENT_VATS, PHASE_ONSET),
        (41300, EVENT_GAZE_AWAY_LONG, PHASE_END),
        (61100, EVENT_VATS, PHASE_END),
    ]
    vats_onset, gaze_end, vats_end = emitted[1][1], emitted[2][1], emitted[3][1]
    assert (vats_onset.t_start_ms, vats_onset.duration_ms, vats_onset.confidence) == (1000, 10000, 1.0)
    # gaze_away_long: [1000, 41300]의 40300 ms 중 자세가 유효하지 않았던 5000 ms를 뺀다.
    assert gaze_end.confidence == pytest.approx(35300.0 / 40300.0)
    # 영역 밖 누적은 [1000, 23000)(자세가 무효가 된 20000부터 3초까지 인정)과 [25000, 41000)이다.
    # 최근 30초의 합은 t > 55000부터 71000 − t이고, 10000 아래로 내려가는 첫 프레임은 61100이다.
    assert (vats_end.t_start_ms, vats_end.duration_ms) == (1000, 60100)
    # vats: [1000, 61100]의 60100 ms 중 5000 ms를 뺀다. 무효 구간은 종료 시점의 30초 창(31100~)보다 앞에 있다.
    assert vats_end.confidence == pytest.approx(55100.0 / 60100.0)


def test_vats_onset_confidence_counts_pose_time_from_t_start(engine: RuleEngine) -> None:
    # 시작 전에 자세가 유효하지 않은 구간(5000~7000)이 있다. 3초보다 짧아 영역 밖 누적은 끊기지 않는다.
    states = [
        st(t, yaw=30.0 if 1000 <= t < 13000 else 0.0, pose_valid=not (5000 <= t < 7000)) for t in range(0, 14001, 100)
    ]
    vats = of_type(feed(engine, states), EVENT_VATS)
    assert brief(vats) == [(11000, EVENT_VATS, PHASE_ONSET)]
    # [1000, 11000]의 10000 ms 중 2000 ms가 유효하지 않았다.
    assert vats[0][1].confidence == pytest.approx(0.8)


def test_pose_window_is_trimmed_to_the_vats_window(engine: RuleEngine, rules_config: RulesConfig) -> None:
    # pose_valid가 프레임마다 켜졌다 꺼지는 100초: pose_valid 구간이 한 프레임 걸러 하나씩 새로 생긴다. 이 창은 vats가
    # 시작될 때 t_start 시점의 누적을 구하는 데만 쓰지만, 30초보다 오래된 구간은 버려야 한다(버리지 않아도 이벤트는
    # 같지만 세션 길이만큼 구간이 쌓인다).
    assert rules_config.vats.window_ms == 30000
    states = [st(t, pose_valid=(t // 100) % 2 == 0) for t in range(0, 100001, 100)]
    assert feed(engine, states) == []
    window = engine._pose_window
    # 마지막 프레임 100000에서 70000보다 오래된 구간이 빠졌다: [70000, 70100), [70200, 70300), ..., [99800, 99900)
    assert len(window._starts) == 150
    assert window.total_ms() == 15000
    assert window.first_start_ms() == 70000


def test_vats_source_is_fixed_at_onset(rules_config: RulesConfig) -> None:
    # 눈동자만 돌린 경우: 시작과 종료 모두 source가 eye다.
    engine = RuleEngine(rules_config)
    emitted = feed(engine, vats_cycles(4, 40000, eye_cycles=(0, 1, 2, 3)))
    assert brief(emitted) == [(17300, EVENT_VATS, PHASE_ONSET), (32700, EVENT_VATS, PHASE_END)]
    assert [e.source for _, e in emitted] == [SOURCE_EYE, SOURCE_EYE]

    # 앞의 세 번은 고개, 시작 이벤트가 나는 네 번째만 눈동자, 다섯 번째는 다시 고개를 돌린다.
    engine = RuleEngine(rules_config)
    emitted = feed(engine, vats_cycles(5, 45000, eye_cycles=(3,)))
    # 다섯 구간의 합 14500. 첫 구간이 빠지면 11600, 둘째 구간(6000~8900)이 1700 ms 빠진 37700에서 9900이 된다.
    assert brief(emitted) == [(17300, EVENT_VATS, PHASE_ONSET), (37700, EVENT_VATS, PHASE_END)]
    onset, end = emitted[0][1], emitted[1][1]
    # source는 시작할 때의 마지막 영역 밖 관측(눈동자)이고, 종료 이벤트도 그 값을 쓴다(그 뒤의 고개 돌림과 무관).
    assert (onset.source, onset.t_start_ms) == (SOURCE_EYE, 1000)
    assert (end.source, end.t_start_ms, end.duration_ms, end.confidence) == (SOURCE_EYE, 1000, 36700, 1.0)


def test_vats_does_not_fire_below_10s(engine: RuleEngine) -> None:
    emitted = feed(engine, vats_cycles(3, 40000))  # 8.7초
    assert emitted == []


def test_vats_window_forgets_old_segments(engine: RuleEngine) -> None:
    # 2.9초 이탈을 12초 간격으로 반복하면 30초 창 안에는 많아야 세 구간(8.7초)만 들어온다.
    states: list[FrameState] = []
    for t in range(0, 120001, 100):
        states.append(st(t, yaw=30.0 if t % 12000 < 2900 else 0.0))
    assert feed(engine, states) == []


def test_vats_stays_active_while_window_sum_is_at_least_10s(engine: RuleEngine) -> None:
    # 20초 동안 영역 밖(1000~21000), 15초 앞을 본 뒤 다시 14초 영역 밖(36000~50000).
    states: list[FrameState] = []
    for t in range(0, 60001, 100):
        out = 1000 <= t < 21000 or 36000 <= t < 50000
        states.append(st(t, yaw=30.0 if out else 0.0))
    vats = of_type(feed(engine, states), EVENT_VATS)
    # 프레임 11000에서 누적 10000으로 시작한다.
    # 그 뒤 최근 30초의 합: 31000까지 20000, 36000에서 15000(첫 구간이 창 밖으로 빠져나가는 중),
    # 36000~50000은 빠져나가는 만큼 새 구간이 늘어 15000 그대로, 51000부터는 둘째 구간 14000.
    # 한 번도 10000 아래로 내려가지 않으므로 60초 안에는 종료도 재시작도 없다.
    assert brief(vats) == [(11000, EVENT_VATS, PHASE_ONSET)]
    assert vats[0][1].t_start_ms == 1000
    assert engine._zone.away.total_ms() == 14000
    assert engine._zone.away.first_start_ms() == 36000


def test_vats_reset_option(rules_config: RulesConfig) -> None:
    # D6: 초기화를 켜면 2.9초 이탈과 2.1초 주시를 반복해도 이벤트가 나지 않는다.
    engine = RuleEngine(apply_overrides(rules_config, {"vats.reset_enabled": True}))
    assert feed(engine, vats_cycles(12, 60000)) == []
    # 끄면(기본값) 같은 입력에서 난다.
    default = RuleEngine(rules_config)
    assert brief(feed(default, vats_cycles(12, 60000)))[0] == (17300, EVENT_VATS, PHASE_ONSET)


def test_vats_reset_needs_full_2s_of_forward(rules_config: RulesConfig) -> None:
    engine = RuleEngine(apply_overrides(rules_config, {"vats.reset_enabled": True}))
    # 2.9초 이탈과 1.9초 주시(주기 4.8초): 전방 주시가 2초에 못 미쳐 초기화되지 않는다.
    states: list[FrameState] = []
    for t in range(0, 40001, 100):
        states.append(st(t, yaw=30.0 if 1000 <= t % 4800 < 3900 else 0.0))
    emitted = feed(engine, states)
    assert [e.type for _, e in emitted][0] == EVENT_VATS


def test_long_continuous_away_gives_both_events(engine: RuleEngine) -> None:
    emitted = feed(engine, gaze_session(1000, 13000, 14000))
    assert brief(emitted) == [
        (4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (11000, EVENT_VATS, PHASE_ONSET),
        (13300, EVENT_GAZE_AWAY_LONG, PHASE_END),
    ]
    assert emitted[1][1].t_start_ms == 1000


# ---------------------------------------------------------------------------
# 분석 불가
# ---------------------------------------------------------------------------


def test_unable_to_analyze_onset_and_end(engine: RuleEngine) -> None:
    states = [st(t, face=not (12000 <= t < 20000)) for t in range(0, 30001, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [
        (17100, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET),
        (25000, EVENT_UNABLE_TO_ANALYZE, PHASE_END),
    ]
    onset, end = emitted[0][1], emitted[1][1]
    # 얼굴 시간은 [0, 12000). 프레임 17100에서 최근 10초의 얼굴 시간이 4900 ms(비율 0.49 < 0.5)가 된다.
    assert onset == DriverStateEvent(
        type=EVENT_UNABLE_TO_ANALYZE,
        severity=1,
        t_start_ms=17100,
        duration_ms=0,
        confidence=0.0,
        source=SOURCE_FACE,
        phase=PHASE_ONSET,
        provisional=False,
    )
    # 얼굴이 20000에 돌아오고 25000에서 최근 10초의 얼굴 시간이 5000 ms(비율 0.5)가 된다.
    assert (end.t_start_ms, end.duration_ms) == (17100, 7900)
    # [17100, 25000] 중 얼굴이 있던 시간은 5000 ms
    assert end.confidence == pytest.approx(5000.0 / 7900.0)


def test_unable_ratio_exactly_half_is_not_unable(engine: RuleEngine) -> None:
    states = [st(t, face=not (12000 <= t < 17000)) for t in range(0, 25001, 100)]
    # 얼굴 없는 시간이 정확히 5초: 비율은 0.5까지만 내려간다.
    assert feed(engine, states) == []


def test_unable_is_not_judged_in_the_first_10s(engine: RuleEngine) -> None:
    states = [st(t, face=False, provisional=True) for t in range(0, 12001, 100)]
    emitted = of_type(feed(engine, states), EVENT_UNABLE_TO_ANALYZE)
    # 세션 시작 후 10초가 지난 첫 프레임에서 처음 판정한다.
    assert brief(emitted) == [(10000, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)]
    assert emitted[0][1].provisional is True


@pytest.mark.parametrize("first_t_ms", [1, 5000, 9999, 10000, 123456])
def test_first_10s_are_measured_from_the_first_frame_whatever_its_timestamp(
    rules_config: RulesConfig, first_t_ms: int
) -> None:
    # "처음 10초"는 t_ms 0이 아니라 세션의 첫 프레임부터 잰다. 처음부터 얼굴이 없는 세션은 첫 프레임에서 정확히
    # 10000 ms 뒤의 프레임에서 분석 불가가 시작된다. 구간의 시작을 0으로 두면(첫 프레임에서 적어 두지 않으면) 첫 t_ms가
    # 5000일 때 5000 ms 뒤의 프레임, 9999일 때 100 ms 뒤의 프레임, 10000일 때 첫 프레임에서 시작한다. 첫 t_ms가 0이나
    # 1이면, 또는 10000을 넘어 끊김 초기화가 그 값을 대신 적어 주면 차이가 드러나지 않는다.
    states = [st(first_t_ms + t, face=False) for t in range(0, 12001, 100)]
    emitted = feed(RuleEngine(rules_config), states)
    assert brief(emitted) == [(first_t_ms + 10000, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms, onset.confidence) == (first_t_ms + 10000, 0, 0.0)


def test_missing_data_counts_as_missing_face(engine: RuleEngine) -> None:
    # 500 ms 간격의 프레임: 프레임마다 200 ms만 인정되므로 얼굴 시간 비율은 0.4다.
    emitted = feed(engine, [st(t) for t in range(0, 15001, 500)])
    assert brief(emitted) == [(10000, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)]


def test_unable_pauses_gaze_and_closure_rules(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 12000, 100)]
    states += [st(t, face=False) for t in range(12000, 20000, 100)]  # 17100에서 분석 불가 시작
    # 얼굴이 돌아오자마자 눈을 감고 고개를 돌린 채 4초: 분석 불가가 끝나는 25000 전에는 누적하지 않는다.
    states += [st(t, 1.0, yaw=30.0) for t in range(20000, 24000, 100)]
    states += [st(t) for t in range(24000, 26000, 100)]
    emitted = feed(engine, states)
    assert brief(emitted) == [
        (17100, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET),
        (25000, EVENT_UNABLE_TO_ANALYZE, PHASE_END),
    ]
    assert engine._zone.away.total_ms() == 0


def test_rules_resume_after_unable_ends(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 12000, 100)]
    states += [st(t, face=False) for t in range(12000, 20000, 100)]
    states += [st(t) for t in range(20000, 25000, 100)]
    states += [st(t, 1.0, yaw=30.0) for t in range(25000, 29001, 100)]
    emitted = feed(engine, states)
    # 분석 불가는 25000에서 끝난다. 그 프레임부터 감김과 시선이 다시 누적된다. 분석 불가 동안 규칙용 검출기는
    # 무효 프레임만 받았으므로 직전 유효 표본이 멀어 감김 시작을 보간하지 않는다(start = 25000).
    assert brief(emitted) == [
        (17100, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET),
        (25000, EVENT_UNABLE_TO_ANALYZE, PHASE_END),
        (26000, EVENT_MICROSLEEP, PHASE_ONSET),
        (28000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (28000, EVENT_SLEEP, PHASE_ONSET),
    ]
    assert emitted[2][1].t_start_ms == 25000
    assert emitted[3][1].t_start_ms == 25000


# ---------------------------------------------------------------------------
# 10초 넘는 끊김 뒤의 초기화
# ---------------------------------------------------------------------------


def test_gap_over_10s_ends_active_events_and_resets(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 12000, 100)]
    states += [st(t, 1.0, yaw=30.0) for t in range(12300, 16001, 100)]  # 감김과 시선 이탈이 함께 진행 중
    emitted = feed(engine, states)
    assert brief(emitted) == [
        (13300, EVENT_MICROSLEEP, PHASE_ONSET),
        (15300, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (15300, EVENT_SLEEP, PHASE_ONSET),
    ]
    # 12초 뒤에 온 프레임: 진행 중이던 이벤트를 끝내고 초기화한다. 세션을 시작할 때처럼 분석 불가의 창도 새로
    # 시작하므로(2026-11-13 소유자 결정) 창 안의 얼굴 시간이 0이어도 분석 불가는 시작되지 않는다.
    after = [(28000, e) for e in engine.update(st(28000, 1.0, yaw=30.0))]
    assert brief(after) == [
        (28000, EVENT_GAZE_AWAY_LONG, PHASE_END),
        (28000, EVENT_SLEEP, PHASE_END),
    ]
    assert after[0][1].t_start_ms == 12300
    assert after[1][1].t_start_ms == 12300
    assert engine._zone.zone == ZONE_IN
    assert engine._zone.away.total_ms() == 0
    assert engine._face_window.total_ms() == 0
    # 그 프레임부터 규칙이 바로 돈다: 감김과 시선 이탈이 28000에서 새로 시작된 것으로 센다(보간할 직전 표본이 없다).
    later = feed(engine, [st(t, 1.0, yaw=30.0) for t in range(28100, 31001, 100)])
    assert brief(later) == [
        (29000, EVENT_MICROSLEEP, PHASE_ONSET),
        (31000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (31000, EVENT_SLEEP, PHASE_ONSET),
    ]
    assert [e.t_start_ms for _, e in later] == [28000, 28000, 28000]


def test_gap_over_10s_drops_a_pending_out_candidate(engine: RuleEngine) -> None:
    # 12000에서 처음 고개를 돌린 프레임 하나(영역 밖 후보)를 본 뒤 10.001초 끊기고, 끊김 뒤에도 고개를 돌리고 있다.
    states = [st(t) for t in range(0, 12000, 100)] + [st(12000, yaw=30.0)]
    assert feed(engine, states) == []
    assert engine._zone.candidate_pending is True
    # 초기화가 걸려 있던 후보를 버린다. 끊김 뒤 첫 프레임(22001)이 새 후보의 시작이고 영역 밖 시간은 거기서부터 센다.
    # (후보가 남아 있으면 22001에서 12000으로 소급해 확정되고 그 프레임에서 gaze_away_long과 vats가 함께 난다.)
    after = [st(22001 + 100 * i, yaw=30.0) for i in range(32)]
    emitted = feed(engine, after)
    assert brief(emitted) == [(25001, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    onset = emitted[0][1]
    assert (onset.t_start_ms, onset.duration_ms, onset.confidence) == (22001, 3000, 1.0)
    assert engine._zone.out_start_ms == 22001
    assert engine._zone.away.total_ms() == 3100  # [22001, 25101)


def test_gap_over_10s_restarts_like_a_fresh_session(rules_config: RulesConfig) -> None:
    # 2026-11-13 소유자 결정의 확인 사례: 30 fps(33 ms), 80 km/h, 12초 끊김 뒤 첫 프레임부터 9초 동안 눈을 감는다.
    # 끊김 뒤 프레임은 t0 + 33n. 길이 33n이 1000 이상이 되는 첫 프레임은 n = 31(1023), 3000 이상은 n = 91(3003).
    # 마지막 감긴 프레임은 n = 272(8976)이고 t0 + 9000의 뜬 프레임에서 끝난다.
    def closed_then_open(t0: int) -> list[FrameState]:
        states = [st(t, 1.0) for t in range(t0, t0 + 9000, 33)]
        return states + [st(t) for t in range(t0 + 9000, t0 + 17000, 33)]

    def run(states: list[FrameState], t0: int) -> tuple[list[tuple[int, str, str, int, int, float]], list]:
        engine = RuleEngine(rules_config)
        policy = AlertPolicy(rules_config)
        events = []
        alerts = []
        for state in states:
            frame_events = engine.update(state)
            for e in frame_events:
                events.append((state.t_ms - t0, e.type, e.phase, e.t_start_ms - t0, e.duration_ms, e.confidence))
            alerts += [(state.t_ms - t0, a.type, a.duration_ms) for a in policy.decide(frame_events, 80.0, state.t_ms)]
        return events, alerts

    before = [st(t) for t in range(0, 30000, 33)]
    assert before[-1].t_ms == 29997
    t0 = 29997 + 12000
    events, alerts = run(before + closed_then_open(t0), t0)
    expected_events = [
        (1023, EVENT_MICROSLEEP, PHASE_ONSET, 0, 1023, 1.0),
        (3003, EVENT_SLEEP, PHASE_ONSET, 0, 3003, 1.0),
        (9000, EVENT_SLEEP, PHASE_END, 0, 9000, 1.0),
    ]
    # 분석 불가 이벤트도 경고도 없다. sleep은 3003 뒤 3초가 지난 첫 프레임(n = 182, 6006)에서 한 번 되풀이한다.
    expected_alerts = [(1023, EVENT_MICROSLEEP, 1023), (3003, EVENT_SLEEP, 3003), (6006, EVENT_SLEEP, 6006)]
    assert events == expected_events
    assert alerts == expected_alerts
    # 세션을 새로 시작해서 같은 프레임을 넣은 것과 같다.
    assert run(closed_then_open(0), 0) == (expected_events, expected_alerts)


def test_gap_over_10s_ends_unable_to_analyze_and_restarts_its_window(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 12000, 100)]
    states += [st(t, face=False) for t in range(12000, 19500, 100)]  # 17100에서 분석 불가 시작
    states += [st(t) for t in range(19500, 20000, 100)]  # 얼굴이 돌아온 프레임 다섯(19500~19900)
    assert brief(feed(engine, states)) == [(17100, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)]
    # 19900에서 최근 10초의 얼굴 시간은 [9900, 12000) 2100 + [19500, 19900) 400 = 2500 ms라 아직 분석 불가다.
    # 10.1초 뒤(30000)의 프레임에서 끝난다. [17100, 30000]의 얼굴 시간은 400 + 마지막 프레임이 유지된 200 = 600 ms.
    after = engine.update(st(30000))
    assert [(e.type, e.phase, e.t_start_ms, e.duration_ms) for e in after] == [
        (EVENT_UNABLE_TO_ANALYZE, PHASE_END, 17100, 12900)
    ]
    assert after[0].confidence == pytest.approx(600.0 / 12900.0)
    assert engine._face_window.total_ms() == 0
    # 30000부터 10초 동안은 판정하지 않는다. 얼굴은 30000~30900에만 있다(그 뒤 창의 얼굴 시간 1000 ms, 비율 0.1).
    # 39999(9999 ms 지남)에서는 아직 판정하지 않고 40000에서 처음 판정해 분석 불가가 시작된다.
    later = [st(t) for t in range(30100, 31000, 100)]
    later += [st(t, face=False) for t in range(31000, 39901, 100)]
    later += [st(39999, face=False), st(40000, face=False), st(40100, face=False)]
    emitted = feed(engine, later)
    assert brief(emitted) == [(40000, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)]
    assert emitted[0][1].t_start_ms == 40000


@pytest.mark.parametrize(
    ("gap_ms", "unable"),
    [
        (5200, False),  # 창의 얼굴 시간 4800 + 유지 200 = 5000 ms: 비율 0.5는 분석 불가가 아니다
        (5201, True),  # 4999 ms
        (10000, True),  # 정확히 10초는 초기화가 아니다: 평소처럼 얼굴 없는 시간으로 센다
        (10001, False),  # 10초를 넘으면 초기화: 창을 새로 시작한다
    ],
)
def test_gap_up_to_10s_counts_as_missing_face_and_longer_gaps_restart(
    rules_config: RulesConfig, gap_ms: int, unable: bool
) -> None:
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    assert feed(engine, [st(t) for t in range(0, 20001, 100)]) == []
    events = engine.update(st(20000 + gap_ms))
    expected = [(EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)] if unable else []
    assert [(e.type, e.phase) for e in events] == expected
    # 분석 불가 경고는 속도로 억제하지 않는다.
    assert [(a.type, a.phase) for a in policy.decide(events, 0.0, 20000 + gap_ms)] == expected


def feed_with_steps(
    engine: RuleEngine,
    states: list[FrameState],
    perclos: Callable[[int], float | None] | None = None,
    p_drowsy: Callable[[int], float | None] | None = None,
) -> Emitted:
    """`DriverPipeline`처럼 부른다: t_ms가 1000의 배수인 프레임에서 스텝 k = t_ms / 1000의 `on_step`을 먼저, 그다음 `update`.

    `perclos(k)`가 None을 주는 k에서는 스텝을 내지 않는다(둘 다 안 주면 스텝이 없다). missing은 0.25로 고정한다.
    """
    out: Emitted = []
    for state in states:
        k = state.t_ms // 1000
        if state.t_ms % 1000 == 0 and k > 0 and (perclos is not None or p_drowsy is not None):
            ratio = perclos(k) if perclos is not None else 0.0
            p = p_drowsy(k) if p_drowsy is not None else None
            if ratio is not None:
                for e in engine.on_step(step(k, perclos=ratio, missing=0.25), None if p is None else probs(p)):
                    out.append((state.t_ms, e))
        for e in engine.update(state):
            out.append((state.t_ms, e))
    return out


def test_gap_over_10s_ends_drowsy_and_resets_the_perclos_rule(engine: RuleEngine) -> None:
    # perclos 0.2가 스텝 1~10에서 이어져 프레임 10000에서 시작한다(t_start는 스텝 1의 끝).
    emitted = feed_with_steps(engine, [st(t) for t in range(0, 12001, 100)], perclos=lambda k: 0.2)
    assert [(t, e.type, e.phase, e.t_start_ms) for t, e in emitted] == [(10000, EVENT_DROWSY, PHASE_ONSET, 1000)]
    # 10.001초 뒤의 프레임에서 끝난다: 발행 시각은 그 프레임의 t_ms, confidence 0.0, provisional은 항상 False.
    after = engine.update(st(22001, provisional=True))
    assert after == [
        DriverStateEvent(
            type=EVENT_DROWSY,
            severity=2,
            t_start_ms=1000,
            duration_ms=21001,
            confidence=0.0,
            source=SOURCE_PERCLOS,
            phase=PHASE_END,
            provisional=False,
        )
    ]
    # 판정기가 처음 상태다: 끊김 뒤의 스텝 23~32(열 개)가 다시 차야 시작하고, t_start는 스텝 23의 끝이다.
    later = feed_with_steps(engine, [st(t) for t in range(22100, 33001, 100)], perclos=lambda k: 0.2)
    assert [(t, e.type, e.phase, e.t_start_ms, e.duration_ms) for t, e in later] == [
        (32000, EVENT_DROWSY, PHASE_ONSET, 23000, 9000)
    ]


def test_gap_over_10s_clears_the_onset_streak(engine: RuleEngine) -> None:
    # 스텝 1~9(아홉 개)까지 조건을 채우고 10.001초 끊긴다. 끊김이 없었다면 스텝 10에서 시작한다.
    assert feed_with_steps(engine, [st(t) for t in range(0, 9001, 100)], perclos=lambda k: 0.2) == []
    assert engine.update(st(19001)) == []
    emitted = feed_with_steps(engine, [st(t) for t in range(19100, 29001, 100)], perclos=lambda k: 0.2)
    # 스텝 20~29가 새로 열 개를 채운다(스텝 번호가 이어지는지와 무관하게 끊김 전의 아홉 개는 버린다).
    assert [(t, e.phase, e.t_start_ms) for t, e in emitted] == [(29000, PHASE_ONSET, 20000)]

    # 번호가 이어지는 스텝을 직접 넣어도 같다: 스텝 10 하나로는 시작하지 않는다.
    other = RuleEngine(engine._cfg)
    assert feed_with_steps(other, [st(t) for t in range(0, 9001, 100)], perclos=lambda k: 0.2) == []
    assert other.update(st(19001)) == []
    assert other.on_step(step(10, perclos=0.2), None) == []
    emitted = [(k, e) for k in range(11, 20) for e in other.on_step(step(k, perclos=0.2), None)]
    assert [(k, e.phase, e.t_start_ms) for k, e in emitted] == [(19, PHASE_ONSET, 10000)]


def test_gap_over_10s_clears_the_refractory_memory(model_rules: RulesConfig) -> None:
    engine = RuleEngine(model_rules)
    # 스텝 1~10에서 시작, 11~20에서 종료(20000). 재시작 금지는 80000까지다.
    emitted = feed_with_steps(
        engine, [st(t) for t in range(0, 20001, 100)], p_drowsy=lambda k: 0.9 if k <= 10 else 0.1
    )
    assert [(t, e.phase) for t, e in emitted] == [(10000, PHASE_ONSET), (20000, PHASE_END)]
    # 끊김 뒤의 첫 프레임: 진행 중인 상태가 없으므로 이벤트는 없다.
    assert engine.update(st(30001)) == []
    # 스텝 31~40이 차면 금지 시간(80000 전)이어도 시작한다. 초기화가 없었다면 스텝 80에서 시작한다.
    later = feed_with_steps(engine, [st(t) for t in range(30100, 41001, 100)], p_drowsy=lambda k: 0.9)
    assert [(t, e.phase, e.t_start_ms, e.source) for t, e in later] == [(40000, PHASE_ONSET, 31000, SOURCE_MODEL)]


def test_gap_of_exactly_10s_keeps_unable_to_analyze(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 12000, 100)]
    states += [st(t, face=False) for t in range(12000, 20000, 100)]  # 17100에서 분석 불가 시작
    assert brief(feed(engine, states)) == [(17100, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)]
    # 19900 뒤 정확히 10초: 초기화가 아니다. 분석 불가는 이어지고 평소처럼 얼굴 시간이 5초가 되는 프레임에서 끝난다.
    assert engine.update(st(29900)) == []
    emitted = feed(engine, [st(t) for t in range(30000, 35001, 100)])
    assert [(t, e.type, e.phase, e.t_start_ms) for t, e in emitted] == [
        (34900, EVENT_UNABLE_TO_ANALYZE, PHASE_END, 17100)
    ]


def test_gap_of_exactly_10s_keeps_drowsy_and_its_streak(engine: RuleEngine) -> None:
    assert feed_with_steps(engine, [st(t) for t in range(0, 9001, 100)], perclos=lambda k: 0.2) == []
    # 9000 뒤 정확히 10초(19000): 스텝 10이 열 번째로 이어져 시작하고, 그 프레임의 update는 drowsy를 끝내지 않는다.
    # (초기화가 아니므로 끊긴 10초는 얼굴 없는 시간이다. 창의 얼굴 시간이 200 ms뿐이라 분석 불가가 시작된다.)
    onset = engine.on_step(step(10, perclos=0.2), None)
    assert [(e.phase, e.t_start_ms) for e in onset] == [(PHASE_ONSET, 1000)]
    assert [(e.type, e.phase) for e in engine.update(st(19000))] == [(EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)]
    # 상태가 살아 있다: 0.10 미만이 열 스텝(11~20) 이어지면 같은 t_start로 끝난다.
    ends = [(k, e) for k in range(11, 21) for e in engine.on_step(step(k, perclos=0.0), None)]
    assert [(k, e.phase, e.t_start_ms, e.duration_ms) for k, e in ends] == [(20, PHASE_END, 1000, 19000)]


def test_gap_over_10s_ends_every_active_state_in_the_documented_order(rules_config: RulesConfig) -> None:
    """모든 상태가 진행 중일 때의 초기화. 프레임은 500 ms 간격(얼굴 시간 비율 0.4)이고 내내 눈을 감고 고개를 돌린다.

    손 계산(프레임마다 200 ms만 유지된 것으로 센다)
    - 0: calibrating 시작(실제로는 없는 조합이지만 순서를 보려고 provisional과 pose_valid를 함께 True로 둔다).
    - 500: 영역 밖 확정(후보 0, 250 ms 이상) -> 영역 밖 상태의 시작은 0.
    - 1000: 감김 길이 1000 -> microsleep.  3000: gaze_away_long(3000 − 0), sleep(3000).
    - 10000: 스텝 1~10의 perclos 0.2 -> drowsy(스텝 이벤트가 먼저). 처음 판정하는 분석 불가: 얼굴 시간
      20 x 200 = 4000 ms(0.4) -> 시작. 영역 밖 합 [0, 10000) = 10000 ms -> vats. 감김은 무효 프레임을 받아 유지 상태다.
    - 20001(10.001초 뒤, provisional=False): 초기화. 유효 시간 누적은 21 x 200 = 4200 ms.
    """
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    states = [st(t, 1.0, yaw=30.0, provisional=True) for t in range(0, 10001, 500)]
    before: Emitted = []
    for state in states:
        events: list[DriverStateEvent] = []
        if state.t_ms % 1000 == 0 and state.t_ms > 0:
            events += engine.on_step(step(state.t_ms // 1000, perclos=0.2, missing=0.25), None)
        events += engine.update(state)
        before += [(state.t_ms, e) for e in events]
        # 5 km/h: 속도 기준이 있는 경고는 모두 막힌다. 분석 불가만 경고한다.
        alerts = policy.decide(events, 5.0, state.t_ms)
        assert [a.type for a in alerts] == ([EVENT_UNABLE_TO_ANALYZE] if state.t_ms == 10000 else [])
    assert brief(before) == [
        (0, EVENT_CALIBRATING, PHASE_ONSET),
        (1000, EVENT_MICROSLEEP, PHASE_ONSET),
        (3000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (3000, EVENT_SLEEP, PHASE_ONSET),
        (10000, EVENT_DROWSY, PHASE_ONSET),
        (10000, EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET),
        (10000, EVENT_VATS, PHASE_ONSET),
    ]
    # 미뤄 둔 경고: microsleep은 sleep 시작에서 지워졌고, sleep은 되풀이 기록으로 기다린다.
    assert set(policy._pending) == {EVENT_GAZE_AWAY_LONG, EVENT_VATS, EVENT_DROWSY}
    assert set(policy._repeat_onset) == {EVENT_SLEEP}

    after = engine.update(st(20001, 1.0, yaw=30.0))
    assert [(e.type, e.phase, e.t_start_ms, e.duration_ms, e.source, e.provisional) for e in after] == [
        (EVENT_DROWSY, PHASE_END, 1000, 19001, SOURCE_PERCLOS, False),
        (EVENT_UNABLE_TO_ANALYZE, PHASE_END, 10000, 10001, SOURCE_FACE, False),
        (EVENT_GAZE_AWAY_LONG, PHASE_END, 0, 20001, SOURCE_HEAD_POSE, False),
        (EVENT_SLEEP, PHASE_END, 0, 20001, SOURCE_CLOSURE, False),
        (EVENT_CALIBRATING, PHASE_END, 0, 20001, SOURCE_FACE, False),
        (EVENT_VATS, PHASE_END, 0, 20001, SOURCE_HEAD_POSE, False),  # 비워진 합 때문에 평소 판정에서 끝난다
    ]
    # drowsy는 0.0. 분석 불가는 [10000, 20001]의 얼굴 시간 200 ms. 나머지는 [0, 20001]의 유효 시간 4200 ms.
    assert [e.confidence for e in after] == pytest.approx(
        [0.0, 200.0 / 10001.0, 4200.0 / 20001.0, 4200.0 / 20001.0, 4200.0 / 20001.0, 4200.0 / 20001.0]
    )
    # 속도가 80 km/h로 올라 막힘이 풀려도, 상태가 모두 끝났으므로 미뤄 둔 경고는 나가지 않는다.
    assert policy.decide(after, 80.0, 20001) == []
    assert policy._pending == {}
    assert policy._repeat_onset == {}
    for t in range(20101, 23000, 100):
        assert policy.decide(engine.update(st(t)), 80.0, t) == []


def test_gap_over_10s_ends_vats_through_the_cleared_sum_and_drops_its_pending_alert(
    rules_config: RulesConfig,
) -> None:
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    # 1000부터 고개를 돌린다: gaze_away_long은 4000, vats는 영역 밖 합이 10초가 되는 11000에서 시작한다.
    # 5 km/h라 둘 다 경고하지 못하고 미뤄 둔다.
    before: Emitted = []
    for t in range(0, 15001, 100):
        events = engine.update(st(t, yaw=30.0 if t >= 1000 else 0.0))
        before += [(t, e) for e in events]
        assert policy.decide(events, 5.0, t) == []
    assert brief(before) == [(4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET), (11000, EVENT_VATS, PHASE_ONSET)]
    assert set(policy._pending) == {EVENT_GAZE_AWAY_LONG, EVENT_VATS}
    # 12초 뒤(27000)에 앞을 보는 프레임, 80 km/h. gaze_away_long은 초기화가 끝내고, vats는 합이 비워져서 같은
    # 프레임의 평소 판정에서 끝난다. [1000, 27000]의 pose_valid 시간은 14000 + 마지막 프레임이 유지된 200 ms.
    after = engine.update(st(27000))
    assert [(e.type, e.phase, e.t_start_ms, e.duration_ms) for e in after] == [
        (EVENT_GAZE_AWAY_LONG, PHASE_END, 1000, 26000),
        (EVENT_VATS, PHASE_END, 1000, 26000),
    ]
    assert [e.confidence for e in after] == pytest.approx([14200.0 / 26000.0, 14200.0 / 26000.0])
    # 막힘이 풀린 호출이지만 두 상태가 그 호출에서 끝났으므로 경고는 없다.
    assert policy.decide(after, 80.0, 27000) == []
    assert policy._pending == {}
    for t in range(27100, 30001, 100):
        assert policy.decide(engine.update(st(t)), 80.0, t) == []


def test_gap_of_exactly_10s_does_not_reset(engine: RuleEngine) -> None:
    states = [st(t) for t in range(0, 12000, 100)]
    states += [st(t, yaw=30.0) for t in range(12000, 16001, 100)]
    feed(engine, states)
    after = engine.update(st(26000, yaw=30.0))
    # 초기화가 아니므로 시선 이벤트는 끝나지 않는다. (얼굴 시간이 모자라 분석 불가만 시작된다.)
    assert [(e.type, e.phase) for e in after] == [(EVENT_UNABLE_TO_ANALYZE, PHASE_ONSET)]
    assert engine._zone.zone == ZONE_OUT


# ---------------------------------------------------------------------------
# 졸음: 모델 판정
# ---------------------------------------------------------------------------


def run_steps(engine: RuleEngine, values: list[float | None], first_k: int = 1) -> list[tuple[int, DriverStateEvent]]:
    """스텝마다 p_drowsy(None이면 확률 없음)를 넣고 (k, 이벤트) 목록을 돌려준다."""
    out: list[tuple[int, DriverStateEvent]] = []
    for i, value in enumerate(values):
        k = first_k + i
        for e in engine.on_step(step(k), None if value is None else probs(value)):
            out.append((k, e))
    return out


def test_model_drowsy_onset_after_10_consecutive_steps(model_rules: RulesConfig) -> None:
    engine = RuleEngine(model_rules)
    emitted = run_steps(engine, [0.5] * 5 + [0.6] * 12)
    assert [(k, e.type, e.phase) for k, e in emitted] == [(15, EVENT_DROWSY, PHASE_ONSET)]
    onset = emitted[0][1]
    # 연속 구간의 첫 스텝은 k = 6. t_start는 그 스텝의 끝, 발행은 열 번째 스텝(k = 15)의 끝.
    assert onset == DriverStateEvent(
        type=EVENT_DROWSY,
        severity=2,
        t_start_ms=6000,
        duration_ms=9000,
        confidence=0.6,
        source=SOURCE_MODEL,
        phase=PHASE_ONSET,
        provisional=False,
    )


def test_model_drowsy_needs_consecutive_steps(model_rules: RulesConfig) -> None:
    engine = RuleEngine(model_rules)
    # 9스텝 뒤에 0.59가 끼면 다시 센다.
    emitted = run_steps(engine, [0.6] * 9 + [0.59] + [0.7] * 9)
    assert emitted == []
    emitted = run_steps(engine, [0.7], first_k=20)
    assert [(k, e.phase) for k, e in emitted] == [(20, PHASE_ONSET)]
    assert emitted[0][1].t_start_ms == 11000


def test_model_drowsy_end_after_10_steps_at_or_below_0_4(model_rules: RulesConfig) -> None:
    engine = RuleEngine(model_rules)
    values: list[float | None] = [0.9] * 10 + [0.41] * 12 + [0.4] * 9 + [0.5] + [0.4] * 10
    emitted = run_steps(engine, values)
    # 0.41은 종료 조건(≤ 0.4)이 아니다. 0.4가 9스텝 이어진 뒤 0.5가 끼면 다시 센다.
    assert [(k, e.phase) for k, e in emitted] == [(10, PHASE_ONSET), (42, PHASE_END)]
    end = emitted[1][1]
    assert (end.t_start_ms, end.duration_ms, end.source) == (1000, 41000, SOURCE_MODEL)
    assert end.confidence == 0.4


def test_model_drowsy_refractory_60s(model_rules: RulesConfig) -> None:
    engine = RuleEngine(model_rules)
    # 스텝 1~10 시작, 11~20 종료(k = 20, 20000 ms). 그 뒤로 계속 0.9.
    emitted = run_steps(engine, [0.9] * 10 + [0.1] * 10 + [0.9] * 70)
    assert [(k, e.phase) for k, e in emitted] == [(10, PHASE_ONSET), (20, PHASE_END), (80, PHASE_ONSET)]
    # 종료 후 60초(80000 ms)가 될 때까지 다시 시작하지 않는다. t_start는 연속 구간의 첫 스텝(k = 21)의 끝이다.
    again = emitted[2][1]
    assert (again.t_start_ms, again.duration_ms) == (21000, 59000)


def test_model_drowsy_missing_probability_breaks_the_run(model_rules: RulesConfig) -> None:
    engine = RuleEngine(model_rules)
    emitted = run_steps(engine, [0.9] * 9 + [None] + [0.9] * 9)
    assert emitted == []
    emitted = run_steps(engine, [0.9], first_k=20)
    assert [(k, e.phase) for k, e in emitted] == [(20, PHASE_ONSET)]


def test_model_drowsy_step_number_jump_breaks_the_run(model_rules: RulesConfig) -> None:
    engine = RuleEngine(model_rules)
    assert run_steps(engine, [0.9] * 9, first_k=1) == []
    # 스텝 번호가 건너뛰면(10~19가 없다) 연속이 아니다.
    emitted = run_steps(engine, [0.9] * 10, first_k=20)
    assert [(k, e.phase) for k, e in emitted] == [(29, PHASE_ONSET)]
    assert emitted[0][1].t_start_ms == 20000


def test_model_source_ignores_perclos(model_rules: RulesConfig) -> None:
    engine = RuleEngine(model_rules)
    for k in range(1, 40):
        assert engine.on_step(step(k, perclos=0.9), probs(0.1)) == []


# ---------------------------------------------------------------------------
# 졸음: PERCLOS 판정, either
# ---------------------------------------------------------------------------


def run_perclos(
    engine: RuleEngine, values: list[float], first_k: int = 1, full: bool = True
) -> list[tuple[int, DriverStateEvent]]:
    out: list[tuple[int, DriverStateEvent]] = []
    for i, value in enumerate(values):
        k = first_k + i
        for e in engine.on_step(step(k, perclos=value, full=full, missing=0.25), None):
            out.append((k, e))
    return out


def test_perclos_drowsy_onset_and_end(engine: RuleEngine) -> None:
    values = [0.14] * 3 + [0.15] * 10 + [0.10] * 12 + [0.0999] * 10
    emitted = run_perclos(engine, values)
    # 시작: perclos_60 ≥ 0.15가 10스텝 연속(k = 4~13). 종료: < 0.10이 10스텝 연속(정확히 0.10은 아니다).
    assert [(k, e.type, e.phase) for k, e in emitted] == [
        (13, EVENT_DROWSY, PHASE_ONSET),
        (35, EVENT_DROWSY, PHASE_END),
    ]
    onset, end = emitted[0][1], emitted[1][1]
    assert onset == DriverStateEvent(
        type=EVENT_DROWSY,
        severity=2,
        t_start_ms=4000,
        duration_ms=9000,
        confidence=0.75,  # 1 − face_missing_ratio_60
        source=SOURCE_PERCLOS,
        phase=PHASE_ONSET,
        provisional=False,
    )
    assert (end.t_start_ms, end.duration_ms, end.source) == (4000, 31000, SOURCE_PERCLOS)


def test_perclos_drowsy_has_no_refractory_by_default(engine: RuleEngine) -> None:
    emitted = run_perclos(engine, [0.2] * 10 + [0.0] * 10 + [0.2] * 10)
    assert [(k, e.phase) for k, e in emitted] == [(10, PHASE_ONSET), (20, PHASE_END), (30, PHASE_ONSET)]


def test_perclos_rule_ignores_steps_without_full_lookback(engine: RuleEngine) -> None:
    # 조회 구간이 덜 찬 스텝은 값이 없는 스텝이다.
    assert run_perclos(engine, [0.9] * 30, first_k=61, full=False) == []
    emitted = run_perclos(engine, [0.9] * 10, first_k=91)
    assert [(k, e.phase) for k, e in emitted] == [(100, PHASE_ONSET)]
    assert emitted[0][1].t_start_ms == 91000


def test_perclos_source_ignores_model_probabilities(engine: RuleEngine) -> None:
    for k in range(1, 40):
        assert engine.on_step(step(k, perclos=0.0), probs(0.99)) == []


def test_either_source_is_the_union(rules_config: RulesConfig) -> None:
    engine = RuleEngine(apply_overrides(rules_config, {"drowsiness.source": "either"}))
    emitted: list[tuple[int, DriverStateEvent]] = []
    for k in range(1, 61):
        p = 0.9 if k <= 30 else 0.1  # 모델: 10에서 시작, 40에서 종료
        perclos = 0.3 if 16 <= k <= 45 else 0.0  # PERCLOS: 25에서 시작, 55에서 종료
        for e in engine.on_step(step(k, perclos=perclos), probs(p)):
            emitted.append((k, e))
    # 둘 중 하나라도 판정 중이면 drowsy다: 모델이 연 상태가 PERCLOS 쪽이 끝날 때까지 이어진다.
    assert [(k, e.phase, e.source) for k, e in emitted] == [
        (10, PHASE_ONSET, SOURCE_MODEL),
        (55, PHASE_END, SOURCE_MODEL),
    ]
    assert [e.t_start_ms for _, e in emitted] == [1000, 1000]


def test_either_source_opened_by_perclos(rules_config: RulesConfig) -> None:
    engine = RuleEngine(apply_overrides(rules_config, {"drowsiness.source": "either"}))
    emitted: list[tuple[int, DriverStateEvent]] = []
    for k in range(1, 40):
        perclos = 0.3 if k <= 12 else 0.0
        for e in engine.on_step(step(k, perclos=perclos, missing=0.1), None):
            emitted.append((k, e))
    assert [(k, e.phase, e.source) for k, e in emitted] == [
        (10, PHASE_ONSET, SOURCE_PERCLOS),
        (22, PHASE_END, SOURCE_PERCLOS),
    ]
    assert emitted[0][1].confidence == pytest.approx(0.9)


def test_either_source_model_episode_then_perclos_episode(rules_config: RulesConfig) -> None:
    engine = RuleEngine(apply_overrides(rules_config, {"drowsiness.source": "either"}))
    emitted: list[tuple[int, DriverStateEvent]] = []
    for k in range(1, 71):
        p = 0.9 if k <= 12 else 0.1  # 모델: 스텝 1~10에서 시작, 13~22에서 종료
        perclos = 0.4 if 25 <= k <= 60 else 0.0  # PERCLOS: 25~34에서 시작, 61~70에서 종료
        for e in engine.on_step(step(k, perclos=perclos, missing=0.25), probs(p)):
            emitted.append((k, e))
    # 두 번째 상태는 PERCLOS 쪽이 열었다: source와 t_start_ms, confidence(1 − face_missing_ratio_60)가 모두 그쪽 것이다.
    # 앞서 모델이 연 상태의 source가 남아 있으면 안 된다.
    assert [(k, e.phase, e.source, e.t_start_ms, e.duration_ms, e.confidence) for k, e in emitted] == [
        (10, PHASE_ONSET, SOURCE_MODEL, 1000, 9000, 0.9),
        (22, PHASE_END, SOURCE_MODEL, 1000, 21000, 0.1),
        (34, PHASE_ONSET, SOURCE_PERCLOS, 25000, 9000, 0.75),
        (70, PHASE_END, SOURCE_PERCLOS, 25000, 45000, 0.75),
    ]


def test_end_streak_is_broken_by_a_step_without_a_value_and_by_a_jump_in_the_step_number() -> None:
    def active_rule() -> StepHysteresis:
        rule = StepHysteresis(onset_steps=2, end_steps=3, refractory_ms=0)
        rule.update(1, 1000, True, True, False)
        rule.update(2, 2000, True, True, False)
        assert rule.active and rule.start_ms == 1000
        return rule

    # 종료 조건 둘(k = 3, 4), 값 없는 스텝(5), 종료 조건 둘(6, 7): 연속 셋이 아니므로 끝나지 않고, k = 8에서 끝난다.
    rule = active_rule()
    for k, has_value in ((3, True), (4, True), (5, False), (6, True), (7, True)):
        rule.update(k, 1000 * k, has_value, False, True)
        assert rule.active, k
    rule.update(8, 8000, True, False, True)
    assert not rule.active

    # 종료 조건 둘(k = 3, 4) 뒤에 번호가 건너뛴다(5가 없다). 6, 7로는 끝나지 않고 k = 8에서 끝난다.
    rule = active_rule()
    for k in (3, 4, 6, 7):
        rule.update(k, 1000 * k, True, False, True)
        assert rule.active, k
    rule.update(8, 8000, True, False, True)
    assert not rule.active


def test_model_drowsy_end_needs_ten_consecutive_steps(model_rules: RulesConfig) -> None:
    # 시작(스텝 10) 뒤 종료 조건(0.1) 아홉, 확률 없는 스텝 하나, 다시 아홉: 끝나지 않는다. 열 번째(k = 30)에서 끝난다.
    engine = RuleEngine(model_rules)
    emitted = run_steps(engine, [0.9] * 10 + [0.1] * 9 + [None] + [0.1] * 9)
    assert [(k, e.phase) for k, e in emitted] == [(10, PHASE_ONSET)]
    emitted = run_steps(engine, [0.1], first_k=30)
    assert [(k, e.phase, e.t_start_ms, e.duration_ms) for k, e in emitted] == [(30, PHASE_END, 1000, 29000)]

    # 종료 조건 아홉(11~19) 뒤에 스텝 번호가 건너뛴다(20~29가 없다). 30~38의 아홉으로는 끝나지 않고 39에서 끝난다.
    engine = RuleEngine(model_rules)
    assert [(k, e.phase) for k, e in run_steps(engine, [0.9] * 10 + [0.1] * 9)] == [(10, PHASE_ONSET)]
    assert run_steps(engine, [0.1] * 9, first_k=30) == []
    emitted = run_steps(engine, [0.1], first_k=39)
    assert [(k, e.phase, e.t_start_ms, e.duration_ms) for k, e in emitted] == [(39, PHASE_END, 1000, 38000)]


def test_step_hysteresis_counts_during_refractory() -> None:
    rule = StepHysteresis(onset_steps=2, end_steps=2, refractory_ms=5000)
    hits = [True, True, False, False, True, True, True, True, True, True]
    ends = [not h for h in hits]
    active = []
    for i, (hit, end) in enumerate(zip(hits, ends)):
        rule.update(i + 1, 1000 * (i + 1), True, hit, end)
        active.append(rule.active)
    # 시작 k=2, 종료 k=4(4000). 다시 시작은 4000 + 5000 = 9000부터: k = 9.
    assert active == [False, True, True, False, False, False, False, False, True, True]
    assert rule.start_ms == 5000  # 다시 시작한 연속 구간의 첫 스텝(k = 5)의 끝


# ---------------------------------------------------------------------------
# 경고 결정: 속도 기준, 재경고 대기, 우선순위
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event_type", "speed", "expected"),
    [
        (EVENT_MICROSLEEP, 19.9, False),
        (EVENT_MICROSLEEP, 20.0, True),
        (EVENT_SLEEP, 19.9, False),
        (EVENT_SLEEP, 20.0, True),
        (EVENT_GAZE_AWAY_LONG, 19.9, False),
        (EVENT_GAZE_AWAY_LONG, 20.0, True),
        (EVENT_VATS, 19.9, False),
        (EVENT_VATS, 20.0, True),
        (EVENT_DROWSY, 49.9, False),  # 졸음 경고는 50 km/h부터
        (EVENT_DROWSY, 50.0, True),
        (EVENT_DROWSY, None, True),  # 속도 입력이 없으면 억제하지 않는다
        (EVENT_MICROSLEEP, None, True),
        (EVENT_UNABLE_TO_ANALYZE, 0.0, True),  # 속도 억제 없음
        (EVENT_UNABLE_TO_ANALYZE, -1.0, True),  # 기준이 0이면 어떤 속도 값에서도 억제하지 않는다(음수가 들어와도)
        (EVENT_MICROSLEEP, -1.0, False),  # 기준이 있는 종류는 기준 미만이라 억제한다
        (EVENT_CALIBRATING, None, False),  # 경고하지 않는다
        (EVENT_CALIBRATING, 100.0, False),
    ],
)
def test_speed_gate(rules_config: RulesConfig, event_type: str, speed: float | None, expected: bool) -> None:
    policy = AlertPolicy(rules_config)
    onset = event(event_type, 1000)
    alerts = policy.decide([onset], speed, 2000)
    assert alerts == ([onset] if expected else [])


def test_end_events_never_alert(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    for event_type in EVENT_TYPES:
        assert policy.decide([event(event_type, 0, PHASE_END)], None, 1000) == []


@pytest.mark.parametrize(
    ("event_type", "cooldown_ms"),
    [
        (EVENT_MICROSLEEP, 5000),
        (EVENT_GAZE_AWAY_LONG, 10000),
        (EVENT_VATS, 30000),
        (EVENT_UNABLE_TO_ANALYZE, 30000),
        (EVENT_DROWSY, 300000),
    ],
)
def test_cooldown_per_type(rules_config: RulesConfig, event_type: str, cooldown_ms: int) -> None:
    policy = AlertPolicy(rules_config)
    t0 = 100000
    assert len(policy.decide([event(event_type, t0)], None, t0)) == 1
    # 재경고 대기 안의 시작 이벤트는 경고하지 않는다.
    assert policy.decide([event(event_type, t0 + 1)], None, t0 + cooldown_ms - 1) == []
    # 대기는 실제로 낸 경고부터 센다: 정확히 cooldown_ms가 지나면 다시 경고한다.
    assert len(policy.decide([event(event_type, t0 + 2)], None, t0 + cooldown_ms)) == 1


def test_cooldown_is_per_type(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    assert len(policy.decide([event(EVENT_MICROSLEEP)], None, 1000)) == 1
    assert len(policy.decide([event(EVENT_GAZE_AWAY_LONG)], None, 1100)) == 1
    assert policy.decide([event(EVENT_MICROSLEEP)], None, 1200) == []


def test_suppressed_onset_does_not_start_cooldown(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    assert policy.decide([event(EVENT_MICROSLEEP)], 10.0, 1000) == []  # 속도로 억제
    assert len(policy.decide([event(EVENT_MICROSLEEP)], 30.0, 2000)) == 1


def test_sleep_repeats_every_3s_until_end(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    onset = event(EVENT_SLEEP, 1000, duration_ms=3000, severity=3, source=SOURCE_CLOSURE)
    alerted_at: list[int] = []
    durations: list[int] = []
    for t in range(4000, 12001, 100):
        events: list[DriverStateEvent] = []
        if t == 4000:
            events = [onset]
        if t == 11000:
            events = [event(EVENT_SLEEP, 1000, PHASE_END, duration_ms=10000, severity=3, source=SOURCE_CLOSURE)]
        for alert in policy.decide(events, None, t):
            assert (alert.type, alert.phase, alert.t_start_ms) == (EVENT_SLEEP, PHASE_ONSET, 1000)
            alerted_at.append(t)
            durations.append(alert.duration_ms)
    assert alerted_at == [4000, 7000, 10000]
    assert durations == [3000, 6000, 9000]


def test_sleep_alert_deferred_until_speed_allows(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    onset = event(EVENT_SLEEP, 1000, duration_ms=3000, severity=3)
    assert policy.decide([onset], 5.0, 4000) == []
    assert policy.decide([], 5.0, 4100) == []
    # 상태가 이어지는 동안 속도 기준을 넘는 첫 프레임에서 경고한다.
    alerts = policy.decide([], 25.0, 4200)
    assert [(a.type, a.duration_ms) for a in alerts] == [(EVENT_SLEEP, 3200)]
    assert policy.decide([], 25.0, 4300) == []
    assert len(policy.decide([], 25.0, 7200)) == 1


def test_repeat_timer_does_not_carry_over_to_the_next_state(rules_config: RulesConfig) -> None:
    # 되풀이 주기를 10초로 늘린 설정(기본 3초에서는 새 sleep이 시작되려면 3초가 걸려 차이가 드러나지 않는다).
    policy = AlertPolicy(apply_overrides(rules_config, {"events.sleep.repeat_ms": 10000}))
    first = event(EVENT_SLEEP, 1000, duration_ms=3000, severity=3, source=SOURCE_CLOSURE)
    assert policy.decide([first], 80.0, 4000) == [first]
    assert policy.decide([event(EVENT_SLEEP, 1000, PHASE_END, duration_ms=3500, severity=3)], 80.0, 4500) == []
    # 새 sleep 상태(8000)는 속도에 막혔다가 9000에서 풀린다. 앞 상태의 마지막 경고(4000)에서 10초가 안 됐지만, 상태가
    # 끝나면서 되풀이 기록이 지워졌으므로 막힘이 풀린 첫 호출에서 경고한다. 그 뒤로는 그 경고에서 10초마다다.
    second = event(EVENT_SLEEP, 5000, duration_ms=3000, severity=3, source=SOURCE_CLOSURE)
    assert policy.decide([second], 5.0, 8000) == []
    assert policy.decide([], 80.0, 9000) == [dataclasses.replace(second, duration_ms=4000)]
    assert policy.decide([], 80.0, 18999) == []
    assert policy.decide([], 80.0, 19000) == [dataclasses.replace(second, duration_ms=14000)]


def test_non_repeating_types_are_deferred_once(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    assert policy.decide([event(EVENT_GAZE_AWAY_LONG, 1000)], 5.0, 4000) == []
    assert policy.decide([], 19.9, 4100) == []
    # 막힘이 풀린 첫 호출에서 한 번 경고한다(2026-11-13 소유자 결정). duration_ms는 4200 − 1000.
    alerts = policy.decide([], 80.0, 4200)
    assert [(a.type, a.phase, a.t_start_ms, a.duration_ms) for a in alerts] == [
        (EVENT_GAZE_AWAY_LONG, PHASE_ONSET, 1000, 3200)
    ]
    # 되풀이하지 않는 종류는 상태가 아무리 이어져도 다시 경고하지 않는다.
    assert policy.decide([], 80.0, 4300) == []
    assert policy.decide([], 80.0, 60000) == []
    assert policy._pending == {}


def test_onset_that_passes_immediately_is_alerted_as_before(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    onset = event(EVENT_GAZE_AWAY_LONG, 1000, duration_ms=3000)
    alerts = policy.decide([onset], 80.0, 4000)
    # 바로 통과한 onset은 사본이 아니라 그 이벤트 그대로이고, 미뤄 두지 않는다.
    assert len(alerts) == 1 and alerts[0] is onset
    assert policy._pending == {}
    for t in range(4100, 60001, 100):
        assert policy.decide([], 80.0, t) == []


def test_deferred_alert_is_a_copy_with_the_duration_at_issue_time(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    onset = DriverStateEvent(
        type=EVENT_GAZE_AWAY_LONG,
        severity=2,
        t_start_ms=1000,
        duration_ms=3000,
        confidence=0.75,
        source=SOURCE_EYE,
        phase=PHASE_ONSET,
        provisional=True,
    )
    assert policy.decide([onset], 19.9, 4000) == []
    assert policy.decide([], 20.0, 4500) == [dataclasses.replace(onset, duration_ms=3500)]


@pytest.mark.parametrize(("lift_speed", "expected_at"), [(100.0, 71000), (50.0, 71000), (49.9, None)])
def test_drowsy_blocked_by_speed_is_alerted_once_when_the_gate_opens(
    rules_config: RulesConfig, lift_speed: float, expected_at: int | None
) -> None:
    # 결정의 확인 사례: 45 km/h에서 시작한 졸음(10초), 71초부터 속도가 오르고 상태는 한 시간 내내 이어진다.
    policy = AlertPolicy(rules_config)
    onset = event(EVENT_DROWSY, 1000, duration_ms=9000, source=SOURCE_PERCLOS)
    alerted: list[tuple[int, DriverStateEvent]] = []
    for k in range(10, 3601):
        speed = 45.0 if k <= 70 else lift_speed
        for alert in policy.decide([onset] if k == 10 else [], speed, 1000 * k):
            alerted.append((1000 * k, alert))
    if expected_at is None:
        assert alerted == []
    else:
        # 50 km/h 이상인 첫 프레임에서 정확히 한 번. 사본의 duration_ms는 71000 − 1000.
        assert alerted == [(expected_at, dataclasses.replace(onset, duration_ms=70000))]


def test_blocked_onset_is_dropped_when_the_state_ends_before_the_block_lifts(rules_config: RulesConfig) -> None:
    onset = event(EVENT_DROWSY, 1000, duration_ms=9000, source=SOURCE_PERCLOS)
    end = event(EVENT_DROWSY, 1000, PHASE_END, duration_ms=59000, source=SOURCE_PERCLOS)
    # (가) 속도가 오르기 전에 끝난다. (나) 속도가 오른 바로 그 호출에 end가 함께 온다.
    for end_k, lift_k in ((60, 71), (71, 71)):
        policy = AlertPolicy(rules_config)
        for k in range(10, 400):
            events = [onset] if k == 10 else [end] if k == end_k else []
            assert policy.decide(events, 45.0 if k < lift_k else 100.0, 1000 * k) == []
        assert policy._pending == {}


def test_onset_inside_the_cooldown_is_alerted_when_the_cooldown_expires(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    first = event(EVENT_DROWSY, 1000, duration_ms=9000)
    second = event(EVENT_DROWSY, 201000, duration_ms=9000)
    assert policy.decide([first], None, 10000) == [first]
    assert policy.decide([event(EVENT_DROWSY, 1000, PHASE_END, duration_ms=109000)], None, 110000) == []
    # 두 번째 시작은 첫 경고 뒤 200초라 재경고 대기(300초)에 걸린다. 대기는 10000 + 300000 = 310000에 끝난다.
    assert policy.decide([second], None, 210000) == []
    for t in range(211000, 309001, 1000):
        assert policy.decide([], None, t) == []
    assert policy.decide([], None, 309999) == []
    assert policy.decide([], None, 310000) == [dataclasses.replace(second, duration_ms=109000)]
    for t in range(311000, 320000, 1000):
        assert policy.decide([], None, t) == []
    # 미뤄서 낸 경고도 대기를 새로 시작한다: 310000 + 300000 = 610000.
    assert policy.decide([event(EVENT_DROWSY, 201000, PHASE_END, duration_ms=119000)], None, 320000) == []
    third = event(EVENT_DROWSY, 600000, duration_ms=9000)
    assert policy.decide([third], None, 609999) == []
    assert policy.decide([], None, 610000) == [dataclasses.replace(third, duration_ms=10000)]


@pytest.mark.parametrize(
    ("event_type", "cooldown_ms"),
    [
        (EVENT_UNABLE_TO_ANALYZE, 30000),
        (EVENT_GAZE_AWAY_LONG, 10000),
        (EVENT_VATS, 30000),
        (EVENT_DROWSY, 300000),
        (EVENT_MICROSLEEP, 5000),
    ],
)
def test_pending_onset_does_not_outlive_its_state(rules_config: RulesConfig, event_type: str, cooldown_ms: int) -> None:
    # 종류마다: 첫 경고(t0) 뒤 대기 중에 다시 시작한 상태는 미뤄 둔다(대기는 속도 기준이 없는 분석 불가도 막는다).
    t0 = 100000

    def blocked_second_onset() -> AlertPolicy:
        policy = AlertPolicy(rules_config)
        assert len(policy.decide([event(event_type, t0)], None, t0)) == 1
        assert policy.decide([event(event_type, t0, PHASE_END)], None, t0 + 100) == []
        assert policy.decide([event(event_type, t0 + 200)], None, t0 + 200) == []
        assert set(policy._pending) == {event_type}
        return policy

    # 상태가 이어지면 대기가 끝나는 첫 호출에서 경고한다(1 ms 전에는 아니다).
    policy = blocked_second_onset()
    assert policy.decide([], None, t0 + cooldown_ms - 1) == []
    alerts = policy.decide([], None, t0 + cooldown_ms)
    assert [(a.type, a.t_start_ms, a.duration_ms) for a in alerts] == [(event_type, t0 + 200, cooldown_ms - 200)]
    assert policy._pending == {}

    # 그 전에 상태가 끝나면 지워지고, 대기가 끝나도 경고하지 않는다.
    policy = blocked_second_onset()
    assert policy.decide([event(event_type, t0 + 200, PHASE_END)], None, t0 + 300) == []
    assert policy._pending == {}
    assert policy.decide([], None, t0 + cooldown_ms) == []
    assert policy.decide([], None, t0 + 2 * cooldown_ms) == []


def test_microsleep_pending_is_dropped_by_sleep_onset_and_by_sleep_end(rules_config: RulesConfig) -> None:
    microsleep = event(EVENT_MICROSLEEP, 1000, duration_ms=1000, severity=3, source=SOURCE_CLOSURE)
    sleep = event(EVENT_SLEEP, 1000, duration_ms=3000, severity=3, source=SOURCE_CLOSURE)
    sleep_end = event(EVENT_SLEEP, 1000, PHASE_END, duration_ms=5000, severity=3, source=SOURCE_CLOSURE)

    # sleep 시작이 오면 microsleep은 지운다. 속도가 올라도 sleep만 경고한다(되풀이하는 종류의 기존 동작).
    policy = AlertPolicy(rules_config)
    assert policy.decide([microsleep], 10.0, 2000) == []
    assert set(policy._pending) == {EVENT_MICROSLEEP}
    assert policy.decide([sleep], 10.0, 4000) == []
    assert policy._pending == {}
    assert [(a.type, a.duration_ms) for a in policy.decide([], 80.0, 4100)] == [(EVENT_SLEEP, 3100)]

    # 규칙 엔진은 sleep까지 간 감김에 microsleep end를 내지 않는다. 감김이 끝난 뒤에 속도가 올라도 경고가 없다.
    policy = AlertPolicy(rules_config)
    assert policy.decide([microsleep], 10.0, 2000) == []
    assert policy.decide([sleep], 10.0, 4000) == []
    assert policy.decide([sleep_end], 10.0, 6000) == []
    for t in range(6100, 20001, 100):
        assert policy.decide([], 80.0, t) == []

    # sleep end만 와도(시작 없이) 지운다.
    policy = AlertPolicy(rules_config)
    assert policy.decide([microsleep], 10.0, 2000) == []
    assert policy.decide([sleep_end], 10.0, 6000) == []
    assert policy._pending == {}
    assert policy.decide([], 80.0, 6100) == []

    # sleep이 되기 전에 막힘이 풀리면 microsleep을 한 번 경고한다.
    policy = AlertPolicy(rules_config)
    assert policy.decide([microsleep], 10.0, 2000) == []
    assert policy.decide([], 20.0, 2100) == [dataclasses.replace(microsleep, duration_ms=1100)]
    assert [a.type for a in policy.decide([sleep], 20.0, 4000)] == [EVENT_SLEEP]


def test_microsleep_blocked_then_sleep_gives_no_late_alert_with_engine(rules_config: RulesConfig) -> None:
    # 결정의 확인 사례: 10 km/h에서 막힌 microsleep이 sleep으로 이어지고, 감김이 끝난 뒤에 속도가 오른다.
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    events: Emitted = []
    for state in closure_session(4500, 12000):
        frame_events = engine.update(state)
        events += [(state.t_ms, e) for e in frame_events]
        speed = 10.0 if state.t_ms <= 4600 else 80.0
        assert policy.decide(frame_events, speed, state.t_ms) == []
    assert brief(events) == [
        (2000, EVENT_MICROSLEEP, PHASE_ONSET),
        (4000, EVENT_SLEEP, PHASE_ONSET),
        (4600, EVENT_SLEEP, PHASE_END),
    ]
    assert policy._pending == {} and policy._repeat_onset == {}


def test_calibrating_is_never_deferred(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    assert policy.decide([event(EVENT_CALIBRATING)], None, 1000) == []
    assert policy._pending == {}
    assert policy.decide([], None, 2000) == []


def test_new_onset_replaces_a_pending_onset(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    assert policy.decide([event(EVENT_GAZE_AWAY_LONG, 1000)], 5.0, 4000) == []
    newer = event(EVENT_GAZE_AWAY_LONG, 9000)
    assert policy.decide([newer], 5.0, 12000) == []
    assert policy.decide([], 80.0, 12100) == [dataclasses.replace(newer, duration_ms=3100)]
    assert policy.decide([], 80.0, 12200) == []

    # 새 onset이 바로 통과하면 그것만 경고하고, 앞서 미뤄 둔 것은 버린다(재경고 대기 10초가 지나도 나오지 않는다).
    policy = AlertPolicy(rules_config)
    assert policy.decide([event(EVENT_GAZE_AWAY_LONG, 1000)], 5.0, 4000) == []
    assert policy.decide([newer], 80.0, 12000) == [newer]
    assert policy._pending == {}
    for t in range(12100, 30001, 100):
        assert policy.decide([], 80.0, t) == []


def test_immediate_deferred_and_repeat_alerts_are_ordered_by_priority(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    blocked = [event(EVENT_DROWSY), event(EVENT_VATS), event(EVENT_GAZE_AWAY_LONG), event(EVENT_SLEEP)]
    assert policy.decide(blocked, 5.0, 1000) == []
    # 속도가 오른 호출: 새로 온 두 onset(바로), 미뤄 둔 셋, sleep(되풀이 기록)이 한꺼번에 나간다.
    alerts = policy.decide([event(EVENT_UNABLE_TO_ANALYZE, 2000), event(EVENT_MICROSLEEP, 2000)], 80.0, 2000)
    assert [(a.type, a.duration_ms) for a in alerts] == [
        (EVENT_SLEEP, 2000),
        (EVENT_MICROSLEEP, 0),
        (EVENT_GAZE_AWAY_LONG, 2000),
        (EVENT_VATS, 2000),
        (EVENT_DROWSY, 2000),
        (EVENT_UNABLE_TO_ANALYZE, 0),
    ]


def test_gaze_alert_deferred_until_speed_allows_with_engine(rules_config: RulesConfig) -> None:
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    alerts: list[tuple[int, str, int]] = []
    for t in range(0, 20001, 100):
        frame_events = engine.update(st(t, yaw=40.0 if t >= 1000 else 0.0))
        speed = 15.0 if t < 5000 else 60.0
        alerts += [(t, a.type, a.duration_ms) for a in policy.decide(frame_events, speed, t)]
    # gaze_away_long은 4000에서 시작하지만 15 km/h라 막힌다. 60 km/h가 된 5000에서 한 번 경고한다(5000 − 1000).
    # vats는 영역 밖 합이 10초가 되는 11000에서 바로 경고한다.
    assert alerts == [(5000, EVENT_GAZE_AWAY_LONG, 4000), (11000, EVENT_VATS, 10000)]


def test_alerts_are_ordered_by_priority(rules_config: RulesConfig) -> None:
    policy = AlertPolicy(rules_config)
    events = [
        event(EVENT_CALIBRATING),
        event(EVENT_UNABLE_TO_ANALYZE),
        event(EVENT_DROWSY),
        event(EVENT_VATS),
        event(EVENT_GAZE_AWAY_LONG),
        event(EVENT_MICROSLEEP),
        event(EVENT_SLEEP),
    ]
    alerts = policy.decide(events, None, 1000)
    assert [a.type for a in alerts] == [
        EVENT_SLEEP,
        EVENT_MICROSLEEP,
        EVENT_GAZE_AWAY_LONG,
        EVENT_VATS,
        EVENT_DROWSY,
        EVENT_UNABLE_TO_ANALYZE,
    ]


def test_state_events_are_emitted_even_when_alerts_are_suppressed(rules_config: RulesConfig) -> None:
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    events: Emitted = []
    alerts: Emitted = []
    for state in closure_session(4500, 6000):
        frame_events = engine.update(state)
        events += [(state.t_ms, e) for e in frame_events]
        alerts += [(state.t_ms, a) for a in policy.decide(frame_events, 10.0, state.t_ms)]
    # 10 km/h: 억제는 경고에만 적용된다(D3).
    assert brief(events) == [
        (2000, EVENT_MICROSLEEP, PHASE_ONSET),
        (4000, EVENT_SLEEP, PHASE_ONSET),
        (4600, EVENT_SLEEP, PHASE_END),
    ]
    assert alerts == []


def test_microsleep_cooldown_with_engine(rules_config: RulesConfig) -> None:
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    # 1.2초 감김을 3초 간격으로 세 번: 시작 이벤트는 2000, 5000, 8000에서 난다.
    states = [st(t, 0.0) for t in range(0, 701, 100)]
    for base in (1000, 4000, 7000):
        states += [st(t, 1.0) for t in range(base, base + 1201, 100)]
        states += [st(t, 0.0) for t in range(base + 1300, base + 2701, 100)]
    onsets: list[int] = []
    alerts: list[int] = []
    for state in states:
        frame_events = engine.update(state)
        onsets += [state.t_ms for e in frame_events if e.phase == PHASE_ONSET]
        alerts += [state.t_ms for _ in policy.decide(frame_events, None, state.t_ms)]
    assert onsets == [2000, 5000, 8000]
    # 5000은 2000의 경고 뒤 3초라 재경고 대기(5초)에 걸린다. 8000은 6초 뒤라 경고한다.
    assert alerts == [2000, 8000]


# ---------------------------------------------------------------------------
# 무작위 세션에서도 지켜지는 성질
# ---------------------------------------------------------------------------


def _random_session(rng: random.Random) -> tuple[list[FrameState], list[float | None]]:
    """구간(앞 보기, 영역 밖, 감김, 얼굴 없음, 자세 무효)을 이어 붙인 프레임 수열과 프레임별 속도.

    프레임 간격은 33~100 ms에 흔들림을 주고, 가끔 0.2~9초 끊김과 10초 넘는 끊김(초기화)을 넣는다.
    """
    t = rng.choice((0, 7, 10**9 + 3, 1_731_480_000_123))
    start = t
    duration_ms = rng.choice((60_000, 150_000, 300_000))
    provisional_ms = rng.choice((0, 0, 20_000))
    base_dt = rng.choice((33, 40, 66, 100))
    states: list[FrameState] = []
    while t - start < duration_ms:
        kind = rng.choice(("forward", "forward", "away", "away", "away_eye", "closed", "no_face", "no_pose", "mixed"))
        segment_end = t + rng.choice((150, 260, 900, 2900, 3100, 6000, 12000, 25000, 45000))
        while t < segment_end:
            provisional = t - start < provisional_ms
            pose_valid = not provisional
            if kind == "forward":
                state = st(t, rng.choice((0.0, 0.2)), provisional=provisional, pose_valid=pose_valid)
            elif kind == "away":
                state = st(t, yaw=rng.choice((30.0, -35.0)), provisional=provisional, pose_valid=pose_valid)
            elif kind == "away_eye":
                state = st(t, gaze_h=0.3, provisional=provisional, pose_valid=pose_valid)
            elif kind == "closed":
                state = st(t, rng.choice((0.75, 1.0)), provisional=provisional, pose_valid=pose_valid)
            elif kind == "no_face":
                state = st(t, face=False, provisional=provisional)
            elif kind == "no_pose":
                state = st(t, yaw=30.0, provisional=provisional, pose_valid=False)
            else:
                state = st(
                    t,
                    rng.random(),
                    valid=rng.random() < 0.85,
                    face=rng.random() < 0.85,
                    yaw=rng.choice((0.0, 0.0, 30.0)),
                    provisional=provisional,
                    pose_valid=pose_valid and rng.random() < 0.9,
                )
            states.append(state)
            roll = rng.random()
            if roll < 0.98:
                t += max(1, base_dt + rng.randrange(-5, 6))
            elif roll < 0.992:
                t += rng.randrange(201, 3200)
            elif roll < 0.998:
                t += rng.randrange(3200, 10001)
            else:
                t += rng.randrange(10001, 25000)
    speeds: list[float | None] = []
    while len(speeds) < len(states):
        speeds += [rng.choice((None, 0.0, 19.9, 20.0, 49.9, 50.0, 90.0))] * rng.randrange(1, 300)
    return states, speeds[: len(states)]


def _held_ms(states: list[FrameState], flags: list[bool], lo_ms: int, hi_ms: int, max_hold_ms: int) -> int:
    """[lo_ms, hi_ms] 안에서 flag가 참이던 시간. 프레임의 상태는 다음 프레임까지, 최대 max_hold_ms 유지된다."""
    total = 0
    for i in range(len(states) - 1):
        if not flags[i]:
            continue
        a = states[i].t_ms
        b = min(states[i + 1].t_ms, a + max_hold_ms)
        a = max(a, lo_ms)
        b = min(b, hi_ms)
        if b > a:
            total += b - a
    return total


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_random_sessions_keep_event_invariants(rules_config: RulesConfig, seed: int) -> None:
    rng = random.Random(seed)
    max_hold_ms = rules_config.clock.max_hold_ms
    counts: dict[tuple[str, str], int] = {}
    n_partial_confidence = 0
    n_vats_end_over_30s = 0
    n_alerts = 0
    n_deferred: dict[str, int] = {}  # 종류 -> 미뤘다가 낸 경고 수
    for _ in range(40):
        cfg = apply_overrides(rules_config, {"vats.reset_enabled": rng.random() < 0.3})
        states, speeds = _random_session(rng)
        face_flags = [state.face for state in states]
        pose_flags = [state.pose_valid for state in states]
        engine = RuleEngine(cfg)
        policy = AlertPolicy(cfg)
        open_events: dict[str, DriverStateEvent] = {}
        last_alert_ms: dict[str, int] = {}
        waiting: dict[str, DriverStateEvent] = {}  # 경고하지 못한 채 상태가 이어지고 있는 onset
        sleep_alert_ms: int | None = None  # 지금 sleep 상태에서 낸 마지막 경고 시각
        for index, state in enumerate(states):
            t = state.t_ms
            events = engine.update(state)
            for e in events:
                counts[(e.type, e.phase)] = counts.get((e.type, e.phase), 0) + 1
                # 공통: 길이는 발행 시각 − t_start_ms, 신뢰도는 0~1, 심각도는 설정값.
                assert e.duration_ms == t - e.t_start_ms
                assert e.duration_ms >= 0
                assert 0.0 <= e.confidence <= 1.0
                assert e.severity == cfg.events.get(e.type).severity
                assert e.provisional == state.provisional
                # 짝: 같은 종류가 열린 채로 다시 시작하지 않고, end는 열려 있는 onset과 같은 t_start_ms, source를 쓴다.
                if e.phase == PHASE_ONSET:
                    assert e.type not in open_events
                    if e.type == EVENT_SLEEP:
                        assert open_events[EVENT_MICROSLEEP].t_start_ms == e.t_start_ms
                    open_events[e.type] = e
                else:
                    onset = open_events.pop(e.type)
                    assert (onset.t_start_ms, onset.source) == (e.t_start_ms, e.source)
                    if e.type == EVENT_SLEEP:
                        del open_events[EVENT_MICROSLEEP]  # 한 감김은 도달한 가장 높은 종류의 end 하나로 끝난다
                    else:
                        assert not (e.type == EVENT_MICROSLEEP and EVENT_SLEEP in open_events)
                # 신뢰도: [t_start_ms, 발행 시각]에서 신호가 유효했던 시간의 비율을 프레임 목록에서 직접 다시 센다.
                if e.type in (EVENT_CALIBRATING, EVENT_UNABLE_TO_ANALYZE, EVENT_GAZE_AWAY_LONG, EVENT_VATS):
                    flags = face_flags if e.type in (EVENT_CALIBRATING, EVENT_UNABLE_TO_ANALYZE) else pose_flags
                    expected = 0.0
                    if e.duration_ms > 0:
                        expected = _held_ms(states, flags, e.t_start_ms, t, max_hold_ms) / e.duration_ms
                    assert e.confidence == pytest.approx(expected, abs=1e-12)
                    if 0.0 < expected < 1.0:
                        n_partial_confidence += 1
                    if e.type == EVENT_VATS and e.phase == PHASE_END and e.duration_ms > cfg.vats.window_ms:
                        n_vats_end_over_30s += 1

            # 경고: onset(바로 또는 미뤄서 한 번)과 이어지는 sleep의 되풀이만, 속도 기준과 재경고 대기를 지킨다.
            speed = speeds[index]
            alerts = policy.decide(events, speed, t)
            alerted_types = [a.type for a in alerts]
            assert len(set(alerted_types)) == len(alerted_types)
            for alert in alerts:
                rule = cfg.events.get(alert.type)
                assert alert.phase == PHASE_ONSET and alert.type != EVENT_CALIBRATING
                assert speed is None or rule.min_speed_kmh <= 0.0 or speed >= rule.min_speed_kmh
                assert alert.type not in last_alert_ms or t - last_alert_ms[alert.type] >= rule.cooldown_ms

            # 이 호출에서 나와야 할 경고를 이벤트만 보고 다시 만든다. `waiting`은 아직 경고하지 못했고 상태가 이어지고
            # 있는 onset이다(되풀이하지 않는 종류). 상태가 끝나면 지운다: 그 종류의 end, microsleep은 sleep의 onset·end도.
            def allowed(event_type: str) -> bool:
                rule = cfg.events.get(event_type)
                gate_open = speed is None or rule.min_speed_kmh <= 0.0 or speed >= rule.min_speed_kmh
                cooled = event_type not in last_alert_ms or t - last_alert_ms[event_type] >= rule.cooldown_ms
                return gate_open and cooled

            for e in events:
                if e.type == EVENT_CALIBRATING:
                    continue
                if e.type == EVENT_SLEEP:
                    waiting.pop(EVENT_MICROSLEEP, None)
                    sleep_alert_ms = None
                elif e.phase == PHASE_ONSET:
                    waiting[e.type] = e
                else:
                    waiting.pop(e.type, None)
            expected: dict[str, DriverStateEvent] = {}
            for event_type in list(waiting):
                if allowed(event_type):
                    onset = waiting.pop(event_type)
                    # 이 프레임의 onset이면 그대로이고(길이가 이미 t − t_start_ms다), 앞선 프레임의 것이면 미뤄 둔 사본이다.
                    expected[event_type] = dataclasses.replace(onset, duration_ms=t - onset.t_start_ms)
                    if onset not in events:
                        n_deferred[event_type] = n_deferred.get(event_type, 0) + 1
            if EVENT_SLEEP in open_events and allowed(EVENT_SLEEP):
                # sleep: 상태가 이어지는 동안, 그 상태의 마지막 경고 뒤 repeat_ms가 지날 때마다(첫 경고는 막힘이 풀리는 대로).
                if sleep_alert_ms is None or t - sleep_alert_ms >= cfg.events.sleep.repeat_ms:
                    onset = open_events[EVENT_SLEEP]
                    expected[EVENT_SLEEP] = dataclasses.replace(onset, duration_ms=t - onset.t_start_ms)
                    sleep_alert_ms = t
            assert alerts == [expected[name] for name in cfg.alert_priority if name in expected]
            # 미뤄 둔 경고는 상태보다 오래 남지 않는다: 정책이 들고 있는 것은 위에서 따로 적은 것과 같다.
            assert policy._pending == waiting
            for alert in alerts:
                last_alert_ms[alert.type] = t
            n_alerts += len(alerts)

    # 이 시험이 실제로 가르는 경우를 충분히 지났는지 확인한다.
    for event_type in (EVENT_UNABLE_TO_ANALYZE, EVENT_GAZE_AWAY_LONG, EVENT_VATS, EVENT_MICROSLEEP):
        assert n_deferred.get(event_type, 0) >= 1, event_type
    assert sum(n_deferred.values()) >= 20
    for event_type in (EVENT_CALIBRATING, EVENT_UNABLE_TO_ANALYZE, EVENT_GAZE_AWAY_LONG, EVENT_VATS, EVENT_MICROSLEEP):
        assert counts.get((event_type, PHASE_ONSET), 0) >= 5, event_type
        assert counts.get((event_type, PHASE_END), 0) >= 5, event_type
    assert counts.get((EVENT_SLEEP, PHASE_END), 0) >= 5
    assert n_partial_confidence >= 50
    assert n_vats_end_over_30s >= 5
    assert n_alerts >= 50


# ---------------------------------------------------------------------------
# 소유자의 결정을 기다리는 동작: 문서(docs/SPEC.md 20장)에 적은 사례가 지금 코드에서 그대로인지
# ---------------------------------------------------------------------------
#
# 아래 값은 문서에 "잰 사례"로 적혀 있다. 결정이 나서 동작이 바뀌면 여기서 드러나고, 그때 문서의 숫자도 함께 고친다.


def _events_and_alerts(
    rules_config: RulesConfig, states: list[FrameState], speed: Callable[[int], float]
) -> tuple[Emitted, Emitted]:
    engine = RuleEngine(rules_config)
    policy = AlertPolicy(rules_config)
    events: Emitted = []
    alerts: Emitted = []
    for state in states:
        frame_events = engine.update(state)
        events += [(state.t_ms, e) for e in frame_events]
        alerts += [(state.t_ms, a) for a in policy.decide(frame_events, speed(state.t_ms), state.t_ms)]
    return events, alerts


@pytest.mark.parametrize("t_resume", [7200, 7201])
def test_waiting_edge_of_the_extra_gaze_onset_condition(rules_config: RulesConfig, t_resume: int) -> None:
    # SPEC 20장 9번. 1000~3900에 영역 밖(연속 3초가 4000에 찬다), 4000에 앞을 본 프레임 하나. 그 관측은 4200까지
    # 유지되고 인정 구간은 7200에 끝난다. 다음 프레임이 얼굴 없이 t_resume에 오고 그 뒤로 앞을 본다.
    states = [st(t) for t in range(0, 1000, 100)]
    states += [st(t, yaw=40.0) for t in range(1000, 4000, 100)]
    states += [st(4000), st(t_resume, face=False)]
    states += [st(t) for t in range(t_resume + 100, t_resume + 700, 100)]
    events, alerts = _events_and_alerts(rules_config, states, lambda t: 80.0)
    if t_resume == 7200:
        # 정확히 인정 구간의 끝: 표지는 UNKNOWN인데 영역 안 후보가 아직 걸려 있어 판정하지 않는다. 다음 프레임의
        # 표지가 IN이라 그대로 넘어간다. 이벤트도 경고도 없다.
        assert events == [] and alerts == []
    else:
        # 1 ms 뒤: 후보를 버리고 이미 찬 3000 ms로 그 프레임에서 시작한다. 7301의 정면 후보가 7601에 확정되어 끝난다.
        assert brief(events) == [(7201, EVENT_GAZE_AWAY_LONG, PHASE_ONSET), (7601, EVENT_GAZE_AWAY_LONG, PHASE_END)]
        assert [(e.t_start_ms, e.duration_ms) for _, e in events] == [(1000, 6201), (1000, 6601)]
        assert brief(alerts) == [(7201, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]


def test_waiting_case_of_a_deferred_vats_alert_long_after_the_driver_looked_forward(rules_config: RulesConfig) -> None:
    # SPEC 20장 8번 (e). 10 km/h에서 1.0~12.0초 동안 옆을 보고 12.0초부터 앞을 본다. 25.0초부터 30 km/h.
    states = [st(t, yaw=40.0 if 1000 <= t < 12000 else 0.0) for t in range(0, 40001, 100)]
    events, alerts = _events_and_alerts(rules_config, states, lambda t: 10.0 if t < 25000 else 30.0)
    assert brief(events) == [
        (4000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (11000, EVENT_VATS, PHASE_ONSET),
        (12300, EVENT_GAZE_AWAY_LONG, PHASE_END),
        (32100, EVENT_VATS, PHASE_END),  # 영역 밖 누적 [1000, 12000)이 30초 창에서 1000 ms 넘게 빠진 프레임
    ]
    # gaze_away_long 경고는 그 상태가 끝나면서 지워졌다. vats는 32.1초까지 이어지므로, 속도가 오른 25.0초에
    # 미뤄 둔 경고가 나간다: 운전자가 앞을 본 지 13초 뒤다.
    assert [(t, a.type, a.t_start_ms, a.duration_ms) for t, a in alerts] == [(25000, EVENT_VATS, 1000, 24000)]


@pytest.mark.parametrize(
    ("clock", "microsleep_at", "sleep_at", "sleep_blink", "n_sleep_alerts", "last_alert"),
    [
        ("33ms", 24123, 72006, 18, 36, 177111),  # 프레임 시각이 33 ms 간격
        ("30fps", 24100, 68133, 17, 38, 179133),  # 정확한 30 fps의 정수 ms 타임스탬프
    ],
)
def test_decided_case_of_lids_hovering_between_the_two_thresholds(
    rules_config: RulesConfig,
    clock: str,
    microsleep_at: int,
    sleep_at: int,
    sleep_blink: int,
    n_sleep_alerts: int,
    last_alert: int,
) -> None:
    # 2026-11-14 소유자 결정("지금대로 둠")으로 확정된 동작과 그 알려진 한계. SPEC 20장 머리의 결정 기록에 이 값들이 있다.
    # 눈꺼풀이 0.6(이탈 0.5와 진입 0.7 사이)에 머물고 4초마다 보통의 깜빡임(0.9, 165 ms)을 한다.
    # 처음 2초는 눈을 뜨고 있고(0.1) 첫 깜빡임은 4초에 온다. 3분, 80 km/h.
    if clock == "33ms":
        times = list(range(0, 180000, 33))
    else:
        times = [(i * 1000 + 15) // 30 for i in range(30 * 180)]
    states = []
    blink_number = {}
    blinks = 0
    was_closed = False
    for t in times:
        closed = t >= 2000 and t % 4000 < 165
        if closed and not was_closed:
            blinks += 1
        was_closed = closed
        blink_number[t] = blinks
        states.append(st(t, 0.1 if t < 2000 else (0.9 if closed else 0.6)))
    events, alerts = _events_and_alerts(rules_config, states, lambda t: 80.0)
    # 감김이 한 번도 끝나지 않는다(0.5 아래로 내려가지 않는다): 깜빡임들의 감긴 시간이 한 감김에 계속 쌓인다.
    assert brief(events) == [(microsleep_at, EVENT_MICROSLEEP, PHASE_ONSET), (sleep_at, EVENT_SLEEP, PHASE_ONSET)]
    assert blink_number[microsleep_at] == 6 and blink_number[sleep_at] == sleep_blink
    assert events[0][1].t_start_ms == events[1][1].t_start_ms  # 첫 깜빡임에서 시작한 감김 하나다
    sleep_alerts = [t for t, a in alerts if a.type == EVENT_SLEEP]
    # sleep이 시작된 뒤에는 눈꺼풀이 머무는 동안 3초마다 경고가 되풀이된다.
    assert (len(sleep_alerts), sleep_alerts[0], sleep_alerts[-1]) == (n_sleep_alerts, sleep_at, last_alert)
    assert [t for t, a in alerts if a.type == EVENT_MICROSLEEP] == [microsleep_at]
