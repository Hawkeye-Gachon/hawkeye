"""`drowsy.pipeline.DriverPipeline` 테스트 (PLAN 3.3절, 4.4절, 9장).

합성 `RawFrame`만 쓴다(실제 얼굴 이미지는 없다). 프레임은 100 ms 간격이다.
합성 얼굴: 640x480, 눈 폭 40 px, 뜬 눈은 눈꺼풀 간격 12 px(EAR 0.30)·eyeBlink 0.1,
감은 눈은 2 px(EAR 0.05)·eyeBlink 0.7이다. 이 얼굴로 60초 보정하면 E = 0.30, B = 0.1이 되어
뜬 눈의 closure는 0, 감은 눈의 closure는 1이다(전역 상수 E = 0.27, B = 0.22로 계산해도 0과 1이다).
반쯤 감은 눈은 `level`(0~1)로 준다: EAR = 0.30 − 0.195·level, eyeBlink = 0.1 + 0.5·level이라
E = 0.30, B = 0.1 기준선에서 a_e = s_e = level, 곧 closure = level이다(rho 0.35, b_closed 0.6, w 0.5).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

import pytest

from drowsy import pipeline as pipeline_module
from drowsy import signals
from drowsy.pipeline import DriverPipeline
from drowsy.spec import PipelineSpec, RulesConfig, apply_overrides
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OK,
    EVENT_CALIBRATING,
    EVENT_DROWSY,
    EVENT_GAZE_AWAY_LONG,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
    F_BLINK_DUR_MEAN_60,
    F_CLOSURE_MEAN_10,
    F_FACE_MISSING_RATIO_10,
    F_MAX_CLOSURE_MS_60,
    F_PERCLOS_60,
    IDENTITY_R0,
    N_FEATURES,
    PHASE_END,
    PHASE_ONSET,
    SOURCE_MODEL,
    SOURCE_PERCLOS,
    Baseline,
    PipelineOutput,
    RawFrame,
    StepFeatures,
)

W = 640
H = 480

# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def colmajor(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> tuple[float, ...]:
    """R = Ry·Rx·Rz를 열 우선 4x4(16개 값)로 적는다. M[i][j] = m[4·j + i]."""
    r = signals.rotation_from_euler(yaw, pitch, roll)
    m = [0.0] * 16
    for i in range(3):
        for j in range(3):
            m[4 * j + i] = r[3 * i + j]
    m[15] = 1.0
    return tuple(m)


def raw(
    spec: PipelineSpec,
    t_ms: int,
    *,
    closed: bool = False,
    face: bool = True,
    yaw: float = 0.0,
    level: float | None = None,
) -> RawFrame:
    """합성 프레임 하나. `level`을 주면 `closed` 대신 그만큼 감은 눈을 만든다(모듈 설명)."""
    if not face:
        return RawFrame(t_ms=t_ms, face=False, w=W, h=H, lm={}, bs={}, m=())
    lm = {index: (0.5, 0.5) for index in spec.landmarks.required_indices()}
    opening = 2.0 if closed else 12.0
    blink = 0.7 if closed else 0.1
    if level is not None:
        opening = 40.0 * (0.30 - 0.195 * level)
        blink = 0.1 + 0.5 * level
    for ear, cx in ((spec.landmarks.ear_r, 250.0), (spec.landmarks.ear_l, 390.0)):
        cy = 200.0
        points = [
            (cx - 20.0, cy),
            (cx - 7.0, cy - opening / 2.0),
            (cx + 7.0, cy - opening / 2.0),
            (cx + 20.0, cy),
            (cx + 7.0, cy + opening / 2.0),
            (cx - 7.0, cy + opening / 2.0),
        ]
        for index, (x, y) in zip(ear, points):
            lm[index] = (x / W, y / H)
    bs = {name: 0.0 for name in spec.blendshapes.required_names()}
    bs[spec.blendshapes.eye_blink_l] = blink
    bs[spec.blendshapes.eye_blink_r] = blink
    return RawFrame(t_ms=t_ms, face=True, w=W, h=H, lm=lm, bs=bs, m=colmajor(yaw))


def synthetic_baseline(t_complete_ms: int = 0) -> Baseline:
    """합성 얼굴의 기준선(보정으로 얻는 값과 같다)."""
    return Baseline(
        e_l=0.3,
        e_r=0.3,
        b_l=0.1,
        b_r=0.1,
        r0=IDENTITY_R0,
        jaw0=0.0,
        gaze_h0=0.0,
        gaze_v0=0.0,
        t_complete_ms=t_complete_ms,
        status=BASELINE_OK,
    )


class StubModel:
    """받은 창을 기록하고 정해진 확률을 돌려준다."""

    def __init__(self, probs: Sequence[float] = (0.2, 0.3, 0.5)) -> None:
        self.probs = probs
        self.windows: list[list[tuple[float, ...]]] = []

    def __call__(self, window: Sequence[Sequence[float]]) -> Sequence[float]:
        self.windows.append([tuple(row) for row in window])
        return self.probs


def run(
    pipe: DriverPipeline,
    spec: PipelineSpec,
    session_times: Sequence[int],
    origin_ms: int = 1,
    closed: Callable[[int], bool] | None = None,
    face: Callable[[int], bool] | None = None,
    yaw: Callable[[int], float] | None = None,
    speed_kmh: float | None = None,
    level: Callable[[int], float | None] | None = None,
) -> list[tuple[int, PipelineOutput]]:
    """세션 시간 목록대로 프레임을 넣고 (세션 시간, 출력) 목록을 돌려준다.

    `level(s)`가 None이 아닌 값을 주는 프레임은 그만큼 감은 눈이고, 나머지는 `closed(s)`를 따른다.
    """
    outputs: list[tuple[int, PipelineOutput]] = []
    for s in session_times:
        frame = raw(
            spec,
            origin_ms + s,
            closed=closed(s) if closed is not None else False,
            face=face(s) if face is not None else True,
            yaw=yaw(s) if yaw is not None else 0.0,
            level=level(s) if level is not None else None,
        )
        out = pipe.update(frame, speed_kmh)
        assert out is not None
        outputs.append((s, out))
    return outputs


def all_steps(outputs: list[tuple[int, PipelineOutput]]) -> list[StepFeatures]:
    return [step for _, out in outputs for step in out.steps]


def all_events(outputs: list[tuple[int, PipelineOutput]]) -> list[tuple[int, str, str]]:
    return [(s, e.type, e.phase) for s, out in outputs for e in out.events]


def all_alerts(outputs: list[tuple[int, PipelineOutput]]) -> list[tuple[int, str, str]]:
    return [(s, e.type, e.phase) for s, out in outputs for e in out.alerts]


def varied_blinks(s: int) -> bool:
    """7초마다 감는다. 길이는 100~500 ms로 돌아가며 바뀐다(스텝마다 특징이 달라지게 하려는 것)."""
    return (s // 1000) % 7 == 0 and s % 1000 < 100 * (1 + (s // 7000) % 5)


# ---------------------------------------------------------------------------
# 합성 얼굴이 설명대로인지
# ---------------------------------------------------------------------------


def test_synthetic_face_signals(pipeline_spec: PipelineSpec) -> None:
    opened = signals.compute(raw(pipeline_spec, 1), pipeline_spec)
    closed = signals.compute(raw(pipeline_spec, 1, closed=True), pipeline_spec)
    assert opened.ear_l == pytest.approx(0.3)
    assert opened.ear_r == pytest.approx(0.3)
    assert closed.ear_l == pytest.approx(0.05)
    assert opened.eye_w_l == pytest.approx(40.0)
    assert (opened.blink_l, closed.blink_r) == (0.1, 0.7)
    turned = signals.compute(raw(pipeline_spec, 1, yaw=30.0), pipeline_spec)
    assert (turned.yaw, turned.pitch, turned.roll) == pytest.approx((30.0, 0.0, 0.0), abs=1e-9)
    # level 0.4: EAR = 0.30 − 0.195·0.4 = 0.222, eyeBlink = 0.1 + 0.5·0.4 = 0.3
    half = signals.compute(raw(pipeline_spec, 1, level=0.4), pipeline_spec)
    assert half.ear_l == pytest.approx(0.222)
    assert half.ear_r == pytest.approx(0.222)
    assert (half.blink_l, half.blink_r) == pytest.approx((0.3, 0.3))


def test_level_frames_give_that_closure(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    for index, level in enumerate((0.0, 0.2, 0.5, 0.794, 1.0)):
        out = pipe.update(raw(pipeline_spec, 1 + 100 * index, level=level))
        assert out is not None
        assert out.state.closure_valid is True
        assert out.state.closure == pytest.approx(level, abs=1e-12)


# ---------------------------------------------------------------------------
# 4.4절의 시간표
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def timeline(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> tuple[DriverPipeline, StubModel, list]:
    """245초 세션(T_b = 60초). 첫 t_ms는 1이다."""
    model = StubModel()
    pipe = DriverPipeline(pipeline_spec, rules_config, model=model)
    outputs = run(pipe, pipeline_spec, range(0, 245001, 100), origin_ms=1, closed=varied_blinks)
    return pipe, model, outputs


def test_timeline_calibration_phase(timeline: tuple[DriverPipeline, StubModel, list]) -> None:
    pipe, _, outputs = timeline
    by_time = dict(outputs)
    # 0 ~ T_b: 기준선 수집. provisional 상태이고 스텝은 내지 않는다.
    for s in (0, 30000, 59900):
        assert by_time[s].state.provisional is True
        assert by_time[s].state.pose_valid is False
        assert by_time[s].steps == ()
    assert by_time[60000].state.provisional is False
    assert by_time[60000].state.pose_valid is True
    events = all_events(outputs)
    assert events[0] == (0, EVENT_CALIBRATING, PHASE_ONSET)
    assert events[1] == (60000, EVENT_CALIBRATING, PHASE_END)
    calibrating_end = by_time[60000].events[0]
    assert (calibrating_end.t_start_ms, calibrating_end.duration_ms) == (1, 60000)
    baseline = pipe.baseline
    assert baseline is not None
    assert baseline.t_complete_ms == 60000
    assert baseline.status == BASELINE_OK
    assert (baseline.e_l, baseline.b_l) == (pytest.approx(0.3), 0.1)
    diagnostics = pipe.baseline_diagnostics
    assert diagnostics is not None
    assert diagnostics.n_face_frames == 600


def test_timeline_first_step_after_t_b(timeline: tuple[DriverPipeline, StubModel, list]) -> None:
    _, _, outputs = timeline
    first_s, first_out = next((s, out) for s, out in outputs if out.steps)
    # T_b 이후에 시작하는 첫 스텝은 k = 61([60000, 61000))이고 세션 시간 61000의 프레임이 발행한다.
    assert first_s == 61000
    assert [step.k for step in first_out.steps] == [61]
    assert first_out.steps[0].t_end_ms == 1 + 61000
    assert first_out.steps[0].full_lookback is False
    assert len(first_out.steps[0].values) == N_FEATURES
    steps = all_steps(outputs)
    assert [step.k for step in steps] == list(range(61, 246))


def test_timeline_first_full_lookback_step(timeline: tuple[DriverPipeline, StubModel, list]) -> None:
    _, _, outputs = timeline
    steps = all_steps(outputs)
    full = [step.k for step in steps if step.full_lookback]
    # 스텝 끝 ≥ T_b + 60초인 첫 스텝은 k = 120이다.
    assert full[0] == 120
    assert full == list(range(120, 246))


def test_timeline_first_model_call_at_239s(timeline: tuple[DriverPipeline, StubModel, list]) -> None:
    pipe, model, outputs = timeline
    with_probs = [(s, out) for s, out in outputs if out.probs is not None]
    # 다 찬 스텝이 연속 120개(k = 120~239)가 되는 세션 시간 239초에 첫 확률이 나온다.
    first_s, first_out = with_probs[0]
    assert first_s == 239000
    assert first_out.steps[-1].k == 239
    assert first_out.probs == (0.2, 0.3, 0.5)
    # 그 뒤로는 스텝마다(매초) 한 번씩 부른다.
    assert [s for s, _ in with_probs] == [239000, 240000, 241000, 242000, 243000, 244000, 245000]
    assert pipe.model_calls == 7
    assert pipe.steps_emitted == 185
    assert pipe.frames_accepted == 2451
    assert pipe.dropped_frames == 0
    assert pipe.nonfinite_windows == 0


def test_model_window_is_120_by_17_oldest_first(timeline: tuple[DriverPipeline, StubModel, list]) -> None:
    _, model, outputs = timeline
    steps = {step.k: step for step in all_steps(outputs)}
    assert len(model.windows) == 7
    first = model.windows[0]
    assert len(first) == 120
    assert all(len(row) == N_FEATURES for row in first)
    # features[0]이 가장 오래된 스텝(k = 120), features[119]가 가장 최근 스텝(k = 239)이다.
    assert first == [steps[k].values for k in range(120, 240)]
    assert model.windows[-1] == [steps[k].values for k in range(126, 246)]
    # 스텝마다 값이 달라서 순서가 뒤집히면 위 비교가 실패한다.
    assert len({row[F_CLOSURE_MEAN_10] for row in first}) > 5
    assert first != first[::-1]


def test_step_probs_align_with_steps_on_single_step_frames(timeline: tuple[DriverPipeline, StubModel, list]) -> None:
    _, _, outputs = timeline
    # 프레임이 끊기지 않은 세션: 프레임마다 스텝은 0개 아니면 1개다. step_probs는 steps와 길이가 같고,
    # probs는 스텝이 있으면 그 스텝의 확률(창이 차기 전에는 None), 없으면 None이다(결정 전의 probs와 같은 값).
    assert {len(out.steps) for _, out in outputs} == {0, 1}
    for s, out in outputs:
        assert len(out.step_probs) == len(out.steps)
        if out.steps == ():
            assert out.step_probs == () and out.probs is None
        elif s < 239000:
            assert out.step_probs == (None,) and out.probs is None
        else:
            assert out.step_probs == ((0.2, 0.3, 0.5),) and out.probs == (0.2, 0.3, 0.5)


def test_step_probs_follow_the_step_order_when_a_frame_emits_several_steps(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    calls: list[int] = []

    def model(window: Sequence[Sequence[float]]) -> Sequence[float]:
        # 부를 때마다 다른 값을 낸다: n번째 호출의 p_drowsy는 0.5 + 0.01n.
        calls.append(len(window))
        return (0.25, 0.25 - 0.01 * len(calls), 0.5 + 0.01 * len(calls))

    def nth(n: int) -> tuple[float, float, float]:
        return (0.25, 0.25 - 0.01 * n, 0.5 + 0.01 * n)

    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0, model=model)
    run(pipe, pipeline_spec, range(0, 177001, 100), closed=varied_blinks)
    assert pipe.model_calls == 0
    # 2초 뒤의 프레임이 스텝 178, 179를 낸다. 창(스텝 60~179)은 179에서 처음 찬다: 178에는 확률이 없고 179에 있다.
    out = pipe.update(raw(pipeline_spec, 1 + 179000))
    assert out is not None
    assert [step.k for step in out.steps] == [178, 179]
    assert out.step_probs == (None, nth(1))
    assert out.probs == nth(1)
    # 2.1초 뒤의 프레임이 스텝 180, 181을 낸다: 호출 순서대로 두 번째, 세 번째 값이고 probs는 마지막 것이다.
    out = pipe.update(raw(pipeline_spec, 1 + 181100))
    assert out is not None
    assert [step.k for step in out.steps] == [180, 181]
    assert out.step_probs == (nth(2), nth(3))
    assert out.probs == nth(3)
    assert pipe.model_calls == 3 and calls == [120, 120, 120]


def test_no_model_means_no_probs(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    outputs = run(pipe, pipeline_spec, range(0, 181001, 100))
    assert all(out.probs is None for _, out in outputs)
    assert all(out.step_probs == (None,) * len(out.steps) for _, out in outputs)
    assert pipe.model_calls == 0
    assert pipe.steps_emitted == 181


# ---------------------------------------------------------------------------
# 기준선 연장과 대체가 시간표를 민다 (D5)
# ---------------------------------------------------------------------------


def test_extended_baseline_shifts_the_timeline(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config)
    # 처음 40초 동안 얼굴이 없다: 60초 시점의 얼굴 프레임 200개(< 300) -> 90초로 연장된다.
    outputs = run(pipe, pipeline_spec, range(0, 92001, 100), face=lambda s: s >= 40000)
    baseline = pipe.baseline
    assert baseline is not None
    assert (baseline.t_complete_ms, baseline.status) == (90000, BASELINE_OK)
    assert [step.k for step in all_steps(outputs)] == [91, 92]
    by_time = dict(outputs)
    assert by_time[89900].state.provisional is True
    assert by_time[90000].state.provisional is False
    assert (90000, EVENT_CALIBRATING, PHASE_END) in all_events(outputs)


def test_accumulation_starts_at_t_b_even_when_the_completing_frame_is_late(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    # 프레임이 59900 다음에 61200에 온다. 기준선은 그 프레임에서 완성되고 T_b는 60000이다. 누적은 T_b부터이므로
    # 그 프레임이 스텝 61([60000, 61000))을 빈 스텝으로 발행한다(누적을 그 프레임의 스텝부터 시작하면 61이 빠진다).
    pipe = DriverPipeline(pipeline_spec, rules_config)
    times = list(range(0, 60000, 100)) + list(range(61200, 181001, 100))
    outputs = run(pipe, pipeline_spec, times)
    assert pipe.baseline is not None and pipe.baseline.t_complete_ms == 60000
    by_time = dict(outputs)
    assert by_time[59900].state.provisional is True and by_time[61200].state.provisional is False
    steps = all_steps(outputs)
    assert [step.k for step in by_time[61200].steps] == [61]
    assert [step.k for step in steps] == list(range(61, 182))
    first, second = steps[0].values, steps[1].values
    # 스텝 61에는 프레임이 없다(59900의 프레임은 누적 시작 전이다): 얼굴 시간 0.
    assert first[F_FACE_MISSING_RATIO_10] == 1.0 and first[F_CLOSURE_MEAN_10] == 0.0
    # 스텝 62: 61200~61900의 프레임 여덟이 100 ms씩 = 800 ms
    assert second[F_FACE_MISSING_RATIO_10] == pytest.approx((10000.0 - 800.0) / 10000.0)
    # 조회 구간은 T_b + 60초에 찬다: 스텝 120부터(61000에서 시작했다면 121부터다).
    assert [step.k for step in steps if step.full_lookback][0] == 120


def test_fallback_baseline_after_180s(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config)
    outputs = run(pipe, pipeline_spec, range(0, 183001, 100), face=lambda s: s >= 170000)
    baseline = pipe.baseline
    assert baseline is not None
    assert (baseline.t_complete_ms, baseline.status) == (180000, BASELINE_FALLBACK_GLOBAL)
    by_time = dict(outputs)
    # 대체된 뒤에는 provisional이 아니지만 시선 규칙용 자세는 계속 무효다.
    assert by_time[180000].state.provisional is False
    assert by_time[180000].state.face is True
    assert by_time[180000].state.pose_valid is False
    assert [step.k for step in all_steps(outputs)] == [181, 182, 183]
    assert (180000, EVENT_CALIBRATING, PHASE_END) in all_events(outputs)


# ---------------------------------------------------------------------------
# 버리는 프레임
# ---------------------------------------------------------------------------


def test_non_increasing_timestamps_are_dropped_and_counted(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    clean = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    times = [1, 101, 201, 201, 150, 1, 301, 300, 401, 1001, 1001, 2001]
    accepted = [1, 101, 201, 301, 401, 1001, 2001]
    results = [pipe.update(raw(pipeline_spec, t, closed=t == 201)) for t in times]
    dropped = [r is None for r in results]
    assert dropped == [False, False, False, True, True, True, False, True, False, False, True, False]
    assert pipe.dropped_frames == 5
    assert pipe.frames_accepted == 7
    # 버린 프레임은 상태에 아무 영향도 주지 않는다.
    expected = [clean.update(raw(pipeline_spec, t, closed=t == 201)) for t in accepted]
    assert [r for r in results if r is not None] == expected
    assert clean.dropped_frames == 0


def test_dropped_frames_leave_runs_and_steps_as_hand_computed(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    # (t_ms, 감았는지). 원점은 1이다. 받아들이는 프레임은 세션 시간 0~2000의 100 ms 간격이고 300, 400, 500에서만 감는다.
    # 사이사이에 중복되거나 되돌아간 프레임을 반대 눈 상태로 끼워 넣는다. 하나라도 처리되면 아래 값이 달라진다.
    stream: list[tuple[int, bool]] = []
    for s in range(0, 2001, 100):
        closed = 300 <= s <= 500
        stream.append((1 + s, closed))
        if s in (200, 400, 500, 1000):
            stream.append((1 + s, not closed))  # 같은 t_ms를 한 번 더
        if s in (300, 600):
            stream.append((1 + s - 50, not closed))  # 되돌아간 t_ms
    results = [pipe.update(raw(pipeline_spec, t, closed=closed)) for t, closed in stream]
    assert pipe.frames_accepted == 21
    assert pipe.dropped_frames == 6
    assert sum(1 for r in results if r is None) == 6

    outputs = [r for r in results if r is not None]
    emitted = [(out.state.t_ms, run) for out in outputs for run in out.runs]
    assert [t for t, _ in emitted] == [601]  # 눈을 뜬 첫 프레임(세션 시간 600)에서 발행한다
    run = emitted[0][1]
    # start: 200(0)과 300(1) 사이에서 0.5 -> 250, end: 500(1)과 600(0) 사이에서 0.5 -> 550. 원점 1을 더한다.
    assert (run.start_ms, run.end_ms, run.duration_ms, run.truncated) == (251, 551, 300, False)
    assert run.amplitude == 1.0
    # 두 속도 모두 1.0 / 0.1초 = 10/s
    assert run.peak_closing_velocity == pytest.approx(10.0)
    assert run.peak_opening_velocity == pytest.approx(10.0)
    assert run.velocity_valid is True

    steps = [step for out in outputs for step in out.steps]
    assert [step.k for step in steps] == [1, 2]
    first = steps[0].values
    # 스텝 1([0, 1000)): 감긴 프레임 셋이 100 ms씩 -> closure 평균 300/1000, 가장 긴 감김과 깜빡임 길이 300 ms.
    assert first[F_CLOSURE_MEAN_10] == pytest.approx(0.3)
    assert first[F_MAX_CLOSURE_MS_60] == 300.0
    assert first[F_BLINK_DUR_MEAN_60] == 300.0
    assert first[F_FACE_MISSING_RATIO_10] == pytest.approx(0.9)  # 10초 창 중 얼굴을 본 시간은 1초


# ---------------------------------------------------------------------------
# 기준선을 넣어 준 경우
# ---------------------------------------------------------------------------


def test_injected_baseline_skips_calibration(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    baseline = synthetic_baseline()
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=baseline, feature_start_ms=5000)
    outputs = run(pipe, pipeline_spec, range(0, 8001, 100), origin_ms=777)
    assert pipe.baseline is baseline
    assert pipe.baseline_diagnostics is None
    first = outputs[0][1]
    assert first.state.provisional is False
    assert first.state.pose_valid is True
    assert all_events(outputs) == []  # calibrating 이벤트가 없다
    # 누적은 feature_start_ms부터: 첫 스텝은 k = 6([5000, 6000))이다.
    steps = all_steps(outputs)
    assert [step.k for step in steps] == [6, 7, 8]
    assert steps[0].t_end_ms == 777 + 6000
    assert next(s for s, out in outputs if out.steps) == 6000


def test_feature_start_defaults_to_baseline_completion_time(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(t_complete_ms=3000))
    outputs = run(pipe, pipeline_spec, range(0, 5001, 100))
    assert [step.k for step in all_steps(outputs)] == [4, 5]


def test_frames_before_the_feature_start_do_not_reach_the_feature_detector(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    # 누적 시작은 세션 시간 5000이다. 세션 시간은 t_ms − 원점이므로 원점이 커도 같다(원점 10000000).
    # 눈을 감는 구간: 3000~3300(누적 시작 전에 끝난다)과 4800~5300(누적 시작에 걸친다).
    origin = 10_000_000
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=5000)
    outputs = run(
        pipe,
        pipeline_spec,
        range(0, 7001, 100),
        origin_ms=origin,
        closed=lambda s: 3000 <= s <= 3300 or 4800 <= s <= 5300,
    )
    emitted = [(s, r) for s, out in outputs for r in out.runs]
    # 첫 감김은 특징용 검출기에 들어가지 않았으므로 구간이 없다. 둘째 감김은 검출기가 처음 받은 프레임(5000)에서
    # 시작한 것으로 잰다(직전 표본이 없어 보간하지 않는다). 끝은 5300(1.0) -> 5400(0.0)의 한가운데 5350이다.
    assert [s for s, _ in emitted] == [5400]
    run_ = emitted[0][1]
    assert (run_.start_ms, run_.end_ms, run_.duration_ms, run_.truncated) == (origin + 5000, origin + 5350, 350, False)
    assert run_.velocity_valid is False  # 감는 쪽의 쌍을 보지 못했다
    steps = all_steps(outputs)
    assert [step.k for step in steps] == [6, 7]
    assert steps[0].values[F_MAX_CLOSURE_MS_60] == 350.0  # 스텝 6([5000, 6000))에 그 구간 하나
    assert steps[0].values[F_BLINK_DUR_MEAN_60] == 350.0


def test_constructor_argument_checks(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    with pytest.raises(ValueError):
        DriverPipeline(pipeline_spec, rules_config, feature_start_ms=60000)  # 기준선 없이 시작 시각만
    with pytest.raises(ValueError):
        DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=60500)
    with pytest.raises(ValueError):
        DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=-1000)
    other_clock = apply_overrides(rules_config, {"clock.max_hold_ms": 300})
    with pytest.raises(ValueError):
        DriverPipeline(pipeline_spec, other_clock)


def test_injected_baseline_reproduces_session_baseline_steps(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    times = range(0, 130001, 100)

    def closed(s: int) -> bool:
        # 보정 구간에도, T_b를 걸쳐서도 감는다(59900~60200).
        return varied_blinks(s) or 59900 <= s < 60200

    own = DriverPipeline(pipeline_spec, rules_config)
    own_outputs = run(own, pipeline_spec, times, closed=closed)
    baseline = own.baseline
    assert baseline is not None

    injected = DriverPipeline(pipeline_spec, rules_config, baseline=baseline, feature_start_ms=baseline.t_complete_ms)
    injected_outputs = run(injected, pipeline_spec, times, closed=closed)
    # 같은 기준선, 같은 시작 시각이면 스텝과 감김 구간이 비트 단위로 같다(세 정규화 조건이 같은 창 집합을 쓴다).
    assert all_steps(injected_outputs) == all_steps(own_outputs)
    own_runs = [r for _, out in own_outputs for r in out.runs]
    injected_runs = [r for _, out in injected_outputs for r in out.runs]
    assert injected_runs == own_runs
    assert len(own_runs) > 5
    # 누적 시작 전의 프레임은 특징용 검출기에 넣지 않는다: 59900에서 시작한 감김은 60000부터 본 것으로 잰다.
    assert own_runs[0].start_ms == 1 + 60000


# ---------------------------------------------------------------------------
# 첫 t_ms가 큰 임의 값인 경우
# ---------------------------------------------------------------------------


def test_large_first_timestamp_gives_the_same_session(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    big = 1_723_456_789_012  # 에포크 ms 수준. 이 크기에서 배정밀도 간격은 2^-12 ms(0.00024)다.
    times = list(range(0, 125001, 100))

    def closed(s: int) -> bool:
        return varied_blinks(s) or 70000 <= s < 71500  # 1.5초 감김(microsleep)

    # 반쯤 감은 프레임을 넣어 보간 비율이 0.5가 아니게 한다.
    # 아래 두 쌍은 임계값에 닿는 오프셋이 반올림 경계 바로 아래(x.49995 ms)다. 반올림하면 x다. 원점을 먼저 더하고
    # 반올림하면 큰 원점에서는 x.5로 읽혀 x + 1이 되므로, 보간이 원점에 의존하면 아래 비교가 깨진다.
    #   69900(0.2) -> 70000: 규칙용 진입 0.7에 닿는 오프셋 = 0.5/(c − 0.2)·100 = 70.49995
    #   100000(0.2) -> 100100: 특징용 진입 0.5에 닿는 오프셋 = 0.3/(c − 0.2)·100 = 50.49995
    crafted = {
        69900: 0.2,
        70000: 0.2 + 0.5 * 100.0 / 70.49995,
        100000: 0.2,
        100100: 0.2 + 0.3 * 100.0 / 50.49995,
        100200: 1.0,
        100300: 1.0,
    }
    assert 0.90 < crafted[70000] < 0.91 and 0.79 < crafted[100100] < 0.80

    def level(s: int) -> float | None:
        if s in crafted:
            return crafted[s]
        if closed(s):
            return None
        if closed(s + 100):
            return 0.37  # 감기 직전 프레임
        if closed(s - 100):
            return 0.23  # 뜬 직후 프레임
        return None

    def yaw(s: int) -> float:
        return 30.0 if 80000 <= s < 84000 else 0.0  # 4초 시선 이탈

    small_pipe = DriverPipeline(pipeline_spec, rules_config)
    big_pipe = DriverPipeline(pipeline_spec, rules_config)
    small = run(small_pipe, pipeline_spec, times, origin_ms=1, closed=closed, yaw=yaw, level=level)
    large = run(big_pipe, pipeline_spec, times, origin_ms=big, closed=closed, yaw=yaw, level=level)

    small_steps = all_steps(small)
    large_steps = all_steps(large)
    assert len(small_steps) == len(large_steps) == 65
    for a, b in zip(small_steps, large_steps):
        assert (a.k, a.full_lookback, a.values) == (b.k, b.full_lookback, b.values)
        assert a.t_end_ms - 1 == b.t_end_ms - big == 1000 * a.k

    # 특징용 감김 구간: 원점을 빼면 같다.
    small_runs = [(r.start_ms - 1, r.end_ms - 1, r.duration_ms) for _, out in small for r in out.runs]
    large_runs = [(r.start_ms - big, r.end_ms - big, r.duration_ms) for _, out in large for r in out.runs]
    assert small_runs == large_runs
    # 손 계산. 63000~63400의 깜빡임: start = 62900 + (0.13/0.63)·100 = +20.6 -> 62921,
    # end = 63400 + (0.5/0.77)·100 = +64.9 -> 63465. 넣어 둔 깜빡임: start = 100000 + 50, end = 100300 + 50.
    assert (62921, 63465, 544) in small_runs
    assert (100050, 100350, 300) in small_runs

    small_events = [(s, e) for s, out in small for e in out.events]
    large_events = [(s, e) for s, out in large for e in out.events]
    assert [(s, e.type, e.phase) for s, e in small_events] == [
        (0, EVENT_CALIBRATING, PHASE_ONSET),
        (60000, EVENT_CALIBRATING, PHASE_END),
        (71000, EVENT_MICROSLEEP, PHASE_ONSET),
        (71500, EVENT_MICROSLEEP, PHASE_END),
        (83000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET),
        (84300, EVENT_GAZE_AWAY_LONG, PHASE_END),
    ]
    # microsleep의 시작은 규칙용 진입 0.7을 지난 시각: 69900 + 70 (손 계산). 프레임 71000에서 1030 ms.
    microsleep = small_events[2][1]
    assert (microsleep.t_start_ms - 1, microsleep.duration_ms) == (69970, 1030)
    assert len(large_events) == len(small_events)
    for (sa, a), (sb, b) in zip(small_events, large_events):
        # 이벤트 시각은 원래 t_ms로 적는다: 원점만큼만 다르다.
        assert sa == sb
        assert a.t_start_ms - 1 == b.t_start_ms - big
        assert (a.type, a.phase, a.duration_ms, a.confidence, a.source, a.provisional, a.severity) == (
            b.type,
            b.phase,
            b.duration_ms,
            b.confidence,
            b.source,
            b.provisional,
            b.severity,
        )
    assert small_pipe.baseline is not None and big_pipe.baseline is not None
    assert small_pipe.baseline == big_pipe.baseline
    # 반쯤 감은 프레임이 섞여도 보정 결과는 합성 얼굴의 뜬 눈 값이다(위 손 계산의 전제).
    assert small_pipe.baseline.e_l == pytest.approx(0.3)
    assert small_pipe.baseline.b_l == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 긴 끊김
# ---------------------------------------------------------------------------


def test_5s_gap_emits_catch_up_steps_and_calls_model_for_each_of_them(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    model = StubModel()
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0, model=model)
    outputs = run(pipe, pipeline_spec, range(0, 182001, 100), closed=varied_blinks)
    # 기준선을 넣어 줬으므로 첫 확률은 60 + 120 − 1 = 179초에 나온다.
    assert [s for s, out in outputs if out.probs is not None] == [179000, 180000, 181000, 182000]
    calls_before = pipe.model_calls
    assert calls_before == 4

    # 5초 끊김: 프레임 하나가 스텝 183~187을 한꺼번에 발행하고, 모델은 밀린 스텝마다 한 번씩 부른다
    # (2026-11-13 소유자 결정. 그 전에는 마지막 스텝에서 한 번만 불렀다).
    out = pipe.update(raw(pipeline_spec, 1 + 187000))
    assert out is not None
    assert [step.k for step in out.steps] == [183, 184, 185, 186, 187]
    assert all(step.full_lookback for step in out.steps)
    assert pipe.model_calls == calls_before + 5
    assert out.step_probs == ((0.2, 0.3, 0.5),) * 5
    assert out.probs == (0.2, 0.3, 0.5)
    steps = {step.k: step for step in all_steps(outputs)}
    for step in out.steps:
        steps[step.k] = step
    # 스텝 k에서 부를 때의 창은 그 스텝까지의 120개(k − 119 ~ k)다. 뒤 스텝이 미리 들어가 있지 않다.
    assert len(model.windows) == 9
    for index, k in enumerate(range(183, 188)):
        assert model.windows[4 + index] == [steps[j].values for j in range(k - 119, k + 1)]
    assert model.windows[-1] == [steps[k].values for k in range(68, 188)]
    # 끊긴 동안의 스텝은 데이터 없는 시간으로 채워진다.
    assert out.steps[-1].values[F_FACE_MISSING_RATIO_10] == pytest.approx((10000.0 - 5000.0 - 200.0) / 10000.0)


def test_12s_gap_resets_and_model_waits_for_a_new_full_window(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    model = StubModel()
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0, model=model)
    # 180.5초에서 눈을 감은 채(감김 진행 중) 프레임이 끊긴다.
    run(pipe, pipeline_spec, range(0, 181001, 100), closed=lambda s: s >= 180500)
    assert pipe.model_calls == 3  # 179, 180, 181
    emitted_before = pipe.steps_emitted

    # 12초 끊김 뒤의 프레임: 끊긴 동안의 스텝을 내지 않고, 진행 중이던 감김도 버린다(기준선은 그대로).
    out = pipe.update(raw(pipeline_spec, 1 + 193000))
    assert out is not None
    assert out.steps == ()
    assert out.step_probs == ()
    assert out.probs is None
    assert out.runs == ()
    assert out.state.provisional is False
    assert pipe.steps_emitted == emitted_before
    # 규칙 엔진도 세션을 시작할 때처럼 다시 시작한다: 끊긴 12초 때문에 분석 불가가 시작되지 않는다.
    assert out.events == ()
    assert out.alerts == ()

    later = run(pipe, pipeline_spec, range(193100, 374001, 100))
    steps = all_steps(later)
    # 새 프레임이 속한 스텝(k = 194)부터 다시 발행한다. 조회 구간은 193000에서 다시 차기 시작한다.
    assert steps[0].k == 194
    assert steps[0].full_lookback is False
    assert steps[0].values[F_MAX_CLOSURE_MS_60] == 0.0  # 버린 감김은 남지 않는다
    assert [step.k for step in steps if step.full_lookback][0] == 253
    # 모델은 다시 120개의 다 찬 스텝(253~372)이 쌓인 뒤에 부른다.
    assert [s for s, out in later if out.probs is not None] == [372000, 373000, 374000]
    assert pipe.model_calls == 6
    assert model.windows[-3] == [step.values for step in steps if 253 <= step.k <= 372]


@pytest.mark.parametrize("gap_ms", [10000, 10001])
def test_feature_detector_and_accumulator_reset_only_above_10s(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, gap_ms: int
) -> None:
    # 5.0초에 눈을 감고, 5.5초의 프레임 다음 프레임이 gap_ms 뒤에 온다(눈을 뜨고 있다). 초기화는 간격이 10초를
    # "넘을" 때만 한다: 정확히 10000 ms이면 특징용 검출기와 누적기가 이어지고, 10001 ms이면 둘 다 다시 시작한다.
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    before = run(pipe, pipeline_spec, range(0, 5501, 100), closed=lambda s: s >= 5000)
    assert [r for _, out in before for r in out.runs] == []
    assert [step.k for step in all_steps(before)] == [1, 2, 3, 4, 5]

    out = pipe.update(raw(pipeline_spec, 1 + 5500 + gap_ms))
    assert out is not None
    later = run(pipe, pipeline_spec, [5500 + gap_ms + 100 * i for i in range(1, 31)])
    later_steps = all_steps(later)
    assert [step.k for step in later_steps] == [16, 17, 18]
    assert [r for _, o in later for r in o.runs] == []
    if gap_ms == 10000:
        # 감김은 4900(0.0) -> 5000(1.0)의 한가운데 4950에서 시작했고 5500까지 봤다. 500 ms 넘게 끊겼으므로 잘린다.
        (run_,) = out.runs
        assert (run_.start_ms, run_.end_ms, run_.duration_ms) == (1 + 4950, 1 + 5500, 550)
        assert (run_.truncated, run_.velocity_valid) == (True, False)
        # 끊긴 동안의 스텝 6~15를 이 프레임이 발행한다. 모두 끊기기 직전의 진행 중 감김(550 ms)을 쓴다.
        assert [step.k for step in out.steps] == list(range(6, 16))
        assert [step.values[F_MAX_CLOSURE_MS_60] for step in out.steps] == [550.0] * 10
        # 잘린 구간은 그 프레임이 속한 스텝 16의 통에 들어간다.
        assert later_steps[0].values[F_MAX_CLOSURE_MS_60] == 550.0
        # 끊긴 10초는 얼굴 없는 시간이다: 규칙 엔진은 초기화하지 않고 분석 불가를 시작한다.
        assert [(e.type, e.phase) for e in out.events] == [("unable_to_analyze", PHASE_ONSET)]
        assert pipe.steps_emitted == 18
    else:
        # 진행 중이던 감김을 버리고(구간 없음) 끊긴 동안의 스텝도 내지 않는다. 스텝 16부터 새로 누적한다.
        assert out.runs == ()
        assert out.steps == ()
        assert later_steps[0].values[F_MAX_CLOSURE_MS_60] == 0.0
        assert out.events == ()
        assert pipe.steps_emitted == 8


def test_12s_gap_then_closed_eyes_is_judged_like_a_fresh_session(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    # 30초 동안 앞을 보다가 12초 끊기고, 끊김 뒤 첫 프레임(42000)부터 9초 동안 눈을 감는다. 80 km/h.
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    times = list(range(0, 30001, 100)) + list(range(42000, 58001, 100))
    outputs = run(pipe, pipeline_spec, times, closed=lambda s: 42000 <= s < 51000, speed_kmh=80.0)
    # 분석 불가 없이 바로 판정한다. 규칙용 검출기를 초기화했으므로 감김의 시작은 42000(보간할 직전 표본이 없다)이고
    # 길이가 1000, 3000이 되는 프레임에서 microsleep, sleep이 시작된다. 뜬 첫 프레임(51000)에서 끝난다.
    events = [(s, e) for s, out in outputs for e in out.events]
    assert [(s, e.type, e.phase, e.t_start_ms, e.duration_ms) for s, e in events] == [
        (43000, EVENT_MICROSLEEP, PHASE_ONSET, 1 + 42000, 1000),
        (45000, EVENT_SLEEP, PHASE_ONSET, 1 + 42000, 3000),
        (51000, EVENT_SLEEP, PHASE_END, 1 + 42000, 9000),
    ]
    # sleep은 3초 뒤(48000)에 한 번 되풀이한다. 51000에서는 상태가 끝났다.
    assert all_alerts(outputs) == [
        (43000, EVENT_MICROSLEEP, PHASE_ONSET),
        (45000, EVENT_SLEEP, PHASE_ONSET),
        (48000, EVENT_SLEEP, PHASE_ONSET),
    ]


@pytest.mark.parametrize("gap_ms", [8000, 12000])
def test_waiting_case_of_closed_eyes_after_a_gap_below_and_above_10s(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, gap_ms: int
) -> None:
    # docs/SPEC.md 20장 6번에 적은 사례(30 fps, 80 km/h). 30초 동안 눈을 뜨고 있다가 gap_ms 끊기고, 끊김 뒤의 첫
    # 프레임부터 5초 동안 눈을 감는다. 10초 이하의 끊김은 얼굴 없는 시간이라(소유자의 결정을 기다리는 동작) 그
    # 5초가 통째로 분석 불가에 들어간다. 10초를 넘으면 세션을 시작할 때처럼 바로 판정한다(2026-11-13 결정).
    def fps30(i: int) -> int:
        return (i * 1000 + 15) // 30

    before = [fps30(i) for i in range(30 * 30)]
    first_after = before[-1] + gap_ms
    assert before[-1] == 29967
    closed_times = [first_after + fps30(i) for i in range(30 * 5)]
    after = closed_times + [first_after + 5000 + fps30(i) for i in range(30 * 2)]
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    outputs = run(pipe, pipeline_spec, before + after, closed=lambda s: s in set(closed_times), speed_kmh=80.0)
    events = [(s - first_after, e.type, e.phase, e.duration_ms) for s, out in outputs for e in out.events]
    alerts = [(s - first_after, e.type) for s, out in outputs for e in out.alerts]
    if gap_ms == 8000:
        assert events == [(0, "unable_to_analyze", PHASE_ONSET, 0), (5000, "unable_to_analyze", PHASE_END, 5000)]
        assert alerts == [(0, "unable_to_analyze")]  # 눈을 감은 5초 동안 microsleep도 sleep도 없다
    else:
        assert events == [
            (1000, EVENT_MICROSLEEP, PHASE_ONSET, 1000),
            (3000, EVENT_SLEEP, PHASE_ONSET, 3000),
            (5000, EVENT_SLEEP, PHASE_END, 5000),
        ]
        assert alerts == [(1000, EVENT_MICROSLEEP), (3000, EVENT_SLEEP)]


@pytest.mark.parametrize("drowsy_after_gap", [True, False])
def test_20_minute_gap_ends_drowsy_and_a_new_episode_can_start_and_alert(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, drowsy_after_gap: bool
) -> None:
    # 매초 300 ms씩 눈을 감는다(PERCLOS 0.3). 200초 뒤 20분 끊기고, 세션 시간 1400000부터 75초 더 이어진다. 80 km/h.
    def closed(s: int) -> bool:
        return s % 1000 < 300 and (s <= 200000 or drowsy_after_gap)

    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    times = list(range(0, 200001, 100)) + list(range(1400000, 1475001, 100))
    outputs = run(pipe, pipeline_spec, times, closed=closed, speed_kmh=80.0)
    steps = {step.k: step for step in all_steps(outputs)}
    assert steps[60].values[F_PERCLOS_60] == pytest.approx(0.3)
    events = [(s, e) for s, out in outputs for e in out.events]
    # 끊김 전: 다 찬 스텝 60~69 -> 프레임 69000에서 시작. 끊김 뒤 첫 프레임에서 끝난다(발행 시각은 그 프레임,
    # confidence 0.0). 끊김 뒤에는 누적이 1400000에서 다시 시작해 스텝 1460부터 다 차고, 1460~1469가 새로 열 개를
    # 채운 프레임 1469000에서 새 졸음이 시작된다. 눈을 뜨고 있으면 새 졸음은 없다.
    expected = [
        (69000, EVENT_DROWSY, PHASE_ONSET, 1 + 60000, 9000),
        (1400000, EVENT_DROWSY, PHASE_END, 1 + 60000, 1340000),
    ]
    expected_alerts = [(69000, EVENT_DROWSY, PHASE_ONSET)]
    if drowsy_after_gap:
        expected.append((1469000, EVENT_DROWSY, PHASE_ONSET, 1 + 1460000, 9000))
        expected_alerts.append((1469000, EVENT_DROWSY, PHASE_ONSET))
    assert [(s, e.type, e.phase, e.t_start_ms, e.duration_ms) for s, e in events] == expected
    end = events[1][1]
    assert (end.confidence, end.source, end.provisional) == (0.0, SOURCE_PERCLOS, False)
    # 첫 경고 뒤 재경고 대기(300초)가 지났으므로 새 졸음도 경고한다. 분석 불가 경고는 없다.
    assert all_alerts(outputs) == expected_alerts
    gap_frame = dict(outputs)[1400000]
    assert gap_frame.steps == () and gap_frame.step_probs == ()
    assert [step.k for step in all_steps(outputs) if step.k > 200][:1] == [1401]


@pytest.mark.parametrize("drowsy_after_gap", [True, False])
def test_deferred_drowsy_alert_is_dropped_by_a_20_minute_gap(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, drowsy_after_gap: bool
) -> None:
    # 위 시험과 같은 세션인데 끊김 전에는 45 km/h(졸음 경고가 막혀 미뤄진다), 끊김 뒤에는 100 km/h다.
    def closed(s: int) -> bool:
        return s % 1000 < 300 and (s <= 200000 or drowsy_after_gap)

    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    outputs = run(pipe, pipeline_spec, range(0, 200001, 100), closed=closed, speed_kmh=45.0)
    assert all_events(outputs) == [(69000, EVENT_DROWSY, PHASE_ONSET)]
    assert all_alerts(outputs) == []
    outputs = run(pipe, pipeline_spec, range(1400000, 1475001, 100), closed=closed, speed_kmh=100.0)
    # 끊김 뒤 첫 프레임에서 상태가 끝나므로, 속도가 올랐어도 20분 전의 졸음을 뒤늦게 경고하지 않는다.
    # 새 졸음(1469000)은 그 프레임에서 바로 경고한다. 눈을 뜨고 있으면 경고가 없다.
    expected_events = [(1400000, EVENT_DROWSY, PHASE_END)]
    expected_alerts = []
    if drowsy_after_gap:
        expected_events.append((1469000, EVENT_DROWSY, PHASE_ONSET))
        expected_alerts.append((1469000, EVENT_DROWSY, PHASE_ONSET))
    assert all_events(outputs) == expected_events
    assert all_alerts(outputs) == expected_alerts
    if drowsy_after_gap:
        frame = dict(outputs)[1469000]
        assert frame.alerts == frame.events  # 미뤄 둔 사본이 아니라 새 onset 그대로다


def test_steps_emitted_by_the_first_frame_after_a_long_gap_carry_no_value(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig
) -> None:
    # 누적 시작(5초) 전에 끊겨서, 누적기가 처음 받는 프레임이 11초 끊김 뒤의 프레임인 경우에만 그런 프레임이 스텝을
    # 발행한다(누적 시작 시각에서 8초 뒤라 누적기는 초기화하지 않고 빈 스텝 6~13을 낸다). 그 스텝들은 조회 구간이
    # 덜 찼고 확률도 없어서, 규칙 엔진의 초기화보다 먼저 처리돼도 졸음 판정을 바꾸지 못한다.
    rules = apply_overrides(rules_config, {"drowsiness.source": "either"})
    model = StubModel((0.05, 0.05, 0.9))
    pipe = DriverPipeline(pipeline_spec, rules, baseline=synthetic_baseline(), feature_start_ms=5000, model=model)

    def closed(s: int) -> bool:
        return s % 1000 < 300

    before = run(pipe, pipeline_spec, range(0, 2001, 100), closed=closed)
    assert all_steps(before) == []
    out = pipe.update(raw(pipeline_spec, 1 + 13000, closed=True))
    assert out is not None
    assert [step.k for step in out.steps] == [6, 7, 8, 9, 10, 11, 12, 13]
    assert [step.full_lookback for step in out.steps] == [False] * 8
    assert out.step_probs == (None,) * 8
    assert out.probs is None
    assert out.events == ()
    assert pipe.model_calls == 0
    # 이후는 평소대로다: 스텝 65부터 다 차고(끝 ≥ 5000 + 60000) PERCLOS 0.3이 65~74에서 열 개를 채운다.
    later = run(pipe, pipeline_spec, range(13100, 76001, 100), closed=closed)
    events = [(s, e) for s, out in later for e in out.events]
    assert [(s, e.type, e.phase, e.t_start_ms, e.source) for s, e in events] == [
        (74000, EVENT_DROWSY, PHASE_ONSET, 1 + 65000, SOURCE_PERCLOS)
    ]
    assert pipe.model_calls == 0  # 창(120개)이 차기 전이다


# ---------------------------------------------------------------------------
# 프레임이 1초 넘게 멈췄을 때의 모델 졸음 판정 (2026-11-13 소유자 결정: 밀린 스텝마다 모델을 부른다)
# ---------------------------------------------------------------------------


def stalled(t_end_ms: int, stalls: Sequence[tuple[int, int]]) -> list[int]:
    """100 ms 간격의 세션 시간 0 ~ t_end_ms에서 [a, b) 구간의 프레임을 뺀 것."""
    return [s for s in range(0, t_end_ms + 1, 100) if not any(a <= s < b for a, b in stalls)]


EVERY_9S = tuple((9000 * i + 6900, 9000 * i + 9000) for i in range(0, 40))  # 9초마다 2.1초 멈춤


@pytest.mark.parametrize(
    ("stalls", "report_frame", "multi_step_frames", "steps_in_report_frame"),
    [
        ((), 188000, 0, 1),  # 멈춤 없음
        (((185050, 187050),), 188000, 1, 1),  # 2.1초 멈춤 한 번: 프레임 187100이 스텝 186, 187을 낸다
        (((184950, 186000),), 188000, 1, 1),  # 1.1초 멈춤이 스텝 경계 둘을 넘는다: 프레임 186000이 185, 186을 낸다
        (EVERY_9S, 189000, 21, 3),  # 9초마다 2.1초: 프레임 189000이 187, 188, 189를 한꺼번에 낸다
        (tuple((9000 * i + 6950, 9000 * i + 7850) for i in range(0, 40)), 188000, 0, 1),  # 0.9초 멈춤(경계 하나)
    ],
)
def test_model_drowsy_onset_is_not_delayed_by_frame_stalls(
    pipeline_spec: PipelineSpec,
    rules_config: RulesConfig,
    stalls: tuple[tuple[int, int], ...],
    report_frame: int,
    multi_step_frames: int,
    steps_in_report_frame: int,
) -> None:
    rules = apply_overrides(rules_config, {"drowsiness.source": "model"})
    model = StubModel((0.05, 0.05, 0.9))
    pipe = DriverPipeline(pipeline_spec, rules, baseline=synthetic_baseline(), feature_start_ms=0, model=model)
    outputs = run(pipe, pipeline_spec, stalled(190000, stalls), speed_kmh=80.0)
    assert sum(1 for _, out in outputs if len(out.steps) > 1) == multi_step_frames
    # 10초 이하의 멈춤은 스텝을 끊지 않는다: 스텝 1~190이 모두 나오고 창은 스텝 179에서 처음 찬다.
    assert [step.k for step in all_steps(outputs)] == list(range(1, 191))
    with_probs = [step.k for _, out in outputs for step, p in zip(out.steps, out.step_probs) if p is not None]
    assert with_probs == list(range(179, 191))
    assert pipe.model_calls == 12
    # p_drowsy 0.9가 스텝 179~188에서 열 개 연속: 멈춤이 있든 없든 스텝 188에서 시작한다. 이벤트의 시각은
    # 스텝 시계라 같고(t_start = 스텝 179의 끝, 길이 9000), 그 스텝을 발행한 프레임만 다르다.
    events = [(s, e) for s, out in outputs for e in out.events]
    assert [(s, e.type, e.phase, e.t_start_ms, e.duration_ms, e.confidence) for s, e in events] == [
        (report_frame, EVENT_DROWSY, PHASE_ONSET, 1 + 179000, 9000, 0.9)
    ]
    frame = dict(outputs)[report_frame]
    assert len(frame.steps) == steps_in_report_frame
    assert frame.step_probs == ((0.05, 0.05, 0.9),) * steps_in_report_frame
    assert all_alerts(outputs) == [(report_frame, EVENT_DROWSY, PHASE_ONSET)]


@pytest.mark.parametrize("stalls", [(), tuple((9000 * i + 6900, 9000 * i + 9000) for i in range(33, 40))])
def test_model_drowsy_end_is_reported_despite_frame_stalls(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, stalls: tuple[tuple[int, int], ...]
) -> None:
    rules = apply_overrides(rules_config, {"drowsiness.source": "model"})
    model = StubModel((0.05, 0.05, 0.9))
    pipe = DriverPipeline(pipeline_spec, rules, baseline=synthetic_baseline(), feature_start_ms=0, model=model)
    times = stalled(320000, stalls)
    outputs = run(pipe, pipeline_spec, [s for s in times if s < 300000])
    model.probs = (0.45, 0.45, 0.1)  # 세션 시간 300초의 프레임부터(스텝 300부터) p_drowsy 0.1
    outputs += run(pipe, pipeline_spec, [s for s in times if s >= 300000])
    if stalls:
        # 첫 멈춤(303.9 ~ 306초) 뒤의 프레임 306000이 스텝 304, 305, 306을 한꺼번에 낸다. 셋 다 확률을 받는다.
        assert [step.k for step in dict(outputs)[306000].steps] == [304, 305, 306]
        assert dict(outputs)[306000].step_probs == ((0.45, 0.45, 0.1),) * 3
    # 0.1이 스텝 300~309에서 열 개 연속이라 스텝 309(프레임 309000)에서 끝난다. 결정 전에는 멈춤이 있으면
    # 앞쪽 스텝에 확률이 없어 연속이 끊기고, 9초마다 멈추는 동안에는 끝나지 않았다.
    events = [(s, e) for s, out in outputs for e in out.events]
    assert [(s, e.type, e.phase, e.t_start_ms, e.duration_ms, e.confidence) for s, e in events] == [
        (188000, EVENT_DROWSY, PHASE_ONSET, 1 + 179000, 9000, 0.9),
        (309000, EVENT_DROWSY, PHASE_END, 1 + 179000, 130000, 0.1),
    ]


# ---------------------------------------------------------------------------
# 규칙과 경고가 이어지는지
# ---------------------------------------------------------------------------


def test_closure_run_output_and_step_attribution(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    outputs = run(pipe, pipeline_spec, range(0, 3001, 100), closed=lambda s: 300 <= s < 600)
    runs = [(s, r) for s, out in outputs for r in out.runs]
    assert len(runs) == 1
    s, run_ = runs[0]
    # closure 0 -> 1 -> 0. 0.5를 지난 시각은 250과 550(원점 1을 더한다). 프레임 600이 발행한다.
    assert s == 600
    assert (run_.start_ms, run_.end_ms, run_.duration_ms, run_.truncated) == (251, 551, 300, False)
    step1 = all_steps(outputs)[0]
    assert step1.k == 1
    assert step1.values[F_BLINK_DUR_MEAN_60] == 300.0
    assert step1.values[F_MAX_CLOSURE_MS_60] == 300.0


def test_microsleep_and_speed_gate(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    def closed(s: int) -> bool:
        return 2000 <= s < 3500

    slow = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    slow_outputs = run(slow, pipeline_spec, range(0, 5001, 100), closed=closed, speed_kmh=10.0)
    # 2000에서 0.7을 지난 시각은 1970. 프레임 3000에서 1030 ms(≥ 1000).
    assert all_events(slow_outputs) == [(3000, EVENT_MICROSLEEP, PHASE_ONSET), (3500, EVENT_MICROSLEEP, PHASE_END)]
    onset = dict(slow_outputs)[3000].events[0]
    assert (onset.t_start_ms, onset.duration_ms, onset.provisional) == (1 + 1970, 1030, False)
    assert all_alerts(slow_outputs) == []  # 10 km/h: 상태 이벤트는 나가고 경고만 억제된다

    fast = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    fast_outputs = run(fast, pipeline_spec, range(0, 5001, 100), closed=closed, speed_kmh=20.0)
    assert all_alerts(fast_outputs) == [(3000, EVENT_MICROSLEEP, PHASE_ONSET)]

    unknown = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    unknown_outputs = run(unknown, pipeline_spec, range(0, 5001, 100), closed=closed)
    assert all_alerts(unknown_outputs) == [(3000, EVENT_MICROSLEEP, PHASE_ONSET)]


def test_provisional_microsleep_during_calibration(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config)
    outputs = run(pipe, pipeline_spec, range(0, 12001, 100), closed=lambda s: 5000 <= s < 6500)
    events = [(s, e) for s, out in outputs for e in out.events]
    assert [(s, e.type, e.phase) for s, e in events] == [
        (0, EVENT_CALIBRATING, PHASE_ONSET),
        (6000, EVENT_MICROSLEEP, PHASE_ONSET),
        (6500, EVENT_MICROSLEEP, PHASE_END),
    ]
    # 기준선 완성 전: 전역 상수로 계산한 임시 closure로 판정하고 provisional=true를 단다(D4).
    assert all(e.provisional for _, e in events)
    assert dict(outputs)[5500].state.closure == pytest.approx(1.0)
    assert dict(outputs)[5500].state.provisional is True
    # calibrating은 경고하지 않고, 임시 판정 microsleep은 경고한다.
    assert all_alerts(outputs) == [(6000, EVENT_MICROSLEEP, PHASE_ONSET)]
    assert all_steps(outputs) == []


def test_gaze_rule_through_pipeline(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)

    def away(s: int) -> float:
        return 30.0 if 2000 <= s < 6000 else 0.0

    outputs = run(pipe, pipeline_spec, range(0, 8001, 100), yaw=away, speed_kmh=30.0)
    assert all_events(outputs) == [(5000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET), (6300, EVENT_GAZE_AWAY_LONG, PHASE_END)]
    assert all_alerts(outputs) == [(5000, EVENT_GAZE_AWAY_LONG, PHASE_ONSET)]
    assert dict(outputs)[3000].state.yaw_rel == pytest.approx(30.0)
    assert pipe.gaze_zone == "in"
    pipe.update(raw(pipeline_spec, 1 + 8100, yaw=30.0))
    pipe.update(raw(pipeline_spec, 1 + 8400, yaw=30.0))
    assert pipe.gaze_zone == "out"


def test_perclos_drowsy_end_to_end(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    # 80초까지 2초마다 400 ms씩 감는다(PERCLOS 0.2). 그 뒤로는 뜨고 있다.
    def closed(s: int) -> bool:
        return s < 80000 and s % 2000 < 400

    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    outputs = run(pipe, pipeline_spec, range(0, 125001, 100), closed=closed, speed_kmh=60.0)
    steps = {step.k: step for step in all_steps(outputs)}
    assert steps[60].values[F_PERCLOS_60] == pytest.approx(0.2)
    # 창 [1000(k−60), 1000k) 안의 감김 주기 수: k가 짝수면 (140 − k)/2, 홀수면 (139 − k)/2. 주기마다 400 ms.
    assert steps[110].values[F_PERCLOS_60] == pytest.approx(0.1)
    assert steps[111].values[F_PERCLOS_60] == pytest.approx(14.0 * 400.0 / 60000.0)

    events = [(s, e) for s, out in outputs for e in out.events]
    # 시작: 다 찬 스텝 60~69가 모두 ≥ 0.15 -> 스텝 69(프레임 69000). t_start는 스텝 60의 끝.
    # 종료: 처음 0.10 미만이 되는 스텝은 111(스텝 109, 110은 정확히 0.10) -> 열 번째인 스텝 120.
    assert [(s, e.type, e.phase) for s, e in events] == [
        (69000, EVENT_DROWSY, PHASE_ONSET),
        (120000, EVENT_DROWSY, PHASE_END),
    ]
    onset, end = events[0][1], events[1][1]
    assert (onset.t_start_ms, onset.duration_ms, onset.source, onset.severity) == (1 + 60000, 9000, SOURCE_PERCLOS, 2)
    assert onset.confidence == pytest.approx(1.0)
    assert (end.t_start_ms, end.duration_ms) == (1 + 60000, 60000)
    # 60 km/h(≥ 50)이므로 졸음 경고가 나간다.
    assert all_alerts(outputs) == [(69000, EVENT_DROWSY, PHASE_ONSET)]


@pytest.mark.parametrize(("lift_at", "alert_frame"), [(75000, 75000), (119900, 119900), (120000, None)])
def test_perclos_drowsy_alert_blocked_by_speed_is_issued_when_the_gate_opens(
    pipeline_spec: PipelineSpec, rules_config: RulesConfig, lift_at: int, alert_frame: int | None
) -> None:
    # 위 시험과 같은 세션: 졸음은 69000에서 시작해 120000에서 끝난다. 속도는 lift_at 전까지 45 km/h, 그 뒤 100 km/h.
    def closed(s: int) -> bool:
        return s < 80000 and s % 2000 < 400

    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0)
    outputs = run(pipe, pipeline_spec, range(0, lift_at, 100), closed=closed, speed_kmh=45.0)
    outputs += run(pipe, pipeline_spec, range(lift_at, 125001, 100), closed=closed, speed_kmh=100.0)
    assert all_events(outputs) == [(69000, EVENT_DROWSY, PHASE_ONSET), (120000, EVENT_DROWSY, PHASE_END)]
    alerts = [(s, a) for s, out in outputs for a in out.alerts]
    if alert_frame is None:
        # 속도가 오른 바로 그 프레임에서 상태가 끝났다: 미뤄 둔 경고는 나가지 않는다.
        assert alerts == []
    else:
        # 상태가 이어지는 동안 50 km/h 이상인 첫 프레임에서 한 번. 시작 이벤트의 사본이고 길이만 그 프레임 기준이다.
        onset = dict(outputs)[69000].events[0]
        assert alerts == [(alert_frame, dataclasses.replace(onset, duration_ms=alert_frame - 60000))]


def test_model_drowsy_end_to_end_and_event_order(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    rules = apply_overrides(rules_config, {"drowsiness.source": "model"})
    model = StubModel((0.1, 0.1, 0.8))

    def closed(s: int) -> bool:
        return 187000 <= s < 188500  # 프레임 188000에서 microsleep이 시작되게 맞춘 감김

    pipe = DriverPipeline(pipeline_spec, rules, baseline=synthetic_baseline(), feature_start_ms=0, model=model)
    outputs = run(pipe, pipeline_spec, range(0, 190001, 100), closed=closed, speed_kmh=45.0)
    events = [(s, e) for s, out in outputs for e in out.events]
    # 첫 확률은 스텝 179. p_drowsy = 0.8이 10스텝 연속인 스텝 188에서 시작한다.
    assert [(s, e.type, e.phase) for s, e in events] == [
        (188000, EVENT_DROWSY, PHASE_ONSET),
        (188000, EVENT_MICROSLEEP, PHASE_ONSET),
        (188500, EVENT_MICROSLEEP, PHASE_END),
    ]
    drowsy = events[0][1]
    assert (drowsy.t_start_ms, drowsy.duration_ms) == (1 + 179000, 9000)
    assert (drowsy.confidence, drowsy.source) == (0.8, SOURCE_MODEL)
    # 한 프레임 안에서는 스텝 이벤트가 먼저, 프레임 이벤트가 나중이다.
    frame = dict(outputs)[188000]
    assert [e.type for e in frame.events] == [EVENT_DROWSY, EVENT_MICROSLEEP]
    assert frame.probs == (0.1, 0.1, 0.8)
    # 45 km/h: microsleep(20 km/h부터)은 경고하고 졸음(50 km/h부터)은 억제한다.
    assert [e.type for e in frame.alerts] == [EVENT_MICROSLEEP]

    # 속도 입력이 없으면 둘 다 경고하고 우선순위(microsleep이 먼저) 순으로 적는다.
    pipe = DriverPipeline(pipeline_spec, rules, baseline=synthetic_baseline(), feature_start_ms=0, model=model)
    outputs = run(pipe, pipeline_spec, range(0, 190001, 100), closed=closed)
    assert [e.type for e in dict(outputs)[188000].alerts] == [EVENT_MICROSLEEP, EVENT_DROWSY]


# ---------------------------------------------------------------------------
# 모델 호출의 방어
# ---------------------------------------------------------------------------


def test_model_must_return_three_probabilities(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    pipe = DriverPipeline(
        pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0, model=StubModel((0.5, 0.5))
    )
    with pytest.raises(ValueError):
        run(pipe, pipeline_spec, range(0, 179001, 100))


def test_is_finite_helper() -> None:
    assert pipeline_module._is_finite(0.0)
    assert pipeline_module._is_finite(-1.5e300)
    assert not pipeline_module._is_finite(math.nan)
    assert not pipeline_module._is_finite(math.inf)
    assert not pipeline_module._is_finite(-math.inf)


def test_non_finite_window_skips_inference(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    model = StubModel()
    pipe = DriverPipeline(pipeline_spec, rules_config, baseline=synthetic_baseline(), feature_start_ms=0, model=model)
    run(pipe, pipeline_spec, range(0, 179001, 100))
    assert pipe.model_calls == 1
    # 정상 입력에서는 생기지 않는 값이다. 방어 동작만 보려고 창의 가장 최근 스텝을 NaN으로 바꿔 놓는다.
    pipe._window[-1] = (math.nan,) * N_FEATURES
    later = run(pipe, pipeline_spec, range(179100, 181001, 100))
    # 창에 유한하지 않은 값이 있는 동안은 추론을 건너뛰고 센다. 스텝과 규칙은 그대로 돈다.
    assert [step.k for step in all_steps(later)] == [180, 181]
    assert all(out.probs is None for _, out in later)
    assert pipe.model_calls == 1
    assert pipe.nonfinite_windows == 2
    assert len(model.windows) == 1
