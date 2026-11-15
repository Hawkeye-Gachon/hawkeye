"""`drowsy.synth` 테스트: 기하 왕복, 반전, 각본 시나리오, 통계 생성기, 데이터셋 쓰기 (PLAN 9장, D14).

합성 값만 쓴다. 통계 생성기로 만든 세션의 수치는 파이프라인 점검용이고 성능이 아니다.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from drowsy import calibrate, signals, synth
from drowsy.config import load_config
from drowsy.dataset import build_manifest, replay_video
from drowsy.extract import (
    drop_sim_latencies,
    flow_limiter_select,
    read_extract_metadata,
    read_raw_frames,
    write_raw_frames,
)
from drowsy.pipeline import DriverPipeline
from drowsy.spec import PipelineSpec, RulesConfig, apply_overrides
from drowsy.synth import (
    Blink,
    Closure,
    FaceMissing,
    HeadTurn,
    Linger,
    LookAside,
    Nod,
    Open,
    Scenario,
    Yawn,
    default_profile,
    frame_from_signals,
    generate_session,
    lid_signals,
    make_subject,
    mirror_frame,
    no_face_frame,
    render,
    scenario_times,
    sequence,
)
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OK,
    EVENT_MICROSLEEP,
    F_LONG_CLOSURE_COUNT_60,
    F_NOD_COUNT_60,
    F_YAWN_COUNT_60,
    IDENTITY_R0,
    PHASE_ONSET,
    Baseline,
    FrameSignals,
    PipelineOutput,
    RawFrame,
)

GEOMETRY_TOLERANCE = 1e-9  # 좌표와 행렬을 거친 값의 허용 오차(실측 최대 1e-12 미만)
FRAME_SIZES = [(640, 480), (480, 640), (1920, 1080), (568, 320), (333, 777)]


def random_signals(rng: random.Random, t_ms: int) -> FrameSignals:
    return FrameSignals(
        t_ms=t_ms,
        face=True,
        ear_l=rng.uniform(0.0, 0.5),
        ear_r=rng.uniform(0.0, 0.5),
        eye_w_l=rng.uniform(6.5, 120.0),
        eye_w_r=rng.uniform(6.5, 120.0),
        blink_l=rng.random(),
        blink_r=rng.random(),
        eye_look=tuple(rng.random() for _ in range(8)),
        jaw_open=rng.random(),
        yaw=rng.uniform(-80.0, 80.0),
        pitch=rng.uniform(-80.0, 80.0),
        roll=rng.uniform(-170.0, 170.0),
        iris_h_l=rng.uniform(0.2, 0.8),
        iris_h_r=rng.uniform(0.2, 0.8),
        iris_v_l=rng.uniform(-0.1, 0.1),
        iris_v_r=rng.uniform(-0.1, 0.1),
    )


def assert_signals_match(computed: FrameSignals, requested: FrameSignals) -> None:
    """블렌드셰이프는 정확히, 좌표와 행렬을 거친 값은 반올림 오차 안에서 같아야 한다."""
    assert computed.t_ms == requested.t_ms and computed.face is True
    assert computed.blink_l == requested.blink_l and computed.blink_r == requested.blink_r
    assert computed.eye_look == requested.eye_look
    assert computed.jaw_open == requested.jaw_open
    for name in ("ear_l", "ear_r", "eye_w_l", "eye_w_r", "yaw", "pitch", "roll"):
        assert getattr(computed, name) == pytest.approx(getattr(requested, name), abs=GEOMETRY_TOLERANCE), name
    for name in ("iris_h_l", "iris_h_r", "iris_v_l", "iris_v_r"):
        assert getattr(computed, name) == pytest.approx(getattr(requested, name), abs=GEOMETRY_TOLERANCE), name


def profile_baseline(spec: PipelineSpec, t_complete_ms: int = 0) -> Baseline:
    """`default_profile(spec)`의 얼굴을 보정했을 때 나오는 기준선."""
    profile = default_profile(spec)
    return Baseline(
        e_l=profile.e_open_l,
        e_r=profile.e_open_r,
        b_l=profile.b_open_l,
        b_r=profile.b_open_r,
        r0=IDENTITY_R0,
        jaw0=profile.jaw_rest,
        gaze_h0=0.0,
        gaze_v0=0.0,
        t_complete_ms=t_complete_ms,
        status=BASELINE_OK,
    )


def run_pipeline(
    frames: list[RawFrame], spec: PipelineSpec, rules: RulesConfig, baseline: Baseline | None = None
) -> list[PipelineOutput]:
    pipe = DriverPipeline(spec, rules, baseline)
    outputs = [pipe.update(frame) for frame in frames]
    assert all(out is not None for out in outputs)
    return [out for out in outputs if out is not None]


# ---------------------------------------------------------------------------
# (a) 기하
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", FRAME_SIZES)
def test_round_trip_from_requested_signals(pipeline_spec: PipelineSpec, size: tuple[int, int]) -> None:
    rng = random.Random(size[0] * 10007 + size[1])
    w, h = size
    for k in range(400):
        requested = random_signals(rng, 1 + k)
        centre = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
        frame = frame_from_signals(requested, w, h, pipeline_spec, centre)
        assert_signals_match(signals.compute(frame, pipeline_spec), requested)


def test_frame_carries_what_the_app_receives(pipeline_spec: PipelineSpec) -> None:
    requested = random_signals(random.Random(1), 77)
    frame = frame_from_signals(requested, 1280, 720, pipeline_spec)
    assert (frame.t_ms, frame.face, frame.w, frame.h) == (77, True, 1280, 720)
    assert sorted(frame.lm) == list(pipeline_spec.landmarks.required_indices()) and len(frame.lm) == 44
    assert sorted(frame.bs) == sorted(pipeline_spec.blendshapes.required_names()) and len(frame.bs) == 11
    assert len(frame.m) == 16 and frame.m[3] == frame.m[7] == frame.m[11] == 0.0 and frame.m[15] == 1.0
    # 열 우선: M[i][j] = m[4·j + i]. 회전 블록은 signals.rotation_from_euler와 같다.
    rotation = signals.rotation_from_euler(requested.yaw, requested.pitch, requested.roll)
    assert signals.rotation_from_colmajor(frame.m) == rotation
    # 비반전 영상: MediaPipe Right 눈(33..133)이 영상 왼쪽, Left 눈(362..263)이 오른쪽에 있다.
    lms = pipeline_spec.landmarks
    assert frame.lm[lms.corner_outer_r][0] < frame.lm[lms.corner_inner_r][0] < frame.lm[lms.corner_inner_l][0]
    assert frame.lm[lms.corner_inner_l][0] < frame.lm[lms.corner_outer_l][0]
    assert frame.lm[lms.face_width[0]][0] < frame.lm[lms.corner_outer_r][0]
    assert frame.lm[lms.face_width[1]][0] > frame.lm[lms.corner_outer_l][0]
    assert all(0.0 < x < 1.0 and 0.0 < y < 1.0 for x, y in frame.lm.values())


def test_default_iris_and_unusable_eye(pipeline_spec: PipelineSpec) -> None:
    base = random_signals(random.Random(2), 5)
    requested = dataclasses.replace(base, iris_h_l=None, iris_v_l=None, iris_h_r=None, iris_v_r=None)
    computed = signals.compute(frame_from_signals(requested, 640, 480, pipeline_spec), pipeline_spec)
    assert computed.iris_h_l == pytest.approx(0.5, abs=GEOMETRY_TOLERANCE)
    assert computed.iris_v_r == pytest.approx(0.0, abs=GEOMETRY_TOLERANCE)

    # 눈 폭이 6 px 미만인 눈은 쓸 수 없는 눈이다. 나머지 눈과 블렌드셰이프, 각도는 그대로 나온다.
    narrow = dataclasses.replace(base, eye_w_r=4.0)
    computed = signals.compute(frame_from_signals(narrow, 640, 480, pipeline_spec), pipeline_spec)
    assert computed.eye_w_r == pytest.approx(4.0, abs=GEOMETRY_TOLERANCE)
    assert computed.ear_r == 0.0 and computed.iris_h_r is None and computed.iris_v_r is None
    assert computed.ear_l == pytest.approx(base.ear_l, abs=GEOMETRY_TOLERANCE)
    assert computed.blink_r == base.blink_r and computed.yaw == pytest.approx(base.yaw, abs=GEOMETRY_TOLERANCE)


def test_no_face_frame_and_wrong_eye_look_length(pipeline_spec: PipelineSpec) -> None:
    frame = no_face_frame(9, 320, 240)
    assert frame == RawFrame(t_ms=9, face=False, w=320, h=240, lm={}, bs={}, m=())
    computed = signals.compute(frame, pipeline_spec)
    assert computed.face is False and computed.ear_l == 0.0 and computed.eye_look == (0.0,) * 8
    assert frame_from_signals(computed, 320, 240, pipeline_spec) == frame  # 얼굴 없는 요청은 얼굴 없는 프레임

    bad = dataclasses.replace(random_signals(random.Random(3), 1), eye_look=(0.1, 0.2))
    with pytest.raises(ValueError, match="eye_look"):
        frame_from_signals(bad, 320, 240, pipeline_spec)


def test_mirror_frame_swaps_sides_and_flips_yaw_and_roll(pipeline_spec: PipelineSpec) -> None:
    rng = random.Random(4)
    baseline = profile_baseline(pipeline_spec)
    for k in range(300):
        requested = random_signals(rng, 1 + k)
        frame = frame_from_signals(requested, 720, 1280, pipeline_spec, (rng.uniform(0.3, 0.7), 0.5))
        mirrored = mirror_frame(frame, pipeline_spec)
        computed = signals.compute(mirrored, pipeline_spec)

        assert sorted(mirrored.lm) == sorted(frame.lm) and sorted(mirrored.bs) == sorted(frame.bs)
        assert computed.blink_l == requested.blink_r and computed.blink_r == requested.blink_l
        assert computed.jaw_open == requested.jaw_open
        assert computed.eye_look == requested.eye_look[4:] + requested.eye_look[:4]
        assert signals.gaze_h(computed.eye_look) == -signals.gaze_h(requested.eye_look)
        assert signals.gaze_v(computed.eye_look) == signals.gaze_v(requested.eye_look)
        assert computed.ear_l == pytest.approx(requested.ear_r, abs=GEOMETRY_TOLERANCE)
        assert computed.ear_r == pytest.approx(requested.ear_l, abs=GEOMETRY_TOLERANCE)
        assert computed.eye_w_l == pytest.approx(requested.eye_w_r, abs=GEOMETRY_TOLERANCE)
        assert computed.yaw == pytest.approx(-requested.yaw, abs=GEOMETRY_TOLERANCE)
        assert computed.pitch == pytest.approx(requested.pitch, abs=GEOMETRY_TOLERANCE)
        assert computed.roll == pytest.approx(-requested.roll, abs=GEOMETRY_TOLERANCE)
        # 홍채 비율은 두 눈이 서로 바뀌고 수직 성분의 부호가 뒤집힌다(눈 꼬리 벡터의 방향이 반대다).
        assert computed.iris_h_l == pytest.approx(requested.iris_h_r, abs=GEOMETRY_TOLERANCE)
        assert computed.iris_v_l == pytest.approx(-requested.iris_v_r, abs=GEOMETRY_TOLERANCE)

        # 모델 특징의 재료인 closure는 반전해도 같다(PLAN 3.2절).
        original_state = calibrate.frame_state(signals.compute(frame, pipeline_spec), baseline, pipeline_spec, False)
        mirrored_state = calibrate.frame_state(computed, baseline, pipeline_spec, False)
        assert mirrored_state.closure_valid == original_state.closure_valid
        assert mirrored_state.closure == pytest.approx(original_state.closure, abs=1e-9)
        assert mirrored_state.yaw_rel == pytest.approx(-original_state.yaw_rel, abs=1e-9)

        # 두 번 반전하면 원래 프레임이다(x 좌표는 1 − (1 − x)의 반올림 오차만큼 다를 수 있다).
        twice = mirror_frame(mirrored, pipeline_spec)
        assert twice.bs == frame.bs and twice.m == frame.m
        for index, (x, y) in frame.lm.items():
            assert twice.lm[index] == (pytest.approx(x, abs=1e-15), y)

    assert mirror_frame(no_face_frame(3, 10, 10), pipeline_spec) == no_face_frame(3, 10, 10)
    extra = frame_from_signals(random_signals(rng, 1), 640, 480, pipeline_spec)
    extra.lm[1] = (0.5, 0.5)  # 짝을 모르는 인덱스
    with pytest.raises(ValueError, match="mirror partner"):
        mirror_frame(extra, pipeline_spec)


def test_layout_check_rejects_other_contour_orders(pipeline_spec: PipelineSpec) -> None:
    lms = pipeline_spec.landmarks
    shuffled = dataclasses.replace(lms, contour_r=lms.contour_r[1:] + lms.contour_r[:1])
    spec = dataclasses.replace(pipeline_spec, landmarks=shuffled)
    with pytest.raises(ValueError, match="contour layout"):
        frame_from_signals(random_signals(random.Random(5), 1), 640, 480, spec)


@pytest.mark.parametrize("w_closure", [0.0, 0.5, 1.0])
def test_lid_signals_invert_the_closure_formula(pipeline_spec: PipelineSpec, w_closure: float) -> None:
    spec = apply_overrides(pipeline_spec, {"closure.w": w_closure})
    for e_open, b_open in ((0.27, 0.22), (0.33, 0.08), (0.2, 0.45)):  # 마지막은 b_span_min이 쓰이는 경우
        for level in (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 1.0):
            ear, blink = lid_signals(level, e_open, b_open, spec)
            closure = calibrate.eye_closure(ear, blink, e_open, b_open, spec.closure)
            assert closure == pytest.approx(level, abs=1e-12)
    assert lid_signals(0.0, 0.3, 0.1, pipeline_spec) == (0.3, 0.1)


# ---------------------------------------------------------------------------
# (b) 각본 시나리오
# ---------------------------------------------------------------------------


def test_scenario_times_regular_jitter_drops_and_gaps() -> None:
    regular = Scenario(segments=(), duration_ms=1000, fps=30.0)
    times = scenario_times(regular)
    assert times[:5] == [0, 33, 67, 100, 133] and len(times) == 30 and times[-1] == 967

    half = Scenario(segments=(), duration_ms=1000, fps=15.0)
    assert scenario_times(half)[:4] == [0, 67, 133, 200]

    gapped = Scenario(segments=(), duration_ms=30_000, fps=10.0, gaps=((5_000, 17_000),))
    times = scenario_times(gapped)
    assert 4_900 in times and 17_000 in times and not any(5_000 <= t < 17_000 for t in times)

    rough = Scenario(segments=(), duration_ms=60_000, fps=30.0, jitter_ms=10, drop_prob=0.2, seed=5)
    times = scenario_times(rough)
    assert times[0] == 0 and all(b > a for a, b in zip(times, times[1:])) and times[-1] < 60_000
    assert 0.7 * 1800 < len(times) < 0.9 * 1800  # 약 20%가 빠진다
    assert any(t != int(k * 1000.0 / 30.0 + 0.5) for k, t in enumerate(times))

    with pytest.raises(ValueError):
        scenario_times(Scenario(segments=(), duration_ms=0))
    with pytest.raises(ValueError):
        scenario_times(Scenario(segments=(), duration_ms=1000, drop_prob=1.0))


def test_scenario_is_deterministic_from_its_seed(pipeline_spec: PipelineSpec) -> None:
    segments = sequence(Open(2_000), Blink(200), Open(1_000), Closure(1_500), Open(500), Yawn(), Nod())
    scenario = Scenario(segments=segments, duration_ms=12_000, fps=24.0, jitter_ms=8, drop_prob=0.1, seed=11)
    first = render(scenario, pipeline_spec)
    assert first == render(scenario, pipeline_spec)
    assert first == render(dataclasses.replace(scenario), pipeline_spec)
    other = render(dataclasses.replace(scenario, seed=12), pipeline_spec)
    assert [f.t_ms for f in other] != [f.t_ms for f in first]
    # 난수가 필요 없는 시나리오는 시드와 무관하다.
    plain = Scenario(segments=segments, duration_ms=12_000, fps=24.0, seed=1)
    assert render(plain, pipeline_spec) == render(dataclasses.replace(plain, seed=2), pipeline_spec)
    assert first[0].t_ms == 1
    assert render(dataclasses.replace(scenario, first_t_ms=987_654), pipeline_spec)[0].t_ms == 987_654


def test_sequence_lays_segments_end_to_end() -> None:
    placed = sequence(Open(1_000), Blink(200), Closure(hold_ms=500), Linger(2_000, 0.4), start_ms=10_000)
    assert [segment.start_ms for segment in placed] == [10_000, 11_000, 11_200, 11_950]
    assert placed[2].duration_ms == 100 + 500 + 150
    assert isinstance(placed[1], Blink) and placed[1].duration_ms == 200


def test_scripted_lid_levels_equal_pipeline_closure(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    # 기본 얼굴은 전역 상수와 같으므로 보정 전(임시)과 보정 후의 closure가 모두 적은 수준과 같다.
    segments = (
        Linger(4_000, 0.4, start_ms=10_000),  # 보정 전
        Linger(4_000, 0.4, start_ms=70_000),  # 보정 후
        Linger(4_000, 0.65, start_ms=80_000),
    )
    frames = render(Scenario(segments=segments, duration_ms=90_000, fps=20.0), pipeline_spec)
    outputs = run_pipeline(frames, pipeline_spec, rules_config)

    def closure_at(session_ms: int) -> tuple[float, bool]:
        out = next(o for o in outputs if o.state.t_ms == 1 + session_ms)
        assert out.state.closure_valid
        return out.state.closure, out.state.provisional

    assert closure_at(12_000) == (pytest.approx(0.4, abs=1e-9), True)
    assert closure_at(72_000) == (pytest.approx(0.4, abs=1e-9), False)
    assert closure_at(82_000) == (pytest.approx(0.65, abs=1e-9), False)
    assert closure_at(5_000) == (pytest.approx(0.0, abs=1e-9), True)
    assert closure_at(10_050)[0] == pytest.approx(0.4 * 50 / 150, abs=1e-9)  # 오르는 구간의 1/3 지점
    # 0.4에 머무는 눈꺼풀은 특징용 검출기(진입 0.5)를 넘지 않고, 0.65는 넘는다.
    runs = [run for out in outputs for run in out.runs]
    assert len(runs) == 1 and runs[0].amplitude == pytest.approx(0.65, abs=1e-9)


def test_scripted_blink_closure_and_face_loss(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    segments = (
        Blink(300, start_ms=5_000),
        Closure(hold_ms=1_500, start_ms=10_000),
        FaceMissing(2_000, start_ms=20_000),
        Closure(hold_ms=600, depth=0.6, start_ms=30_000),
    )
    frames = render(Scenario(segments=segments, duration_ms=35_000, fps=30.0), pipeline_spec)
    assert [f.face for f in frames if 20_000 <= f.t_ms - 1 < 22_000] == [False] * 60
    assert all(f.face for f in frames if not 20_000 <= f.t_ms - 1 < 22_000)

    outputs = run_pipeline(frames, pipeline_spec, rules_config, profile_baseline(pipeline_spec))
    runs = [run for out in outputs for run in out.runs]
    assert len(runs) == 3
    blink, closure, shallow = runs
    # 깜빡임 300 ms(감는 데 120, 뜨는 데 180): 0.5를 넘는 시간은 그 절반인 150 ms다.
    assert blink.start_ms == pytest.approx(1 + 5_060, abs=2) and blink.duration_ms == pytest.approx(150, abs=3)
    assert blink.amplitude > 0.9 and blink.velocity_valid
    # 감김: 0.5를 넘는 시간 = 50(감는 구간의 절반) + 1500 + 75(뜨는 구간의 절반)
    assert closure.start_ms == pytest.approx(1 + 10_050, abs=2) and closure.duration_ms == pytest.approx(1_625, abs=3)
    assert closure.amplitude == pytest.approx(1.0, abs=1e-9) and not closure.truncated
    assert shallow.amplitude == pytest.approx(0.6, abs=1e-9)
    onsets = [e for out in outputs for e in out.events if e.type == EVENT_MICROSLEEP and e.phase == PHASE_ONSET]
    assert len(onsets) == 1 and not onsets[0].provisional  # 1.5초 감김만 마이크로슬립이다


def test_scripted_pose_gaze_yawn_and_nod(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    segments = (
        HeadTurn(4_000, yaw=30.0, start_ms=5_000),
        HeadTurn(4_000, pitch=12.0, roll=5.0, start_ms=12_000),
        LookAside(3_000, gaze_h=0.3, start_ms=20_000),
        LookAside(3_000, gaze_h=-0.2, gaze_v=-0.4, start_ms=25_000),
        Yawn(start_ms=30_000),
        Nod(start_ms=40_000),
    )
    frames = render(Scenario(segments=segments, duration_ms=75_000, fps=30.0), pipeline_spec)
    outputs = run_pipeline(frames, pipeline_spec, rules_config, profile_baseline(pipeline_spec))
    states = {out.state.t_ms - 1: out.state for out in outputs}

    assert states[7_000].yaw_rel == pytest.approx(30.0, abs=1e-9)
    assert states[7_000].pitch_rel == pytest.approx(0.0, abs=1e-9)
    assert states[5_133].yaw_rel == pytest.approx(30.0 * 133 / 300, abs=1e-9)  # 오르는 구간
    assert states[14_000].pitch_rel == pytest.approx(12.0, abs=1e-9)
    rolled = signals.compute(next(f for f in frames if f.t_ms == 14_001), pipeline_spec)
    assert rolled.roll == pytest.approx(5.0, abs=1e-9)
    assert states[21_000].gaze_h_rel == pytest.approx(0.3, abs=1e-12) and states[21_000].gaze_v_rel == 0.0
    assert states[26_000].gaze_h_rel == pytest.approx(-0.2, abs=1e-12)
    assert states[26_000].gaze_v_rel == pytest.approx(-0.4, abs=1e-12)
    assert states[32_000].jaw_rel == pytest.approx(0.6, abs=1e-12)
    assert states[40_500].pitch_rel == pytest.approx(20.0, abs=1e-9)  # 끄덕임의 가장 낮은 지점
    assert states[3_000].yaw_rel == pytest.approx(0.0, abs=1e-9)
    assert states[3_000].jaw_rel == pytest.approx(0.0, abs=1e-12)

    last = [out.steps[-1] for out in outputs if out.steps][-1]
    assert last.k == 74 and last.full_lookback
    assert last.values[F_YAWN_COUNT_60] == 1.0 and last.values[F_NOD_COUNT_60] == 1.0
    assert last.values[F_LONG_CLOSURE_COUNT_60] == 0.0


def test_scripted_profile_changes_the_raw_signals(pipeline_spec: PipelineSpec) -> None:
    profile = dataclasses.replace(
        default_profile(pipeline_spec), e_open_l=0.31, e_open_r=0.29, b_open_l=0.1, b_open_r=0.15,
        yaw=8.0, pitch=-5.0, roll=2.0, eye_look_rest=(0.1,) * 8, eye_w_px=55.0, centre=(0.4, 0.6),
    )  # fmt: skip
    scenario = Scenario(segments=(Closure(1_000, start_ms=1_000),), duration_ms=3_000, w=1280, h=720, profile=profile)
    frames = render(scenario, pipeline_spec)
    opened = signals.compute(frames[0], pipeline_spec)
    assert (opened.ear_l, opened.ear_r) == (pytest.approx(0.31, abs=1e-9), pytest.approx(0.29, abs=1e-9))
    assert (opened.blink_l, opened.blink_r) == (0.1, 0.15) and opened.eye_look == (0.1,) * 8
    assert (opened.yaw, opened.pitch, opened.roll) == (pytest.approx(8.0), pytest.approx(-5.0), pytest.approx(2.0))
    assert opened.eye_w_l == pytest.approx(55.0, abs=1e-9) and (frames[0].w, frames[0].h) == (1280, 720)
    closed = signals.compute(next(f for f in frames if f.t_ms == 1 + 1_500), pipeline_spec)
    rho = pipeline_spec.closure.rho
    assert closed.ear_l == pytest.approx(0.31 * rho, abs=1e-9) and closed.blink_l == pytest.approx(0.6, abs=1e-12)


# ---------------------------------------------------------------------------
# (c) 통계 생성기
# ---------------------------------------------------------------------------


def test_subjects_are_deterministic_and_distinct() -> None:
    assert make_subject(7, 0) == make_subject(7, 0)
    assert make_subject(7, 0) != make_subject(8, 0) and make_subject(7, 0) != make_subject(7, 1)
    for subject in range(1, 40):
        profile = make_subject(subject, 0)
        assert 0.2 < profile.e_open_l < 0.36 and 0.05 < profile.b_open_l < 0.31
        assert profile.b_closed >= 0.6 and profile.ear_closed_ratio <= 0.35  # 완전히 감으면 closure가 1에 닿는다
        assert synth.FPS_MIN <= profile.fps <= synth.FPS_MAX and (profile.w, profile.h) in synth.RESOLUTIONS
        assert profile.eye_w_px >= 12.0  # 쓸 수 없는 눈(6 px 미만)이 되지 않는다


def test_generated_session_is_deterministic_with_valid_timestamps(pipeline_spec: PipelineSpec) -> None:
    profile = make_subject(3, 0)
    frames = list(generate_session(profile, 1, 90_000, 0, pipeline_spec))
    assert frames == list(generate_session(profile, 1, 90_000, 0, pipeline_spec))
    assert frames != list(generate_session(profile, 1, 90_000, 1, pipeline_spec))  # 시드
    assert frames != list(generate_session(profile, 1, 90_000, 0, pipeline_spec, part=2))  # 나뉜 부분은 다른 세션

    times = np.array([f.t_ms for f in frames])
    intervals = np.diff(times)
    assert times[0] == 1 and times[-1] <= 90_000 and np.all(intervals > 0)
    # 프레임 간격은 12~30 fps 범위(33~84 ms)이고, 버려진 프레임 자리만 더 길다.
    assert intervals.min() >= 33 and np.mean(intervals <= 84) > 0.93
    assert all((f.w, f.h) == (profile.w, profile.h) for f in frames)
    face_ratio = np.mean([f.face for f in frames])
    assert 0.7 < face_ratio < 1.0
    with pytest.raises(ValueError):
        next(generate_session(profile, 3, 90_000, 0, pipeline_spec))
    with pytest.raises(ValueError):
        next(generate_session(profile, 0, 0, 0, pipeline_spec))


def test_recording_properties_do_not_depend_on_state(pipeline_spec: PipelineSpec) -> None:
    """해상도와 얼굴 크기는 피험자의 세 세션이 같고, 프레임 간격의 분포는 상태와 무관하다."""
    mean_intervals: list[list[float]] = [[], [], []]
    for subject in range(1, 13):
        profile = make_subject(subject, 0)
        for class_index in range(3):
            frames = list(generate_session(profile, class_index, 60_000, 0, pipeline_spec))
            assert {(f.w, f.h) for f in frames} == {(profile.w, profile.h)}
            widths = {round(signals.compute(f, pipeline_spec).eye_w_l, 6) for f in frames[:50] if f.face}
            assert widths == {round(profile.eye_w_px, 6)}
            mean_intervals[class_index].append(float(np.mean(np.diff([f.t_ms for f in frames]))))
    per_class = np.array(mean_intervals)  # [상태, 피험자]
    # 같은 피험자의 세션들은 같은 명목 프레임률에서 나온다: 상태 사이의 차이가 피험자 사이의 차이보다 훨씬 작다.
    between_states = np.abs(per_class.mean(axis=1) - per_class.mean()).max()
    between_subjects = per_class.mean(axis=0).std()
    assert between_states < 0.25 * between_subjects


def test_states_differ_on_average_but_overlap(pipeline_spec: PipelineSpec, rules_config: RulesConfig) -> None:
    """상태 차이는 평균적으로 뚜렷하고 세션 단위로는 겹친다(합성 데이터의 점검이며 성능 수치가 아니다)."""
    long_runs: list[list[int]] = [[], [], []]
    durations: list[list[float]] = [[], [], []]
    for subject in range(1, 11):
        profile = make_subject(subject, 0)
        for class_index in range(3):
            frames = generate_session(profile, class_index, 150_000, 0, pipeline_spec)
            result = replay_video(frames, pipeline_spec, None, None, rules=rules_config)
            # 거치 각도가 15도를 넘는 세션은 off_axis로 표시될 수 있다. 전역 상수로 대체되는 일은 없어야 한다.
            assert result.baseline is not None and result.baseline.status != BASELINE_FALLBACK_GLOBAL
            runs = [run for run in result.runs if not run.truncated]
            long_runs[class_index].append(sum(run.duration_ms >= 500 for run in runs))
            durations[class_index].append(float(np.median([run.duration_ms for run in runs])))
    mean_long = [float(np.mean(values)) for values in long_runs]
    mean_duration = [float(np.mean(values)) for values in durations]
    assert mean_long[0] < mean_long[1] < mean_long[2] and mean_long[2] > mean_long[0] + 1.0
    assert mean_duration[0] < mean_duration[1] < mean_duration[2]
    # 겹침: 어떤 low_vigilant 세션은 어떤 alert 세션보다 깜빡임이 짧다.
    assert min(durations[1]) < max(durations[0])


# ---------------------------------------------------------------------------
# (d) 데이터셋 쓰기와 명령
# ---------------------------------------------------------------------------


def test_split_subjects_and_session_layout() -> None:
    assert synth.split_subjects(60) == (32, 49) and synth.split_subjects(49) == (32, 49)
    assert synth.split_subjects(48) == (24, 48) and synth.split_subjects(4) == (2, 4)
    assert synth.split_subjects(2) == (1, 2) and synth.split_subjects(1) == (1,)

    jobs = synth.session_jobs(60, 10.0, 0)
    paths = [job.rel_path for job in jobs]
    assert len(jobs) == 182 and len(set(paths)) == 182  # RLDD와 같은 파일 수
    assert paths[:3] == ["Fold1_part1/01/0.parquet", "Fold1_part1/01/5.parquet", "Fold1_part1/01/10.parquet"]
    assert "Fold1_part2/07/0.parquet" in paths and "Fold5_part2/60/10.parquet" in paths
    assert "Fold3_part2/32/10_1.parquet" in paths and "Fold3_part2/32/10_2.parquet" in paths
    assert "Fold5_part1/49/10_2.parquet" in paths and "Fold3_part2/32/10.parquet" not in paths
    assert all(job.duration_ms == 600_000 for job in jobs)
    with pytest.raises(ValueError):
        synth.session_jobs(0, 10.0, 0)
    with pytest.raises(ValueError):
        synth.session_jobs(3, 0.0, 0)


def test_cli_writes_extraction_schema_in_the_rldd_layout(
    tmp_path: Path, config_path: Path, pipeline_spec: PipelineSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "synth"
    argv = ["--config", str(config_path), "--out", str(out), "--subjects", "3", "--minutes", "0.25", "--seed", "4"]
    assert synth.main(argv + ["--workers", "1"]) == 0
    assert capsys.readouterr().out.startswith("SYNTHETIC data (pipeline check only): wrote 11 sessions")

    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.parquet"))
    assert files == [
        "Fold1_part1/01/0.parquet", "Fold1_part1/01/10_1.parquet", "Fold1_part1/01/10_2.parquet",
        "Fold1_part1/01/5.parquet",
        "Fold1_part1/02/0.parquet", "Fold1_part1/02/10.parquet", "Fold1_part1/02/5.parquet",
        "Fold1_part1/03/0.parquet", "Fold1_part1/03/10_1.parquet", "Fold1_part1/03/10_2.parquet",
        "Fold1_part1/03/5.parquet",
    ]  # fmt: skip
    info = json.loads((out / synth.INFO_FILE).read_text(encoding="utf-8"))
    assert (info["subjects"], info["seed"], info["split_subjects"]) == (3, 4, [1, 3]) and "SYNTHETIC" in info["note"]

    # 추출이 쓰는 것과 같은 형식: 같은 열과 자료형, 같은 읽기 함수.
    reference = tmp_path / "reference.parquet"
    hand = frame_from_signals(random_signals(random.Random(0), 1), 640, 480, pipeline_spec)
    write_raw_frames([hand], reference, pipeline_spec)
    drowsy_part = out / "Fold1_part1/03/10_2.parquet"
    assert pq.read_schema(drowsy_part).remove_metadata() == pq.read_schema(reference).remove_metadata()
    metadata = read_extract_metadata(drowsy_part)
    assert metadata["synthetic"] is True and (metadata["subject"], metadata["label"], metadata["part"]) == (3, 10, 2)
    expected = list(generate_session(make_subject(3, 4), 2, 15_000, 4, pipeline_spec, part=2))
    assert list(read_raw_frames(drowsy_part)) == expected and expected[0].t_ms == 1

    manifest = build_manifest(out)
    assert len(manifest) == 11 and manifest["video_id"].nunique() == 9
    assert manifest.loc[manifest["subject"] == 3, "session_id"].tolist() == ["S03_0", "S03_5", "S03_10_1", "S03_10_2"]

    # 같은 폴더에 다시 쓰지 않는다(이전 실행의 파일이 섞이지 않게).
    assert synth.main(argv) == 2
    assert "already holds parquet files" in capsys.readouterr().err
    no_config = ["--config", str(tmp_path / "missing.yaml"), "--out", str(tmp_path / "x"), "--subjects", "1"]
    assert synth.main(no_config) == 2
    assert synth.main(["--config", str(config_path), "--out", str(tmp_path / "y"), "--subjects", "0"]) == 2
    capsys.readouterr()


def test_frame_drop_simulated_dataset_is_a_thinned_copy(
    tmp_path: Path, config_path: Path, pipeline_spec: PipelineSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    common = ["--config", str(config_path), "--subjects", "2", "--minutes", "0.5", "--seed", "1", "--workers", "1"]
    assert synth.main(common + ["--out", str(tmp_path / "native")]) == 0
    assert synth.main(common + ["--out", str(tmp_path / "fps15"), "--fps-sim", "15"]) == 0
    assert "device frame drops simulated at about 15 fps" in capsys.readouterr().out

    drop_sim = load_config(config_path).extraction.drop_sim
    for rel_path in ("Fold1_part1/01/0.parquet", "Fold1_part1/02/10_2.parquet"):
        native = list(read_raw_frames(tmp_path / "native" / rel_path))
        thinned = list(read_raw_frames(tmp_path / "fps15" / rel_path))
        # 추출의 누락 모사와 같은 흐름 제한기, 같은 지연 수열로 고른 프레임이다.
        latencies = list(itertools.islice(drop_sim_latencies(drop_sim), len(native)))
        kept = flow_limiter_select([frame.t_ms for frame in native], latencies)
        assert thinned == [native[index] for index in kept]
        assert len(thinned) < len(native) and thinned[0] == native[0]
        mean_fps = 1000.0 * (len(thinned) - 1) / (thinned[-1].t_ms - thinned[0].t_ms)
        assert mean_fps <= 15.5
        metadata = read_extract_metadata(tmp_path / "fps15" / rel_path)
        assert metadata["drop_sim"] == {"latency_ms_mean": 66.7, "latency_ms_std": 8.0, "seed": 0}
        assert metadata["synthetic"] is True
        assert read_extract_metadata(tmp_path / "native" / rel_path)["drop_sim"] is None
    assert synth.main(common + ["--out", str(tmp_path / "bad"), "--fps-sim", "0"]) == 2
    assert "--fps-sim must be positive" in capsys.readouterr().err


def test_dataset_does_not_depend_on_worker_count(tmp_path: Path, pipeline_spec: PipelineSpec) -> None:
    serial = synth.write_dataset(tmp_path / "serial", 2, 0.2, 9, pipeline_spec, workers=1)
    parallel = synth.write_dataset(tmp_path / "parallel", 2, 0.2, 9, pipeline_spec, workers=2)
    assert serial == parallel and len(serial) == 8
    for rel_path in serial:
        assert list(read_raw_frames(tmp_path / "serial" / rel_path)) == list(
            read_raw_frames(tmp_path / "parallel" / rel_path)
        )
    with pytest.raises(ValueError, match="workers"):
        synth.write_dataset(tmp_path / "bad", 1, 0.2, 9, pipeline_spec, workers=0)
