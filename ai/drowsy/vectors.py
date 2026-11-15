"""테스트 벡터 생성과 재검증 (PLAN 10.2절, 10.1절의 규칙 경계값). 이식 대상이 아니다.

    python -m drowsy.vectors generate --out tests/vectors
    python -m drowsy.vectors verify --dir tests/vectors

벡터 하나는 JSON 파일 하나다. 입력(프레임, 넣어 준 기준선, 규칙 설정 변경, 차량 속도, 모델)과 참조 구현이
그 입력에서 낸 출력(expected)을 함께 담는다. Android 팀은 같은 입력을 Kotlin 이식 코드에 넣어 expected와
비교하고, 저장소의 pytest는 expected를 다시 계산해 저장된 값과 비교한다(참조 구현이 바뀌면 바로 드러난다).

코드 경로는 하나다(PLAN 3.1절). 입력은 `drowsy.synth`의 각본 시나리오나 같은 모듈의 프레임 만들기 함수로 손으로 놓은
프레임(`_hand_frames`)이고, expected는 이식 대상 모듈을 그대로 불러서 얻는다. 신호 묶음은 `signals.compute`, 모델 묶음은 onnxruntime, 나머지는 모두 `DriverPipeline`
(`BaselineEstimator`, `frame_state`, `ClosureRunDetector`, `StepAggregator`, `RuleEngine`, `AlertPolicy`를 잇는
그 클래스)이다. 이 모듈에는 신호·closure·감김 검출·스텝 특징·규칙의 계산이 없다.

파일 형식 (PLAN 10.2절의 형식에 설명·허용 오차·모델 항목을 더한 것)
    name              파일 이름(확장자 제외)
    bundle            signals | baseline | runs | steps | rules | model | e2e
    description       이 벡터가 고정하는 것(한국어)
    pipeline_hash     만든 시점의 config/feature_spec.json pipeline_hash
    rules_hash        만든 시점의 규칙 설정 해시(config_overrides를 적용하기 전의 값)
    tolerances        expected의 항목별 허용 오차 (PLAN 10.2절 표)
    baseline          넣어 준 기준선(`Baseline`의 필드) 또는 null. null이면 세션 앞부분에서 보정한다
    config_overrides  규칙 설정에 덮어쓸 값. 점으로 이은 경로 -> 값 (`spec.apply_overrides`)
    model             null | {"kind": "stub", "probs": [[3] ...]} | {"kind": "onnx", "file": ..., "sha256": ...}
    frames            [{t_ms, face, w, h, lm: {인덱스: [x, y]}, bs: {이름: 값}, m: [16]}] (얼굴이 없으면 lm {}, bs {}, m [])
    speed             [[t_ms, kmh] ...]. t_ms 이상인 프레임부터 그 속도다. 첫 항목 전의 프레임은 속도 입력이 없다(None)
    model_inputs      (모델 묶음만) 원시 스텝 특징 [120][17]의 목록. features[0]이 가장 오래된 스텝이다
    expected          아래 항목 중 이 벡터에 해당하는 것
        signals       프레임마다 `FrameSignals`의 필드 (홍채 비율이 없으면 null)
        closure       받아들인 프레임마다 `FrameState`의 필드
        baseline      세션에서 확정된 기준선과 그것을 처음 쓴 프레임의 t_ms. 확정되지 않았으면 null
        runs          특징용 감김 구간(`ClosureRun`의 필드)과 그것을 발행한 프레임의 t_ms
        steps         스텝(`StepFeatures`의 필드)과 그것을 발행시킨 프레임의 t_ms
        model_inputs  모델을 처음 부른 창: {t_ms, k, features [120][17]}. features[0]이 가장 오래된 스텝이다
        step_probs    발행된 스텝마다 {t_ms, k, probs [3] 또는 null}. `PipelineOutput.step_probs`를 스텝 순서대로 편 것이다.
                      t_ms는 그 스텝을 발행시킨 프레임, probs는 그 스텝에서 모델을 불렀으면 확률, 아니면 null이다.
                      프레임 하나가 스텝을 여럿 발행했으면(프레임 끊김) 같은 t_ms의 원소가 여럿이고 저마다 확률을 가질 수 있다
        probs         마지막 스텝에 확률이 있는 프레임마다 {t_ms, k, probs [3]}. `PipelineOutput.probs`(그 프레임이 발행시킨
                      마지막 스텝의 확률)이고 k는 그 스텝의 번호다 (모델 묶음에서는 {input, probs})
        events        상태 이벤트(`DriverStateEvent`의 필드)와 그것을 발행한 프레임의 t_ms, 발행 순서
        alerts        경고(`AlertPolicy`의 출력), 같은 모양
        counters      `DriverPipeline`이 센 값(받아들인 프레임, 버린 프레임, 스텝, 모델 호출, 건너뛴 창)

가정: 기준선을 넣어 줄 때 특징 누적을 시작할 세션 시간은 `baseline.t_complete_ms`로 전한다(별도 항목을 두지 않는다).
가정: config_overrides는 규칙 설정(`RulesConfig`)에만 적용한다. `pipeline` 블록의 상수는 pipeline_hash로 고정된
  값이므로 벡터에서 바꾸지 않는다.
가정: 스텁 모델은 i번째 호출(0부터)에 probs[i]를 돌려주고, 목록이 끝나면 마지막 값을 되풀이한다.
가정: 모델을 스텝마다 부르게 되면서(2026-11-13 소유자 결정) 스텝별 확률을 `step_probs` 항목으로 따로 적는다.
  `probs` 항목의 뜻과 모양은 그대로 둔다(프레임의 마지막 스텝). 프레임이 스텝을 하나씩만 발행하는 벡터에서는
  `step_probs`에서 null이 아닌 원소를 모은 것이 `probs`와 같다.
가정: 경계값 벡터의 일부는 각본 구간 대신 프레임의 시각과 값을 직접 정해서 만든다(`_hand_frames`). 정수 ms의 경계
  (정확히 250 ms, 1000 ms, 10000 ms, 10001 ms)에 프레임을 놓으려는 것이다. 값은 계단 모양으로 바뀐다.
가정: 실수로 계산되는 값의 임계(closure, 각도, 눈동자 합성값)는 등호 자리를 벡터로 고정하지 않는다. 그 자리의 표본은
  수학 함수의 마지막 자리 차이로 갈리므로 두지 않고(아래), 임계의 바로 안쪽과 바깥쪽 값으로 양쪽을 고정한다.
  모델 확률의 종료 임계(0.4)도 같다: 모델 출력은 float32이고 0.4는 float32로 정확히 적을 수 없어서(0.4보다 큰 값이
  된다), 스텁이 Double을 돌려주는 구현과 Float를 돌려주는 구현이 "이하"의 등호 자리에서 갈린다. 등호를 고정하는 것은
  정수 ms의 비교, 입력으로 그대로 들어오는 값(차량 속도 20.0, 50.0), 정수 ms의 몫으로 정확히 나오는 값(perclos_60
  0.15, 0.10), 그리고 시작 임계와 같은 p_drowsy 0.6이다(0.6을 float32로 적으면 0.6보다 커서 어느 구현에서든 시작
  조건이므로 벡터에 둬도 옳은 구현이 떨어지지 않는다).
가정: 긴 세션(기준선, 스텝 특징, 졸음 판정, 전 과정)은 파일 크기를 줄이려고 3~10 fps의 프레임을 쓴다. 시간 누적은
  프레임률과 무관하게 정의돼 있고(프레임 간격에서 `clock.max_hold_ms`(200 ms)를 넘는 부분만 "데이터 없음"이다),
  프레임률에 따른 감김 검출은 감김 검출 묶음(30 fps, 15 fps, 누락)이 따로 고정한다.
가정: 이벤트의 confidence 허용 오차(절대 1e-4)는 계획서에 없어서 모델 확률의 허용 오차와 같게 정했다.
가정: 모델 묶음의 입력은 배정밀도로 적고, 텐서에 넣을 때 float32로 바꾼다(앱이 스텝 특징을 넣는 방식과 같다).
가정: 생성 결과가 바이트 단위로 같은 것은 같은 환경(같은 수학 라이브러리와 onnxruntime)에서의 일이다.
가정: 옳게 이식한 코드가 다른 기기에서도 같은 결과를 내도록, 판정이 갈리는 자리에 걸친 값은 벡터에 두지 않는다.
  판정 임계값에서 1e-6 안에 있는 표본이 생기는 각본은 만들 때 거부하고(`_threshold_problem`), 보간한 시각이
  반올림 경계(x.5 ms)에 붙지 않는지는 tests/test_vectors.py가 확인한다. 경계값 벡터는 정수 ms의 비교로 경계를 잡는다
  (예: 감김 시작에서 정확히 3000 ms 뒤의 프레임).
가정: 기본 설정에서는 드러나지 않는 동작을 고정하려고 규칙 설정을 덮어쓴 벡터가 있다. `rules_gap_over_10s_drowsy_model`은
  `drowsiness.model.refractory_ms`를 600000으로 둔다(10초 넘는 끊김 뒤에 모델 창이 다시 차려면 179초가 걸려서, 기본
  60초로는 재시작 금지 기억이 지워졌는지 드러나지 않는다). 덮어쓴 값은 그 벡터 안에서만 쓰인다.
가정: `runs_two_humps`의 "같은 최댓값이 다시 나온 표본"은 같은 내용의 프레임을 두 번 쓴 것이라 어느 구현에서든 두 표본의
  closure가 같다. 임계값과의 비교가 아니므로 위의 "판정이 갈리는 자리에 값을 두지 않는다"에 걸리지 않는다.
가정: sleep의 end가 미뤄 둔 microsleep 경고를 지우는 규칙은 벡터에 없다. 규칙 엔진은 sleep의 onset을 먼저 내고 그때 이미
  지워지므로 엔진을 거친 이벤트로는 그 자리를 만들 수 없다. 대신 다른 종류의 end가 그 경고를 지우지 않는다는 것을
  `rules_alert_pending_survives_end`가, 다른 종류의 onset도 지우지 않는다는 것을 `rules_alert_pending_survives_onset`이
  고정한다.
가정: 첫 t_ms가 작은 세션(`rules_unable_small_t0`)의 첫 t_ms는 5000으로 정했다. 0보다 크고(벡터의 t_ms는 양수다)
  `unable_to_analyze.window_ms`(10000)보다 작아서, "처음 10초"를 첫 프레임이 아니라 t_ms 0부터 재는 구현이 5초 일찍
  판정을 시작하는 값이다. 첫 t_ms가 1인 다른 벡터에서는 그 차이가 1 ms라 드러나지 않는다.
가정: 끄덕임의 느린 평균을 고정하는 벡터(`steps_nod_slow_mean`)는 끄덕임이 시작되는 프레임의 d = pitch_rel − m이 시작
  임계 10도에서 0.028도 위에 오도록 pitch(25.25도)를 골랐다. m의 갱신 속도가 1%만 달라도(α = dt/τ로 구한 경우 d가
  9.984) 끄덕임이 시작되지 않게 하려는 것이다. 등호 자리가 아니고 수학 함수의 마지막 자리 차이(1e-12 수준)보다 훨씬
  멀어서 옳게 이식한 코드는 어느 기기에서든 같은 쪽에 놓인다. d는 저장되는 값이 아니라서 `_threshold_problem`이 보는
  값에 들지 않는다. 손으로 계산한 값은 tests/test_vectors.py에 있다.
가정: 손으로 놓은 보정 세션 셋(`baseline_odd_sample_count`, `baseline_burst_frames`, `baseline_exactly_300_frames`)은
  신호를 몇 가지 값으로만 둔다(분위수와 중앙값을 손으로 셀 수 있게). `baseline_burst_frames`의 얼굴 검출 비율 0.5는 정수
  ms의 몫(45000 / 90000)으로 정확히 나오는 값이라 등호를 고정하고, `baseline_exactly_300_frames`는 프레임 수의 등호
  (정확히 300개)를 고정한다. E의 범위 끝(0.12, 0.45)의 등호는 실수 임계라 벡터로 고정하지 않았다(단위 테스트가 고정한다).
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import functools
import json
import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from drowsy import signals, synth
from drowsy.pipeline import DriverPipeline, ModelFn
from drowsy.spec import (
    DEFAULT_FEATURE_SPEC_PATH,
    DEFAULT_RULES_PATH,
    FeatureSpec,
    PipelineSpec,
    RulesConfig,
    SpecError,
    apply_overrides,
    file_sha256,
    load_feature_spec,
    load_rules,
    rules_hash,
)
from drowsy.synth import (
    Blink,
    Closure,
    FaceMissing,
    FaceProfile,
    HeadTurn,
    Linger,
    LookAside,
    Nod,
    Open,
    Scenario,
    Segment,
    Yawn,
)
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OFF_AXIS,
    BASELINE_OK,
    CLASS_DROWSY,
    EL_DOWN_L,
    EL_DOWN_R,
    EL_IN_L,
    EL_IN_R,
    EL_OUT_L,
    EL_OUT_R,
    EL_UP_L,
    EL_UP_R,
    EVENT_DROWSY,
    EVENT_GAZE_AWAY_LONG,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
    EVENT_VATS,
    F_BLINK_DUR_MEAN_60,
    F_BLINK_DUR_STD_60,
    F_BLINK_RATE_60,
    F_CLOSURE_MEAN_10,
    F_CLOSURE_MEAN_60,
    F_FACE_MISSING_RATIO_10,
    F_FACE_MISSING_RATIO_60,
    F_LONG_CLOSURE_COUNT_60,
    F_MAX_CLOSURE_MS_10,
    F_MAX_CLOSURE_MS_60,
    F_NOD_COUNT_60,
    F_PERCLOS_10,
    F_PERCLOS_60,
    F_PITCH_STD_60,
    F_YAWN_COUNT_60,
    FEATURE_NAMES,
    IDENTITY_R0,
    PHASE_ONSET,
    SOURCE_EYE,
    SOURCE_HEAD_POSE,
    SOURCE_MODEL,
    SOURCE_PERCLOS,
    Baseline,
    ClosureRun,
    DriverStateEvent,
    FrameSignals,
    FrameState,
    RawFrame,
    StepFeatures,
)

# ---------------------------------------------------------------------------
# 이름과 상수
# ---------------------------------------------------------------------------

BUNDLE_SIGNALS = "signals"
BUNDLE_BASELINE = "baseline"
BUNDLE_RUNS = "runs"
BUNDLE_STEPS = "steps"
BUNDLE_RULES = "rules"
BUNDLE_MODEL = "model"
BUNDLE_E2E = "e2e"
BUNDLES: tuple[str, ...] = (
    BUNDLE_SIGNALS,
    BUNDLE_BASELINE,
    BUNDLE_RUNS,
    BUNDLE_STEPS,
    BUNDLE_RULES,
    BUNDLE_MODEL,
    BUNDLE_E2E,
)

SECTION_SIGNALS = "signals"
SECTION_CLOSURE = "closure"
SECTION_BASELINE = "baseline"
SECTION_RUNS = "runs"
SECTION_STEPS = "steps"
SECTION_MODEL_INPUTS = "model_inputs"
SECTION_STEP_PROBS = "step_probs"
SECTION_PROBS = "probs"
SECTION_EVENTS = "events"
SECTION_ALERTS = "alerts"
SECTION_COUNTERS = "counters"

# 항목별 허용 오차 (PLAN 10.2절 표). 실수는 |실제 − 기대| ≤ abs + rel·|기대|, 시각은 ±time_abs_ms로 비교한다.
# 여기에 없는 값(종류, 단계, 순서, 불리언, 스텝 번호, 발행 프레임의 t_ms, counters)은 정확히 같아야 한다.
TOLERANCES: dict[str, dict[str, float]] = {
    SECTION_SIGNALS: {"abs": 1e-6, "angle_abs_deg": 1e-3},
    SECTION_CLOSURE: {"abs": 1e-6},
    SECTION_BASELINE: {"abs": 1e-6},
    SECTION_RUNS: {"time_abs_ms": 1, "rel": 1e-4},
    SECTION_STEPS: {"abs": 1e-5, "rel": 1e-5},
    SECTION_MODEL_INPUTS: {"abs": 1e-5, "rel": 1e-5},
    SECTION_STEP_PROBS: {"abs": 1e-4},
    SECTION_PROBS: {"abs": 1e-4},
    SECTION_EVENTS: {"time_abs_ms": 1, "confidence_abs": 1e-4},
    SECTION_ALERTS: {"time_abs_ms": 1, "confidence_abs": 1e-4},
}
_ANGLE_KEYS = frozenset({"yaw", "pitch", "roll"})
_TIME_KEYS: dict[str, frozenset[str]] = {
    SECTION_RUNS: frozenset({"start_ms", "end_ms", "duration_ms"}),
    SECTION_EVENTS: frozenset({"t_start_ms", "duration_ms"}),
    SECTION_ALERTS: frozenset({"t_start_ms", "duration_ms"}),
}

MODEL_KIND_STUB = "stub"
MODEL_KIND_ONNX = "onnx"
DEFAULT_MODEL_PATH = "export/drowsiness_v0.onnx"
DEFAULT_MODEL_DIR = "export"

# MediaPipe 공식 기대 출력 파일의 변환 행렬(packed_data, 열 우선). 기대 각도는 약 2.24, 6.32, 0.99도 (PLAN 2.2절).
OFFICIAL_MATRIX: tuple[float, ...] = (
    0.9995292, 0.01715696, -0.03715533, 0.0,
    -0.01294756, 0.9937692, 0.11070588, 0.0,
    0.038823195, -0.1101321, 0.99315894, 0.0,
    -0.3691378, 22.75809, -65.765925, 1.0,
)  # fmt: skip
OFFICIAL_ANGLES_DEG: tuple[float, float, float] = (2.24, 6.32, 0.99)

# 첫 t_ms가 큰 임의 값인 세션의 시작 시각. 32비트 정수 범위를 넘는다(Kotlin에서 Long이어야 한다).
LARGE_FIRST_T_MS = 1_731_465_123_457
# 첫 t_ms가 작은 세션의 시작 시각. 0보다 크고 `unable_to_analyze.window_ms`(10000)보다 작다(까닭은 모듈 설명에 있다).
SMALL_FIRST_T_MS = 5000

# 벡터 이름. 묶음별로 적는다. `build_vectors`가 만든 이름과 같아야 한다.
VECTOR_NAMES: tuple[str, ...] = (
    # 신호
    "signals_ear_square",
    "signals_ear_non_square",
    "signals_official_matrix",
    "signals_mirrored_matrix",
    "signals_random_matrices",
    "signals_one_eye_closed",
    "signals_zero_width_eye",
    "signals_no_face",
    "signals_iris_tilted_eyes",
    # 기준선과 closure
    "baseline_normal",
    "baseline_extended",
    "baseline_fallback",
    "closure_yaw_weighting",
    "closure_no_face",
    "baseline_odd_sample_count",
    "baseline_burst_frames",
    "baseline_exactly_300_frames",
    "closure_blink_span_floor",
    # 감김 검출
    "runs_blinks_30fps",
    "runs_blinks_15fps",
    "runs_dropped_frames",
    "runs_one_sample_blink",
    "runs_closure_600ms",
    "runs_truncated_by_gap",
    "runs_duplicate_timestamp",
    "runs_lingering_lids",
    "runs_velocity_after_gap",
    "runs_rounding_large_t0",
    "runs_multi_span",
    "runs_multi_span_truncated",
    "runs_two_humps",
    "runs_one_sample_truncated",
    "runs_two_rises_before_peak",
    # 스텝 특징
    "steps_five_minutes",
    "steps_five_minutes_large_t0",
    "steps_gap_over_10s",
    "steps_multi_span_runs",
    "steps_closure_across_gap",
    "steps_zero_division",
    "steps_gap_exactly_10s",
    "steps_yawn_edges",
    "steps_nod_edges",
    "steps_nod_slow_mean",
    "steps_nod_after_long_gap",
    "steps_two_closures_in_one_step",
    "steps_closure_above_cap",
    "steps_closure_exactly_500ms",
    "steps_pitch_std_at_5000ms",
    # 규칙
    "rules_closure_900ms",
    "rules_closure_1100ms",
    "rules_closure_2500ms",
    "rules_closure_3000ms",
    "rules_closure_6000ms",
    "rules_gaze_2900ms",
    "rules_gaze_3100ms",
    "rules_gaze_eye_only",
    "rules_gaze_face_lost_in_turn",
    "rules_vats",
    "rules_drowsy_model_stub",
    "rules_drowsy_perclos",
    "rules_calibrating_provisional",
    "rules_unable_to_analyze",
    "rules_speed_gates",
    "rules_cooldown",
    "rules_sleep_repeat",
    "rules_closure_dwell_below_1s",
    "rules_closure_dwell_reaches_1s",
    "rules_closure_three_spans",
    "rules_alert_deferred_cooldown",
    "rules_alert_blocked_microsleep_sleep",
    "rules_drowsy_perclos_exact",
    "rules_drowsy_model_stall",
    "rules_gap_over_10s",
    "rules_gap_exactly_10s",
    "rules_gap_over_10s_drowsy",
    "rules_gap_unable_boundary",
    "rules_unable_sparse_frames",
    "rules_gaze_credit_window_missed",
    "rules_gaze_credit_boundary",
    "rules_gaze_debounce_250ms",
    "rules_forward_zone_edges",
    "rules_vats_reset_enabled",
    "rules_alert_priority",
    "rules_gap_over_10s_candidate",
    "rules_gaze_candidate_dropped",
    "rules_forward_zone_eye_edges",
    "rules_gaze_unable_pauses_timer",
    "rules_drowsy_either",
    "rules_gap_over_10s_drowsy_model",
    "rules_alert_pending_survives_end",
    "rules_closure_after_dropped_run",
    "rules_gaze_glance_and_credit_times",
    "rules_calibrating_large_t0",
    "rules_vats_window_edge",
    "rules_vats_confidence_face_lost",
    "rules_gaze_one_frame_glances",
    "rules_microsleep_confidence_face_lost",
    "rules_alert_pending_survives_onset",
    "rules_unable_small_t0",
    "rules_drowsy_off_step_boundary",
    # 전 과정
    "e2e_250s",
    # 모델
    "model_v0_inputs",
)
E2E_NAME = "e2e_250s"


class VectorError(ValueError):
    """벡터를 만들 수 없거나(각본이 의도한 경우를 내지 못함) 파일이 형식에 맞지 않을 때."""


# ---------------------------------------------------------------------------
# 값 <-> JSON
# ---------------------------------------------------------------------------


def frame_to_json(frame: RawFrame) -> dict[str, Any]:
    """`RawFrame`을 벡터의 frames 항목 하나로 바꾼다. 랜드마크는 인덱스 순, 블렌드셰이프는 이름 순으로 적는다."""
    return {
        "t_ms": frame.t_ms,
        "face": frame.face,
        "w": frame.w,
        "h": frame.h,
        "lm": {str(index): [frame.lm[index][0], frame.lm[index][1]] for index in sorted(frame.lm)},
        "bs": {name: frame.bs[name] for name in sorted(frame.bs)},
        "m": list(frame.m),
    }


def frame_from_json(data: Mapping[str, Any]) -> RawFrame:
    """벡터의 frames 항목 하나를 `RawFrame`으로 되돌린다."""
    return RawFrame(
        t_ms=int(data["t_ms"]),
        face=bool(data["face"]),
        w=int(data["w"]),
        h=int(data["h"]),
        lm={int(index): (float(xy[0]), float(xy[1])) for index, xy in data["lm"].items()},
        bs={str(name): float(value) for name, value in data["bs"].items()},
        m=tuple(float(value) for value in data["m"]),
    )


def baseline_to_json(baseline: Baseline) -> dict[str, Any]:
    out = dataclasses.asdict(baseline)
    out["r0"] = list(baseline.r0)
    return out


def baseline_from_json(data: Mapping[str, Any]) -> Baseline:
    return Baseline(
        e_l=float(data["e_l"]),
        e_r=float(data["e_r"]),
        b_l=float(data["b_l"]),
        b_r=float(data["b_r"]),
        r0=tuple(float(value) for value in data["r0"]),
        jaw0=float(data["jaw0"]),
        gaze_h0=float(data["gaze_h0"]),
        gaze_v0=float(data["gaze_v0"]),
        t_complete_ms=int(data["t_complete_ms"]),
        status=str(data["status"]),
    )


def _signals_to_json(s: FrameSignals) -> dict[str, Any]:
    out = dataclasses.asdict(s)
    out["eye_look"] = list(s.eye_look)
    return out


def _state_to_json(state: FrameState) -> dict[str, Any]:
    return dataclasses.asdict(state)


def _run_to_json(t_ms: int, run: ClosureRun) -> dict[str, Any]:
    return {"t_ms": t_ms, **dataclasses.asdict(run)}


def _step_to_json(t_ms: int, step: StepFeatures) -> dict[str, Any]:
    return {
        "t_ms": t_ms,
        "k": step.k,
        "t_end_ms": step.t_end_ms,
        "full_lookback": step.full_lookback,
        "values": list(step.values),
    }


def _event_to_json(t_ms: int, event: DriverStateEvent) -> dict[str, Any]:
    return {"t_ms": t_ms, **dataclasses.asdict(event)}


# ---------------------------------------------------------------------------
# 실행 환경
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Context:
    """벡터를 만들고 검증할 때 쓰는 설정.

    - spec, pipeline_hash: config/feature_spec.json의 `pipeline` 블록과 그 해시.
    - rules, rules_hash: config/rules.yaml과 그 해시(`spec.rules_hash`, 앱에 내보내는 rules.json 바이트의 sha256).
    - window_steps, n_features: 모델 입력의 모양(120, 17).
    - model_dir: 벡터의 `model.file`을 찾는 폴더.
    """

    spec: PipelineSpec
    pipeline_hash: str
    rules: RulesConfig
    rules_hash: str
    window_steps: int
    n_features: int
    model_dir: Path


def load_context(
    spec_path: str | Path = DEFAULT_FEATURE_SPEC_PATH,
    rules_path: str | Path = DEFAULT_RULES_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> Context:
    """설정 파일을 읽어 `Context`를 만든다."""
    feature_spec: FeatureSpec = load_feature_spec(spec_path)
    rules = load_rules(rules_path)
    return Context(
        spec=feature_spec.pipeline,
        pipeline_hash=feature_spec.pipeline_hash,
        rules=rules,
        rules_hash=rules_hash(rules),
        window_steps=feature_spec.window_steps,
        n_features=feature_spec.n_features,
        model_dir=Path(model_dir),
    )


@functools.lru_cache(maxsize=4)
def _session(model_path: str) -> ort.InferenceSession:
    """onnxruntime CPU 세션. 스레드를 1개로 고정해 같은 입력에 같은 출력이 나오게 한다."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])


def _onnx_probs(model_path: Path, window: Sequence[Sequence[float]], ctx: Context) -> list[float]:
    """원시 스텝 특징 창 하나(가장 오래된 스텝이 먼저)를 모델에 넣어 확률 3개를 얻는다."""
    session = _session(str(model_path))
    tensor = np.asarray(window, dtype=np.float32).reshape(1, ctx.window_steps, ctx.n_features)
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return [float(value) for value in output[0]]


def _model_path(model: Mapping[str, Any], ctx: Context) -> Path:
    return ctx.model_dir / str(model["file"])


def _model_fn(model: Mapping[str, Any] | None, ctx: Context) -> ModelFn | None:
    """벡터의 model 항목에서 `DriverPipeline`에 넣을 모델 함수를 만든다."""
    if model is None:
        return None
    kind = model.get("kind")
    if kind == MODEL_KIND_STUB:
        scripted = [[float(value) for value in probs] for probs in model["probs"]]
        if len(scripted) == 0:
            raise VectorError("stub model needs at least one probability triple")
        calls = [0]

        def stub(window: Sequence[Sequence[float]]) -> Sequence[float]:
            index = min(calls[0], len(scripted) - 1)
            calls[0] += 1
            return scripted[index]

        return stub
    if kind == MODEL_KIND_ONNX:
        path = _model_path(model, ctx)

        def onnx_model(window: Sequence[Sequence[float]]) -> Sequence[float]:
            return _onnx_probs(path, window, ctx)

        return onnx_model
    raise VectorError(f"unknown model kind: {kind!r}")


# ---------------------------------------------------------------------------
# expected 계산
# ---------------------------------------------------------------------------


def compute_expected(vector: Mapping[str, Any], ctx: Context) -> dict[str, Any]:
    """벡터의 입력에서 참조 구현의 출력을 계산한다. 그 묶음에서 계산할 수 있는 항목을 모두 돌려준다.

    생성과 검증이 이 함수 하나를 쓴다. 입력은 JSON에서 읽은 모양 그대로 받는다.
    """
    bundle = vector["bundle"]
    if bundle == BUNDLE_SIGNALS:
        frames = [frame_from_json(item) for item in vector["frames"]]
        return {SECTION_SIGNALS: [_signals_to_json(signals.compute(frame, ctx.spec)) for frame in frames]}
    if bundle == BUNDLE_MODEL:
        path = _model_path(vector["model"], ctx)
        return {
            SECTION_PROBS: [
                {"input": index, "probs": _onnx_probs(path, window, ctx)}
                for index, window in enumerate(vector["model_inputs"])
            ]
        }
    if bundle in BUNDLES:
        return _run_pipeline(vector, ctx)
    raise VectorError(f"unknown bundle: {bundle!r}")


def _run_pipeline(vector: Mapping[str, Any], ctx: Context) -> dict[str, Any]:
    """프레임을 `DriverPipeline`에 차례로 넣고 나온 것을 항목별로 모은다."""
    rules = ctx.rules
    if len(vector["config_overrides"]) > 0:
        rules = apply_overrides(rules, vector["config_overrides"])
    injected = None if vector["baseline"] is None else baseline_from_json(vector["baseline"])

    model = _model_fn(vector["model"], ctx)
    windows: list[list[list[float]]] = []
    recording: ModelFn | None = None
    if model is not None:
        inner = model

        def recording_model(window: Sequence[Sequence[float]]) -> Sequence[float]:
            if len(windows) == 0:  # 처음 부른 창만 적어 둔다
                windows.append([list(row) for row in window])
            return inner(window)

        recording = recording_model

    pipe = DriverPipeline(ctx.spec, rules, injected, None, recording)
    speed_times = [int(item[0]) for item in vector["speed"]]
    speed_values = [float(item[1]) for item in vector["speed"]]
    if speed_times != sorted(speed_times):
        raise VectorError("speed entries must be sorted by t_ms")

    closure: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    model_inputs: list[dict[str, Any]] = []
    step_probs: list[dict[str, Any]] = []
    probs: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None

    for item in vector["frames"]:
        frame = frame_from_json(item)
        speed: float | None = None
        position = bisect.bisect_right(speed_times, frame.t_ms) - 1
        if position >= 0:
            speed = speed_values[position]
        out = pipe.update(frame, speed)
        if out is None:
            continue
        t_ms = frame.t_ms
        if injected is None and baseline is None and pipe.baseline is not None:
            baseline = {"t_ms": t_ms, **baseline_to_json(pipe.baseline)}
        closure.append(_state_to_json(out.state))
        runs.extend(_run_to_json(t_ms, run) for run in out.runs)
        steps.extend(_step_to_json(t_ms, step) for step in out.steps)
        # 모델은 스텝마다 불린다. 스텝별 확률은 step_probs에, 프레임의 마지막 스텝의 확률은 probs에 적는다.
        for step, step_p in zip(out.steps, out.step_probs, strict=True):
            step_probs.append({"t_ms": t_ms, "k": step.k, "probs": None if step_p is None else list(step_p)})
            if step_p is not None and len(model_inputs) == 0:  # 처음 부른 창은 확률을 처음 받은 스텝의 것이다
                model_inputs.append({"t_ms": t_ms, "k": step.k, "features": windows[0]})
        if out.probs is not None:
            probs.append({"t_ms": t_ms, "k": out.steps[-1].k, "probs": list(out.probs)})
        events.extend(_event_to_json(t_ms, event) for event in out.events)
        alerts.extend(_event_to_json(t_ms, event) for event in out.alerts)

    return {
        SECTION_CLOSURE: closure,
        SECTION_BASELINE: baseline,
        SECTION_RUNS: runs,
        SECTION_STEPS: steps,
        SECTION_MODEL_INPUTS: model_inputs,
        SECTION_STEP_PROBS: step_probs,
        SECTION_PROBS: probs,
        SECTION_EVENTS: events,
        SECTION_ALERTS: alerts,
        SECTION_COUNTERS: {
            "frames_accepted": pipe.frames_accepted,
            "dropped_frames": pipe.dropped_frames,
            "steps_emitted": pipe.steps_emitted,
            "model_calls": pipe.model_calls,
            "nonfinite_windows": pipe.nonfinite_windows,
        },
    }


# ---------------------------------------------------------------------------
# 비교와 검증
# ---------------------------------------------------------------------------

_MAX_PROBLEMS = 20


def _compare(
    section: str,
    key: str,
    expected: Any,
    actual: Any,
    tolerance: Mapping[str, float],
    path: str,
    problems: list[str],
) -> None:
    """expected와 actual을 허용 오차 안에서 비교하고 다른 곳을 problems에 적는다. key는 가장 가까운 필드 이름이다."""
    if len(problems) >= _MAX_PROBLEMS:
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            problems.append(f"{path}: field names differ")
            return
        for name in expected:
            _compare(section, name, expected[name], actual[name], tolerance, f"{path}.{name}", problems)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            actual_len = len(actual) if isinstance(actual, list) else "no list"
            problems.append(f"{path}: expected {len(expected)} items, got {actual_len}")
            return
        for index in range(len(expected)):
            _compare(section, key, expected[index], actual[index], tolerance, f"{path}[{index}]", problems)
    elif expected is None or isinstance(expected, (bool, str)):
        if type(actual) is not type(expected) or actual != expected:
            problems.append(f"{path}: expected {expected!r}, got {actual!r}")
    elif isinstance(expected, int):
        limit = 0
        if key in _TIME_KEYS.get(section, frozenset()):
            limit = int(tolerance["time_abs_ms"])
        if isinstance(actual, bool) or not isinstance(actual, int) or abs(actual - expected) > limit:
            problems.append(f"{path}: expected {expected!r} (tolerance {limit}), got {actual!r}")
    else:
        if section == SECTION_SIGNALS and key in _ANGLE_KEYS:
            abs_tol, rel_tol = tolerance["angle_abs_deg"], 0.0
        elif section in (SECTION_EVENTS, SECTION_ALERTS):
            abs_tol, rel_tol = tolerance["confidence_abs"], 0.0
        else:
            abs_tol, rel_tol = tolerance.get("abs", 0.0), tolerance.get("rel", 0.0)
        ok = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        if not ok or abs(actual - expected) > abs_tol + rel_tol * abs(expected):
            problems.append(f"{path}: expected {expected!r} (abs {abs_tol}, rel {rel_tol}), got {actual!r}")


_REQUIRED_KEYS: tuple[str, ...] = (
    "name",
    "bundle",
    "description",
    "pipeline_hash",
    "rules_hash",
    "tolerances",
    "baseline",
    "config_overrides",
    "model",
    "frames",
    "speed",
    "expected",
)


def verify_vector(vector: Mapping[str, Any], ctx: Context) -> list[str]:
    """벡터 하나를 다시 계산해 저장된 expected와 비교한다. 문제가 없으면 빈 목록을 돌려준다.

    확인하는 것: 필수 항목, pipeline_hash와 rules_hash가 지금 설정과 같은지, 모델 파일의 sha256,
    expected의 모든 항목이 그 항목의 허용 오차 안에서 다시 계산한 값과 같은지.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in vector]
    if missing:
        return [f"missing keys: {', '.join(missing)}"]
    problems: list[str] = []
    if vector["pipeline_hash"] != ctx.pipeline_hash:
        problems.append(f"pipeline_hash is {vector['pipeline_hash']}, current spec has {ctx.pipeline_hash}")
    if vector["rules_hash"] != ctx.rules_hash:
        problems.append(f"rules_hash is {vector['rules_hash']}, current rules have {ctx.rules_hash}")

    model = vector["model"]
    if model is not None and model.get("kind") == MODEL_KIND_ONNX:
        path = _model_path(model, ctx)
        if not path.is_file():
            return [*problems, f"model file not found: {path}"]
        digest = file_sha256(path)
        if digest != model["sha256"]:
            problems.append(f"model sha256 is {model['sha256']}, {path} has {digest}")

    try:
        actual = compute_expected(vector, ctx)
    except (VectorError, SpecError, KeyError, ValueError, TypeError) as error:
        return [*problems, f"could not recompute: {type(error).__name__}: {error}"]

    expected = vector["expected"]
    if len(expected) == 0:
        problems.append("expected is empty")
    for section, stored in expected.items():
        if section not in actual:
            problems.append(f"expected.{section}: not an output of bundle {vector['bundle']!r}")
            continue
        tolerance = vector["tolerances"].get(section, {})
        if section in TOLERANCES and set(tolerance) != set(TOLERANCES[section]):
            problems.append(f"tolerances.{section}: expected the keys {sorted(TOLERANCES[section])}")
            continue
        _compare(section, section, stored, actual[section], tolerance, f"expected.{section}", problems)
    return problems


def read_vector(path: Path) -> dict[str, Any]:
    """벡터 파일 하나를 읽는다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VectorError(f"{path}: expected a JSON object")
    return data


def vector_files(directory: Path) -> list[Path]:
    """폴더의 벡터 파일(*.json)을 이름 순으로 돌려준다."""
    return sorted(directory.glob("*.json"))


def verify_dir(directory: Path, ctx: Context) -> dict[str, list[str]]:
    """폴더의 모든 벡터를 검증한다. 파일 이름(확장자 제외) -> 문제 목록."""
    results: dict[str, list[str]] = {}
    for path in vector_files(directory):
        try:
            vector = read_vector(path)
        except (VectorError, json.JSONDecodeError) as error:
            results[path.stem] = [f"could not read: {error}"]
            continue
        problems = verify_vector(vector, ctx)
        if vector.get("name") != path.stem:
            problems.append(f"name is {vector.get('name')!r}, file is {path.name}")
        results[path.stem] = problems
    return results


# ---------------------------------------------------------------------------
# 파일 쓰기
# ---------------------------------------------------------------------------

_LINE_LIMIT = 4000  # 이보다 긴 목록과 객체는 원소마다 줄을 바꾼다. 프레임 하나(약 1.5~3 KB)는 한 줄에 들어간다


def _compact(value: Any) -> str:
    # 실수는 Python의 repr(왕복이 정확한 가장 짧은 십진 표기)로 적힌다. NaN과 무한대는 거부한다.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _format(value: Any, indent: int) -> str:
    """한 줄에 들어가면 한 줄로, 아니면 원소마다 줄을 바꿔서 적는다."""
    text = _compact(value)
    if not isinstance(value, (dict, list)) or len(value) == 0 or len(text) <= _LINE_LIMIT:
        return text
    if isinstance(value, dict):
        pad = " " * (indent + 2)
        items = [f"{pad}{_compact(name)}: {_format(item, indent + 2)}" for name, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    return _rows(value, indent)


def _rows(items: Sequence[Any], indent: int) -> str:
    """목록을 원소마다 한 줄씩 적는다."""
    if len(items) == 0:
        return "[]"
    pad = " " * (indent + 2)
    return "[\n" + ",\n".join(pad + _format(item, indent + 2) for item in items) + "\n" + " " * indent + "]"


def dumps_vector(vector: Mapping[str, Any]) -> str:
    """벡터를 파일 내용으로 바꾼다. 같은 벡터는 언제나 같은 문자열이 된다.

    최상위 항목은 한 줄에 하나씩 적는다. frames와 model_inputs, expected의 목록 항목은 원소(프레임, 스텝,
    이벤트)마다 줄을 바꾼다. 읽어 들이면 같은 값이 되는지 확인한다(실수의 왕복, 정수와 실수의 구분).
    """
    lines: list[str] = []
    for name, value in vector.items():
        if name == "expected":
            sections = [
                f"    {_compact(section)}: {_rows(items, 4) if isinstance(items, list) else _format(items, 4)}"
                for section, items in value.items()
            ]
            body = "{\n" + ",\n".join(sections) + "\n  }"
        elif name in ("frames", "model_inputs"):
            body = _rows(value, 2)
        else:
            body = _format(value, 2)
        lines.append(f"  {_compact(name)}: {body}")
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    if _compact(json.loads(text)) != _compact(vector):
        raise VectorError(f"vector {vector.get('name')!r} does not survive a JSON round trip")
    return text


def write_vectors(vectors: Sequence[Mapping[str, Any]], out_dir: Path) -> list[Path]:
    """벡터를 `{name}.json`으로 쓴다. 줄바꿈 변환을 막으려고 바이트로 쓴다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for vector in vectors:
        path = out_dir / f"{vector['name']}.json"
        path.write_bytes(dumps_vector(vector).encode("utf-8"))
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 벡터 정의
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Case:
    """벡터 하나의 입력과, 만들 때 확인할 의도.

    - sections: expected에 저장할 항목.
    - events, alerts: 이 각본이 내야 하는 이벤트와 경고의 "type:phase" 순서. None이면 확인하지 않는다.
    - check: 그 밖에 확인할 것. 계산한 출력 전체를 받아서 의도와 다르면 `VectorError`를 낸다.
    각본이 의도한 경우를 내지 못하면 생성이 실패한다. 참조 구현이 바뀌어 벡터가 더는 그 경계를 지나지 않게 된 것을
    조용히 넘기지 않기 위해서다.
    """

    name: str
    bundle: str
    description: str
    sections: tuple[str, ...]
    frames: tuple[RawFrame, ...] = ()
    baseline: Baseline | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    speed: tuple[tuple[int, float], ...] = ()
    model: Mapping[str, Any] | None = None
    model_inputs: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    events: tuple[str, ...] | None = None
    alerts: tuple[str, ...] | None = None
    check: Callable[[Mapping[str, Any]], None] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VectorError(message)


def _kinds(items: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(f"{item['type']}:{item['phase']}" for item in items)


def _build_vector(case: _Case, ctx: Context) -> dict[str, Any]:
    """입력을 파일 모양으로 적고, 그 모양에서 expected를 계산해 채운다."""
    vector: dict[str, Any] = {
        "name": case.name,
        "bundle": case.bundle,
        "description": case.description,
        "pipeline_hash": ctx.pipeline_hash,
        "rules_hash": ctx.rules_hash,
        "tolerances": {section: dict(TOLERANCES[section]) for section in case.sections if section in TOLERANCES},
        "baseline": None if case.baseline is None else baseline_to_json(case.baseline),
        "config_overrides": dict(case.overrides),
        "model": None if case.model is None else dict(case.model),
        "frames": [frame_to_json(frame) for frame in case.frames],
        "speed": [[t_ms, kmh] for t_ms, kmh in case.speed],
    }
    if case.model_inputs is not None:
        vector["model_inputs"] = [[list(row) for row in window] for window in case.model_inputs]
    _require(all(frame.t_ms > 0 for frame in case.frames), f"{case.name}: frame times must be positive")

    computed = compute_expected(vector, ctx)
    if SECTION_CLOSURE in computed:
        problem = _threshold_problem(computed[SECTION_CLOSURE], ctx)
        _require(problem is None, f"{case.name}: {problem}")
    if case.events is not None:
        got = _kinds(computed[SECTION_EVENTS])
        _require(got == case.events, f"{case.name}: events are {got}, the script intends {case.events}")
    if case.alerts is not None:
        got = _kinds(computed[SECTION_ALERTS])
        _require(got == case.alerts, f"{case.name}: alerts are {got}, the script intends {case.alerts}")
    if case.check is not None:
        try:
            case.check(computed)
        except VectorError as error:
            raise VectorError(f"{case.name}: {error}") from error
    vector["expected"] = {section: computed[section] for section in case.sections}
    return vector


_THRESHOLD_MARGIN = 1e-6


def _threshold_problem(rows: Sequence[Mapping[str, Any]], ctx: Context) -> str | None:
    """판정 임계값에 붙어 있는 표본이 있으면 그 내용을, 없으면 None을 돌려준다.

    각본의 선형 구간에서는 표본이 임계값과 정확히 같아질 수 있다(예: 깜빡임의 중간에서 closure 0.5). 그런 표본은
    수학 함수(hypot, atan2)의 마지막 자리 차이만으로 다른 쪽으로 넘어가서, 옳게 이식한 코드가 다른 기기에서 다른
    결과를 낸다. 벡터에는 임계값에서 1e-6(closure의 허용 오차) 넘게 떨어진 표본만 둔다.
    보는 임계값: 감김 검출기 둘의 진입·이탈과 PERCLOS 임계(closure), closure 유효 한계와 전방 주시 영역(자세),
    눈동자 보조(gaze), 하품(jaw_rel).
    """
    spec = ctx.spec
    zone = ctx.rules.forward_zone
    rule_detector = ctx.rules.closure.detector
    closure_thresholds = (
        spec.run_detector.enter_thr,
        spec.run_detector.exit_thr,
        rule_detector.enter_thr,
        rule_detector.exit_thr,
        spec.step_features.perclos_thr,
    )
    for row in rows:
        samples: list[tuple[str, float, float]] = []
        if row["closure_valid"]:
            samples += [("closure", row["closure"], threshold) for threshold in closure_thresholds]
        if row["face"]:
            samples.append(("|yaw_rel|", abs(row["yaw_rel"]), spec.closure.valid_max_abs_yaw_deg))
            samples.append(("jaw_rel", row["jaw_rel"], spec.yawn.enter_thr))
            samples.append(("jaw_rel", row["jaw_rel"], spec.yawn.exit_thr))
        if row["pose_valid"]:
            samples += [
                ("|yaw_rel|", abs(row["yaw_rel"]), zone.yaw_abs_max_deg),
                ("pitch_rel", row["pitch_rel"], -zone.pitch_up_max_deg),
                ("pitch_rel", row["pitch_rel"], zone.pitch_down_max_deg),
                ("|gaze_h_rel|", abs(row["gaze_h_rel"]), zone.gaze_h_abs_max),
                ("|gaze_v_rel|", abs(row["gaze_v_rel"]), zone.gaze_v_abs_max),
            ]
        for name, value, threshold in samples:
            if abs(value - threshold) < _THRESHOLD_MARGIN:
                return f"{name} = {value!r} at t_ms {row['t_ms']} sits on the threshold {threshold}"
    return None


# 6 fps 세션의 감김 깊이. 깊이가 1이면 감는 동안의 표본이 0과 1뿐이라 0.5를 지난 시각이 두 프레임의 한가운데
# (간격이 홀수 ms이면 x.5 ms)로 보간되고, 반올림이 마지막 자리 오차에 따라 갈린다. 0.97이면 그런 값이 나오지 않는다.
_LOW_FPS_DEPTH = 0.97


def _personal_baseline(spec: PipelineSpec) -> Baseline:
    """`synth.default_profile`의 얼굴에 맞는 개인 기준선(status ok, T_b = 0).

    값은 전역 상수와 같으므로 각본에 적은 눈꺼풀 수준이 그대로 closure가 된다. 전역 상수 기준선과 다른 점은
    status가 ok라서 자세가 유효하다는 것(시선 규칙이 동작한다)이다. T_b = 0이므로 첫 프레임부터 특징을 누적한다.
    """
    constants = spec.global_constants
    return Baseline(
        e_l=constants.e,
        e_r=constants.e,
        b_l=constants.b,
        b_r=constants.b,
        r0=IDENTITY_R0,
        jaw0=constants.jaw0,
        gaze_h0=constants.gaze_h0,
        gaze_v0=constants.gaze_v0,
        t_complete_ms=0,
        status=BASELINE_OK,
    )


def _render(
    spec: PipelineSpec, segments: Sequence[Segment], duration_ms: int, fps: float = 30.0, **options: Any
) -> tuple[RawFrame, ...]:
    return tuple(synth.render(Scenario(segments=tuple(segments), duration_ms=duration_ms, fps=fps, **options), spec))


# 손으로 놓는 프레임의 첫 t_ms. 각본 시나리오의 기본값(`Scenario.first_t_ms`)과 같다. 세션 시간 = t_ms − 1.
_FIRST_T_MS = 1


@dataclass(frozen=True, slots=True)
class _Shot:
    """손으로 놓는 프레임의 내용. 각본 구간 대신 프레임마다 값을 직접 정할 때 쓴다(`_hand_frames`).

    - level: 눈꺼풀 수준. `_personal_baseline`을 넣은 벡터에서는 closure가 이 값이다.
    - yaw, pitch: 머리 각도(도). 그 기준선의 R0가 단위 행렬이라 yaw_rel, pitch_rel이 이 값이다.
    - gaze_h, gaze_v: 눈동자 합성값. 그 기준선의 gaze0가 0이라 gaze_h_rel, gaze_v_rel이 이 값이다.
    - jaw: 쉬는 값에 더하는 턱 벌림. 그 기준선의 jaw0가 쉬는 값이라 jaw_rel이 이 값이다.
    - face: False이면 얼굴 없는 프레임이다(다른 값은 쓰지 않는다).
    """

    level: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    gaze_h: float = 0.0
    gaze_v: float = 0.0
    jaw: float = 0.0
    face: bool = True


def _shot_frame(spec: PipelineSpec, t_ms: int, shot: _Shot) -> RawFrame:
    """`_Shot` 하나를 `synth.default_profile`의 얼굴로 그린 프레임."""
    if not shot.face:
        return synth.no_face_frame(t_ms, 640, 480)
    profile = synth.default_profile(spec)
    ear, blink = synth.lid_signals(shot.level, profile.e_open_l, profile.b_open_l, spec)
    look = [0.0] * 8
    if shot.gaze_h > 0.0:  # 피험자의 왼쪽: 오른눈은 코 쪽(In), 왼눈은 관자놀이 쪽(Out)
        look[EL_IN_R] = shot.gaze_h
        look[EL_OUT_L] = shot.gaze_h
    elif shot.gaze_h < 0.0:
        look[EL_OUT_R] = -shot.gaze_h
        look[EL_IN_L] = -shot.gaze_h
    if shot.gaze_v > 0.0:
        look[EL_UP_L] = shot.gaze_v
        look[EL_UP_R] = shot.gaze_v
    elif shot.gaze_v < 0.0:
        look[EL_DOWN_L] = -shot.gaze_v
        look[EL_DOWN_R] = -shot.gaze_v
    s = FrameSignals(
        t_ms=t_ms,
        face=True,
        ear_l=ear,
        ear_r=ear,
        eye_w_l=profile.eye_w_px,
        eye_w_r=profile.eye_w_px,
        blink_l=blink,
        blink_r=blink,
        eye_look=tuple(look),
        jaw_open=profile.jaw_rest + shot.jaw,
        yaw=shot.yaw,
        pitch=shot.pitch,
        roll=0.0,
    )
    return synth.frame_from_signals(s, 640, 480, spec)


def _hand_frames(
    spec: PipelineSpec, times: Sequence[int], plan: Sequence[tuple[int, _Shot]], first_t_ms: int = _FIRST_T_MS
) -> tuple[RawFrame, ...]:
    """세션 시간 `times`(엄격히 증가)에 프레임을 하나씩 놓는다. `t_ms = first_t_ms + 세션 시간`.

    `plan`은 (시작 세션 시간, 내용)의 목록이고 시작 시간 순이다. 프레임의 내용은 시작 시간이 그 프레임의 시각
    이하인 마지막 항목이다. 값은 계단 모양으로 바뀐다(구간 사이를 잇는 경사가 없다).
    """
    starts = [start for start, _ in plan]
    _require(len(times) > 0 and len(plan) > 0, "hand frames need at least one time and one plan entry")
    _require(starts == sorted(starts) and starts[0] <= times[0], "plan must be sorted and cover the first frame")
    _require(all(later > earlier for earlier, later in zip(times, times[1:])), "frame times must increase")
    frames: list[RawFrame] = []
    for t in times:
        shot = plan[bisect.bisect_right(starts, t) - 1][1]
        frames.append(_shot_frame(spec, first_t_ms + t, shot))
    return tuple(frames)


_OPEN = _Shot()
_SHUT = _Shot(level=0.9)  # 두 검출기의 진입 임계(0.5, 0.7)와 PERCLOS 임계(0.8)를 모두 넘는 감김


# --- 신호 -------------------------------------------------------------------

_SIGNALS_ONLY: tuple[str, ...] = (SECTION_SIGNALS,)


def _face_signals(t_ms: int, **changes: Any) -> FrameSignals:
    """신호 묶음의 기본 얼굴. 좌우 값을 모두 다르게 둬서 좌우가 바뀌면 드러나게 한다."""
    base = FrameSignals(
        t_ms=t_ms,
        face=True,
        ear_l=0.31,
        ear_r=0.27,
        eye_w_l=52.0,
        eye_w_r=48.0,
        blink_l=0.18,
        blink_r=0.25,
        eye_look=(0.05, 0.12, 0.03, 0.2, 0.07, 0.02, 0.04, 0.22),
        jaw_open=0.11,
        yaw=12.5,
        pitch=-7.25,
        roll=3.0,
        iris_h_l=0.46,
        iris_h_r=0.55,
        iris_v_l=0.02,
        iris_v_r=-0.03,
    )
    return replace(base, **changes)


def _signal_cases(ctx: Context) -> list[_Case]:
    spec = ctx.spec

    def frame(t_ms: int, w: int, h: int, **changes: Any) -> RawFrame:
        return synth.frame_from_signals(_face_signals(t_ms, **changes), w, h, spec)

    def check_same_ear(computed: Mapping[str, Any]) -> None:
        rows = computed[SECTION_SIGNALS]
        for row in rows:
            exact = abs(row["ear_l"] - 0.31) < 1e-9 and abs(row["ear_r"] - 0.27) < 1e-9
            _require(exact, "EAR is not the requested value")

    official = replace(frame(1, 640, 480), m=OFFICIAL_MATRIX)

    def check_official(computed: Mapping[str, Any]) -> None:
        row = computed[SECTION_SIGNALS][0]
        for name, angle in zip(("yaw", "pitch", "roll"), OFFICIAL_ANGLES_DEG):
            _require(abs(row[name] - angle) < 0.01, f"{name} is {row[name]}, the official output expects about {angle}")

    def check_mirror(computed: Mapping[str, Any]) -> None:
        plain, mirrored = computed[SECTION_SIGNALS]
        _require(mirrored["yaw"] == -plain["yaw"] and mirrored["roll"] == -plain["roll"], "yaw and roll must flip sign")
        _require(mirrored["pitch"] == plain["pitch"], "pitch must not change")
        _require(mirrored["blink_l"] == plain["blink_r"] and mirrored["blink_r"] == plain["blink_l"], "L/R must swap")
        _require(abs(mirrored["ear_l"] - plain["ear_r"]) < 1e-9, "EAR must swap with the eyes")

    rng = random.Random(20261113)
    random_frames: list[RawFrame] = []
    random_angles: list[tuple[float, float, float]] = []
    for index in range(8):
        angles = (rng.uniform(-170.0, 170.0), rng.uniform(-80.0, 80.0), rng.uniform(-170.0, 170.0))
        translation = (rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0), rng.uniform(-90.0, -30.0))
        base = frame(1 + 33 * index, 640, 480)
        random_frames.append(replace(base, m=synth.pose_matrix(*angles, translation)))
        random_angles.append(angles)

    def check_random(computed: Mapping[str, Any]) -> None:
        for row, angles in zip(computed[SECTION_SIGNALS], random_angles):
            for name, angle in zip(("yaw", "pitch", "roll"), angles):
                _require(abs(row[name] - angle) < 1e-9, f"{name} does not round-trip through the matrix")

    left_closed = frame(1, 640, 480, ear_l=0.06, blink_l=0.72, ear_r=0.29, blink_r=0.2)

    def check_laterality(computed: Mapping[str, Any]) -> None:
        plain, mirrored = computed[SECTION_SIGNALS]
        _require(plain["ear_l"] < 0.1 < plain["ear_r"] and plain["blink_l"] > 0.5 > plain["blink_r"], "left eye closed")
        _require(mirrored["ear_r"] < 0.1 < mirrored["ear_l"], "mirrored: right eye closed")
        _require(mirrored["blink_r"] > 0.5 > mirrored["blink_l"], "mirrored: right eye closed")

    def check_width(computed: Mapping[str, Any]) -> None:
        zero, narrow, usable = computed[SECTION_SIGNALS]
        _require(zero["eye_w_l"] == 0.0 and zero["ear_l"] == 0.0 and zero["iris_h_l"] is None, "zero-width left eye")
        _require(zero["ear_r"] > 0.0 and zero["iris_h_r"] is not None, "the other eye stays usable")
        _require(narrow["eye_w_r"] < 6.0 and narrow["ear_r"] == 0.0 and narrow["iris_v_r"] is None, "narrow right eye")
        _require(usable["ear_l"] > 0.0 and usable["ear_r"] > 0.0, "7 px eyes are usable")

    def tilted(base: RawFrame, turn_l: tuple[float, float], turn_r: tuple[float, float]) -> RawFrame:
        """두 눈의 윤곽 점과 홍채 점을 그 눈의 중점(두 눈 꼬리의 가운데) 둘레로 픽셀 좌표에서 돌린 프레임.

        turn은 (cos, sin)이다. 돌려도 점 사이의 픽셀 거리는 그대로라서 EAR, 눈 폭, 홍채 비율은 돌리기 전과 같아야 한다.
        """
        lms = spec.landmarks
        lm = dict(base.lm)
        for contour, iris, outer, inner, (cos, sin) in (
            (lms.contour_l, lms.iris_l, lms.corner_outer_l, lms.corner_inner_l, turn_l),
            (lms.contour_r, lms.iris_r, lms.corner_outer_r, lms.corner_inner_r, turn_r),
        ):
            cx = 0.5 * (base.lm[outer][0] + base.lm[inner][0]) * base.w
            cy = 0.5 * (base.lm[outer][1] + base.lm[inner][1]) * base.h
            for index in (*contour, *iris):
                x = base.lm[index][0] * base.w - cx
                y = base.lm[index][1] * base.h - cy
                lm[index] = ((cx + cos * x - sin * y) / base.w, (cy + sin * x + cos * y) / base.h)
        return replace(base, lm=lm)

    # 3:4:5 삼각형의 각도(36.87도, 53.13도)로 돌린다: 눈 폭 52 px인 Left 눈의 두 꼬리는 세로로 31.2 px 또는 41.6 px 떨어진다.
    tilted_frames = (
        tilted(frame(1, 640, 480), (0.8, 0.6), (0.6, -0.8)),
        tilted(frame(34, 1280, 720), (0.6, -0.8), (0.8, 0.6)),
    )

    def check_tilted(computed: Mapping[str, Any]) -> None:
        check_same_ear(computed)
        lms = spec.landmarks
        drops = []
        for item in tilted_frames:
            for outer, inner in ((lms.corner_outer_l, lms.corner_inner_l), (lms.corner_outer_r, lms.corner_inner_r)):
                drops.append(abs(item.lm[outer][1] - item.lm[inner][1]) * item.h)
        wanted = (52 * 0.6, 48 * 0.8, 52 * 0.8, 48 * 0.6)  # 눈 폭 x sin: 두 꼬리의 세로 거리(px)
        _require(all(abs(got - want) < 1e-9 for got, want in zip(drops, wanted)), f"eye axes are not tilted: {drops}")
        for row in computed[SECTION_SIGNALS]:
            _require(abs(row["eye_w_l"] - 52.0) < 1e-9 and abs(row["eye_w_r"] - 48.0) < 1e-9, "eye width changed")
            iris = (row["iris_h_l"], row["iris_v_l"], row["iris_h_r"], row["iris_v_r"])
            wanted_iris = (0.46, 0.02, 0.55, -0.03)  # `_face_signals`에 적은 값: 눈 축이 기울어도 그대로다
            _require(all(abs(got - want) < 1e-9 for got, want in zip(iris, wanted_iris)), f"iris ratios are {iris}")

    return [
        _Case(
            name="signals_ear_square",
            bundle=BUNDLE_SIGNALS,
            description=(
                "정사각 프레임(720x720) 한 장의 원시 신호. EAR은 픽셀 좌표 (x·w, y·h)에서 구한다. 좌우 값이 모두 달라서 "
                "L/R 랜드마크 집합이나 블렌드셰이프 이름을 바꿔 읽으면 드러난다. 홍채 비율과 eyeLook 8개도 확인한다."
            ),
            sections=_SIGNALS_ONLY,
            frames=(frame(1, 720, 720),),
            check=check_same_ear,
        ),
        _Case(
            name="signals_ear_non_square",
            bundle=BUNDLE_SIGNALS,
            description=(
                "비정사각 프레임 세 장(1280x720, 720x1280, 568x320)에 같은 픽셀 기하의 눈을 놓았다. EAR은 세 장 모두 "
                "ear_l 0.31, ear_r 0.27이다. 정규화 좌표로 EAR을 구하면 프레임마다 다른 값이 나온다."
            ),
            sections=_SIGNALS_ONLY,
            frames=(frame(1, 1280, 720), frame(34, 720, 1280), frame(67, 568, 320)),
            check=check_same_ear,
        ),
        _Case(
            name="signals_official_matrix",
            bundle=BUNDLE_SIGNALS,
            description=(
                "MediaPipe 공식 기대 출력 파일의 변환 행렬(열 우선 16개 값). yaw, pitch, roll은 약 2.24, 6.32, 0.99도다. "
                "행 우선으로 읽으면 전치가 되어 부호가 뒤집힌다."
            ),
            sections=_SIGNALS_ONLY,
            frames=(official,),
            check=check_official,
        ),
        _Case(
            name="signals_mirrored_matrix",
            bundle=BUNDLE_SIGNALS,
            description=(
                "공식 행렬의 프레임과 그것을 좌우 반전한 프레임. 반전하면 yaw와 roll의 부호가 뒤집히고 pitch는 그대로이며 "
                "_l과 _r의 신호가 서로 바뀐다."
            ),
            sections=_SIGNALS_ONLY,
            frames=(official, replace(synth.mirror_frame(official, spec), t_ms=34)),
            check=check_mirror,
        ),
        _Case(
            name="signals_random_matrices",
            bundle=BUNDLE_SIGNALS,
            description=(
                "무작위 자세 여덟 개(yaw ±170도, pitch ±80도, roll ±170도, 평행이동도 무작위)의 행렬에서 각도를 되찾는다. "
                "R = Ry(yaw)·Rx(pitch)·Rz(roll). 평행이동은 각도에 영향을 주지 않는다."
            ),
            sections=_SIGNALS_ONLY,
            frames=tuple(random_frames),
            check=check_random,
        ),
        _Case(
            name="signals_one_eye_closed",
            bundle=BUNDLE_SIGNALS,
            description=(
                "한쪽 눈만 감은 프레임 둘. 첫 프레임은 MediaPipe Left 눈(L 집합 362.., eyeBlinkLeft)만 감겼고, "
                "둘째는 그것을 좌우 반전해 Right 눈만 감겼다. 좌우 짝이 맞는지 확인한다."
            ),
            sections=_SIGNALS_ONLY,
            frames=(left_closed, replace(synth.mirror_frame(left_closed, spec), t_ms=34)),
            check=check_laterality,
        ),
        _Case(
            name="signals_zero_width_eye",
            bundle=BUNDLE_SIGNALS,
            description=(
                "눈 폭이 6 px 미만인 눈은 쓸 수 없는 눈이다(ear 0.0, 홍채 비율 null, 눈 폭은 측정값 그대로). 프레임 셋: "
                "Left 눈 폭 0 px(0으로 나누지 않아야 한다), Right 눈 폭 5 px, 두 눈 모두 7 px(쓸 수 있음)."
            ),
            sections=_SIGNALS_ONLY,
            frames=(
                frame(1, 640, 480, eye_w_l=0.0),
                frame(34, 640, 480, eye_w_r=5.0),
                frame(67, 640, 480, eye_w_l=7.0, eye_w_r=7.0),
            ),
            check=check_width,
        ),
        _Case(
            name="signals_no_face",
            bundle=BUNDLE_SIGNALS,
            description="얼굴이 검출되지 않은 프레임. 모든 실수 신호는 0.0, eyeLook은 0.0 여덟 개, 홍채 비율은 null이다.",
            sections=_SIGNALS_ONLY,
            frames=(synth.no_face_frame(1, 640, 480),),
        ),
        _Case(
            name="signals_iris_tilted_eyes",
            bundle=BUNDLE_SIGNALS,
            description=(
                "눈 축이 영상에서 기울어진 프레임 둘(640x480, 1280x720). 다른 신호 벡터와 같은 얼굴인데 두 눈의 윤곽 점과 "
                "홍채 점을 눈의 중점 둘레로 픽셀 좌표에서 돌렸다: 첫 프레임은 Left 눈 +36.87도, Right 눈 -53.13도, 둘째는 "
                "Left 눈 -53.13도, Right 눈 +36.87도(3:4:5 삼각형의 각도). 머리가 기울면(roll) 실제 영상에서도 이렇게 된다. "
                "홍채 비율 h = dot(d, a)/|a|², v = cross(a, d)/|a|²는 눈 축의 방향과 무관해야 한다: iris_h_l 0.46, "
                "iris_v_l 0.02, iris_h_r 0.55, iris_v_r -0.03, EAR 0.31과 0.27, 눈 폭 52 px과 48 px이 그대로 나온다. 다른 "
                "신호 벡터는 두 눈 꼬리가 같은 높이에 있어서(a의 세로 성분이 0) 식에서 a의 세로 성분이 든 항의 부호가 "
                "틀려도 드러나지 않는다. 홍채 비율은 기록용이고 이식 대상의 다른 모듈은 이 값을 읽지 않는다."
            ),
            sections=_SIGNALS_ONLY,
            frames=tilted_frames,
            check=check_tilted,
        ),
    ]


# --- 기준선과 closure --------------------------------------------------------

_BASELINE_SECTIONS: tuple[str, ...] = (SECTION_CLOSURE, SECTION_BASELINE, SECTION_COUNTERS)
_CLOSURE_SECTIONS: tuple[str, ...] = (SECTION_CLOSURE, SECTION_COUNTERS)

# 기준선 벡터의 얼굴: 좌우 눈의 뜬 눈 수준이 다르고, 쉬는 자세와 쉬는 시선이 0이 아니다.
_BASELINE_PROFILE = FaceProfile(
    e_open_l=0.31,
    e_open_r=0.29,
    b_open_l=0.19,
    b_open_r=0.23,
    jaw_rest=0.06,
    yaw=6.0,
    pitch=-4.0,
    roll=1.5,
    eye_look_rest=(0.04, 0.10, 0.02, 0.15, 0.06, 0.03, 0.02, 0.16),
    eye_w_px=44.0,
    centre=(0.48, 0.42),
)


def _with_noise(
    frames: Sequence[RawFrame], spec: PipelineSpec, seed: int, centre: tuple[float, float]
) -> tuple[RawFrame, ...]:
    """각본 프레임의 신호를 프레임마다 조금씩 흔든다(시드가 같으면 결과가 같다).

    각본 시나리오의 신호는 몇 가지 값만 되풀이하므로 분위수와 중앙값의 정의가 달라도 결과가 같아질 수 있다.
    값을 흩어 놓으면 최근접 순위 분위수와 보간 분위수, 짝수 표본의 중앙값 정의가 서로 다른 답을 낸다.
    신호는 `signals.compute`로 읽고 `synth.frame_from_signals`로 다시 프레임을 만든다.
    """
    rng = random.Random(seed)
    out: list[RawFrame] = []
    for frame in frames:
        if not frame.face:
            out.append(frame)
            continue
        s = signals.compute(frame, spec)
        shaken = replace(
            s,
            ear_l=s.ear_l * (1.0 + rng.uniform(-0.04, 0.04)),
            ear_r=s.ear_r * (1.0 + rng.uniform(-0.04, 0.04)),
            blink_l=s.blink_l + rng.uniform(-0.015, 0.015),
            blink_r=s.blink_r + rng.uniform(-0.015, 0.015),
            eye_look=tuple(value + rng.uniform(0.0, 0.02) for value in s.eye_look),
            jaw_open=s.jaw_open + rng.uniform(-0.01, 0.01),
            yaw=s.yaw + rng.uniform(-1.5, 1.5),
            pitch=s.pitch + rng.uniform(-1.0, 1.0),
            roll=s.roll + rng.uniform(-0.5, 0.5),
        )
        out.append(synth.frame_from_signals(shaken, frame.w, frame.h, spec, centre))
    return tuple(out)


def _baseline_cases(ctx: Context) -> list[_Case]:
    spec = ctx.spec

    def session(segments: Sequence[Segment], duration_ms: int, profile: FaceProfile, seed: int) -> tuple[RawFrame, ...]:
        frames = _render(spec, segments, duration_ms, fps=6.0, profile=profile)
        return _with_noise(frames, spec, seed, profile.centre)

    def check_baseline(status: str, t_complete_ms: int) -> Callable[[Mapping[str, Any]], None]:
        def check(computed: Mapping[str, Any]) -> None:
            baseline = computed[SECTION_BASELINE]
            _require(baseline is not None, "the baseline was not decided")
            _require(baseline["status"] == status, f"baseline status is {baseline['status']}")
            _require(baseline["t_complete_ms"] == t_complete_ms, f"T_b is {baseline['t_complete_ms']}")
            for row in computed[SECTION_CLOSURE]:
                before = row["t_ms"] < baseline["t_ms"]
                _require(row["provisional"] == before, "provisional must end with the baseline")
                if status == BASELINE_FALLBACK_GLOBAL:
                    _require(not row["pose_valid"], "pose must stay invalid with the global fallback")

        return check

    normal = [Blink(400, start_ms=start) for start in range(3000, 60000, 5200)]
    normal += [
        LookAside(2500, gaze_h=0.15, gaze_v=-0.1, start_ms=12000),
        Yawn(3000, jaw=0.5, start_ms=20000),
        HeadTurn(3000, yaw=20.0, pitch=5.0, start_ms=30000),
        FaceMissing(2000, start_ms=45000),
        Closure(hold_ms=600, start_ms=61000),
    ]
    extended = [Blink(400, start_ms=start) for start in range(2000, 90000, 6100)]
    extended += [FaceMissing(34000, start_ms=8000), Closure(hold_ms=600, start_ms=90700)]
    fallback = [Blink(400, start_ms=start) for start in range(2000, 12000, 4300)]
    fallback += [FaceMissing(166000, start_ms=12000), Linger(2000, 0.9, start_ms=180500)]

    # closure의 눈 가중: 기준 자세가 (10, 5, 2)도인 기준선을 넣고, Left 눈만 많이 감긴 얼굴을 여러 yaw에서 본다.
    weighting_baseline = Baseline(
        e_l=0.30,
        e_r=0.28,
        b_l=0.20,
        b_r=0.24,
        r0=signals.rotation_from_euler(10.0, 5.0, 2.0),
        jaw0=0.06,
        gaze_h0=0.01,
        gaze_v0=-0.02,
        t_complete_ms=0,
        status=BASELINE_OK,
    )
    ear_l, blink_l = synth.lid_signals(0.9, weighting_baseline.e_l, weighting_baseline.b_l, spec)
    ear_r, blink_r = synth.lid_signals(0.2, weighting_baseline.e_r, weighting_baseline.b_r, spec)
    yaw_offsets = (0.0, 10.0, -10.0, 12.0, -12.0, 17.0, -17.0, 20.0, -20.0, 23.0, -23.0, 28.0, -28.0, 37.0, -37.0)
    invalid_offsets = (44.0, -44.0)
    eye_widths = ((0.0, 42.0), (42.0, 3.0), (0.0, 0.0))  # (Left, Right) 눈 폭: Left만, Right만, 둘 다 쓸 수 없음

    def weighting_frame(index: int, yaw_offset: float, eye_w_l: float = 42.0, eye_w_r: float = 42.0) -> RawFrame:
        s = FrameSignals(
            t_ms=1 + 100 * index,
            face=True,
            ear_l=ear_l,
            ear_r=ear_r,
            eye_w_l=eye_w_l,
            eye_w_r=eye_w_r,
            blink_l=blink_l,
            blink_r=blink_r,
            eye_look=(0.03, 0.08, 0.02, 0.1, 0.05, 0.02, 0.02, 0.12),
            jaw_open=0.09,
            yaw=10.0 + yaw_offset,
            pitch=5.0,
            roll=2.0,
        )
        return synth.frame_from_signals(s, 640, 480, spec)

    weighting_frames = [weighting_frame(index, offset) for index, offset in enumerate(yaw_offsets + invalid_offsets)]
    first_width = len(weighting_frames)
    weighting_frames += [weighting_frame(first_width + index, 0.0, *widths) for index, widths in enumerate(eye_widths)]

    def check_weighting(computed: Mapping[str, Any]) -> None:
        rows = computed[SECTION_CLOSURE]
        by_offset = dict(zip(yaw_offsets + invalid_offsets, rows))
        _require(abs(by_offset[0.0]["closure"] - 0.55) < 1e-9, "both eyes weigh the same at yaw_rel 0")
        _require(abs(by_offset[10.0]["closure"] - 0.55) < 1e-9, "both eyes weigh the same below 15 degrees")
        _require(abs(by_offset[28.0]["closure"] - 0.2) < 1e-9, "only the near (Right) eye counts beyond 25 degrees")
        _require(abs(by_offset[-28.0]["closure"] - 0.9) < 1e-9, "only the near (Left) eye counts beyond -25 degrees")
        _require(0.2 < by_offset[20.0]["closure"] < 0.55 < by_offset[-20.0]["closure"] < 0.9, "far eye ramp")
        _require(by_offset[37.0]["closure_valid"] and by_offset[-37.0]["closure_valid"], "valid up to 40 degrees")
        _require(not by_offset[44.0]["closure_valid"] and not by_offset[-44.0]["closure_valid"], "invalid beyond 40")
        only_right, only_left, neither = rows[first_width:]
        _require(abs(only_right["closure"] - 0.2) < 1e-9, "Left eye unusable: closure is the Right eye's")
        _require(abs(only_left["closure"] - 0.9) < 1e-9, "Right eye unusable: closure is the Left eye's")
        _require(not neither["closure_valid"] and neither["closure"] == 0.0, "no usable eye")

    profile = synth.default_profile(spec)
    lid_open = synth.lid_signals(0.35, profile.e_open_l, profile.b_open_l, spec)
    lid_half = synth.lid_signals(0.6, profile.e_open_l, profile.b_open_l, spec)

    def plain_frame(t_ms: int, lids: tuple[float, float]) -> RawFrame:
        s = FrameSignals(
            t_ms=t_ms,
            face=True,
            ear_l=lids[0],
            ear_r=lids[0],
            eye_w_l=profile.eye_w_px,
            eye_w_r=profile.eye_w_px,
            blink_l=lids[1],
            blink_r=lids[1],
            eye_look=profile.eye_look_rest,
            jaw_open=profile.jaw_rest,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
        )
        return synth.frame_from_signals(s, 640, 480, spec)

    def check_no_face(computed: Mapping[str, Any]) -> None:
        before, lost_a, lost_b, after = computed[SECTION_CLOSURE]
        _require(abs(before["closure"] - 0.35) < 1e-9, "closure of the first face frame")
        _require(abs(after["closure"] - 0.6) < 1e-9, "closure of the last face frame")
        for row in (lost_a, lost_b):
            _require(not row["face"] and not row["closure_valid"] and not row["pose_valid"], "no face: nothing valid")
            _require(row["closure"] == 0.0 and row["yaw_rel"] == 0.0 and row["jaw_rel"] == 0.0, "no face: zeros")

    # --- 손으로 놓은 보정 세션 둘. 신호는 몇 가지 값뿐이라 기준선의 모든 값을 손으로 셀 수 있다 -------------------------

    def calibration_frame(
        t: int,
        ears: tuple[float, float],
        blinks: tuple[float, float],
        jaw_open: float,
        pitch: float = 0.0,
        look_left: float = 0.0,
        look_up: float = 0.0,
        eye_w_r: float = 44.0,
        eye_w_l: float = 44.0,
    ) -> RawFrame:
        """세션 시간 t의 얼굴 프레임(t_ms = t + 1). ears와 blinks는 (Left, Right) 순서다."""
        look = [0.0] * 8
        look[EL_IN_R] = look_left  # gaze_h = ((InRight + OutLeft) − (OutRight + InLeft)) / 2 = look_left
        look[EL_OUT_L] = look_left
        look[EL_UP_L] = look_up  # gaze_v = 두 눈의 Up 평균 − Down 평균 = look_up
        look[EL_UP_R] = look_up
        s = FrameSignals(
            t_ms=_FIRST_T_MS + t,
            face=True,
            ear_l=ears[0],
            ear_r=ears[1],
            eye_w_l=eye_w_l,
            eye_w_r=eye_w_r,
            blink_l=blinks[0],
            blink_r=blinks[1],
            eye_look=tuple(look),
            jaw_open=jaw_open,
            yaw=0.0,
            pitch=pitch,
            roll=0.0,
        )
        return synth.frame_from_signals(s, 640, 480, spec)

    # (1) 표본이 홀수 개인 기준선. 얼굴 프레임 331개: 처음 1분에 300개(200 ms 간격, Right 눈은 폭 3 px이라 쓸 수 없다),
    # 65초·72초·79초에 셋(Right 눈의 EAR이 0.50으로 e_max 0.45를 넘는다), 90~117초에 1초 간격으로 28개(Right 눈 0.28).
    # 앞의 165개는 "낮은 자세"(pitch 0, 턱 0.05, 눈동자 0), 뒤의 166개는 "높은 자세"(pitch 4, 턱 0.09, 눈동자 0.1과 0.06)다.
    odd_times = [200 * index for index in range(300)] + [65000, 72000, 79000] + [90000 + 1000 * i for i in range(28)]
    odd_blinks = (0.20, 0.24)

    def odd_frame(t: int, high: bool, ear_r: float, eye_w_r: float = 44.0, lids: float = 0.0) -> RawFrame:
        ear_l, blink_l = synth.lid_signals(lids, 0.30, odd_blinks[0], spec)
        ear_r, blink_r = synth.lid_signals(lids, ear_r, odd_blinks[1], spec)
        if high:
            return calibration_frame(t, (ear_l, ear_r), (blink_l, blink_r), 0.09, 4.0, 0.1, 0.06, eye_w_r)
        return calibration_frame(t, (ear_l, ear_r), (blink_l, blink_r), 0.05, eye_w_r=eye_w_r)

    odd_frames: list[RawFrame] = []
    for index, t in enumerate(odd_times):
        if index < 300:
            odd_frames.append(odd_frame(t, index >= 165, 0.28, eye_w_r=3.0))
        else:
            odd_frames.append(odd_frame(t, True, 0.50 if index < 303 else 0.28))
    # 완성 뒤의 세 프레임: 높은 자세(기준선과 같은 얼굴), 낮은 자세, 높은 자세에서 두 눈을 0.6만큼 감은 얼굴
    odd_frames += [odd_frame(120000, True, 0.28), odd_frame(120200, False, 0.28), odd_frame(120400, True, 0.28, lids=0.6)]

    def check_odd(computed: Mapping[str, Any]) -> None:
        check_baseline(BASELINE_OK, 120000)(computed)
        got = computed[SECTION_BASELINE]
        _require(got["t_ms"] == _FIRST_T_MS + 120000, f"the baseline is first used at t_ms {got['t_ms']}")
        cos4, sin4 = math.cos(math.radians(4.0)), math.sin(math.radians(4.0))
        wanted = {
            "e_l": 0.30,  # 331개 모두 0.30
            "e_r": 0.28,  # 쓸 수 있는 31개(0.28이 28개, 0.50이 3개)의 28번째(ceil(0.9 x 31))
            "b_l": 0.20,
            "b_r": 0.24,
            "jaw0": 0.09,  # 331개의 166번째: 0.05가 165개, 0.09가 166개
            "gaze_h0": 0.1,
            "gaze_v0": 0.06,
        }
        for name, value in wanted.items():
            _require(abs(got[name] - value) < 1e-9, f"baseline {name} is {got[name]}, the script intends {value}")
        r0 = (1.0, 0.0, 0.0, 0.0, cos4, -sin4, 0.0, sin4, cos4)  # Rx(4도): 중앙값은 yaw 0, pitch 4, roll 0
        _require(all(abs(a - b) < 1e-9 for a, b in zip(got["r0"], r0)), f"R0 is {got['r0']}")
        same, low, shut = computed[SECTION_CLOSURE][-3:]
        for row, pitch_rel, jaw_rel, gaze_h, gaze_v, closure in (
            (same, 0.0, 0.0, 0.0, 0.0, 0.0),
            (low, -4.0, -0.04, -0.1, -0.06, 0.0),
            (shut, 0.0, 0.0, 0.0, 0.0, 0.6),
        ):
            values = (row["pitch_rel"], row["jaw_rel"], row["gaze_h_rel"], row["gaze_v_rel"], row["closure"])
            wanted_row = (pitch_rel, jaw_rel, gaze_h, gaze_v, closure)
            _require(all(abs(a - b) < 1e-9 for a, b in zip(values, wanted_row)), f"row {row} is not {wanted_row}")

    # (2) 프레임이 몰려서 오는 세션. 처음 59초는 초마다 5장(0, 75, 150, 225, 300 ms), 마지막 1초는 5장(25 ms 간격)과
    # 59900의 한 장. 얼굴 없는 프레임 셋(500, 1500, 2500). 60100부터 200 ms 간격으로 75장, 그리고 90000부터 셋.
    burst_times = [1000 * second + offset for second in range(59) for offset in (0, 75, 150, 225, 300)]
    burst_times += [59000, 59025, 59050, 59075, 59100, 59900]
    burst_times += [60100 + 200 * index for index in range(75)] + [90000, 90200, 90400]
    burst_missing = (500, 1500, 2500)
    burst_frames = [calibration_frame(t, (0.30, 0.30), (0.20, 0.20), 0.10) for t in burst_times]
    burst_frames += [synth.no_face_frame(_FIRST_T_MS + t, 640, 480) for t in burst_missing]
    burst_frames.sort(key=lambda item: item.t_ms)

    def check_burst(computed: Mapping[str, Any]) -> None:
        check_baseline(BASELINE_OK, 90000)(computed)
        got = computed[SECTION_BASELINE]
        _require(got["t_ms"] == _FIRST_T_MS + 90000, f"the baseline is first used at t_ms {got['t_ms']}")
        for name, value in (("e_l", 0.30), ("e_r", 0.30), ("b_l", 0.20), ("b_r", 0.20), ("jaw0", 0.10)):
            _require(abs(got[name] - value) < 1e-9, f"baseline {name} is {got[name]}")
        # 얼굴 프레임이 대표한 시간을 프레임 시각에서 다시 센다(다음 프레임까지, 최대 200 ms, 마감에서 자른다).
        face_times = sorted(burst_times)
        every = sorted(burst_times + list(burst_missing))
        for deadline, n_face, face_ms in ((60000, 301, 29900), (90000, 376, 45000)):
            held = 0
            for t in face_times:
                if t >= deadline:
                    break
                later = every[every.index(t) + 1]
                held += min(later, deadline, t + ctx.spec.clock.max_hold_ms) - t
            count = sum(1 for t in face_times if t < deadline)
            _require((count, held) == (n_face, face_ms), f"deadline {deadline}: {count} face frames, {held} ms")

    # (3) 얼굴 프레임이 정확히 300개인 기준선. 0~59600의 299개(200 ms 간격)는 Left 눈의 폭이 3 px이라 쓸 수 없고,
    # 300번째(60000)에서만 Left 눈을 쓸 수 있다. 그 뒤 90000까지 프레임이 없다. 모든 프레임의 다른 신호는 같다
    # (Left EAR 0.30, Right EAR 0.28, eyeBlink 0.20과 0.24, 턱 0.05, 자세 0도).
    exact_times = [200 * index for index in range(299)] + [60000]

    def exact_frame(t: int, eye_w_l: float = 44.0, lids: float = 0.0) -> RawFrame:
        ear_l, blink_l = synth.lid_signals(lids, 0.30, 0.20, spec)
        ear_r, blink_r = synth.lid_signals(lids, 0.28, 0.24, spec)
        return calibration_frame(t, (ear_l, ear_r), (blink_l, blink_r), 0.05, eye_w_l=eye_w_l)

    exact_frames = [exact_frame(t, eye_w_l=3.0) for t in exact_times[:299]] + [exact_frame(exact_times[299])]
    # 완성 뒤의 세 프레임: 기준선과 같은 얼굴 둘, 두 눈을 0.6만큼 감은 얼굴
    exact_frames += [exact_frame(90000), exact_frame(90200), exact_frame(90400, lids=0.6)]

    def check_exact(computed: Mapping[str, Any]) -> None:
        check_baseline(BASELINE_OK, 90000)(computed)
        got = computed[SECTION_BASELINE]
        _require(got["t_ms"] == _FIRST_T_MS + 90000, f"the baseline is first used at t_ms {got['t_ms']}")
        _require(ctx.spec.baseline.min_face_frames == 300, "the session is built for min_face_frames = 300")
        collected = [signals.compute(item, spec) for item in exact_frames[:300]]
        _require(all(s.face for s in collected), "all collected frames show a face")
        _require(sum(1 for t in exact_times if t < 60000) == 299, "299 face frames before the 60 s deadline")
        _require(sum(1 for t in exact_times if t < 90000) == 300, "exactly 300 face frames before the 90 s deadline")
        usable_l = [s.eye_w_l >= spec.signals.min_eye_width_px for s in collected]
        _require(usable_l == [False] * 299 + [True], "only the 300th frame has a usable Left eye")
        wanted = {
            "e_l": 0.30,  # 표본이 하나(300번째 프레임)뿐이다: ceil(0.9 x 1) = 1번째
            "e_r": 0.28,  # 300개 모두 0.28
            "b_l": 0.20,
            "b_r": 0.24,
            "jaw0": 0.05,
            "gaze_h0": 0.0,
            "gaze_v0": 0.0,
        }
        for name, value in wanted.items():
            _require(abs(got[name] - value) < 1e-9, f"baseline {name} is {got[name]}, the script intends {value}")
        _require(all(abs(a - b) < 1e-9 for a, b in zip(got["r0"], IDENTITY_R0)), f"R0 is {got['r0']}")
        # 얼굴 시간: 299장 x 200 ms(59600의 프레임은 60000의 프레임까지 400 ms 가운데 200 ms) = 59800 ms,
        # 60000의 프레임이 200 ms를 더해 90초 마감에서 60000 ms(비율 0.6667).
        every = exact_times + [90000]
        held = sum(min(later - earlier, ctx.spec.clock.max_hold_ms) for earlier, later in zip(every, every[1:]))
        _require(held == 60000 and held - 200 == 59800, f"face time at the 90 s deadline is {held} ms")
        same, _, shut = computed[SECTION_CLOSURE][-3:]
        _require(abs(same["closure"]) < 1e-9 and abs(shut["closure"] - 0.6) < 1e-9, "closure after completion: 0 and 0.6")

    # (4) 눈꺼풀 블렌드셰이프의 분모 하한. Left 눈의 B가 0.50이라 b_closed − B = 0.10이 b_span_min 0.20에 못 미친다.
    floor_baseline = Baseline(
        e_l=0.30,
        e_r=0.30,
        b_l=0.50,
        b_r=0.30,
        r0=IDENTITY_R0,
        jaw0=0.08,
        gaze_h0=0.0,
        gaze_v0=0.0,
        t_complete_ms=0,
        status=BASELINE_OK,
    )
    floor_blinks = ((0.60, 0.30), (0.66, 0.45), (0.75, 0.54))
    floor_frames = tuple(
        calibration_frame(100 * index, (0.30, 0.30), blinks, 0.08) for index, blinks in enumerate(floor_blinks)
    )

    def check_floor(computed: Mapping[str, Any]) -> None:
        got = [row["closure"] for row in computed[SECTION_CLOSURE]]
        # EAR 쪽은 0(EAR = E). Left: (blink − 0.50) / 0.20 = 0.5, 0.8, 1.25(1로 자른다). Right: (blink − 0.30) / 0.30
        # = 0, 0.5, 0.8. c_e = 0.5·s_e, closure = 두 눈 평균.
        wanted = ((0.25 + 0.0) / 2, (0.4 + 0.25) / 2, (0.5 + 0.4) / 2)
        _require(all(abs(a - b) < 1e-9 for a, b in zip(got, wanted)), f"closure is {got}, the script intends {wanted}")

    return [
        _Case(
            name="baseline_normal",
            bundle=BUNDLE_BASELINE,
            description=(
                "정상 기준선. 6 fps로 63초, 신호가 프레임마다 조금씩 흩어져 있다(분위수와 중앙값의 정의가 드러나게). "
                "깜빡임, 눈만 돌림, 하품, 고개 돌림, 2초의 얼굴 소실이 수집 구간에 섞여 있다. 세션 60000 ms에 "
                "status ok로 완성되고(T_b = 60000), 그 프레임부터 provisional이 false이고 개인 기준선으로 정규화한다. "
                "E는 EAR의 90 분위수, B는 eyeBlink의 10 분위수(최근접 순위), R0·jaw0·gaze0는 중앙값이다. "
                "완성 뒤의 감김 한 번으로 개인 기준선의 closure를 확인한다."
            ),
            sections=_BASELINE_SECTIONS,
            frames=session(normal, 63000, _BASELINE_PROFILE, 11),
            check=check_baseline(BASELINE_OK, 60000),
        ),
        _Case(
            name="baseline_extended",
            bundle=BUNDLE_BASELINE,
            description=(
                "기준선 연장. 세션 8~42초에 얼굴이 없어서 60초 시점에는 얼굴 프레임이 300개에 못 미치고 얼굴 검출 "
                "비율도 0.5 아래다. 30초 연장한 90000 ms에 완성된다(T_b = 90000). 카메라가 옆에 있어 기준 yaw가 "
                "15도를 넘으므로 status는 off_axis다."
            ),
            sections=_BASELINE_SECTIONS,
            frames=session(extended, 92500, replace(_BASELINE_PROFILE, yaw=18.0), 12),
            check=check_baseline(BASELINE_OFF_AXIS, 90000),
        ),
        _Case(
            name="baseline_fallback",
            bundle=BUNDLE_BASELINE,
            description=(
                "기준선 대체. 얼굴이 처음 12초와 178초 이후에만 보인다. 60, 90, 120, 150, 180초의 마감을 모두 "
                "통과하지 못해 180000 ms에 전역 상수로 대체한다(status fallback_global, T_b = 180000). 그 뒤의 "
                "프레임은 provisional이 false이고 전역 상수로 closure를 계산하며 pose_valid는 계속 false다."
            ),
            sections=_BASELINE_SECTIONS,
            frames=session(fallback, 183500, _BASELINE_PROFILE, 13),
            check=check_baseline(BASELINE_FALLBACK_GLOBAL, 180000),
        ),
        _Case(
            name="closure_yaw_weighting",
            bundle=BUNDLE_BASELINE,
            description=(
                "yaw에 따른 눈 가중. 기준 자세가 (10, 5, 2)도이고 두 눈의 E, B가 다른 기준선을 넣었다. Left 눈은 c = 0.9, "
                "Right 눈은 c = 0.2인 얼굴을 기준 대비 yaw 0, ±10, ±12, ±17, ±20, ±23, ±28, ±37, ±44도에서 본다. "
                "|yaw_rel| 15도까지는 두 눈 평균(0.55), 25도부터는 가까운 눈만(yaw_rel > 0이면 Right의 0.2, 아니면 "
                "Left의 0.9), 40도를 넘으면 closure가 유효하지 않다. yaw_rel은 Euler(R0ᵀ·R)이라 단순 차와 조금 "
                "다르다. 끝의 세 프레임은 Left 눈, Right 눈, 두 눈 모두를 쓸 수 없는 경우(눈 폭 6 px 미만)다."
            ),
            sections=_CLOSURE_SECTIONS,
            frames=tuple(weighting_frames),
            baseline=weighting_baseline,
            check=check_weighting,
        ),
        _Case(
            name="closure_no_face",
            bundle=BUNDLE_BASELINE,
            description=(
                "얼굴 없음. 얼굴 프레임(closure 0.35), 얼굴 없는 프레임 둘, 얼굴 프레임(closure 0.6). 얼굴이 없으면 "
                "closure_valid와 pose_valid가 false이고 모든 실수 필드가 0.0이다(이전 값을 잇거나 보간하지 않는다)."
            ),
            sections=_CLOSURE_SECTIONS,
            frames=(
                plain_frame(1, lid_open),
                synth.no_face_frame(101, 640, 480),
                synth.no_face_frame(201, 640, 480),
                plain_frame(301, lid_half),
            ),
            baseline=_personal_baseline(spec),
            check=check_no_face,
        ),
        _Case(
            name="baseline_odd_sample_count",
            bundle=BUNDLE_BASELINE,
            description=(
                "표본이 홀수 개인 기준선과 눈 조건에 의한 연장 두 번(손으로 놓은 프레임, 신호는 몇 가지 값뿐이다). 얼굴 "
                "프레임은 331개다. 처음 1분의 300개(200 ms 간격)는 Right 눈의 폭이 3 px이라 쓸 수 없다: 60초 마감에서 "
                "프레임 수와 얼굴 시간은 충분하지만 Right 눈의 E를 정할 표본이 없어 연장한다. 65, 72, 79초의 셋은 Right "
                "눈의 EAR이 0.50이다: 90초 마감에서 E_r = 0.50이 e_max 0.45를 넘어 다시 연장한다. 90~117초의 28개는 Right "
                "눈의 EAR이 0.28이다: 120초 마감에서 E_r은 표본 31개의 28번째(ceil(0.9 x 31)) 값 0.28이라 완성된다"
                "(T_b = 120000, status ok). 331개 가운데 앞의 165개는 pitch 0, 턱 0.05, 눈동자 합성값 0이고 뒤의 166개는 "
                "pitch 4, 턱 0.09, 눈동자 합성값 0.1과 0.06이다. 표본이 홀수 개이면 중앙값은 가운데 표본(166번째) 하나라서 "
                "R0 = Rx(4도), jaw0 = 0.09, gaze_h0 = 0.1, gaze_v0 = 0.06이다. 가운데 두 표본의 평균을 내면 pitch 2, "
                "턱 0.07이 된다. 완성 뒤의 세 프레임: 기준선과 같은 얼굴(pitch_rel 0, jaw_rel 0), 앞쪽 자세의 얼굴"
                "(pitch_rel -4, jaw_rel -0.04, gaze_h_rel -0.1, gaze_v_rel -0.06), 두 눈을 0.6만큼 감은 얼굴(closure 0.6)."
            ),
            sections=_BASELINE_SECTIONS,
            frames=tuple(odd_frames),
            check=check_odd,
        ),
        _Case(
            name="baseline_burst_frames",
            bundle=BUNDLE_BASELINE,
            description=(
                "프레임이 몰려서 오는 세션의 얼굴 검출 비율(손으로 놓은 프레임, 모든 얼굴 프레임의 신호가 같다). 얼굴 "
                "프레임 하나는 다음 프레임까지, 최대 200 ms를 대표하고 마감을 넘는 부분은 세지 않는다. 처음 59초는 초마다 "
                "다섯 장(0, 75, 150, 225, 300 ms)이라 초당 75 x 4 + 200 = 500 ms, 마지막 1초는 25 ms 간격의 다섯 장"
                "(100 + 200 ms)과 59900의 한 장(마감까지 100 ms)이다. 60초 마감에서 얼굴 프레임은 301개로 충분하지만 얼굴 "
                "시간이 29900 ms(비율 0.4983)라 연장한다. 200 ms 상한 없이 프레임 간격을 그대로 더하거나, 59900의 프레임이 "
                "마감 뒤까지 대표한 시간(다음 프레임은 60100)을 더하거나, 얼굴 없는 프레임(500, 1500, 2500)이 대표한 "
                "시간을 더하면 0.5 이상이 되어 60초에 완성된다. 60100부터 200 ms 간격의 75장이 15000 ms를 더해 90초 "
                "마감의 얼굴 시간은 정확히 45000 ms, 비율은 정확히 0.5다: '이상'이므로 완성된다(T_b = 90000)."
            ),
            sections=_BASELINE_SECTIONS,
            frames=tuple(burst_frames),
            check=check_burst,
        ),
        _Case(
            name="baseline_exactly_300_frames",
            bundle=BUNDLE_BASELINE,
            description=(
                "얼굴 프레임이 정확히 300개인 기준선과 Left 눈의 표본(손으로 놓은 프레임, 모든 프레임의 신호가 같다). "
                "0~59600 ms의 299개(200 ms 간격)는 Left 눈의 폭이 3 px이라 쓸 수 없고, 300번째 프레임(60000 ms)에서만 "
                "Left 눈을 쓸 수 있다. 그 뒤 90000 ms까지 프레임이 없다. 60초 마감(60000의 프레임이 오면 그 프레임을 넣기 "
                "전에 검사한다)에서는 얼굴 프레임이 299개이고 Left 눈의 E를 정할 표본이 없어 연장한다. 90초 마감에서는 얼굴 "
                "프레임이 정확히 300개다: min_face_frames 300 '이상'이므로 완성된다(T_b = 90000, status ok). 얼굴 시간은 "
                "299 x 200 + 200 = 60000 ms(비율 0.6667)다. E_l은 Left 눈의 표본 하나(0.30)이고 E_r은 0.28이다. 프레임 수를 "
                "'초과'로 세면 완성되지 않고, 쓸 수 없는 Left 눈의 EAR(0.0)을 표본에 넣으면 E_l이 270번째 값인 0.0이 되어 "
                "완성되지 않으며, Left 눈의 표본이 없는 60초 마감에서 분위수를 구하려 들면 빈 목록에서 오류가 난다. 완성 "
                "뒤의 세 프레임: 기준선과 같은 얼굴 둘(closure 0), 두 눈을 0.6만큼 감은 얼굴(closure 0.6)."
            ),
            sections=_BASELINE_SECTIONS,
            frames=tuple(exact_frames),
            check=check_exact,
        ),
        _Case(
            name="closure_blink_span_floor",
            bundle=BUNDLE_BASELINE,
            description=(
                "눈꺼풀 블렌드셰이프 쪽 감김 s_e의 분모 하한. s_e = clip((blink − B) / max(b_closed − B, b_span_min), 0, 1)"
                "이고 b_closed = 0.6, b_span_min = 0.2다. 넣어 준 기준선은 Left 눈의 B가 0.50(b_closed − B = 0.10이라 "
                "분모는 하한 0.20), Right 눈의 B가 0.30(분모 0.30)이다. EAR은 두 눈 모두 E와 같아 EAR 쪽 감김은 0이다. "
                "프레임 셋의 (eyeBlinkLeft, eyeBlinkRight)는 (0.60, 0.30), (0.66, 0.45), (0.75, 0.54)이고 closure는 "
                "0.125, 0.325, 0.45다(셋째 프레임의 Left 눈은 1.25가 1로 잘린다). 하한을 빼면 첫 프레임이 0.25가 된다."
            ),
            sections=_CLOSURE_SECTIONS,
            frames=floor_frames,
            baseline=floor_baseline,
            check=check_floor,
        ),
    ]


# --- 감김 검출 ---------------------------------------------------------------

_RUN_SECTIONS: tuple[str, ...] = (SECTION_CLOSURE, SECTION_RUNS, SECTION_COUNTERS)


def _run_cases(ctx: Context) -> list[_Case]:
    spec = ctx.spec
    baseline = _personal_baseline(spec)

    def check_runs(truncated: Sequence[bool], dropped_frames: int = 0) -> Callable[[Mapping[str, Any]], None]:
        def check(computed: Mapping[str, Any]) -> None:
            got = [run["truncated"] for run in computed[SECTION_RUNS]]
            _require(got == list(truncated), f"runs (truncated flags) are {got}, the script intends {list(truncated)}")
            dropped = computed[SECTION_COUNTERS]["dropped_frames"]
            _require(dropped == dropped_frames, f"{dropped} frames were dropped, the script intends {dropped_frames}")

        return check

    blinks = synth.sequence(
        Open(210),
        Blink(200),
        Open(300),
        Blink(300, depth=0.9),
        Open(310),
        Blink(120),
        Open(290),
        Blink(250, depth=0.45),
        Open(200),
    )
    blinks_ms = sum(segment.duration_ms for segment in blinks)

    def check_600(computed: Mapping[str, Any]) -> None:
        check_runs([False])(computed)
        run = computed[SECTION_RUNS][0]
        _require(run["duration_ms"] == 600 and run["velocity_valid"], f"run is {run}")

    truncated = [
        Closure(hold_ms=500, start_ms=200),
        FaceMissing(300, start_ms=400),
        Closure(hold_ms=1000, start_ms=1500),
        Closure(hold_ms=1000, start_ms=3500),
        FaceMissing(700, start_ms=3800),
    ]

    # 중복 타임스탬프: 눈을 뜬 구간에 "감긴 눈" 프레임을 같은 t_ms와 더 이른 t_ms로 끼워 넣는다. 버리지 않으면 결과가 달라진다.
    two_blinks = [Blink(200, start_ms=210), Blink(300, depth=0.9, start_ms=710)]
    clean = list(_render(spec, two_blinks, 1300))
    closed_eye = clean[8]
    duplicated = clean[:15] + [replace(closed_eye, t_ms=clean[14].t_ms)] + clean[15:21]
    duplicated += [replace(closed_eye, t_ms=clean[20].t_ms - 50)] + clean[21:]

    def check_duplicate(computed: Mapping[str, Any]) -> None:
        check_runs([False, False], dropped_frames=2)(computed)
        _require(signals.compute(closed_eye, spec).ear_l < 0.15, "the inserted frame must show closed eyes")
        _require(len(computed[SECTION_CLOSURE]) == len(clean), "dropped frames must not produce a frame state")

    lingering = [
        Linger(1000, 0.4, start_ms=200),
        Closure(hold_ms=300, start_ms=1500),
        Linger(800, 0.4, ramp_ms=100, start_ms=1800),
        Closure(hold_ms=200, start_ms=2400),
        Closure(hold_ms=300, start_ms=3500),
        Linger(800, 0.4, ramp_ms=100, start_ms=3800),
    ]

    def check_lingering(computed: Mapping[str, Any]) -> None:
        check_runs([False, False])(computed)
        across, before = computed[SECTION_RUNS]
        # 첫 구간은 머문 구간을 건너 이어지지만(끝 − 시작 > 1100 ms) 길이는 0.5 이상이던 두 토막의 합이다.
        # 0.4에 머문 약 470 ms는 들어가지 않는다(예전 정의에서는 1225 ms였다).
        _require(across["end_ms"] - across["start_ms"] > 1100, "the first run must span the lingering stretch")
        _require(700 < across["duration_ms"] < 800, "its length must be the sum of the two closed spans only")
        _require(before["duration_ms"] < 500, "the second run ends before the lingering stretch")
        _require(before["duration_ms"] == before["end_ms"] - before["start_ms"], "a single span: length = end - start")
        _require(before["t_ms"] - before["end_ms"] > 500, "and is emitted only when the lids open")

    # 감는 속도의 거슬러 보기 (쓸 수 없는 쌍에서 끊는다). 네 토막을 한 세션에 잇는다. 시각은 세션 시간이다.
    #  A   0~840: 오르는 쌍 셋 뒤 640 ms 끊김(> hold_ms) -> 잘린 구간. 끊김 뒤 첫 표본에서 새 구간이 시작한다.
    #             그 구간의 감는 속도는 구간 안의 쌍(0.90 -> 0.92, 40 ms)의 0.5/s이고, 잘린 앞 구간의 증가율이 아니다.
    #  B   2000~: 오르는 쌍(0.10 -> 0.45, 33 ms) 뒤 약 9초의 프레임 끊김(10초 이하라 초기화는 없다), 끊김 뒤 첫 표본이
    #             감긴 눈. 감는 쪽에 쓸 수 있는 쌍이 없으므로 velocity_valid = false.
    #  C  13000~: 같은 오르는 쌍 뒤 얼굴 없는 프레임 여덟(유효 표본 간격 267 ms > max_dt_ms). B와 같다.
    #  D  15000~: 같은 오르는 쌍 뒤 얼굴 없는 프레임 하나(유효 표본 간격 67 ms). 쌍이 이어지므로 거슬러 본다.
    lookback_plan: list[tuple[int, _Shot]] = [
        (0, _Shot(level=0.1)),
        (40, _Shot(level=0.6)),
        (80, _Shot(level=0.82)),
        (120, _Shot(level=0.95)),
        (760, _Shot(level=0.9)),
        (800, _Shot(level=0.92)),
        (840, _Shot(level=0.1)),
        (2033, _Shot(level=0.45)),
        (11000, _SHUT),
        (11080, _Shot(level=0.1)),
        (13033, _Shot(level=0.45)),
        (13066, _Shot(face=False)),
        (13300, _SHUT),
        (13380, _Shot(level=0.1)),
        (15033, _Shot(level=0.45)),
        (15066, _Shot(face=False)),
        (15100, _SHUT),
        (15180, _Shot(level=0.1)),
    ]
    lookback_times = [0, 40, 80, 120, 760, 800, 840, 2000, 2033, 11000, 11040, 11080, 13000, 13033]
    lookback_times += [13066, 13100, 13133, 13166, 13200, 13233, 13266, 13300, 13340, 13380]
    lookback_times += [15000, 15033, 15066, 15100, 15140, 15180, 15220]
    rise_rate = (0.45 - 0.1) * 1000.0 / 33.0  # 끊김 앞에서 잰 증가율(1/s). 새 구간으로 넘어오면 안 되는 값

    def check_lookback(computed: Mapping[str, Any]) -> None:
        check_runs([True, False, False, False, False])(computed)
        cut, after_cut, after_gap, after_invalid, chained = computed[SECTION_RUNS]
        _require(not cut["velocity_valid"], "a truncated run has no velocities")
        _require(after_cut["start_ms"] == _FIRST_T_MS + 760, "the run after the cut starts at its first sample")
        _require(after_cut["velocity_valid"], "a rise inside the run gives a closing velocity")
        _require(abs(after_cut["peak_closing_velocity"] - 0.5) < 1e-6, "0.90 -> 0.92 in 40 ms is 0.5 /s")
        for run in (after_gap, after_invalid):
            _require(not run["velocity_valid"], "no usable pair on the closing side: velocity_valid must be false")
            _require(run["peak_closing_velocity"] == 0.0 and run["peak_opening_velocity"] == 0.0, "velocities are 0.0")
        _require(chained["velocity_valid"], "the pair across one invalid frame is usable (67 ms)")
        _require(abs(chained["peak_closing_velocity"] - rise_rate) < 1e-6, "the lookback reaches the pair before entry")

    # 반올림과 t_ms의 원점. 첫 t_ms가 1.7e12인 세션. 표본은 세션 시간 100(0.05), 133(0.95), 173(0.4045), 213(0.05)이다.
    # 0.5를 위로 지난 쌍 안의 오프셋은 16.5 ms에서 7e-5 ms 모자란 값이다: 쌍 안의 오프셋을 반올림하면 16이고 구간은
    # 50 ms(min_duration_ms와 같아서 남는다), 절대 시각을 더한 뒤 반올림하면 그 크기의 Double(간격 2.4e-4 ms)에서 합이
    # x.5로 올라가 17이 되고 구간은 49 ms라 버려진다. 7e-5 ms는 두 표본의 closure가 허용 오차(1e-6)만큼 달라도
    # 넘지 못하는 거리다(오프셋의 변화는 3.7e-5 ms 이하). 아래로 지난 자리는 133의 표본에서 정확히 33 ms 뒤다.
    tie_peak = 0.05 + 0.45 * 33.0 / (16.5 - 7e-5)
    tie_half = tie_peak - (tie_peak - 0.5) * 40.0 / 33.0
    tie_plan = [(0, _Shot(level=0.05)), (133, _Shot(level=tie_peak)), (173, _Shot(level=tie_half))]
    tie_plan += [(213, _Shot(level=0.05))]
    tie_times = [0, 33, 66, 100, 133, 173, 213, 246]

    def check_tie(computed: Mapping[str, Any]) -> None:
        check_runs([False])(computed)
        run = computed[SECTION_RUNS][0]
        rows = {row["t_ms"]: row for row in computed[SECTION_CLOSURE]}
        before, entry = rows[LARGE_FIRST_T_MS + 100], rows[LARGE_FIRST_T_MS + 133]
        enter = spec.run_detector.enter_thr
        gap_closure = entry["closure"] - before["closure"]
        offset = (enter - before["closure"]) / gap_closure * 33
        _require(6e-5 < 16.5 - offset < 8e-5, f"the in-pair offset is {offset!r}, the script intends 16.5 - 7e-5")
        # 두 표본의 closure가 1e-6씩 달라질 때 오프셋이 움직이는 최대 폭. 이보다 x.5에서 멀어야 한다.
        slack = 1e-6 * 33 * ((enter - before["closure"]) + (entry["closure"] - enter)) / (gap_closure * gap_closure)
        _require(slack < 5e-5, f"a closure error of 1e-6 moves the offset by {slack!r} ms")
        absolute = float(before["t_ms"]) + offset  # 절대 시각에 더하면 마지막 자리가 잘려 x.5가 된다
        _require(absolute - float(before["t_ms"]) == 16.5, "at this origin the absolute sum must land on x.5")
        _require(run["start_ms"] == before["t_ms"] + 16, f"start_ms is {run['start_ms']}")
        _require(run["end_ms"] == entry["t_ms"] + 33, f"end_ms is {run['end_ms']}")
        _require(run["duration_ms"] == spec.run_detector.min_duration_ms, f"the run is {run['duration_ms']} ms long")

    # 토막이 여럿인 구간(2026-11-13 소유자 결정: 길이는 0.5 이상이던 시간의 합). 100 ms 간격, 값은 계단 모양.
    half_open = _Shot(level=0.4)  # 이탈(0.3)과 진입(0.5) 사이
    no_face = _Shot(face=False)
    spans_plan = [
        (0, _OPEN),
        # A: 토막 셋. 0.4에 300 ms, 그리고 600 ms 머문다
        (500, _SHUT),
        (800, half_open),
        (1200, _SHUT),
        (1400, half_open),
        (2000, _SHUT),
        (2300, _OPEN),
        # B: 토막 안의 끊김(두 표본이 모두 0.5 이상, 400 ms)은 길이에 들어간다. 그 뒤 0.4로 내려간 채 얼굴이
        #    사라졌다가(HOLD) 400 ms 만에 감긴 눈으로 돌아오면 새 토막은 그 프레임의 시각에서 시작한다(보간 불가)
        (3000, _SHUT),
        (3200, no_face),
        (3500, _SHUT),
        (3700, half_open),
        (3800, no_face),
        (4100, _SHUT),
        (4300, _OPEN),
    ]

    def check_spans(computed: Mapping[str, Any]) -> None:
        check_runs([False, False])(computed)
        three, held = computed[SECTION_RUNS]
        first = _FIRST_T_MS
        # 0.0 -> 0.9는 100 ms 중 56 ms에, 0.4 -> 0.9는 20 ms에, 0.9 -> 0.4는 80 ms에, 0.9 -> 0.0은 44 ms에 0.5를 지난다
        _require((three["start_ms"], three["end_ms"]) == (first + 456, first + 2244), f"three-span run is {three}")
        _require(three["duration_ms"] == (780 - 456) + (1380 - 1120) + (2244 - 1920), "sum of the three spans")
        _require((held["start_ms"], held["end_ms"]) == (first + 2956, first + 4244), f"run with a hold is {held}")
        _require(held["duration_ms"] == (3680 - 2956) + (4244 - 4100), "the dropout counts, the dwell does not")
        _require(three["velocity_valid"] and held["velocity_valid"], "both runs end normally")

    cut_spans_plan = [
        (0, _OPEN),
        # A: 둘째 토막에서 얼굴이 사라져 잘린다. end_ms는 0.5 이상이던 마지막 프레임(1100)
        (500, _SHUT),
        (700, half_open),
        (1000, _SHUT),
        (1200, no_face),
        (1900, _OPEN),
        # B: 둘째 토막 뒤 0.4로 내려간 다음에 얼굴이 사라져 잘린다. 0.5를 아래로 지난 시각(3580)을 보간해 두었지만
        #    잘린 구간의 끝은 0.5 이상이던 마지막 프레임(3500)이다
        (3000, _SHUT),
        (3200, half_open),
        (3400, _SHUT),
        (3600, half_open),
        (3800, no_face),
        (4500, _OPEN),
        # C: 둘째 토막 뒤 무효 프레임 하나, 그다음 프레임에서 눈을 떴다(HOLD에서 c < exit: 잘린 구간)
        (5000, _SHUT),
        (5200, half_open),
        (5300, _SHUT),
        (5400, no_face),
        (5500, _OPEN),
    ]

    def check_cut_spans(computed: Mapping[str, Any]) -> None:
        check_runs([True, True, True])(computed)
        first = _FIRST_T_MS
        expected = [
            (first + 456, first + 1100, (680 - 456) + (1100 - 920), first + 1700),
            (first + 2956, first + 3500, (3180 - 2956) + (3500 - 3320), first + 4300),
            (first + 4956, first + 5300, (5180 - 4956) + (5300 - 5220), first + 5500),
        ]
        got = [(run["start_ms"], run["end_ms"], run["duration_ms"], run["t_ms"]) for run in computed[SECTION_RUNS]]
        _require(got == expected, f"truncated multi-span runs are {got}, the script intends {expected}")

    # 정점이 뒤의 표본으로 옮겨지는 구간과 두 속도. 시각은 세션 시간, 값은 계단 모양이다.
    #  A   0~766: 0.55까지 감았다가(200 ms 쌍, 2.5/s) 0.42로 내려가고(33 ms 쌍, 3.94/s) 0.54로 다시 오른 뒤(33 ms 쌍,
    #             3.64/s. 진폭 0.55를 넘지 않아 아직 "정점 뒤의 증가") 0.82로 오른다(2.8/s. 정점이 옮겨진다).
    #             감는 속도는 정점까지의 쌍 전부에서 가장 큰 3.64/s이고, 뜨는 속도는 새 정점 뒤의 쌍만 본 3.7/s다
    #             (앞 봉우리에서 내려간 3.94/s는 버린다).
    #  B   2000~: 0.9, 0.6, 0.9, 0.1. 같은 최댓값이 다시 나와도 정점은 처음 닿은 표본이다. 뜨는 속도는 그 뒤의 쌍
    #             전부에서 가장 큰 9.09/s(0.9 → 0.6, 33 ms)이고 감는 속도는 진입 쌍(0.1 → 0.9)의 8.0/s다.
    #  C   4000~: 구간 둘. 앞 구간은 정점(0.9) 뒤에 0.55 → 0.86으로 급히 올랐다가(9.39/s. 정점을 넘지 않는다) 뜬다.
    #             그 증가율은 어느 구간의 감는 속도도 아니다: 앞 구간은 8.0/s, 뒤 구간(정점이 옮겨진다)은 2.7/s다.
    humps_plan: list[tuple[int, _Shot]] = [
        (0, _Shot(level=0.05)),
        (400, _Shot(level=0.55)),
        (433, _Shot(level=0.42)),
        (466, _Shot(level=0.54)),
        (566, _Shot(level=0.82)),
        (666, _Shot(level=0.45)),
        (766, _Shot(level=0.1)),
        (2200, _SHUT),
        (2233, _Shot(level=0.6)),
        (2333, _SHUT),
        (2533, _Shot(level=0.1)),
        (4200, _SHUT),
        (4300, _Shot(level=0.55)),
        (4333, _Shot(level=0.86)),
        (4433, _Shot(level=0.1)),
        (5400, _Shot(level=0.55)),
        (5500, _Shot(level=0.82)),
        (5600, _Shot(level=0.1)),
    ]
    humps_times = [0, 200, 400, 433, 466, 566, 666, 766, 2000, 2100, 2200, 2233, 2333, 2533, 2633]
    humps_times += [4000, 4100, 4200, 4300, 4333, 4433, 5000, 5200, 5400, 5500, 5600, 5700]

    def check_humps(computed: Mapping[str, Any]) -> None:
        check_runs([False, False, False, False])(computed)
        first = _FIRST_T_MS
        moved, tied, before, after = computed[SECTION_RUNS]
        expected = [
            # (start_ms, end_ms, duration_ms, 감는 속도, 뜨는 속도)
            (first + 380, first + 652, 33 + (652 - 455), 0.12 / 0.033, 0.37 / 0.1),
            (first + 2150, first + 2433, 283, 0.8 / 0.1, 0.3 / 0.033),
            (first + 4150, first + 4380, 230, 0.8 / 0.1, 0.76 / 0.1),
            (first + 5378, first + 5544, 166, 0.27 / 0.1, 0.72 / 0.1),
        ]
        for run, (start, end, length, closing, opening) in zip((moved, tied, before, after), expected, strict=True):
            _require((run["start_ms"], run["end_ms"], run["duration_ms"]) == (start, end, length), f"run is {run}")
            _require(run["velocity_valid"], f"both velocities must be measured: {run}")
            _require(abs(run["peak_closing_velocity"] - closing) < 1e-6, f"closing velocity of {run}")
            _require(abs(run["peak_opening_velocity"] - opening) < 1e-6, f"opening velocity of {run}")
        _require(abs(moved["amplitude"] - 0.82) < 1e-6 and abs(tied["amplitude"] - 0.9) < 1e-6, "amplitudes")

    # 감긴 표본이 하나뿐인 채로 잘리는 구간. 0.42 → 0.9(100 ms)는 17 ms에 0.5를 지나므로 길이는 83 ms다.
    #  (1) 그 표본 뒤 프레임이 600 ms 끊긴다. (2) 그 표본 뒤 얼굴이 사라진다(500 ms를 넘긴 첫 무효 프레임에서 잘린다).
    #  (3) 0.05 → 0.9는 53 ms에 0.5를 지나 길이가 47 ms다: 잘린 구간에도 min_duration_ms(50)를 적용하므로 버려진다.
    lone_plan: list[tuple[int, _Shot]] = [
        (0, _Shot(level=0.05)),
        (1000, _Shot(level=0.42)),
        (1100, _SHUT),
        (1700, _Shot(level=0.05)),
        (3100, _Shot(level=0.42)),
        (3200, _SHUT),
        (3300, _Shot(face=False)),
        (3900, _Shot(level=0.05)),
        (5200, _SHUT),
        (5800, _Shot(level=0.05)),
    ]
    lone_times = list(range(0, 1200, 100)) + list(range(1700, 5300, 100)) + [5800, 5900]

    def check_lone(computed: Mapping[str, Any]) -> None:
        check_runs([True, True])(computed)
        first = _FIRST_T_MS
        got = [(run["start_ms"], run["end_ms"], run["duration_ms"], run["t_ms"]) for run in computed[SECTION_RUNS]]
        expected = [(first + 1017, first + 1100, 83, first + 1700), (first + 3117, first + 3200, 83, first + 3800)]
        _require(got == expected, f"truncated one-sample runs are {got}, the script intends {expected}")
        closed = [row["t_ms"] - first for row in computed[SECTION_CLOSURE] if row["closure"] >= 0.5]
        _require(closed == [1100, 3200, 5200], f"closed samples are at {closed}")

    # 정점이 옮겨지기 전에 증가가 두 번 있는 구간(큰 것이 먼저). 시각은 세션 시간, 값은 계단 모양이다.
    #   0.05 → 0.55(200 ms, 진입 쌍 2.5/s) → 0.42 → 0.54(33 ms, 3.64/s) → 0.45 → 0.48(33 ms, 0.91/s)
    #   → 0.82(100 ms, 3.4/s. 정점이 이 표본으로 옮겨진다) → 0.45(100 ms, 3.7/s) → 0.1
    # 정점을 넘지 못한 증가율은 "가장 큰 것"을 들고 있다가 정점이 옮겨질 때 감는 속도에 합친다: 3.64/s.
    # "마지막 것"을 들고 있으면 0.91/s가 남아 감는 속도가 3.4/s가 된다(`runs_two_humps`에는 그런 증가가 한 번뿐이다).
    rises_plan: list[tuple[int, _Shot]] = [
        (0, _Shot(level=0.05)),
        (400, _Shot(level=0.55)),
        (433, _Shot(level=0.42)),
        (466, _Shot(level=0.54)),
        (499, _Shot(level=0.45)),
        (532, _Shot(level=0.48)),
        (632, _Shot(level=0.82)),
        (732, _Shot(level=0.45)),
        (832, _Shot(level=0.1)),
    ]
    rises_times = [0, 200, 400, 433, 466, 499, 532, 632, 732, 832, 1000]

    def check_rises(computed: Mapping[str, Any]) -> None:
        check_runs([False])(computed)
        (run,) = computed[SECTION_RUNS]
        first = _FIRST_T_MS
        # 토막 셋: 380~413, 455~481, 538~718
        _require((run["start_ms"], run["end_ms"]) == (first + 380, first + 718), f"run is {run}")
        _require(run["duration_ms"] == (413 - 380) + (481 - 455) + (718 - 538), f"run is {run}")
        _require(run["velocity_valid"] and abs(run["amplitude"] - 0.82) < 1e-6, f"run is {run}")
        _require(abs(run["peak_closing_velocity"] - 0.12 / 0.033) < 1e-6, "the larger of the two rises, not the last")
        _require(abs(run["peak_opening_velocity"] - 0.37 / 0.1) < 1e-6, f"opening velocity of {run}")

    return [
        _Case(
            name="runs_blinks_30fps",
            bundle=BUNDLE_RUNS,
            description=(
                "30 fps의 깜빡임 넷: 200 ms, 300 ms(깊이 0.9), 120 ms, 250 ms(깊이 0.45). 앞의 셋은 감김 구간이 되고 "
                "깊이 0.45는 진입 임계 0.5에 못 미쳐 구간이 없다. start_ms와 end_ms는 0.5를 지난 시각을 앞뒤 표본으로 "
                "선형 보간하고 floor(x + 0.5)로 반올림한 값이다. 속도는 1/s 단위다."
            ),
            sections=_RUN_SECTIONS,
            frames=_render(spec, blinks, blinks_ms),
            baseline=baseline,
            check=check_runs([False, False, False]),
        ),
        _Case(
            name="runs_blinks_15fps",
            bundle=BUNDLE_RUNS,
            description=(
                "runs_blinks_30fps와 같은 눈꺼풀 움직임을 15 fps로 본 것. 앞의 두 깜빡임은 검출되고, 120 ms 깜빡임은 "
                "0.5 이상인 표본이 하나뿐이고 보간한 길이가 min_duration_ms(50) 미만이라 버려진다."
            ),
            sections=_RUN_SECTIONS,
            frames=_render(spec, blinks, blinks_ms, fps=15.0),
            baseline=baseline,
            check=check_runs([False, False]),
        ),
        _Case(
            name="runs_dropped_frames",
            bundle=BUNDLE_RUNS,
            description=(
                "runs_blinks_30fps와 같은 움직임에서 프레임을 확률 0.35로 무작위로 버리고(66개 중 29개가 빠졌다) "
                "시각을 ±6 ms 흔든 것. 프레임 "
                "간격이 고르지 않아도 보간과 속도는 실제 간격으로 계산한다. 첫 깜빡임은 정점 표본이 빠져 진폭이 "
                "작게 읽히고, 세 번째 깜빡임은 0.5 이상인 표본이 남지 않아 검출되지 않는다."
            ),
            sections=_RUN_SECTIONS,
            frames=_render(spec, blinks, blinks_ms, drop_prob=0.35, jitter_ms=6, seed=4),
            baseline=baseline,
            check=check_runs([False, False]),
        ),
        _Case(
            name="runs_one_sample_blink",
            bundle=BUNDLE_RUNS,
            description=(
                "15 fps에서 한 표본짜리 깜빡임 셋. 정점 0.9가 표본 하나에 걸린 깜빡임은 구간이 된다(앞뒤 표본으로 "
                "보간한 길이 약 60 ms). 정점 0.6이 걸린 깜빡임은 보간한 길이가 50 ms 미만이라 버려진다. 두 프레임 "
                "사이에 통째로 들어간 60 ms 깜빡임은 보이지 않는다."
            ),
            sections=_RUN_SECTIONS,
            frames=_render(
                spec,
                [Blink(100, depth=0.9, start_ms=160), Blink(100, depth=0.6, start_ms=560), Blink(60, start_ms=1003)],
                1400,
                fps=15.0,
            ),
            baseline=baseline,
            check=check_runs([False]),
        ),
        _Case(
            name="runs_closure_600ms",
            bundle=BUNDLE_RUNS,
            description=(
                "600 ms 감김(30 fps). closure가 0.5를 넘은 시간이 정확히 600 ms다. duration_ms는 히스테리시스의 "
                "이탈 임계(0.3)가 아니라 진입 임계(0.5)를 지난 두 시각의 차다."
            ),
            sections=_RUN_SECTIONS,
            frames=_render(spec, [Closure(hold_ms=475, start_ms=300)], 1325),
            baseline=baseline,
            check=check_600,
        ),
        _Case(
            name="runs_truncated_by_gap",
            bundle=BUNDLE_RUNS,
            description=(
                "감김 중에 유효 프레임이 끊기는 세 경우(30 fps). (1) 얼굴이 300 ms 사라졌다 돌아오면 hold_ms(500) "
                "이하이므로 같은 구간으로 이어진다. (2) 프레임 자체가 700 ms 끊기면 다음 프레임에서 먼저 구간을 "
                "닫고(truncated, end_ms는 0.5 이상이었던 마지막 프레임), 그 프레임을 OPEN에서 다시 처리해 새 구간이 "
                "시작된다. (3) 얼굴이 700 ms 사라지면 끊긴 시간이 500 ms를 넘은 첫 무효 프레임에서 잘린 구간을 낸다. "
                "잘린 구간과 보간할 수 없는 구간은 velocity_valid가 false다."
            ),
            sections=_RUN_SECTIONS,
            frames=_render(spec, truncated, 5000, gaps=((1800, 2500),)),
            baseline=baseline,
            check=check_runs([False, True, False, True, False]),
        ),
        _Case(
            name="runs_duplicate_timestamp",
            bundle=BUNDLE_RUNS,
            description=(
                "중복 타임스탬프. 눈을 뜬 구간에 감긴 눈의 프레임 둘을 끼워 넣었다. 하나는 직전 프레임과 같은 t_ms, "
                "하나는 더 이른 t_ms다. 둘 다 버리고 세며(counters.dropped_frames = 2), 프레임 상태와 감김 구간은 "
                "끼워 넣지 않았을 때와 같다."
            ),
            sections=_RUN_SECTIONS,
            frames=tuple(duplicated),
            baseline=baseline,
            check=check_duplicate,
        ),
        _Case(
            name="runs_lingering_lids",
            bundle=BUNDLE_RUNS,
            description=(
                "눈꺼풀이 임계값 사이(0.4)에 머무는 경우(30 fps). (1) 0.4에만 머물면 구간이 없다. (2) 감았다가 0.4에 "
                "머문 뒤 다시 감으면 한 구간이고, end_ms는 마지막으로 0.5를 아래로 지난 시각이다. duration_ms는 "
                "end_ms − start_ms가 아니라 0.5 이상이던 두 토막의 길이를 더한 값이다(머문 시간은 들어가지 않는다). "
                "(3) 감았다가 0.4에 머문 뒤 뜨면 구간은 0.3 아래로 내려간 프레임에서 발행되지만 end_ms는 0.5를 아래로 "
                "지난 시각이다."
            ),
            sections=_RUN_SECTIONS,
            frames=_render(spec, lingering, 5200),
            baseline=baseline,
            check=check_lingering,
        ),
        _Case(
            name="runs_velocity_after_gap",
            bundle=BUNDLE_RUNS,
            description=(
                "감는 속도의 거슬러 보기는 쓸 수 없는 쌍(직전 유효 표본과의 간격이 max_dt_ms 200을 넘는 쌍)에서 끊긴다. "
                "구간 다섯: (1) 640 ms 끊김으로 잘린 구간. (2) 그 끊김 뒤 첫 표본에서 시작한 구간: 감는 속도는 구간 "
                "안의 쌍(0.90 → 0.92, 40 ms)의 0.5/s이고 잘린 앞 구간의 증가율이 넘어오지 않는다. (3) 오르는 쌍"
                "(0.10 → 0.45, 33 ms) 뒤 약 9초의 프레임 끊김, 끊김 뒤 첫 표본이 감긴 눈: velocity_valid = false이고 "
                "두 속도는 0.0이다. (4) 같은 오르는 쌍 뒤 얼굴 없는 프레임 여덟(유효 표본 간격 267 ms): (3)과 같다. "
                "(5) 같은 오르는 쌍 뒤 얼굴 없는 프레임 하나(유효 표본 간격 67 ms): 쌍이 이어지므로 진입 두 표본 전의 "
                "증가율(약 10.6/s)이 감는 속도다. 시각은 프레임 시각 그대로 손으로 놓았다(세션 시간 = t_ms − 1)."
            ),
            sections=_RUN_SECTIONS,
            frames=_hand_frames(spec, lookback_times, lookback_plan),
            baseline=baseline,
            check=check_lookback,
        ),
        _Case(
            name="runs_rounding_large_t0",
            bundle=BUNDLE_RUNS,
            description=(
                "보간 시각의 반올림은 t_ms의 원점과 무관하다. 첫 t_ms가 1731465123457인 세션에서 closure 0.05, 0.95, "
                "0.4045, 0.05를 33 ms, 40 ms, 40 ms 간격으로 본다. 0.5를 위로 지난 시각은 직전 표본에서 16.49993 ms "
                "뒤다. 보간 시각 = 직전 표본의 시각 + floor(쌍 안의 오프셋 + 0.5)이므로 +16이고, 0.5를 아래로 지난 "
                "시각은 정점 표본의 33 ms 뒤라서 구간은 50 ms로 min_duration_ms(50)와 같아 남는다. 절대 시각에 "
                "오프셋을 더한 뒤 반올림하면 이 크기의 Double에서는 합이 x.5가 되어 +17, 구간이 49 ms가 되어 "
                "버려진다(구간이 하나도 나오지 않는다). 오프셋은 x.5에서 7e-5 ms 떨어져 있어서 closure가 허용 오차"
                "(1e-6)만큼 달라도 반올림이 갈리지 않는다."
            ),
            sections=_RUN_SECTIONS,
            frames=_hand_frames(spec, tie_times, tie_plan, first_t_ms=LARGE_FIRST_T_MS),
            baseline=baseline,
            check=check_tie,
        ),
        _Case(
            name="runs_multi_span",
            bundle=BUNDLE_RUNS,
            description=(
                "토막이 여럿인 구간(10 fps, 값은 계단 모양). 구간의 길이는 closure가 0.5 이상이던 토막들의 합이고 "
                "0.3~0.5에 머문 시간은 들어가지 않는다. (1) 감김 300 ms, 0.4에 머묾, 감김 200 ms, 0.4에 600 ms 머묾, "
                "감김 300 ms: 한 구간이고 duration_ms 908 = 324 + 260 + 324(end_ms − start_ms는 1788). 뒤 토막의 "
                "시작도 start_ms처럼 앞뒤 표본으로 보간한다. (2) 감긴 채 얼굴이 400 ms 사라졌다 돌아온 끊김은 앞뒤 "
                "표본이 모두 0.5 이상이라 토막 안이다(길이에 들어간다). 그 뒤 0.4로 내려간 채 얼굴이 사라졌다가 "
                "400 ms 만에 감긴 눈으로 돌아오면(HOLD에서 복귀) 새 토막은 보간할 수 없어 그 프레임의 시각에서 "
                "시작한다. duration_ms 868 = 724 + 144."
            ),
            sections=_RUN_SECTIONS,
            frames=_hand_frames(spec, range(0, 4600, 100), spans_plan),
            baseline=baseline,
            check=check_spans,
        ),
        _Case(
            name="runs_multi_span_truncated",
            bundle=BUNDLE_RUNS,
            description=(
                "토막이 둘인 구간이 잘리는 세 경우(10 fps). 잘린 구간의 end_ms는 closure가 0.5 이상이던 마지막 유효 "
                "프레임의 시각이고, duration_ms는 끝난 토막의 길이 + (그 시각 − 마지막 토막의 시작)이다. (1) 둘째 토막 "
                "도중에 얼굴이 사라져 500 ms 넘게 끊긴다. (2) 둘째 토막 뒤 0.4로 내려간 다음에 얼굴이 사라진다: "
                "0.5를 아래로 지난 시각을 보간해 두었어도 잘린 구간의 끝에는 쓰지 않는다. (3) 둘째 토막 뒤 무효 프레임 "
                "하나, 그다음 프레임에서 눈을 떴다(HOLD에서 이탈 임계 아래: 끊긴 시간이 짧아도 잘린 구간이다)."
            ),
            sections=_RUN_SECTIONS,
            frames=_hand_frames(spec, range(0, 5800, 100), cut_spans_plan),
            baseline=baseline,
            check=check_cut_spans,
        ),
        _Case(
            name="runs_two_humps",
            bundle=BUNDLE_RUNS,
            description=(
                "정점이 뒤의 표본으로 옮겨지는 구간의 두 속도(값은 계단 모양, 시각은 세션 시간). 구간 넷: (1) 0.55까지 "
                "감았다가(진입 쌍 2.5/s) 0.42로 내려가고(이탈 0.3과 진입 0.5 사이, 33 ms에 3.94/s) 0.54로 다시 "
                "올랐다가(3.64/s. 그때의 진폭 0.55를 넘지 않는다) 0.82까지 오른 뒤(2.8/s) 뜬다. 정점은 0.82의 "
                "표본이다. peak_closing_velocity는 정점까지의 모든 쌍 가운데 가장 큰 3.64/s(정점을 넘지 못했던 "
                "증가도 정점이 옮겨지면 감는 쪽이다), peak_opening_velocity는 새 정점 뒤의 쌍만 본 3.7/s다(앞 "
                "봉우리에서 내려간 3.94/s는 버린다). duration_ms 230 = 33 + 197. (2) 0.9, 0.6, 0.9, 0.1: 같은 "
                "최댓값이 다시 나와도 정점은 처음 닿은 표본이다. 뜨는 속도는 그 뒤의 가장 큰 감소율 9.09/s"
                "(0.9 → 0.6, 33 ms)다. (3)과 (4): 앞 구간은 정점 0.9 뒤에 0.55 → 0.86으로 급히 올랐다가(9.39/s) "
                "뜬다. 정점을 넘지 않았으므로 그 증가율은 앞 구간의 감는 속도(8.0/s)가 아니고, 다음 구간으로도 "
                "넘어가지 않는다(뒤 구간의 감는 속도는 2.7/s)."
            ),
            sections=_RUN_SECTIONS,
            frames=_hand_frames(spec, humps_times, humps_plan),
            baseline=baseline,
            check=check_humps,
        ),
        _Case(
            name="runs_one_sample_truncated",
            bundle=BUNDLE_RUNS,
            description=(
                "감긴 표본이 하나뿐인 채로 잘리는 구간(10 fps). (1) 0.42 → 0.9로 감긴 표본 하나 뒤 프레임이 600 ms "
                "끊긴다. 다음 프레임에서 잘린 구간이 나오고 end_ms는 그 감긴 표본의 시각, duration_ms는 보간한 "
                "시작부터 그 시각까지인 83 ms다. (2) 같은 표본 뒤 얼굴이 사라진다: 끊긴 시간이 500 ms를 넘은 첫 무효 "
                "프레임(600 ms 뒤)에서 같은 구간이 나온다. (3) 0.05 → 0.9로 감긴 표본 하나(보간한 길이 47 ms) 뒤 "
                "600 ms 끊긴다: 잘린 구간에도 min_duration_ms(50)를 적용하므로 구간이 나오지 않는다. 잘린 구간의 두 "
                "속도는 0.0이고 velocity_valid는 false다."
            ),
            sections=_RUN_SECTIONS,
            frames=_hand_frames(spec, lone_times, lone_plan),
            baseline=baseline,
            check=check_lone,
        ),
        _Case(
            name="runs_two_rises_before_peak",
            bundle=BUNDLE_RUNS,
            description=(
                "정점이 옮겨지기 전에 증가가 두 번 있는 구간(값은 계단 모양, 시각은 세션 시간). 0.55까지 감았다가"
                "(진입 쌍 2.5/s) 0.42로 내려가 0.54로 오르고(33 ms에 3.64/s) 0.45로 내려가 0.48로 오른 뒤(33 ms에 "
                "0.91/s) 0.82까지 오르고(100 ms에 3.4/s. 정점이 이 표본으로 옮겨진다) 뜬다. 정점을 넘지 못한 "
                "증가율은 가장 큰 것을 들고 있다가 정점이 옮겨질 때 감는 속도에 합친다: peak_closing_velocity는 "
                "3.64/s다. 마지막 증가율(0.91/s)만 들고 있으면 3.4/s가 된다(runs_two_humps에는 그런 증가가 한 "
                "번뿐이라 두 구현이 같은 값을 낸다). peak_opening_velocity는 새 정점 뒤의 3.7/s다. 토막이 셋이고 "
                "duration_ms 239 = 33 + 26 + 180(end_ms − start_ms는 338)."
            ),
            sections=_RUN_SECTIONS,
            frames=_hand_frames(spec, rises_times, rises_plan),
            baseline=baseline,
            check=check_rises,
        ),
    ]


# --- 스텝 특징 ---------------------------------------------------------------

_STEP_SECTIONS: tuple[str, ...] = (SECTION_RUNS, SECTION_STEPS, SECTION_COUNTERS)


def _five_minute_frames(spec: PipelineSpec, first_t_ms: int) -> tuple[RawFrame, ...]:
    """스텝 특징 5분 세션. 6 fps에 시각을 ±20 ms 흔들고 2%를 버려서 200 ms를 넘는 간격도 섞여 있다."""
    quiet = ((118000, 152000), (158000, 190000), (248000, 258000), (268000, 275000))  # 깜빡임을 넣지 않는 구간
    starts = [t for t in range(2000, 298000, 4500) if not any(lo <= t < hi for lo, hi in quiet)]
    durations = (303, 397, 491, 347)  # 5의 배수가 아니라서 정수 ms의 프레임이 임계값에 정확히 놓이지 않는다
    segments: list[Segment] = [
        Blink(durations[index % 4], depth=1.0 if index % 5 else 0.7, start_ms=start)
        for index, start in enumerate(starts)
    ]
    segments += [
        Yawn(4000, start_ms=20300),
        Nod(1000, pitch=20.0, start_ms=35200),
        Closure(hold_ms=700, start_ms=70300),
        Closure(hold_ms=2500, start_ms=95400),
        Closure(hold_ms=1500, start_ms=159300),  # 감긴 채로 얼굴이 사라진다
        FaceMissing(12000, start_ms=160000),  # 10초 구간에 유효 시간이 하나도 없는 스텝이 생긴다
        HeadTurn(4000, pitch=12.0, start_ms=205000),
        Yawn(5000, start_ms=212000),
        Nod(1200, pitch=18.0, start_ms=222300),
        Nod(800, pitch=22.0, start_ms=240500),
        HeadTurn(6000, yaw=50.0, start_ms=250000),  # |yaw_rel| > 40도: closure가 유효하지 않은 시간
        Linger(3000, 0.85, start_ms=270000),
    ]
    return _render(
        spec,
        segments,
        300000,
        fps=6.0,
        jitter_ms=20,
        drop_prob=0.02,
        gaps=((181000, 188200),),
        seed=7,
        first_t_ms=first_t_ms,
    )


def _step_cases(ctx: Context) -> list[_Case]:
    spec = ctx.spec
    baseline = _personal_baseline(spec)

    def check_session(computed: Mapping[str, Any]) -> None:
        steps = computed[SECTION_STEPS]
        _require([step["k"] for step in steps] == list(range(1, 300)), "steps 1..299, none skipped")
        _require([step["k"] for step in steps if step["full_lookback"]][0] == 60, "full lookback from step 60")
        per_frame: dict[int, int] = {}
        for step in steps:
            per_frame[step["t_ms"]] = per_frame.get(step["t_ms"], 0) + 1
        _require(max(per_frame.values()) >= 7, "the multi-second gap must emit several steps at once")
        full = [step["values"] for step in steps if step["full_lookback"]]
        _require(min(values[F_BLINK_RATE_60] for values in full) < 2.0, "the no-blink stretch must lower blink_rate_60")
        _require(max(values[F_FACE_MISSING_RATIO_10] for values in full) > 0.7, "face loss must raise the 10 s ratio")
        _require(max(values[F_YAWN_COUNT_60] for values in full) >= 1.0, "a yawn must be counted")
        _require(max(values[F_NOD_COUNT_60] for values in full) >= 2.0, "nods must be counted")
        _require(max(values[F_LONG_CLOSURE_COUNT_60] for values in full) >= 2.0, "long closures must be counted")
        _require(any(run["truncated"] for run in computed[SECTION_RUNS]), "a closure must be truncated by face loss")
        empty = [values for values in full if values[F_FACE_MISSING_RATIO_10] == 1.0]
        _require(len(empty) > 0, "some steps must have no face time in the last 10 s")
        for values in empty:  # 0으로 나누게 되는 비율은 0 (PLAN 3.3절)
            _require(values[F_PERCLOS_10] == 0.0 and values[F_CLOSURE_MEAN_10] == 0.0, "undefined 10 s ratios are 0")
        _require(any(not run["velocity_valid"] for run in computed[SECTION_RUNS]), "some velocities must be invalid")

    gap = [Blink(400, start_ms=start) for start in range(3000, 62000, 5000)]
    gap += [Yawn(5000, start_ms=61500), Closure(hold_ms=3000, depth=_LOW_FPS_DEPTH, start_ms=62500)]
    gap += [Blink(400, start_ms=start) for start in range(78000, 86000, 4000)]

    # 고개를 숙인 채로 가만히 있는 얼굴: pitch_rel이 0이 아닌 일정한 값이라 분산이 반올림 오차만큼 0에서 벗어난다.
    still = replace(synth.default_profile(spec), pitch=7.3)

    def check_gap(computed: Mapping[str, Any]) -> None:
        steps = computed[SECTION_STEPS]
        _require(all(step["values"][F_PITCH_STD_60] < 1e-6 for step in steps), "a constant pitch has no spread")
        numbers = [step["k"] for step in steps]
        _require(numbers == list(range(1, 64)) + list(range(77, 87)), f"step numbers are {numbers}")
        _require(steps[61]["full_lookback"] and not steps[63]["full_lookback"], "full_lookback must restart")
        _require(steps[62]["values"][F_FACE_MISSING_RATIO_60] < 0.01, "before the gap: a full minute of face time")
        _require(steps[63]["values"][F_FACE_MISSING_RATIO_60] > 0.98, "after the gap: one second of face time")
        _require(all(not run["truncated"] for run in computed[SECTION_RUNS]), "the ongoing closure is dropped")

    # 토막이 여럿인 구간의 스텝 특징. 100 ms 간격, 값은 계단 모양(0.4는 이탈 0.3과 진입 0.5 사이).
    half_open = _Shot(level=0.4)
    multi_span_plan = [
        (0, _OPEN),
        # A: 토막 둘. 0.5 이상이던 시간은 224 + 224 = 448 ms, 처음부터 끝까지는 988 ms
        (1000, _SHUT),
        (1200, half_open),
        (1800, _SHUT),
        (2000, _OPEN),
        # B: 토막 셋. 324 + 260 + 324 = 908 ms. 스텝 5가 끝날 때(세션 5000) 진행 중이다
        (4000, _SHUT),
        (4300, half_open),
        (4500, _SHUT),
        (4700, half_open),
        (4900, _SHUT),
        (5200, _OPEN),
        # C: 토막 둘 사이에 0.4로 2.2초 머문다(스텝 8, 9의 끝을 지난다). 424 + 224 = 648 ms
        (7000, _SHUT),
        (7400, half_open),
        (9600, _SHUT),
        (9800, _OPEN),
    ]

    def check_multi_span(computed: Mapping[str, Any]) -> None:
        runs = computed[SECTION_RUNS]
        got = [(run["duration_ms"], run["end_ms"] - run["start_ms"], run["truncated"]) for run in runs]
        _require(got == [(448, 988, False), (908, 1188, False), (648, 2788, False)], f"runs are {got}")
        steps = {step["k"]: step["values"] for step in computed[SECTION_STEPS]}
        _require(sorted(steps) == list(range(1, 13)), "steps 1..12")
        # 스텝 3: 구간 A 하나. 길이는 448 ms라 긴 감김(500 ms 이상)이 아니다(처음부터 끝까지로 세면 988 ms)
        _require(steps[3][F_BLINK_DUR_MEAN_60] == 448.0 and steps[3][F_LONG_CLOSURE_COUNT_60] == 0.0, "run A at step 3")
        _require(steps[3][F_MAX_CLOSURE_MS_60] == 448.0 and steps[3][F_MAX_CLOSURE_MS_10] == 448.0, "run A at step 3")
        # 스텝 5: 구간 B가 진행 중. 끝난 두 토막 584 ms + 셋째 토막의 80 ms
        _require(steps[5][F_LONG_CLOSURE_COUNT_60] == 1.0 and steps[5][F_MAX_CLOSURE_MS_60] == 664.0, "run B ongoing")
        _require(steps[5][F_BLINK_DUR_MEAN_60] == 448.0, "an ongoing run is not a blink event yet")
        # 스텝 6: 구간 B가 끝났다. 이벤트 둘의 평균 (448 + 908) / 2
        _require(steps[6][F_BLINK_DUR_MEAN_60] == 678.0 and steps[6][F_LONG_CLOSURE_COUNT_60] == 1.0, "run B at step 6")
        _require(steps[6][F_BLINK_RATE_60] == 2 * 60000.0 / 6000.0, "two events in 6 s of valid time")
        # 스텝 8, 9: 구간 C가 0.4에 머무는 중. 진행 중인 길이는 첫 토막의 344 ms에서 늘지 않는다
        for k in (8, 9):
            _require(steps[k][F_LONG_CLOSURE_COUNT_60] == 1.0 and steps[k][F_MAX_CLOSURE_MS_10] == 908.0, "dwell")
        # 스텝 10: 구간 C가 끝났다(648 ms, 긴 감김). 다시 감긴 프레임에서 머문 시간이 더해지지 않았다
        _require(steps[10][F_LONG_CLOSURE_COUNT_60] == 2.0 and steps[10][F_BLINK_DUR_MEAN_60] == 668.0, "run C")
        _require(steps[12][F_MAX_CLOSURE_MS_60] == 908.0, "the longest run is 908 ms, not 2788 ms from start to end")

    # 눈을 감은 채 프레임이 끊긴다(10초 이하). 200 ms 간격. 12.0초에 감고, 12.8초의 프레임 다음 프레임이 20.6초에 온다.
    across_gap_times = list(range(0, 13000, 200)) + list(range(20600, 32200, 200))
    across_gap_plan = [(0, _OPEN), (12000, _SHUT), (20600, _OPEN)]

    def check_across_gap(computed: Mapping[str, Any]) -> None:
        steps = {step["k"]: step for step in computed[SECTION_STEPS]}
        _require(sorted(steps) == list(range(1, 33)), "steps 1..32, none skipped (the gap is below 10 s)")
        emitter = _FIRST_T_MS + 20600
        at_once = [k for k, step in steps.items() if step["t_ms"] == emitter]
        _require(at_once == list(range(13, 21)), f"the frame after the gap emits steps {at_once}")
        (run,) = computed[SECTION_RUNS]
        _require(run["truncated"] and run["t_ms"] == emitter and run["duration_ms"] == 889, f"run is {run}")
        for k in range(13, 21):  # 끊긴 동안의 스텝: 끊기기 직전 프레임의 진행 중 감김(889 ms)을 쓴다
            values = steps[k]["values"]
            _require(values[F_LONG_CLOSURE_COUNT_60] == 1.0, "the ongoing closure counts once in every gap step")
            _require(values[F_MAX_CLOSURE_MS_60] == 889.0 and values[F_MAX_CLOSURE_MS_10] == 889.0, "ongoing 889 ms")
        for k in range(21, 31):  # 잘린 구간은 스텝 21의 통에 들어간다: 두 번 세지 않고, 10초 구간에 10스텝 더 남는다
            _require(steps[k]["values"][F_LONG_CLOSURE_COUNT_60] == 1.0, "the truncated run is not counted twice")
            _require(steps[k]["values"][F_MAX_CLOSURE_MS_10] == 889.0, "it stays in the 10 s window")
        _require(steps[31]["values"][F_MAX_CLOSURE_MS_10] == 0.0, "and leaves it after gap + 10 s")
        perclos = [steps[k]["values"][F_PERCLOS_10] for k in range(13, 24)]
        _require(perclos[0] == 0.1 and perclos[7] == 1000.0 / 3000.0, f"perclos_10 over the gap is {perclos}")
        _require(max(perclos) == 1000.0 / 2400.0 and perclos[-1] == 0.0, f"perclos_10 over the gap is {perclos}")

    # 0으로 나누게 되는 비율. 200 ms 간격. 0~6초 얼굴 없음, 6~12초 고개를 50도 돌림(얼굴은 있고 closure는 무효),
    # 12초부터 정면. 깜빡임 하나(13.0~13.2초)와 감긴 채 얼굴이 사라져 잘리는 감김 하나(16.0~16.2초).
    zero_plan = [
        (0, _Shot(face=False)),
        (6000, _Shot(yaw=50.0, pitch=5.0)),
        (12000, _Shot(pitch=5.0)),
        (13000, _Shot(level=0.9, pitch=5.0)),
        (13400, _Shot(pitch=5.0)),
        (16000, _Shot(level=0.9, pitch=5.0)),
        (16400, _Shot(face=False)),
        (17200, _Shot(pitch=5.0)),
    ]

    def check_zero(computed: Mapping[str, Any]) -> None:
        steps = {step["k"]: step["values"] for step in computed[SECTION_STEPS]}
        _require(sorted(steps) == list(range(1, 21)), "steps 1..20")
        for k in range(1, 7):  # 얼굴이 없다: 분모(유효 시간, 얼굴 시간, 이벤트 수)가 모두 0이다
            _require(all(value == 0.0 for value in steps[k][:15]), f"step {k}: every ratio and mean is 0")
            _require(steps[k][F_FACE_MISSING_RATIO_60] == 1.0 and steps[k][F_FACE_MISSING_RATIO_10] == 1.0, "no face")
        for k in range(7, 13):  # 얼굴은 있고 closure는 무효: 유효 시간 0
            _require(steps[k][F_PERCLOS_60] == 0.0 and steps[k][F_CLOSURE_MEAN_60] == 0.0, "no valid closure time")
            _require(steps[k][F_PERCLOS_10] == 0.0 and steps[k][F_CLOSURE_MEAN_10] == 0.0, "no valid closure time")
        _require(steps[10][F_PITCH_STD_60] == 0.0 and steps[11][F_FACE_MISSING_RATIO_60] == 55.0 / 60.0, "face time")
        # 스텝 13: 유효 시간 1초. perclos_10은 2초 미만이라 0이고 closure_mean_10은 남은 유효 시간으로 구한다
        _require(steps[13][F_PERCLOS_10] == 0.0 and steps[13][F_CLOSURE_MEAN_10] == 0.0, "step 13")
        _require(steps[14][F_PERCLOS_10] == 0.2 and steps[14][F_CLOSURE_MEAN_10] > 0.17, "step 14: 400 of 2000 ms")
        _require(steps[14][F_BLINK_DUR_MEAN_60] > 0.0 and steps[14][F_BLINK_DUR_STD_60] == 0.0, "one event: no spread")
        for k in range(13, 18):  # 유효 시간 1000, 2000, 3000, 4000, 4400 ms: 5초에 못 미친다
            _require(steps[k][F_BLINK_RATE_60] == 0.0 and steps[k][F_PERCLOS_60] == 0.0, "below 5 s of valid time")
        # 스텝 18: 유효 시간 5200 ms, 그중 0.8 이상이 800 ms, 잘리지 않은 구간 하나
        _require(steps[18][F_BLINK_RATE_60] == 60000.0 / 5200.0 and steps[18][F_PERCLOS_60] == 800.0 / 5200.0, "5.2 s")
        runs = computed[SECTION_RUNS]
        _require([run["truncated"] for run in runs] == [False, True], "one blink, one truncated closure")
        for values in steps.values():
            _require(all(value - value == 0.0 for value in values), "every value is finite")

    # 끊김이 정확히 10000 ms인 경우와 10001 ms인 경우. 200 ms 간격. 두 번 모두 눈을 감은 지 0.6초 만에 끊긴다
    # (감긴 프레임 넷: 4800~5400, 19800~20400). 5400 다음 프레임은 15400, 20400 다음 프레임은 30401이다.
    exact_gap_times = list(range(0, 5600, 200)) + list(range(15400, 20600, 200))
    exact_gap_times += [30401 + 200 * index for index in range(16)]
    exact_gap_plan = [(0, _OPEN), (4800, _SHUT), (15400, _OPEN), (19800, _SHUT), (30401, _OPEN)]

    def check_exact_gap(computed: Mapping[str, Any]) -> None:
        first = _FIRST_T_MS
        steps = {step["k"]: step for step in computed[SECTION_STEPS]}
        _require(sorted(steps) == [*range(1, 21), 31, 32, 33], f"steps are {sorted(steps)}")
        at_once = [k for k, step in steps.items() if step["t_ms"] == first + 15400]
        _require(at_once == list(range(6, 16)), f"the frame after the 10000 ms gap emits steps {at_once}")
        # 감김의 시작은 4600(0.0) -> 4800(0.9)의 200 ms 중 111 ms: 4711. 5400까지 본 길이는 689 ms다.
        (run,) = computed[SECTION_RUNS]
        got = (run["t_ms"], run["start_ms"], run["end_ms"], run["duration_ms"], run["truncated"])
        _require(got == (first + 15400, first + 4711, first + 5400, 689, True), f"run is {run}")
        for k in range(6, 16):  # 끊긴 동안의 스텝: 끊기기 직전 프레임의 진행 중 감김
            values = steps[k]["values"]
            _require(values[F_MAX_CLOSURE_MS_60] == 689.0 and values[F_LONG_CLOSURE_COUNT_60] == 1.0, f"step {k}")
        for k in range(16, 21):  # 잘린 구간이 스텝 16의 통에 들어 있다(두 번 세지 않는다)
            _require(steps[k]["values"][F_LONG_CLOSURE_COUNT_60] == 1.0, f"step {k}")
            _require(steps[k]["values"][F_MAX_CLOSURE_MS_10] == 689.0, f"step {k}")
        # 10001 ms: 초기화. 둘째 감김은 버려지고(구간이 하나뿐이다) 스텝 21~30은 없다. 스텝 31부터 다시 쌓는다.
        _require(steps[31]["t_ms"] == first + 31001, f"step 31 is emitted by the frame at {steps[31]['t_ms']}")
        for k in (31, 32, 33):
            values = steps[k]["values"]
            _require(values[F_MAX_CLOSURE_MS_60] == 0.0 and values[F_LONG_CLOSURE_COUNT_60] == 0.0, f"step {k}")
            _require(values[F_PERCLOS_60] == 0.0 and values[F_CLOSURE_MEAN_60] == 0.0, f"step {k}")
        missing = {k: steps[k]["values"][F_FACE_MISSING_RATIO_60] for k in (20, 31)}
        _require(abs(missing[20] - (60000.0 - 10200.0) / 60000.0) < 1e-12, "10.2 s of face time before the reset")
        _require(abs(missing[31] - (60000.0 - 599.0) / 60000.0) < 1e-12, "599 ms of face time after the reset")

    # --- 하품·끄덕임 계수기와 감김 특징의 경계(손으로 놓은 프레임, 값은 계단 모양, 세션 시간 = t_ms − 1) ------------

    def feature_by_step(computed: Mapping[str, Any], index: int) -> list[float]:
        return [step["values"][index] for step in computed[SECTION_STEPS]]

    def check_feature(index: int, wanted: Sequence[float], first_k: int = 1) -> Callable[[Mapping[str, Any]], None]:
        def check(computed: Mapping[str, Any]) -> None:
            numbers = [step["k"] for step in computed[SECTION_STEPS]]
            _require(numbers == list(range(first_k, first_k + len(wanted))), f"step numbers are {numbers}")
            got = feature_by_step(computed, index)
            _require(got == list(wanted), f"{FEATURE_NAMES[index]} by step is {got}, the script intends {list(wanted)}")

        return check

    no_face = _Shot(face=False)

    # 하품(jaw_rel 0.40 이상으로 시작, 0.25 미만으로 끝, 2000 ms에 도달한 프레임에서 한 번, 무효 500 ms 초과면 취소).
    # 200 ms 간격에 6500의 프레임 하나를 더했다. 턱: 0.5(시작 임계 이상), 0.3(두 임계 사이), 0(쉬는 값).
    wide = _Shot(jaw=0.5)
    ajar = _Shot(jaw=0.3)
    yawn_times = sorted([*range(0, 15200, 200), 6500])
    yawn_plan = [
        (0, ajar),  # A: 첫 프레임부터 0.3으로 2.2초. 시작 임계를 넘은 적이 없으므로 하품이 아니다
        (2400, _OPEN),
        (2600, wide),  # B: 2600~4000 벌림, 얼굴 없음 4200~4600, 4800~5600 벌림. 4000 -> 4800은 800 ms라 취소된다
        (4200, no_face),
        (4800, wide),
        (5800, _OPEN),
        (6000, wide),  # C: 6000~6500 벌림, 얼굴 없음 6600·6800, 7000부터 벌림. 6500 -> 7000은 정확히 500 ms라 이어진다
        (6600, no_face),
        (7000, wide),  # 8000의 프레임에서 8000 − 6000 = 2000 ms: 센다(스텝 9)
        (8400, _OPEN),
        (9000, wide),  # D: 9000~9800 벌림, 10000~11000은 0.3(끝 임계 이상이라 이어진다). 11000에서 정확히 2000 ms: 센다
        (10000, ajar),
        (11200, _OPEN),  # 바로 다음 프레임에서 닫는다
        (12000, wide),  # E: 12000~13800 벌림, 정확히 2000 ms 뒤인 14000의 프레임이 끝 임계 미만: 세지 않는다
        (14000, _OPEN),
    ]
    yawn_counts = [0.0] * 8 + [1.0] * 3 + [2.0] * 4

    # 끄덕임(d = pitch_rel − m이 10 이상으로 시작, 5 미만으로 끝, 길이 200~3000 ms면 끝날 때 센다). 200 ms 간격에
    # 7700의 프레임 하나를 더했다. 고개는 20도 숙이거나 0도다. 느린 평균 m은 이 세션에서 0과 6.3 사이라서
    # 숙인 프레임의 d는 13.7 이상, 든 프레임의 d는 0 이하다.
    bowed = _Shot(pitch=20.0)
    nod_times = sorted([*range(0, 8200, 200), 7700])
    nod_plan = [
        (0, _OPEN),
        (200, bowed),  # A: 200에 숙이고 400에 든다. 길이 정확히 200 ms: 센다(스텝 1)
        (400, _OPEN),
        (1000, bowed),  # B: 1000~3800 숙임, 4000에 든다. 길이 정확히 3000 ms: 센다(스텝 5)
        (4000, _OPEN),
        (5000, bowed),  # C: 5000~5400 숙임, 얼굴 없음 5600~6200. 5400 -> 6000은 600 ms라 취소된다: 세지 않는다
        (5600, no_face),
        (6400, _OPEN),
        (7000, bowed),  # D: 7000·7200 숙임, 얼굴 없음 7400·7600, 7700 숙임(7200 -> 7700은 정확히 500 ms라 이어진다)
        (7400, no_face),
        (7700, bowed),
        (7800, _OPEN),  # 7800에 든다. 길이 800 ms: 센다(스텝 8)
    ]
    nod_counts = [1.0] * 4 + [2.0] * 3 + [3.0]

    # 느린 평균. 자세 12도로 시작(첫 표본이 m의 초기값), 400부터 20도: 100 ms 간격 열 표본(α = 100/10100 = 1/101),
    # 1.7초의 프레임 끊김 뒤 3000의 표본(dt를 200으로 잘라 α = 200/10200 = 1/51), 200 ms 간격 스무 표본(α = 1/51).
    # 그 뒤 m = 20 − 8·(100/101)^10·(50/51)^21 = 15.2217. 7200·7400에 25.25도(d = 10.028: 시작), 7600에 20도로 돌아온다
    # (d = 20 − 15.6111 = 4.389: 끝, 길이 400 ms).
    mean_times = [0, 100, 200, 300, *range(400, 1400, 100), *range(3000, 8200, 200)]
    mean_plan = [(0, _Shot(pitch=12.0)), (400, bowed), (7200, _Shot(pitch=25.25)), (7600, bowed)]
    mean_counts = [0.0] * 7 + [1.0]

    # 10초 넘는 끊김 뒤의 끄덕임 검출기. 0~1000은 0도, 10400 ms 끊긴 뒤 11400~12200은 12도, 12400부터 0도.
    restart_times = [*range(0, 1200, 200), *range(11400, 14200, 200)]
    restart_plan = [(0, _OPEN), (11400, _Shot(pitch=12.0)), (12400, _OPEN)]

    def check_restart(computed: Mapping[str, Any]) -> None:
        numbers = [step["k"] for step in computed[SECTION_STEPS]]
        _require(numbers == [1, 12, 13, 14], f"step numbers are {numbers}")
        got = feature_by_step(computed, F_NOD_COUNT_60)
        _require(got == [0.0, 0.0, 0.0, 0.0], f"nod_count_60 by step is {got}: the slow mean must restart at 12 degrees")

    # 한 스텝 안에서 끝난 감김 둘. 100 ms 간격, 0.0 -> 0.9는 56 ms, 0.9 -> 0.0은 44 ms에 0.5를 지난다.
    pair_plan = [
        (0, _OPEN),
        (200, _SHUT),  # 156~544: 388 ms
        (600, _OPEN),
        (800, _SHUT),  # 756~844: 88 ms
        (900, _OPEN),
        (1100, _SHUT),  # 1056~1144: 88 ms
        (1200, _OPEN),
        (1400, _SHUT),  # 1356~1844: 488 ms
        (1900, _OPEN),
    ]

    def check_pair(computed: Mapping[str, Any]) -> None:
        got = [(run["duration_ms"], run["truncated"]) for run in computed[SECTION_RUNS]]
        _require(got == [(388, False), (88, False), (88, False), (488, False)], f"runs are {got}")
        check_feature(F_MAX_CLOSURE_MS_60, [388.0, 488.0])(computed)
        check_feature(F_MAX_CLOSURE_MS_10, [388.0, 488.0])(computed)
        check_feature(F_BLINK_DUR_MEAN_60, [(388 + 88) / 2, (388 + 88 + 88 + 488) / 4])(computed)
        spread = feature_by_step(computed, F_BLINK_DUR_STD_60)  # sqrt(79144 − 238²), sqrt(101044 − 263²)
        _require(abs(spread[0] - 150.0) < 1e-9 and abs(spread[1] - math.sqrt(31875.0)) < 1e-9, f"std is {spread}")
        check_feature(F_LONG_CLOSURE_COUNT_60, [0.0, 0.0])(computed)

    # 상한 10000 ms를 넘는 감김. 200 ms 간격. 200(0.0) -> 400(0.9)의 111 ms에 0.5를 위로 지나고(311), 11800 -> 12000의
    # 89 ms에 아래로 지난다(11889). 길이 11578 ms. 스텝 k(1~12)의 끝에서 진행 중인 길이는 (1000k − 200) − 311이다.
    cap_plan = [(0, _OPEN), (400, _SHUT), (12000, _OPEN)]
    cap_ongoing = [float(min(1000 * k - 511, 10000)) for k in range(1, 13)]

    def check_cap(computed: Mapping[str, Any]) -> None:
        (run,) = computed[SECTION_RUNS]
        got = (run["start_ms"], run["end_ms"], run["duration_ms"], run["truncated"], run["t_ms"])
        _require(got == (_FIRST_T_MS + 311, _FIRST_T_MS + 11889, 11578, False, _FIRST_T_MS + 12000), f"run is {run}")
        _require(cap_ongoing[9:] == [9489.0, 10000.0, 10000.0], "steps 10..12: 9489, then the cap")
        check_feature(F_MAX_CLOSURE_MS_60, [*cap_ongoing, 10000.0])(computed)  # 스텝 13: 끝난 감김 11578도 상한
        check_feature(F_MAX_CLOSURE_MS_10, [*cap_ongoing, 10000.0])(computed)
        check_feature(F_LONG_CLOSURE_COUNT_60, [0.0] + [1.0] * 12)(computed)  # 스텝 1의 진행 중 길이는 489 ms
        check_feature(F_BLINK_DUR_MEAN_60, [0.0] * 12 + [2000.0])(computed)  # 깜빡임 길이의 상한은 2000 ms

    # 길이가 정확히 500 ms인 감김. 100 ms 간격에 1950의 프레임 하나를 더했다. 눈을 뜬 값이 0.1이라 0.1 <-> 0.9의
    # 쌍에서 0.5를 지나는 자리는 정확히 가운데(50 ms)다.
    slit = _Shot(level=0.1)
    edge_times = sorted([*range(0, 3100, 100), 1950])
    edge_plan = [
        (0, slit),
        (200, _SHUT),  # 150~650: 정확히 500 ms. 700의 프레임에서 끝난다(스텝 1)
        (700, slit),
        (1500, _SHUT),  # 1450에 시작. 스텝 2의 마지막 프레임 1950에서 진행 중인 길이가 정확히 500 ms
        (2100, slit),  # 2050에 끝난다: 600 ms(스텝 3)
    ]

    def check_edge(computed: Mapping[str, Any]) -> None:
        got = [(run["start_ms"] - _FIRST_T_MS, run["end_ms"] - _FIRST_T_MS, run["duration_ms"]) for run in computed[SECTION_RUNS]]
        _require(got == [(150, 650, 500), (1450, 2050, 600)], f"runs are {got}")
        check_feature(F_LONG_CLOSURE_COUNT_60, [1.0, 2.0, 2.0])(computed)
        check_feature(F_MAX_CLOSURE_MS_60, [500.0, 500.0, 600.0])(computed)
        check_feature(F_BLINK_DUR_MEAN_60, [500.0, 500.0, 550.0])(computed)

    # pitch가 바뀌는 세션의 pitch_std_60. 200 ms 간격, 모든 프레임에 얼굴이 있다: 스텝 k의 얼굴 시간은 정확히 1000k ms.
    # pitch는 0~2000에 0도, 그 뒤 5도. 스텝 5: 평균 3, 제곱 평균 15, 분산 6. 스텝 6: 평균 10/3, 제곱 평균 50/3, 분산 50/9.
    spread_plan = [(0, _OPEN), (2000, _Shot(pitch=5.0))]

    def check_spread(computed: Mapping[str, Any]) -> None:
        numbers = [step["k"] for step in computed[SECTION_STEPS]]
        _require(numbers == [1, 2, 3, 4, 5, 6], f"step numbers are {numbers}")
        got = feature_by_step(computed, F_PITCH_STD_60)
        _require(got[:4] == [0.0, 0.0, 0.0, 0.0], f"below 5000 ms of face time pitch_std_60 is 0, got {got}")
        _require(abs(got[4] - math.sqrt(6.0)) < 1e-9, f"step 5: exactly 5000 ms of face time, got {got[4]}")
        _require(abs(got[5] - math.sqrt(50.0 / 9.0)) < 1e-9, f"step 6: got {got[5]}")
        missing = feature_by_step(computed, F_FACE_MISSING_RATIO_60)
        _require(missing[4] == (60000.0 - 5000.0) / 60000.0, f"step 5 must hold exactly 5000 ms of face time: {missing}")

    return [
        _Case(
            name="steps_five_minutes",
            bundle=BUNDLE_STEPS,
            description=(
                "스텝 특징 5분 세션(기준선을 넣어 줘서 첫 프레임부터 누적, 스텝 1~299). 초기 구간(스텝 59까지는 "
                "full_lookback이 false), 깜빡임 없는 구간(118~152초), 감긴 채 얼굴이 사라지는 12초(160~172초, 잘린 감김. "
                "10초 구간에 유효 시간이 없는 스텝의 perclos_10과 closure_mean_10은 정의 불가 값 0), 7.2초의 프레임 "
                "끊김(181~188.2초, 다음 프레임 하나가 스텝 여러 개를 발행), 하품 둘, 끄덕임 셋, 긴 감김, |yaw_rel| > 40도인 "
                "고개 돌림, 0.85에 머무는 눈꺼풀을 담았다. 약 6 fps이고 프레임 간격이 200 ms를 넘는 곳에서는 넘는 "
                "부분이 '데이터 없음'이 된다. 스텝 k의 특징은 세션 시간 1000k 이전의 프레임만 쓴다."
            ),
            sections=_STEP_SECTIONS,
            frames=_five_minute_frames(spec, 1),
            baseline=baseline,
            check=check_session,
        ),
        _Case(
            name="steps_five_minutes_large_t0",
            bundle=BUNDLE_STEPS,
            description=(
                "steps_five_minutes와 같은 세션을 첫 t_ms = 1731465123457에서 시작한 것. 세션 시간은 첫 프레임 "
                "대비이므로 스텝 번호와 특징 값은 steps_five_minutes와 같고 t_ms, t_end_ms, 감김 구간의 시각만 "
                "그만큼 밀린다. t_ms는 32비트 정수 범위를 넘는다."
            ),
            sections=_STEP_SECTIONS,
            frames=_five_minute_frames(spec, LARGE_FIRST_T_MS),
            baseline=baseline,
            check=check_session,
        ),
        _Case(
            name="steps_gap_over_10s",
            bundle=BUNDLE_STEPS,
            description=(
                "10초 넘는 끊김 뒤의 초기화. 63.9초까지 프레임이 오다가(스텝 60부터 full_lookback) 감김과 하품이 "
                "진행 중일 때 12.7초 동안 끊긴다. 끊긴 동안의 스텝(64~76)은 발행하지 않고, 진행 중이던 감김과 하품은 "
                "버린다. 새 프레임이 속한 스텝 77부터 누적을 다시 시작하므로 full_lookback이 false로 돌아가고 "
                "face_missing_ratio_60은 1 근처에서 다시 내려온다. 스텝 번호는 세션 시간 그대로다. 얼굴은 고개를 "
                "7.3도 숙인 채 움직이지 않는다: pitch_std_60의 분산은 0 이상으로 자른 뒤 제곱근을 구한다(0 또는 1e-6 "
                "미만의 값이 나와야 하고 NaN이 나오면 안 된다)."
            ),
            sections=_STEP_SECTIONS,
            frames=_render(spec, gap, 86500, fps=6.0, gaps=((63900, 76600),), profile=still),
            baseline=baseline,
            check=check_gap,
        ),
        _Case(
            name="steps_multi_span_runs",
            bundle=BUNDLE_STEPS,
            description=(
                "토막이 여럿인 감김의 스텝 특징(10 fps, 12초, 스텝 1~12). 감김의 길이(특징 #5, #6, #9~#11)는 closure가 "
                "0.5 이상이던 시간의 합이고 0.3~0.5에 머문 시간은 들어가지 않는다. 구간 A(토막 둘, 448 ms, 처음부터 "
                "끝까지는 988 ms)는 긴 감김(500 ms 이상)이 아니다. 구간 B(토막 셋, 908 ms)는 스텝 5의 끝에서 진행 "
                "중이고(진행 중인 길이 664 ms) 스텝 6에서 이벤트가 된다. 구간 C는 0.4에 2.2초 머무는 동안 진행 중인 "
                "길이가 344 ms에서 늘지 않고, 다시 감긴 뒤 648 ms로 끝난다(처음부터 끝까지는 2788 ms). 깜빡임 둘이 "
                "머문 구간으로 이어지면 이벤트는 하나(blink_rate_60)이고 길이는 두 토막의 합이다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, range(0, 12100, 100), multi_span_plan),
            baseline=baseline,
            check=check_multi_span,
        ),
        _Case(
            name="steps_closure_across_gap",
            bundle=BUNDLE_STEPS,
            description=(
                "눈을 감은 채 프레임이 7.8초 끊긴다(5 fps, 10초 이하라 초기화는 없다). 12.0초에 감고 12.8초의 프레임 "
                "다음 프레임이 20.6초에 온다. 그 프레임 하나가 스텝 13~20을 발행하고, 그 스텝들은 모두 끊기기 직전 "
                "프레임의 진행 중 감김(889 ms)을 쓴다: long_closure_count_60 = 1, max_closure_ms_60 = "
                "max_closure_ms_10 = 889. 같은 프레임에서 검출기가 그 감김을 잘린 구간으로 내고 스텝 21의 통에 "
                "들어가므로 두 번 세지 않지만, max_closure_ms_10에는 스텝 30까지(끊긴 시간 + 10초) 남는다. 끊긴 동안 "
                "10초 구간의 유효 시간이 줄어 perclos_10이 0.1에서 0.4167까지 올랐다가 감긴 시간이 구간을 벗어난 "
                "스텝 23에서 0이 된다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, across_gap_times, across_gap_plan),
            baseline=baseline,
            check=check_across_gap,
        ),
        _Case(
            name="steps_zero_division",
            bundle=BUNDLE_STEPS,
            description=(
                "0으로 나누게 되는 비율은 0이다(PLAN 3.3절, 4.6절). 5 fps, 20초. 0~6초는 얼굴이 없다: 유효 시간, 얼굴 "
                "시간, 이벤트 수가 모두 0이라 특징 #0~#14가 0이고 face_missing_ratio는 1이다. 6~12초는 고개를 50도 "
                "돌려 얼굴은 있지만 closure가 무효다: perclos와 closure_mean은 계속 0이고 얼굴 시간만 쌓인다"
                "(pitch_std_60은 얼굴 시간 5초부터 계산하고, 일정한 pitch라 0이다). 12초부터 정면: 유효 시간이 2초에 "
                "못 미치는 스텝 13의 perclos_10은 0, 5초에 못 미치는 스텝 13~17의 perclos_60과 blink_rate_60은 0이다"
                "(16.4~17.2초에 감긴 채 얼굴이 사라져 유효 시간이 스텝 18에서야 5초를 넘는다). 깜빡임이 하나뿐이라 "
                "blink_dur_std_60은 0이고, 잘린 감김은 이벤트 수에 들어가지 않는다. 모든 값이 유한해야 한다(NaN이나 "
                "무한대가 나오면 안 된다)."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, range(0, 20200, 200), zero_plan),
            baseline=baseline,
            check=check_zero,
        ),
        _Case(
            name="steps_gap_exactly_10s",
            bundle=BUNDLE_STEPS,
            description=(
                "특징 쪽의 초기화 경계(5 fps): 프레임 간격이 10초를 '넘을' 때만 특징용 감김 검출기와 누적기를 "
                "초기화한다. 두 번 모두 눈을 감은 지 0.6초 만에 프레임이 끊긴다. (1) 5400의 프레임 다음 프레임이 "
                "정확히 10000 ms 뒤(15400)에 온다: 초기화하지 않는다. 그 프레임이 진행 중이던 감김을 잘린 구간"
                "(689 ms, end_ms는 마지막 감긴 프레임 5400)으로 내고 스텝 6~15를 한꺼번에 발행한다. 그 스텝들은 "
                "끊기기 직전의 진행 중 감김(689 ms)을 쓴다(long_closure_count_60 = 1, max_closure_ms_60 = 689). "
                "(2) 20400의 프레임 다음 프레임이 10001 ms 뒤(30401)에 온다: 초기화한다. 진행 중이던 감김은 구간 "
                "없이 버려지고 스텝 21~30은 발행되지 않는다. 새 프레임이 속한 스텝 31부터 누적을 다시 시작하므로 "
                "스텝 31~33의 감김 특징은 0이고 face_missing_ratio_60은 1 근처에서 다시 내려온다. '이상'으로 "
                "구현하면 (1)에서 구간과 스텝 6~15가 나오지 않는다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, exact_gap_times, exact_gap_plan),
            baseline=baseline,
            check=check_exact_gap,
        ),
        _Case(
            name="steps_yawn_edges",
            bundle=BUNDLE_STEPS,
            description=(
                "하품 계수의 경계(5 fps, 15초, 스텝 1~15). jaw_rel이 0.40 이상이면 시작, 0.25 미만이면 끝이고, 시작한 "
                "프레임에서 2000 ms가 된 프레임에서 한 번 센다. 얼굴 없는 시간이 500 ms를 넘으면 취소한다. 장면 다섯: "
                "(A) 첫 프레임부터 0.3으로 2.2초: 시작 임계에 닿은 적이 없어 세지 않는다. (B) 1.4초 벌리고 얼굴이 "
                "사라졌다가 800 ms 뒤에 벌린 채 돌아와 0.8초 더: 취소되고 새로 시작하므로 세지 않는다(취소하지 않으면 "
                "돌아온 프레임에서 2200 ms가 되어 센다). (C) 0.5초 벌리고 얼굴이 사라졌다가 정확히 500 ms 뒤에 돌아온다: "
                "'넘으면'이므로 이어지고 8000의 프레임에서 센다. (D) 0.8초 벌린 뒤 0.3으로 내려가 머문다: 끝 임계 "
                "이상이라 이어지고 정확히 2000 ms인 11000의 프레임에서 센다. 다음 프레임에서 닫으므로 '이상'을 '초과'로 "
                "구현하면 세지 못한다. (E) 1.8초 벌리고 정확히 2000 ms 뒤의 프레임에서 닫는다: 끝 임계 아래로 내려간 "
                "프레임에서는 세지 않는다. yawn_count_60은 스텝 1~8이 0, 9~11이 1, 12~15가 2다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, yawn_times, yawn_plan),
            baseline=baseline,
            check=check_feature(F_YAWN_COUNT_60, yawn_counts),
        ),
        _Case(
            name="steps_nod_edges",
            bundle=BUNDLE_STEPS,
            description=(
                "끄덕임 계수의 경계(5 fps, 8초, 스텝 1~8). d = pitch_rel − m(느린 평균)이 10도 이상이면 시작, 5도 미만이면 "
                "끝이고, 끝난 프레임에서 길이가 200~3000 ms이면 센다. 얼굴 없는 시간이 500 ms를 넘으면 취소한다. 고개는 "
                "20도 숙이거나 0도다. 장면 넷: (A) 한 프레임만 숙였다가 200 ms 뒤에 든다: 길이가 정확히 200 ms라 센다. "
                "(B) 3초 숙였다가 든다: 정확히 3000 ms라 센다. (C) 0.6초 숙이다가 얼굴이 0.8초 사라진다: 얼굴 없는 "
                "프레임은 값을 내지 않고 600 ms째에 판정이 취소된다. 얼굴 없는 프레임을 pitch 0의 표본으로 넣으면 첫 무효 "
                "프레임에서 길이 600 ms로 끝나 하나를 더 센다. (D) 0.4초 숙이다가 얼굴이 사라지고 정확히 500 ms 뒤에 "
                "숙인 채 돌아와 바로 든다: 이어지므로 길이 800 ms로 센다(취소하면 100 ms짜리가 되어 세지 않는다). "
                "nod_count_60은 스텝 1~4가 1, 5~7이 2, 8이 3이다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, nod_times, nod_plan),
            baseline=baseline,
            check=check_feature(F_NOD_COUNT_60, nod_counts),
        ),
        _Case(
            name="steps_nod_slow_mean",
            bundle=BUNDLE_STEPS,
            description=(
                "끄덕임의 느린 평균 m(스텝 1~8). 유효 표본마다 d = pitch_rel − m을 먼저 구한 뒤 m += α·(pitch_rel − m), "
                "α = dt/(10000 + dt), dt = min(직전 유효 표본과의 간격, 200)으로 갱신한다. 자세 12도로 시작한다: 첫 "
                "표본이 m의 초기값이라 d는 0이다(m을 0에서 시작하면 첫 프레임부터 d = 12로 끄덕임이 시작되어 끝나지 않는다). "
                "400 ms부터 자세가 20도다(d = 8에서 줄어든다. 시작 임계 10 미만). 100 ms 간격 열 표본(α = 1/101), 1.7초의 "
                "프레임 끊김 뒤 한 표본(dt를 200으로 잘라 α = 1/51), 200 ms 간격 스무 표본(α = 1/51)을 지나면 "
                "m = 20 − 8·(100/101)^10·(50/51)^21 = 15.2217이다. 7200과 7400의 프레임이 25.25도다: d = 10.028로 "
                "시작하고, 7600에 20도로 돌아오면 d = 4.389로 끝난다(길이 400 ms). nod_count_60은 스텝 1~7이 0, 스텝 "
                "8이 1이다. 25.25도는 m의 갱신 속도가 1%만 달라도 d가 10에 못 미치도록 고른 값이다: α = dt/10000이면 "
                "d = 9.984, α = dt/(10000 − dt)이면 9.938, dt를 자르지 않으면 9.416, dt를 늘 200으로 두면 9.580이다. "
                "m을 갱신하지 않거나 d = pitch_rel + m으로 구하면 끄덕임이 끝나지 않아 하나도 세지 않는다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, mean_times, mean_plan),
            baseline=baseline,
            check=check_feature(F_NOD_COUNT_60, mean_counts),
        ),
        _Case(
            name="steps_nod_after_long_gap",
            bundle=BUNDLE_STEPS,
            description=(
                "10초 넘는 끊김 뒤의 끄덕임 검출기(5 fps). 0~1000 ms는 고개 0도(m = 0), 프레임이 10400 ms 끊긴 뒤 "
                "11400~12200은 12도, 12400부터 0도다. 끊김 뒤에는 느린 평균도 처음 상태로 돌아가므로 끊김 뒤 첫 표본이 "
                "m의 새 초기값이다(12도, d = 0). 12400에 0도로 돌아와도 d = −12라 끄덕임이 없다: nod_count_60은 스텝 1, "
                "12, 13, 14 모두 0이다(스텝 2~11은 발행되지 않는다). 느린 평균을 끊김 앞의 값(0)으로 두면 11400의 "
                "프레임에서 d = 12로 시작해 12400에 끝나는 1000 ms짜리 끄덕임을 스텝 13에서 센다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, restart_times, restart_plan),
            baseline=baseline,
            check=check_restart,
        ),
        _Case(
            name="steps_two_closures_in_one_step",
            bundle=BUNDLE_STEPS,
            description=(
                "한 스텝 안에서 끝난 감김이 둘일 때의 max_closure_ms(10 fps, 스텝 1~2). 스텝 1에서는 388 ms짜리가 먼저, "
                "88 ms짜리가 나중에 끝난다. 스텝 2에서는 88 ms짜리가 먼저, 488 ms짜리가 나중에 끝난다. 스텝의 값은 그 "
                "스텝에 귀속된 감김 가운데 가장 긴 것이다: max_closure_ms_60과 max_closure_ms_10은 스텝 1이 388, 스텝 2가 "
                "488이다. 나중에 끝난 감김으로 덮어쓰면 스텝 1이 88이 되고, 먼저 끝난 감김을 남기면 스텝 2가 388이 된다. "
                "blink_dur_mean_60은 238과 263, blink_dur_std_60은 150과 178.536, 긴 감김(500 ms 이상)은 없다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, range(0, 2100, 100), pair_plan),
            baseline=baseline,
            check=check_pair,
        ),
        _Case(
            name="steps_closure_above_cap",
            bundle=BUNDLE_STEPS,
            description=(
                "상한을 넘는 감김(5 fps, 스텝 1~13). 세션 311 ms에 감아 11889 ms에 뜬다(길이 11578 ms). 스텝 k의 끝에서 "
                "진행 중인 길이는 1000k − 511 ms다: max_closure_ms_60과 max_closure_ms_10은 스텝 10이 9489이고, 스텝 11과 "
                "12는 10489와 11489가 아니라 상한 max_closure_cap_ms인 10000이다. 감김이 끝나 스텝 13의 통에 들어간 뒤에도 "
                "11578이 아니라 10000이다. 깜빡임 길이의 상한은 따로 있다: blink_dur_mean_60은 스텝 13에서 2000"
                "(blink_duration_cap_ms)이다. long_closure_count_60은 스텝 1(진행 중 489 ms)이 0, 그 뒤로 1이다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, range(0, 13200, 200), cap_plan),
            baseline=baseline,
            check=check_cap,
        ),
        _Case(
            name="steps_closure_exactly_500ms",
            bundle=BUNDLE_STEPS,
            description=(
                "길이가 정확히 500 ms인 감김은 긴 감김이다(long_closure_min_ms 500 '이상'. 10 fps, 스텝 1~3). 눈을 뜬 값이 "
                "0.1, 감은 값이 0.9라서 0.5를 지나는 자리가 두 프레임의 정확히 가운데다. 첫 감김은 150~650 ms로 정확히 "
                "500 ms이고 스텝 1에서 끝난다: long_closure_count_60 = 1. 둘째 감김은 1450에 시작하고 스텝 2의 마지막 "
                "프레임(1950)에서 진행 중인 길이가 정확히 500 ms다: 진행 중인 감김도 '이상'으로 세어 스텝 2는 2다. "
                "둘째 감김은 600 ms로 끝나 스텝 3도 2다. 끝난 감김을 '초과'로 세면 0, 1, 1이, 진행 중인 감김을 '초과'로 세면 "
                "1, 1, 2가 된다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, edge_times, edge_plan),
            baseline=baseline,
            check=check_edge,
        ),
        _Case(
            name="steps_pitch_std_at_5000ms",
            bundle=BUNDLE_STEPS,
            description=(
                "pitch_std_60은 60초 구간의 얼굴 시간이 pitch_std_min_face_ms(5000) '이상'일 때 계산한다(5 fps, 스텝 "
                "1~6, 모든 프레임에 얼굴이 있다). pitch_rel은 0~2000 ms에 0도, 그 뒤 5도다. 스텝 3과 4는 pitch가 이미 "
                "바뀌었지만 얼굴 시간이 3000, 4000 ms라 0이다. 스텝 5는 얼굴 시간이 정확히 5000 ms다: 시간 가중 평균 3, "
                "제곱 평균 15, 표준편차 sqrt(6) = 2.449490. 스텝 6은 sqrt(50/9) = 2.357023. '초과'로 구현하면 스텝 5가 "
                "0이 된다. 프레임이 200 ms 이하 간격으로 끊기지 않고 오는 세션은 모두 다섯째 스텝에서 이 자리를 지난다."
            ),
            sections=_STEP_SECTIONS,
            frames=_hand_frames(spec, range(0, 6200, 200), spread_plan),
            baseline=baseline,
            check=check_spread,
        ),
    ]


# --- 규칙 -------------------------------------------------------------------

_RULE_SECTIONS: tuple[str, ...] = (SECTION_EVENTS, SECTION_ALERTS, SECTION_COUNTERS)

_ONSET = PHASE_ONSET
_MICROSLEEP_ON = f"{EVENT_MICROSLEEP}:{_ONSET}"
_MICROSLEEP_END = f"{EVENT_MICROSLEEP}:end"
_SLEEP_ON = f"{EVENT_SLEEP}:{_ONSET}"
_SLEEP_END = f"{EVENT_SLEEP}:end"
_GAZE_ON = f"{EVENT_GAZE_AWAY_LONG}:{_ONSET}"
_GAZE_END = f"{EVENT_GAZE_AWAY_LONG}:end"
_VATS_ON = f"{EVENT_VATS}:{_ONSET}"
_VATS_END = f"{EVENT_VATS}:end"
_DROWSY_ON = f"{EVENT_DROWSY}:{_ONSET}"
_DROWSY_END = f"{EVENT_DROWSY}:end"
_CALIBRATING_ON = "calibrating:onset"
_CALIBRATING_END = "calibrating:end"
_UNABLE_ON = "unable_to_analyze:onset"
_UNABLE_END = "unable_to_analyze:end"

# 감김 경계값 벡터의 프레임 간격(ms). 30 fps의 프레임 시각은 100 ms마다 같은 자리로 돌아와서 3000 ms 뒤의 프레임이
# 임계 시각에 정확히 올 수 없다(오면 그 표본의 closure가 임계값과 같아진다). 35 ms 간격이면 감김의 시작(보간한 값)에서
# 정확히 3000 ms 뒤에 프레임이 오고, 그 앞뒤 표본은 임계값에서 멀다.
_BOUNDARY_FRAME_MS = 35.0

# 스텁 모델의 확률 (alert, low_vigilant, drowsy)
_STUB_MID = (0.3, 0.2, 0.5)  # 시작 조건(≥ 0.6)도 종료 조건(≤ 0.4)도 아닌 값
_STUB_HIGH = (0.1, 0.2, 0.7)
_STUB_LOW = (0.5, 0.2, 0.3)
# p_drowsy가 시작 임계(0.6)와 같은 값. 스텁이 Double을 돌려주면 "이상"의 등호 자리를 지나고, Float를 돌려주는 구현에서는
# 0.6f가 0.6보다 조금 커서(0.60000002) 어느 쪽이든 시작 조건이다. 종료 임계 0.4는 같은 방법으로 고정할 수 없다
# (0.4f는 0.4보다 커서 "이하"가 아니게 된다).
_STUB_AT_ONSET = (0.2, 0.2, 0.6)


def _rule_cases(ctx: Context) -> list[_Case]:
    spec = ctx.spec
    baseline = _personal_baseline(spec)
    closure_rules = ctx.rules.closure
    events_cfg = ctx.rules.events

    def closure_case(name: str, description: str, hold_ms: int, events: Sequence[str], alerts: Sequence[str]) -> _Case:
        # 0.7을 넘는 시간 = 감는 100 ms 중 30 + 머무는 hold_ms + 뜨는 150 ms 중 45
        return _Case(
            name=name,
            bundle=BUNDLE_RULES,
            description=description,
            sections=_RULE_SECTIONS,
            frames=_render(spec, [Closure(hold_ms=hold_ms, start_ms=510)], hold_ms + 1350),
            baseline=baseline,
            events=tuple(events),
            alerts=tuple(alerts),
        )

    def check_sleep_at_boundary(repeat: bool) -> Callable[[Mapping[str, Any]], None]:
        def check(computed: Mapping[str, Any]) -> None:
            sleep = [event for event in computed[SECTION_EVENTS] if event["type"] == EVENT_SLEEP][0]
            _require(sleep["duration_ms"] == closure_rules.sleep_ms, f"sleep onset after {sleep['duration_ms']} ms")
            if repeat:
                first, second = [alert for alert in computed[SECTION_ALERTS] if alert["type"] == EVENT_SLEEP]
                waited = second["t_ms"] - first["t_ms"]
                repeat_ms = events_cfg.sleep.repeat_ms
                _require(repeat_ms <= waited < repeat_ms + _BOUNDARY_FRAME_MS, f"repeat after {waited} ms")
                _require(second["t_start_ms"] == first["t_start_ms"], "the repeat is a copy of the onset")
                _require(second["duration_ms"] == second["t_ms"] - second["t_start_ms"], "with its duration updated")

        return check

    def gaze_case(name: str, description: str, turn_ms: int, events: Sequence[str]) -> _Case:
        # yaw 35도, 오르내림 300 ms: yaw가 23도를 넘는 시간은 turn_ms − 394 ms
        return _Case(
            name=name,
            bundle=BUNDLE_RULES,
            description=description,
            sections=_RULE_SECTIONS,
            frames=_render(spec, [HeadTurn(turn_ms, yaw=35.0, start_ms=500)], turn_ms + 1300),
            baseline=baseline,
            events=tuple(events),
            alerts=tuple(event for event in events if event.endswith(_ONSET)),
        )

    def check_source(source: str, exact_ms: int | None = None) -> Callable[[Mapping[str, Any]], None]:
        def check(computed: Mapping[str, Any]) -> None:
            for event in computed[SECTION_EVENTS]:
                _require(event["source"] == source, f"source is {event['source']}")
            if exact_ms is not None:
                onset = computed[SECTION_EVENTS][0]
                _require(onset["duration_ms"] == exact_ms, f"onset after {onset['duration_ms']} ms")

        return check

    def check_face_lost(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["confidence"] < 0.5, "the onset must come while the face is lost (little valid pose time)")
        _require(end["duration_ms"] > 5500, "the state lasts until the head is back")

    def check_vats(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["source"] == SOURCE_HEAD_POSE and onset["t_start_ms"] == end["t_start_ms"], "one VATS state")
        _require(onset["duration_ms"] < ctx.rules.vats.window_ms < end["duration_ms"], "the window slides past it")

    vats_starts = (2000, 8000, 14000, 20000)
    cooldown_starts = (1000, 4000, 8500)
    stub_probs = [_STUB_MID] * 3 + [_STUB_HIGH] * 10 + [_STUB_MID] * 3 + [_STUB_LOW] * 10 + [_STUB_MID] * 2
    stub_probs += [_STUB_HIGH]

    def check_stub(computed: Mapping[str, Any]) -> None:
        first_on, end, second_on = computed[SECTION_EVENTS]
        probs = computed[SECTION_PROBS]
        _require(probs[0]["k"] == 179, f"the first model call is at step {probs[0]['k']}")
        for event in (first_on, end, second_on):
            _require(event["source"] == SOURCE_MODEL and not event["provisional"], "model-driven drowsy events")
        _require(first_on["confidence"] == _STUB_HIGH[CLASS_DROWSY], "confidence is p_drowsy of the emitting step")
        _require(end["confidence"] == _STUB_LOW[CLASS_DROWSY], "confidence is p_drowsy of the emitting step")
        refractory_ms = ctx.rules.drowsiness.model.refractory_ms
        _require(second_on["t_ms"] - end["t_ms"] == refractory_ms, "the second onset waits for the refractory time")
        _require(second_on["duration_ms"] > 10000, "its start is the first step of the run, inside the refractory time")

    perclos = [Closure(hold_ms=800, start_ms=30000 + 2000 * index) for index in range(15)]

    def check_perclos(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["source"] == SOURCE_PERCLOS and end["source"] == SOURCE_PERCLOS, "PERCLOS-driven drowsy events")
        steps = {step["k"]: step for step in computed[SECTION_STEPS]}
        first_k = min(k for k, step in steps.items() if step["full_lookback"])
        _require(onset["t_start_ms"] == steps[first_k]["t_end_ms"], "the run starts at the first full-lookback step")
        _require(onset["duration_ms"] == 9000, "onset at the tenth consecutive step")
        _require(onset["confidence"] == 1.0 - steps[first_k + 9]["values"][F_FACE_MISSING_RATIO_60], "confidence")
        _require(steps[first_k - 1]["values"][F_PERCLOS_60] > 0.15, "steps without full lookback do not count")

    calibrating = [Blink(400, start_ms=start) for start in range(3000, 60000, 7000)]
    calibrating += [
        Closure(hold_ms=1500, depth=_LOW_FPS_DEPTH, start_ms=10000),
        HeadTurn(4000, yaw=35.0, start_ms=20000),
        FaceMissing(2000, start_ms=40000),  # 보정 중의 얼굴 소실: calibrating end의 confidence가 1보다 작아진다
        Closure(hold_ms=1500, depth=_LOW_FPS_DEPTH, start_ms=62000),
        HeadTurn(4000, yaw=35.0, start_ms=65000),
    ]

    def check_calibrating(computed: Mapping[str, Any]) -> None:
        events = computed[SECTION_EVENTS]
        flags = [event["provisional"] for event in events]
        _require(flags == [True, True, True, False, False, False, False, False], f"provisional flags are {flags}")
        _require(computed[SECTION_BASELINE]["t_complete_ms"] == 60000, "the baseline completes at 60 s")
        _require(events[3]["t_ms"] == computed[SECTION_BASELINE]["t_ms"], "calibrating ends with the baseline")
        _require(0.9 < events[3]["confidence"] < 0.98, "2 s without a face: the end confidence is the face-time ratio")

    unable = [
        FaceMissing(7000, start_ms=12000),
        Closure(hold_ms=2000, start_ms=19500),
        Closure(hold_ms=1500, start_ms=26000),
    ]

    def check_unable(computed: Mapping[str, Any]) -> None:
        onset, end, microsleep = computed[SECTION_EVENTS][:3]
        _require(onset["duration_ms"] == 0 and onset["t_start_ms"] == onset["t_ms"], "starts at the deciding frame")
        _require(microsleep["t_ms"] > end["t_ms"] + 2000, "the closure during unable_to_analyze must not count")

    speed_gates = [
        Closure(hold_ms=1400, start_ms=2000),
        Closure(hold_ms=1400, start_ms=6000),
        Closure(hold_ms=3900, start_ms=10000),
        Closure(hold_ms=3900, start_ms=17000),
        HeadTurn(4000, yaw=35.0, start_ms=24000),
        HeadTurn(4000, yaw=35.0, start_ms=31000),
        HeadTurn(4000, yaw=35.0, start_ms=37000),
    ]
    speed_series = ((1, 10.0), (5001, 25.0), (9001, 10.0), (13601, 25.0), (23001, 10.0), (30001, 25.0), (36001, 10.0))

    def check_speed(computed: Mapping[str, Any]) -> None:
        late_sleep = computed[SECTION_ALERTS][1]
        _require(late_sleep["type"] == EVENT_SLEEP and late_sleep["t_ms"] == 13601, "sleep alerts once speed allows it")
        elapsed = late_sleep["t_ms"] - late_sleep["t_start_ms"]
        _require(late_sleep["duration_ms"] == elapsed, "with its duration updated")

    def check_cooldown(computed: Mapping[str, Any]) -> None:
        onsets = [event["t_ms"] for event in computed[SECTION_EVENTS] if event["phase"] == _ONSET]
        cooldown_ms = events_cfg.microsleep.cooldown_ms
        _require(onsets[1] - onsets[0] < cooldown_ms <= onsets[2] - onsets[0], "second inside, third outside the wait")
        _require(onsets[2] - onsets[1] < cooldown_ms, "a suppressed onset must not restart the wait")

    def check_repeat(computed: Mapping[str, Any]) -> None:
        times = [alert["t_ms"] for alert in computed[SECTION_ALERTS] if alert["type"] == EVENT_SLEEP]
        _require(len(times) == 3, f"{len(times)} sleep alerts")
        _require(times[1] - times[0] == 3000 and times[2] - times[1] == 3000, f"sleep alerts at {times}")

    boundary_fps = 1000.0 / _BOUNDARY_FRAME_MS
    return [
        closure_case(
            "rules_closure_900ms",
            "0.9초 감김(closure가 규칙용 진입 임계 0.7을 넘은 시간이 900 ms). 1.0초에 못 미치므로 이벤트가 없다.",
            825,
            (),
            (),
        ),
        closure_case(
            "rules_closure_1100ms",
            (
                "1.1초 감김. 유효한 감김 프레임에서 ongoing_ms가 1000 이상이 된 순간 microsleep onset, 감김이 끝난 "
                "프레임(closure < 0.5)에서 end. t_start_ms는 0.7을 위로 지난 시각(보간)이다. 속도 입력이 없으면 "
                "경고는 억제되지 않는다."
            ),
            1025,
            (_MICROSLEEP_ON, _MICROSLEEP_END),
            (_MICROSLEEP_ON,),
        ),
        closure_case(
            "rules_closure_2500ms",
            "2.5초 감김. microsleep만 나고 sleep(3.0초)은 나지 않는다. end는 microsleep의 것이다.",
            2425,
            (_MICROSLEEP_ON, _MICROSLEEP_END),
            (_MICROSLEEP_ON,),
        ),
        _Case(
            name="rules_closure_3000ms",
            bundle=BUNDLE_RULES,
            description=(
                "3.0초 감김의 경계(D2: sleep은 3.0초 '이상'). 프레임 간격 35 ms. 감김의 시작에서 정확히 3000 ms 뒤에 "
                "감긴 프레임이 오고 그다음 프레임에서는 눈을 떴다. 그 프레임에서 duration_ms = 3000인 sleep onset이 "
                "나야 한다('초과'로 구현하면 sleep이 나지 않는다). 한 감김은 microsleep onset, sleep onset, 도달한 "
                "가장 높은 종류(sleep)의 end 하나를 낸다."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(
                spec, [Closure(hold_ms=2972, close_ms=100, open_ms=50, start_ms=500)], 4200, fps=boundary_fps
            ),
            baseline=baseline,
            events=(_MICROSLEEP_ON, _SLEEP_ON, _SLEEP_END),
            alerts=(_MICROSLEEP_ON, _SLEEP_ON),
            check=check_sleep_at_boundary(repeat=False),
        ),
        _Case(
            name="rules_closure_6000ms",
            bundle=BUNDLE_RULES,
            description=(
                "6초 감김. 프레임 간격 35 ms. 3000 ms에 sleep onset과 경고, 그 경고에서 3000 ms 이상 지난 첫 프레임"
                "(감김 시작 뒤 6010 ms)에 sleep 경고가 한 번 되풀이되고 바로 다음 프레임에서 감김이 끝난다. 되풀이 "
                "경고는 onset의 사본이고 duration_ms만 그 시각 기준이다. 6초를 넘는 감김에 따로 나는 이벤트는 "
                "없다(무반응 운전자는 범위 밖, D15)."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(
                spec, [Closure(hold_ms=5982, close_ms=100, open_ms=50, start_ms=500)], 7200, fps=boundary_fps
            ),
            baseline=baseline,
            events=(_MICROSLEEP_ON, _SLEEP_ON, _SLEEP_END),
            alerts=(_MICROSLEEP_ON, _SLEEP_ON, _SLEEP_ON),
            check=check_sleep_at_boundary(repeat=True),
        ),
        gaze_case(
            "rules_gaze_2900ms",
            "2.9초 시선 이탈(상대 yaw가 23도를 넘은 시간이 약 2.9초인 고개 돌림). 3.0초에 못 미치므로 이벤트가 없다.",
            3294,
            (),
        ),
        replace(
            gaze_case(
                "rules_gaze_3100ms",
                (
                    "3.1초 시선 이탈(고개 돌림, source head_pose). 영역 밖이 시작된 프레임에서 정확히 3000 ms 뒤의 "
                    "프레임에 gaze_away_long onset(duration_ms = 3000), 영역 안이 250 ms 유지되어 확정된 프레임에 "
                    "end. t_start_ms는 영역 밖을 처음 본 프레임으로 소급한 값이다."
                ),
                3494,
                (_GAZE_ON, _GAZE_END),
            ),
            check=check_source(SOURCE_HEAD_POSE, exact_ms=3000),
        ),
        _Case(
            name="rules_gaze_eye_only",
            bundle=BUNDLE_RULES,
            description=(
                "눈만 돌린 시선 이탈(D12). 머리는 정면이고 gaze_h의 기준선 대비 변화가 0.25를 넘는다. "
                "gaze_away_long의 source는 eye다."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(spec, [LookAside(3600, gaze_h=0.4, start_ms=500)], 4900, fps=15.0),
            baseline=baseline,
            events=(_GAZE_ON, _GAZE_END),
            alerts=(_GAZE_ON,),
            check=check_source(SOURCE_EYE),
        ),
        _Case(
            name="rules_gaze_face_lost_in_turn",
            bundle=BUNDLE_RULES,
            description=(
                "고개를 60도 돌린 6초 동안 얼굴이 4.5초 사라진다. 영역 밖에서 얼굴이 사라지면 3초까지는 영역 밖으로 "
                "치므로 얼굴이 없는 동안 gaze_away_long onset이 난다(confidence는 그 구간의 pose_valid 시간 비율이라 "
                "낮다). 3초가 지나면 '알 수 없음'으로 타이머가 멈추고, 얼굴이 돌아와 정면이 확정되면 end가 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(
                spec, [HeadTurn(6000, yaw=60.0, start_ms=1010), FaceMissing(4500, start_ms=1500)], 8000, fps=15.0
            ),
            baseline=baseline,
            events=(_GAZE_ON, _GAZE_END),
            alerts=(_GAZE_ON,),
            check=check_face_lost,
        ),
        _Case(
            name="rules_vats",
            bundle=BUNDLE_RULES,
            description=(
                "VATS 누적. 약 2.6초씩의 시선 이탈 넷(각각은 3초에 못 미쳐 gaze_away_long이 없다)이 6초 간격으로 온다. "
                "최근 30초의 영역 밖 누적이 10초에 도달한 프레임에서 vats onset, 첫 이탈이 30초 창을 벗어나 누적이 "
                "10초 아래로 내려간 프레임에서 end. t_start_ms는 시작할 때 창 안의 첫 영역 밖 구간의 시작이다. "
                "속도 30 km/h(기준 20 이상)라 경고가 나간다."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(spec, [HeadTurn(3000, yaw=35.0, start_ms=start) for start in vats_starts], 36000, fps=10.0),
            baseline=baseline,
            speed=((1, 30.0),),
            events=(_VATS_ON, _VATS_END),
            alerts=(_VATS_ON,),
            check=check_vats,
        ),
        _Case(
            name="rules_drowsy_model_stub",
            bundle=BUNDLE_RULES,
            description=(
                "모델 졸음 판정(drowsiness.source = model)을 스텁 확률로 확인한다. 3 fps의 눈 뜬 프레임 266초. "
                "모델은 스텝 179(full_lookback 스텝이 120개 이어진 첫 스텝)부터 매 스텝 불린다. p_drowsy: 0.5 셋, "
                "0.7 열(열 번째에 onset, t_start_ms는 그 연속 구간의 첫 스텝 끝), 0.5 셋, 0.3 열(열 번째에 end), "
                "0.5 둘, 그 뒤 0.7. 다시 열 번이 찬 뒤에도 종료 후 60초 동안은 시작하지 않고, 60초가 된 스텝에서 "
                "onset이 난다(t_start_ms는 그 연속 구간의 첫 스텝 끝). confidence는 발행한 스텝의 p_drowsy다. "
                "속도 60 km/h(기준 50 이상)라 첫 onset은 경고가 되고, 둘째는 재경고 대기(300초)에 걸린다."
            ),
            sections=(SECTION_STEP_PROBS, SECTION_PROBS, *_RULE_SECTIONS),
            frames=_render(spec, [], 266500, fps=3.0),
            baseline=baseline,
            overrides={"drowsiness.source": "model"},
            speed=((1, 60.0),),
            model={"kind": MODEL_KIND_STUB, "probs": [list(probs) for probs in stub_probs]},
            events=(_DROWSY_ON, _DROWSY_END, _DROWSY_ON),
            alerts=(_DROWSY_ON,),
            check=check_stub,
        ),
        _Case(
            name="rules_drowsy_perclos",
            bundle=BUNDLE_RULES,
            description=(
                "PERCLOS 졸음 판정(기본 판정원). 5 fps, 30~60초에 0.9초 감김 열다섯(각각 1초에 못 미쳐 microsleep이 "
                "없다)으로 perclos_60이 0.2가 된다. full_lookback인 스텝만 세므로 스텝 60~69의 열 번째에 onset"
                "(t_start_ms는 스텝 60의 끝), 감김이 60초 구간을 벗어나 0.10 미만이 열 번 이어진 스텝에서 end. "
                "confidence는 1 − face_missing_ratio_60이다. 속도 40 km/h(기준 50 미만)라 경고는 억제되고 상태 "
                "이벤트는 그대로 난다."
            ),
            sections=(SECTION_STEPS, *_RULE_SECTIONS),
            frames=_render(spec, perclos, 120000, fps=5.0),
            baseline=baseline,
            speed=((1, 40.0),),
            events=(_DROWSY_ON, _DROWSY_END),
            alerts=(),
            check=check_perclos,
        ),
        _Case(
            name="rules_calibrating_provisional",
            bundle=BUNDLE_RULES,
            description=(
                "보정 중과 임시 판정(D4). 기준선을 넣지 않은 세션. 첫 프레임에 calibrating onset, 기준선이 완성된 "
                "프레임(세션 60초)에 end. 보정 중의 1.5초 감김은 전역 상수 closure로 판정해 provisional = true인 "
                "microsleep이 되고 경고도 나간다. 보정 중의 고개 돌림(4초)은 이벤트가 없다(시선 규칙은 기준선 완성 "
                "뒤에 시작). 완성 뒤의 같은 감김과 고개 돌림은 provisional = false인 microsleep과 gaze_away_long이다. "
                "보정 중 40~42초에 얼굴이 없어서 calibrating end의 confidence(그 구간의 얼굴 시간 비율)는 1보다 작다."
            ),
            sections=(SECTION_BASELINE, *_RULE_SECTIONS),
            frames=_render(spec, calibrating, 70500, fps=6.0),
            events=(
                _CALIBRATING_ON,
                _MICROSLEEP_ON,
                _MICROSLEEP_END,
                _CALIBRATING_END,
                _MICROSLEEP_ON,
                _MICROSLEEP_END,
                _GAZE_ON,
                _GAZE_END,
            ),
            alerts=(_MICROSLEEP_ON, _MICROSLEEP_ON, _GAZE_ON),
            check=check_calibrating,
        ),
        _Case(
            name="rules_unable_to_analyze",
            bundle=BUNDLE_RULES,
            description=(
                "분석 불가. 12~19초에 얼굴이 없다. 최근 10초의 얼굴 시간 비율이 0.5 아래로 내려간 프레임에서 onset"
                "(t_start_ms는 그 프레임, duration_ms 0), 0.5 이상으로 돌아온 프레임에서 end. 분석 불가인 동안의 "
                "2초 감김(19.5초~)은 microsleep이 되지 않고, 끝난 뒤의 1.5초 감김은 microsleep이 된다."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(spec, unable, 29000, fps=10.0),
            baseline=baseline,
            events=(_UNABLE_ON, _UNABLE_END, _MICROSLEEP_ON, _MICROSLEEP_END),
            alerts=(_UNABLE_ON, _MICROSLEEP_ON),
            check=check_unable,
        ),
        _Case(
            name="rules_speed_gates",
            bundle=BUNDLE_RULES,
            description=(
                "속도 억제(D3: 20 km/h 미만에서 microsleep, sleep, gaze_away_long, vats 경고를 억제). 속도는 10과 "
                "25 km/h를 오간다. 10일 때의 microsleep, gaze_away_long, vats는 상태 이벤트만 나고 경고가 없다. "
                "25일 때의 microsleep, sleep, gaze_away_long은 경고가 된다. 10에서 시작된 sleep은 onset 때 억제되고, "
                "상태가 이어지는 동안 속도가 25가 된 첫 프레임(13601)에 경고가 나간다(되풀이하는 종류만 그렇다). "
                "vats의 기준 이상은 rules_vats, drowsy의 기준(50 km/h) 위아래는 rules_drowsy_model_stub과 "
                "rules_drowsy_perclos가 고정한다."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(spec, speed_gates, 42000, fps=10.0),
            baseline=baseline,
            speed=speed_series,
            events=(
                _MICROSLEEP_ON,
                _MICROSLEEP_END,
                _MICROSLEEP_ON,
                _MICROSLEEP_END,
                _MICROSLEEP_ON,
                _SLEEP_ON,
                _SLEEP_END,
                _MICROSLEEP_ON,
                _SLEEP_ON,
                _SLEEP_END,
                _GAZE_ON,
                _GAZE_END,
                _GAZE_ON,
                _GAZE_END,
                _VATS_ON,
                _GAZE_ON,
                _GAZE_END,
            ),
            alerts=(_MICROSLEEP_ON, _SLEEP_ON, _MICROSLEEP_ON, _SLEEP_ON, _GAZE_ON),
            check=check_speed,
        ),
        _Case(
            name="rules_cooldown",
            bundle=BUNDLE_RULES,
            description=(
                "재경고 대기(microsleep 5초). 1.5초 감김 셋. 첫 onset은 경고, 3초 뒤의 둘째는 대기에 걸려 경고가 "
                "없고, 첫 경고에서 5초 넘게 지난 셋째는 경고가 된다. 대기에 걸린 onset은 대기를 새로 시작하지 않는다"
                "(셋째는 둘째에서 5초가 안 됐다). 상태 이벤트는 세 번 모두 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(spec, [Closure(hold_ms=1400, start_ms=start) for start in cooldown_starts], 11000, fps=15.0),
            baseline=baseline,
            events=(_MICROSLEEP_ON, _MICROSLEEP_END) * 3,
            alerts=(_MICROSLEEP_ON, _MICROSLEEP_ON),
            check=check_cooldown,
        ),
        _Case(
            name="rules_sleep_repeat",
            bundle=BUNDLE_RULES,
            description=(
                "sleep 경고의 되풀이. 10초 감김(10 fps). 1초에 microsleep, 3초에 sleep 경고가 나가고, 상태가 "
                "이어지는 동안 3000 ms마다 sleep 경고가 되풀이된다(모두 세 번). 상태 이벤트는 sleep onset 하나와 "
                "end 하나뿐이다."
            ),
            sections=_RULE_SECTIONS,
            frames=_render(spec, [Closure(hold_ms=10000, start_ms=1000)], 12500, fps=10.0),
            baseline=baseline,
            events=(_MICROSLEEP_ON, _SLEEP_ON, _SLEEP_END),
            alerts=(_MICROSLEEP_ON, _SLEEP_ON, _SLEEP_ON, _SLEEP_ON),
            check=check_repeat,
        ),
    ]


# --- 규칙: 경계와 2026-11-13 결정 --------------------------------------------

# 3 fps의 프레임 시각(세션 시간). 1초에 0, 333, 667 ms의 세 장이고 프레임마다 200 ms(`clock.max_hold_ms`)씩 인정된다.
_THIRD_MS: tuple[int, ...] = (0, 333, 667)


def _three_fps(start_s: int, stop_s: int) -> list[int]:
    return [1000 * second + offset for second in range(start_s, stop_s) for offset in _THIRD_MS]


def _boundary_rule_cases(ctx: Context) -> list[_Case]:
    """손으로 놓은 프레임으로 만든 규칙 벡터: 정수 ms의 경계, 프레임 끊김, 2026-11-13 소유자 결정 네 가지.

    시각은 세션 시간으로 적는다(`t_ms` = 세션 시간 + 1). 감김은 0.0 → 0.9의 계단이라 규칙용 검출기(진입 0.7)에서
    감김의 시작은 100 ms 간격이면 첫 감긴 프레임의 22 ms 앞, 200 ms 간격이면 44 ms 앞이다(0.7 / 0.9를 지난 자리).
    0.6(이탈 0.5와 진입 0.7 사이)에서 0.9로 오를 때는 100 ms 중 33 ms, 0.9에서 0.6으로 내릴 때는 67 ms에 0.7을 지난다.
    """
    spec = ctx.spec
    baseline = _personal_baseline(spec)
    rules = ctx.rules
    first = _FIRST_T_MS
    half_open = _Shot(level=0.6)  # 규칙용 검출기의 이탈(0.5)과 진입(0.7) 사이
    no_face = _Shot(face=False)
    away = _Shot(yaw=35.0)
    away_shut = _Shot(yaw=35.0, level=0.9)

    def times_of(items: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, int]]:
        return [(item["type"], item["phase"], item["t_ms"] - first) for item in items]

    def expect(
        events: Sequence[tuple[str, str, int]], alerts: Sequence[tuple[str, str, int]]
    ) -> Callable[[Mapping[str, Any]], None]:
        """이벤트와 경고의 (종류, 단계, 세션 시간)이 각본의 의도와 같은지 확인한다."""

        def check(computed: Mapping[str, Any]) -> None:
            got = times_of(computed[SECTION_EVENTS])
            _require(got == list(events), f"events are {got}, the script intends {list(events)}")
            got = times_of(computed[SECTION_ALERTS])
            _require(got == list(alerts), f"alerts are {got}, the script intends {list(alerts)}")

        return check

    def both(*checks: Callable[[Mapping[str, Any]], None]) -> Callable[[Mapping[str, Any]], None]:
        def check(computed: Mapping[str, Any]) -> None:
            for item in checks:
                item(computed)

        return check

    ms_on, ms_end = (EVENT_MICROSLEEP, _ONSET), (EVENT_MICROSLEEP, "end")
    sleep_on, sleep_end = (EVENT_SLEEP, _ONSET), (EVENT_SLEEP, "end")
    gaze_on, gaze_end = (EVENT_GAZE_AWAY_LONG, _ONSET), (EVENT_GAZE_AWAY_LONG, "end")
    vats_on, vats_end = (EVENT_VATS, _ONSET), (EVENT_VATS, "end")
    drowsy_on, drowsy_end = (EVENT_DROWSY, _ONSET), (EVENT_DROWSY, "end")
    unable_on, unable_end = ("unable_to_analyze", _ONSET), ("unable_to_analyze", "end")

    # ---- 감김의 길이 (결정 1) ----
    dwell_plan = [(0, _OPEN), (1000, _SHUT), (1400, half_open), (6400, _SHUT)]

    def check_reaches(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        # 첫 토막 978~1367(389 ms), 둘째 토막은 6333에서 시작한다. 6944의 프레임에서 389 + 611 = 1000 ms다.
        _require(onset["t_start_ms"] == first + 978 and onset["duration_ms"] == 6944 - 978, f"onset is {onset}")
        _require(end["duration_ms"] == 7300 - 978, f"end is {end}")
        _require(onset["duration_ms"] > 5 * rules.closure.microsleep_ms, "duration_ms covers the half-open dwell")

    def check_three_spans(computed: Mapping[str, Any]) -> None:
        microsleep, sleep, end = computed[SECTION_EVENTS]
        # 토막 978~1467(489), 1733~2067(334), 2333~. 2510에서 823 + 177 = 1000, 4510에서 823 + 2177 = 3000 ms다.
        _require(microsleep["t_start_ms"] == sleep["t_start_ms"] == end["t_start_ms"] == first + 978, "one run")
        _require(microsleep["duration_ms"] == 2510 - 978 and sleep["duration_ms"] == 4510 - 978, "event lengths")

    # ---- 미뤄 둔 경고 (결정 2) ----
    cooldown_plan = [(0, _OPEN), (1000, _SHUT), (2200, _OPEN), (3000, _SHUT), (4300, _OPEN), (5000, _SHUT)]
    cooldown_plan += [(7600, _OPEN), (8000, _SHUT), (9300, _OPEN)]

    def check_deferred_cooldown(computed: Mapping[str, Any]) -> None:
        immediate, deferred = computed[SECTION_ALERTS]
        third_onset = computed[SECTION_EVENTS][4]
        _require(deferred["t_ms"] - immediate["t_ms"] == rules.events.microsleep.cooldown_ms, "exactly at the expiry")
        _require(deferred["t_start_ms"] == third_onset["t_start_ms"] == first + 4978, "a copy of the third onset")
        _require(deferred["duration_ms"] == 7000 - 4978 and third_onset["duration_ms"] == 6000 - 4978, "durations")
        _require(deferred["confidence"] == third_onset["confidence"], "the copy keeps the onset's confidence")

    blocked_plan = [(0, _OPEN), (1000, _SHUT), (4500, _OPEN), (7000, _SHUT), (11500, _OPEN)]
    blocked_speed = ((first, 10.0), (first + 5000, 25.0), (first + 6000, 10.0), (first + 10500, 20.0))

    def check_blocked(computed: Mapping[str, Any]) -> None:
        (alert,) = computed[SECTION_ALERTS]
        _require(alert["type"] == EVENT_SLEEP and alert["duration_ms"] == 10500 - 6978, f"the alert is {alert}")
        _require(rules.events.sleep.min_speed_kmh == 20.0, "the vector pins the speed gate at exactly 20.0 km/h")

    # ---- PERCLOS 판정의 등호 자리와 속도로 미뤄진 drowsy 경고 (결정 2) ----
    # 3 fps: 60초 구간에 프레임 180장, 유효 시간 36000 ms. 감긴 프레임(0.9) 27장이면 5400 / 36000 = 0.15,
    # 18장이면 3600 / 36000 = 0.10이다(둘 다 정수 ms의 몫이라 임계값과 정확히 같다).
    exact_times = _three_fps(0, 112)
    exact_closed = [11, 14, 17, 41, 44, 47, 50, 53, 56]  # 감긴 프레임 세 장이 들어가는 초
    exact_plan: list[tuple[int, _Shot]] = [(0, _OPEN)]
    for second in exact_closed:
        exact_plan += [(1000 * second, _SHUT), (1000 * second + 1000, _OPEN)]

    def check_perclos_exact(computed: Mapping[str, Any]) -> None:
        steps = {step["k"]: step for step in computed[SECTION_STEPS]}
        onset_ratio, end_ratio = rules.drowsiness.perclos.onset_ratio, rules.drowsiness.perclos.end_ratio
        for k in range(60, 72):
            _require(steps[k]["values"][F_PERCLOS_60] == onset_ratio, f"step {k}: perclos_60 must be the onset ratio")
        for k in range(78, 102):
            _require(steps[k]["values"][F_PERCLOS_60] == end_ratio, f"step {k}: perclos_60 must equal the end ratio")
        _require(steps[102]["values"][F_PERCLOS_60] < end_ratio, "step 102 is the first below the end ratio")
        onset, end = computed[SECTION_EVENTS]
        _require(onset["t_start_ms"] == steps[60]["t_end_ms"] and onset["duration_ms"] == 9000, f"onset is {onset}")
        (alert,) = computed[SECTION_ALERTS]
        _require(alert["duration_ms"] == 75000 - 60000 and alert["confidence"] == onset["confidence"], "deferred copy")
        _require(rules.events.drowsy.min_speed_kmh == 50.0, "the vector pins the speed gate at exactly 50.0 km/h")

    # ---- 밀린 스텝마다 모델 호출 (결정 3) ----
    # 176초까지 3 fps, 그 뒤는 5 fps(끊김이 있어도 최근 10초의 얼굴 시간이 5초 밑으로 내려가지 않게 한다. 3 fps로
    # 두면 끊김마다 unable_to_analyze가 함께 난다). 모델은 스텝 179부터 불린다. 프레임이 2600 ms 끊기는 곳이 둘:
    # 시작 조건이 이어지는 중(스텝 185~187을 한 프레임이 발행)과 종료 조건이 이어지는 중(스텝 198~200).
    stall_times = _three_fps(0, 176) + list(range(176000, 204400, 200))
    stall_times = [t for t in stall_times if not (184400 < t < 187000 or 197400 < t < 200000)]
    stall_probs = [_STUB_MID] * 3 + [_STUB_AT_ONSET] * 10 + [_STUB_MID] * 3 + [_STUB_LOW] * 10

    def check_stall(computed: Mapping[str, Any]) -> None:
        step_probs = computed[SECTION_STEP_PROBS]
        called = [entry["k"] for entry in step_probs if entry["probs"] is not None]
        _require([entry["k"] for entry in step_probs] == list(range(1, 205)), "one entry per emitted step")
        _require(called == list(range(179, 205)), f"the model must run at every step from 179, got {called}")
        for index, entry in enumerate(entry for entry in step_probs if entry["probs"] is not None):
            _require(entry["probs"] == list(stall_probs[index]), "call order = step order")
        for emitter, ks in ((187000, [185, 186, 187]), (200000, [198, 199, 200])):
            got = [entry["k"] for entry in step_probs if entry["t_ms"] == first + emitter]
            _require(got == ks, f"the frame after the stall emits steps {got}")
        per_frame = [entry["k"] for entry in computed[SECTION_PROBS]]
        _require(185 not in per_frame and 186 not in per_frame and 187 in per_frame, "probs holds the last step only")
        onset, end = computed[SECTION_EVENTS]
        _require(onset["t_start_ms"] == first + 182000 and onset["duration_ms"] == 9000, f"onset is {onset}")
        _require(end["duration_ms"] == 204000 - 182000, f"end is {end}")
        _require(computed[SECTION_COUNTERS]["model_calls"] == len(stall_probs) == 26, "26 model calls")

    # ---- 10초 넘는 끊김 (결정 4) ----
    gap_lead = list(range(0, 14800, 200))  # 마지막 프레임 14600
    gap_plan = [(0, _OPEN), (1000, away), (11400, away_shut)]

    def check_gap_over(computed: Mapping[str, Any]) -> None:
        events = computed[SECTION_EVENTS]
        at_gap = [event for event in events if event["t_ms"] == first + 24601]
        _require([event["type"] for event in at_gap] == [EVENT_GAZE_AWAY_LONG, EVENT_SLEEP, EVENT_VATS], "end order")
        gaze, sleep, vats = at_gap
        _require((gaze["duration_ms"], sleep["duration_ms"], vats["duration_ms"]) == (23601, 13245, 23601), "ends")
        _require(gaze["confidence"] == 13800 / 23601 and vats["confidence"] == 13800 / 23601, "pose-valid time ratio")
        _require(sleep["confidence"] == 3400 / 13201, "closure-valid time ratio")
        after = [event for event in events if event["t_ms"] > first + 24601 and event["phase"] == _ONSET]
        _require([event["duration_ms"] for event in after] == [1000, 3000, 3000], "restart: exact 1000 and 3000 ms")
        _require(all(event["t_start_ms"] == first + 24601 for event in after), "new states start at the first frame")
        _require(not any(event["type"] == "unable_to_analyze" for event in events), "no unable_to_analyze")

    def check_gap_exact(computed: Mapping[str, Any]) -> None:
        events = {(event["type"], event["phase"], event["t_ms"] - first): event for event in computed[SECTION_EVENTS]}
        sleep = events[(EVENT_SLEEP, "end", 24600)]
        _require(sleep["duration_ms"] == 24600 - 11356 and sleep["confidence"] == 3400 / 13200, f"sleep end is {sleep}")
        unable = events[("unable_to_analyze", "end", 31600)]
        _require(unable["duration_ms"] == 7000 and unable["confidence"] == 5000 / 7000, f"unable end is {unable}")
        gaze = events[(EVENT_GAZE_AWAY_LONG, "end", 32000)]
        _require(gaze["t_start_ms"] == first + 1000 and gaze["confidence"] == 19200 / 31000, f"gaze end is {gaze}")
        vats = events[(EVENT_VATS, "end", 38000)]
        _require(vats["duration_ms"] == 37000 > rules.vats.window_ms, "the vats state outlasts the 30 s window")
        _require(vats["confidence"] == 25200 / 37000, f"vats end confidence is {vats['confidence']}")

    # 결정 4 + 2: 졸음 상태와 미뤄 둔 졸음 경고가 10초 넘는 끊김에서 끝난다. 3 fps, 감긴 프레임 27장으로 perclos_60 0.15.
    drowsy_gap_times = [t for t in _three_fps(0, 71) if t <= 70000] + [80001 + 333 * index for index in range(10)]
    drowsy_gap_plan: list[tuple[int, _Shot]] = [(0, _OPEN)]
    for second in exact_closed:
        drowsy_gap_plan += [(1000 * second, _SHUT), (1000 * second + 1000, _OPEN)]
    drowsy_gap_plan += [(66000, away), (80001, _OPEN)]

    def check_drowsy_gap(computed: Mapping[str, Any]) -> None:
        onset, _, end, gaze = computed[SECTION_EVENTS]
        _require(end["t_start_ms"] == onset["t_start_ms"] == first + 60000 and end["source"] == SOURCE_PERCLOS, "pair")
        _require(end["duration_ms"] == 80001 - 60000 and end["confidence"] == 0.0, f"drowsy end is {end}")
        _require(gaze["duration_ms"] == 80001 - 66000, f"gaze end is {gaze}")
        steps = [step["k"] for step in computed[SECTION_STEPS]]
        _require(steps == list(range(1, 71)) + [81, 82], f"steps are {steps}")

    boundary_times = list(range(0, 10200, 200)) + list(range(15200, 25600, 200)) + list(range(30601, 36000, 200))

    def check_gap_boundary(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(end["duration_ms"] == 5000 and end["confidence"] == 1.0, f"unable end is {end}")
        gaps = [later - earlier for earlier, later in zip(boundary_times, boundary_times[1:])]
        _require(sorted(set(gaps)) == [200, 5200, 5201], "one gap of 5200 ms (no event) and one of 5201 ms (onset)")

    sparse_times = list(range(0, 12500, 500)) + list(range(22500, 41000, 500))

    def check_sparse(computed: Mapping[str, Any]) -> None:
        first_on, end, second_on = computed[SECTION_EVENTS]
        _require(end["duration_ms"] == 12500 and end["confidence"] == 1000 / 12500, f"unable end is {end}")
        immediate, deferred = computed[SECTION_ALERTS]
        _require(deferred["t_ms"] - immediate["t_ms"] == rules.events.unable_to_analyze.cooldown_ms, "at the expiry")
        _require(deferred["t_start_ms"] == second_on["t_start_ms"] and deferred["duration_ms"] == 7500, "deferred copy")

    # ---- 전방 주시의 경계 ----
    missed_plan = [(0, _OPEN), (1000, away), (1300, no_face), (5100, _OPEN)]
    missed_times = list(range(0, 4000, 100)) + list(range(4300, 5700, 100))

    def check_missed(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["t_start_ms"] == first + 1000 and onset["duration_ms"] == 3300, f"onset is {onset}")
        _require(onset["confidence"] == 300 / 3300, "three pose-valid frames of 100 ms in 3300 ms")
        frames = {frame.t_ms: frame for frame in missed_frames}
        _require(not frames[onset["t_ms"]].face and first + 4200 not in frames, "onset on the first frame after it")

    missed_frames = _hand_frames(spec, missed_times, missed_plan)

    credit_plan = [
        (0, _OPEN),
        # 1: 0.4초 이탈, 앞을 본 프레임 하나, 그 뒤 인정 구간이 끝나는 바로 그 시각(1400 + 200 + 3000)에 얼굴 없는 프레임
        (1000, away),
        (1400, _OPEN),
        (4600, no_face),
        (4800, _OPEN),
        # 2: 이탈 3.5초(gaze_away_long 진행 중), 앞을 본 프레임 하나, 인정 구간이 끝나는 바로 그 시각(14700)의 프레임
        (8000, away),
        (11500, _OPEN),
        (14700, no_face),
        (14900, _OPEN),
        # 3: 같은 모양인데 다음 프레임이 인정 구간이 끝나기 50 ms 전(21500 + 200 + 2950)에 온다
        (18000, away),
        (21500, _OPEN),
        (24650, no_face),
        (24700, _OPEN),
    ]
    credit_times = list(range(0, 1500, 100)) + list(range(4600, 11600, 100)) + list(range(14700, 21600, 100))
    credit_times += [24650] + list(range(24700, 25200, 100))

    debounce_plan = [(0, away), (3500, _OPEN), (3700, away), (4500, _OPEN)]
    debounce_times = list(range(0, 3400, 100)) + list(range(3400, 5050, 50))

    def check_debounce(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["duration_ms"] == 3000 and end["duration_ms"] == 4750, "end exactly 250 ms after the candidate")

    # 영역의 여섯 경계: 바깥쪽 값으로 3.0초(정확히 3000 ms에 onset), 안쪽 값으로 돌아와 400 ms 뒤 확정(end)
    edges = (
        (_Shot(yaw=23.01), _Shot(yaw=22.99), SOURCE_HEAD_POSE),
        (_Shot(yaw=-23.01), _Shot(yaw=-22.99), SOURCE_HEAD_POSE),
        (_Shot(pitch=-25.01), _Shot(pitch=-24.99), SOURCE_HEAD_POSE),
        (_Shot(gaze_h=-0.251), _Shot(gaze_h=-0.249), SOURCE_EYE),
        (_Shot(gaze_v=0.351), _Shot(gaze_v=0.349), SOURCE_EYE),
        (_Shot(pitch=18.01), _Shot(pitch=17.99), SOURCE_HEAD_POSE),
    )
    edge_plan: list[tuple[int, _Shot]] = [(0, _OPEN)]
    for index, (outside, inside, _) in enumerate(edges):
        edge_plan += [(1000 + 3800 * index, outside), (1000 + 3800 * index + 3200, inside)]

    def check_edges(computed: Mapping[str, Any]) -> None:
        gaze = [event for event in computed[SECTION_EVENTS] if event["type"] == EVENT_GAZE_AWAY_LONG]
        _require(len(gaze) == 12, f"{len(gaze)} gaze_away_long events, the script intends six pairs")
        for index, (_, _, source) in enumerate(edges):
            onset, end = gaze[2 * index], gaze[2 * index + 1]
            start = first + 1000 + 3800 * index
            _require(onset["t_start_ms"] == start and onset["duration_ms"] == 3000, f"onset {index} is {onset}")
            _require(end["t_ms"] == start + 3600 and onset["source"] == end["source"] == source, f"end is {end}")
        (vats,) = [event for event in computed[SECTION_EVENTS] if event["type"] == EVENT_VATS]
        _require(vats["source"] == SOURCE_EYE and vats["t_ms"] == first + 12800, f"vats onset is {vats}")

    # vats.reset_enabled: 2.8초 이탈 다섯 번. 첫 이탈 뒤에는 전방 주시가 정확히 2000 ms 이어져 합이 지워지고,
    # 그 뒤로는 1800 ms씩이라 지워지지 않는다.
    reset_plan: list[tuple[int, _Shot]] = [(0, _OPEN), (1000, away), (3800, _OPEN)]
    for start in (6000, 10600, 15200, 19800):
        reset_plan += [(start, away), (start + 2800, _OPEN)]

    def check_vats_reset(computed: Mapping[str, Any]) -> None:
        (onset,) = computed[SECTION_EVENTS]
        _require(onset["t_start_ms"] == first + 6000 and onset["duration_ms"] == 15400, f"vats onset is {onset}")

    def check_priority(computed: Mapping[str, Any]) -> None:
        events = [event for event in computed[SECTION_EVENTS] if event["t_ms"] == first + 4000]
        alerts = [alert for alert in computed[SECTION_ALERTS] if alert["t_ms"] == first + 4000]
        _require([event["type"] for event in events] == [EVENT_GAZE_AWAY_LONG, EVENT_MICROSLEEP], "event order")
        _require([alert["type"] for alert in alerts] == [EVENT_MICROSLEEP, EVENT_GAZE_AWAY_LONG], "alert order")

    # ---- 2차 검증에서 나온 자리: 벡터가 잡지 못하던 동작 ----
    def gaze_events(computed: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [event for event in computed[SECTION_EVENTS] if event["type"] == EVENT_GAZE_AWAY_LONG]

    # 영역 밖 후보가 걸린 채 10초 넘게 끊긴다. 2000의 프레임 하나만 고개를 돌렸고 다음 프레임은 10001 ms 뒤(12001)다.
    candidate_times = list(range(0, 2200, 200)) + [12001 + 200 * index for index in range(20)]

    def check_candidate_gap(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["t_start_ms"] == first + 12001 and onset["duration_ms"] == 3000, f"onset is {onset}")
        _require(onset["confidence"] == 1.0 and end["duration_ms"] == 3800, f"events are {onset}, {end}")

    # 영역 안 후보가 인정 구간 뒤의 프레임에서 버려진다. 2.0초 이탈, 앞을 본 프레임 하나(3000), 3.3초 뒤에 다시 이탈.
    dropped_times = list(range(0, 3100, 100)) + list(range(6300, 8400, 100))

    def check_dropped(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["t_start_ms"] == first + 1000 and onset["duration_ms"] == 6300, f"onset is {onset}")
        _require(onset["confidence"] == 3200 / 6300, "pose-valid time: 2000 + 200 + 1000 ms of 6300")
        _require(end["duration_ms"] == 7300 and end["confidence"] == 4200 / 7300, f"end is {end}")

    # 눈동자 보조의 나머지 두 방향: gaze_h는 양수 쪽, gaze_v는 음수 쪽(아래). 바깥 값으로 3.2초, 안쪽 값으로 돌아온다.
    eye_edges = (
        (_Shot(gaze_h=0.251), _Shot(gaze_h=0.249)),
        (_Shot(gaze_v=-0.351), _Shot(gaze_v=-0.349)),
    )
    eye_edge_plan: list[tuple[int, _Shot]] = [(0, _OPEN)]
    for index, (outside, inside) in enumerate(eye_edges):
        eye_edge_plan += [(1000 + 3800 * index, outside), (1000 + 3800 * index + 3200, inside)]

    def check_eye_edges(computed: Mapping[str, Any]) -> None:
        gaze = gaze_events(computed)
        _require(len(gaze) == 4 and len(computed[SECTION_EVENTS]) == 4, f"{len(gaze)} gaze_away_long events")
        for event in gaze:
            _require(event["source"] == SOURCE_EYE, f"source is {event['source']}")
        _require(gaze[0]["duration_ms"] == 3000 and gaze[2]["duration_ms"] == 3000, "onsets after exactly 3000 ms")

    # 분석 불가가 연속 영역 밖 시간을 멈춘다. 200 ms 간격. 얼굴: 0~11800 정면, 12000~15800 없음, 16000~16400 고개 돌림,
    # 16600~39800 없음, 40000~55800 고개 돌림, 56000부터 정면.
    pause_plan = [(0, _OPEN), (12000, no_face), (16000, away), (16600, no_face), (40000, away), (56000, _OPEN)]

    def check_pause(computed: Mapping[str, Any]) -> None:
        unable_start, unable_stop, gaze, vats, gaze_stop = computed[SECTION_EVENTS]
        _require(unable_stop["duration_ms"] == 27200 and unable_stop["confidence"] == 5000 / 27200, "unable end")
        # 분석 불가가 시작될 때까지 1800 ms(16000~17800), 끝난 뒤 1200 ms(45000~46200)
        _require(gaze["t_start_ms"] == first + 16000 and gaze["duration_ms"] == 30200, f"gaze onset is {gaze}")
        _require(gaze["confidence"] == 6800 / 30200, "pose-valid time: 600 + 6200 ms of 30200")
        # 창 안의 첫 영역 밖 구간은 45000에서 시작한다. 16000~17800의 구간은 47800에 30초 창을 벗어났다.
        _require(vats["t_start_ms"] == first + 45000 and vats["duration_ms"] == 10000, f"vats onset is {vats}")
        _require(vats["confidence"] == 1.0 and gaze_stop["confidence"] == 17000 / 40400, "confidences")

    # either: 모델이 연 상태가 끝난 뒤 PERCLOS가 상태를 연다. 3 fps. 모델은 스텝 179부터 불린다: 0.7 열 번
    # (스텝 188에 onset), 0.3 열 번(스텝 198에 end), 그 뒤 0.5. 감긴 프레임 세 장이 든 초가 165, 168, ..., 189의 아홉이라
    # perclos_60은 스텝 190부터 5400 / 36000 = 0.15다(스텝 189는 4800 / 36000). 스텝 190~199의 열 번째에 onset.
    either_plan: list[tuple[int, _Shot]] = [(0, _OPEN)]
    for second in range(165, 190, 3):
        either_plan += [(1000 * second, _SHUT), (1000 * second + 1000, _OPEN)]
    either_probs = [_STUB_HIGH] * 10 + [_STUB_LOW] * 10 + [_STUB_MID]

    def check_either(computed: Mapping[str, Any]) -> None:
        steps = {step["k"]: step for step in computed[SECTION_STEPS]}
        onset_ratio = rules.drowsiness.perclos.onset_ratio
        _require(steps[189]["values"][F_PERCLOS_60] < onset_ratio, "step 189 is below the onset ratio")
        for k in range(190, 201):
            _require(steps[k]["values"][F_PERCLOS_60] == onset_ratio, f"step {k}: perclos_60 must be the onset ratio")
        by_model, model_end, by_perclos = computed[SECTION_EVENTS]
        _require(by_model["source"] == SOURCE_MODEL and model_end["source"] == SOURCE_MODEL, "the model opens first")
        _require(by_model["t_start_ms"] == model_end["t_start_ms"] == first + 179000, "model episode")
        _require(model_end["confidence"] == _STUB_LOW[CLASS_DROWSY], "a model end carries p_drowsy")
        _require(by_perclos["source"] == SOURCE_PERCLOS, f"the second state is opened by PERCLOS: {by_perclos}")
        _require(by_perclos["t_start_ms"] == first + 190000 and by_perclos["duration_ms"] == 9000, "PERCLOS episode")
        _require(abs(by_perclos["confidence"] - 0.6) < 1e-12, "confidence is 1 - face_missing_ratio_60 = 0.6")
        _require(computed[SECTION_COUNTERS]["model_calls"] == 22, "the model runs at steps 179..200")

    # 모델 판정기도 10초 넘는 끊김에서 처음 상태로 돌아간다. 3 fps. 스텝 188에 onset, 198에 end. 199000의 프레임 다음
    # 프레임이 10001 ms 뒤(209001)에 온다. 누적이 209000에서 다시 시작하므로 full_lookback은 스텝 269부터, 모델은
    # 스텝 388부터 다시 불린다. 0.7이 열 번 이어진 스텝 397에 onset이 난다.
    model_gap_times = [t for t in _three_fps(0, 200) if t <= 199000] + [209001 + t for t in _three_fps(0, 190)]
    model_gap_probs = [_STUB_HIGH] * 10 + [_STUB_LOW] * 10 + [_STUB_MID, _STUB_HIGH]
    model_gap_refractory_ms = 600000

    def check_model_gap(computed: Mapping[str, Any]) -> None:
        steps = computed[SECTION_STEPS]
        _require([step["k"] for step in steps] == [*range(1, 200), *range(210, 399)], "steps 1..199 and 210..398")
        full_after = [step["k"] for step in steps if step["k"] >= 210 and step["full_lookback"]]
        _require(full_after[0] == 269, f"full lookback restarts at step {full_after[0]}")
        called = [entry["k"] for entry in computed[SECTION_STEP_PROBS] if entry["probs"] is not None]
        _require(called == [*range(179, 200), *range(388, 399)], f"the model runs at steps {called}")
        first_on, end, second_on = computed[SECTION_EVENTS]
        _require(end["t_ms"] == first + 198000 and end["duration_ms"] == 19000, f"end is {end}")
        _require(second_on["t_ms"] == first + 397001, f"the second onset is emitted at {second_on['t_ms']}")
        _require(second_on["t_start_ms"] == first + 388000 and second_on["duration_ms"] == 9000, "second episode")
        waited = second_on["t_ms"] - end["t_ms"]
        _require(waited < model_gap_refractory_ms, "the second onset must fall inside the old refractory time")
        _require(second_on["t_ms"] - first < model_gap_refractory_ms, "and before one refractory time from t = 0")
        for event in (first_on, end, second_on):
            _require(event["source"] == SOURCE_MODEL, f"source is {event['source']}")

    # 미뤄 둔 microsleep 경고는 다른 종류의 end로는 지워지지 않는다. 200 ms 간격, 속도 10 km/h.
    survive_plan = [(0, _OPEN), (1000, away), (5000, away_shut), (6200, _SHUT), (7200, _OPEN)]

    def check_survives(computed: Mapping[str, Any]) -> None:
        (alert,) = computed[SECTION_ALERTS]
        microsleep = computed[SECTION_EVENTS][1]
        _require(alert["type"] == EVENT_MICROSLEEP and alert["t_start_ms"] == first + 4956, f"the alert is {alert}")
        _require(alert["duration_ms"] == 6800 - 4956 and microsleep["duration_ms"] == 6000 - 4956, "durations")
        _require(alert["confidence"] == microsleep["confidence"], "the copy keeps the onset's confidence")

    # 버려진 짧은 감김 뒤에 같은 프레임에서 시작한 새 감김. 100 ms 간격. 1000의 프레임 하나만 감겼고(규칙용 검출기로
    # 22 ms) 다음 프레임이 700 ms 뒤(1700)에 감긴 채 온다: 앞 감김은 잘리고 50 ms 미만이라 구간 없이 버려진다.
    restart_times = list(range(0, 1100, 100)) + list(range(1700, 3300, 100))

    def check_restart(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["t_start_ms"] == first + 1700 and onset["duration_ms"] == 1000, f"onset is {onset}")
        _require(onset["confidence"] == 1.0 and end["duration_ms"] == 1300, f"events are {onset}, {end}")

    # 잠깐 앞을 본 구간, 얼굴이 사라진 시각, 인정 구간의 끝에 온 관측. 200 ms 간격(12000 뒤의 프레임만 100 ms 뒤).
    # 1000부터 고개를 돌리고 2600의 프레임 하나만 앞을 본다. 12000에 앞을 본 프레임 하나, 12100에 얼굴 없는 프레임,
    # 다음 프레임은 15150. 15250부터 앞을 보다가 21000의 프레임 하나에서 고개를 돌리고, 다음 프레임은 24200.
    glance_times = list(range(0, 12200, 200)) + [12100, 15150, 15250, 15350, 15450, 15550, 15650]
    glance_times += list(range(15800, 21200, 200)) + [24200]
    glance_times.sort()
    glance_plan = [(0, _OPEN), (1000, away), (2600, _OPEN), (2800, away), (12000, _OPEN), (12100, no_face)]
    glance_plan += [(15250, _OPEN), (21000, away)]

    def check_glance(computed: Mapping[str, Any]) -> None:
        gaze, vats, gaze_stop, credited = computed[SECTION_EVENTS]
        _require(gaze["t_start_ms"] == first + 1000 and gaze["duration_ms"] == 3000, f"gaze onset is {gaze}")
        _require(vats["t_start_ms"] == first + 1000 and vats["duration_ms"] == 10000, f"vats onset is {vats}")
        _require(gaze_stop["duration_ms"] == 14550 and gaze_stop["confidence"] == 11400 / 14550, f"end is {gaze_stop}")
        _require(credited["t_start_ms"] == first + 21000 and credited["duration_ms"] == 3200, f"onset is {credited}")
        _require(credited["confidence"] == 200 / 3200, "one pose-valid frame of 200 ms in 3200 ms")

    # 첫 t_ms가 큰 값인 세션의 규칙과 경고. 보정부터 한다(6 fps). 59700~61200의 프레임이 없어서 기준선은 61333의
    # 프레임에서 완성된다(T_b = 60000). 보정 중의 1.5초 감김은 provisional microsleep, 완성 뒤의 고개 돌림은
    # gaze_away_long이다.
    large_segments = [
        Closure(hold_ms=1500, depth=_LOW_FPS_DEPTH, start_ms=10000),
        HeadTurn(4000, yaw=35.0, start_ms=63000),
    ]
    large_frames = _render(
        spec, large_segments, 68000, fps=6.0, gaps=((59700, 61200),), first_t_ms=LARGE_FIRST_T_MS
    )

    def check_large(computed: Mapping[str, Any]) -> None:
        origin = LARGE_FIRST_T_MS
        events = computed[SECTION_EVENTS]
        _require(not any(event["type"] == "unable_to_analyze" for event in events), "no unable_to_analyze")
        calib_on, micro_on, micro_end, calib_end, gaze, gaze_stop = events
        _require(calib_on["t_ms"] == origin and calib_on["t_start_ms"] == origin, f"calibrating onset is {calib_on}")
        _require(micro_on["provisional"] and micro_end["provisional"], "microsleep during calibration is provisional")
        _require(micro_on["t_start_ms"] > 2**31, "event times must not fit in 32 bits")
        _require(calib_end["t_ms"] == origin + 61333 and calib_end["duration_ms"] == 61333, f"end is {calib_end}")
        _require(abs(calib_end["confidence"] - 59867 / 61333) < 1e-12, "face time 59867 ms of 61333")
        _require(gaze["t_ms"] == origin + 66333 and gaze["duration_ms"] == 3000, f"gaze onset is {gaze}")
        _require(not gaze["provisional"] and gaze_stop["t_ms"] == origin + 67167, f"gaze end is {gaze_stop}")
        baseline_out = computed[SECTION_BASELINE]
        _require(baseline_out["t_complete_ms"] == 60000 and baseline_out["t_ms"] == origin + 61333, "baseline")
        steps = computed[SECTION_STEPS]
        _require([step["k"] for step in steps] == list(range(61, 68)), "steps 61..67")
        _require(steps[0]["t_ms"] == origin + 61333 and steps[0]["t_end_ms"] == origin + 61000, "step 61 is late")
        _require(steps[0]["values"][F_FACE_MISSING_RATIO_60] == 1.0, "step 61 holds no frame")

    # ---- 3차 검증과 그 뒤의 정비에서 나온 자리: 벡터 81개를 모두 통과하던 한 줄짜리 이식 오류 ----
    # 30초 창의 가장자리. 200 ms 간격. 영역 밖 구간 [1000, 2000)과 [21000, ...). 29000의 프레임 다음 프레임이 3000 ms
    # 뒤(32000)에 온다: 끊긴 시간은 29000 + 200 + 3000 전까지 영역 밖으로 치므로 둘째 구간이 [21000, 32000)이 되고,
    # 30초 창의 시작(2000)은 첫 구간의 끝과 정확히 같다. 합은 9000(29000의 프레임)에서 11000으로 건너뛴다.
    window_edge_times = list(range(0, 29200, 200)) + list(range(32000, 33200, 200))
    window_edge_plan = [(0, _OPEN), (1000, away), (2000, _OPEN), (21000, away)]

    def check_window_edge(computed: Mapping[str, Any]) -> None:
        gaze, vats = computed[SECTION_EVENTS]
        _require(gaze["t_start_ms"] == first + 21000 and gaze["duration_ms"] == 3000, f"gaze onset is {gaze}")
        _require(vats["t_start_ms"] == first + 21000 and vats["duration_ms"] == 11000, f"vats onset is {vats}")
        _require(vats["confidence"] == 8200 / 11000, "pose-valid time: 41 frames of 200 ms in 11000 ms")
        gaps = [later - earlier for earlier, later in zip(window_edge_times, window_edge_times[1:])]
        _require(sorted(set(gaps)) == [200, 3000], "one frame gap of 3000 ms")
        _require(32000 - rules.vats.window_ms == 2000, "the first away interval must end exactly at the window start")

    # vats 시작의 confidence가 1보다 작은 경우. 200 ms 간격. 400과 600의 프레임에 얼굴이 없고(t_start_ms보다 앞에서
    # 끝나는 pose_valid 구간 [0, 400)을 만든다), 1000부터 고개를 돌리고, 5000~6800의 프레임 열에 얼굴이 없다(3초 안이라
    # 영역 밖으로 친다). 7000부터 다시 고개를 돌린 채 보이고 12000부터 앞을 본다.
    lost_plan = [(0, _OPEN), (400, no_face), (800, _OPEN), (1000, away), (5000, no_face), (7000, away)]
    lost_plan += [(12000, _OPEN)]

    def check_face_lost_inside(computed: Mapping[str, Any]) -> None:
        gaze, vats, gaze_stop = computed[SECTION_EVENTS]
        _require(gaze["t_start_ms"] == first + 1000 and gaze["confidence"] == 1.0, f"gaze onset is {gaze}")
        _require(vats["t_start_ms"] == first + 1000 and vats["duration_ms"] == 10000, f"vats onset is {vats}")
        _require(vats["confidence"] == 8000 / 10000, "pose-valid time: 4000 + 4000 ms of 10000")
        _require(gaze_stop["duration_ms"] == 11400 and gaze_stop["confidence"] == 9400 / 11400, f"end is {gaze_stop}")

    # 한 프레임짜리 곁눈질. 200 ms 간격. 1000부터 32200까지 400 ms마다 한 프레임만 고개를 돌린다(79번, 31.4초).
    # 영역 밖 후보는 250 ms를 채우지 못하고 다음 프레임에서 버려진다. 영역 안 상태에서 버려진 후보는 영역 밖 시간이 아니다.
    one_frame_plan: list[tuple[int, _Shot]] = [(0, _OPEN)]
    for start in range(1000, 32400, 400):
        one_frame_plan += [(start, away), (start + 200, _OPEN)]

    def check_one_frame(computed: Mapping[str, Any]) -> None:
        turned = len(one_frame_plan) // 2
        _require(turned == 79 and 200 * turned > rules.vats.cumulative_ms, "the glances add up to more than 10 s")
        _require(400 * (turned - 1) + 200 > rules.vats.window_ms, "and they go on for more than the 30 s window")
        _require(200 < rules.forward_zone.debounce_ms, "one frame of 200 ms must stay below the debounce time")

    # microsleep의 confidence가 1보다 작은 경우. 100 ms 간격. 1000부터 감고, 1300~1500의 프레임 셋에 얼굴이 없다
    # (끊긴 시간 400 ms는 hold_ms 500 이하라 같은 감김이고, 앞뒤 표본이 모두 진입 임계 이상이라 길이에 들어간다).
    closure_lost_plan = [(0, _OPEN), (1000, _SHUT), (1300, no_face), (1600, _SHUT), (2600, _OPEN)]

    def check_closure_lost(computed: Mapping[str, Any]) -> None:
        onset, end = computed[SECTION_EVENTS]
        _require(onset["t_start_ms"] == first + 978 and onset["duration_ms"] == 2000 - 978, f"onset is {onset}")
        _require(onset["confidence"] == 700 / 1000, "closure-valid time: 300 + 400 ms of 1000")
        _require(end["duration_ms"] == 2600 - 978 and end["confidence"] == 1300 / 1600, f"end is {end}")

    # 미뤄 둔 microsleep 경고는 다른 종류의 onset으로도 지워지지 않는다. 200 ms 간격, 속도 10 km/h.
    onset_plan = [(0, _OPEN), (1000, away), (2800, away_shut), (5000, _OPEN)]

    def check_survives_onset(computed: Mapping[str, Any]) -> None:
        microsleep, gaze = computed[SECTION_EVENTS][:2]
        first_alert, second_alert = computed[SECTION_ALERTS]
        _require(microsleep["t_start_ms"] == first + 2756 and microsleep["duration_ms"] == 3800 - 2756, "microsleep")
        _require(first_alert["type"] == EVENT_MICROSLEEP and first_alert["duration_ms"] == 4400 - 2756, "first alert")
        _require(second_alert["type"] == EVENT_GAZE_AWAY_LONG and second_alert["duration_ms"] == 3400, "second alert")
        _require(first_alert["t_start_ms"] == microsleep["t_start_ms"], "a copy of the microsleep onset")
        _require(second_alert["t_start_ms"] == gaze["t_start_ms"] == first + 1000, "a copy of the gaze onset")

    # 첫 t_ms가 작은 세션(5000). 200 ms 간격. 세션 12000 전의 프레임에는 얼굴이 없다.
    small_times = list(range(0, 17400, 200))
    small_frames = _hand_frames(spec, small_times, [(0, no_face), (12000, _OPEN)], first_t_ms=SMALL_FIRST_T_MS)

    def check_small_t0(computed: Mapping[str, Any]) -> None:
        origin = SMALL_FIRST_T_MS
        _require(0 < origin < rules.unable_to_analyze.window_ms, "the first t_ms must be below the window length")
        got = [(item["phase"], item["t_ms"] - origin) for item in computed[SECTION_EVENTS]]
        _require(got == [(_ONSET, 10000), ("end", 17000)], f"unable_to_analyze events are {got}")
        onset, end = computed[SECTION_EVENTS]
        _require(onset["type"] == end["type"] == "unable_to_analyze", f"events are {onset}, {end}")
        _require(onset["t_start_ms"] == origin + 10000 and onset["duration_ms"] == 0, f"onset is {onset}")
        _require(end["duration_ms"] == 7000 and end["confidence"] == 5000 / 7000, f"end is {end}")
        _require(_kinds(computed[SECTION_ALERTS]) == (_UNABLE_ON,), "one alert, at the onset")

    # 스텝 경계에서 벗어난 프레임이 내는 drowsy. rules_drowsy_perclos_exact와 같은 감김을 3 fps로 72초 동안 두고, 첫 프레임을
    # 뺀 모든 프레임을 50 ms 늦춘다(세션 0, 383, 717, 1050, ...). 스텝 k는 세션 1000k + 50의 프레임이 발행한다.
    off_times = [0] + [t + 50 for t in _three_fps(0, 72) if t > 0]

    def check_off_boundary(computed: Mapping[str, Any]) -> None:
        steps = {step["k"]: step for step in computed[SECTION_STEPS]}
        _require(sorted(steps) == list(range(1, 72)), f"steps are {sorted(steps)}")
        for k in range(60, 72):
            _require(steps[k]["values"][F_PERCLOS_60] == rules.drowsiness.perclos.onset_ratio, f"step {k}: perclos_60")
            _require(steps[k]["t_ms"] == first + 1000 * k + 50, f"step {k} must be emitted 50 ms after its end")
            _require(steps[k]["t_end_ms"] == first + 1000 * k, f"step {k} ends at {steps[k]['t_end_ms']}")
        (onset,) = computed[SECTION_EVENTS]
        (alert,) = computed[SECTION_ALERTS]
        _require(onset["t_ms"] == first + 69050 and onset["t_start_ms"] == first + 60000, f"onset is {onset}")
        _require(onset["duration_ms"] == 9000, "the emit time of a drowsy event is the step end, not the frame")
        _require(alert == onset, "an immediate alert is the onset event itself, not a copy re-timed to the frame")
        _require(rules.events.drowsy.min_speed_kmh <= 60.0, "60 km/h must not block the drowsy alert")

    return [
        _Case(
            name="rules_closure_dwell_below_1s",
            bundle=BUNDLE_RULES,
            description=(
                "감김의 길이는 closure가 규칙용 진입 임계(0.7) 이상이던 시간의 합이다(2026-11-13 결정). 10 fps. 0.9로 "
                "300 ms, 0.6(이탈 0.5와 진입 0.7 사이)으로 5초, 다시 0.9로 300 ms, 그리고 눈을 뜬다. 한 감김이고 "
                "길이는 두 토막의 합 778 ms라 이벤트가 없다. 머문 5초를 길이에 넣으면 둘째 감김의 첫 프레임에서 "
                "microsleep과 sleep이 함께 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 7400, 100), [*dwell_plan, (6800, _OPEN)]),
            baseline=baseline,
            events=(),
            alerts=(),
        ),
        _Case(
            name="rules_closure_dwell_reaches_1s",
            bundle=BUNDLE_RULES,
            description=(
                "rules_closure_dwell_below_1s와 같은 앞부분에 둘째 감김이 더 길다. 첫 토막이 389 ms(978~1367), 둘째 "
                "토막이 6333에서 시작하므로 누적 감김 시간은 세션 6944 ms의 프레임에서 정확히 1000 ms가 된다. 그 "
                "프레임에서 microsleep onset이 난다('초과'로 구현하면 다음 프레임에서 난다). t_start_ms는 감김의 "
                "시작(978)이고 duration_ms는 발행 시각 − t_start_ms = 5966이다: 누적 감김 시간(1000)이 아니라 머문 "
                "5초를 포함한 값이다. end는 눈을 뜬 프레임(7300)에서 난다. sleep은 없다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, sorted({*range(0, 7600, 100), 6944}), [*dwell_plan, (7300, _OPEN)]),
            baseline=baseline,
            check=both(expect([(*ms_on, 6944), (*ms_end, 7300)], [(*ms_on, 6944)]), check_reaches),
        ),
        _Case(
            name="rules_closure_three_spans",
            bundle=BUNDLE_RULES,
            description=(
                "토막이 셋인 감김(10 fps). 0.9로 500 ms, 0.6으로 300 ms, 0.9로 300 ms, 0.6으로 300 ms, 0.9로 2.3초. "
                "0.7 아래로 내려가 있던 시간은 두 번 모두 266 ms다. 끝난 두 토막이 489 + 334 = 823 ms, 셋째 토막이 "
                "2333에서 시작하므로 누적 감김 시간은 2510의 프레임에서 정확히 1000 ms(microsleep onset), 4510의 "
                "프레임에서 정확히 3000 ms(sleep onset)다. 두 이벤트의 t_start_ms는 감김의 시작(978)으로 같고 "
                "duration_ms는 1532와 3532다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(
                spec,
                sorted({*range(0, 5000, 100), 2510, 4510}),
                [(0, _OPEN), (1000, _SHUT), (1500, half_open), (1800, _SHUT), (2100, half_open), (2400, _SHUT)]
                + [(4700, _OPEN)],
            ),
            baseline=baseline,
            check=both(
                expect([(*ms_on, 2510), (*sleep_on, 4510), (*sleep_end, 4700)], [(*ms_on, 2510), (*sleep_on, 4510)]),
                check_three_spans,
            ),
        ),
        _Case(
            name="rules_alert_deferred_cooldown",
            bundle=BUNDLE_RULES,
            description=(
                "재경고 대기에 걸린 onset은 미뤄 두었다가 대기가 풀렸을 때 상태가 이어지고 있으면 한 번 경고한다"
                "(2026-11-13 결정). 10 fps, 감김 셋. 첫 microsleep onset(2000)은 바로 경고한다. 둘째 onset(4000)은 "
                "대기 5초에 걸리고, 대기가 풀리기 전에 감김이 끝나므로(4300) 경고가 없다. 셋째 onset(6000)도 걸리지만 "
                "감김이 이어져서, 첫 경고에서 정확히 5000 ms 뒤의 프레임(7000)에 경고가 나간다. 그 경고는 셋째 onset의 "
                "사본이고 duration_ms만 그 시각 기준(2022)이다. 대기는 '미만'으로 비교한다(5000 ms가 지난 프레임은 "
                "걸리지 않는다). 미뤘다 낸 경고도 대기를 새로 시작한다: 넷째 onset(9000)은 그 경고(7000)의 대기에 "
                "걸리고 감김이 먼저 끝나 경고가 없다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 9600, 100), cooldown_plan),
            baseline=baseline,
            check=both(
                expect(
                    [(*ms_on, 2000), (*ms_end, 2200), (*ms_on, 4000), (*ms_end, 4300), (*ms_on, 6000), (*ms_end, 7600)]
                    + [(*ms_on, 9000), (*ms_end, 9300)],
                    [(*ms_on, 2000), (*ms_on, 7000)],
                ),
                check_deferred_cooldown,
            ),
        ),
        _Case(
            name="rules_alert_blocked_microsleep_sleep",
            bundle=BUNDLE_RULES,
            description=(
                "속도에 막힌 microsleep이 sleep으로 넘어가면 microsleep 경고는 뒤늦게 나가지 않는다(2026-11-13 결정). "
                "10 fps, 속도 10 km/h. 첫 감김(3.5초)은 microsleep과 sleep이 모두 막힌 채 끝나고, 그 뒤 속도가 25가 "
                "되어도 경고가 없다. 둘째 감김(4.5초)은 10 km/h에서 microsleep(8000)과 sleep(10000)이 시작되고, 감김이 "
                "이어지는 중에 속도가 정확히 20.0 km/h가 된 프레임(10500)에서 sleep 경고 하나만 나간다(속도는 "
                "'미만'일 때만 막는다). microsleep은 sleep onset에서 지워졌으므로 경고가 없다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 12000, 100), blocked_plan),
            baseline=baseline,
            speed=blocked_speed,
            check=both(
                expect(
                    [(*ms_on, 2000), (*sleep_on, 4000), (*sleep_end, 4500)]
                    + [(*ms_on, 8000), (*sleep_on, 10000), (*sleep_end, 11500)],
                    [(*sleep_on, 10500)],
                ),
                check_blocked,
            ),
        ),
        _Case(
            name="rules_drowsy_perclos_exact",
            bundle=BUNDLE_RULES,
            description=(
                "PERCLOS 판정의 등호 자리와, 속도에 막혔다가 나가는 drowsy 경고(2026-11-13 결정). 3 fps, 112초. 60초 "
                "구간의 유효 시간은 36000 ms다. 스텝 60~71의 perclos_60은 5400 / 36000으로 시작 비율 0.15와 정확히 "
                "같다: '이상'이므로 스텝 60부터 열 번째인 스텝 69에 onset이 난다. 스텝 78~101은 3600 / 36000으로 종료 "
                "비율 0.10과 정확히 같다: '미만'이 아니므로 종료 조건이 아니고, 0.10 아래로 내려간 스텝 102부터 열 "
                "번째인 스텝 111에 end가 난다. 속도는 45 km/h라 onset 때 경고가 막히고, 정확히 50.0 km/h가 된 "
                "프레임(75000)에서 한 번 나간다(onset의 사본, duration_ms 15000)."
            ),
            sections=(SECTION_STEPS, *_RULE_SECTIONS),
            frames=_hand_frames(spec, exact_times, exact_plan),
            baseline=baseline,
            speed=((first, 45.0), (first + 75000, 50.0)),
            check=both(
                expect([(*drowsy_on, 69000), (*drowsy_end, 111000)], [(*drowsy_on, 75000)]), check_perclos_exact
            ),
        ),
        _Case(
            name="rules_drowsy_model_stall",
            bundle=BUNDLE_RULES,
            description=(
                "밀린 스텝마다 모델을 부른다(2026-11-13 결정). drowsiness.source = model, 스텁 확률, 눈 뜬 프레임 "
                "204초(176초까지 3 fps, 그 뒤 5 fps). 프레임이 2600 ms 끊긴 곳이 둘이다: 184400 다음 프레임이 "
                "187000(스텝 185, 186, 187을 한 프레임이 발행), 197400 다음 프레임이 200000(스텝 198, 199, 200). "
                "모델은 스텝 179부터 스텝마다 그 "
                "스텝까지의 창으로 불리고(끊김 뒤의 프레임에서 세 번), expected.step_probs에 스텝마다 하나씩 적힌다. "
                "p_drowsy는 호출 순서대로 0.5 셋, 0.6 열(스텝 182~191. 시작 임계와 같은 값이고 '이상'이라 시작 "
                "조건이다), 0.5 셋, 0.3 열(스텝 195~204)이다. 끊김이 "
                "연속 수를 끊지 않으므로 스텝 191에 onset, 스텝 204에 end가 난다. expected.probs는 프레임마다 마지막 "
                "스텝의 확률만 담는다. 속도 60 km/h."
            ),
            sections=(SECTION_STEP_PROBS, SECTION_PROBS, *_RULE_SECTIONS),
            frames=_hand_frames(spec, stall_times, [(0, _OPEN)]),
            baseline=baseline,
            overrides={"drowsiness.source": "model"},
            speed=((first, 60.0),),
            model={"kind": MODEL_KIND_STUB, "probs": [list(probs) for probs in stall_probs]},
            check=both(expect([(*drowsy_on, 191000), (*drowsy_end, 204000)], [(*drowsy_on, 191000)]), check_stall),
        ),
        _Case(
            name="rules_gap_over_10s",
            bundle=BUNDLE_RULES,
            description=(
                "10초 넘는 끊김 뒤에는 규칙 엔진이 세션을 시작할 때처럼 다시 시작한다(2026-11-13 결정). 5 fps. 고개를 "
                "35도 돌린 채(1000~) gaze_away_long(4000)과 vats(11000, 누적이 정확히 10000 ms)가 시작되고, 눈을 "
                "감아(11400~) microsleep과 sleep이 시작된 뒤, 14600의 프레임 다음 프레임이 10001 ms 뒤(24601)에 온다. "
                "그 프레임에서 end가 gaze_away_long, sleep, vats 순으로 나고(발행 시각은 그 프레임), 고개를 돌리고 "
                "눈을 감은 채이므로 그 프레임부터 새로 센다: microsleep이 정확히 1000 ms 뒤, gaze_away_long과 "
                "sleep이 정확히 3000 ms 뒤의 같은 프레임에 시작한다. 그 프레임의 경고는 우선순위 순(sleep, "
                "gaze_away_long)이다. 끊김 뒤에 unable_to_analyze는 없다. 속도는 15 km/h로 시작해 4200에 정확히 "
                "20.0 km/h가 된다: 첫 gaze_away_long onset(4000)의 경고는 막혔다가 4200에 한 번 나가고"
                "(duration_ms 3200), 상태가 그 경고의 재경고 대기(10초)를 넘겨 이어져도 다시 나가지 않는다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(
                spec,
                gap_lead + [24601 + 200 * index for index in range(21)],
                [*gap_plan, (28001, _OPEN)],
            ),
            baseline=baseline,
            speed=((first, 15.0), (first + 4200, 20.0)),
            check=both(
                expect(
                    [(*gaze_on, 4000), (*vats_on, 11000), (*ms_on, 12400), (*sleep_on, 14400)]
                    + [(*gaze_end, 24601), (*sleep_end, 24601), (*vats_end, 24601)]
                    + [(*ms_on, 25601), (*gaze_on, 27601), (*sleep_on, 27601), (*sleep_end, 28001), (*gaze_end, 28401)],
                    [(*gaze_on, 4200), (*vats_on, 11000), (*ms_on, 12400), (*sleep_on, 14400)]
                    + [(*ms_on, 25601), (*sleep_on, 27601), (*gaze_on, 27601)],
                ),
                check_gap_over,
            ),
        ),
        _Case(
            name="rules_gap_exactly_10s",
            bundle=BUNDLE_RULES,
            description=(
                "rules_gap_over_10s와 같은 앞부분에서 끊김이 정확히 10000 ms다(초과가 아니므로 초기화하지 않는다). "
                "끊긴 시간은 얼굴 없는 시간이라 그 프레임(24600)에서 unable_to_analyze가 시작되고, 감김은 500 ms 넘게 "
                "끊겨 잘리므로 sleep end가 난다. gaze_away_long과 vats는 끝나지 않는다. 끊김 뒤 2초는 얼굴이 없고 "
                "그다음부터 정면을 본다: 얼굴 시간이 5000 ms가 된 프레임(31600)에서 unable_to_analyze end, 영역 안이 "
                "확정된 프레임(32000)에서 gaze_away_long end, 영역 밖 누적이 10초 아래로 내려간 프레임(38000)에서 "
                "vats end가 난다. vats는 37초 이어졌고 end의 confidence는 구간 전체의 pose_valid 시간 비율 "
                "25200 / 37000이다(끊긴 시간과 얼굴 없는 시간이 빠진다. 최근 30초로 자르지 않는다). 얼굴이 사라진 뒤 "
                "영역 밖으로 치는 시간은 3초까지다(14800~17800)."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(
                spec,
                gap_lead + list(range(24600, 38400, 200)),
                [*gap_plan, (24600, no_face), (26600, _OPEN)],
            ),
            baseline=baseline,
            check=both(
                expect(
                    [(*gaze_on, 4000), (*vats_on, 11000), (*ms_on, 12400), (*sleep_on, 14400)]
                    + [(*unable_on, 24600), (*sleep_end, 24600), (*unable_end, 31600), (*gaze_end, 32000)]
                    + [(*vats_end, 38000)],
                    [(*gaze_on, 4000), (*vats_on, 11000), (*ms_on, 12400), (*sleep_on, 14400), (*unable_on, 24600)],
                ),
                check_gap_exact,
            ),
        ),
        _Case(
            name="rules_gap_over_10s_drowsy",
            bundle=BUNDLE_RULES,
            description=(
                "10초 넘는 끊김은 진행 중인 drowsy도 끝낸다(2026-11-13 결정). 3 fps, 속도 45 km/h. perclos_60이 "
                "0.15라 스텝 69에 drowsy onset이 나지만 경고는 속도(50 km/h 미만)에 막혀 미뤄진다. 고개를 돌려 같은 "
                "프레임에서 gaze_away_long도 시작한다(스텝 이벤트가 먼저). 70000의 프레임 다음 프레임이 10001 ms "
                "뒤(80001)에 온다. 그 프레임에서 drowsy end(발행 시각은 그 프레임의 t_ms, confidence 0.0, onset과 "
                "같은 t_start_ms와 source)와 gaze_away_long end가 이 순서로 난다. 끊김 뒤 속도는 80 km/h지만 "
                "미뤄 둔 drowsy 경고는 상태가 끝났으므로 나가지 않는다. 두 졸음 판정기도 처음 상태로 돌아가므로 "
                "끊김 뒤의 스텝(81, 82, full_lookback false)에서 drowsy가 다시 시작되지 않는다."
            ),
            sections=(SECTION_STEPS, *_RULE_SECTIONS),
            frames=_hand_frames(spec, drowsy_gap_times, drowsy_gap_plan),
            baseline=baseline,
            speed=((first, 45.0), (first + 80001, 80.0)),
            check=both(
                expect(
                    [(*drowsy_on, 69000), (*gaze_on, 69000), (*drowsy_end, 80001), (*gaze_end, 80001)],
                    [(*gaze_on, 69000)],
                ),
                check_drowsy_gap,
            ),
        ),
        _Case(
            name="rules_gap_unable_boundary",
            bundle=BUNDLE_RULES,
            description=(
                "10초 이하의 끊김은 얼굴 없는 시간으로 센다(초기화하지 않는다). 5 fps, 얼굴은 계속 보인다. 끊김이 "
                "5200 ms이면 다음 프레임에서 최근 10초의 얼굴 시간이 4800 + 200 = 5000 ms(비율 0.5)라 "
                "unable_to_analyze가 시작되지 않는다. 5201 ms이면 4999 ms(비율 0.4999)라 그 프레임(30601)에서 "
                "시작되고 경고가 나간다. 얼굴 시간이 다시 5000 ms가 된 프레임(35601)에서 끝난다. 시작은 '미만', "
                "종료는 '이상'이다. 둘째 끊김은 고개를 돌린 지 0.6초 만에(24800~25400) 시작된다: 끊긴 동안 3초까지 "
                "영역 밖으로 쳐서 연속 영역 밖 시간이 3초를 넘지만, 끊김 뒤의 프레임은 분석 불가 중이라 "
                "gaze_away_long이 나지 않는다(운전자는 정면을 보고 있다)."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, boundary_times, [(0, _OPEN), (24800, away), (30601, _OPEN)]),
            baseline=baseline,
            check=both(expect([(*unable_on, 30601), (*unable_end, 35601)], [(*unable_on, 30601)]), check_gap_boundary),
        ),
        _Case(
            name="rules_unable_sparse_frames",
            bundle=BUNDLE_RULES,
            description=(
                "드문 프레임과 분석 불가. 얼굴이 계속 보이지만 프레임 간격이 500 ms다. 한 프레임은 200 ms까지만 "
                "인정되므로 최근 10초의 얼굴 시간은 4000 ms(비율 0.4)다. 세션의 처음 10초는 판정하지 않고, 정확히 "
                "10000 ms의 프레임에서 unable_to_analyze가 시작된다. 12000의 프레임 다음 프레임이 10500 ms 뒤"
                "(22500)에 온다: 10초를 넘는 끊김이라 그 프레임에서 end가 나고(confidence 1000 / 12500) 창과 "
                "'처음 10초'가 다시 시작한다(2026-11-13 결정). 그래서 그 프레임에서 정확히 10000 ms 뒤(32500)에 "
                "다시 시작된다. 이 onset은 첫 경고의 재경고 대기(30초)에 걸려 미뤄지고, 상태가 이어지므로 첫 "
                "경고에서 정확히 30000 ms 뒤의 프레임(40000)에 나간다(onset의 사본, duration_ms 7500)."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, sparse_times, [(0, _OPEN)]),
            baseline=baseline,
            check=both(
                expect(
                    [(*unable_on, 10000), (*unable_end, 22500), (*unable_on, 32500)],
                    [(*unable_on, 10000), (*unable_on, 40000)],
                ),
                check_sparse,
            ),
        ),
        _Case(
            name="rules_gaze_credit_window_missed",
            bundle=BUNDLE_RULES,
            description=(
                "얼굴 소실 뒤의 인정 구간에 프레임이 오지 않은 경우(10 fps). 고개를 돌린 프레임 셋(1000, 1100, 1200) "
                "뒤 얼굴이 사라진다. 얼굴이 사라진 시각은 1300이고 4300까지 영역 밖으로 친다. 연속 영역 밖 시간이 "
                "3000 ms가 되는 4000~4200의 프레임이 빠졌고, 다음 프레임(4300, 얼굴 없음)은 표지가 UNKNOWN이다. "
                "gaze_away_long onset은 그 프레임에서 난다: 임계 시간은 이미 찼고 UNKNOWN 동안에는 시간을 세지 "
                "않으므로 일찍 나는 것이 아니다. duration_ms 3300, confidence 300 / 3300. 얼굴이 정면으로 "
                "돌아와 확정된 프레임(5400)에서 end가 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=missed_frames,
            baseline=baseline,
            check=both(expect([(*gaze_on, 4300), (*gaze_end, 5400)], [(*gaze_on, 4300)]), check_missed),
        ),
        _Case(
            name="rules_gaze_credit_boundary",
            bundle=BUNDLE_RULES,
            description=(
                "인정 구간의 끝에 걸린 프레임(10 fps). 마지막 유효 관측에서 200 ms + 3000 ms가 되는 시각 '전'까지만 "
                "표지를 이어받는다. (1) 0.4초 이탈 뒤 앞을 본 프레임 하나(1400), 다음 프레임은 정확히 4600에 얼굴 "
                "없이 온다: 표지는 UNKNOWN이고 영역 안 후보가 아직 걸려 있으므로 gaze_away_long이 나지 않는다"
                "(후보는 다음 프레임에서 버려진다). (2) 3.5초 이탈로 gaze_away_long이 진행 중일 때 앞을 본 프레임 "
                "하나(11500), 다음 프레임은 정확히 14700: 이어받지 않으므로 영역 안이 확정되지 않고, 얼굴이 "
                "돌아와 250 ms 이상 정면을 본 프레임(15200)에서 end가 난다. (3) 같은 모양에서 다음 프레임이 "
                "24650(끝나기 50 ms 전)에 오면 영역 안 표지를 이어받아 그 프레임에서 확정되고 end가 난다. "
                "둘째 onset(21000)은 첫 경고(11000)에서 정확히 10000 ms 뒤라 재경고 대기에 걸리지 않는다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, credit_times, credit_plan),
            baseline=baseline,
            check=expect(
                [(*gaze_on, 11000), (*gaze_end, 15200), (*gaze_on, 21000), (*gaze_end, 24650)],
                [(*gaze_on, 11000), (*gaze_on, 21000)],
            ),
        ),
        _Case(
            name="rules_gaze_debounce_250ms",
            bundle=BUNDLE_RULES,
            description=(
                "디바운스 250 ms의 경계. 고개를 돌려 gaze_away_long이 시작된 뒤(3000), 50 ms 간격의 프레임에서 "
                "200 ms 동안만 정면을 보면(3500~3650의 네 프레임) 영역 안이 확정되지 않아 상태가 이어진다. 다시 "
                "정면을 본 첫 프레임(4500)에서 정확히 250 ms 뒤의 프레임(4750)에 확정되어 end가 난다('이상'. "
                "'초과'로 구현하면 4800에 난다)."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, debounce_times, debounce_plan),
            baseline=baseline,
            check=both(expect([(*gaze_on, 3000), (*gaze_end, 4750)], [(*gaze_on, 3000)]), check_debounce),
        ),
        _Case(
            name="rules_forward_zone_edges",
            bundle=BUNDLE_RULES,
            description=(
                "전방 주시 영역의 경계 여섯(5 fps): 상대 yaw +23과 −23도, pitch −25도(위)와 +18도(아래), 눈동자 "
                "합성값 gaze_h 0.25, gaze_v 0.35. 경계마다 바로 바깥 값(yaw ±23.01, pitch −25.01과 18.01, gaze_h "
                "−0.251, gaze_v 0.351)으로 3.2초 있다가 바로 안쪽 값(±22.99, −24.99와 17.99, −0.249, 0.349)으로 "
                "돌아온다. 바깥 값에서는 정확히 3000 ms에 gaze_away_long onset이 나고, 안쪽 값은 영역 안이라 "
                "400 ms 뒤 확정되어 end가 난다. source는 머리 조건이 head_pose, 눈동자가 eye다. 네 번째 이탈"
                "(눈동자) 도중에 최근 30초의 영역 밖 누적이 10초가 되어 vats가 시작되고 source는 eye다. 등호 "
                "자리(정확히 23.0도 등)는 벡터에 두지 않는다. gaze_away_long 경고는 재경고 대기 10초에 걸린 것이 "
                "빠진다(대기가 풀리기 전에 상태가 끝난다)."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 24400, 200), edge_plan),
            baseline=baseline,
            check=check_edges,
        ),
        _Case(
            name="rules_vats_reset_enabled",
            bundle=BUNDLE_RULES,
            description=(
                "vats.reset_enabled = true(기본은 false). 5 fps, 2.8초의 시선 이탈 다섯 번(각각 3초에 못 미쳐 "
                "gaze_away_long은 없다). 첫 이탈 뒤에는 정면을 본 시간이 정확히 2000 ms가 되는 프레임(5800)이 있어 "
                "거기서 영역 밖 누적을 지운다('이상'). 그 뒤의 정면은 1800 ms씩이라 지워지지 않는다. 그래서 vats "
                "onset은 다섯 번째 이탈 도중(21400)에 나고 t_start_ms는 둘째 이탈의 시작(6000)이다. 지우지 "
                "않으면 네 번째 이탈 도중에 나고 t_start_ms가 첫 이탈의 시작이 된다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 22000, 200), reset_plan),
            baseline=baseline,
            overrides={"vats.reset_enabled": True},
            check=both(expect([(*vats_on, 21400)], [(*vats_on, 21400)]), check_vats_reset),
        ),
        _Case(
            name="rules_alert_priority",
            bundle=BUNDLE_RULES,
            description=(
                "한 프레임에 경고가 둘(10 fps). 고개를 돌린 채(1000~) 눈을 감아(3000~) 같은 프레임(4000)에서 "
                "gaze_away_long과 microsleep이 시작한다. 상태 이벤트는 판정 순서(gaze_away_long, microsleep)로, "
                "경고는 alert_priority 순서(microsleep, gaze_away_long)로 나온다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(
                spec, range(0, 5000, 100), [(0, _OPEN), (1000, away), (3000, away_shut), (4400, _OPEN)]
            ),
            baseline=baseline,
            check=both(
                expect(
                    [(*gaze_on, 4000), (*ms_on, 4000), (*ms_end, 4400), (*gaze_end, 4700)],
                    [(*ms_on, 4000), (*gaze_on, 4000)],
                ),
                check_priority,
            ),
        ),
        _Case(
            name="rules_gap_over_10s_candidate",
            bundle=BUNDLE_RULES,
            description=(
                "10초 넘는 끊김은 걸려 있던 전방 주시 후보도 버린다(5 fps). 2000의 프레임 하나에서 고개를 돌린 것을 "
                "보고(영역 밖 후보, 아직 확정 전) 다음 프레임이 10001 ms 뒤(12001)에 온다. 끊김 뒤에도 고개를 돌리고 "
                "있다. 규칙 엔진이 다시 시작하므로 영역 밖은 12001의 프레임부터 새로 센다: 그 프레임에서 정확히 "
                "3000 ms 뒤(15001)에 gaze_away_long onset이 나고 t_start_ms는 12001이다. 후보를 버리지 않으면 "
                "12001에서 2000으로 소급해 확정되고 그 프레임에서 gaze_away_long(10001 ms)과 vats가 함께 난다. "
                "정면으로 돌아와 확정된 프레임(15801)에서 end가 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, candidate_times, [(0, _OPEN), (2000, away), (15401, _OPEN)]),
            baseline=baseline,
            check=both(expect([(*gaze_on, 15001), (*gaze_end, 15801)], [(*gaze_on, 15001)]), check_candidate_gap),
        ),
        _Case(
            name="rules_gaze_candidate_dropped",
            bundle=BUNDLE_RULES,
            description=(
                "영역 안 후보가 프레임 끊김으로 버려지면 그 후보 구간은 영역 밖 시간에 들어가지 않는다(10 fps). 2.0초 "
                "동안 고개를 돌리고(1000~2900) 앞을 본 프레임 하나(3000) 뒤 3.3초 동안 프레임이 없다가 고개를 돌린 "
                "채 돌아온다(6300~). 3000의 관측을 이어받는 구간은 6200에 끝났으므로 6300에서 후보를 버린다. 상태는 "
                "영역 밖 그대로이고 연속 영역 밖 시간은 2000 ms에서 이어진다(3000~6300의 3300 ms는 세지 않는다). "
                "그래서 gaze_away_long onset은 7300에 나고 t_start_ms는 1000, duration_ms는 6300이다. confidence는 "
                "그 구간의 pose_valid 시간 비율 3200 / 6300이다. 정면이 확정된 프레임(8300)에서 end가 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(
                spec, dropped_times, [(0, _OPEN), (1000, away), (3000, _OPEN), (6300, away), (8000, _OPEN)]
            ),
            baseline=baseline,
            check=both(expect([(*gaze_on, 7300), (*gaze_end, 8300)], [(*gaze_on, 7300)]), check_dropped),
        ),
        _Case(
            name="rules_forward_zone_eye_edges",
            bundle=BUNDLE_RULES,
            description=(
                "눈동자 보조 판정의 나머지 두 방향(5 fps). rules_forward_zone_edges는 gaze_h의 음수 쪽과 gaze_v의 "
                "양수 쪽만 지난다. 여기서는 gaze_h +0.251(바깥)과 +0.249(안쪽), gaze_v −0.351(바깥, 아래를 봄)과 "
                "−0.349(안쪽)다. 눈동자 합성값은 절댓값으로 비교하므로 두 이탈 모두 정확히 3000 ms에 gaze_away_long "
                "onset(source eye)이 나고 안쪽 값으로 돌아온 뒤 400 ms에 end가 난다. 둘째 onset의 경고는 재경고 "
                "대기(10초)에 걸린 채 상태가 끝나 나가지 않는다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 9000, 200), eye_edge_plan),
            baseline=baseline,
            check=both(
                expect(
                    [(*gaze_on, 4000), (*gaze_end, 4600), (*gaze_on, 7800), (*gaze_end, 8400)],
                    [(*gaze_on, 4000)],
                ),
                check_eye_edges,
            ),
        ),
        _Case(
            name="rules_gaze_unable_pauses_timer",
            bundle=BUNDLE_RULES,
            description=(
                "분석 불가 동안에는 연속 영역 밖 시간이 멈춘다(5 fps). 얼굴이 12000~15800에 없다가 고개를 돌린 채 "
                "세 프레임(16000~16400) 보이고 다시 사라진다. 영역 밖은 16000부터이고 얼굴이 사라진 뒤에도 "
                "이어받지만, 17800에서 최근 10초의 얼굴 시간이 4800 ms가 되어 unable_to_analyze가 시작된다. 그때까지의 "
                "연속 영역 밖 시간은 1800 ms다. 분석 불가 중인 프레임의 표지는 UNKNOWN이라 시간을 세지 않는다. "
                "얼굴은 40000에 고개를 돌린 채 돌아오고 45000에 분석 불가가 끝난다(confidence 5000 / 27200). 그 "
                "프레임부터 다시 세어 1800 + 1200 = 3000 ms가 되는 46200에 gaze_away_long onset이 난다(t_start_ms "
                "16000, duration_ms 30200). 멈춘 시간을 빼지 않으면 45000에서 바로 난다. vats는 45000부터의 영역 밖 "
                "구간이 10000 ms가 되는 55000에 시작하고 t_start_ms는 45000이다: 그때의 30초 창에 든 첫 영역 밖 "
                "구간의 시작이다(16000~17800의 구간은 창을 벗어났고, 분석 불가 동안의 프레임은 구간을 만들지 "
                "않는다). 정면이 확정된 56400에 gaze_away_long end가 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 57000, 200), pause_plan),
            baseline=baseline,
            check=both(
                expect(
                    [(*unable_on, 17800), (*unable_end, 45000), (*gaze_on, 46200), (*vats_on, 55000)]
                    + [(*gaze_end, 56400)],
                    [(*unable_on, 17800), (*gaze_on, 46200), (*vats_on, 55000)],
                ),
                check_pause,
            ),
        ),
        _Case(
            name="rules_drowsy_either",
            bundle=BUNDLE_RULES,
            description=(
                "drowsiness.source = either: 모델이 연 상태가 끝난 뒤 PERCLOS가 상태를 연다. 3 fps, 201초, 스텁 확률, "
                "속도 60 km/h. 모델은 스텝 179부터 불린다. p_drowsy 0.7이 열 번(스텝 179~188)이라 스텝 188에 "
                "onset(source model, t_start_ms는 스텝 179의 끝), 0.3이 열 번(189~198)이라 스텝 198에 end(같은 "
                "source와 t_start_ms, confidence는 그 스텝의 p_drowsy 0.3)가 난다. 그 뒤의 p_drowsy는 0.5다. 감긴 "
                "프레임 세 장이 든 초가 165, 168, ..., 189의 아홉이라 perclos_60은 스텝 190부터 5400 / 36000 = 0.15이고"
                "(스텝 189는 0.1333) 스텝 190~199의 열 번째인 스텝 199에 둘째 onset이 난다. 이 상태는 PERCLOS 쪽이 "
                "열었으므로 source는 perclos, t_start_ms는 스텝 190의 끝, confidence는 1 − face_missing_ratio_60 = "
                "0.6이다(앞 상태의 source model이 남으면 안 된다). 둘째 onset의 경고는 재경고 대기(300초)에 걸린다."
            ),
            sections=(SECTION_STEPS, SECTION_STEP_PROBS, SECTION_PROBS, *_RULE_SECTIONS),
            frames=_hand_frames(spec, _three_fps(0, 201), either_plan),
            baseline=baseline,
            overrides={"drowsiness.source": "either"},
            speed=((first, 60.0),),
            model={"kind": MODEL_KIND_STUB, "probs": [list(probs) for probs in either_probs]},
            check=both(
                expect(
                    [(*drowsy_on, 188000), (*drowsy_end, 198000), (*drowsy_on, 199000)],
                    [(*drowsy_on, 188000)],
                ),
                check_either,
            ),
        ),
        _Case(
            name="rules_gap_over_10s_drowsy_model",
            bundle=BUNDLE_RULES,
            description=(
                "10초 넘는 끊김은 모델 졸음 판정기도 처음 상태로 되돌린다: 재시작 금지 시간이 끊김을 넘어 이어지지 "
                "않는다. drowsiness.source = model, 스텁 확률, 3 fps, 속도 60 km/h. 끊김 뒤에 모델 창(120스텝)이 다시 "
                "차려면 179초가 걸려 기본 금지 시간(60초)으로는 드러나지 않으므로 이 벡터는 "
                "drowsiness.model.refractory_ms를 600000으로 덮어쓴다. 스텝 188에 onset, 스텝 198에 end가 난다(금지 "
                "시간은 그때부터 600초). 199000의 프레임 다음 프레임이 10001 ms 뒤(209001)에 온다. 끊김 때 진행 중인 "
                "상태가 없으므로 그 프레임의 이벤트는 없다. 누적이 209000에서 다시 시작해 full_lookback은 스텝 "
                "269부터, 모델 호출은 스텝 388부터다. p_drowsy 0.7이 열 번 이어진 스텝 397(프레임 397001)에 새 "
                "onset이 난다(t_start_ms는 스텝 388의 끝). 앞 상태가 끝난 지 199초라 금지 시간 안이지만 판정기가 "
                "처음 상태로 돌아갔으므로 막지 않는다. 초기화하지 않으면 이 onset이 나지 않는다. 경고는 첫 onset "
                "하나다(둘째는 재경고 대기 300초에 걸린다)."
            ),
            sections=(SECTION_STEP_PROBS, SECTION_PROBS, *_RULE_SECTIONS),
            frames=_hand_frames(spec, model_gap_times, [(0, _OPEN)]),
            baseline=baseline,
            overrides={"drowsiness.source": "model", "drowsiness.model.refractory_ms": model_gap_refractory_ms},
            speed=((first, 60.0),),
            model={"kind": MODEL_KIND_STUB, "probs": [list(probs) for probs in model_gap_probs]},
            check=both(
                expect(
                    [(*drowsy_on, 188000), (*drowsy_end, 198000), (*drowsy_on, 397001)],
                    [(*drowsy_on, 188000)],
                ),
                check_model_gap,
            ),
        ),
        _Case(
            name="rules_alert_pending_survives_end",
            bundle=BUNDLE_RULES,
            description=(
                "미뤄 둔 microsleep 경고는 다른 종류의 end로는 지워지지 않는다(5 fps). 속도 10 km/h. 고개를 돌려"
                "(1000~) gaze_away_long이 시작되고(4000) 그 상태에서 눈을 감아(5000~) microsleep이 시작된다(6000). 두 "
                "경고 모두 속도에 막혀 미뤄진다. 눈을 감은 채 정면으로 돌아와 gaze_away_long이 끝난다(6600): 미뤄 둔 "
                "gaze_away_long 경고는 지워지지만 microsleep 경고는 남는다. 속도가 25 km/h가 된 프레임(6800)에서 "
                "microsleep 경고 하나가 나간다(onset의 사본, duration_ms 1844). microsleep 경고를 지우는 것은 "
                "microsleep의 end와 sleep의 onset·end뿐이다. 눈을 뜬 프레임(7200)에서 microsleep end가 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 7800, 200), survive_plan),
            baseline=baseline,
            speed=((first, 10.0), (first + 6800, 25.0)),
            check=both(
                expect(
                    [(*gaze_on, 4000), (*ms_on, 6000), (*gaze_end, 6600), (*ms_end, 7200)],
                    [(*ms_on, 6800)],
                ),
                check_survives,
            ),
        ),
        _Case(
            name="rules_closure_after_dropped_run",
            bundle=BUNDLE_RULES,
            description=(
                "버려진 짧은 감김 뒤에 같은 프레임에서 시작한 새 감김(10 fps). 1000의 프레임 하나만 눈을 감았고"
                "(규칙용 검출기의 감김은 978에서 시작, 그 프레임까지 22 ms) 다음 프레임은 700 ms 뒤(1700)에 감긴 채 "
                "온다. 끊긴 시간이 hold_ms(500)를 넘었으므로 앞 감김은 잘리는데, 길이가 min_duration_ms(50) 미만이라 "
                "구간 없이 버려진다. 같은 프레임에서 새 감김이 시작한다(직전 표본과 700 ms 떨어져 보간하지 않으므로 "
                "시작은 1700). 규칙 엔진은 진행 중인 감김의 시작이 바뀐 것으로 새 감김을 알아본다: microsleep onset은 "
                "1700에서 정확히 1000 ms 뒤(2700)에 나고 t_start_ms는 1700, confidence는 1.0이다. 앞 감김의 "
                "기록(시작 978, 처음 본 프레임 1000)이 남으면 t_start_ms와 duration_ms, confidence가 달라진다. "
                "눈을 뜬 프레임(3000)에서 end가 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, restart_times, [(0, _OPEN), (1000, _SHUT), (3000, _OPEN)]),
            baseline=baseline,
            check=both(expect([(*ms_on, 2700), (*ms_end, 3000)], [(*ms_on, 2700)]), check_restart),
        ),
        _Case(
            name="rules_gaze_glance_and_credit_times",
            bundle=BUNDLE_RULES,
            description=(
                "잠깐 앞을 본 구간, 얼굴이 사라진 시각, 인정 구간의 끝에 온 관측(5 fps. 12000 다음 프레임만 100 ms 뒤에 "
                "온다). (1) 1000부터 "
                "고개를 돌리고 있다가 2600의 프레임 하나에서만 앞을 본다. 영역 안 후보는 250 ms를 채우지 못하고 "
                "2800의 영역 밖 표지로 버려지며, 그 200 ms도 영역 밖 시간으로 친다(연속이 끊기지 않는다). 그래서 "
                "gaze_away_long은 4000, vats는 영역 밖 누적이 정확히 10000 ms가 되는 11000에 시작한다(그 200 ms를 "
                "빼면 11200이다). (2) 12000에 앞을 본 프레임 하나(영역 안 후보), 100 ms 뒤(12100)에 얼굴 없는 프레임이 "
                "오고 그 뒤 프레임이 끊긴다. 마지막 관측이 유지된 끝은 다음 프레임의 시각과 관측 + 200 ms 중 이른 "
                "쪽이므로 12100이고, 표지를 이어받는 구간은 15100에 끝난다. 다음 프레임(15150, 얼굴 없음)은 그 "
                "뒤라서 후보가 버려지고 영역 안은 확정되지 않는다. 정면을 250 ms 이상 본 프레임(15550)에서야 "
                "gaze_away_long end가 난다. 얼굴이 사라진 시각을 언제나 관측 + 200 ms로 잡으면 15150의 프레임이 "
                "표지를 이어받아 거기서 end가 난다. (3) 21000의 프레임 하나에서 고개를 돌리고(영역 밖 후보) 다음 "
                "프레임이 정확히 인정 구간의 끝(21000 + 200 + 3000 = 24200)에 고개를 돌린 채 온다. 후보는 간격이 "
                "인정 구간을 '넘을' 때만 버리므로 살아 있고, 그 프레임에서 21000으로 소급해 확정된다. 연속 영역 밖 "
                "시간이 3200 ms라 같은 프레임에서 gaze_away_long onset이 난다(t_start_ms 21000, confidence 200 / "
                "3200). '이상'으로 버리면 24200에서 새 후보가 시작될 뿐 onset이 없다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, glance_times, glance_plan),
            baseline=baseline,
            check=both(
                expect(
                    [(*gaze_on, 4000), (*vats_on, 11000), (*gaze_end, 15550), (*gaze_on, 24200)],
                    [(*gaze_on, 4000), (*vats_on, 11000), (*gaze_on, 24200)],
                ),
                check_glance,
            ),
        ),
        _Case(
            name="rules_calibrating_large_t0",
            bundle=BUNDLE_RULES,
            description=(
                "첫 t_ms가 1731465123457인 세션의 규칙과 경고(6 fps, 68초, 기준선을 넣지 않아 보정부터 한다). 규칙 "
                "엔진과 경고 결정의 시각도 Long이어야 하고, '처음 10초'와 calibrating의 구간은 첫 프레임 대비로 잰다. "
                "첫 프레임에 calibrating onset(t_start_ms는 그 프레임의 t_ms). 보정 중의 1.5초 감김은 provisional = "
                "true인 microsleep이다. 세션 59700~61200의 프레임이 없어서(1.6초 끊김) 기준선은 61333의 프레임에서 "
                "완성된다. T_b는 60000이므로 그 프레임이 스텝 61([60000, 61000), 프레임이 없는 빈 스텝)을 발행하고 "
                "누적은 60000부터다. 같은 프레임에서 calibrating end가 난다(duration_ms 61333, confidence는 얼굴 시간 "
                "59867 / 61333). 처음 10초가 지난 뒤에도 얼굴 시간이 충분하므로 unable_to_analyze는 없다. 완성 뒤의 "
                "고개 돌림은 gaze_away_long이다(영역 밖을 처음 본 프레임에서 정확히 3000 ms 뒤에 onset)."
            ),
            sections=(SECTION_BASELINE, SECTION_STEPS, *_RULE_SECTIONS),
            frames=large_frames,
            events=(
                _CALIBRATING_ON,
                _MICROSLEEP_ON,
                _MICROSLEEP_END,
                _CALIBRATING_END,
                _GAZE_ON,
                _GAZE_END,
            ),
            alerts=(_MICROSLEEP_ON, _GAZE_ON),
            check=check_large,
        ),
        _Case(
            name="rules_vats_window_edge",
            bundle=BUNDLE_RULES,
            description=(
                "30초 창의 가장자리에서 끝나는 영역 밖 구간(5 fps). 고개를 1000~1800의 프레임에서 돌려 영역 밖 구간 "
                "[1000, 2000)이 생기고, 21000부터 다시 돌린다(gaze_away_long onset 24000). 29000의 프레임에서 "
                "최근 30초의 합은 1000 + 8000 = 9000 ms다. 다음 프레임은 3000 ms 뒤(32000)에 고개를 돌린 채 온다. "
                "끊긴 시간은 마지막 관측 + 200 + 3000 ms 전까지 영역 밖으로 치므로 둘째 구간이 [21000, 32000)이 되어 "
                "합이 11000 ms다. 그 프레임에서 30초 창의 시작은 2000이고 첫 구간은 정확히 거기서 끝난다: 끝이 창의 "
                "시작 '이하'인 구간은 버린다. 그래서 vats onset의 t_start_ms는 21000, duration_ms는 11000, "
                "confidence는 그 구간의 pose_valid 시간 비율 8200 / 11000이다(끊긴 시간 중 200 ms만 유효). 끝이 창의 "
                "시작과 같은 구간을 남기면 길이 0인 구간의 시작(2000)이 t_start_ms가 되어 duration_ms가 30000이 된다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, window_edge_times, window_edge_plan),
            baseline=baseline,
            check=both(
                expect([(*gaze_on, 24000), (*vats_on, 32000)], [(*gaze_on, 24000), (*vats_on, 32000)]),
                check_window_edge,
            ),
        ),
        _Case(
            name="rules_vats_confidence_face_lost",
            bundle=BUNDLE_RULES,
            description=(
                "vats onset의 confidence가 1보다 작은 경우(5 fps). confidence는 [t_start_ms, 발행 시각]의 pose_valid "
                "시간 비율이고, pose_valid 구간이 여럿이면 구간마다 t_start_ms 뒤의 부분만 더한다. 400과 600의 프레임에 "
                "얼굴이 없다(t_start_ms보다 앞에서 끝나는 pose_valid 구간 [0, 400)이 생긴다. 이 구간은 더하지 않는다). "
                "1000부터 고개를 돌려 gaze_away_long이 시작되고(4000), 5000~6800의 프레임 열에 얼굴이 없다가(3초 "
                "안이라 영역 밖으로 친다) 7000부터 다시 고개를 돌린 채 보인다. 영역 밖 누적이 10000 ms가 되는 "
                "11000에 vats onset: t_start_ms 1000, duration_ms 10000, confidence는 (4000 + 4000) / 10000 = 0.8이다. "
                "뒤 구간을 t_start_ms부터 통째로 세면 1.4가 되고, 앞에서 끝난 구간을 음수로 더하면 0.74가 된다. "
                "12000부터 앞을 보고 확정된 프레임(12400)에서 gaze_away_long end가 난다(confidence 9400 / 11400)."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 12800, 200), lost_plan),
            baseline=baseline,
            check=both(
                expect(
                    [(*gaze_on, 4000), (*vats_on, 11000), (*gaze_end, 12400)],
                    [(*gaze_on, 4000), (*vats_on, 11000)],
                ),
                check_face_lost_inside,
            ),
        ),
        _Case(
            name="rules_gaze_one_frame_glances",
            bundle=BUNDLE_RULES,
            description=(
                "한 프레임짜리 곁눈질은 영역 밖 시간이 아니다(5 fps, 33초). 1000부터 32200까지 400 ms마다 프레임 "
                "하나에서만 고개를 돌린다(79번, 31.4초 동안. 고개를 돌린 프레임의 시간을 더하면 15.8초다). 영역 밖 "
                "후보는 디바운스 250 ms를 채우지 못하고 다음 프레임의 영역 안 표지로 버려진다. 영역 안 상태에서 "
                "버려진 영역 밖 후보의 구간은 영역 밖 누적에 넣지 않으므로 이벤트도 경고도 없다. 버려진 후보의 "
                "구간을 상태와 무관하게 더하면 누적이 10000 ms가 되는 20800에 vats가 난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 33200, 200), one_frame_plan),
            baseline=baseline,
            events=(),
            alerts=(),
            check=check_one_frame,
        ),
        _Case(
            name="rules_microsleep_confidence_face_lost",
            bundle=BUNDLE_RULES,
            description=(
                "microsleep의 confidence가 1보다 작은 경우(10 fps). 1000부터 눈을 감고(감김의 시작 978), 1300~1500의 "
                "프레임 셋에 얼굴이 없다가 1600에 감긴 채 돌아온다. 끊긴 시간 400 ms는 hold_ms(500) 이하라 같은 "
                "감김이고 앞뒤 표본이 모두 진입 임계 이상이라 길이에 들어간다: 2000의 프레임에서 1022 ms가 되어 "
                "microsleep onset이 난다. confidence는 감김을 처음 본 프레임(1000)부터 발행 시각까지의 "
                "closure_valid 시간 비율이다: 얼굴 없는 프레임이 유지된 시간은 유효 시간이 아니므로 (300 + 400) / "
                "1000 = 0.7이다. 눈을 뜬 프레임(2600)의 end는 1300 / 1600 = 0.8125다. 모든 프레임의 유지 시간을 "
                "세면 둘 다 1.0이 된다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 3100, 100), closure_lost_plan),
            baseline=baseline,
            check=both(expect([(*ms_on, 2000), (*ms_end, 2600)], [(*ms_on, 2000)]), check_closure_lost),
        ),
        _Case(
            name="rules_alert_pending_survives_onset",
            bundle=BUNDLE_RULES,
            description=(
                "미뤄 둔 microsleep 경고는 다른 종류의 onset으로도 지워지지 않는다(5 fps). 속도 10 km/h. 고개를 "
                "돌린 채(1000~) 눈을 감아(2800~) microsleep이 먼저 시작되고(3800) 경고는 속도에 막혀 미뤄진다. 다음 "
                "프레임(4000)에서 gaze_away_long onset이 나고 그 경고도 미뤄진다: 이 onset은 미뤄 둔 microsleep "
                "경고를 지우지 않는다(지우는 것은 microsleep의 end와 sleep의 onset·end뿐이다). 속도가 25 km/h가 된 "
                "프레임(4400)에서 두 경고가 우선순위 순(microsleep, gaze_away_long)으로 함께 나간다. 둘 다 onset의 "
                "사본이고 duration_ms만 그 시각 기준(1644, 3400)이다. 눈을 뜨고 앞을 본 프레임(5000)에서 microsleep "
                "end, 영역 안이 확정된 프레임(5400)에서 gaze_away_long end가 난다. rules_alert_pending_survives_end는 "
                "다른 종류의 end만 지난다."
            ),
            sections=_RULE_SECTIONS,
            frames=_hand_frames(spec, range(0, 6200, 200), onset_plan),
            baseline=baseline,
            speed=((first, 10.0), (first + 4400, 25.0)),
            check=both(
                expect(
                    [(*ms_on, 3800), (*gaze_on, 4000), (*ms_end, 5000), (*gaze_end, 5400)],
                    [(*ms_on, 4400), (*gaze_on, 4400)],
                ),
                check_survives_onset,
            ),
        ),
        _Case(
            name="rules_unable_small_t0",
            bundle=BUNDLE_RULES,
            description=(
                "첫 t_ms가 작은 세션(첫 프레임의 t_ms가 5000, 5 fps). 아래의 시각은 세션 시간(t_ms − 5000)이다. "
                "처음부터 얼굴이 없다. 분석 불가를 판정하지 않는 '처음 10초'는 t_ms 0이 아니라 첫 프레임부터 잰다: "
                "unable_to_analyze onset은 첫 프레임에서 정확히 10000 ms 뒤의 프레임(t_ms 15000)에서 난다. 0부터 "
                "재면 t_ms 10000의 프레임(세션 5000)에서 난다(첫 t_ms가 1인 다른 벡터에서는 이 차이가 드러나지 "
                "않는다). 12000부터 얼굴이 보이고, 최근 10초의 얼굴 시간이 5000 ms가 된 프레임(17000)에서 end가 "
                "난다(duration_ms 7000, confidence 5000 / 7000)."
            ),
            sections=_RULE_SECTIONS,
            frames=small_frames,
            baseline=baseline,
            check=check_small_t0,
        ),
        _Case(
            name="rules_drowsy_off_step_boundary",
            bundle=BUNDLE_RULES,
            description=(
                "스텝 경계에서 벗어난 프레임이 내는 drowsy(3 fps, 72초, 속도 60 km/h). rules_drowsy_perclos_exact와 "
                "같은 감김인데 첫 프레임을 뺀 모든 프레임이 50 ms 늦다(세션 0, 383, 717, 1050, ...). 그래서 스텝 k는 "
                "세션 1000k + 50의 프레임이 발행한다. perclos_60이 스텝 60부터 0.15라 열 번째인 스텝 69에 drowsy "
                "onset이 나고, 그 스텝을 발행한 프레임은 69050이다. drowsy 이벤트의 발행 시각은 그 프레임이 아니라 "
                "스텝의 끝(69000)이다: t_start_ms는 스텝 60의 끝(60000), duration_ms는 9000이다. 속도가 기준(50 km/h) "
                "이상이라 경고가 그 프레임에서 바로 나가고, 바로 낸 경고는 onset 이벤트 그대로다(duration_ms 9000). "
                "duration_ms를 프레임 시각 기준으로 고치는 것은 미뤘다 낸 경고와 되풀이 경고뿐이다. 이벤트나 경고의 "
                "길이를 프레임 시각에서 재면 9050이 된다. 다른 벡터에서는 drowsy 이벤트를 낸 프레임의 시각이 스텝의 "
                "끝과 같거나 1 ms 차이라 이 차이가 드러나지 않는다."
            ),
            sections=(SECTION_STEPS, *_RULE_SECTIONS),
            frames=_hand_frames(spec, off_times, exact_plan),
            baseline=baseline,
            speed=((first, 60.0),),
            check=both(expect([(*drowsy_on, 69050)], [(*drowsy_on, 69050)]), check_off_boundary),
        ),
    ]


# --- 전 과정과 모델 ------------------------------------------------------------


def _onnx_model(model_file: str, ctx: Context) -> dict[str, Any]:
    path = ctx.model_dir / model_file
    if not path.is_file():
        raise VectorError(f"model file not found: {path}")
    return {"kind": MODEL_KIND_ONNX, "file": model_file, "sha256": file_sha256(path)}


def _e2e_case(ctx: Context, model: Mapping[str, Any]) -> _Case:
    spec = ctx.spec
    segments: list[Segment] = [Blink(400, start_ms=start) for start in range(3000, 90000, 6000)]
    segments += [Closure(hold_ms=700, depth=_LOW_FPS_DEPTH, start_ms=90000 + 1900 * index) for index in range(16)]
    segments += [Closure(hold_ms=1500, depth=_LOW_FPS_DEPTH, start_ms=125000)]
    segments += [Blink(400, start_ms=start) for start in range(131000, 250000, 6000)]
    segments += [Yawn(4000, start_ms=140500), Nod(1000, pitch=20.0, start_ms=200300)]

    def check(computed: Mapping[str, Any]) -> None:
        steps = computed[SECTION_STEPS]
        probs = computed[SECTION_PROBS]
        window = computed[SECTION_MODEL_INPUTS][0]
        _require(computed[SECTION_BASELINE]["t_complete_ms"] == 60000, "the baseline completes at 60 s")
        _require(steps[0]["k"] == 61 and steps[-1]["k"] == 250, "steps 61..250")
        _require([entry["k"] for entry in probs] == list(range(239, 251)), "the model runs every step from 239")
        _require(window["k"] == 239 and len(window["features"]) == ctx.window_steps, "the first window ends at 239")
        by_k = {step["k"]: step["values"] for step in steps}
        _require(window["features"][0] == by_k[120] and window["features"][-1] == by_k[239], "oldest step first")
        drowsy = [event for event in computed[SECTION_EVENTS] if event["type"] == EVENT_DROWSY]
        _require(all(event["source"] == SOURCE_PERCLOS for event in drowsy), "drowsy is PERCLOS-driven here")
        _require(max(entry["probs"][CLASS_DROWSY] for entry in probs) < ctx.rules.drowsiness.model.onset_p, "p_drowsy")

    return _Case(
        name=E2E_NAME,
        bundle=BUNDLE_E2E,
        description=(
            "전 과정 250초(6 fps, 속도 80 km/h). 프레임 → 보정(60초에 기준선 완성) → 스텝 61~250 → 스텝 239부터 "
            f"120스텝 창(스텝 120~239, 가장 오래된 스텝이 먼저)을 {model['file']}에 넣어 probs → 이벤트와 "
            "경고. 90~120초의 긴 깜빡임으로 perclos_60이 올라가 스텝 129에 drowsy onset, 내려간 뒤 end가 난다. "
            "이 모델은 무작위 가중치라 실제로 나올 수 있는 특징(0 이상)에서는 p_drowsy가 0.6에 닿지 않는다. 그래서 "
            "이 벡터의 drowsy는 기본 판정원인 PERCLOS 규칙이 낸 것이고(source perclos), 모델 확률은 probs로만 "
            "확인한다. 모델이 판정을 내는 경로는 rules_drowsy_model_stub이 고정한다."
        ),
        sections=(
            SECTION_BASELINE,
            SECTION_STEPS,
            SECTION_MODEL_INPUTS,
            SECTION_STEP_PROBS,
            SECTION_PROBS,
            SECTION_EVENTS,
            SECTION_ALERTS,
            SECTION_COUNTERS,
        ),
        frames=_render(spec, segments, 250500, fps=6.0),
        speed=((1, 80.0),),
        model=model,
        events=(_CALIBRATING_ON, _CALIBRATING_END, _MICROSLEEP_ON, _MICROSLEEP_END, _DROWSY_ON, _DROWSY_END),
        alerts=(_MICROSLEEP_ON, _DROWSY_ON),
        check=check,
    )


def _model_case(ctx: Context, model: Mapping[str, Any], real_window: Sequence[Sequence[float]]) -> _Case:
    """모델 입력 셋: 0으로 채운 창, 전 과정 벡터의 실제 창, 범위를 벗어난 값이 섞인 무작위 창."""
    zeros = tuple(tuple(0.0 for _ in range(ctx.n_features)) for _ in range(ctx.window_steps))
    real = tuple(tuple(float(value) for value in row) for row in real_window)
    rng = random.Random(17)
    wide = tuple(
        tuple(round(rng.uniform(-3.0, 9.0), 3) for _ in range(ctx.n_features)) for _ in range(ctx.window_steps)
    )

    def check(computed: Mapping[str, Any]) -> None:
        rows = [entry["probs"] for entry in computed[SECTION_PROBS]]
        for probs in rows:
            _require(len(probs) == 3 and abs(sum(probs) - 1.0) < 1e-5, f"probs must sum to 1, got {probs}")
        _require(rows[0] != rows[1] and rows[1] != rows[2], "the three inputs must give different outputs")

    return _Case(
        name="model_v0_inputs",
        bundle=BUNDLE_MODEL,
        description=(
            f"모델 파일 {model['file']}(sha256은 model 항목)에 넣는 원시 스텝 특징 셋과 probs(alert, low_vigilant, "
            "drowsy). model_inputs[i]는 [120][17]이고 [0]이 가장 오래된 스텝, [119]가 가장 최근 스텝이다"
            "(ONNX 입력 features[0][t][f]). 입력 0은 0으로 채운 창, 1은 e2e_250s의 첫 창(실제 척도), 2는 -3~9의 "
            "무작위 값(그래프 안의 log1p(max(x, 0)), 표준화, ±5 자르기를 지난다)이다. 표준화하지 않은 값을 그대로 "
            "float32로 바꿔 넣는다."
        ),
        sections=(SECTION_PROBS,),
        model=model,
        model_inputs=(zeros, real, wide),
        check=check,
    )


# ---------------------------------------------------------------------------
# 생성과 명령
# ---------------------------------------------------------------------------


def build_vectors(ctx: Context, model_file: str = Path(DEFAULT_MODEL_PATH).name) -> list[dict[str, Any]]:
    """모든 벡터를 만든다. 순서와 이름은 `VECTOR_NAMES`와 같다. 모델 파일은 `ctx.model_dir`에서 찾는다."""
    model = _onnx_model(model_file, ctx)
    cases = _signal_cases(ctx) + _baseline_cases(ctx) + _run_cases(ctx) + _step_cases(ctx)
    cases += _rule_cases(ctx) + _boundary_rule_cases(ctx)
    cases.append(_e2e_case(ctx, model))
    vectors = [_build_vector(case, ctx) for case in cases]
    by_name = {vector["name"]: vector for vector in vectors}

    real_window = by_name[E2E_NAME]["expected"][SECTION_MODEL_INPUTS][0]["features"]
    vectors.append(_build_vector(_model_case(ctx, model, real_window), ctx))

    names = tuple(vector["name"] for vector in vectors)
    _require(names == VECTOR_NAMES, f"vector names differ from VECTOR_NAMES: {names}")
    plain = by_name["steps_five_minutes"]["expected"][SECTION_STEPS]
    shifted = by_name["steps_five_minutes_large_t0"]["expected"][SECTION_STEPS]
    _require(
        [(step["k"], step["values"]) for step in plain] == [(step["k"], step["values"]) for step in shifted],
        "a large first t_ms must not change step numbers or feature values",
    )
    return vectors


def generate(out_dir: Path, ctx: Context, model_file: str = Path(DEFAULT_MODEL_PATH).name) -> list[Path]:
    """벡터를 만들어 out_dir에 쓰고 쓴 파일 목록을 돌려준다. 같은 입력이면 파일이 바이트 단위로 같다."""
    return write_vectors(build_vectors(ctx, model_file), out_dir)


def _cmd_generate(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    out_dir = Path(args.out)
    try:
        ctx = load_context(args.spec, args.rules, model_path.parent)
        paths = generate(out_dir, ctx, model_path.name)
    except (VectorError, SpecError, OSError) as error:
        print(f"could not generate vectors: {error}", file=sys.stderr)
        return 1
    total = 0
    for path in paths:
        size = path.stat().st_size
        total += size
        print(f"{size:>9d}  {path.name}")
    print(f"{total:>9d}  total, {len(paths)} vectors in {out_dir}")
    stale = sorted(path.name for path in vector_files(out_dir) if path not in paths)
    if stale:
        print(f"warning: not written by this run (stale?): {', '.join(stale)}", file=sys.stderr)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    try:
        ctx = load_context(args.spec, args.rules, args.model_dir)
    except (SpecError, OSError) as error:
        print(f"could not load the configuration: {error}", file=sys.stderr)
        return 1
    results = verify_dir(directory, ctx)
    if len(results) == 0:
        print(f"no vector files in {directory}", file=sys.stderr)
        return 1
    failed = 0
    for name, problems in results.items():
        if problems:
            failed += 1
            print(f"FAIL  {name}")
            for problem in problems:
                print(f"      {problem}")
        else:
            print(f"ok    {name}")
    missing = sorted(set(VECTOR_NAMES) - set(results))
    if missing:
        print(f"FAIL  missing vectors: {', '.join(missing)}")
    print(f"{len(results) - failed} of {len(results)} vectors verified in {directory}")
    return 1 if failed or missing else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m drowsy.vectors", description="test vectors for the ported code")
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("generate", help="write the test vectors")
    make.add_argument("--out", required=True, help="output folder, e.g. tests/vectors")
    make.add_argument("--model", default=DEFAULT_MODEL_PATH, help="ONNX model for the model and e2e vectors")

    check = commands.add_parser("verify", help="recompute every expected value and compare")
    check.add_argument("--dir", required=True, help="folder holding the vector files")
    check.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="folder holding the model named in the vectors")

    for command in (make, check):
        command.add_argument("--spec", default=DEFAULT_FEATURE_SPEC_PATH)
        command.add_argument("--rules", default=DEFAULT_RULES_PATH)

    args = parser.parse_args(argv)
    if args.command == "generate":
        return _cmd_generate(args)
    return _cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
