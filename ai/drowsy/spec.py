"""feature_spec.json과 규칙 설정의 자료형, 로드, 검증, 해시.

- import 시점에는 표준 라이브러리만 쓴다. yaml은 `load_rules` 안에서만 import한다.
  이식 대상 모듈이 이 모듈의 자료형을 import해도 numpy나 yaml이 따라 올라오지 않는다.
- `PipelineSpec`: feature_spec.json의 `pipeline` 블록. 이식 코드가 쓰는 상수 전부.
- `FeatureSpec`: feature_spec.json 전체(특징 목록, 표준화 상수, pipeline, pipeline_hash).
- `RulesConfig`: config/rules.yaml. 앱에는 같은 내용을 export/rules.json으로 내보낸다.

해시 (PLAN 4.7절)
- `pipeline_hash`: `pipeline` 블록을 정규 JSON(키 정렬, 공백 없는 구분자, ASCII 이스케이프)으로 쓴 것의 sha256.
  Python만 계산하고 Kotlin은 문자열 비교만 한다.
- `feature_spec_hash`: 내보낸 feature_spec.json 파일 바이트의 sha256.
- `rules_hash`: 내보낸 rules.json 파일 바이트의 sha256.

명령 (저장소 루트에서 실행)
    python -m drowsy.spec check
    python -m drowsy.spec update-hash
    python -m drowsy.spec export-rules --out export/rules.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drowsy.types import (
    CLASS_ORDER,
    DROWSY_SOURCES,
    EVENT_CALIBRATING,
    EVENT_DROWSY,
    EVENT_GAZE_AWAY_LONG,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
    EVENT_TYPES,
    EVENT_UNABLE_TO_ANALYZE,
    EVENT_VATS,
    FEATURE_NAMES,
    N_FEATURES,
)

TRANSFORM_NONE = "none"
TRANSFORM_LOG1P = "log1p"
TRANSFORMS: tuple[str, ...] = (TRANSFORM_NONE, TRANSFORM_LOG1P)

STANDARDIZATION_IN_GRAPH = "in_graph"

DEFAULT_FEATURE_SPEC_PATH = "config/feature_spec.json"
DEFAULT_RULES_PATH = "config/rules.yaml"


class SpecError(ValueError):
    """spec이나 설정 파일의 내용이 계약과 맞지 않을 때."""


# ---------------------------------------------------------------------------
# PipelineSpec: feature_spec.json의 pipeline 블록
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LandmarkSpec:
    """랜드마크 인덱스 집합 (PLAN 2.1절). R은 MediaPipe RIGHT_EYE, L은 LEFT_EYE다."""

    ear_r: tuple[int, ...]  # EAR p1..p6 = 33, 160, 158, 133, 153, 144
    ear_l: tuple[int, ...]  # EAR p1..p6 = 362, 385, 387, 263, 373, 380
    iris_r: tuple[int, ...]  # 468(중심), 469~472. 순서는 중심, +x, 위, -x, 아래
    iris_l: tuple[int, ...]  # 473(중심), 474~477. 순서는 같다
    corner_outer_r: int  # 33  (시선 대용치용 눈 꼬리: 관자놀이 쪽)
    corner_inner_r: int  # 133 (코 쪽)
    corner_outer_l: int  # 263
    corner_inner_l: int  # 362
    contour_r: tuple[int, ...]  # 눈 윤곽 16점: 바깥 꼬리에서 윗눈꺼풀을 따라 안쪽 꼬리까지, 아랫눈꺼풀로 돌아옴
    contour_l: tuple[int, ...]
    face_width: tuple[int, ...]  # 얼굴 폭 측정용 2점 (234, 454). 품질 통계에만 쓴다

    def required_indices(self) -> tuple[int, ...]:
        """RawFrame.lm에 들어 있어야 하는 인덱스 44개(오름차순)."""
        merged = set(self.contour_r) | set(self.contour_l) | set(self.iris_r) | set(self.iris_l) | set(self.face_width)
        return tuple(sorted(merged))


@dataclass(frozen=True, slots=True)
class BlendshapeSpec:
    """블렌드셰이프 이름 (PLAN 2.1절). RawFrame.bs는 category_name으로 찾는다."""

    eye_blink_l: str  # eyeBlinkLeft (인덱스 9), L 집합과 짝
    eye_blink_r: str  # eyeBlinkRight (인덱스 10), R 집합과 짝
    eye_look: tuple[str, ...]  # 8개. FrameSignals.eye_look과 같은 순서 (in_l, out_l, up_l, down_l, in_r, out_r, up_r, down_r)
    jaw_open: str  # jawOpen (인덱스 25)
    order: tuple[str, ...]  # 52개 전체. 위치가 MediaPipe Category.index와 같다

    def required_names(self) -> tuple[str, ...]:
        """RawFrame.bs에 들어 있어야 하는 이름 11개."""
        return (self.eye_blink_l, self.eye_blink_r) + self.eye_look + (self.jaw_open,)

    def index_of(self, name: str) -> int:
        """MediaPipe 블렌드셰이프 인덱스."""
        return self.order.index(name)


@dataclass(frozen=True, slots=True)
class PoseSpec:
    """각도 규약 (PLAN 2.2절). 계산식은 고정이고 이 값들은 규약을 해시에 묶어 두는 기록이다."""

    order: str  # "Ry*Rx*Rz"
    yaw_pos: str  # "subject_left"
    pitch_pos: str  # "chin_down"
    roll_pos: str  # "ccw_in_image"
    matrix_layout: str  # "column_major": M[i][j] = m[4*j + i]
    angle_unit: str  # "deg"


@dataclass(frozen=True, slots=True)
class ClockSpec:
    """시간 규칙 (PLAN 3.3절)."""

    step_ms: int  # 1000. 스텝 길이
    max_hold_ms: int  # 200. 한 프레임의 상태를 다음 프레임까지 인정하는 최대 시간. 넘는 부분은 "데이터 없음"
    reset_gap_ms: int  # 10000. 프레임 간격이 이 값을 넘으면 검출기와 누적기를 초기화(기준선은 유지)


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """원시 신호 (PLAN 4.1절)."""

    min_eye_width_px: float  # 6.0. 눈 폭이 이 값 미만이면 쓸 수 없는 눈


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """기준선 (PLAN 4.2절, D5).

    분위수는 최근접 순위 `sorted[ceil(q*n) - 1]`, 중앙값만 표본이 짝수 개일 때 가운데 두 값의 평균이다.
    """

    duration_ms: int  # 60000. 기본 수집 시간(세션 시간)
    extend_step_ms: int  # 30000. 완성 조건 미달 시 한 번에 연장하는 시간
    max_duration_ms: int  # 180000. 여기서도 미달이면 전역 상수로 대체
    min_face_frames: int  # 300. 얼굴 프레임 수 하한
    min_face_ratio: float  # 0.5. 얼굴 검출 비율(시간 기준) 하한
    e_min: float  # 0.12. E의 허용 범위 하한
    e_max: float  # 0.45. E의 허용 범위 상한
    ear_quantile: float  # 0.9. E = EAR의 90 분위수
    blink_quantile: float  # 0.1. B = eyeBlink의 10 분위수
    blink_record_quantiles: tuple[float, ...]  # (0.5, 0.9). 분포 기록용
    off_axis_yaw_deg: float  # 15.0. |yaw0|이 이 값을 넘으면 status = off_axis


@dataclass(frozen=True, slots=True)
class ClosureSpec:
    """closure 식의 상수 (PLAN 4.3절).

    a_e = clip((E_e − ear_e) / (E_e · (1 − rho)), 0, 1)
    s_e = clip((blink_e − B_e) / max(b_closed − B_e, b_span_min), 0, 1)
    c_e = w · a_e + (1 − w) · s_e
    w_far = clip((far_eye_zero_yaw_deg − |yaw_rel|) / far_eye_ramp_deg, 0, 1)
    """

    rho: float  # 0.35 (잠정)
    b_closed: float  # 0.60 (잠정)
    b_span_min: float  # 0.2. 분모 max(b_closed − B_e, ·)의 하한
    w: float  # 0.5. EAR 쪽 비중
    far_eye_zero_yaw_deg: float  # 25.0. |yaw_rel|이 이 값 이상이면 가까운 눈만 쓴다
    far_eye_ramp_deg: float  # 10.0. 먼 쪽 눈 비중이 1에서 0으로 줄어드는 구간 폭
    valid_max_abs_yaw_deg: float  # 40.0. |yaw_rel|이 이 값을 넘으면 closure 무효


@dataclass(frozen=True, slots=True)
class RunDetectorSpec:
    """`ClosureRunDetector` 설정 (PLAN 4.5절).

    같은 검출기를 두 번 쓴다. `PipelineSpec.run_detector`는 스텝 특징용(0.5 / 0.3),
    `RulesConfig.closure.detector`는 규칙용(0.7 / 0.5)이고 나머지 값은 같다.
    검출기는 이 자료형의 값만 있으면 만들 수 있다.
    """

    enter_thr: float  # closure가 이 값 이상이면 감김 구간 시작
    exit_thr: float  # 이 값 미만이면 구간 종료
    min_duration_ms: int  # 50. 이보다 짧은 이벤트는 버림
    max_dt_ms: int  # 200. 이보다 긴 프레임 간격은 속도 계산과 시각 보간에서 제외
    hold_ms: int  # 500. 감긴 중 유효 프레임이 끊겨도 이만큼은 같은 구간으로 이음
    velocity_lookback_samples: int  # 2. 감는 속도는 진입 이 표본 수 전부터 정점까지에서 구한다


@dataclass(frozen=True, slots=True)
class StepFeatureSpec:
    """스텝 특징 상수 (PLAN 4.4절, 4.6절)."""

    names: tuple[str, ...]  # 17개, 고정 순서. drowsy.types.FEATURE_NAMES와 같다
    window_steps: int  # 120. 모델 입력 창 길이(스텝). full_lookback 스텝이 연속 이만큼 쌓이면 모델을 부른다
    lookback_long_ms: int  # 60000. 긴 조회 구간. full_lookback 기준(스텝 끝 ≥ T_b + 이 값)이기도 하다
    lookback_short_ms: int  # 10000. 짧은 조회 구간
    perclos_thr: float  # 0.8. closure가 이 값 이상인 시간을 센다 (잠정)
    perclos_min_valid_ms_60: int  # 5000. 유효 시간이 이 값 미만이면 perclos_60 = 0
    perclos_min_valid_ms_10: int  # 2000. 유효 시간이 이 값 미만이면 perclos_10 = 0
    blink_rate_min_valid_ms: int  # 5000. 유효 시간이 이 값 미만이면 blink_rate_60 = 0
    blink_duration_cap_ms: int  # 2000. 지속 시간 평균과 표준편차에서 이벤트별 상한
    long_closure_min_ms: int  # 500. 이 길이 이상이면 긴 감김으로 센다
    max_closure_cap_ms: int  # 10000. max_closure_ms_*의 상한
    pitch_std_min_face_ms: int  # 5000. 얼굴 시간이 이 값 미만이면 pitch_std_60 = 0


@dataclass(frozen=True, slots=True)
class YawnSpec:
    """하품 이벤트 (PLAN 4.6절, 잠정). 입력값은 jaw_rel = jaw_open − jaw0."""

    enter_thr: float  # 0.40. 이 값 이상이면 시작
    exit_thr: float  # 0.25. 이 값 미만이면 끝
    count_at_ms: int  # 2000. 지속 시간이 이 값에 도달하면 한 번 센다
    cancel_gap_ms: int  # 500. 무효 구간이 이 값을 넘으면 진행 중이던 판정을 취소


@dataclass(frozen=True, slots=True)
class NodSpec:
    """끄덕임 이벤트 (PLAN 4.6절, 잠정). 입력값은 d = pitch_rel − m, m은 pitch_rel의 느린 평균.

    유효 프레임마다 d를 먼저 계산한 뒤 m += α·(pitch_rel − m), α = dt / (ema_tau_ms + dt),
    dt = min(프레임 간격, ema_max_dt_ms). m의 초기값은 첫 유효 표본이다.
    """

    enter_deg: float  # 10.0. d가 이 값 이상이면 시작
    exit_deg: float  # 5.0. 이 값 미만이면 끝
    min_duration_ms: int  # 200. 끝날 때 지속 시간이 [min, max] 안이면 센다
    max_duration_ms: int  # 3000
    ema_tau_ms: int  # 10000
    ema_max_dt_ms: int  # 200
    cancel_gap_ms: int  # 500. 무효 구간이 이 값을 넘으면 진행 중이던 판정을 취소


@dataclass(frozen=True, slots=True)
class GlobalConstants:
    """전역 상수 (PLAN 4.3절). 기준선 대체, 기준선 완성 전의 임시 closure, 전역 정규화 조건에 같은 값을 쓴다.

    spec v0에서는 설계값이고 E0에서 한 번 바꾼다. 두 눈에 같은 e, b를 쓴다.
    가정: 계획서는 E, B, jaw0, R0만 정했다. gaze_h0과 gaze_v0은 0.0으로 둔다(대체 상태에서는 시선 규칙이 꺼진다).
    """

    e: float  # 0.27
    b: float  # 0.22
    jaw0: float  # 0.08
    gaze_h0: float  # 0.0
    gaze_v0: float  # 0.0
    r0: tuple[float, ...]  # 단위 행렬, 행 우선 9개


@dataclass(frozen=True, slots=True)
class MediaPipeSpec:
    """MediaPipe 전제 (Q2, Q3)."""

    model_sha256: str  # face_landmarker.task (float16 v1)의 sha256
    num_faces: int  # 1. 1이 아니면 시간 평활이 꺼진다


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """feature_spec.json의 `pipeline` 블록: 이식 코드가 쓰는 상수 전부."""

    landmarks: LandmarkSpec
    blendshapes: BlendshapeSpec
    pose: PoseSpec
    clock: ClockSpec
    signals: SignalSpec
    baseline: BaselineSpec
    closure: ClosureSpec
    run_detector: RunDetectorSpec
    step_features: StepFeatureSpec
    yawn: YawnSpec
    nod: NodSpec
    global_constants: GlobalConstants
    mediapipe: MediaPipeSpec


# ---------------------------------------------------------------------------
# FeatureSpec: feature_spec.json 전체
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureDef:
    """스텝 특징 하나 (PLAN 4.6절)."""

    index: int
    name: str
    window_sec: int  # 60 또는 10
    unit: str
    formula: str  # 한 줄 설명. 상수 이름은 pipeline.step_features의 키
    undefined_value: float  # 정의할 수 없을 때의 값
    transform: str  # "none" 또는 "log1p". 그래프 안에서 표준화 전에 적용
    mean: float  # 변환 뒤 값의 학습 데이터 평균. v0은 0.0
    std: float  # 변환 뒤 값의 학습 데이터 표준편차(std_floor 이상). v0은 1.0


@dataclass(frozen=True, slots=True)
class Standardization:
    """그래프 안 전처리 상수 (PLAN 5.1절, Q6)."""

    location: str  # "in_graph". 앱은 원시 스텝 특징을 넣는다
    z_clip: float  # 5.0. 표준화 뒤 ±z_clip으로 자른다
    std_floor: float  # 0.001. 표준편차 하한
    temperature: float  # 로짓을 이 값으로 나눈 뒤 소프트맥스. v0은 1.0


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """feature_spec.json 전체 (PLAN 4.7절)."""

    spec_version: int
    step_sec: int
    window_steps: int
    class_order: tuple[str, ...]
    features: tuple[FeatureDef, ...]
    standardization: Standardization
    pipeline: PipelineSpec
    pipeline_hash: str

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @property
    def means(self) -> tuple[float, ...]:
        return tuple(f.mean for f in self.features)

    @property
    def stds(self) -> tuple[float, ...]:
        return tuple(f.std for f in self.features)

    @property
    def log1p_mask(self) -> tuple[bool, ...]:
        """특징별로 log1p(max(x, 0))를 적용하는지."""
        return tuple(f.transform == TRANSFORM_LOG1P for f in self.features)


# ---------------------------------------------------------------------------
# RulesConfig: config/rules.yaml (PLAN 8장)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForwardZoneRules:
    """전방 주시 영역 (PLAN 8.1절). 값은 모두 잠정.

    영역 안: |yaw_rel| ≤ yaw_abs_max_deg 이고 −pitch_up_max_deg ≤ pitch_rel ≤ pitch_down_max_deg.
    눈동자 보조(D12): |gaze_h_rel| > gaze_h_abs_max 또는 |gaze_v_rel| > gaze_v_abs_max이면 영역 밖.
    """

    yaw_abs_max_deg: float  # 23.0
    pitch_up_max_deg: float  # 25.0. 위쪽 한계(pitch는 턱이 내려갈 때 양수이므로 −25까지)
    pitch_down_max_deg: float  # 18.0. 아래쪽 한계
    eye_aux_enabled: bool  # true. 눈동자 보조 판정을 쓸지
    gaze_h_abs_max: float  # 0.25
    gaze_v_abs_max: float  # 0.35
    debounce_ms: int  # 250. 상태 변화는 이만큼 유지돼야 확정되고, 시작 시각은 처음 나타난 프레임으로 소급
    face_loss_credit_ms: int  # 3000. 영역 밖에서 얼굴이 사라지면 이 시간까지는 영역 밖으로 친다


@dataclass(frozen=True, slots=True)
class GazeAwayLongRules:
    """장시간 시선 이탈."""

    onset_ms: int  # 3000. 영역 밖이 연속 이 시간에 도달하면 시작


@dataclass(frozen=True, slots=True)
class VatsRules:
    """VATS: 최근 window_ms 중 영역 밖 누적이 cumulative_ms에 도달하면 시작 (D6)."""

    window_ms: int  # 30000
    cumulative_ms: int  # 10000
    reset_enabled: bool  # false. 켜면 전방 주시가 reset_forward_ms 이어질 때 누적을 지운다
    reset_forward_ms: int  # 2000 (Euro NCAP 2023 값). reset_enabled일 때만 쓴다


@dataclass(frozen=True, slots=True)
class ClosureRules:
    """눈 감김 규칙 (PLAN 8.1절, D2).

    임계 시간은 유효한 감김 프레임이 도착했을 때 `ongoing_ms` ≥ 값으로 판정한다.
    """

    detector: RunDetectorSpec  # 규칙용 검출기. 진입 0.7 / 이탈 0.5, 나머지는 특징용과 같은 값
    microsleep_ms: int  # 1000
    sleep_ms: int  # 3000. "3.0초 이상". 명세의 "3초 초과"로 바꾸려면 3001


@dataclass(frozen=True, slots=True)
class ModelDrowsyRules:
    """모델 졸음 판정 (PLAN 6장)."""

    onset_p: float  # 0.6. p_drowsy ≥ 이 값이
    onset_steps: int  # 10. 이 스텝 수만큼 연속이면 시작
    end_p: float  # 0.4. p_drowsy ≤ 이 값이
    end_steps: int  # 10. 이 스텝 수만큼 연속이면 종료
    refractory_ms: int  # 60000. 종료 후 이 시간 동안은 다시 시작하지 않는다


@dataclass(frozen=True, slots=True)
class PerclosDrowsyRules:
    """PERCLOS 졸음 판정 (PLAN 8.2절).

    가정: 계획서는 PERCLOS 판정의 재시작 금지 시간을 적지 않았다. 0(없음)으로 둔다.
    """

    onset_ratio: float  # 0.15. perclos_60 ≥ 이 값이
    onset_steps: int  # 10. 이 스텝 수만큼 연속이면 시작
    end_ratio: float  # 0.10. perclos_60 < 이 값이
    end_steps: int  # 10. 이 스텝 수만큼 연속이면 종료
    refractory_ms: int  # 0


@dataclass(frozen=True, slots=True)
class DrowsinessRules:
    """졸음 판정원 (D11).

    가정: 교차검증 기준(D11)을 통과하기 전에는 모델이 무작위 가중치이므로 기본값을 perclos로 둔다.
    """

    source: str  # "model" | "perclos" | "either"
    model: ModelDrowsyRules
    perclos: PerclosDrowsyRules


@dataclass(frozen=True, slots=True)
class UnableToAnalyzeRules:
    """분석 불가 (PLAN 8.1절)."""

    window_ms: int  # 10000
    min_face_ratio: float  # 0.5. 최근 window_ms의 얼굴 검출 비율(시간 기준, 데이터 없음 포함)이 이 값 미만이면 분석 불가


@dataclass(frozen=True, slots=True)
class EventPolicy:
    """이벤트 종류 하나의 심각도와 경고 정책 (PLAN 8.2절 표, D3).

    가정: 표의 "없음"은 0으로 적는다. sleep의 "없음(3초마다 반복)"은 cooldown_ms = 0, repeat_ms = 3000이다.
    """

    severity: int  # 0~3
    cooldown_ms: int  # 재경고 대기. 0이면 없음
    repeat_ms: int  # 상태가 이어지는 동안 경고를 되풀이하는 주기. 0이면 되풀이하지 않음
    min_speed_kmh: float  # 속도가 이 값 미만이면 경고 억제. 0이면 억제 없음. 속도 입력이 없으면 억제하지 않는다


@dataclass(frozen=True, slots=True)
class EventPolicies:
    """이벤트 종류별 정책."""

    calibrating: EventPolicy
    unable_to_analyze: EventPolicy
    gaze_away_long: EventPolicy
    vats: EventPolicy
    drowsy: EventPolicy
    microsleep: EventPolicy
    sleep: EventPolicy

    def get(self, event_type: str) -> EventPolicy:
        """EVENT_* 문자열로 정책을 찾는다."""
        if event_type == EVENT_CALIBRATING:
            return self.calibrating
        if event_type == EVENT_UNABLE_TO_ANALYZE:
            return self.unable_to_analyze
        if event_type == EVENT_GAZE_AWAY_LONG:
            return self.gaze_away_long
        if event_type == EVENT_VATS:
            return self.vats
        if event_type == EVENT_DROWSY:
            return self.drowsy
        if event_type == EVENT_MICROSLEEP:
            return self.microsleep
        if event_type == EVENT_SLEEP:
            return self.sleep
        raise KeyError(f"unknown event type: {event_type!r}")


@dataclass(frozen=True, slots=True)
class RulesConfig:
    """규칙 엔진과 경고 결정의 설정 전체 (PLAN 8장).

    가정: 규칙 엔진은 이 설정만 받는다(`RuleEngine(cfg)`). 그래서 PipelineSpec에 있는 값 중 규칙 엔진이
    쓰는 것(시간 규칙, 감김 검출기 설정)을 여기에도 둔다. 두 곳의 값이 같은지는 테스트가 검사한다.
    가정: 계획서의 경고 우선순위는 sleep, microsleep, gaze_away_long, vats, drowsy 다섯 종류만 정했다.
    alert_priority에는 그 뒤에 unable_to_analyze, calibrating을 두어 모든 종류를 싣는다.
    """

    clock: ClockSpec  # 시간 규칙(PLAN 3.3절). pipeline.clock과 같은 값
    forward_zone: ForwardZoneRules
    gaze_away_long: GazeAwayLongRules
    vats: VatsRules
    closure: ClosureRules
    drowsiness: DrowsinessRules
    unable_to_analyze: UnableToAnalyzeRules
    events: EventPolicies
    alert_priority: tuple[str, ...]  # 높은 순위부터. 모든 이벤트 종류가 한 번씩 들어 있다


# ---------------------------------------------------------------------------
# dict <-> dataclass 변환
# ---------------------------------------------------------------------------

_UNION_TYPE = type(int | None)


def build_dataclass(cls: type, data: Any, where: str, strict_float: bool = False) -> Any:
    """중첩 mapping을 dataclass 트리로 바꾼다. 모르는 키, 빠진 키, 틀린 자료형은 SpecError다.

    strict_float=True이면 float 자리에 정수를 받지 않는다. 해시 대상 JSON에서 25와 25.0이
    섞이면 같은 상수의 해시가 달라지기 때문이다.
    """
    if not isinstance(data, Mapping):
        raise SpecError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(str(key) for key in data if key not in names)
    if unknown:
        raise SpecError(f"{where}: unknown keys {unknown}")
    missing = [name for name in names if name not in data]
    if missing:
        raise SpecError(f"{where}: missing keys {missing}")
    kwargs = {name: _convert(hints[name], data[name], f"{where}.{name}", strict_float) for name in names}
    return cls(**kwargs)


def _convert(tp: Any, value: Any, where: str, strict_float: bool) -> Any:
    # bool은 int의 하위형이라 type()으로 정확히 가린다.
    if tp is bool:
        if type(value) is not bool:
            raise SpecError(f"{where}: expected bool, got {value!r}")
        return value
    if tp is int:
        if type(value) is not int:
            raise SpecError(f"{where}: expected int, got {value!r}")
        return value
    if tp is float:
        if type(value) is int and not strict_float:
            value = float(value)
        if type(value) is int:
            raise SpecError(f"{where}: expected float, got integer {value!r} (write {float(value)!r})")
        if type(value) is not float:
            raise SpecError(f"{where}: expected float, got {value!r}")
        if not math.isfinite(value):
            raise SpecError(f"{where}: expected a finite float, got {value!r}")
        return value
    if tp is str:
        if type(value) is not str:
            raise SpecError(f"{where}: expected str, got {value!r}")
        return value
    if tp is Path:
        if type(value) is not str:
            raise SpecError(f"{where}: expected a path string, got {value!r}")
        return Path(value)
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is _UNION_TYPE:
        inner = [arg for arg in typing.get_args(tp) if arg is not type(None)]
        if len(inner) != 1:
            raise TypeError(f"{where}: unsupported union type {tp!r}")
        if value is None:
            return None
        return _convert(inner[0], value, where, strict_float)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise SpecError(f"{where}: expected a list, got {value!r}")
        elem = typing.get_args(tp)[0]
        return tuple(_convert(elem, item, f"{where}[{i}]", strict_float) for i, item in enumerate(value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise SpecError(f"{where}: expected a mapping, got {value!r}")
        key_tp, val_tp = typing.get_args(tp)
        result = {}
        for key, item in value.items():
            converted_key = _convert(key_tp, key, f"{where} key", strict_float)
            result[converted_key] = _convert(val_tp, item, f"{where}[{key!r}]", strict_float)
        return result
    if dataclasses.is_dataclass(tp):
        return build_dataclass(tp, value, where, strict_float)
    raise TypeError(f"{where}: unsupported field type {tp!r}")


def to_plain(obj: Any) -> Any:
    """dataclass 트리를 JSON으로 쓸 수 있는 dict/list로 바꾼다. 키 순서는 필드 선언 순서다."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_plain(item) for item in obj]
    if isinstance(obj, Mapping):
        return {key: to_plain(item) for key, item in obj.items()}
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SpecError(message)


# ---------------------------------------------------------------------------
# 해시
# ---------------------------------------------------------------------------


def compute_pipeline_hash(pipeline_block: Mapping[str, Any]) -> str:
    """`pipeline` 블록의 정규 JSON(키 정렬, 구분자 ",", ":", ASCII 이스케이프)의 sha256."""
    canonical = json.dumps(pipeline_block, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """파일 바이트의 sha256."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# PipelineSpec, FeatureSpec 로드와 검증
# ---------------------------------------------------------------------------


def _validate_clock(clock: ClockSpec, where: str) -> None:
    _check(clock.step_ms > 0, f"{where}.step_ms must be > 0")
    _check(0 < clock.max_hold_ms < clock.reset_gap_ms, f"{where}: need 0 < max_hold_ms < reset_gap_ms")


def _validate_run_detector(det: RunDetectorSpec, clock: ClockSpec, where: str) -> None:
    """감김 검출기 설정 하나를 검증한다. `clock`은 그 검출기와 함께 쓰이는 시간 규칙이다(같은 파일의 `clock` 블록)."""
    _check(0.0 < det.exit_thr < det.enter_thr <= 1.0, f"{where}: need 0 < exit_thr < enter_thr <= 1")
    _check(det.min_duration_ms >= 0, f"{where}.min_duration_ms must be >= 0")
    _check(det.max_dt_ms > 0, f"{where}.max_dt_ms must be > 0")
    _check(det.hold_ms >= 0, f"{where}.hold_ms must be >= 0")
    # hold_ms를 넘는 끊김으로 구간이 잘리면 그 끊김을 건너는 쌍은 속도와 보간에도 쓸 수 없어야 한다.
    # 검출기는 "쓸 수 없는 쌍"에서만 감는 속도의 거슬러 보기를 끊으므로, 이 조건이 없으면 잘린 구간의
    # 증가율이 바로 이어 시작한 구간의 감는 속도로 넘어갈 수 있다(drowsy/blink.py의 호출 계약).
    _check(det.max_dt_ms <= det.hold_ms, f"{where}: need max_dt_ms <= hold_ms")
    # 초기화를 일으키는 끊김(reset_gap_ms 초과)은 언제나 진행 중인 감김을 자르는 끊김(hold_ms 초과)이기도 해야 한다.
    # hold_ms가 reset_gap_ms 이상이면 끊김 앞의 감김과 뒤의 감김을 가르는 것이 호출자의 `reset()` 호출 하나뿐이
    # 되고, 그 호출을 빠뜨린 이식 코드는 두 감김을 하나로 이어도 벡터로는 드러나지 않는다.
    _check(
        det.hold_ms < clock.reset_gap_ms,
        f"{where}: need hold_ms < clock.reset_gap_ms (hold_ms is {det.hold_ms}, reset_gap_ms is {clock.reset_gap_ms})",
    )
    _check(det.velocity_lookback_samples >= 1, f"{where}.velocity_lookback_samples must be >= 1")


def _validate_pipeline(p: PipelineSpec) -> None:
    lm = p.landmarks
    _check(len(lm.ear_r) == 6 and len(lm.ear_l) == 6, "pipeline.landmarks: EAR sets need 6 points each")
    _check(len(lm.iris_r) == 5 and len(lm.iris_l) == 5, "pipeline.landmarks: iris sets need 5 points each")
    for name, contour in (("contour_r", lm.contour_r), ("contour_l", lm.contour_l)):
        _check(len(contour) == 16 and len(set(contour)) == 16, f"pipeline.landmarks.{name}: need 16 distinct points")
    _check(set(lm.ear_r) <= set(lm.contour_r), "pipeline.landmarks: ear_r must be a subset of contour_r")
    _check(set(lm.ear_l) <= set(lm.contour_l), "pipeline.landmarks: ear_l must be a subset of contour_l")
    # EAR의 p1, p4가 눈 꼬리다. 시선 대용치와 눈 폭이 같은 두 점을 쓴다.
    _check(
        {lm.corner_outer_r, lm.corner_inner_r} == {lm.ear_r[0], lm.ear_r[3]},
        "pipeline.landmarks: right eye corners must be p1 and p4 of ear_r",
    )
    _check(
        {lm.corner_outer_l, lm.corner_inner_l} == {lm.ear_l[0], lm.ear_l[3]},
        "pipeline.landmarks: left eye corners must be p1 and p4 of ear_l",
    )
    _check(len(lm.face_width) == 2, "pipeline.landmarks.face_width: need 2 points")
    required = lm.required_indices()
    _check(len(required) == 44, f"pipeline.landmarks: expected 44 distinct indices, got {len(required)}")
    _check(required[0] >= 0 and required[-1] <= 477, "pipeline.landmarks: indices must be within 0..477")

    bs = p.blendshapes
    _check(len(bs.order) == 52 and len(set(bs.order)) == 52, "pipeline.blendshapes.order: need 52 distinct names")
    _check(len(bs.eye_look) == 8, "pipeline.blendshapes.eye_look: need 8 names")
    names = bs.required_names()
    _check(len(set(names)) == 11, "pipeline.blendshapes: the 11 used names must be distinct")
    _check(set(names) <= set(bs.order), "pipeline.blendshapes: used names must appear in order")

    clock = p.clock
    _validate_clock(clock, "pipeline.clock")

    _check(p.signals.min_eye_width_px > 0.0, "pipeline.signals.min_eye_width_px must be > 0")

    base = p.baseline
    _check(0 < base.duration_ms <= base.max_duration_ms, "pipeline.baseline: need 0 < duration_ms <= max_duration_ms")
    _check(base.extend_step_ms > 0, "pipeline.baseline.extend_step_ms must be > 0")
    # T_b가 스텝 경계에 놓여야 "T_b 이후에 시작하는 스텝"이 정확히 정해진다.
    _check(
        base.duration_ms % clock.step_ms == 0 and base.extend_step_ms % clock.step_ms == 0,
        "pipeline.baseline: duration_ms and extend_step_ms must be multiples of clock.step_ms",
    )
    _check(
        (base.max_duration_ms - base.duration_ms) % base.extend_step_ms == 0,
        "pipeline.baseline: max_duration_ms must be duration_ms plus a whole number of extensions",
    )
    _check(base.min_face_frames >= 1, "pipeline.baseline.min_face_frames must be >= 1")
    _check(0.0 <= base.min_face_ratio <= 1.0, "pipeline.baseline.min_face_ratio must be within 0..1")
    _check(0.0 < base.e_min < base.e_max, "pipeline.baseline: need 0 < e_min < e_max")
    quantiles = (base.ear_quantile, base.blink_quantile) + base.blink_record_quantiles
    _check(all(0.0 < q <= 1.0 for q in quantiles), "pipeline.baseline: quantiles must be within (0, 1]")
    _check(base.off_axis_yaw_deg > 0.0, "pipeline.baseline.off_axis_yaw_deg must be > 0")

    clo = p.closure
    _check(0.0 < clo.rho < 1.0, "pipeline.closure.rho must be within (0, 1)")
    _check(0.0 < clo.b_closed <= 1.0, "pipeline.closure.b_closed must be within (0, 1]")
    _check(clo.b_span_min > 0.0, "pipeline.closure.b_span_min must be > 0")
    _check(0.0 <= clo.w <= 1.0, "pipeline.closure.w must be within 0..1")
    _check(
        0.0 < clo.far_eye_ramp_deg <= clo.far_eye_zero_yaw_deg,
        "pipeline.closure: need 0 < far_eye_ramp_deg <= far_eye_zero_yaw_deg",
    )
    _check(clo.valid_max_abs_yaw_deg > 0.0, "pipeline.closure.valid_max_abs_yaw_deg must be > 0")

    _validate_run_detector(p.run_detector, clock, "pipeline.run_detector")

    sf = p.step_features
    _check(sf.names == FEATURE_NAMES, "pipeline.step_features.names must equal drowsy.types.FEATURE_NAMES")
    _check(sf.window_steps >= 1, "pipeline.step_features.window_steps must be >= 1")
    _check(
        0 < sf.lookback_short_ms < sf.lookback_long_ms,
        "pipeline.step_features: need 0 < lookback_short_ms < lookback_long_ms",
    )
    _check(
        sf.lookback_short_ms % clock.step_ms == 0 and sf.lookback_long_ms % clock.step_ms == 0,
        "pipeline.step_features: windows must be multiples of clock.step_ms",
    )
    _check(0.0 < sf.perclos_thr <= 1.0, "pipeline.step_features.perclos_thr must be within (0, 1]")
    for name in (
        "perclos_min_valid_ms_60",
        "perclos_min_valid_ms_10",
        "blink_rate_min_valid_ms",
        "blink_duration_cap_ms",
        "long_closure_min_ms",
        "max_closure_cap_ms",
        "pitch_std_min_face_ms",
    ):
        _check(getattr(sf, name) > 0, f"pipeline.step_features.{name} must be > 0")

    _check(0.0 < p.yawn.exit_thr < p.yawn.enter_thr, "pipeline.yawn: need 0 < exit_thr < enter_thr")
    _check(p.yawn.count_at_ms > 0 and p.yawn.cancel_gap_ms > 0, "pipeline.yawn: times must be > 0")
    _check(0.0 < p.nod.exit_deg < p.nod.enter_deg, "pipeline.nod: need 0 < exit_deg < enter_deg")
    _check(
        0 < p.nod.min_duration_ms <= p.nod.max_duration_ms,
        "pipeline.nod: need 0 < min_duration_ms <= max_duration_ms",
    )
    _check(
        p.nod.ema_tau_ms > 0 and p.nod.ema_max_dt_ms > 0 and p.nod.cancel_gap_ms > 0,
        "pipeline.nod: times must be > 0",
    )
    # 초기화를 일으키는 끊김(reset_gap_ms 초과)은 언제나 진행 중인 하품·끄덕임 판정을 취소하는 끊김(cancel_gap_ms 초과)
    # 이기도 해야 한다(감김 검출기의 hold_ms와 같은 까닭이다). cancel_gap_ms가 reset_gap_ms 이상이면 끊김 앞의 하품과
    # 뒤의 하품을 가르는 것이 `StepAggregator`가 끊김 뒤에 부르는 계수기의 `reset()` 호출 하나뿐이 되고, 그 호출을
    # 빠뜨린 이식 코드는 두 하품을 하나로 이어도 벡터로는 드러나지 않는다.
    _check(
        p.yawn.cancel_gap_ms < clock.reset_gap_ms,
        "pipeline.yawn: need cancel_gap_ms < clock.reset_gap_ms "
        f"(cancel_gap_ms is {p.yawn.cancel_gap_ms}, reset_gap_ms is {clock.reset_gap_ms})",
    )
    _check(
        p.nod.cancel_gap_ms < clock.reset_gap_ms,
        "pipeline.nod: need cancel_gap_ms < clock.reset_gap_ms "
        f"(cancel_gap_ms is {p.nod.cancel_gap_ms}, reset_gap_ms is {clock.reset_gap_ms})",
    )
    # nod.ema_max_dt_ms에는 같은 조건을 두지 않는다. 느린 평균의 dt 상한은 초기화와 얽히지 않기 때문이다: 끊김 뒤의
    # 초기화에서 느린 평균을 버리는 것은 이 값이 얼마든(기본 200이든 reset_gap_ms 이상이든) 드러나는 동작이고
    # (벡터 steps_nod_after_long_gap이 고정한다), 초기화의 나머지 문장은 이 값이 얼마든 드러나지 않는다. 초기화 뒤의
    # 첫 표본은 dt를 쓰지 않고 느린 평균의 초기값이 된다. 2026-11-14에 이 값을 200, 9999, 10000, 20000으로 두고
    # 초기화의 문장을 하나씩 뺀 구현과 301개 세션에서 비교해 확인했다(네 값 모두 같은 문장만 차이를 냈다).

    glob = p.global_constants
    _check(base.e_min <= glob.e <= base.e_max, "pipeline.global_constants.e must be within baseline e_min..e_max")
    _check(0.0 <= glob.b < clo.b_closed, "pipeline.global_constants.b must be within [0, closure.b_closed)")
    _check(len(glob.r0) == 9, "pipeline.global_constants.r0: need 9 values (row-major 3x3)")

    sha = p.mediapipe.model_sha256
    _check(
        len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha),
        "pipeline.mediapipe.model_sha256 must be 64 lowercase hex characters",
    )
    _check(p.mediapipe.num_faces >= 1, "pipeline.mediapipe.num_faces must be >= 1")


def _validate_feature_spec(spec: FeatureSpec) -> None:
    pipe = spec.pipeline
    _check(spec.spec_version >= 0, "spec_version must be >= 0")
    _check(spec.step_sec * 1000 == pipe.clock.step_ms, "step_sec must match pipeline.clock.step_ms")
    _check(
        spec.window_steps == pipe.step_features.window_steps,
        "window_steps must match pipeline.step_features.window_steps",
    )
    _check(spec.class_order == CLASS_ORDER, f"class_order must be {list(CLASS_ORDER)}")
    _check(len(spec.features) == N_FEATURES, f"features: expected {N_FEATURES} entries, got {len(spec.features)}")

    std = spec.standardization
    _check(std.location == STANDARDIZATION_IN_GRAPH, f"standardization.location must be {STANDARDIZATION_IN_GRAPH!r}")
    _check(std.z_clip > 0.0, "standardization.z_clip must be > 0")
    _check(std.std_floor > 0.0, "standardization.std_floor must be > 0")
    _check(std.temperature > 0.0, "standardization.temperature must be > 0")

    windows_sec = (pipe.step_features.lookback_long_ms // 1000, pipe.step_features.lookback_short_ms // 1000)
    for i, feat in enumerate(spec.features):
        where = f"features[{i}]"
        _check(feat.index == i, f"{where}.index must be {i}")
        _check(feat.name == FEATURE_NAMES[i], f"{where}.name must be {FEATURE_NAMES[i]!r}, got {feat.name!r}")
        _check(feat.window_sec in windows_sec, f"{where}.window_sec must be one of {windows_sec}")
        _check(feat.name.endswith(f"_{feat.window_sec}"), f"{where}: name suffix must match window_sec")
        _check(feat.transform in TRANSFORMS, f"{where}.transform must be one of {TRANSFORMS}")
        _check(feat.std >= std.std_floor, f"{where}.std must be >= standardization.std_floor")


def pipeline_spec_from_dict(block: Mapping[str, Any]) -> PipelineSpec:
    """`pipeline` 블록 dict를 검증해 PipelineSpec으로 만든다."""
    spec = build_dataclass(PipelineSpec, block, "pipeline", strict_float=True)
    _validate_pipeline(spec)
    return spec


def pipeline_to_dict(spec: PipelineSpec) -> dict[str, Any]:
    """PipelineSpec을 `pipeline` 블록 dict로 되돌린다. `pipeline_spec_from_dict`의 역이다."""
    return to_plain(spec)


def feature_spec_from_dict(data: Mapping[str, Any]) -> FeatureSpec:
    """feature_spec.json의 dict를 검증해 FeatureSpec으로 만든다. pipeline_hash가 맞지 않으면 SpecError다."""
    spec = build_dataclass(FeatureSpec, data, "feature_spec", strict_float=True)
    _validate_pipeline(spec.pipeline)
    _validate_feature_spec(spec)
    expected = compute_pipeline_hash(data["pipeline"])
    if spec.pipeline_hash != expected:
        raise SpecError(
            f"pipeline_hash mismatch: file has {spec.pipeline_hash!r}, pipeline block hashes to {expected!r}; "
            "run `python -m drowsy.spec update-hash` if the change is intended"
        )
    return spec


def feature_spec_to_dict(spec: FeatureSpec) -> dict[str, Any]:
    """FeatureSpec을 feature_spec.json의 dict로 되돌린다."""
    return to_plain(spec)


def read_feature_spec_dict(path: str | Path) -> dict[str, Any]:
    """feature_spec.json을 검증 없이 dict로 읽는다(평균·표준편차를 고쳐 다시 쓸 때 쓴다)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SpecError(f"{path}: expected a JSON object")
    return data


def load_feature_spec(path: str | Path) -> FeatureSpec:
    """feature_spec.json을 읽어 검증한다."""
    return feature_spec_from_dict(read_feature_spec_dict(path))


def write_feature_spec(spec_dict: Mapping[str, Any], path: str | Path) -> str:
    """feature_spec dict를 검증한 뒤 고정된 형식(들여쓰기 2, UTF-8, 끝 줄바꿈)으로 쓴다.

    쓴 파일 바이트의 sha256(= feature_spec_hash)을 돌려준다.
    """
    feature_spec_from_dict(spec_dict)
    payload = (json.dumps(spec_dict, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 바이너리로 써서 플랫폼의 줄바꿈 변환을 막는다(파일 바이트가 해시 대상).
    out.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# RulesConfig 로드, 검증, 내보내기
# ---------------------------------------------------------------------------


def _validate_rules(cfg: RulesConfig) -> None:
    _validate_clock(cfg.clock, "clock")

    fz = cfg.forward_zone
    _check(fz.yaw_abs_max_deg > 0.0, "forward_zone.yaw_abs_max_deg must be > 0")
    _check(fz.pitch_up_max_deg > 0.0 and fz.pitch_down_max_deg > 0.0, "forward_zone pitch limits must be > 0")
    _check(fz.gaze_h_abs_max > 0.0 and fz.gaze_v_abs_max > 0.0, "forward_zone gaze limits must be > 0")
    _check(fz.debounce_ms >= 0 and fz.face_loss_credit_ms >= 0, "forward_zone times must be >= 0")

    _check(cfg.gaze_away_long.onset_ms > 0, "gaze_away_long.onset_ms must be > 0")
    _check(0 < cfg.vats.cumulative_ms <= cfg.vats.window_ms, "vats: need 0 < cumulative_ms <= window_ms")
    _check(cfg.vats.reset_forward_ms > 0, "vats.reset_forward_ms must be > 0")

    _validate_run_detector(cfg.closure.detector, cfg.clock, "closure.detector")
    _check(0 < cfg.closure.microsleep_ms <= cfg.closure.sleep_ms, "closure: need 0 < microsleep_ms <= sleep_ms")

    drowsy = cfg.drowsiness
    _check(drowsy.source in DROWSY_SOURCES, f"drowsiness.source must be one of {DROWSY_SOURCES}")
    _check(0.0 <= drowsy.model.end_p < drowsy.model.onset_p <= 1.0, "drowsiness.model: need 0 <= end_p < onset_p <= 1")
    _check(
        0.0 < drowsy.perclos.end_ratio < drowsy.perclos.onset_ratio <= 1.0,
        "drowsiness.perclos: need 0 < end_ratio < onset_ratio <= 1",
    )
    for name, block in (("model", drowsy.model), ("perclos", drowsy.perclos)):
        _check(block.onset_steps >= 1 and block.end_steps >= 1, f"drowsiness.{name}: step counts must be >= 1")
        _check(block.refractory_ms >= 0, f"drowsiness.{name}.refractory_ms must be >= 0")

    _check(cfg.unable_to_analyze.window_ms > 0, "unable_to_analyze.window_ms must be > 0")
    _check(0.0 < cfg.unable_to_analyze.min_face_ratio <= 1.0, "unable_to_analyze.min_face_ratio must be within (0, 1]")

    for event_type in EVENT_TYPES:
        policy = cfg.events.get(event_type)
        _check(0 <= policy.severity <= 3, f"events.{event_type}.severity must be within 0..3")
        _check(policy.cooldown_ms >= 0 and policy.repeat_ms >= 0, f"events.{event_type}: times must be >= 0")
        _check(policy.min_speed_kmh >= 0.0, f"events.{event_type}.min_speed_kmh must be >= 0")

    _check(
        sorted(cfg.alert_priority) == sorted(EVENT_TYPES),
        "alert_priority must list every event type exactly once",
    )


def rules_from_dict(data: Mapping[str, Any]) -> RulesConfig:
    """규칙 설정 dict(rules.yaml 또는 rules.json의 내용)를 검증해 RulesConfig로 만든다."""
    cfg = build_dataclass(RulesConfig, data, "rules", strict_float=False)
    _validate_rules(cfg)
    return cfg


def rules_to_dict(cfg: RulesConfig) -> dict[str, Any]:
    """RulesConfig를 dict로 되돌린다. `rules_from_dict`의 역이다."""
    return to_plain(cfg)


def load_rules(path: str | Path) -> RulesConfig:
    """규칙 설정을 읽는다. 확장자가 .json이면 내보낸 rules.json으로, 아니면 YAML로 읽는다."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml  # 지연 import: 이 모듈을 import하는 이식 대상 코드에 yaml이 따라오지 않게 한다.

        data = yaml.safe_load(text)
    return rules_from_dict(data)


def rules_to_json(cfg: RulesConfig) -> str:
    """앱에 내보내는 rules.json의 내용. 같은 설정이면 항상 같은 문자열이다."""
    return json.dumps(rules_to_dict(cfg), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def rules_hash(cfg: RulesConfig) -> str:
    """`export_rules_json`이 쓸 파일 바이트의 sha256. 파일을 쓰지 않고 해시만 필요할 때 쓴다."""
    return hashlib.sha256(rules_to_json(cfg).encode("utf-8")).hexdigest()


def export_rules_json(cfg: RulesConfig, out_path: str | Path) -> str:
    """rules.json을 쓰고 rules_hash(쓴 파일 바이트의 sha256)를 돌려준다."""
    payload = rules_to_json(cfg).encode("utf-8")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def apply_overrides(obj: Any, overrides: Mapping[str, Any]) -> Any:
    """점으로 이은 경로로 값을 바꾼 사본을 만든다(테스트 벡터의 config_overrides와 실험용).

    obj는 RulesConfig 또는 PipelineSpec이다. 예: `{"vats.reset_enabled": True}`.
    바꾼 결과도 같은 검증을 거친다. 없는 경로는 SpecError다.
    """
    plain = to_plain(obj)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = plain
        for depth, part in enumerate(parts):
            if not isinstance(node, dict) or part not in node:
                raise SpecError(f"override path not found: {dotted!r}")
            if depth == len(parts) - 1:
                node[part] = value
            else:
                node = node[part]
    if isinstance(obj, RulesConfig):
        return rules_from_dict(plain)
    if isinstance(obj, PipelineSpec):
        return pipeline_spec_from_dict(plain)
    raise TypeError(f"apply_overrides supports RulesConfig and PipelineSpec, got {type(obj).__name__}")


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------


def _cmd_check(spec_path: str) -> int:
    raw = read_feature_spec_dict(spec_path)
    stored = raw.get("pipeline_hash")
    computed = compute_pipeline_hash(raw.get("pipeline", {}))
    if stored != computed:
        print(f"pipeline_hash MISMATCH in {spec_path}", file=sys.stderr)
        print(f"  file:     {stored}", file=sys.stderr)
        print(f"  computed: {computed}", file=sys.stderr)
        print("  run `python -m drowsy.spec update-hash` if the pipeline change is intended", file=sys.stderr)
        return 1
    try:
        feature_spec_from_dict(raw)
    except SpecError as error:
        print(f"invalid feature spec {spec_path}: {error}", file=sys.stderr)
        return 1
    print(f"pipeline_hash OK: {computed}")
    print(f"feature_spec_hash: {file_sha256(spec_path)}")
    return 0


def _cmd_update_hash(spec_path: str) -> int:
    raw = read_feature_spec_dict(spec_path)
    old = raw.get("pipeline_hash")
    new = compute_pipeline_hash(raw.get("pipeline", {}))
    raw["pipeline_hash"] = new
    try:
        file_hash = write_feature_spec(raw, spec_path)
    except SpecError as error:
        print(f"invalid feature spec {spec_path}: {error}", file=sys.stderr)
        return 1
    if old == new:
        print(f"pipeline_hash unchanged: {new}")
    else:
        print(f"pipeline_hash updated: {old} -> {new}")
    print(f"feature_spec_hash: {file_hash}")
    return 0


def _cmd_export_rules(rules_path: str, out_path: str) -> int:
    try:
        cfg = load_rules(rules_path)
    except SpecError as error:
        print(f"invalid rules {rules_path}: {error}", file=sys.stderr)
        return 1
    digest = export_rules_json(cfg, out_path)
    print(f"wrote {out_path}")
    print(f"rules_hash: {digest}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m drowsy.spec", description="feature_spec and rules utilities")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="verify that pipeline_hash matches the pipeline block")
    check.add_argument("--spec", default=DEFAULT_FEATURE_SPEC_PATH)

    update = commands.add_parser("update-hash", help="recompute pipeline_hash and rewrite the file")
    update.add_argument("--spec", default=DEFAULT_FEATURE_SPEC_PATH)

    export = commands.add_parser("export-rules", help="write rules.json for the app and print rules_hash")
    export.add_argument("--rules", default=DEFAULT_RULES_PATH)
    export.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "check":
        return _cmd_check(args.spec)
    if args.command == "update-hash":
        return _cmd_update_hash(args.spec)
    return _cmd_export_rules(args.rules, args.out)


if __name__ == "__main__":
    sys.exit(main())
