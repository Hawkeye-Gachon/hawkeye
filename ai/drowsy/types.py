"""파이프라인이 주고받는 값 자료형과 문자열 상수 (이식 대상, 표준 라이브러리만 사용).

공통 규약
- 시간: 모든 시각과 길이는 정수 ms다. `t_ms`, `start_ms`, `end_ms`, `t_start_ms`, `t_end_ms`는
  앱이 MediaPipe에 넘긴 타임스탬프와 같은 시계(원래 t_ms)로 적는다. 세션 시간
  (= t_ms − 처음 받아들인 프레임의 t_ms, PLAN 3.3절)으로 적는 값은 `Baseline.t_complete_ms`(T_b)와
  스텝 번호 `StepFeatures.k` 둘뿐이다.
  가정: 계획서는 이벤트 시각만 원래 t_ms로 적으라고 했고 나머지 자료형의 시계는 정하지 않았다.
  시계를 바꾸는 지점을 없애려고 모든 자료형을 원래 t_ms로 통일한다. 세션 원점(처음 받아들인 프레임의
  t_ms)이 필요한 계산(기준선 경과 시간, 스텝 경계와 k)은 그 값을 따로 받아서 쓴다.
- 실수: 모두 배정밀도. NaN과 무한대를 쓰지 않는다. 값이 없으면 0.0을 넣고 옆의 bool 필드로 구분한다.
- 좌우: `_l`은 MediaPipe Left(L 집합 362.., `*Left` 블렌드셰이프), `_r`은 Right(R 집합 33.., `*Right`)다.
  반전하지 않은 영상에서 피험자 자신의 좌우다(PLAN 2.1절).
- 각도: 도(degree). 부호는 PLAN 2.2절(yaw 양수 = 피험자가 자기 왼쪽으로 돌림, pitch 양수 = 턱이 내려감,
  roll 양수 = 영상에서 반시계 방향).
- 모든 자료형은 불변이다. dict 필드(`RawFrame.lm`, `RawFrame.bs`)도 만든 뒤에는 고치지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 문자열 상수
# ---------------------------------------------------------------------------

# 이벤트 종류 (PLAN 8.2절)
EVENT_CALIBRATING = "calibrating"
EVENT_UNABLE_TO_ANALYZE = "unable_to_analyze"
EVENT_GAZE_AWAY_LONG = "gaze_away_long"
EVENT_VATS = "vats"
EVENT_DROWSY = "drowsy"
EVENT_MICROSLEEP = "microsleep"
EVENT_SLEEP = "sleep"
EVENT_TYPES: tuple[str, ...] = (
    EVENT_CALIBRATING,
    EVENT_UNABLE_TO_ANALYZE,
    EVENT_GAZE_AWAY_LONG,
    EVENT_VATS,
    EVENT_DROWSY,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
)

# 이벤트 단계 (Q5): 임계 시간에 도달한 순간 onset, 상태가 끝날 때 end
PHASE_ONSET = "onset"
PHASE_END = "end"
PHASES: tuple[str, ...] = (PHASE_ONSET, PHASE_END)

# 이벤트 출처 (PLAN 8.2절의 source 열)
SOURCE_FACE = "face"
SOURCE_HEAD_POSE = "head_pose"
SOURCE_EYE = "eye"
SOURCE_MODEL = "model"
SOURCE_PERCLOS = "perclos"
SOURCE_CLOSURE = "closure"
SOURCES: tuple[str, ...] = (
    SOURCE_FACE,
    SOURCE_HEAD_POSE,
    SOURCE_EYE,
    SOURCE_MODEL,
    SOURCE_PERCLOS,
    SOURCE_CLOSURE,
)

# 규칙 설정 `drowsiness.source`의 값 (D11)
DROWSY_SOURCE_MODEL = "model"
DROWSY_SOURCE_PERCLOS = "perclos"
DROWSY_SOURCE_EITHER = "either"
DROWSY_SOURCES: tuple[str, ...] = (DROWSY_SOURCE_MODEL, DROWSY_SOURCE_PERCLOS, DROWSY_SOURCE_EITHER)

# 기준선 상태 (PLAN 4.2절)
# 가정: 계획서는 fallback_global과 off_axis만 이름을 정했다. 정상 완성은 "ok"로 적는다.
# 대체 상태의 R0은 단위 행렬(yaw0 = 0)이므로 off_axis와 fallback_global은 함께 나올 수 없고,
# status는 이 셋 중 하나의 문자열이다. 연장 여부는 t_complete_ms로 안다.
BASELINE_OK = "ok"
BASELINE_OFF_AXIS = "off_axis"
BASELINE_FALLBACK_GLOBAL = "fallback_global"
BASELINE_STATUSES: tuple[str, ...] = (BASELINE_OK, BASELINE_OFF_AXIS, BASELINE_FALLBACK_GLOBAL)

# 모델 출력 클래스 순서 (PLAN 7장). probs[CLASS_DROWSY]가 p_drowsy다.
CLASS_ORDER: tuple[str, ...] = ("alert", "low_vigilant", "drowsy")
CLASS_ALERT = 0
CLASS_LOW_VIGILANT = 1
CLASS_DROWSY = 2

# 스텝 특징의 고정 순서 (PLAN 4.6절). feature_spec.json의 features와 같아야 하며 spec 로더가 검사한다.
FEATURE_NAMES: tuple[str, ...] = (
    "perclos_60",
    "perclos_10",
    "closure_mean_60",
    "closure_mean_10",
    "blink_rate_60",
    "blink_dur_mean_60",
    "blink_dur_std_60",
    "closing_vel_mean_60",
    "opening_vel_mean_60",
    "long_closure_count_60",
    "max_closure_ms_60",
    "max_closure_ms_10",
    "yawn_count_60",
    "pitch_std_60",
    "nod_count_60",
    "face_missing_ratio_60",
    "face_missing_ratio_10",
)
N_FEATURES = 17
F_PERCLOS_60 = 0
F_PERCLOS_10 = 1
F_CLOSURE_MEAN_60 = 2
F_CLOSURE_MEAN_10 = 3
F_BLINK_RATE_60 = 4
F_BLINK_DUR_MEAN_60 = 5
F_BLINK_DUR_STD_60 = 6
F_CLOSING_VEL_MEAN_60 = 7
F_OPENING_VEL_MEAN_60 = 8
F_LONG_CLOSURE_COUNT_60 = 9
F_MAX_CLOSURE_MS_60 = 10
F_MAX_CLOSURE_MS_10 = 11
F_YAWN_COUNT_60 = 12
F_PITCH_STD_60 = 13
F_NOD_COUNT_60 = 14
F_FACE_MISSING_RATIO_60 = 15
F_FACE_MISSING_RATIO_10 = 16

# FrameSignals.eye_look의 고정 순서와 위치
EYE_LOOK_ORDER: tuple[str, ...] = ("in_l", "out_l", "up_l", "down_l", "in_r", "out_r", "up_r", "down_r")
EL_IN_L = 0
EL_OUT_L = 1
EL_UP_L = 2
EL_DOWN_L = 3
EL_IN_R = 4
EL_OUT_R = 5
EL_UP_R = 6
EL_DOWN_R = 7

# 단위 회전(행 우선 9개). 전역 상수 기준선의 R0.
IDENTITY_R0: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 자료형
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawFrame:
    """앱이 MediaPipe FaceLandmarker에서 프레임 하나에 대해 받는 것 (PLAN 4.1절).

    전제(Q3): 똑바로 세운 비반전 영상, numFaces=1, 타임스탬프는 양수이고 엄격히 증가하는 실제 ms.

    - t_ms: 프레임 타임스탬프(ms).
    - face: 얼굴이 검출됐는지. MediaPipe 결과 리스트가 비었는지로만 판정한다.
    - w, h: MediaPipe에 넘긴 영상의 너비와 높이(픽셀). 정규화 좌표를 픽셀로 되돌릴 때 쓴다.
    - lm: 랜드마크 인덱스 -> 정규화 좌표 (x, y). x는 너비, y는 높이로 정규화된 값이다.
      `PipelineSpec.landmarks.required_indices()`의 44개(눈 윤곽 32, 홍채 10, 얼굴 폭 2)를 담는다.
      더 많은 인덱스가 들어 있어도 된다.
    - bs: 블렌드셰이프 category_name -> score. `PipelineSpec.blendshapes.required_names()`의 11개는
      반드시 있어야 한다(추출 parquet은 52개 전부를 담는다).
    - m: 얼굴 변환 행렬 4x4의 열 우선 16개 값. Python의 `M[i][j]`는 `m[4*j + i]`다(PLAN 2.2절).

    얼굴이 없으면(face=False) lm과 bs는 빈 dict, m은 빈 튜플이다.
    face=True이면 lm, bs, m이 모두 채워져 있어야 한다.
    """

    t_ms: int
    face: bool
    w: int
    h: int
    lm: dict[int, tuple[float, float]]
    bs: dict[str, float]
    m: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FrameSignals:
    """프레임 하나의 원시 신호 (`signals.compute`의 출력, PLAN 4.1절).

    - ear_l, ear_r: 픽셀 좌표 (x*w, y*h)에서 구한 2D EAR.
    - eye_w_l, eye_w_r: 눈 폭 |p1 − p4| (픽셀).
      이 값이 `PipelineSpec.signals.min_eye_width_px`(6 px) 미만인 눈은 "쓸 수 없는 눈"이다.
      별도 플래그는 없고 이 비교로 판정한다. 쓸 수 없는 눈의 ear는 0.0, 홍채 비율은 None이다.
    - blink_l, blink_r: `eyeBlinkLeft`, `eyeBlinkRight` 값 그대로.
    - eye_look: `eyeLook*` 8개. 순서는 EYE_LOOK_ORDER
      (in_l, out_l, up_l, down_l, in_r, out_r, up_r, down_r)로 고정이다.
    - jaw_open: `jawOpen` 값 그대로.
    - yaw, pitch, roll: 변환 행렬에서 구한 절대 오일러 각(도), R = Ry(yaw)·Rx(pitch)·Rz(roll).
    - iris_h_l, iris_h_r, iris_v_l, iris_v_r: 선택 항목인 홍채 비율(h = dot(d,a)/|a|², v = cross(a,d)/|a|²).
      계산하지 않았거나 계산할 수 없으면 None이다.

    얼굴이 없으면(face=False) 모든 실수 필드는 0.0, eye_look은 0.0 여덟 개, 홍채 비율은 None이다.
    """

    t_ms: int
    face: bool
    ear_l: float
    ear_r: float
    eye_w_l: float
    eye_w_r: float
    blink_l: float
    blink_r: float
    eye_look: tuple[float, ...]
    jaw_open: float
    yaw: float
    pitch: float
    roll: float
    iris_h_l: float | None = None
    iris_h_r: float | None = None
    iris_v_l: float | None = None
    iris_v_r: float | None = None


@dataclass(frozen=True, slots=True)
class Baseline:
    """개인 기준선 (PLAN 4.2절).

    - e_l, e_r: 각 눈 EAR의 90 분위수(뜬 눈 기준).
    - b_l, b_r: 각 눈 `eyeBlink`의 10 분위수(뜬 눈 기준).
    - r0: 기준 회전 3x3을 행 우선 9개 값으로 적은 것. `R0[i][j]`는 `r0[3*i + j]`다.
      yaw, pitch, roll 중앙값으로 Ry·Rx·Rz를 만든 것이고, 전역 상수 기준선에서는 단위 행렬이다.
    - jaw0, gaze_h0, gaze_v0: `jaw_open`, `gaze_h`, `gaze_v`의 중앙값.
    - t_complete_ms: 기준선이 완성된 세션 시간 T_b(ms). 정상이면 60000, 연장되면 90000~180000.
      가정: 보정 없이 전역 상수로 바로 만든 기준선(`calibrate.global_baseline`)은 0으로 둔다.
    - status: BASELINE_OK, BASELINE_OFF_AXIS, BASELINE_FALLBACK_GLOBAL 중 하나.
      전역 상수로 만든 기준선은 언제나 BASELINE_FALLBACK_GLOBAL이다(보정 실패로 대체한 것,
      기준선 완성 전의 임시 계산용, 전역 정규화 조건용 모두).

    가정: 분포 기록용 `eyeBlink` 50, 90 분위수는 계획서의 필드 목록에 없으므로 여기 두지 않는다.
    """

    e_l: float
    e_r: float
    b_l: float
    b_r: float
    r0: tuple[float, ...]
    jaw0: float
    gaze_h0: float
    gaze_v0: float
    t_complete_ms: int
    status: str


@dataclass(frozen=True, slots=True)
class FrameState:
    """기준선으로 정규화한 프레임 상태 (`calibrate.frame_state`의 출력, PLAN 4.3절).

    - t_ms: 원래 프레임 타임스탬프(RawFrame.t_ms와 같은 값).
    - closure: 0~1. closure_valid=False이면 0.0이고 의미가 없다(보간하거나 이전 값을 잇지 않는다).
    - closure_valid: 얼굴 있음, 쓸 수 있는 눈이 하나 이상, |yaw_rel| ≤ 40도일 때 True.
    - provisional: 기준선 완성 전에 전역 상수로 계산한 프레임이면 True(D4). 기준선이 완성되면
      (전역 상수로 대체된 경우 포함) False다. 규칙 엔진은 True인 동안 감김 이벤트에 provisional=true를 달고,
      False인 첫 프레임에서 calibrating을 끝낸다.
    - yaw_rel, pitch_rel: Euler(R0ᵀ·R)의 yaw와 pitch(도). face=True이면 항상 채운다.
      전역 상수 기준선에서는 R0이 단위 행렬이므로 절대 각도와 같다. closure의 눈 가중과
      스텝 특징(pitch_std, 끄덕임)은 pose_valid가 아니라 face만 보고 이 값을 쓴다.
    - pose_valid: 시선 규칙이 yaw_rel, pitch_rel, gaze_h_rel, gaze_v_rel을 써도 되는지.
      face=True이고 기준선에 개인 기준 회전이 있을 때(status가 fallback_global이 아닐 때)만 True다.
      가정: 계획서는 "대체 상태에서는 R0이 없으므로 시선 규칙을 끈다"(4.2절), "시선 규칙은 기준선 완성 후
      시작한다"(D4)고만 했다. 규칙 엔진은 FrameState만 받으므로 두 경우 모두 이 필드를 False로 해서 전한다.
      기준선 완성 전에는 전역 상수 기준선(status = fallback_global)으로 계산하므로 저절로 False가 된다.
    - gaze_h_rel, gaze_v_rel: 눈동자 합성값의 기준선 대비 변화(gaze_h − gaze_h0, gaze_v − gaze_v0).
    - jaw_rel: jaw_open − jaw0.

    얼굴이 없으면(face=False) closure_valid와 pose_valid는 False, 모든 실수 필드는 0.0이다.
    gaze_h_rel, gaze_v_rel, jaw_rel은 face=True이면 항상 채운다.
    """

    t_ms: int
    face: bool
    closure: float
    closure_valid: bool
    provisional: bool
    yaw_rel: float
    pitch_rel: float
    pose_valid: bool
    gaze_h_rel: float
    gaze_v_rel: float
    jaw_rel: float


@dataclass(frozen=True, slots=True)
class ClosureRun:
    """감김 구간 하나 (`ClosureRunDetector`가 구간이 끝날 때 발행, PLAN 4.5절).

    - start_ms: closure가 enter를 처음 위로 지난 시각(보간했으면 floor(x + 0.5)로 반올림한 값).
    - end_ms: 나가기 전 마지막으로 enter를 아래로 지난 시각. 잘린 구간은 closure가 enter 이상이었던
      마지막 유효 프레임의 시각.
    - duration_ms: 구간 안에서 closure가 enter 이상이었던 시간의 합(2026-11-13 소유자 결정). 구간 도중에 enter
      아래로 내려갔다가(exit 아래로는 내려가지 않고) 다시 올라왔으면 그 사이는 넣지 않는다. 그래서
      duration_ms ≤ end_ms − start_ms이고, 다시 올라온 적이 없는 구간에서는 둘이 같다.
      가정: enter 이상이던 토막마다 반올림된 두 시각(위로 지난 시각, 아래로 지난 시각)의 차를 구해 더한다.
    - amplitude: 구간 중 최대 closure.
    - peak_closing_velocity, peak_opening_velocity: closure 증가율과 감소율의 최댓값(1/s, 0 이상).
    - velocity_valid: 두 속도를 스텝 특징(#7, #8)에 써도 되는지. False이면 두 속도 필드는 0.0이다.
      가정: 감는 속도와 뜨는 속도가 둘 다 측정됐을 때만 True로 한다(플래그가 하나뿐이므로).
    - truncated: 유효 프레임이 끊겨서 닫힌 구간이면 True.
    """

    start_ms: int
    end_ms: int
    duration_ms: int
    amplitude: float
    peak_closing_velocity: float
    peak_opening_velocity: float
    velocity_valid: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class StepFeatures:
    """1초 스텝 하나의 특징 (PLAN 4.4절, 4.6절).

    - k: 세션 스텝 번호. 스텝 k는 세션 시간 [1000(k−1), 1000k) ms를 대표한다.
      T_b = 60000이면 처음 발행되는 스텝은 k = 61이다.
    - t_end_ms: 스텝 끝 시각을 원래 t_ms 시계로 적은 것 = (처음 받아들인 프레임의 t_ms) + 1000·k.
      졸음 이벤트의 t_start_ms("연속 구간의 첫 스텝 끝")는 이 값을 그대로 쓴다.
    - full_lookback: 60초 조회 구간이 다 찼으면(스텝 끝 ≥ T_b + 60초) True. 모델 창에는 True인 스텝만 들어간다.
    - values: FEATURE_NAMES 순서의 원시 특징 17개. 표준화도 log1p도 하지 않은 값이며 모두 유한하다.
    """

    k: int
    t_end_ms: int
    full_lookback: bool
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DriverStateEvent:
    """운전자 상태 이벤트 (PLAN 8.2절, 뒤 두 필드는 Q5).

    - type: EVENT_* 중 하나.
    - severity: 0~3. 규칙 설정의 종류별 값.
    - t_start_ms: 상태가 시작된 시각(원래 t_ms 시계). 종류별 정의는 8.2절 표.
    - duration_ms: t_start_ms부터 이 이벤트를 발행한 시점까지의 길이.
    - confidence: 규칙 이벤트는 해당 구간의 유효 시간 비율(0~1), 모델 졸음 이벤트는 p_drowsy.
    - source: SOURCE_* 중 하나.
    - phase: PHASE_ONSET(임계 시간 도달) 또는 PHASE_END(상태 종료).
    - provisional: 전역 상수로 계산한 closure로 판정한 이벤트면 True(D4).
    """

    type: str
    severity: int
    t_start_ms: int
    duration_ms: int
    confidence: float
    source: str
    phase: str
    provisional: bool


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """`DriverPipeline.update`가 프레임 하나에 대해 돌려주는 것.

    - state: 이 프레임의 상태.
    - runs: 이 프레임에서 끝난 특징용(진입 0.5 / 이탈 0.3) 감김 구간. 보통 비어 있다.
    - steps: 이 프레임이 발행시킨 스텝. 프레임이 끊겼다가 오면 여러 개일 수 있고 오래된 것부터 적는다.
    - step_probs: `steps`와 길이와 순서가 같다. 원소는 그 스텝에서 모델을 불렀으면 그 스텝까지의 창으로 얻은
      (alert, low_vigilant, drowsy) 확률, 부르지 않았으면(모델 없음, 창이 덜 참, 입력이 유한하지 않음) None이다.
      모델은 스텝마다 부르므로 스텝을 여러 개 발행한 프레임에서는 확률도 여러 개일 수 있다.
    - probs: 이 프레임이 스텝을 발행시켰으면 `step_probs`의 마지막 원소(가장 최근 스텝의 확률 또는 None),
      스텝이 없으면 None. 스텝을 하나만 발행시킨 프레임에서는 "이 프레임에서 모델을 불렀을 때만 확률"이다.
    - events: 이 프레임에서 발행된 상태 이벤트(발행 순서).
    - alerts: `AlertPolicy`가 이 프레임에서 실제로 경고하기로 한 이벤트. 이 프레임의 `onset`일 수도 있고,
      앞선 프레임의 `onset`을 미뤘다가 내는 사본이나 되풀이 경고의 사본일 수도 있다(`duration_ms`만 다르다).
    """

    state: FrameState
    runs: tuple[ClosureRun, ...]
    steps: tuple[StepFeatures, ...]
    step_probs: tuple[tuple[float, float, float] | None, ...]
    probs: tuple[float, float, float] | None
    events: tuple[DriverStateEvent, ...]
    alerts: tuple[DriverStateEvent, ...]
