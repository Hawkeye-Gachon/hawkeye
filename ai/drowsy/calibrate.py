"""기준선 추정과 closure 계산 (이식 대상, 표준 라이브러리만 사용).

PLAN 4.2절(기준선), 4.3절(closure), D4(기준선 완성 전의 임시 계산), D5(연장과 대체).

- `BaselineEstimator`: 세션 앞부분의 얼굴 프레임으로 개인 기준선을 한 번 만든다.
- `global_baseline`: 전역 상수로 만든 기준선(대체, 기준선 완성 전의 임시 계산, 전역 정규화 조건).
- `frame_state`: 원시 신호와 기준선으로 프레임 하나의 정규화 상태(closure, 상대 자세, 상대 시선·턱)를 만든다.
  상태를 갖지 않는다.

시간: 추정기는 처음 받은 프레임의 `t_ms`를 세션 원점으로 삼고 그 뒤로는 세션 시간(ms)으로 계산한다.
호출자(`DriverPipeline`)는 받아들인 프레임만, `t_ms`가 엄격히 증가하는 순서로 넘긴다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from drowsy import signals
from drowsy.spec import ClosureSpec, PipelineSpec
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    BASELINE_OFF_AXIS,
    BASELINE_OK,
    Baseline,
    FrameSignals,
    FrameState,
)

# ---------------------------------------------------------------------------
# 분위수와 중앙값
# ---------------------------------------------------------------------------


def nearest_rank_quantile(values: Sequence[float], q: float) -> float:
    """최근접 순위 분위수 `sorted[ceil(q·n) − 1]` (PLAN 4.2절).

    값을 보간하지 않으므로 결과는 항상 표본 중 하나다. q·n은 배정밀도 곱 그대로 올림한다
    (Kotlin의 `ceil(q * n)`과 같은 값). spec의 분위수 0.1, 0.5, 0.9는 q·n이 정수일 때 곱도 정확히 그 정수가
    되어 순위가 밀리지 않는다(n ≤ 5400에서 테스트로 확인).
    가정: 순위는 1 이상 n 이하로 자른다(q = 0이면 최솟값).
    """
    n = len(values)
    if n == 0:
        raise ValueError("quantile of an empty sequence is undefined")
    ordered = sorted(values)
    rank = int(math.ceil(q * n))
    if rank < 1:
        rank = 1
    if rank > n:
        rank = n
    return ordered[rank - 1]


def median(values: Sequence[float]) -> float:
    """중앙값. 표본이 짝수 개면 가운데 두 값의 평균이다 (PLAN 4.2절).

    최근접 순위를 쓰지 않는 이유: 좌우 반전 입력(yaw와 roll의 부호가 뒤집힌다)에서
    median(−x) = −median(x)가 정확히 성립해야 기준 회전 R0이 정확히 대칭이 된다.
    """
    n = len(values)
    if n == 0:
        raise ValueError("median of an empty sequence is undefined")
    ordered = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# ---------------------------------------------------------------------------
# 기준선
# ---------------------------------------------------------------------------


def global_baseline(spec: PipelineSpec) -> Baseline:
    """전역 상수로 만든 기준선 (PLAN 4.3절).

    두 눈에 같은 E, B를 쓰고 R0은 단위 행렬이다. status는 fallback_global, t_complete_ms는 0이다.
    기준선 완성 전의 임시 closure(D4)와 전역 정규화 조건이 이 값을 쓴다.
    """
    return _baseline_from_constants(spec, 0)


def _baseline_from_constants(spec: PipelineSpec, t_complete_ms: int) -> Baseline:
    constants = spec.global_constants
    return Baseline(
        e_l=constants.e,
        e_r=constants.e,
        b_l=constants.b,
        b_r=constants.b,
        r0=constants.r0,
        jaw0=constants.jaw0,
        gaze_h0=constants.gaze_h0,
        gaze_v0=constants.gaze_v0,
        t_complete_ms=t_complete_ms,
        status=BASELINE_FALLBACK_GLOBAL,
    )


@dataclass(frozen=True, slots=True)
class BaselineDiagnostics:
    """기준선을 확정한 시점의 판정 통계와 분포 기록. `Baseline`에는 넣지 않는 값이다.

    - n_face_frames: 수집 구간의 얼굴 프레임 수.
    - face_ratio: 얼굴 검출 비율(시간 기준) = 얼굴 프레임이 대표한 시간 / 수집 구간 길이.
    - blink_l_quantiles, blink_r_quantiles: `eyeBlink`의 기록용 분위수.
      순서는 `spec.baseline.blink_record_quantiles`(50, 90 분위수)와 같다. 얼굴 프레임이 없으면 0.0이다.
    """

    n_face_frames: int
    face_ratio: float
    blink_l_quantiles: tuple[float, ...]
    blink_r_quantiles: tuple[float, ...]


class BaselineEstimator:
    """세션 앞 60초의 얼굴 프레임으로 기준선을 만든다 (PLAN 4.2절, D5).

    사용법: 받아들인 프레임마다 `update(signals)`를 부른다. 기준선이 확정되는 프레임에서 `Baseline`을
    한 번 돌려주고, 그 전과 그 뒤에는 None을 돌려준다. 확정 뒤에는 `diagnostics`가 채워진다.

    수집 구간은 세션 시간 [0, 마감)이다. 마감은 60000 ms에서 시작한다.
    세션 시간이 마감 이상인 첫 프레임이 오면 그 프레임을 넣기 전에 완성 조건을 검사한다.
      1. 얼굴 프레임 수 ≥ min_face_frames(300)
      2. 얼굴 검출 비율(시간 기준) ≥ min_face_ratio(0.5)
      3. 두 눈 모두 e_min(0.12) ≤ E ≤ e_max(0.45)
    모두 만족하면 status ok(|yaw0| > 15도면 off_axis)로 확정하고 `t_complete_ms`(T_b)는 마감 값이다.
    하나라도 어기면 마감을 30000 ms 늦추고(최대 180000 ms) 수집을 계속한다. 180000 ms에서도 어기면
    전역 상수로 대체한다(status fallback_global, t_complete_ms = 180000).

    통계: E는 EAR의 90 분위수, B는 `eyeBlink`의 10 분위수, R0은 yaw·pitch·roll 중앙값으로 만든 회전,
    jaw0·gaze_h0·gaze_v0은 중앙값이다. 모두 수집 구간의 얼굴 프레임에서 구한다.

    가정: 얼굴 검출 비율(시간 기준)은 PLAN 3.3절의 시간 규칙으로 센다. 얼굴 프레임은 다음 프레임까지의
      간격만큼, 최대 `clock.max_hold_ms`(200 ms)까지의 시간을 대표하고, 마감을 넘는 부분은 세지 않는다.
      분모는 마감 값(데이터 없는 시간을 포함한 수집 구간 전체)이다.
    가정: E의 표본에는 쓸 수 있는 눈(눈 폭 ≥ 6 px)의 EAR만 넣는다. 쓸 수 없는 눈의 ear 0.0은 측정값이 아니다.
      한쪽 눈이라도 쓸 수 있는 표본이 하나도 없으면 그 눈의 E를 정할 수 없으므로 조건 3을 어긴 것으로 본다.
    가정: 마감 시각에 정확히 온 프레임은 수집 구간에 넣지 않는다(스텝 경계 규칙과 같다. PLAN 4.4절).
    가정: 프레임이 한참 끊겼다가 와서 마감을 여러 번 지나면 지난 마감을 순서대로 하나씩 검사한다.
      T_b는 조건을 만족한 첫 마감이다.
    가정: 전역 상수로 대체한 기준선의 `t_complete_ms`는 대체를 결정한 마감(180000)이다. 특징 누적은 그때 시작한다.
    가정: `t_ms`가 직전 프레임 이하이면 ValueError를 낸다. 그런 프레임은 호출자가 버리고 넘기지 않는다.
    """

    def __init__(self, spec: PipelineSpec) -> None:
        self._spec = spec
        self._cfg = spec.baseline
        self._max_hold_ms = spec.clock.max_hold_ms
        self._min_eye_width_px = spec.signals.min_eye_width_px

        self._done = False
        self._deadline_ms = self._cfg.duration_ms

        # 직전 프레임(세션 시간). 얼굴 프레임이 대표하는 시간을 다음 프레임이 올 때 확정하려고 둔다.
        self._has_prev = False
        self._origin_ms = 0
        self._prev_t = 0
        self._prev_face = False
        self._face_ms = 0

        self._ear_l: list[float] = []
        self._ear_r: list[float] = []
        self._blink_l: list[float] = []
        self._blink_r: list[float] = []
        self._yaw: list[float] = []
        self._pitch: list[float] = []
        self._roll: list[float] = []
        self._jaw: list[float] = []
        self._gaze_h: list[float] = []
        self._gaze_v: list[float] = []

        self.diagnostics: BaselineDiagnostics | None = None

    def update(self, s: FrameSignals) -> Baseline | None:
        """프레임 하나를 받는다. 이 프레임에서 기준선이 확정되면 `Baseline`을, 아니면 None을 돌려준다."""
        if self._done:
            return None

        if not self._has_prev:
            self._origin_ms = s.t_ms
        t = s.t_ms - self._origin_ms
        if self._has_prev and t <= self._prev_t:
            raise ValueError("t_ms must be strictly greater than the previous frame's t_ms")

        # 직전 프레임이 얼굴 프레임이면 이 프레임까지의 간격을 최대 max_hold_ms까지 대표한다.
        held_ms = 0
        if self._has_prev and self._prev_face:
            held_ms = t - self._prev_t
            if held_ms > self._max_hold_ms:
                held_ms = self._max_hold_ms

        while t >= self._deadline_ms:
            held_before_deadline = self._deadline_ms - self._prev_t
            if held_before_deadline > held_ms:
                held_before_deadline = held_ms
            baseline = self._try_complete(self._face_ms + held_before_deadline)
            if baseline is not None:
                self._done = True
                return baseline
            next_deadline_ms = self._deadline_ms + self._cfg.extend_step_ms
            if next_deadline_ms > self._cfg.max_duration_ms:
                next_deadline_ms = self._cfg.max_duration_ms
            self._deadline_ms = next_deadline_ms

        self._face_ms += held_ms
        if s.face:
            self._collect(s)
        self._has_prev = True
        self._prev_t = t
        self._prev_face = s.face
        return None

    def _collect(self, s: FrameSignals) -> None:
        if s.eye_w_l >= self._min_eye_width_px:
            self._ear_l.append(s.ear_l)
        if s.eye_w_r >= self._min_eye_width_px:
            self._ear_r.append(s.ear_r)
        self._blink_l.append(s.blink_l)
        self._blink_r.append(s.blink_r)
        self._yaw.append(s.yaw)
        self._pitch.append(s.pitch)
        self._roll.append(s.roll)
        self._jaw.append(s.jaw_open)
        self._gaze_h.append(signals.gaze_h(s.eye_look))
        self._gaze_v.append(signals.gaze_v(s.eye_look))

    def _try_complete(self, face_ms: int) -> Baseline | None:
        """지금 마감에서 완성 조건을 검사한다. 확정(완성 또는 대체)이면 기준선을, 연장이면 None을 돌려준다."""
        cfg = self._cfg
        deadline_ms = self._deadline_ms
        n_face = len(self._blink_l)
        face_ratio = face_ms / deadline_ms

        e_l = 0.0
        e_r = 0.0
        e_in_range = False
        if len(self._ear_l) > 0 and len(self._ear_r) > 0:
            e_l = nearest_rank_quantile(self._ear_l, cfg.ear_quantile)
            e_r = nearest_rank_quantile(self._ear_r, cfg.ear_quantile)
            e_in_range = e_l >= cfg.e_min and e_l <= cfg.e_max and e_r >= cfg.e_min and e_r <= cfg.e_max

        complete = n_face >= cfg.min_face_frames and face_ratio >= cfg.min_face_ratio and e_in_range
        if not complete and deadline_ms < cfg.max_duration_ms:
            return None

        self.diagnostics = BaselineDiagnostics(
            n_face_frames=n_face,
            face_ratio=face_ratio,
            blink_l_quantiles=self._record_quantiles(self._blink_l),
            blink_r_quantiles=self._record_quantiles(self._blink_r),
        )
        if not complete:
            return _baseline_from_constants(self._spec, deadline_ms)

        yaw0 = median(self._yaw)
        pitch0 = median(self._pitch)
        roll0 = median(self._roll)
        status = BASELINE_OK
        if abs(yaw0) > cfg.off_axis_yaw_deg:
            status = BASELINE_OFF_AXIS
        return Baseline(
            e_l=e_l,
            e_r=e_r,
            b_l=nearest_rank_quantile(self._blink_l, cfg.blink_quantile),
            b_r=nearest_rank_quantile(self._blink_r, cfg.blink_quantile),
            r0=signals.rotation_from_euler(yaw0, pitch0, roll0),
            jaw0=median(self._jaw),
            gaze_h0=median(self._gaze_h),
            gaze_v0=median(self._gaze_v),
            t_complete_ms=deadline_ms,
            status=status,
        )

    def _record_quantiles(self, values: Sequence[float]) -> tuple[float, ...]:
        out: list[float] = []
        for q in self._cfg.blink_record_quantiles:
            if len(values) > 0:
                out.append(nearest_rank_quantile(values, q))
            else:
                out.append(0.0)
        return tuple(out)


# ---------------------------------------------------------------------------
# closure와 프레임 상태
# ---------------------------------------------------------------------------


def eye_closure(ear: float, blink: float, e: float, b: float, cfg: ClosureSpec) -> float:
    """눈 하나의 감김 정도 c_e (0~1, PLAN 4.3절).

    a_e = clip((E − ear) / (E·(1 − rho)), 0, 1)             뜬 눈 기준 대비 얼마나 좁아졌나
    s_e = clip((blink − B) / max(b_closed − B, b_span_min), 0, 1)   뜬 눈 수준에서 감긴 수준까지 얼마나 갔나
    c_e = w·a_e + (1 − w)·s_e

    가정: E·(1 − rho)가 0 이하이면 a_e = 0으로 둔다(PLAN 3.3절: 0으로 나누게 되는 비율은 0).
    """
    a = 0.0
    ear_span = e * (1.0 - cfg.rho)
    if ear_span > 0.0:
        a = _clip01((e - ear) / ear_span)
    blink_span = cfg.b_closed - b
    if blink_span < cfg.b_span_min:
        blink_span = cfg.b_span_min
    s = _clip01((blink - b) / blink_span)
    return cfg.w * a + (1.0 - cfg.w) * s


def frame_state(s: FrameSignals, b: Baseline, spec: PipelineSpec, provisional: bool) -> FrameState:
    """프레임 하나의 정규화 상태 (PLAN 4.3절).

    (yaw_rel, pitch_rel, roll_rel) = Euler(R0ᵀ·R)
    w_far   = clip((25 − |yaw_rel|) / 10, 0, 1)      먼 쪽 눈: yaw_rel > 0이면 L, 아니면 R
    closure = (c_near + w_far·c_far) / (1 + w_far)
    쓸 수 있는 눈(눈 폭 ≥ 6 px)이 하나뿐이면 그 눈의 c_e를 그대로 쓴다(먼 쪽 눈 가중을 하지 않는다).
    유효 조건: 얼굴 있음, 쓸 수 있는 눈이 하나 이상, |yaw_rel| ≤ 40도.
    유효하지 않으면 closure = 0.0, closure_valid = False다. NaN을 쓰지 않고 이전 값을 잇지도 않는다.

    - 얼굴이 있으면 yaw_rel, pitch_rel, gaze_h_rel, gaze_v_rel, jaw_rel을 항상 채운다.
    - pose_valid = 얼굴 있음이고 기준선 status가 fallback_global이 아님(시선 규칙의 사용 조건).
    - provisional은 받은 값을 그대로 적는다(기준선 완성 전이면 호출자가 True를 준다).

    좌우 반전 입력에서는 yaw_rel의 부호와 눈 집합이 함께 바뀌므로 closure가 같다.

    가정: `FrameSignals`에는 변환 행렬이 없으므로 R은 (yaw, pitch, roll)로 다시 만든다.
    가정: yaw_rel이 정확히 0이면 R을 먼 쪽으로 친다. 이때 w_far = 1이라 결과는 두 눈 평균으로 같다.
    """
    if not s.face:
        return FrameState(
            t_ms=s.t_ms,
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

    rotation = signals.rotation_from_euler(s.yaw, s.pitch, s.roll)
    relative = signals.euler_from_rotation(signals.relative_rotation(rotation, b.r0))
    yaw_rel = relative[0]
    pitch_rel = relative[1]
    abs_yaw_rel = abs(yaw_rel)

    cfg = spec.closure
    usable_l = s.eye_w_l >= spec.signals.min_eye_width_px
    usable_r = s.eye_w_r >= spec.signals.min_eye_width_px

    closure = 0.0
    closure_valid = False
    if (usable_l or usable_r) and abs_yaw_rel <= cfg.valid_max_abs_yaw_deg:
        closure_valid = True
        if usable_l and usable_r:
            c_l = eye_closure(s.ear_l, s.blink_l, b.e_l, b.b_l, cfg)
            c_r = eye_closure(s.ear_r, s.blink_r, b.e_r, b.b_r, cfg)
            w_far = _clip01((cfg.far_eye_zero_yaw_deg - abs_yaw_rel) / cfg.far_eye_ramp_deg)
            if yaw_rel > 0.0:
                c_far = c_l
                c_near = c_r
            else:
                c_far = c_r
                c_near = c_l
            closure = (c_near + w_far * c_far) / (1.0 + w_far)
        elif usable_l:
            closure = eye_closure(s.ear_l, s.blink_l, b.e_l, b.b_l, cfg)
        else:
            closure = eye_closure(s.ear_r, s.blink_r, b.e_r, b.b_r, cfg)

    return FrameState(
        t_ms=s.t_ms,
        face=True,
        closure=closure,
        closure_valid=closure_valid,
        provisional=provisional,
        yaw_rel=yaw_rel,
        pitch_rel=pitch_rel,
        pose_valid=b.status != BASELINE_FALLBACK_GLOBAL,
        gaze_h_rel=signals.gaze_h(s.eye_look) - b.gaze_h0,
        gaze_v_rel=signals.gaze_v(s.eye_look) - b.gaze_v0,
        jaw_rel=s.jaw_open - b.jaw0,
    )


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
