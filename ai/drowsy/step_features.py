"""하품·끄덕임 검출과 1초 스텝 특징 (이식 대상, 표준 라이브러리만 사용). PLAN 3.3절, 4.4절, 4.6절.

- `ThresholdEventCounter`: 히스테리시스 임계 이벤트 계수기. 하품과 끄덕임이 같이 쓴다.
- `NodDetector`: pitch_rel의 느린 평균을 빼서 `ThresholdEventCounter`에 넣는다.
- `StepAggregator`: 프레임마다 `update`를 받아 1초 스텝이 끝날 때마다 특징 17개(`StepFeatures`)를 낸다.

시간 규칙 (PLAN 3.3절). 아래에서 t는 세션 시간 = `state.t_ms − origin_ms`다.
- 한 프레임의 상태는 다음 프레임이 올 때까지 유지된 것으로 본다. 단 최대 `clock.max_hold_ms`(200 ms)까지다.
  넘는 부분은 "데이터 없음"이고 얼굴 없는 시간과 똑같이 취급한다(어느 누적값에도 들어가지 않는다).
- 그래서 시간 누적은 다음 프레임이 도착했을 때 한다. 유지된 구간이 스텝 경계에 걸치면 경계에서 정확히 나눈다.
- 스텝 k는 [1000(k−1), 1000k)를 대표한다. t ≥ 1000k인 첫 프레임이 스텝 k를 발행시키고, 그 프레임은
  스텝 k에 들어가지 않는다(정확히 1000k에 온 프레임은 스텝 k+1에 속한다).
- 프레임이 끊겼다가 오면 그 사이에 끝난 스텝을 오래된 것부터 모두 발행한다.
- 프레임 간격이 `clock.reset_gap_ms`(10초)를 넘으면 누적값과 하품·끄덕임 검출기를 초기화한다.

특징 (PLAN 4.6절 표, 순서는 `drowsy.types.FEATURE_NAMES`)
- 스텝마다 누적값을 담는 통(bucket)을 하나씩 두고, 스텝이 끝날 때 최근 60개(60초 구간)와 최근 10개
  (10초 구간)를 오래된 것부터 더해서 특징을 만든다. 더한 값을 들고 다니지 않으므로 오차가 쌓이지 않는다.
- 감김 구간(`ClosureRun`)과 하품·끄덕임 이벤트는 그것을 발행한 프레임이 속한 스텝의 통에 넣는다.
- 감김의 길이(#5, #6, #9~#11)는 `ClosureRun.duration_ms`와 `ongoing_ms`를 그대로 쓴다. 둘 다 closure가 진입
  임계(0.5) 이상이었던 시간의 합이다(2026-11-13 소유자 결정, `drowsy.blink`). 눈꺼풀이 이탈과 진입 사이(0.3~0.5)에
  머문 시간은 구간을 끝내지 않지만 길이에도 들어가지 않는다. `end_ms − start_ms`는 어디에도 쓰지 않는다.
  그래서 깜빡이고 반쯤 뜬 채로 몇 초 있다가 다시 깜빡이면 이벤트는 하나(#4)이고 길이는 두 깜빡임의 합이다.
- 0으로 나누게 되는 비율은 0, 분산은 제곱근 전에 0 이상으로 자른다. 입력이 유한하면 출력도 모두 유한하다.

계획서가 정하지 않아서 정한 것
가정: 스텝 k의 특징은 t < 1000k인 프레임까지의 정보만 쓴다. "진행 중인 감김"(#9~#11)은 스텝을 발행시킨
  프레임이 아니라 그 직전 프레임에서 받은 `ongoing_ms`다. 발행시킨 프레임의 값을 쓰면, 그 프레임에서 끝난
  긴 감김이 스텝 k에서는 진행 중도 아니고 귀속되지도 않아(귀속은 스텝 k+1) 한 스텝 동안 사라진다.
  프레임이 끊겼다가 와서 프레임 하나가 스텝을 여럿 발행할 때(10초 이하의 끊김)도 같은 규칙이다(검토 K5).
  그 스텝들은 모두 끊기기 직전 마지막 프레임의 `ongoing_ms`를 쓴다. 끊긴 동안 감김이 어떻게 됐는지는 알 수
  없으므로 늘리지도 0으로 되돌리지도 않는다(0으로 두면 감김이 끊긴 동안 사라졌다가 끊김 뒤에 잘린 구간으로
  다시 나타난다). 눈을 감은 채 끊겼을 때의 결과는 다음과 같다. 동작은 바꾸지 않았고 문서로만 남긴다.
  · 끊긴 동안의 스텝마다 그 감김이 진행 중인 감김으로 들어간다(long_closure_count_60, max_closure_ms_60,
    max_closure_ms_10). 끊김 뒤 첫 프레임에서 검출기가 그 감김을 잘린 구간으로 내면 그 프레임이 속한 스텝의
    통에 들어가고 진행 중인 값은 0이 되므로 두 번 세지는 않는다. 다만 max_closure_ms_10에는 10초가 아니라
    끊긴 시간 + 10초 동안 남는다.
  · 끊긴 동안의 스텝에서는 10초 구간의 유효 시간이 줄어드는데 감긴 시간은 그대로 구간에 남아 perclos_10이
    올라간다. 유효 시간이 `perclos_min_valid_ms_10`(2초) 아래로 내려가야 0이 된다.
  · 예(실행해서 확인한 값): 20.0초에 눈을 감고, 20.8초의 프레임 다음 프레임이 28.5초에 오면 스텝 21~28이
    모두 그 프레임에서 발행되고 그 스텝들의 max_closure_ms_60과 max_closure_ms_10은 812,
    long_closure_count_60은 1이다. 잘린 구간(812 ms)은 스텝 29에 들어가고 max_closure_ms_10은 스텝 38까지
    812다(스텝 39에서 0). perclos_10은 스텝 21의 0.100에서 스텝 29~30의 0.400까지 올랐다가 스텝 31에서
    0이 된다.
가정: 누적을 시작하기 전의 시간(start_ms 이전, 초기화 이전)은 "데이터 없음"으로 센다. 그래서 조회 구간이
  덜 찬 스텝의 face_missing_ratio는 1에 가깝고, 유효 시간이 모자란 동안 perclos 등은 정의 불가 값(0)이다.
가정: start_ms에는 "데이터 없음" 상태의 가상 프레임이 있다고 본다. 첫 프레임이 start_ms보다 늦게 오면 그
  사이에 끝난 스텝은 빈 스텝으로 발행하고, 10초 넘게 늦으면 아래의 초기화 규칙을 따른다.
가정: 10초 넘는 끊김 뒤에는 끊긴 동안의 스텝을 발행하지 않는다(끊기기 직전에 채우던 스텝도 버린다).
  새 프레임이 속한 스텝의 시작 경계를 새 누적 시작 시각으로 삼고 그 스텝부터 다시 발행한다.
  `full_lookback`도 새 누적 시작 시각 + 60초부터 다시 True가 된다. 스텝 번호 k는 세션 시간 그대로다.
가정: start_ms 이전의 프레임은 받아도 누적하지 않는다(빈 목록을 돌려준다).
가정: start_ms는 `clock.step_ms`의 배수여야 한다(T_b는 spec 검증으로 항상 배수다). 아니면 ValueError.
"""

from __future__ import annotations

import math

from drowsy.spec import NodSpec, PipelineSpec, YawnSpec
from drowsy.types import ClosureRun, FrameState, StepFeatures

_MS_PER_MINUTE = 60000.0


class ThresholdEventCounter:
    """히스테리시스 임계 이벤트 계수기 (PLAN 4.6절의 하품과 끄덕임 공용).

    값이 `enter_thr` 이상이면 이벤트가 시작되고 `exit_thr` 미만이면 끝난다. 길이는 프레임 시각의 차다.
    - `count_at_end=False`(하품): 진행 중인 이벤트의 길이가 `min_duration_ms`에 도달한 프레임에서 한 번 센다.
      `max_duration_ms`는 쓰지 않는다.
    - `count_at_end=True`(끄덕임): 끝나는 프레임에서 길이가 `min_duration_ms` 이상 `max_duration_ms` 이하이면 센다.
    무효 구간(마지막 유효 프레임 뒤로 흐른 시간)이 `cancel_gap_ms`를 넘으면 진행 중이던 판정을 취소한다.

    `update`는 프레임마다 한 번, 시각이 증가하는 순서로 부른다. 무효 프레임도 valid=False로 넘긴다.

    가정: 길이는 (지금 프레임 시각) − (enter 이상을 처음 본 프레임 시각)이다. 보간하지 않는다.
    가정: 하품은 값이 exit 이상으로 유지되는 프레임에서만 센다. exit 미만으로 내려간 프레임에서는
      길이가 기준을 넘었더라도 세지 않는다.
    가정: 무효 구간은 무효 프레임이 없이 프레임 자체가 끊긴 시간도 포함한다(PLAN 3.3절).
    """

    def __init__(
        self,
        enter_thr: float,
        exit_thr: float,
        min_duration_ms: int,
        max_duration_ms: int,
        count_at_end: bool,
        cancel_gap_ms: int,
    ) -> None:
        self._enter_thr = enter_thr
        self._exit_thr = exit_thr
        self._min_duration_ms = min_duration_ms
        self._max_duration_ms = max_duration_ms
        self._count_at_end = count_at_end
        self._cancel_gap_ms = cancel_gap_ms

        self._active = False
        self._start_ms = 0
        self._last_valid_ms = 0  # 진행 중일 때만 의미가 있다
        self._counted = False  # 진행 중인 이벤트를 이미 셌는지(count_at_end=False에서만 쓴다)

    @property
    def active(self) -> bool:
        """이벤트가 진행 중인지."""
        return self._active

    def reset(self) -> None:
        """처음 상태로 되돌린다. 진행 중이던 판정은 버린다."""
        self._active = False
        self._start_ms = 0
        self._last_valid_ms = 0
        self._counted = False

    def update(self, t_ms: int, value: float, valid: bool) -> bool:
        """프레임 하나를 처리한다. 이 프레임에서 이벤트를 하나 셌으면 True."""
        if self._active and t_ms - self._last_valid_ms > self._cancel_gap_ms:
            self._active = False
        if not valid:
            return False

        counted = False
        if not self._active:
            if value >= self._enter_thr:
                self._active = True
                self._start_ms = t_ms
                self._counted = False
        elif value < self._exit_thr:
            self._active = False
            if self._count_at_end:
                duration_ms = t_ms - self._start_ms
                counted = duration_ms >= self._min_duration_ms and duration_ms <= self._max_duration_ms
        elif not self._count_at_end and not self._counted:
            if t_ms - self._start_ms >= self._min_duration_ms:
                self._counted = True
                counted = True
        self._last_valid_ms = t_ms
        return counted


def yawn_counter(cfg: YawnSpec) -> ThresholdEventCounter:
    """하품 계수기. 입력값은 jaw_rel = jaw_open − jaw0이다.

    0.40 이상으로 시작, 0.25 미만으로 끝, 길이가 2000 ms에 도달하면 한 번 센다(잠정값).
    """
    return ThresholdEventCounter(
        enter_thr=cfg.enter_thr,
        exit_thr=cfg.exit_thr,
        min_duration_ms=cfg.count_at_ms,
        max_duration_ms=cfg.count_at_ms,
        count_at_end=False,
        cancel_gap_ms=cfg.cancel_gap_ms,
    )


class NodDetector:
    """끄덕임 검출기 (PLAN 4.6절, 잠정값).

    d = pitch_rel − m이 10도 이상으로 시작, 5도 미만으로 끝, 길이가 200~3000 ms면 끝날 때 센다.
    m은 pitch_rel의 느린 평균이다. 유효 프레임마다 d를 먼저 계산한 뒤
    m += α·(pitch_rel − m), α = dt / (ema_tau_ms + dt)로 갱신한다. m의 초기값은 첫 유효 표본이고
    (그 프레임의 d는 0이다) 무효 프레임에서는 갱신하지 않는다.

    가정: dt는 직전 유효 표본과의 간격이고 `ema_max_dt_ms`(200 ms)를 넘으면 그 값으로 자른다.
    """

    def __init__(self, cfg: NodSpec) -> None:
        self._tau_ms = cfg.ema_tau_ms
        self._max_dt_ms = cfg.ema_max_dt_ms
        self._counter = ThresholdEventCounter(
            enter_thr=cfg.enter_deg,
            exit_thr=cfg.exit_deg,
            min_duration_ms=cfg.min_duration_ms,
            max_duration_ms=cfg.max_duration_ms,
            count_at_end=True,
            cancel_gap_ms=cfg.cancel_gap_ms,
        )
        self._has_mean = False
        self._mean = 0.0
        self._prev_valid_ms = 0

    @property
    def mean(self) -> float:
        """느린 평균 m. 유효 표본을 아직 받지 않았으면 0.0이다."""
        return self._mean

    def reset(self) -> None:
        """처음 상태로 되돌린다. 느린 평균도 버린다."""
        self._counter.reset()
        self._has_mean = False
        self._mean = 0.0
        self._prev_valid_ms = 0

    def update(self, t_ms: int, pitch_rel: float, valid: bool) -> bool:
        """프레임 하나를 처리한다. 이 프레임에서 끄덕임을 하나 셌으면 True."""
        if not valid:
            return self._counter.update(t_ms, 0.0, False)

        d = 0.0
        if self._has_mean:
            d = pitch_rel - self._mean
            dt_ms = t_ms - self._prev_valid_ms
            if dt_ms > self._max_dt_ms:
                dt_ms = self._max_dt_ms
            alpha = float(dt_ms) / float(self._tau_ms + dt_ms)
            self._mean += alpha * (pitch_rel - self._mean)
        else:
            self._has_mean = True
            self._mean = pitch_rel
        self._prev_valid_ms = t_ms
        return self._counter.update(t_ms, d, True)


class _Bucket:
    """스텝 하나(1초)의 누적값. 시간은 정수 ms로 더한다."""

    __slots__ = (
        "valid_ms",
        "closed_ms",
        "closure_sum",
        "face_ms",
        "pitch_sum",
        "pitch_sq_sum",
        "n_events",
        "dur_sum",
        "dur_sq_sum",
        "n_velocity",
        "closing_vel_sum",
        "opening_vel_sum",
        "n_long",
        "max_run_ms",
        "n_yawn",
        "n_nod",
    )

    def __init__(self) -> None:
        self.valid_ms = 0  # closure가 유효한 시간
        self.closed_ms = 0  # 그중 closure ≥ perclos_thr인 시간
        self.closure_sum = 0.0  # Σ closure·ms (유효 시간에 대해)
        self.face_ms = 0  # 얼굴이 있는 시간
        self.pitch_sum = 0.0  # Σ pitch_rel·ms (얼굴 시간에 대해)
        self.pitch_sq_sum = 0.0  # Σ pitch_rel²·ms
        self.n_events = 0  # 잘리지 않은 감김 이벤트 수
        self.dur_sum = 0  # Σ min(duration_ms, 상한)
        self.dur_sq_sum = 0  # Σ min(duration_ms, 상한)²
        self.n_velocity = 0  # 그중 속도가 유효한 이벤트 수
        self.closing_vel_sum = 0.0
        self.opening_vel_sum = 0.0
        self.n_long = 0  # 길이 ≥ long_closure_min_ms인 감김 수(잘린 것 포함)
        self.max_run_ms = 0  # 귀속된 감김 중 가장 긴 것의 길이(잘린 것 포함)
        self.n_yawn = 0
        self.n_nod = 0


class StepAggregator:
    """프레임 상태를 1초 스텝 특징으로 누적한다 (PLAN 3.3절, 4.4절, 4.6절).

    - `start_ms`: 누적을 시작하는 세션 시간 T_b. 첫 스텝은 [start_ms, start_ms + 1000)이다.
    - `origin_ms`: 처음 받아들인 프레임의 `t_ms`. 세션 시간 = `state.t_ms − origin_ms`.

    `update(state, run, ongoing_ms)`를 받아들인 프레임마다, 특징용 감김 검출기를 갱신한 뒤에 부른다.
    - `run`: 그 프레임에서 특징용 검출기가 발행한 구간(없으면 None).
    - `ongoing_ms`: 그 프레임을 처리한 뒤의 `ClosureRunDetector.ongoing_ms()`.
    이 프레임이 발행시킨 스텝을 오래된 것부터 돌려준다. 시간 규칙과 가정은 모듈 설명에 있다.
    """

    def __init__(self, spec: PipelineSpec, start_ms: int, origin_ms: int) -> None:
        self._step_ms = spec.clock.step_ms
        if start_ms < 0 or start_ms % self._step_ms != 0:
            raise ValueError("start_ms must be a non-negative multiple of clock.step_ms")
        self._max_hold_ms = spec.clock.max_hold_ms
        self._reset_gap_ms = spec.clock.reset_gap_ms

        sf = spec.step_features
        self._lookback_long_ms = sf.lookback_long_ms
        self._lookback_short_ms = sf.lookback_short_ms
        self._n_long = sf.lookback_long_ms // self._step_ms
        self._n_short = sf.lookback_short_ms // self._step_ms
        self._perclos_thr = sf.perclos_thr
        self._perclos_min_valid_ms_60 = sf.perclos_min_valid_ms_60
        self._perclos_min_valid_ms_10 = sf.perclos_min_valid_ms_10
        self._blink_rate_min_valid_ms = sf.blink_rate_min_valid_ms
        self._blink_duration_cap_ms = sf.blink_duration_cap_ms
        self._long_closure_min_ms = sf.long_closure_min_ms
        self._max_closure_cap_ms = sf.max_closure_cap_ms
        self._pitch_std_min_face_ms = sf.pitch_std_min_face_ms

        self._origin_ms = origin_ms
        self._start_ms = start_ms
        self._accum_start_ms = start_ms  # 지금 누적을 시작한 세션 시간(초기화하면 뒤로 옮겨진다)

        self._cur_k = start_ms // self._step_ms + 1  # 지금 채우고 있는 스텝 번호
        self._cur = _Bucket()
        self._history: list[_Bucket] = []  # 끝난 스텝의 통, 오래된 것부터 최대 n_long개

        # 직전 프레임(세션 시간). 프레임을 받기 전에는 start_ms에 "데이터 없음" 프레임이 있다고 본다.
        self._has_prev = False
        self._prev_t = start_ms
        self._prev_face = False
        self._prev_valid = False
        self._prev_closure = 0.0
        self._prev_pitch = 0.0
        self._prev_ongoing_ms = 0

        self._yawn = yawn_counter(spec.yawn)
        self._nod = NodDetector(spec.nod)

    def update(self, state: FrameState, run: ClosureRun | None, ongoing_ms: int) -> list[StepFeatures]:
        """프레임 하나를 누적한다. 이 프레임이 발행시킨 스텝을 오래된 것부터 돌려준다."""
        t = state.t_ms - self._origin_ms
        if t < self._start_ms:
            return []
        if self._has_prev and t <= self._prev_t:
            raise ValueError("t_ms must be strictly greater than the previous frame's t_ms")

        steps: list[StepFeatures] = []
        if t - self._prev_t > self._reset_gap_ms:
            self._restart(t)
        else:
            self._advance_to(t, steps)

        # 이 프레임에서 생긴 이벤트는 이 프레임이 속한 스텝(지금 채우는 통)에 넣는다.
        if run is not None:
            self._add_run(run)
        if self._yawn.update(t, state.jaw_rel, state.face):
            self._cur.n_yawn += 1
        if self._nod.update(t, state.pitch_rel, state.face):
            self._cur.n_nod += 1

        self._has_prev = True
        self._prev_t = t
        self._prev_face = state.face
        self._prev_valid = state.closure_valid
        self._prev_closure = state.closure
        self._prev_pitch = state.pitch_rel
        self._prev_ongoing_ms = ongoing_ms
        return steps

    def _restart(self, t: int) -> None:
        """10초 넘는 끊김 뒤의 초기화. t가 속한 스텝부터 다시 누적한다."""
        self._cur_k = t // self._step_ms + 1
        self._accum_start_ms = (self._cur_k - 1) * self._step_ms
        self._cur = _Bucket()
        self._history = []
        self._has_prev = False
        self._prev_ongoing_ms = 0
        self._yawn.reset()
        self._nod.reset()

    def _advance_to(self, t: int, steps: list[StepFeatures]) -> None:
        """직전 프레임의 상태가 유지된 구간을 누적하면서, t까지 끝난 스텝을 발행한다."""
        pos = self._prev_t
        hold_end = self._prev_t
        if self._has_prev:
            held_ms = t - self._prev_t
            if held_ms > self._max_hold_ms:
                held_ms = self._max_hold_ms
            hold_end = self._prev_t + held_ms

        while self._cur_k * self._step_ms <= t:
            boundary = self._cur_k * self._step_ms
            if pos < hold_end:
                piece_end = hold_end
                if piece_end > boundary:
                    piece_end = boundary
                self._add_time(piece_end - pos)
                pos = piece_end
            steps.append(self._emit())
        if pos < hold_end:
            self._add_time(hold_end - pos)

    def _add_time(self, ms: int) -> None:
        """직전 프레임의 상태를 ms만큼 지금 통에 더한다."""
        bucket = self._cur
        if self._prev_face:
            bucket.face_ms += ms
            bucket.pitch_sum += self._prev_pitch * ms
            bucket.pitch_sq_sum += self._prev_pitch * self._prev_pitch * ms
        if self._prev_valid:
            bucket.valid_ms += ms
            bucket.closure_sum += self._prev_closure * ms
            if self._prev_closure >= self._perclos_thr:
                bucket.closed_ms += ms

    def _add_run(self, run: ClosureRun) -> None:
        bucket = self._cur
        if not run.truncated:
            duration_ms = run.duration_ms
            if duration_ms > self._blink_duration_cap_ms:
                duration_ms = self._blink_duration_cap_ms
            bucket.n_events += 1
            bucket.dur_sum += duration_ms
            bucket.dur_sq_sum += duration_ms * duration_ms
            if run.velocity_valid:
                bucket.n_velocity += 1
                bucket.closing_vel_sum += run.peak_closing_velocity
                bucket.opening_vel_sum += run.peak_opening_velocity
        if run.duration_ms >= self._long_closure_min_ms:
            bucket.n_long += 1
        if run.duration_ms > bucket.max_run_ms:
            bucket.max_run_ms = run.duration_ms

    def _emit(self) -> StepFeatures:
        """지금 통을 닫고 그 스텝의 특징을 만든다."""
        self._history.append(self._cur)
        if len(self._history) > self._n_long:
            del self._history[0]
        k = self._cur_k
        end_ms = k * self._step_ms
        step = StepFeatures(
            k=k,
            t_end_ms=self._origin_ms + end_ms,
            full_lookback=end_ms >= self._accum_start_ms + self._lookback_long_ms,
            values=self._features(),
        )
        self._cur = _Bucket()
        self._cur_k = k + 1
        return step

    def _features(self) -> tuple[float, ...]:
        """방금 끝난 스텝의 특징 17개. `_history`의 마지막 통이 그 스텝이다."""
        n = len(self._history)
        first_short = n - self._n_short
        if first_short < 0:
            first_short = 0

        valid_60 = 0
        closed_60 = 0
        closure_sum_60 = 0.0
        face_60 = 0
        pitch_sum = 0.0
        pitch_sq_sum = 0.0
        n_events = 0
        dur_sum = 0
        dur_sq_sum = 0
        n_velocity = 0
        closing_vel_sum = 0.0
        opening_vel_sum = 0.0
        n_long = 0
        max_run_60 = 0
        n_yawn = 0
        n_nod = 0
        valid_10 = 0
        closed_10 = 0
        closure_sum_10 = 0.0
        face_10 = 0
        max_run_10 = 0

        for i in range(n):
            b = self._history[i]
            valid_60 += b.valid_ms
            closed_60 += b.closed_ms
            closure_sum_60 += b.closure_sum
            face_60 += b.face_ms
            pitch_sum += b.pitch_sum
            pitch_sq_sum += b.pitch_sq_sum
            n_events += b.n_events
            dur_sum += b.dur_sum
            dur_sq_sum += b.dur_sq_sum
            n_velocity += b.n_velocity
            closing_vel_sum += b.closing_vel_sum
            opening_vel_sum += b.opening_vel_sum
            n_long += b.n_long
            if b.max_run_ms > max_run_60:
                max_run_60 = b.max_run_ms
            n_yawn += b.n_yawn
            n_nod += b.n_nod
            if i >= first_short:
                valid_10 += b.valid_ms
                closed_10 += b.closed_ms
                closure_sum_10 += b.closure_sum
                face_10 += b.face_ms
                if b.max_run_ms > max_run_10:
                    max_run_10 = b.max_run_ms

        # 진행 중인 감김: 이 스텝에 속한 마지막 프레임에서 받은 값이다.
        ongoing_ms = self._prev_ongoing_ms
        if ongoing_ms >= self._long_closure_min_ms:
            n_long += 1
        if ongoing_ms > max_run_60:
            max_run_60 = ongoing_ms
        if ongoing_ms > max_run_10:
            max_run_10 = ongoing_ms
        if max_run_60 > self._max_closure_cap_ms:
            max_run_60 = self._max_closure_cap_ms
        if max_run_10 > self._max_closure_cap_ms:
            max_run_10 = self._max_closure_cap_ms

        perclos_60 = 0.0
        if valid_60 >= self._perclos_min_valid_ms_60:
            perclos_60 = float(closed_60) / float(valid_60)
        perclos_10 = 0.0
        if valid_10 >= self._perclos_min_valid_ms_10:
            perclos_10 = float(closed_10) / float(valid_10)

        closure_mean_60 = 0.0
        if valid_60 > 0:
            closure_mean_60 = closure_sum_60 / float(valid_60)
        closure_mean_10 = 0.0
        if valid_10 > 0:
            closure_mean_10 = closure_sum_10 / float(valid_10)

        blink_rate_60 = 0.0
        if valid_60 >= self._blink_rate_min_valid_ms:
            blink_rate_60 = float(n_events) * _MS_PER_MINUTE / float(valid_60)

        blink_dur_mean_60 = 0.0
        blink_dur_std_60 = 0.0
        if n_events > 0:
            blink_dur_mean_60 = float(dur_sum) / float(n_events)
        if n_events >= 2:
            blink_dur_std_60 = _std_from_moments(blink_dur_mean_60, float(dur_sq_sum) / float(n_events))

        closing_vel_mean_60 = 0.0
        opening_vel_mean_60 = 0.0
        if n_velocity > 0:
            closing_vel_mean_60 = closing_vel_sum / float(n_velocity)
            opening_vel_mean_60 = opening_vel_sum / float(n_velocity)

        pitch_std_60 = 0.0
        if face_60 >= self._pitch_std_min_face_ms:
            pitch_std_60 = _std_from_moments(pitch_sum / float(face_60), pitch_sq_sum / float(face_60))

        # 통이 없는 시간(누적 시작 전)도 얼굴이 없던 시간으로 센다.
        face_missing_ratio_60 = float(self._lookback_long_ms - face_60) / float(self._lookback_long_ms)
        face_missing_ratio_10 = float(self._lookback_short_ms - face_10) / float(self._lookback_short_ms)

        return (
            perclos_60,
            perclos_10,
            closure_mean_60,
            closure_mean_10,
            blink_rate_60,
            blink_dur_mean_60,
            blink_dur_std_60,
            closing_vel_mean_60,
            opening_vel_mean_60,
            float(n_long),
            float(max_run_60),
            float(max_run_10),
            float(n_yawn),
            pitch_std_60,
            float(n_nod),
            face_missing_ratio_60,
            face_missing_ratio_10,
        )


def _std_from_moments(mean: float, mean_sq: float) -> float:
    """평균과 제곱 평균으로 구한 모표준편차. 분산은 제곱근 전에 0 이상으로 자른다(PLAN 3.3절)."""
    variance = mean_sq - mean * mean
    if variance < 0.0:
        variance = 0.0
    return math.sqrt(variance)
