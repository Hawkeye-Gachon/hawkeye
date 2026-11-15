"""스트리밍 감김 구간 검출기 `ClosureRunDetector` (이식 대상, 표준 라이브러리만 사용). PLAN 4.5절.

closure에 히스테리시스를 거는 상태 머신이다. 같은 클래스를 설정만 바꿔 두 번 쓴다.
스텝 특징용은 진입 0.5 / 이탈 0.3(`PipelineSpec.run_detector`), 규칙용은 진입 0.7 / 이탈 0.5
(`RulesConfig.closure.detector`)다. 임계값은 모두 설정에서 읽는다.

상태 전이 (프레임이 도착할 때만 판정한다. c = closure, 끊긴 시간 = 이 프레임 시각 − 마지막 유효 프레임 시각)

OPEN
- 유효, c ≥ enter          : CLOSED로. start_ms 기록
- 유효, exit ≤ c < enter   : 그대로
- 유효, c < exit           : 그대로
- 무효 프레임              : 그대로
CLOSED
- 유효, c ≥ enter          : 그대로. 정점 갱신. 직전 유효 표본이 enter 미만이었으면 새 토막을 연다(아래 "길이")
- 유효, exit ≤ c < enter   : 그대로. 직전 표본이 enter 이상이었으면 last_down_ms 기록
- 유효, c < exit           : 구간 발행, OPEN으로
- 무효 프레임              : HOLD로
HOLD
- 유효, c ≥ enter          : 끊긴 시간 ≤ hold_ms면 CLOSED로 복귀. 직전 유효 표본이 enter 미만이었으면 새 토막을 연다
- 유효, exit ≤ c < enter   : 같음(CLOSED로 복귀). 직전 유효 표본이 enter 이상이었으면 last_down_ms 기록
- 유효, c < exit           : 구간 발행(truncated=true), OPEN으로
- 무효 프레임              : 끊긴 시간 > hold_ms면 구간 발행(truncated=true), OPEN으로. 아니면 그대로

CLOSED나 HOLD에서 끊긴 시간이 hold_ms를 넘은 뒤 도착한 프레임은 먼저 구간을 닫고(truncated=true),
그 프레임을 OPEN 상태에서 다시 처리한다.

시각
- start_ms: closure가 enter를 (그 구간에서 처음) 위로 지난 시각. 직전 유효 표본과 이 표본 사이를 선형 보간한다.
- end_ms: 나가기 전 마지막으로 enter를 아래로 지난 시각(last_down_ms). 같은 방식으로 보간한다.
- 잘린 구간(truncated)의 end_ms: closure가 enter 이상이었던 마지막 유효 프레임의 시각.
- 보간은 두 유효 표본의 간격이 max_dt_ms 이하일 때만 한다.
- 보간한 시각 = 직전 유효 표본의 시각 + floor(fraction · 간격 + 0.5). fraction은 두 표본 사이에서 enter에 닿는
  비율(0~1)이다. 쌍 안에서의 오프셋(ms)만 반올림하고 직전 표본의 시각(정수)은 그대로 더하므로, 결과는
  두 closure 값과 간격만으로 정해지고 `t_ms`의 원점(첫 프레임의 절대 시각)이 달라져도 바뀌지 않는다.

길이 (2026-11-13 소유자 결정: 길이에는 closure가 enter 이상이었던 시간만 더한다)
- 한 구간은 "감긴 토막" 하나 이상으로 이루어진다. 토막은 closure가 enter를 위로 지난 시각에 시작해서 그다음에
  enter를 아래로 지난 시각에 끝난다. 첫 토막의 시작은 start_ms다.
- 다시 감김: 구간이 이어지는 중에(CLOSED, 또는 HOLD에서 복귀한 프레임) 직전 유효 표본이 enter 미만이었고 이번
  표본이 enter 이상인 경우다. 이때 끝난 토막을 더하고 새 토막을 연다.
      sealed_ms    += last_down_ms − span_start_ms
      span_start_ms = 이번에 enter를 위로 지난 시각(start_ms와 같은 조건, 같은 반올림으로 보간한다.
                      간격이 max_dt_ms를 넘어 보간할 수 없으면 이 프레임의 시각)
- duration_ms = sealed_ms + (end_ms − span_start_ms). 잘린 구간도 같은 식이고, 그때의 end_ms는 위에 적은 대로
  enter 이상이었던 마지막 유효 프레임의 시각이다(마지막 토막도 거기서 끝난 것으로 센다. 그 뒤에 enter를 아래로
  지난 시각을 보간해 두었더라도 쓰지 않는다). 따라서 duration_ms ≤ end_ms − start_ms이고, 토막이 하나뿐인
  구간(enter 아래로 내려갔다가 다시 올라온 적이 없는 구간)에서는 둘이 같다.
- 눈꺼풀이 exit와 enter 사이에 머문 시간은 구간을 끝내지 않지만(히스테리시스는 상태 전이에만 쓰인다) 길이에도
  들어가지 않는다. 그 뒤에 다시 감겨도 머문 시간이 뒤늦게 더해지지 않는다.
- 두 표본이 모두 enter 이상이면 그 사이의 끊김(hold_ms 이하)은 토막 안이다. 그 시간은 길이에 들어간다.
- min_duration_ms는 duration_ms에 적용한다. 그보다 짧은 구간은 버린다(발행하지 않는다).
- `ongoing_ms()` = sealed_ms + (enter 이상이었던 마지막 유효 프레임의 시각 − span_start_ms). OPEN에서는 0이다.
  exit와 enter 사이에 머무는 동안과 HOLD 동안에는 늘지 않고, 한 구간 안에서는 줄지 않는다.
- `run_start_ms()` = 진행 중인 구간의 start_ms. 토막이 여럿이면 `t_ms − ongoing_ms()`가 start_ms가 아니므로,
  구간의 시작이 필요한 쪽(규칙 엔진)은 이 값을 읽는다.

속도 (단위 1/s)
- 쌍: 직전 유효 표본과 이번 유효 표본. 간격이 max_dt_ms 이하일 때만 쓴다("쓸 수 있는 쌍").
  직전 유효 표본이 없거나 간격이 max_dt_ms를 넘으면 "쓸 수 없는 쌍"이다.
- peak_closing_velocity: 진입 `velocity_lookback_samples`(2) 표본 전부터 정점까지의 쌍 중 closure 증가율 최댓값.
  진입 전의 쌍은 진입 표본까지 쓸 수 있는 쌍이 끊기지 않고 이어진 것만 본다. 쓸 수 없는 쌍을 만나면 그보다
  오래된 쌍은 모두 버린다(거슬러 보기를 끊는다). 무효 프레임이나 프레임 누락이 max_dt_ms 넘게 이어진 뒤,
  hold_ms를 넘는 끊김으로 구간이 잘린 뒤, `reset()` 뒤의 첫 유효 표본이 모두 여기에 해당한다. 따라서 끊김 앞에서
  잰 증가율이나 잘린 앞 구간의 증가율이 새 구간의 감는 속도로 넘어오지 않는다.
- peak_opening_velocity: 정점부터 구간을 닫은 표본까지의 쌍 중 closure 감소율 최댓값.
- velocity_valid가 False이면 두 속도는 0.0이다.

호출 계약
- 프레임 하나에 `update`를 한 번 부른다. 얼굴이 없거나 closure가 유효하지 않은 프레임도 valid=False로 넘긴다
  (끊긴 시간과 HOLD 판정에 필요하다). 이때 closure 값은 읽지 않는다.
- `t_ms`는 호출마다 엄격히 증가해야 한다. `t_ms`가 직전 값 이하인 프레임(중복 타임스탬프 포함)은
  호출자(`DriverPipeline`)가 버리고 센다(PLAN 3.3절). 이 클래스에 그런 프레임을 넘기면 ValueError가 난다.
- 프레임 간격이 `clock.reset_gap_ms`(10초)를 넘으면 호출자가 `reset()`을 부른 뒤 그 프레임을 넘긴다.
- `update` 한 번은 구간을 최대 하나 발행한다.
- 설정은 `max_dt_ms ≤ hold_ms`여야 한다(`drowsy.spec`의 검증이 거부한다. 기본값은 200 / 500).
  그래야 hold_ms를 넘는 끊김으로 구간이 잘렸을 때 그 끊김을 건너는 쌍이 항상 쓸 수 없는 쌍이 된다.

계획서가 정하지 않아서 정한 것
가정: 끊긴 시간은 HOLD뿐 아니라 CLOSED에서도 검사한다. 무효 프레임 없이 프레임 자체가 hold_ms 넘게 끊겨도
  같은 구간으로 잇지 않는다(PLAN 3.3절: 데이터 없는 시간은 얼굴 없는 시간과 똑같이 취급한다).
  따라서 CLOSED에서 끊긴 시간이 hold_ms를 넘은 무효 프레임은 HOLD를 거치지 않고 바로 구간을 닫는다.
가정: 쌍과 보간은 "직전 유효 표본"만 본다. 두 유효 표본 사이에 무효 프레임이 끼어 있어도 간격이
  max_dt_ms 이하이면 쌍으로 쓴다.
가정: 감는 속도의 거슬러 보기는 쓸 수 없는 쌍에서만 끊는다. 위 가정대로 무효 프레임이 끼어 있어도 간격이
  max_dt_ms 이하인 쌍은 쓸 수 있는 쌍이므로 끊지 않는다(PLAN 4.5절의 "연속 유효 쌍"을 시각 기준으로 읽은 것).
가정: 진입 쌍이 쓸 수 없는 쌍이어도(감는 모습을 보지 못한 구간) 구간 안에서 정점까지 오른 쌍이 있으면
  그 증가율을 감는 속도로 쓴다(계획서의 "정점까지" 그대로다. velocity_valid에 진입 쌍 조건을 더하지 않는다).
가정: 보간할 수 없을 때 start_ms(와 뒤 토막의 시작)는 enter 이상을 처음 본 프레임의 시각, end_ms(와 앞 토막의
  끝)는 enter 이상이었던 마지막 유효 프레임의 시각이다. 감긴 것을 직접 본 시간만 길이에 넣는다.
가정: 잘린 구간의 end_ms는 그 전에 enter 아래로 내려간 표본을 봤더라도 enter 이상이었던 마지막 유효
  프레임의 시각이다(계획서 문장 그대로).
가정: 정점은 최대 closure에 처음 도달한 표본이다(같은 값이 다시 나와도 옮기지 않는다).
가정: velocity_valid는 잘리지 않은 구간에서 0보다 큰 감는 속도와 뜨는 속도를 둘 다 쟀을 때만 True다.
가정: min_duration_ms는 잘린 구간에도 적용한다.
가정: `ongoing_ms()`의 끝은 enter 이상이었던 마지막 유효 프레임의 시각이다(프레임 시각, 보간하지 않음).
  그래서 구간이 끝난 뒤의 duration_ms는 마지막 `ongoing_ms()`보다 보간한 만큼 길 수 있다.
가정: `run_start_ms()`는 OPEN에서 0을 돌려준다(`ongoing_ms()`와 같다. 진행 중인 구간이 없으므로 뜻이 없는 값이다).
가정: `t_ms`가 직전 값 이하인 호출은 조용히 넘기지 않고 ValueError로 거부한다(호출자가 버린다는 계약을 드러낸다).
가정: `reset()`은 진행 중이던 구간을 발행하지 않고 버린다(10초 넘는 끊김 뒤에는 누적기도 함께 초기화된다).
"""

from __future__ import annotations

import math

from drowsy.spec import RunDetectorSpec
from drowsy.types import ClosureRun

STATE_OPEN = "open"
STATE_CLOSED = "closed"
STATE_HOLD = "hold"


class ClosureRunDetector:
    """감김 구간 검출기. 전이표, 시각·길이·속도의 정의, 호출 계약은 모듈 설명에 있다."""

    def __init__(self, settings: RunDetectorSpec) -> None:
        self._enter_thr = settings.enter_thr
        self._exit_thr = settings.exit_thr
        self._min_duration_ms = settings.min_duration_ms
        self._max_dt_ms = settings.max_dt_ms
        self._hold_ms = settings.hold_ms
        self._lookback = settings.velocity_lookback_samples

        self._state = STATE_OPEN

        # 마지막으로 받은 프레임(유효 여부와 무관). 시간 순서 검사에만 쓴다.
        self._has_frame = False
        self._last_t_ms = 0

        # 직전 유효 표본. 무효 프레임이 와도 지우지 않는다.
        self._has_prev = False
        self._prev_t_ms = 0
        self._prev_c = 0.0

        # 최근 쌍들의 closure 증가율(0 이상), 오래된 것부터. 길이는 velocity_lookback_samples로 고정이다.
        # 진입 순간 이 목록의 최댓값이 "진입 두 표본 전부터 진입 표본까지"의 최대 감는 속도다.
        # 쓸 수 없는 쌍이 오면 전부 0으로 지운다(그 쌍보다 오래된 증가율은 새 구간과 이어지지 않는다).
        self._recent_up: list[float] = [0.0] * self._lookback

        # 진행 중인 구간(CLOSED, HOLD에서만 의미가 있다)
        self._start_ms = 0
        self._sealed_ms = 0  # 끝난 토막들의 길이 합
        self._span_start_ms = 0  # 지금 토막이 시작된 시각(첫 토막이면 start_ms)
        self._last_closed_ms = 0  # closure가 enter 이상이었던 마지막 유효 프레임의 시각
        self._last_down_ms = 0  # 마지막으로 enter를 아래로 지난 시각
        self._amplitude = 0.0
        self._closing_peak = 0.0  # 정점까지의 최대 증가율
        self._opening_peak = 0.0  # 정점 뒤의 최대 감소율
        self._pending_up = 0.0  # 정점 뒤의 최대 증가율. 정점이 뒤로 옮겨지면 감는 속도에 합친다

    @property
    def state(self) -> str:
        """현재 상태(STATE_OPEN, STATE_CLOSED, STATE_HOLD)."""
        return self._state

    def reset(self) -> None:
        """처음 상태로 되돌린다. 진행 중이던 구간은 발행하지 않고 버린다.

        프레임 간격이 10초를 넘었을 때 호출자가 부른다(PLAN 3.3절).
        """
        self._state = STATE_OPEN
        self._has_frame = False
        self._last_t_ms = 0
        self._has_prev = False
        self._prev_t_ms = 0
        self._prev_c = 0.0
        for i in range(self._lookback):
            self._recent_up[i] = 0.0

    def ongoing_ms(self) -> int:
        """진행 중인 구간의 길이(ms). 진행 중인 구간이 없으면 0이다.

        closure가 enter 이상이었던 시간의 합이다: 끝난 토막들의 길이에, 지금 토막의 시작부터 closure가 enter
        이상이었던 가장 최근 프레임 시각까지를 더한다. 토막이 하나뿐이면 그 프레임 시각 − start_ms다.
        눈꺼풀이 exit와 enter 사이에 머무는 동안과 HOLD 동안에는 늘어나지 않고, 한 구간 안에서는 줄지 않는다.
        """
        if self._state == STATE_OPEN:
            return 0
        return self._sealed_ms + (self._last_closed_ms - self._span_start_ms)

    def run_start_ms(self) -> int:
        """진행 중인 구간의 start_ms(closure가 enter를 처음 위로 지난 시각). 진행 중인 구간이 없으면 0이다.

        구간이 끝나서 발행될 `ClosureRun.start_ms`와 같은 값이다. 상태가 OPEN이 아닐 때만 뜻이 있다.
        """
        if self._state == STATE_OPEN:
            return 0
        return self._start_ms

    def update(self, t_ms: int, closure: float, valid: bool) -> ClosureRun | None:
        """프레임 하나를 처리한다. 이 프레임에서 구간이 끝났으면 `ClosureRun`을, 아니면 None을 돌려준다."""
        if self._has_frame and t_ms <= self._last_t_ms:
            raise ValueError("t_ms must be strictly greater than the previous frame's t_ms")
        self._has_frame = True
        self._last_t_ms = t_ms

        run: ClosureRun | None = None

        # 끊긴 시간이 hold_ms를 넘었으면 먼저 구간을 닫고, 이 프레임은 OPEN에서 처리한다.
        if self._state != STATE_OPEN and t_ms - self._prev_t_ms > self._hold_ms:
            run = self._finish(True)

        if not valid:
            if self._state == STATE_CLOSED:
                self._state = STATE_HOLD
            return run

        # 직전 유효 표본과의 쌍
        pair_ok = self._has_prev and t_ms - self._prev_t_ms <= self._max_dt_ms
        up_rate = 0.0
        down_rate = 0.0
        if pair_ok:
            rate = (closure - self._prev_c) * 1000.0 / (t_ms - self._prev_t_ms)
            if rate > 0.0:
                up_rate = rate
            elif rate < 0.0:
                down_rate = -rate
        else:
            # 쓸 수 없는 쌍(직전 유효 표본이 없거나 간격이 max_dt_ms 초과): 거슬러 보기를 끊는다.
            # 끊김 앞의 증가율이 이 표본에서 시작하는 구간의 감는 속도로 넘어가지 않게 한다.
            for i in range(self._lookback):
                self._recent_up[i] = 0.0
        for i in range(self._lookback - 1):
            self._recent_up[i] = self._recent_up[i + 1]
        self._recent_up[self._lookback - 1] = up_rate

        if self._state == STATE_OPEN:
            if closure >= self._enter_thr:
                self._begin(t_ms, closure, pair_ok)
        else:
            resumed_from_hold = self._state == STATE_HOLD
            self._state = STATE_CLOSED

            if closure > self._amplitude:
                # 정점이 이 표본으로 옮겨졌다. 여기까지의 쌍은 모두 감는 쪽이고 뜨는 쪽은 다시 잰다.
                self._amplitude = closure
                if self._pending_up > self._closing_peak:
                    self._closing_peak = self._pending_up
                if up_rate > self._closing_peak:
                    self._closing_peak = up_rate
                self._pending_up = 0.0
                self._opening_peak = 0.0
            else:
                if up_rate > self._pending_up:
                    self._pending_up = up_rate
                if down_rate > self._opening_peak:
                    self._opening_peak = down_rate

            if closure >= self._enter_thr:
                if self._prev_c < self._enter_thr:
                    # 다시 감김: enter 아래로 내려갔다가 구간이 끝나기 전에 다시 올라왔다. 앞 토막을 마지막으로
                    # 내려간 시각에서 닫아 더하고, 이번에 올라온 시각에서 새 토막을 연다. 그 사이는 길이에 넣지 않는다.
                    self._sealed_ms += self._last_down_ms - self._span_start_ms
                    self._span_start_ms = self._up_crossing_ms(t_ms, closure, pair_ok)
                self._last_closed_ms = t_ms
            else:
                if self._prev_c >= self._enter_thr:
                    self._last_down_ms = self._down_crossing_ms(t_ms, closure, pair_ok)
                if closure < self._exit_thr:
                    run = self._finish(resumed_from_hold)

        self._has_prev = True
        self._prev_t_ms = t_ms
        self._prev_c = closure
        return run

    def _begin(self, t_ms: int, closure: float, pair_ok: bool) -> None:
        """OPEN -> CLOSED. 이 표본이 enter 이상을 처음 본 표본이다."""
        start_ms = self._up_crossing_ms(t_ms, closure, pair_ok)

        closing_peak = 0.0
        for i in range(self._lookback):
            if self._recent_up[i] > closing_peak:
                closing_peak = self._recent_up[i]

        self._state = STATE_CLOSED
        self._start_ms = start_ms
        self._sealed_ms = 0
        self._span_start_ms = start_ms
        self._last_closed_ms = t_ms
        self._last_down_ms = t_ms
        self._amplitude = closure
        self._closing_peak = closing_peak
        self._opening_peak = 0.0
        self._pending_up = 0.0

    def _up_crossing_ms(self, t_ms: int, closure: float, pair_ok: bool) -> int:
        """직전 유효 표본과 이 표본(enter 이상) 사이에서 enter를 위로 지난 시각. 구간과 뒤 토막의 시작에 쓴다.

        쓸 수 있는 쌍이고 직전 표본이 enter 미만일 때만 보간한다. 아니면 이 프레임의 시각이다.
        """
        if not pair_ok or self._prev_c >= self._enter_thr:
            return t_ms
        fraction = (self._enter_thr - self._prev_c) / (closure - self._prev_c)
        # 쌍 안의 오프셋만 반올림한다(절대 시각을 더한 뒤 반올림하면 원점에 따라 1 ms 달라질 수 있다).
        return self._prev_t_ms + _round_half_up(fraction * (t_ms - self._prev_t_ms))

    def _down_crossing_ms(self, t_ms: int, closure: float, pair_ok: bool) -> int:
        """직전 유효 표본(enter 이상)과 이 표본(enter 미만) 사이에서 enter를 아래로 지난 시각."""
        if not pair_ok:
            return self._prev_t_ms
        fraction = (self._prev_c - self._enter_thr) / (self._prev_c - closure)
        return self._prev_t_ms + _round_half_up(fraction * (t_ms - self._prev_t_ms))

    def _finish(self, truncated: bool) -> ClosureRun | None:
        """진행 중인 구간을 닫고 OPEN으로 간다. min_duration_ms보다 짧으면 None을 돌려준다."""
        self._state = STATE_OPEN
        if truncated:
            end_ms = self._last_closed_ms
        else:
            end_ms = self._last_down_ms
        # 끝난 토막들에 마지막 토막을 더한다. 토막이 하나뿐이면 end_ms − start_ms다.
        duration_ms = self._sealed_ms + (end_ms - self._span_start_ms)
        if duration_ms < self._min_duration_ms:
            return None

        velocity_valid = (not truncated) and self._closing_peak > 0.0 and self._opening_peak > 0.0
        closing_velocity = 0.0
        opening_velocity = 0.0
        if velocity_valid:
            closing_velocity = self._closing_peak
            opening_velocity = self._opening_peak
        return ClosureRun(
            start_ms=self._start_ms,
            end_ms=end_ms,
            duration_ms=duration_ms,
            amplitude=self._amplitude,
            peak_closing_velocity=closing_velocity,
            peak_opening_velocity=opening_velocity,
            velocity_valid=velocity_valid,
            truncated=truncated,
        )


def _round_half_up(x: float) -> int:
    """floor(x + 0.5). 계획서가 정한 반올림이다(Python의 round는 짝수 쪽으로 반올림하므로 쓰지 않는다)."""
    return int(math.floor(x + 0.5))
