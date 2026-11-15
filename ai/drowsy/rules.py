"""규칙 엔진과 경고 결정 (이식 대상, 표준 라이브러리만 사용). PLAN 8장.

- `RuleEngine.update(state)`: 받아들인 프레임마다 한 번 부른다. 이 프레임에서 생긴 상태 이벤트를 돌려준다
  (calibrating, unable_to_analyze, gaze_away_long, vats, microsleep, sleep 순서로 판정한다).
  10초 넘는 끊김 뒤의 첫 프레임에서는 초기화가 내는 `end`가 그 앞에 온다(아래 "시간").
- `RuleEngine.on_step(step, probs)`: 스텝이 발행될 때마다 부른다. drowsy 이벤트를 돌려준다.
  프레임이 스텝을 발행시켰으면 그 프레임의 `update`보다 먼저, 오래된 스텝부터 부른다.
- `AlertPolicy.decide(events, speed_kmh, t_ms)`: 프레임마다 한 번 부른다. 실제로 경고할 이벤트를 돌려준다.

이벤트 공통 규칙
- 판정은 프레임(drowsy는 스텝)이 도착할 때만 한다. 임계 시간에 도달한 뒤 처음 판정하는 프레임에서 `onset`,
  상태가 끝난 프레임에서 `end`를 낸다.
- `duration_ms` = 발행 시각 − `t_start_ms`. 발행 시각은 프레임 이벤트는 그 프레임의 t_ms, drowsy는 스텝 끝 시각이다
  (끊김 초기화가 끝내는 drowsy `end`만 그 프레임의 t_ms다).
  `end`는 짝이 되는 `onset`과 같은 `t_start_ms`와 `source`를 쓴다.
- `severity`는 설정의 종류별 값, `provisional`은 발행한 프레임의 `state.provisional`이다(drowsy는 항상 False).
- `confidence`(규칙 이벤트)는 "그 이벤트 구간에서 해당 신호가 유효했던 시간 / 구간 길이"다. 길이가 0이면 0.0이다.
  유효 시간은 PLAN 3.3절의 유지 규칙으로 센다(프레임의 상태는 다음 프레임까지, 최대 `clock.max_hold_ms`).
    calibrating, unable_to_analyze : [t_start_ms, 발행 시각]에서 얼굴이 있던 시간의 비율
    gaze_away_long, vats           : [t_start_ms, 발행 시각]에서 `pose_valid`였던 시간의 비율
    microsleep, sleep              : [감김을 처음 본 프레임, 발행 시각]에서 `closure_valid`였던 시간의 비율
    drowsy(model)                  : 발행한 스텝의 p_drowsy (확률이 없는 스텝에서 끝나면 0.0)
    drowsy(perclos)                : 발행한 스텝의 1 − face_missing_ratio_60
    drowsy(끊김 초기화의 end)      : 0.0
  가정: 위 구간과 신호의 선택은 계획서의 "해당 구간의 유효 시간 비율"을 종류별로 풀어 쓴 것이다.
  가정: 끊김 초기화가 끝내는 drowsy `end`는 그 값을 실어 줄 스텝이 없으므로 confidence를 0.0으로 둔다.

시간
- `t_ms`는 호출마다 엄격히 증가해야 한다(아니면 ValueError). 버리는 것은 호출자(`DriverPipeline`)의 일이다.
- 끊김 초기화 (PLAN 3.3절, 2026-11-13 소유자 결정): 프레임 간격이 `clock.reset_gap_ms`(10초)를 넘으면(정확히
  10초인 간격은 넘은 것이 아니다) 그 프레임에서 규칙 엔진이 세션을 시작할 때처럼 다시 시작한다.
  그 프레임의 `update`는 아래 순서로 한다.
    1. 진행 중이던 상태의 `end`를 drowsy, unable_to_analyze, gaze_away_long, microsleep 또는 sleep(그 감김이
       도달한 가장 높은 종류 하나) 순서로 낸다. 발행 시각은 모두 그 프레임의 t_ms이고, `t_start_ms`와 `source`는
       짝이 되는 `onset`의 것, confidence는 위 공통 규칙의 값이다.
    2. 처음 상태로 되돌린다: 두 졸음 판정기(연속 수, 판정 상태, 재시작 금지 기억), 분석 불가의 얼굴 창(비우고
       "처음 `window_ms` 동안은 판정하지 않는다"는 구간을 그 프레임부터 다시 센다), 전방 주시 상태와 VATS 누적,
       규칙용 감김 검출기.
    3. 그 프레임을 평소 순서로 판정한다. 그 프레임부터 `window_ms` 동안은 분석 불가가 아니므로 시선과 감김
       규칙이 바로 돈다. 진행 중이던 vats는 합이 비워졌으므로 여기서 `end`가 난다(1번의 `end`들 뒤이고,
       같은 프레임에 calibrating `end`가 있으면 그 뒤다).
  남기는 것은 calibrating 상태(기준선은 유지된다, PLAN 3.3절)와 유효 시간 누적값(신뢰도 계산용. 끊김 전의
  값은 끊김 뒤에 시작하는 이벤트의 신뢰도에 들어가지 않는다)뿐이다.
- 간격이 `reset_gap_ms` 이하이면 초기화하지 않는다. 데이터 없는 시간은 얼굴 없는 시간이므로(PLAN 8.1절),
  끊기기 전 10초 동안 얼굴이 계속 있었더라도 5.2초를 넘고 10초 이하인 끊김 뒤에는 첫 프레임에서 분석 불가가
  시작된다(끊긴 구간 중 얼굴 시간으로 인정되는 것은 `max_hold_ms` 200 ms뿐이다. 처음 `window_ms`가 지나
  판정을 시작한 뒤의 이야기다). 분석 불가 경고는 속도로 억제하지 않으므로 재경고 대기 중이 아니면 경고도 나간다.
  가정: `on_step`은 `update`보다 먼저 불리므로, 10초 넘는 끊김 뒤의 첫 프레임이 발행시킨 스텝이 있다면 그 스텝은
    초기화 전의 판정기로 처리되고 그 뒤에 1번의 drowsy `end`가 온다. `DriverPipeline`에서는 그런 스텝이 값을 갖지
    못한다(누적기가 같은 조건으로 다시 시작해 스텝이 없다. 누적기가 그 프레임에서 처음 프레임을 받는 경우에만
    스텝이 나오는데 조회 구간이 덜 찬 빈 스텝이다). 그래서 끊김 뒤 첫 프레임의 스텝이 drowsy를 열거나 닫아
    초기화 직전에 상태가 바뀌는 일은 없다.
"""

from __future__ import annotations

from collections.abc import Sequence

from drowsy.blink import STATE_OPEN, ClosureRunDetector
from drowsy.spec import ForwardZoneRules, RulesConfig
from drowsy.types import (
    CLASS_DROWSY,
    DROWSY_SOURCE_EITHER,
    DROWSY_SOURCE_MODEL,
    DROWSY_SOURCE_PERCLOS,
    EVENT_CALIBRATING,
    EVENT_DROWSY,
    EVENT_GAZE_AWAY_LONG,
    EVENT_MICROSLEEP,
    EVENT_SLEEP,
    EVENT_UNABLE_TO_ANALYZE,
    EVENT_VATS,
    F_FACE_MISSING_RATIO_60,
    F_PERCLOS_60,
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

# 전방 주시 표지. `ForwardZone.zone`은 IN과 OUT만 갖고, UNKNOWN은 프레임 표지(`label`)에만 쓴다.
ZONE_IN = "in"
ZONE_OUT = "out"
ZONE_UNKNOWN = "unknown"

# 한 감김이 도달한 가장 높은 종류
_LEVEL_NONE = 0
_LEVEL_MICROSLEEP = 1
_LEVEL_SLEEP = 2


class IntervalWindow:
    """최근 `window_ms` 안에 든 시간 구간들의 길이 합 (VATS, 분석 불가, 유효 시간 비율에 쓴다).

    구간 [start, end)를 시간 순서로, 겹치지 않게 넣는다. 앞 구간의 끝과 바로 이어지면 하나로 합친다.
    `trim(now)`는 now − window_ms보다 오래된 부분을 정확히 잘라 낸다(걸친 구간은 그 시각에서 자른다).
    """

    def __init__(self, window_ms: int) -> None:
        self._window_ms = window_ms
        self._starts: list[int] = []
        self._ends: list[int] = []

    def clear(self) -> None:
        self._starts = []
        self._ends = []

    def add(self, start_ms: int, end_ms: int) -> None:
        if end_ms <= start_ms:
            return
        n = len(self._ends)
        if n > 0 and self._ends[n - 1] == start_ms:
            self._ends[n - 1] = end_ms
        else:
            self._starts.append(start_ms)
            self._ends.append(end_ms)

    def trim(self, now_ms: int) -> None:
        cut_ms = now_ms - self._window_ms
        while len(self._ends) > 0 and self._ends[0] <= cut_ms:
            del self._starts[0]
            del self._ends[0]
        if len(self._starts) > 0 and self._starts[0] < cut_ms:
            self._starts[0] = cut_ms

    def total_ms(self) -> int:
        """들어 있는 구간 길이의 합. `trim` 뒤에 부르면 최근 window_ms 안의 합이다."""
        total = 0
        for i in range(len(self._starts)):
            total += self._ends[i] - self._starts[i]
        return total

    def since_ms(self, from_ms: int) -> int:
        """from_ms 이후 부분의 길이 합."""
        total = 0
        for i in range(len(self._starts)):
            start_ms = self._starts[i]
            if start_ms < from_ms:
                start_ms = from_ms
            if self._ends[i] > start_ms:
                total += self._ends[i] - start_ms
        return total

    def first_start_ms(self) -> int:
        """가장 오래된 구간의 시작. 비어 있을 때는 부르지 않는다."""
        return self._starts[0]


class ForwardZone:
    """전방 주시 상태 머신 (PLAN 8.1절).

    관측(프레임마다)
    - 영역 안: |yaw_rel| ≤ yaw_abs_max_deg 이고 −pitch_up_max_deg ≤ pitch_rel ≤ pitch_down_max_deg.
    - 머리 조건이 영역 밖이면 source는 head_pose다. 머리는 안인데 눈동자 보조(D12)가
      |gaze_h_rel| > gaze_h_abs_max 또는 |gaze_v_rel| > gaze_v_abs_max이면 영역 밖이고 source는 eye다.
    - `state.pose_valid`가 False이거나 분석 불가 중이면(`analyzable=False`) 관측이 없다.

    표지(`label`)
    - 관측이 있으면 그 값이다.
    - 관측이 없으면 마지막 유효 관측을 이어받는다. 단 얼굴이 사라진 시각 + `face_loss_credit_ms`(3초) 전까지만이고,
      그 뒤에는 UNKNOWN이다. 얼굴이 사라진 시각은 마지막 유효 관측이 유지된 끝
      (다음 프레임 시각과 그 관측 시각 + `max_hold_ms` 중 이른 쪽)이다.
    - 분석 불가 중인 프레임은 이어받지 않고 바로 UNKNOWN이다.

    상태(`zone`)와 디바운스
    - 표지가 지금 상태와 다르면 그 프레임을 후보의 시작으로 기록한다. 같은 표지가 이어져서 후보 시작 뒤
      `debounce_ms`(250 ms) 이상 지난 프레임이 오면 확정하고, 상태의 시작 시각은 후보의 시작으로 소급한다.
    - 그 전에 지금 상태와 같은 표지가 오면 후보를 버린다(상태는 끊기지 않고 이어진 것으로 본다).
    - UNKNOWN 표지는 상태를 바꾸지 않는다. 영역 밖 상태에서 UNKNOWN이 이어지는 동안에는 시간을 세지 않는다.

    영역 밖 시간 (`away`, `unknown_ms`)
    - 확정된 영역 밖 상태에 속하면서 표지가 OUT이던 시간(이어받은 시간 포함)을 `away`에 구간으로 넣는다.
      후보가 걸려 있는 동안은 미뤄 두었다가, 확정되거나 버려질 때 한꺼번에 정산한다.
    - 영역 밖 상태가 시작된 뒤 세지 않은 시간은 `unknown_ms`에 더한다.
      연속 영역 밖 시간 = t − out_start_ms − unknown_ms.

    가정: 초기 상태는 영역 안이다.
    가정: 이어받기는 마지막 유효 관측이 영역 안이었을 때도 똑같이 한다(계획서는 영역 밖일 때만 적었다).
      영역 밖 상태에서 잠깐 앞을 본 직후 얼굴이 사라지면 영역 안으로 확정된다.
    가정: 확정은 프레임이 도착할 때만 판정한다. 영역 밖 후보는 이어받은 표지로도 확정된다(고개를 빨리 돌려
      250 ms가 차기 전에 얼굴이 사라지는 경우를 놓치지 않으려는 것이다).
    가정: 후보가 확정되지 못한 채 UNKNOWN 구간을 만나면 후보를 버리고, 후보 구간은 세지 않는다.
    가정: `out_source`는 마지막 유효 영역 밖 관측의 source다. 시선 이벤트는 시작할 때의 이 값을 source로 쓴다.
    """

    def __init__(self, cfg: ForwardZoneRules, max_hold_ms: int, away_window_ms: int) -> None:
        self._cfg = cfg
        self._max_hold_ms = max_hold_ms
        self.away = IntervalWindow(away_window_ms)

        self.zone = ZONE_IN
        self.label = ZONE_UNKNOWN
        self.confirmed_in = False  # 이 프레임에서 영역 안이 확정됐는지
        self.out_start_ms = 0  # 지금(또는 마지막) 영역 밖 상태가 시작된 프레임
        self.out_pose_ms = 0  # 그 프레임에서의 pose_valid 누적 시간
        self.out_source = SOURCE_HEAD_POSE
        self.unknown_ms = 0
        self.in_start_ms = 0  # 지금(또는 마지막) 영역 안 상태가 시작된 프레임

        self._has_prev = False
        self._prev_t_ms = 0
        self._prev_observed = False  # 직전 프레임에 유효 관측이 있었는지
        self._cover_label = ZONE_UNKNOWN  # 이어받을 표지
        self._lost_ms = 0  # 마지막 유효 관측이 유지된 끝
        self._acct_ms = 0  # 여기까지 영역 밖 시간을 정산했다

        self._cand = False  # 반대 표지의 후보가 걸려 있는지
        self._cand_ms = 0
        self._cand_pose_ms = 0

    @property
    def candidate_pending(self) -> bool:
        """반대 표지의 후보가 확정도 폐기도 되지 않은 채 걸려 있는지.

        걸려 있는 동안에는 후보 구간이 `unknown_ms`에 아직 들어가지 않았으므로 `t − out_start_ms − unknown_ms`가
        후보 구간만큼 크게 읽힌다. 그 값을 연속 영역 밖 시간으로 쓰려면 이 값이 False여야 한다.
        """
        return self._cand

    def reset(self) -> None:
        """처음 상태로 되돌린다. 영역 밖 누적도 지운다."""
        self.away.clear()
        self.zone = ZONE_IN
        self.label = ZONE_UNKNOWN
        self.confirmed_in = False
        self.unknown_ms = 0
        self._has_prev = False
        self._prev_observed = False
        self._cover_label = ZONE_UNKNOWN
        self._cand = False

    def update(self, state: FrameState, analyzable: bool, pose_ms: int) -> None:
        """프레임 하나를 처리한다. `pose_ms`는 이 프레임까지의 pose_valid 누적 시간이다."""
        t = state.t_ms
        credit_ms = self._cfg.face_loss_credit_ms
        self.confirmed_in = False

        observed = analyzable and state.pose_valid
        obs = ZONE_UNKNOWN
        if observed:
            obs = self._observe(state)

        if not self._has_prev:
            self.in_start_ms = t
        else:
            if self._prev_observed:
                held_ms = t - self._prev_t_ms
                if held_ms > self._max_hold_ms:
                    held_ms = self._max_hold_ms
                self._lost_ms = self._prev_t_ms + held_ms
            cover_end_ms = self._lost_ms + credit_ms

            # 지난 구간 [직전 프레임, 이 프레임)의 정산
            if self._cand:
                if self._cover_label == ZONE_UNKNOWN or t > cover_end_ms:
                    if self.zone == ZONE_OUT:
                        self.unknown_ms += t - self._cand_ms
                    self._cand = False
            elif self.zone == ZONE_OUT:
                away_end_ms = self._acct_ms
                if self._cover_label == ZONE_OUT:
                    away_end_ms = t
                    if away_end_ms > cover_end_ms:
                        away_end_ms = cover_end_ms
                    if away_end_ms < self._acct_ms:
                        away_end_ms = self._acct_ms
                self.away.add(self._acct_ms, away_end_ms)
                self.unknown_ms += t - away_end_ms

        # 이 프레임의 표지
        label = obs
        if not observed and analyzable and self._has_prev:
            if self._cover_label != ZONE_UNKNOWN and t < self._lost_ms + credit_ms:
                label = self._cover_label
        self._cover_label = label

        # 디바운스
        if label != ZONE_UNKNOWN:
            if label == self.zone:
                if self._cand:
                    if self.zone == ZONE_OUT:
                        self.away.add(self._cand_ms, t)
                    self._cand = False
            else:
                if not self._cand:
                    self._cand = True
                    self._cand_ms = t
                    self._cand_pose_ms = pose_ms
                if t - self._cand_ms >= self._cfg.debounce_ms:
                    self._cand = False
                    self.zone = label
                    if label == ZONE_OUT:
                        self.out_start_ms = self._cand_ms
                        self.out_pose_ms = self._cand_pose_ms
                        self.unknown_ms = 0
                        self.away.add(self._cand_ms, t)
                    else:
                        self.in_start_ms = self._cand_ms
                        self.confirmed_in = True

        self.label = label
        self._acct_ms = t
        self._has_prev = True
        self._prev_t_ms = t
        self._prev_observed = observed

    def _observe(self, state: FrameState) -> str:
        cfg = self._cfg
        yaw_abs = state.yaw_rel
        if yaw_abs < 0.0:
            yaw_abs = -yaw_abs
        head_out = yaw_abs > cfg.yaw_abs_max_deg
        if state.pitch_rel < -cfg.pitch_up_max_deg or state.pitch_rel > cfg.pitch_down_max_deg:
            head_out = True
        if head_out:
            self.out_source = SOURCE_HEAD_POSE
            return ZONE_OUT
        if cfg.eye_aux_enabled:
            gaze_h_abs = state.gaze_h_rel
            if gaze_h_abs < 0.0:
                gaze_h_abs = -gaze_h_abs
            gaze_v_abs = state.gaze_v_rel
            if gaze_v_abs < 0.0:
                gaze_v_abs = -gaze_v_abs
            if gaze_h_abs > cfg.gaze_h_abs_max or gaze_v_abs > cfg.gaze_v_abs_max:
                self.out_source = SOURCE_EYE
                return ZONE_OUT
        return ZONE_IN


class StepHysteresis:
    """연속 스텝 수로 시작과 종료를 정하는 판정기. 졸음 판정의 모델 쪽과 PERCLOS 쪽이 같이 쓴다 (PLAN 6장, 8.2절).

    - 시작 조건을 만족한 스텝이 `onset_steps`개 연속이면 시작한다. `start_ms`는 그 연속 구간의 첫 스텝 끝이다.
    - 종료 조건을 만족한 스텝이 `end_steps`개 연속이면 끝난다.
    - 끝난 뒤 `refractory_ms` 동안은 다시 시작하지 않는다.
    - 값이 없는 스텝(`has_value=False`)과 스텝 번호가 이어지지 않는 경우에는 두 연속 수를 0으로 되돌린다.
      상태(`active`)는 그대로다.
    - `reset()`은 만들었을 때의 상태로 되돌린다(규칙 엔진이 10초 넘는 끊김 뒤에 부른다).

    가정: 재시작 금지 시간 동안에도 연속 수는 센다. 금지 시간이 끝난 첫 스텝에서 연속 수가 이미 찼으면
      그 스텝에서 시작하고, `start_ms`는 그 연속 구간의 첫 스텝 끝(금지 시간 안일 수 있다)이다.
    """

    def __init__(self, onset_steps: int, end_steps: int, refractory_ms: int) -> None:
        self._onset_steps = onset_steps
        self._end_steps = end_steps
        self._refractory_ms = refractory_ms

        self.active = False
        self.start_ms = 0  # 지금(또는 마지막) 판정 구간의 t_start_ms

        self._has_k = False
        self._last_k = 0
        self._onset_run = 0
        self._end_run = 0
        self._run_start_ms = 0  # 지금 세고 있는 시작 조건 연속 구간의 첫 스텝 끝
        self._has_ended = False
        self._last_end_ms = 0

    def reset(self) -> None:
        """처음 상태로 되돌린다: 판정 상태, 두 연속 수, 스텝 번호의 이어짐, 재시작 금지 기억을 모두 지운다."""
        self.active = False
        self.start_ms = 0
        self._has_k = False
        self._last_k = 0
        self._onset_run = 0
        self._end_run = 0
        self._run_start_ms = 0
        self._has_ended = False
        self._last_end_ms = 0

    def update(self, k: int, t_end_ms: int, has_value: bool, onset_hit: bool, end_hit: bool) -> None:
        """스텝 하나를 처리한다. 결과는 `active`와 `start_ms`로 읽는다."""
        if self._has_k and k != self._last_k + 1:
            self._onset_run = 0
            self._end_run = 0
        self._has_k = True
        self._last_k = k

        if not has_value:
            self._onset_run = 0
            self._end_run = 0
            return

        if onset_hit:
            if self._onset_run == 0:
                self._run_start_ms = t_end_ms
            self._onset_run += 1
        else:
            self._onset_run = 0
        if end_hit:
            self._end_run += 1
        else:
            self._end_run = 0

        if not self.active:
            if self._onset_run >= self._onset_steps:
                if not self._has_ended or t_end_ms - self._last_end_ms >= self._refractory_ms:
                    self.active = True
                    self.start_ms = self._run_start_ms
        elif self._end_run >= self._end_steps:
            self.active = False
            self._has_ended = True
            self._last_end_ms = t_end_ms


class RuleEngine:
    """규칙 엔진 (PLAN 8장). 이벤트 공통 규칙과 초기화 규칙은 모듈 설명에 있다.

    calibrating (D4)
    - 첫 프레임이 provisional이면 그 프레임에서 시작하고, provisional이 False인 첫 프레임에서 끝난다.
      첫 프레임부터 provisional이 False이면(기준선을 넣어 준 경우) 이벤트를 내지 않는다.

    unable_to_analyze (8.1절)
    - 최근 `window_ms`(10초)의 얼굴 시간 비율(데이터 없는 시간은 얼굴 없는 시간)이 `min_face_ratio`(0.5)
      미만인 첫 프레임에서 시작하고(`t_start_ms`는 그 프레임), 0.5 이상인 첫 프레임에서 끝난다.
    - 지속되는 동안 시선과 감김의 누적을 멈춘다: 전방 주시 표지는 UNKNOWN, 규칙용 감김 검출기에는 무효 프레임을 준다.
    - 10초 넘는 끊김 뒤의 첫 프레임에서는 진행 중이던 분석 불가를 끝내고(`end`) 창을 비운다. 아래의 "판정하지 않는
      구간"도 그 프레임부터 다시 센다. 그래서 그 프레임과 이후 `window_ms` 동안은 분석 불가가 아니다.
      10초 이하의 끊김은 평소대로 얼굴 없는 시간으로 센다(모듈 설명의 "시간").
    가정: 세션의 첫 프레임(끊김 초기화가 있었으면 그 뒤의 첫 프레임)부터 `window_ms`가 지나기 전에는 판정하지
      않는다(창이 덜 차서 생기는 오판정을 막는다).

    gaze_away_long, vats (8.1절, 8.2절. 전방 주시 상태는 `ForwardZone`)
    - gaze_away_long: 영역 밖 상태에서 연속 영역 밖 시간이 `onset_ms`(3초) 이상이 된 뒤 처음 도착한 프레임에서
      시작한다. 그 프레임의 표지는 OUT이거나, 분석 불가가 아닌 프레임의 UNKNOWN이어야 한다. 뒤쪽은 얼굴 소실 뒤
      인정 구간(3초) 안에서 3초가 찼는데 그 구간에 프레임이 하나도 도착하지 않은 경우다(UNKNOWN 동안에는 시간을
      세지 않으므로 이미 찬 시간만으로 판정한다). 표지가 IN인 프레임(영역 안 후보가 걸려 있다)과 분석 불가 중인
      프레임에서는 시작하지 않는다.
      가정: 표지가 UNKNOWN인 프레임에 영역 안 후보가 아직 걸려 있으면(`ForwardZone.candidate_pending`. 인정 구간이
      끝나는 바로 그 시각에 프레임이 온 경우) 그 프레임에서는 판정하지 않는다. 후보 구간이 연속 시간에 섞여
      잠깐 앞을 본 뒤의 끊김이 시선 이탈로 잡히는 것을 막는다.
      `t_start_ms`는 영역 밖 상태가 시작된 프레임. 영역 안이 확정된 프레임에서 끝난다.
    - vats: 최근 `window_ms`(30초)의 영역 밖 시간 합이 `cumulative_ms`(10초) 이상인 첫 프레임에서 시작,
      미만인 첫 프레임에서 끝난다. `t_start_ms`는 시작할 때 그 30초 안의 첫 영역 밖 구간 시작이다
      (30초 창의 시작에 걸친 구간이면 창의 시작 시각).
    - `vats.reset_enabled`이면 표지가 IN인 프레임에서 영역 안 상태가 `reset_forward_ms` 이상 이어졌을 때 합을 지운다.

    microsleep, sleep (8.1절, 8.2절, D2)
    - 규칙용 `ClosureRunDetector`(진입 0.7 / 이탈 0.5)를 돌린다. 유효한 감김 프레임(closure ≥ 진입)이 도착했을 때
      `ongoing_ms`가 `microsleep_ms`(1초) 이상이면 microsleep 시작, `sleep_ms`(3초) 이상이면 sleep 시작이다.
    - `ongoing_ms`는 그 감김에서 closure가 진입 임계 이상이었던 시간의 합이다(2026-11-13 소유자 결정). 이탈과 진입
      사이(0.5~0.7)에 머문 시간은 감김을 끝내지 않지만 길이에도 들어가지 않는다. 잠깐 감았다가 반쯤 뜬 채로 몇 초
      있다가 다시 감아도 그 몇 초가 뒤늦게 더해져 시작 이벤트가 나지는 않는다.
    - 둘의 `t_start_ms`는 같고, 그 감김에서 closure가 진입 임계를 처음 위로 지난 시각이다(검출기의 start_ms.
      진행 중에는 `ClosureRunDetector.run_start_ms()`로 읽는다).
      가정: `duration_ms`는 이벤트 공통 규칙대로 발행 시각 − `t_start_ms`다. 감김 도중에 반쯤 뜬 시간이 있었으면
      시작 이벤트를 낸 누적 감김 시간(`ongoing_ms`)보다 길다(vats의 `duration_ms`가 영역 밖 시간의 합보다 긴 것과 같다).
    - 감김이 끝난 프레임에서 그 감김이 도달한 가장 높은 종류의 `end`를 하나 낸다.
    - 끊긴 구간(500 ms 이하)은 앞뒤 표본이 모두 진입 임계 이상이면 길이에 들어가지만, 그 안에서는 시작 이벤트를
      내지 않는다.

    drowsy (`on_step`, 6장, 8.2절, D11)
    - 모델: p_drowsy ≥ onset_p가 onset_steps 연속이면 시작, ≤ end_p가 end_steps 연속이면 종료,
      종료 후 refractory_ms 동안은 다시 시작하지 않는다. 확률이 없는 스텝(probs=None)은 값이 없는 스텝이다.
    - PERCLOS: perclos_60 ≥ onset_ratio가 onset_steps 연속이면 시작, < end_ratio가 end_steps 연속이면 종료.
    - `drowsiness.source`가 model이면 모델만, perclos면 PERCLOS만 쓴다. either면 둘 중 하나라도 판정 중일 때
      drowsy 상태다(둘 다 끝나야 끝난다). 시작 이벤트의 source와 `t_start_ms`는 상태를 연 쪽의 것이고
      (같은 스텝에서 함께 열리면 model), 종료 이벤트도 같은 source와 `t_start_ms`를 쓴다.
    - 10초 넘는 끊김 뒤의 첫 프레임(`update`)에서 진행 중이던 drowsy를 끝내고 두 판정기를 처음 상태로 되돌린다
      (모듈 설명의 "시간"). 그 `end`는 발행 시각이 그 프레임의 t_ms이고 confidence가 0.0이다. 끊김 뒤에는
      시작 조건을 처음부터 다시 채워야 새 drowsy가 시작되고, 재시작 금지 시간도 이어지지 않는다.
    가정: PERCLOS 판정은 `full_lookback`인 스텝만 쓴다. 조회 구간이 덜 찬 스텝은 값이 없는 스텝으로 본다
      (perclos_60이 60초 구간의 값일 때만 임계값과 비교한다).
    """

    def __init__(self, cfg: RulesConfig) -> None:
        self._cfg = cfg
        self._max_hold_ms = cfg.clock.max_hold_ms
        self._reset_gap_ms = cfg.clock.reset_gap_ms

        # 직전 프레임
        self._has_prev = False
        self._first_t_ms = 0  # 세션의 첫 프레임. 끊김 초기화 뒤에는 그 프레임(분석 불가를 판정하지 않는 구간의 시작)
        self._prev_t_ms = 0
        self._prev_face = False
        self._prev_pose_valid = False
        self._prev_closure_valid = False

        # 유효 시간 누적(유지 규칙 적용). 이벤트 구간의 시작에서 값을 적어 두고 차로 비율을 구한다.
        self._face_ms = 0
        self._pose_ms = 0
        self._closure_valid_ms = 0
        self._face_window = IntervalWindow(cfg.unable_to_analyze.window_ms)
        # vats의 t_start_ms는 시작 프레임보다 과거(최근 30초 안)다. 시작할 때 "t_start_ms까지의 pose_valid 누적"을
        # 구하려고 최근 30초의 pose_valid 구간을 들고 있는다(그 용도로만 쓴다).
        self._pose_window = IntervalWindow(cfg.vats.window_ms)

        # calibrating
        self._calibrating = False
        self._calib_start_ms = 0

        # unable_to_analyze
        self._unable = False
        self._unable_start_ms = 0
        self._unable_face_ms0 = 0

        # 시선
        self._zone = ForwardZone(cfg.forward_zone, cfg.clock.max_hold_ms, cfg.vats.window_ms)
        self._gaze_active = False
        self._gaze_start_ms = 0
        self._gaze_pose_ms0 = 0
        self._gaze_source = SOURCE_HEAD_POSE
        self._vats_active = False
        self._vats_start_ms = 0
        self._vats_pose_ms0 = 0  # t_start_ms 시점의 pose_valid 누적 시간
        self._vats_source = SOURCE_HEAD_POSE

        # 감김
        self._detector = ClosureRunDetector(cfg.closure.detector)
        self._run_tracked = False  # 진행 중인 감김의 시작을 기록했는지
        self._run_start_ms = 0  # 그 감김의 start_ms(검출기의 run_start_ms()). 이벤트의 t_start_ms다
        self._run_first_ms = 0  # 그 감김을 처음 본 프레임
        self._run_valid_ms0 = 0  # 그 프레임에서의 closure_valid 누적 시간
        self._closure_level = _LEVEL_NONE

        # 졸음
        model_cfg = cfg.drowsiness.model
        perclos_cfg = cfg.drowsiness.perclos
        self._model_rule = StepHysteresis(model_cfg.onset_steps, model_cfg.end_steps, model_cfg.refractory_ms)
        self._perclos_rule = StepHysteresis(perclos_cfg.onset_steps, perclos_cfg.end_steps, perclos_cfg.refractory_ms)
        self._drowsy_active = False
        self._drowsy_start_ms = 0
        self._drowsy_source = SOURCE_PERCLOS

    @property
    def gaze_zone(self) -> str:
        """마지막 프레임까지의 전방 주시 상태(표시용). ZONE_IN, ZONE_OUT, 판단할 수 없으면 ZONE_UNKNOWN.

        확정된 상태(디바운스 뒤)를 돌려준다. 마지막 프레임의 표지가 UNKNOWN이면(자세 무효, 얼굴 소실 3초 경과,
        분석 불가) 상태와 무관하게 ZONE_UNKNOWN이다.
        """
        if self._zone.label == ZONE_UNKNOWN:
            return ZONE_UNKNOWN
        return self._zone.zone

    # ------------------------------------------------------------------
    # 프레임
    # ------------------------------------------------------------------

    def update(self, state: FrameState) -> list[DriverStateEvent]:
        """프레임 하나를 처리하고 이 프레임에서 생긴 이벤트를 발행 순서대로 돌려준다."""
        t = state.t_ms
        if self._has_prev and t <= self._prev_t_ms:
            raise ValueError("t_ms must be strictly greater than the previous frame's t_ms")
        events: list[DriverStateEvent] = []

        if not self._has_prev:
            self._first_t_ms = t
        else:
            gap_ms = t - self._prev_t_ms
            held_ms = gap_ms
            if held_ms > self._max_hold_ms:
                held_ms = self._max_hold_ms
            held_end_ms = self._prev_t_ms + held_ms
            if self._prev_face:
                self._face_ms += held_ms
                self._face_window.add(self._prev_t_ms, held_end_ms)
            if self._prev_pose_valid:
                self._pose_ms += held_ms
                self._pose_window.add(self._prev_t_ms, held_end_ms)
            if self._prev_closure_valid:
                self._closure_valid_ms += held_ms
            if gap_ms > self._reset_gap_ms:
                self._reset_after_gap(state, events)

        self._rule_calibrating(state, events)
        analyzable = self._rule_unable(state, events)
        self._rule_gaze(state, analyzable, events)
        self._rule_closure(state, analyzable, events)

        self._has_prev = True
        self._prev_t_ms = t
        self._prev_face = state.face
        self._prev_pose_valid = state.pose_valid
        self._prev_closure_valid = state.closure_valid
        return events

    def _reset_after_gap(self, state: FrameState, events: list[DriverStateEvent]) -> None:
        """10초 넘는 끊김 뒤의 초기화. 진행 중이던 상태를 끝내고 세션을 시작할 때의 상태로 되돌린다.

        `end`를 내는 순서는 drowsy, unable_to_analyze, gaze_away_long, microsleep 또는 sleep이다(모듈 설명의 "시간").
        vats의 `end`는 여기서 내지 않는다: 합을 비워 두면 이 프레임의 `_rule_gaze`가 낸다.
        calibrating 상태와 유효 시간 누적값(`_face_ms`, `_pose_ms`, `_closure_valid_ms`)은 건드리지 않는다.
        """
        t = state.t_ms

        # 졸음: 값을 실어 줄 스텝이 없으므로 confidence는 0.0이다. provisional은 다른 drowsy 이벤트처럼 False다.
        if self._drowsy_active:
            self._drowsy_active = False
            events.append(
                self._event(EVENT_DROWSY, self._drowsy_start_ms, t, 0.0, self._drowsy_source, PHASE_END, False)
            )
        self._model_rule.reset()
        self._perclos_rule.reset()

        # 분석 불가: 끝내고, 창을 비우고, 판정하지 않는 구간(처음 window_ms)을 이 프레임부터 다시 센다.
        # 직전 프레임이 유지된 시간은 호출 전에 이미 `_face_ms`에 들어갔으므로 신뢰도는 공통 규칙 그대로다.
        if self._unable:
            self._unable = False
            confidence = _ratio(self._face_ms - self._unable_face_ms0, t - self._unable_start_ms)
            events.append(
                self._event(
                    EVENT_UNABLE_TO_ANALYZE,
                    self._unable_start_ms,
                    t,
                    confidence,
                    SOURCE_FACE,
                    PHASE_END,
                    state.provisional,
                )
            )
        self._face_window.clear()
        self._first_t_ms = t

        # 시선: 전방 주시 상태와 VATS 누적
        if self._gaze_active:
            self._gaze_active = False
            events.append(self._gaze_event(state, PHASE_END))
        self._zone.reset()

        # 감김
        if self._closure_level != _LEVEL_NONE:
            events.append(self._closure_event(state, self._closure_level, PHASE_END))
            self._closure_level = _LEVEL_NONE
        self._detector.reset()
        self._run_tracked = False

    def _rule_calibrating(self, state: FrameState, events: list[DriverStateEvent]) -> None:
        t = state.t_ms
        if not self._has_prev:
            if state.provisional:
                self._calibrating = True
                self._calib_start_ms = t
                events.append(self._event(EVENT_CALIBRATING, t, t, 0.0, SOURCE_FACE, PHASE_ONSET, state.provisional))
        elif self._calibrating and not state.provisional:
            self._calibrating = False
            confidence = _ratio(self._face_ms, t - self._calib_start_ms)
            events.append(
                self._event(
                    EVENT_CALIBRATING, self._calib_start_ms, t, confidence, SOURCE_FACE, PHASE_END, state.provisional
                )
            )

    def _rule_unable(self, state: FrameState, events: list[DriverStateEvent]) -> bool:
        """분석 불가를 판정하고, 이 프레임을 시선과 감김 규칙이 써도 되는지(분석 불가가 아닌지)를 돌려준다."""
        cfg = self._cfg.unable_to_analyze
        t = state.t_ms
        self._face_window.trim(t)
        # 세션의 첫 프레임(끊김 초기화가 있었으면 그 뒤의 첫 프레임)부터 window_ms가 지나기 전에는 판정하지 않는다.
        if t - self._first_t_ms < cfg.window_ms:
            return True
        face_ratio = float(self._face_window.total_ms()) / float(cfg.window_ms)
        if not self._unable:
            if face_ratio < cfg.min_face_ratio:
                self._unable = True
                self._unable_start_ms = t
                self._unable_face_ms0 = self._face_ms
                events.append(
                    self._event(EVENT_UNABLE_TO_ANALYZE, t, t, 0.0, SOURCE_FACE, PHASE_ONSET, state.provisional)
                )
        elif face_ratio >= cfg.min_face_ratio:
            self._unable = False
            confidence = _ratio(self._face_ms - self._unable_face_ms0, t - self._unable_start_ms)
            events.append(
                self._event(
                    EVENT_UNABLE_TO_ANALYZE,
                    self._unable_start_ms,
                    t,
                    confidence,
                    SOURCE_FACE,
                    PHASE_END,
                    state.provisional,
                )
            )
        return not self._unable

    def _rule_gaze(self, state: FrameState, analyzable: bool, events: list[DriverStateEvent]) -> None:
        t = state.t_ms
        zone = self._zone
        zone.update(state, analyzable, self._pose_ms)

        # gaze_away_long
        if self._gaze_active:
            if zone.confirmed_in:
                self._gaze_active = False
                events.append(self._gaze_event(state, PHASE_END))
        elif zone.zone == ZONE_OUT and (
            zone.label == ZONE_OUT or (zone.label == ZONE_UNKNOWN and analyzable and not zone.candidate_pending)
        ):
            # 표지가 UNKNOWN인 프레임도 본다: 얼굴 소실 뒤 인정 구간 안에서 임계 시간이 찼는데 그 구간에 프레임이
            # 없었으면 여기서 낸다. UNKNOWN 동안에는 unknown_ms가 함께 늘어 연속 시간이 멈춰 있으므로 일찍 나지 않는다.
            # 분석 불가 중인 프레임(analyzable=False)과 영역 안 후보가 걸린 프레임(표지 IN)에서는 내지 않는다.
            # 표지가 OUT인 프레임에는 걸린 후보가 없다. 분석 가능한 프레임에서 표지가 UNKNOWN인데 후보가 걸려 있는 것은
            # 인정 구간이 끝나는 바로 그 시각(t = 소실 시각 + face_loss_credit_ms)에 프레임이 온 경우뿐이고, 그때는
            # 후보 구간이 아직 연속 시간에서 빠지지 않았으므로 판정하지 않는다(다음 프레임에서 후보가 버려진 뒤 판정한다).
            if t - zone.out_start_ms - zone.unknown_ms >= self._cfg.gaze_away_long.onset_ms:
                self._gaze_active = True
                self._gaze_start_ms = zone.out_start_ms
                self._gaze_pose_ms0 = zone.out_pose_ms
                self._gaze_source = zone.out_source
                events.append(self._gaze_event(state, PHASE_ONSET))

        # vats
        vats = self._cfg.vats
        if vats.reset_enabled and zone.zone == ZONE_IN and zone.label == ZONE_IN:
            if t - zone.in_start_ms >= vats.reset_forward_ms:
                zone.away.clear()
        zone.away.trim(t)
        self._pose_window.trim(t)
        away_ms = zone.away.total_ms()
        if not self._vats_active:
            if away_ms >= vats.cumulative_ms:
                self._vats_active = True
                self._vats_start_ms = zone.away.first_start_ms()
                # t_start_ms는 방금 자른 30초 창 안에 있으므로, 그 뒤의 pose_valid 시간이 전부 _pose_window에 남아 있다.
                # 전체 누적에서 빼면 t_start_ms 시점의 누적값이고, 이후의 유효 시간은 gaze_away_long처럼 차로 구한다.
                self._vats_pose_ms0 = self._pose_ms - self._pose_window.since_ms(self._vats_start_ms)
                self._vats_source = zone.out_source
                events.append(self._vats_event(state, PHASE_ONSET))
        elif away_ms < vats.cumulative_ms:
            self._vats_active = False
            events.append(self._vats_event(state, PHASE_END))

    def _rule_closure(self, state: FrameState, analyzable: bool, events: list[DriverStateEvent]) -> None:
        cfg = self._cfg.closure
        t = state.t_ms
        detector = self._detector
        valid = state.closure_valid and analyzable
        run = detector.update(t, state.closure, valid)

        # 유효한 감김 프레임(closure ≥ 진입)에서 진행 중인 감김의 길이와 시작을 검출기에서 읽는다.
        # 길이(ongoing_ms)는 closure가 진입 임계 이상이었던 시간의 합이라 t − start_ms보다 짧을 수 있다.
        # 그래서 감김의 시작은 길이에서 거꾸로 계산하지 않고 run_start_ms()로 받는다.
        closed_frame = valid and state.closure >= cfg.detector.enter_thr and detector.state != STATE_OPEN
        ongoing_ms = 0
        start_ms = 0
        if closed_frame:
            ongoing_ms = detector.ongoing_ms()
            start_ms = detector.run_start_ms()

        # 기록해 둔 감김이 끝났으면(같은 프레임에서 새 감김이 시작됐어도) 도달한 가장 높은 종류의 end를 하나 낸다.
        ended = run is not None or detector.state == STATE_OPEN
        if closed_frame and self._run_tracked and start_ms != self._run_start_ms:
            ended = True
        if ended:
            if self._closure_level != _LEVEL_NONE:
                events.append(self._closure_event(state, self._closure_level, PHASE_END))
                self._closure_level = _LEVEL_NONE
            self._run_tracked = False

        # 임계 시간은 유효한 감김 프레임에서만 판정한다.
        if closed_frame:
            if not self._run_tracked:
                self._run_tracked = True
                self._run_start_ms = start_ms
                self._run_first_ms = t
                self._run_valid_ms0 = self._closure_valid_ms
            if self._closure_level < _LEVEL_MICROSLEEP and ongoing_ms >= cfg.microsleep_ms:
                self._closure_level = _LEVEL_MICROSLEEP
                events.append(self._closure_event(state, _LEVEL_MICROSLEEP, PHASE_ONSET))
            if self._closure_level < _LEVEL_SLEEP and ongoing_ms >= cfg.sleep_ms:
                self._closure_level = _LEVEL_SLEEP
                events.append(self._closure_event(state, _LEVEL_SLEEP, PHASE_ONSET))

    # ------------------------------------------------------------------
    # 스텝
    # ------------------------------------------------------------------

    def on_step(self, step: StepFeatures, probs: Sequence[float] | None) -> list[DriverStateEvent]:
        """스텝 하나를 처리하고 drowsy 이벤트(없으면 빈 목록)를 돌려준다."""
        cfg = self._cfg.drowsiness
        use_model = cfg.source == DROWSY_SOURCE_MODEL or cfg.source == DROWSY_SOURCE_EITHER
        use_perclos = cfg.source == DROWSY_SOURCE_PERCLOS or cfg.source == DROWSY_SOURCE_EITHER

        has_p = probs is not None
        p_drowsy = 0.0
        if probs is not None:
            p_drowsy = probs[CLASS_DROWSY]
        perclos = step.values[F_PERCLOS_60]

        if use_model:
            self._model_rule.update(
                step.k, step.t_end_ms, has_p, p_drowsy >= cfg.model.onset_p, p_drowsy <= cfg.model.end_p
            )
        if use_perclos:
            self._perclos_rule.update(
                step.k,
                step.t_end_ms,
                step.full_lookback,
                perclos >= cfg.perclos.onset_ratio,
                perclos < cfg.perclos.end_ratio,
            )
        model_on = use_model and self._model_rule.active
        perclos_on = use_perclos and self._perclos_rule.active

        if not self._drowsy_active and (model_on or perclos_on):
            self._drowsy_active = True
            if model_on:
                self._drowsy_source = SOURCE_MODEL
                self._drowsy_start_ms = self._model_rule.start_ms
            else:
                self._drowsy_source = SOURCE_PERCLOS
                self._drowsy_start_ms = self._perclos_rule.start_ms
            return [self._drowsy_event(step, p_drowsy, PHASE_ONSET)]
        if self._drowsy_active and not (model_on or perclos_on):
            self._drowsy_active = False
            return [self._drowsy_event(step, p_drowsy, PHASE_END)]
        return []

    # ------------------------------------------------------------------
    # 이벤트 만들기
    # ------------------------------------------------------------------

    def _event(
        self,
        event_type: str,
        t_start_ms: int,
        t_emit_ms: int,
        confidence: float,
        source: str,
        phase: str,
        provisional: bool,
    ) -> DriverStateEvent:
        return DriverStateEvent(
            type=event_type,
            severity=self._cfg.events.get(event_type).severity,
            t_start_ms=t_start_ms,
            duration_ms=t_emit_ms - t_start_ms,
            confidence=confidence,
            source=source,
            phase=phase,
            provisional=provisional,
        )

    def _gaze_event(self, state: FrameState, phase: str) -> DriverStateEvent:
        t = state.t_ms
        confidence = _ratio(self._pose_ms - self._gaze_pose_ms0, t - self._gaze_start_ms)
        return self._event(
            EVENT_GAZE_AWAY_LONG, self._gaze_start_ms, t, confidence, self._gaze_source, phase, state.provisional
        )

    def _vats_event(self, state: FrameState, phase: str) -> DriverStateEvent:
        t = state.t_ms
        # [t_start_ms, 발행 시각] 전체의 pose_valid 시간 비율. 구간이 30초보다 길어도(end는 대개 그렇다) 잘리지 않는다.
        confidence = _ratio(self._pose_ms - self._vats_pose_ms0, t - self._vats_start_ms)
        return self._event(EVENT_VATS, self._vats_start_ms, t, confidence, self._vats_source, phase, state.provisional)

    def _drowsy_event(self, step: StepFeatures, p_drowsy: float, phase: str) -> DriverStateEvent:
        confidence = p_drowsy
        if self._drowsy_source == SOURCE_PERCLOS:
            confidence = 1.0 - step.values[F_FACE_MISSING_RATIO_60]
        return self._event(
            EVENT_DROWSY, self._drowsy_start_ms, step.t_end_ms, confidence, self._drowsy_source, phase, False
        )

    def _closure_event(self, state: FrameState, level: int, phase: str) -> DriverStateEvent:
        t = state.t_ms
        event_type = EVENT_MICROSLEEP
        if level == _LEVEL_SLEEP:
            event_type = EVENT_SLEEP
        confidence = _ratio(self._closure_valid_ms - self._run_valid_ms0, t - self._run_first_ms)
        # t_start_ms는 감김의 start_ms, duration_ms는 공통 규칙대로 t − t_start_ms다. 임계 시간을 판정한 길이
        # (진입 임계 이상이었던 시간의 합)가 아니므로, 감김 도중에 반쯤 뜬 시간이 있었으면 그만큼 더 길다.
        return self._event(event_type, self._run_start_ms, t, confidence, SOURCE_CLOSURE, phase, state.provisional)


class AlertPolicy:
    """경고 결정 (PLAN 8.2절, D3). 상태 이벤트와 차량 속도를 받아 실제로 경고할 이벤트를 돌려준다.

    프레임마다 `decide`를 한 번 부른다(이벤트가 없어도 부른다. 미뤄 둔 경고와 되풀이 경고의 시각을 재야 한다).
    - 경고가 되는 것은 `onset` 이벤트뿐이다. `end`는 상태가 끝났다는 것만 기록한다.
    - calibrating은 경고하지 않는다.
    - 재경고 대기: 같은 종류의 마지막 경고 뒤 `cooldown_ms`가 지나기 전에는 경고하지 않는다(0이면 대기 없음).
    - 속도 억제(D3): 속도가 `min_speed_kmh` 미만이면 경고하지 않는다. 속도가 None이거나 기준이 0이면 억제하지 않는다.
      억제는 경고에만 적용된다(상태 이벤트는 규칙 엔진이 그대로 낸다).
    - 미뤄 둔 경고(2026-11-13 소유자 결정. calibrating을 뺀 모든 종류): `onset`이 온 호출에서 대기나 속도에 막혀
      경고하지 못하면 그 `onset`을 종류별로 하나 미뤄 둔다. 이후의 호출마다, 그 호출의 이벤트를 모두 처리한 뒤에,
      미뤄 둔 종류가 더는 막히지 않으면 한 번 경고하고 미뤄 둔 것을 지운다. 이 경고는 미뤄 둔 `onset`의 사본이고
      `duration_ms`만 지금 시각 기준(`t_ms − t_start_ms`)으로 고친다. 상태가 끝나면 경고 없이 지운다.
        · 그 종류의 `end`가 온 호출에서 지운다(막힘이 풀린 바로 그 호출에 `end`가 함께 와도 경고하지 않는다).
        · microsleep은 sleep의 `onset`이나 `end`가 와도 지운다. 규칙 엔진은 한 감김이 도달한 가장 높은 종류의
          `end` 하나만 내므로, sleep까지 간 감김의 microsleep에는 자기 `end`가 오지 않는다.
      한 `onset`은 바로든 미뤄서든 한 번만 경고한다. 상태가 오래 이어져도 주기적으로 다시 경고하지 않는다
      (다시 경고하는 것은 아래의 되풀이뿐이다).
    - 되풀이(`repeat_ms` > 0, 기본 설정에서는 sleep의 3초): 상태가 이어지는 동안 마지막 경고 뒤 `repeat_ms`가
      지날 때마다 다시 경고한다. 되풀이 경고는 시작 이벤트의 사본이고 `duration_ms`만 지금 시각 기준으로 고친다.
      시작 경고가 막혔으면 상태가 이어지는 동안 막힘이 풀린 첫 호출에서 경고하고 거기서부터 주기를 센다
      (미뤄 둔 경고와 결과가 같아서, 되풀이하는 종류는 따로 미뤄 두지 않는다).
    - 우선순위: 한 번의 호출에서 여러 경고(바로 낸 것, 미뤄 둔 것, 되풀이)가 나오면 `alert_priority` 순서
      (높은 것부터)로 돌려준다.

    종류별로 상태가 끝나는 길(미뤄 둔 경고가 상태보다 오래 남지 않는 근거)
    - unable_to_analyze, gaze_away_long, vats, drowsy, sleep: 규칙 엔진은 상태를 끝낼 때 반드시 그 종류의 `end`를
      낸다. 10초 넘는 끊김의 초기화도 `end`를 낸다(vats는 같은 프레임에서 비워진 합 때문에 `end`가 난다).
    - microsleep: 자기 `end`, 또는 위의 sleep `onset`·`end`.
    - calibrating: 경고하지 않으므로 미뤄 두지 않는다.

    가정: 같은 호출에서 통과한 경고는 모두 돌려준다(낮은 순위를 버리지 않는다). 앱은 첫 원소를 대표로 쓰면 된다.
    가정: 억제되거나 대기에 걸린 `onset`은 재경고 대기를 새로 시작하지 않는다(대기는 실제로 낸 경고부터 센다).
    가정: 미뤄 둔 종류의 `onset`이 `end` 없이 다시 오면(규칙 엔진은 그렇게 내지 않는다) 새 `onset`으로 바꾼다.
    가정: sleep의 `onset`·`end`는 microsleep의 되풀이 기록도 함께 지운다(microsleep에 `repeat_ms`를 준 설정에서
      감김이 끝난 뒤에도 microsleep 경고가 되풀이되는 것을 막는다. 기본 설정에서는 해당 없음).
    가정: unable_to_analyze의 "속도 억제 없음"은 설정값(`min_speed_kmh: 0`)으로 표현한다.
    """

    def __init__(self, cfg: RulesConfig) -> None:
        self._cfg = cfg
        self._last_alert_ms: dict[str, int] = {}  # 종류 -> 마지막 경고 시각
        self._pending: dict[str, DriverStateEvent] = {}  # 되풀이하지 않는 종류 -> 막혀서 미뤄 둔 시작 이벤트
        self._repeat_onset: dict[str, DriverStateEvent] = {}  # 되풀이하는 종류 -> 이어지는 상태의 시작 이벤트
        self._repeat_alert_ms: dict[str, int] = {}  # 되풀이하는 종류 -> 그 상태에서 낸 마지막 경고 시각

    def decide(
        self, events: Sequence[DriverStateEvent], speed_kmh: float | None, t_ms: int
    ) -> list[DriverStateEvent]:
        """이 프레임에서 실제로 경고할 이벤트를 우선순위 순으로 돌려준다."""
        passed: list[DriverStateEvent] = []

        # 1. 이 호출의 이벤트. 상태가 끝났으면 그 종류의 기록을 지우고, 시작이면 경고하거나 미뤄 둔다.
        for event in events:
            policy = self._cfg.events.get(event.type)
            if event.phase == PHASE_END:
                self._forget(event.type)
                if event.type == EVENT_SLEEP:
                    self._forget(EVENT_MICROSLEEP)
                continue
            if event.type == EVENT_CALIBRATING:
                continue
            self._forget(event.type)
            if event.type == EVENT_SLEEP:
                self._forget(EVENT_MICROSLEEP)
            if policy.repeat_ms > 0:
                self._repeat_onset[event.type] = event
            if self._allowed(event.type, speed_kmh, t_ms):
                self._mark(event.type, t_ms)
                passed.append(event)
            elif policy.repeat_ms == 0:
                self._pending[event.type] = event

        # 2. 미뤄 둔 경고: 막힘이 풀렸으면 한 번 내고 지운다. 이 호출에서 방금 미뤄 둔 것은 같은 조건으로 다시
        #    보는 것이라 여기서 통과하지 못한다(그래서 "이후의 호출"에서만 나간다).
        for event_type in self._cfg.alert_priority:
            if event_type not in self._pending:
                continue
            if self._allowed(event_type, speed_kmh, t_ms):
                self._mark(event_type, t_ms)
                passed.append(_reissued(self._pending[event_type], t_ms))
                del self._pending[event_type]

        # 3. 되풀이
        for event_type in self._cfg.alert_priority:
            if event_type not in self._repeat_onset:
                continue
            if event_type in self._repeat_alert_ms:
                if t_ms - self._repeat_alert_ms[event_type] < self._cfg.events.get(event_type).repeat_ms:
                    continue
            if self._allowed(event_type, speed_kmh, t_ms):
                self._mark(event_type, t_ms)
                passed.append(_reissued(self._repeat_onset[event_type], t_ms))

        alerts: list[DriverStateEvent] = []
        for event_type in self._cfg.alert_priority:
            for event in passed:
                if event.type == event_type:
                    alerts.append(event)
        return alerts

    def _allowed(self, event_type: str, speed_kmh: float | None, t_ms: int) -> bool:
        policy = self._cfg.events.get(event_type)
        if event_type in self._last_alert_ms:
            if t_ms - self._last_alert_ms[event_type] < policy.cooldown_ms:
                return False
        if speed_kmh is not None and policy.min_speed_kmh > 0.0 and speed_kmh < policy.min_speed_kmh:
            return False
        return True

    def _mark(self, event_type: str, t_ms: int) -> None:
        self._last_alert_ms[event_type] = t_ms
        if event_type in self._repeat_onset:
            self._repeat_alert_ms[event_type] = t_ms

    def _forget(self, event_type: str) -> None:
        """그 종류의 상태가 끝났거나 새로 시작됐다: 미뤄 둔 경고와 되풀이 기록을 지운다(마지막 경고 시각은 남긴다)."""
        if event_type in self._pending:
            del self._pending[event_type]
        if event_type in self._repeat_onset:
            del self._repeat_onset[event_type]
        if event_type in self._repeat_alert_ms:
            del self._repeat_alert_ms[event_type]


def _reissued(onset: DriverStateEvent, t_ms: int) -> DriverStateEvent:
    """시작 이벤트의 사본. `duration_ms`만 지금 시각 기준으로 고친다(미뤄 둔 경고와 되풀이 경고)."""
    return DriverStateEvent(
        type=onset.type,
        severity=onset.severity,
        t_start_ms=onset.t_start_ms,
        duration_ms=t_ms - onset.t_start_ms,
        confidence=onset.confidence,
        source=onset.source,
        phase=onset.phase,
        provisional=onset.provisional,
    )


def _ratio(numerator_ms: int, denominator_ms: int) -> float:
    """시간 비율. 분모가 0 이하이면 0.0이다(PLAN 3.3절)."""
    if denominator_ms <= 0:
        return 0.0
    return float(numerator_ms) / float(denominator_ms)
