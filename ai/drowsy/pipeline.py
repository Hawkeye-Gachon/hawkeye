"""`DriverPipeline`: 이식 대상 모듈을 한 줄로 잇는 스트리밍 파이프라인 (이식 대상, 표준 라이브러리만 사용).

PLAN 3.1절(코드 경로는 하나), 3.3절(시간 규칙), 4.4절(세션 시간 흐름), 9장.
학습 특징 생성, 데모, 테스트 벡터 생성이 모두 이 클래스를 프레임 하나씩 호출한다.

프레임 하나의 처리 순서
 1. `t_ms`가 직전에 받아들인 프레임 이하이면 버리고 센다(None을 돌려준다).
    처음 받아들인 프레임의 `t_ms`가 세션 원점이다. 세션 시간 = `t_ms` − 원점.
 2. 원시 신호 (`signals.compute`).
 3. 기준선: 완성 전에는 `BaselineEstimator`에 넣는다. 완성된 프레임부터 개인 기준선을 쓴다.
 4. 프레임 상태 (`calibrate.frame_state`). 기준선 완성 전에는 전역 상수 기준선과 provisional=True (D4).
 5. 세션 시간이 누적 시작 시각(T_b) 이상이면 특징용 감김 검출기(진입 0.5 / 이탈 0.3)와 `StepAggregator`.
    직전 프레임과의 간격이 `clock.reset_gap_ms`(10초)를 넘으면 검출기를 먼저 초기화한다
    (`StepAggregator`와 `RuleEngine`은 같은 조건으로 스스로 초기화한다).
 6. 스텝과 모델 (2026-11-13 소유자 결정: 밀린 스텝마다 모델을 부른다). 이 프레임이 발행시킨 스텝마다, 오래된
    것부터 차례로:
    a. 모델 창에 넣는다. 창은 스텝 번호가 이어진 `full_lookback` 스텝을 최대 `window_steps`(120)개 담는다.
       조회 구간이 덜 찬 스텝이나 번호가 건너뛴 스텝을 만나면 창을 비운다.
    b. 모델이 있고 창에 120개가 찼으면 그 스텝까지의 창으로 모델을 부른다. 입력은 120 x 17 원시 특징이고
       가장 오래된 스텝이 먼저다. 아니면 그 스텝의 확률은 없다(None).
    c. `RuleEngine.on_step`에 그 스텝과 그 스텝의 확률을 준다.
    프레임 하나가 스텝을 n개 발행시키면(프레임이 끊겼다가 왔을 때) 모델을 최대 n번 부른다. 스텝을 하나만
    발행시키는 프레임에서는 예전과 같다. 오프라인 경보 평가(`evaluate.alarm_videos`)도 창이 찬 모든 스텝에 확률을 주고
    10초 넘는 끊김에서는 `RuleEngine`과 같은 자리에서 판정기를 다시 시작하므로, 모델 졸음 판정은 끊김의 길이와 무관하게
    앱과 평가에서 같다. 남는 차이는 앱의 PERCLOS 판정기가 모델보다 119스텝 먼저 값을 받는 것 하나다(`alarm_videos`의 설명).
 7. 프레임 규칙: `RuleEngine.update`.
 8. 경고: `AlertPolicy.decide`.

기준선을 넣어 준 경우(학습용): 보정을 건너뛰고 첫 프레임부터 그 기준선을 쓴다(provisional=False,
calibrating 이벤트 없음). 특징 누적은 `feature_start_ms`(세션 시간)부터 시작한다. 그 전의 프레임은 특징용
검출기와 누적기에 넣지 않으므로, 같은 영상을 자기 기준선으로 돌렸을 때와 같은 프레임에서 누적이 시작된다.

가정: 한 프레임의 이벤트는 스텝 이벤트(drowsy)를 먼저, 프레임 이벤트를 나중에 적는다
  (스텝은 이 프레임보다 앞선 시간을 대표한다).
가정: 10초 넘는 끊김 뒤의 첫 프레임에서 규칙 엔진의 초기화는 `RuleEngine.update` 안에서, 곧 그 프레임의 스텝을
  처리한 뒤에 일어난다. 그 프레임의 이벤트는 초기화의 `end`(drowsy, unable_to_analyze, gaze_away_long,
  microsleep 또는 sleep 순), 그 프레임의 판정 순이다. 그 프레임이 발행시킨 스텝은 졸음 판정을 바꾸지 못한다:
  `StepAggregator`가 같은 끊김으로 다시 시작하므로 스텝이 없고, 누적기가 그 프레임에서 처음 프레임을 받는 경우
  (끊김이 누적 시작 시각에 걸친 경우)에만 스텝이 나오는데 데이터가 없는 `full_lookback=False` 스텝이라
  PERCLOS 판정에도 모델 창에도 값이 없다. 그래서 스텝 이벤트가 초기화의 drowsy `end`보다 앞에 오는 일은 없다.
가정: 기준선을 주고 `feature_start_ms`를 주지 않으면 `baseline.t_complete_ms`를 쓴다.
가정: `rules.clock`은 `spec.clock`과 같아야 한다(다르면 누적기와 규칙 엔진의 초기화 시점이 어긋난다). 아니면 ValueError.
가정: 모델 입력에 유한하지 않은 값이 있으면 추론을 건너뛰고 센다(PLAN 4.6절: 유한하지 않으면 추론을 건너뛴다).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from drowsy import calibrate, signals
from drowsy.blink import ClosureRunDetector
from drowsy.calibrate import BaselineDiagnostics, BaselineEstimator
from drowsy.rules import AlertPolicy, RuleEngine
from drowsy.spec import PipelineSpec, RulesConfig
from drowsy.step_features import StepAggregator
from drowsy.types import (
    Baseline,
    ClosureRun,
    DriverStateEvent,
    PipelineOutput,
    RawFrame,
    StepFeatures,
)

ModelFn = Callable[[Sequence[Sequence[float]]], Sequence[float]]


class DriverPipeline:
    """프레임 하나를 받아 `PipelineOutput` 하나를 돌려준다. 처리 순서는 모듈 설명에 있다.

    - `baseline`: 주면 보정을 건너뛰고 이 기준선을 쓴다.
    - `feature_start_ms`: 기준선을 줄 때 특징 누적을 시작할 세션 시간. `clock.step_ms`의 배수여야 한다.
    - `model`: 120 x 17 원시 스텝 특징(가장 오래된 스텝이 먼저)을 받아 (alert, low_vigilant, drowsy) 확률을
      돌려주는 함수. None이면 확률은 항상 None이다.

    세는 값: `frames_accepted`, `dropped_frames`(버린 프레임), `steps_emitted`, `model_calls`(모델을 부른 횟수.
    스텝마다 세므로 한 프레임에서 여럿일 수 있다), `nonfinite_windows`(유한하지 않은 값 때문에 추론을 건너뛴 횟수.
    역시 스텝마다 센다).
    읽기 전용 상태: `baseline`, `baseline_diagnostics`, `gaze_zone`.
    """

    def __init__(
        self,
        spec: PipelineSpec,
        rules: RulesConfig,
        baseline: Baseline | None = None,
        feature_start_ms: int | None = None,
        model: ModelFn | None = None,
    ) -> None:
        if rules.clock != spec.clock:
            raise ValueError("rules.clock must equal spec.clock")
        if baseline is None and feature_start_ms is not None:
            raise ValueError("feature_start_ms requires an injected baseline")

        self._spec = spec
        self._model = model
        self._reset_gap_ms = spec.clock.reset_gap_ms
        self._window_steps = spec.step_features.window_steps

        self._global_baseline = calibrate.global_baseline(spec)
        self._baseline = baseline
        self._estimator: BaselineEstimator | None = None
        self._feature_start_ms = 0
        if baseline is None:
            self._estimator = BaselineEstimator(spec)
        else:
            self._feature_start_ms = baseline.t_complete_ms
            if feature_start_ms is not None:
                self._feature_start_ms = feature_start_ms
            if self._feature_start_ms < 0 or self._feature_start_ms % spec.clock.step_ms != 0:
                raise ValueError("feature_start_ms must be a non-negative multiple of clock.step_ms")

        self._has_frame = False
        self._origin_ms = 0
        self._prev_t_ms = 0

        self._detector = ClosureRunDetector(spec.run_detector)
        self._aggregator: StepAggregator | None = None
        self._engine = RuleEngine(rules)
        self._policy = AlertPolicy(rules)

        # 모델 창: 번호가 이어진 full_lookback 스텝의 특징, 오래된 것부터 최대 window_steps개
        self._window: list[tuple[float, ...]] = []
        self._window_has_k = False
        self._window_last_k = 0

        self.frames_accepted = 0
        self.dropped_frames = 0
        self.steps_emitted = 0
        self.model_calls = 0
        self.nonfinite_windows = 0

    @property
    def baseline(self) -> Baseline | None:
        """지금 쓰는 개인 기준선. 보정이 끝나기 전에는 None이다."""
        return self._baseline

    @property
    def baseline_diagnostics(self) -> BaselineDiagnostics | None:
        """이 세션에서 기준선을 확정했을 때의 판정 통계. 기준선을 넣어 줬거나 아직 확정 전이면 None이다."""
        if self._estimator is None:
            return None
        return self._estimator.diagnostics

    @property
    def gaze_zone(self) -> str:
        """마지막 프레임까지의 전방 주시 상태(표시용). `RuleEngine.gaze_zone`과 같다."""
        return self._engine.gaze_zone

    def update(self, frame: RawFrame, speed_kmh: float | None = None) -> PipelineOutput | None:
        """프레임 하나를 처리한다. 버린 프레임(`t_ms`가 직전에 받아들인 값 이하)이면 None을 돌려준다."""
        t_ms = frame.t_ms
        if self._has_frame and t_ms <= self._prev_t_ms:
            self.dropped_frames += 1
            return None

        long_gap = False
        if not self._has_frame:
            self._has_frame = True
            self._origin_ms = t_ms
        elif t_ms - self._prev_t_ms > self._reset_gap_ms:
            long_gap = True
        self._prev_t_ms = t_ms
        self.frames_accepted += 1
        session_ms = t_ms - self._origin_ms

        frame_signals = signals.compute(frame, self._spec)

        if self._baseline is None and self._estimator is not None:
            completed = self._estimator.update(frame_signals)
            if completed is not None:
                self._baseline = completed
                self._feature_start_ms = completed.t_complete_ms
        if self._baseline is None:
            state = calibrate.frame_state(frame_signals, self._global_baseline, self._spec, True)
        else:
            state = calibrate.frame_state(frame_signals, self._baseline, self._spec, False)

        run: ClosureRun | None = None
        steps: list[StepFeatures] = []
        if self._baseline is not None and session_ms >= self._feature_start_ms:
            if self._aggregator is None:
                self._aggregator = StepAggregator(self._spec, self._feature_start_ms, self._origin_ms)
            if long_gap:
                self._detector.reset()
            run = self._detector.update(t_ms, state.closure, state.closure_valid)
            steps = self._aggregator.update(state, run, self._detector.ongoing_ms())

        # 스텝마다: 모델 창에 넣고, 창이 찼으면 그 시점의 창으로 모델을 부르고, 그 확률로 졸음 판정을 한다.
        events: list[DriverStateEvent] = []
        step_probs: list[tuple[float, float, float] | None] = []
        probs: tuple[float, float, float] | None = None
        n_steps = len(steps)
        self.steps_emitted += n_steps
        for i in range(n_steps):
            self._push_step(steps[i])
            probs = None
            if self._model is not None and len(self._window) == self._window_steps:
                probs = self._call_model()
            step_probs.append(probs)
            events.extend(self._engine.on_step(steps[i], probs))
        # 여기서 probs는 마지막 스텝의 확률이다(스텝이 없으면 None).
        events.extend(self._engine.update(state))
        alerts = self._policy.decide(events, speed_kmh, t_ms)

        runs: tuple[ClosureRun, ...] = ()
        if run is not None:
            runs = (run,)
        return PipelineOutput(
            state=state,
            runs=runs,
            steps=tuple(steps),
            step_probs=tuple(step_probs),
            probs=probs,
            events=tuple(events),
            alerts=tuple(alerts),
        )

    def _push_step(self, step: StepFeatures) -> None:
        """모델 창을 갱신한다. 조회 구간이 덜 찬 스텝이나 번호가 건너뛴 스텝을 만나면 창을 비운다."""
        if not step.full_lookback:
            self._window = []
        elif self._window_has_k and step.k != self._window_last_k + 1:
            self._window = []
        if step.full_lookback:
            self._window.append(step.values)
            if len(self._window) > self._window_steps:
                del self._window[0]
        self._window_has_k = True
        self._window_last_k = step.k

    def _call_model(self) -> tuple[float, float, float] | None:
        for values in self._window:
            for value in values:
                if not _is_finite(value):
                    self.nonfinite_windows += 1
                    return None
        if self._model is None:
            return None
        out = self._model(list(self._window))
        if len(out) != 3:
            raise ValueError("model must return 3 probabilities (alert, low_vigilant, drowsy)")
        self.model_calls += 1
        return (float(out[0]), float(out[1]), float(out[2]))


def _is_finite(x: float) -> bool:
    """NaN과 ±무한대에서만 False다(Kotlin의 `Double.isFinite()`)."""
    return x - x == 0.0
