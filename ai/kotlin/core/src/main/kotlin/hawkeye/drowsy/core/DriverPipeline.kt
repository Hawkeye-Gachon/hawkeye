package hawkeye.drowsy.core

import hawkeye.drowsy.core.internal.AlertPolicy
import hawkeye.drowsy.core.internal.BaselineEstimator
import hawkeye.drowsy.core.internal.ClosureRunDetector
import hawkeye.drowsy.core.internal.FrameSignals
import hawkeye.drowsy.core.internal.RuleEngine
import hawkeye.drowsy.core.internal.StepAggregator
import hawkeye.drowsy.core.internal.compute
import hawkeye.drowsy.core.internal.frameState
import hawkeye.drowsy.core.internal.globalBaseline
import hawkeye.drowsy.core.spec.PipelineSpec
import hawkeye.drowsy.core.spec.RulesConfig
import hawkeye.drowsy.core.spec.SpecValidation
import java.util.Collections

// 원본: drowsy/pipeline.py (SPEC 6, 7, 14장). 계획 5.7절. `internal`이다: 앱이 시작 검사를 건너뛴 파이프라인을 만들 길이
// 없게 한다(계획 4.1절). 벡터 시험은 같은 Gradle 모듈의 시험이라 기준선 주입 생성자를 쓴다.
//
// 처리 순서(SPEC 7.2의 1~9단계)와 원본이 정한 가정(pipeline.py:33-43: 한 프레임의 이벤트는 스텝 이벤트를 먼저 적는다,
// 끊김 뒤 첫 프레임의 규칙 엔진 초기화는 RuleEngine.update 안에서 그 프레임의 스텝을 처리한 뒤에 일어난다, 기준선을 주고
// feature_start_ms를 주지 않으면 baseline.t_complete_ms를 쓴다, rules.clock은 spec.clock과 같아야 한다, 모델 입력에 유한하지
// 않은 값이 있으면 추론을 건너뛰고 센다)은 원본 모듈 설명과 SPEC 7장에 있다. 이 파일은 그것을 문장 순서 그대로 옮긴다.
//
// 세 입구가 같은 코드를 지나도록 update를 private 함수 셋으로 나눈다(계획 5.7절, pipeline.py:159-215의 순서 그대로):
// accept(tMs) → (signals.compute) → stateFromSignals(sig) → advance(accepted, state, speedKmh).
// - update = accept → compute → stateFromSignals → advance.
// - updateFromSignals(차등 비교 SIG 모드) = accept(sig.tMs) → stateFromSignals → advance.
// - updateFromState(차등 비교 STATE 모드) = accept(state.tMs) → (기준선 대입만) → advance.

/**
 * pipeline.py:68-244 `DriverPipeline`. 프레임 하나를 받아 PipelineOutput 하나를 돌려준다.
 *
 * - baseline: 주면 보정을 건너뛰고 이 기준선을 쓴다(학습과 벡터용).
 * - featureStartMs: 기준선을 줄 때 특징 누적을 시작할 세션 시간. clock.step_ms의 배수여야 한다. 주지 않으면
 *   baseline.tCompleteMs다(원본의 가정, pipeline.py:41).
 * - model: 120 x 17 원시 스텝 특징(가장 오래된 스텝이 먼저)을 받아 (alert, low_vigilant, drowsy) 확률을 돌려주는 함수.
 *   null이면 확률은 언제나 null이다.
 *
 * 생성 검사 셋(pipeline.py:90-93, 105-111): rules.clock != spec.clock, featureStartMs인데 baseline 없음, featureStart가
 * 음수이거나 step_ms의 배수가 아님 → IllegalArgumentException(Python은 ValueError, 메시지는 같다).
 * 세는 값: framesAccepted, droppedFrames, stepsEmitted, modelCalls(스텝마다 세므로 한 프레임에서 여럿일 수 있다),
 * nonfiniteWindows(유한하지 않은 값 때문에 추론을 건너뛴 횟수, 역시 스텝마다). 읽기 전용 상태: baseline,
 * baselineDiagnostics, gazeZone. 스레드에 안전하지 않다. 운전 한 번에 인스턴스 하나다(G11).
 */
internal class DriverPipeline internal constructor(
    private val spec: PipelineSpec,
    rules: RulesConfig,
    baseline: Baseline?,
    featureStartMs: Long?,
    private val model: ModelFn?,
) {
    // 특징 누적을 시작할 세션 시간(T_b). 기준선이 없으면 0이고, 기준선이 확정되는 프레임에서 정해진다.
    // pipeline.py:90-93, 104-111: 생성 검사 셋의 순서와 메시지는 Python 그대로다(SpecValidation, 단계 1).
    // Kotlin은 속성 초기화를 적힌 순서대로 돌리므로 이 검사를 첫 초기화로 두어, 아래의 어떤 객체(전역 기준선, 추정기,
    // 검출기, 엔진, 정책, 창)보다 먼저 돌게 한다. 원본은 검사 둘 뒤에 global_baseline을 만들고(pipeline.py:100) 셋째 검사는
    // 기준선을 줄 때만 한다. global_baseline은 예외를 내지 않으므로 셋을 모두 먼저 해도 판정과 메시지가 같다.
    // 검사를 거치지 않은 spec(velocity_lookback_samples < 0)에서 검출기가 먼저 만들어지면 Python의 ValueError 대신
    // NegativeArraySizeException이 나왔다(f_s4, DriverPipelineConstructionOrderTest).
    private var featureStartMsValue: Long =
        SpecValidation.checkPipelineConstruction(spec, rules, baseline, featureStartMs)

    private val resetGapMs: Long = spec.clock.resetGapMs
    private val windowSteps: Int = spec.stepFeatures.windowSteps

    private val globalBaseline: Baseline = globalBaseline(spec)
    private var baselineValue: Baseline? = baseline
    private val estimator: BaselineEstimator? = if (baseline == null) BaselineEstimator(spec) else null

    private var hasFrame = false
    private var originMs: Long = 0L
    private var prevTMs: Long = 0L

    private val detector = ClosureRunDetector(spec.runDetector)
    private var aggregator: StepAggregator? = null
    private val engine: RuleEngine = RuleEngine(rules)
    private val policy: AlertPolicy = AlertPolicy(rules)

    // 모델 창: 번호가 이어진 full_lookback 스텝의 특징, 오래된 것부터 최대 windowSteps개(R13: ArrayDeque).
    // 창에는 step.values(읽기 전용 목록)를 그대로 넣는다(계획 5.7절).
    private val window = ArrayDeque<List<Double>>()
    private var windowHasK = false
    private var windowLastK: Int = 0

    /** 받아들인 프레임 수 (pipeline.py:127). */
    var framesAccepted: Long = 0L
        private set

    /** 버린 프레임 수: tMs가 직전에 받아들인 값 이하 (pipeline.py:128). */
    var droppedFrames: Long = 0L
        private set

    /** 발행한 스텝 수 (pipeline.py:129). */
    var stepsEmitted: Long = 0L
        private set

    /** 모델을 부른 횟수 (pipeline.py:130). 스텝마다 센다. */
    var modelCalls: Long = 0L
        private set

    /** 유한하지 않은 값 때문에 추론을 건너뛴 횟수 (pipeline.py:131). 스텝마다 센다. */
    var nonfiniteWindows: Long = 0L
        private set

    /** pipeline.py:133-136. 지금 쓰는 개인 기준선. 보정이 끝나기 전에는 null이다. */
    val baseline: Baseline?
        get() = baselineValue

    /** pipeline.py:138-143. 이 세션에서 기준선을 확정했을 때의 판정 통계. 기준선을 넣어 줬거나 확정 전이면 null이다. */
    val baselineDiagnostics: BaselineDiagnostics?
        get() = estimator?.diagnostics

    /** pipeline.py:145-148. 마지막 프레임까지의 전방 주시 상태(표시용). RuleEngine.gazeZone과 같다(읽을 때마다 계산). */
    val gazeZone: GazeZone
        get() = engine.gazeZone

    /** accept가 받아들인 프레임 하나의 값 (pipeline.py:152-165). */
    private class Accepted(val tMs: Long, val longGap: Boolean, val sessionMs: Long)

    /**
     * pipeline.py:150-217 `update`. 프레임 하나를 처리한다. 버린 프레임(tMs가 직전에 받아들인 값 이하)이면 null을 돌려주고
     * 다른 상태는 하나도 바꾸지 않는다(SPEC 6장).
     */
    fun update(frame: RawFrame, speedKmh: Double?): PipelineOutput? {
        val accepted = accept(frame.tMs) ?: return null
        val sig = compute(frame, spec)
        val state = stateFromSignals(sig)
        return advance(accepted, state, speedKmh)
    }

    /**
     * 시험 입구(차등 비교 SIG 모드, 계획 9.2.1절): 참조의 FrameSignals를 받아 기준선 추정, frameState와 그 뒤 전부를 돈다.
     * update와 같은 코드를 지난다(signals.compute만 건너뛴다).
     */
    fun updateFromSignals(sig: FrameSignals, speedKmh: Double?): PipelineOutput? {
        val accepted = accept(sig.tMs) ?: return null
        val state = stateFromSignals(sig)
        return advance(accepted, state, speedKmh)
    }

    /**
     * 시험 입구(차등 비교 STATE 모드, 계획 5.7절, 9.2.1절): 참조의 FrameState를 받아 감김 검출기, StepAggregator, 모델 창,
     * RuleEngine, AlertPolicy를 돈다. completedBaseline이 있고 아직 기준선이 없으면 stateFromSignals의 대입만 한다
     * (baseline = completedBaseline, featureStartMs = completedBaseline.tCompleteMs). 참조가 기준선을 처음 가진 프레임에서
     * 그 기준선을 준다. 기준선 추정기는 돌지 않으므로 baselineDiagnostics는 null로 남는다.
     */
    fun updateFromState(state: FrameState, completedBaseline: Baseline?, speedKmh: Double?): PipelineOutput? {
        val accepted = accept(state.tMs) ?: return null
        if (completedBaseline != null && baselineValue == null) {
            baselineValue = completedBaseline
            featureStartMsValue = completedBaseline.tCompleteMs
        }
        return advance(accepted, state, speedKmh)
    }

    /** pipeline.py:152-165 (SPEC 7.2의 1, 2단계). 버릴 프레임이면 dropped를 세고 null. */
    private fun accept(tMs: Long): Accepted? {
        if (hasFrame && tMs <= prevTMs) {
            droppedFrames += 1
            return null
        }

        var longGap = false
        if (!hasFrame) {
            hasFrame = true
            originMs = tMs
        } else if (tMs - prevTMs > resetGapMs) {
            longGap = true
        }
        prevTMs = tMs
        framesAccepted += 1
        val sessionMs = tMs - originMs
        return Accepted(tMs = tMs, longGap = longGap, sessionMs = sessionMs)
    }

    /** pipeline.py:169-177 (SPEC 7.2의 4단계). R15: var 속성은 지역 변수로 복사해 null 검사하고, 대입 뒤에는 다시 읽는다. */
    private fun stateFromSignals(sig: FrameSignals): FrameState {
        val est = estimator
        if (baselineValue == null && est != null) {
            val completed = est.update(sig)
            if (completed != null) {
                baselineValue = completed
                featureStartMsValue = completed.tCompleteMs
            }
        }
        val b = baselineValue
        return if (b == null) {
            // R10, port_map B28.4(pipeline.py:175-177): 이름 붙인 인자로 부른다.
            frameState(s = sig, b = globalBaseline, spec = spec, provisional = true)
        } else {
            frameState(s = sig, b = b, spec = spec, provisional = false)
        }
    }

    /** pipeline.py:179-217 (SPEC 7.2의 5~9단계). */
    private fun advance(a: Accepted, state: FrameState, speedKmh: Double?): PipelineOutput {
        var run: ClosureRun? = null
        var steps: List<StepFeatures> = Collections.emptyList()
        if (baselineValue != null && a.sessionMs >= featureStartMsValue) {
            var agg = aggregator
            if (agg == null) {
                // R10: Long 둘(startMs, originMs)이 나란히 있으므로 이름 붙인 인자로 부른다.
                agg = StepAggregator(spec = spec, startMs = featureStartMsValue, originMs = originMs)
                aggregator = agg
            }
            if (a.longGap) {
                detector.reset()
            }
            run = detector.update(a.tMs, state.closure, state.closureValid)
            steps = agg.update(state, run, detector.ongoingMs())
        }

        // 스텝마다: 모델 창에 넣고, 창이 찼으면 그 시점의 창으로 모델을 부르고, 그 확률로 졸음 판정을 한다.
        val events = ArrayList<DriverStateEvent>()
        val stepProbs = ArrayList<ClassProbs?>(steps.size)
        var probs: ClassProbs? = null
        val nSteps = steps.size
        stepsEmitted += nSteps.toLong()
        for (i in 0 until nSteps) {
            pushStep(steps[i])
            probs = null
            if (model != null && window.size == windowSteps) {
                probs = callModel()
            }
            stepProbs.add(probs)
            events.addAll(engine.onStep(steps[i], probs))
        }
        // 여기서 probs는 마지막 스텝의 확률이다(스텝이 없으면 null).
        events.addAll(engine.update(state))
        val alerts = policy.decide(events, speedKmh, a.tMs)

        val runs: List<ClosureRun> = if (run != null) Collections.singletonList(run) else Collections.emptyList()
        return PipelineOutput(
            state = state,
            runs = runs,
            steps = steps,
            stepProbs = Collections.unmodifiableList(stepProbs),
            probs = probs,
            events = Collections.unmodifiableList(events),
            alerts = Collections.unmodifiableList(alerts),
        )
    }

    /**
     * 시험 전용. tests/test_pipeline.py:1191 `pipe._window[-1] = (math.nan,) * N_FEATURES`를 옮기려고 둔다(정상 입력에서는
     * 창에 유한하지 않은 값이 생기지 않는다). 모델 창의 가장 새 행을 row로 바꾼다. 창이 비었으면 IllegalStateException.
     */
    internal fun replaceNewestWindowRowForTest(row: List<Double>) {
        check(window.isNotEmpty()) { "model window is empty" }
        window[window.size - 1] = row
    }

    /**
     * 시험 전용. 차등 비교 자료의 `ongoing_feature_ms` 열(FORMATS.md 1.3절)은 참조 내보내기가 프레임마다 읽은
     * `p._detector.ongoing_ms()`다(kotlin/difftools/hk.py run_session). 그 값을 같은 자리에서 읽으려고 둔다. 상태를 바꾸지 않는다.
     */
    internal fun featureDetectorOngoingMsForTest(): Long = detector.ongoingMs()

    /** pipeline.py:219-230 `_push_step`. 모델 창을 갱신한다. 조회 구간이 덜 찬 스텝이나 번호가 건너뛴 스텝을 만나면 창을 비운다. */
    private fun pushStep(step: StepFeatures) {
        if (!step.fullLookback) {
            window.clear()
        } else if (windowHasK && step.k.toLong() != windowLastK.toLong() + 1L) {
            // Python `step.k != self._window_last_k + 1`: Python int는 넘치지 않으므로 Long으로 더한다.
            window.clear()
        }
        if (step.fullLookback) {
            window.addLast(step.values)
            if (window.size > windowSteps) {
                window.removeFirst()
            }
        }
        windowHasK = true
        windowLastK = step.k
    }

    /**
     * pipeline.py:232-244 `_call_model`. 창에 유한하지 않은 값이 있으면 추론을 건너뛰고 센다(null). 모델에는 창의 사본을
     * 준다(pipeline.py:240 `list(self._window)`). 모델이 3개가 아닌 값을 돌려주면 IllegalArgumentException(Python은
     * ValueError, 메시지는 같다). 그 경우 modelCalls는 늘지 않는다.
     */
    private fun callModel(): ClassProbs? {
        for (values in window) {
            for (value in values) {
                if (!isFinite(value)) {
                    nonfiniteWindows += 1
                    return null
                }
            }
        }
        val m = model ?: return null
        val out = m.predict(Collections.unmodifiableList(window.toList()))
        if (out.size != 3) {
            throw IllegalArgumentException("model must return 3 probabilities (alert, low_vigilant, drowsy)")
        }
        modelCalls += 1
        return ClassProbs(alert = out[0], lowVigilant = out[1], drowsy = out[2])
    }
}

/** pipeline.py:247-249 `_is_finite`. NaN과 ±무한대에서만 false다(Kotlin의 Double.isFinite()와 같다). */
internal fun isFinite(x: Double): Boolean = x - x == 0.0
