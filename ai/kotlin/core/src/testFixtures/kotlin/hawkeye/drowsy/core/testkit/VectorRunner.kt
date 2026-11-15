package hawkeye.drowsy.core.testkit

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.ClosureRun
import hawkeye.drowsy.core.DriverPipeline
import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.ModelFn
import hawkeye.drowsy.core.PipelineOutput
import hawkeye.drowsy.core.StepFeatures
import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.internal.FrameSignals
import hawkeye.drowsy.core.internal.compute
import hawkeye.drowsy.core.spec.FeatureSpec
import hawkeye.drowsy.core.spec.MiniJson
import hawkeye.drowsy.core.spec.PyRepr
import hawkeye.drowsy.core.spec.RulesConfig
import hawkeye.drowsy.core.spec.SpecException
import hawkeye.drowsy.core.spec.SpecLoader
import java.io.File

/**
 * 벡터 실행기 (계획 3.2절 "묶음별 실행기", "config_overrides 적용", 9.0절). drowsy/vectors.py:503-660 `compute_expected`,
 * `_run_pipeline`, `_model_fn`과 729-771 `verify_vector`를 옮겼다. 시험 고정물이다(배포하지 않는다). :core 시험과
 * :onnx 시험이 같이 쓴다. :onnx 시험은 :core의 internal(DriverPipeline, signals.compute)을 볼 수 없으므로 여기를 거친다.
 * 공개 함수의 형은 모두 공개 형이다(VectorFile, ModelFn, JSON 나무).
 *
 * 출력의 모양은 vectors.py `_run_pipeline` 그대로다: 받아들인 프레임마다 closure 한 원소, runs·steps·events·alerts에는
 * 그것을 낸 프레임의 t_ms, step_probs는 스텝마다, probs는 마지막 스텝에 확률이 있는 프레임마다, model_inputs는 모델을 처음
 * 부른 창, 기준선은 처음 가진 프레임의 t_ms와 함께, counters 다섯.
 * ONNX 모델은 :core에 없으므로 부르는 쪽이 [OnnxOpener]로 준다. 없으면 ONNX 벡터는 "could not recompute"로 실패한다.
 */
object VectorRunner {

    /** 벡터의 model.kind = onnx: 파일(export/의 그 이름)에서 모델 함수를 연다. 닫는 것은 부르는 쪽의 일이다. */
    fun interface OnnxOpener {
        fun open(file: File, windowSteps: Int, nFeatures: Int): ModelFn
    }

    /**
     * vectors.py:465-500 `Context`. 시험하는 feature_spec(파이프라인 상수와 pipeline_hash), rules.json 바이트(그 sha256이
     * rules_hash이고 그 나무가 config_overrides의 바탕이다), 모델 폴더.
     */
    class Context(val featureSpec: FeatureSpec, val rulesBytes: ByteArray, val modelDir: File) {
        val rulesSha256: String = sha256Hex(rulesBytes)
        internal val rules: RulesConfig = SpecLoader.parseRules(rulesBytes)

        companion object {
            /** 배포 파일 셋: export/feature_spec_v0.json, export/rules.json, 모델 폴더 export/. */
            fun shipped(): Context = Context(
                featureSpec = SpecLoader.parseFeatureSpec(File(DiffData.export, "feature_spec_v0.json").readBytes()),
                rulesBytes = File(DiffData.export, "rules.json").readBytes(),
                modelDir = DiffData.export,
            )
        }
    }

    /** 벡터 하나의 검증 결과. problems = 0단계, 모델 파일, 다시 계산, SPEC 16장 문제. result는 비교까지 간 경우만. */
    class Outcome(
        val name: String,
        val bundle: String,
        val stageZero: List<String>,
        val problems: List<String>,
        val result: VectorComparator.Result?,
        val actual: Map<String, Any?>?,
    ) {
        val passed: Boolean get() = stageZero.isEmpty() && problems.isEmpty() && result != null && result.passed
    }

    /**
     * vectors.py:729-771 `verify_vector`와 계획 9.0절 실행기의 0단계: 0단계 문제가 있으면 아무것도 비교하지 않는다.
     * 그다음 ONNX 모델 파일의 sha256, 다시 계산, 항목마다 SPEC 16장 판정과 엄격 층([VectorComparator.compareVector]).
     */
    fun verify(fileStem: String, v: VectorFile, ctx: Context, onnx: OnnxOpener?): Outcome {
        val stage0 = v.stageZeroProblems(fileStem, ctx.featureSpec.pipelineHash, ctx.rulesSha256)
        if (stage0.isNotEmpty()) return Outcome(v.name, v.bundle, stage0, emptyList(), null, null)
        val problems = ArrayList<String>()
        val model = v.model
        if (model is VectorModel.Onnx) {
            val path = File(ctx.modelDir, model.file)
            if (!path.isFile) return Outcome(v.name, v.bundle, stage0, listOf("model file not found: $path"), null, null)
            val digest = sha256Hex(path.readBytes())
            if (digest != model.sha256) problems.add("model sha256 is ${model.sha256}, $path has $digest")
        }
        val actual: Map<String, Any?> = try {
            computeExpected(v, ctx, onnx)
        } catch (e: IllegalArgumentException) {
            // vectors.py:757-758: (VectorError, SpecError, KeyError, ValueError, TypeError)
            problems.add("could not recompute: ${e.javaClass.simpleName}: ${e.message}")
            return Outcome(v.name, v.bundle, stage0, problems, null, null)
        }
        val result = VectorComparator.compareVector(v, actual)
        return Outcome(v.name, v.bundle, stage0, problems, result, actual)
    }

    /** vectors.py:556-576 `compute_expected`. 그 묶음에서 계산할 수 있는 항목을 모두 돌려준다. */
    fun computeExpected(v: VectorFile, ctx: Context, onnx: OnnxOpener?): Map<String, Any?> {
        val spec = ctx.featureSpec.pipeline
        return when (v.bundle) {
            "signals" -> linkedMapOf("signals" to v.frames.map { signalsJson(compute(it, spec)) })
            "model" -> {
                val fn = modelFn(v.model, ctx, onnx) ?: throw IllegalArgumentException("model bundle without a model")
                val inputs = v.modelInputs ?: throw IllegalArgumentException("model bundle without model_inputs")
                linkedMapOf(
                    "probs" to inputs.mapIndexed { i, w -> linkedMapOf<String, Any?>("input" to i.toLong(), "probs" to ArrayList<Any?>(fn.predict(w))) },
                )
            }
            "baseline", "runs", "steps", "rules", "e2e" -> runPipeline(v, ctx, onnx)
            else -> throw IllegalArgumentException("unknown bundle: ${PyRepr.str(v.bundle)}")
        }
    }

    /**
     * vectors.py:525-553 `_model_fn`. stub: i번째 호출(0부터)에 probs[min(i, len − 1)]를 돌려준다(목록이 끝나면 마지막 값을
     * 되풀이한다). onnx: [OnnxOpener]가 연 모델.
     */
    private fun modelFn(model: VectorModel?, ctx: Context, onnx: OnnxOpener?): ModelFn? = when (model) {
        null -> null
        is VectorModel.Stub -> {
            val scripted = model.probs
            if (scripted.isEmpty()) throw IllegalArgumentException("stub model needs at least one probability triple")
            var calls = 0
            ModelFn { _ ->
                val index = minOf(calls, scripted.size - 1)
                calls += 1
                scripted[index]
            }
        }
        is VectorModel.Onnx -> {
            val opener = onnx ?: throw IllegalArgumentException("no ONNX opener given for model file ${model.file} (run this vector in :onnx)")
            opener.open(File(ctx.modelDir, model.file), ctx.featureSpec.windowSteps, ctx.featureSpec.features.size)
        }
    }

    /** vectors.py:579-660 `_run_pipeline`. 프레임을 DriverPipeline에 차례로 넣고 나온 것을 항목별로 모은다([Collector]). */
    private fun runPipeline(v: VectorFile, ctx: Context, onnx: OnnxOpener?): Map<String, Any?> {
        var rules = ctx.rules
        if (v.configOverrides.isNotEmpty()) rules = applyOverrides(MiniJson.parse(ctx.rulesBytes), v.configOverrides)
        val injected = v.baseline

        val model = modelFn(v.model, ctx, onnx)
        val collector = Collector(baselineInjected = injected != null)
        val pipe = DriverPipeline(
            spec = ctx.featureSpec.pipeline, rules = rules, baseline = injected, featureStartMs = null,
            model = model?.let { collector.recording(it) },
        )
        for (frame in v.frames) {
            val out = pipe.update(frame, v.speedAt(frame.tMs)) ?: continue
            collector.add(frame.tMs, out, pipe.baseline)
        }
        return collector.result(
            framesAccepted = pipe.framesAccepted,
            droppedFrames = pipe.droppedFrames,
            stepsEmitted = pipe.stepsEmitted,
            modelCalls = pipe.modelCalls,
            nonfiniteWindows = pipe.nonfiniteWindows,
        )
    }

    /**
     * vectors.py:579-660 `_run_pipeline`의 모으는 쪽. 받아들인 프레임마다 [add]를 부르고 끝에 [result]로 항목별 JSON 나무를
     * 얻는다. 파이프라인 밖(예: `DrowsinessMonitor`가 낸 DriverState)에서도 같은 모양을 만들도록 공개로 둔다.
     * 모델 입력(model_inputs)은 [recording]으로 감싼 모델 함수가 처음 받은 창이다.
     */
    class Collector(private val baselineInjected: Boolean) {
        private val windows = ArrayList<List<List<Double>>>()
        private val closure = ArrayList<Any?>()
        private val runs = ArrayList<Any?>()
        private val steps = ArrayList<Any?>()
        private val modelInputs = ArrayList<Any?>()
        private val stepProbs = ArrayList<Any?>()
        private val probs = ArrayList<Any?>()
        private val events = ArrayList<Any?>()
        private val alerts = ArrayList<Any?>()
        private var baseline: Map<String, Any?>? = null

        /** 모델 함수를 감싸 처음 부른 창만 적어 둔다(vectors.py의 기록 모델과 같다). */
        fun recording(model: ModelFn): ModelFn = ModelFn { window ->
            if (windows.isEmpty()) windows.add(window.map { ArrayList(it) })
            model.predict(window)
        }

        /** 받아들인 프레임 하나. baselineAfter = 그 update 직후 파이프라인의 baseline 속성. */
        fun add(tMs: Long, out: PipelineOutput, baselineAfter: Baseline?) {
            if (!baselineInjected && baseline == null && baselineAfter != null) {
                val m = LinkedHashMap<String, Any?>()
                m["t_ms"] = tMs
                m.putAll(baselineJson(baselineAfter))
                baseline = m
            }
            closure.add(stateJson(out.state))
            for (run in out.runs) runs.add(runJson(tMs, run))
            for (step in out.steps) steps.add(stepJson(tMs, step))
            // 모델은 스텝마다 불린다. 스텝별 확률은 step_probs에, 프레임의 마지막 스텝의 확률은 probs에 적는다.
            require(out.steps.size == out.stepProbs.size) { "steps and step_probs differ in length" } // zip(strict=True)
            for (j in out.steps.indices) {
                val step = out.steps[j]
                val p = out.stepProbs[j]
                stepProbs.add(
                    linkedMapOf<String, Any?>(
                        "t_ms" to tMs,
                        "k" to step.k.toLong(),
                        "probs" to p?.let { arrayListOf<Any?>(it.alert, it.lowVigilant, it.drowsy) },
                    ),
                )
                if (p != null && modelInputs.isEmpty()) { // 처음 부른 창은 확률을 처음 받은 스텝의 것이다
                    modelInputs.add(linkedMapOf<String, Any?>("t_ms" to tMs, "k" to step.k.toLong(), "features" to windows[0]))
                }
            }
            val outProbs = out.probs
            if (outProbs != null) {
                probs.add(
                    linkedMapOf<String, Any?>(
                        "t_ms" to tMs,
                        "k" to out.steps[out.steps.size - 1].k.toLong(),
                        "probs" to arrayListOf<Any?>(outProbs.alert, outProbs.lowVigilant, outProbs.drowsy),
                    ),
                )
            }
            for (e in out.events) events.add(eventJson(tMs, e))
            for (e in out.alerts) alerts.add(eventJson(tMs, e))
        }

        /** 항목별 JSON 나무(vectors.py `_run_pipeline`의 돌려주는 값과 같은 키). */
        fun result(
            framesAccepted: Long,
            droppedFrames: Long,
            stepsEmitted: Long,
            modelCalls: Long,
            nonfiniteWindows: Long,
        ): Map<String, Any?> = linkedMapOf(
            "closure" to closure,
            "baseline" to baseline,
            "runs" to runs,
            "steps" to steps,
            "model_inputs" to modelInputs,
            "step_probs" to stepProbs,
            "probs" to probs,
            "events" to events,
            "alerts" to alerts,
            "counters" to linkedMapOf<String, Any?>(
                "frames_accepted" to framesAccepted,
                "dropped_frames" to droppedFrames,
                "steps_emitted" to stepsEmitted,
                "model_calls" to modelCalls,
                "nonfinite_windows" to nonfiniteWindows,
            ),
        )
    }

    // -----------------------------------------------------------------------------------------------------------------
    // config_overrides (drowsy/spec.py:969-990 `apply_overrides`)
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * spec.py:969-990 `apply_overrides`의 RulesConfig 쪽: 규칙 설정의 평범한 나무(`to_plain(cfg)`; 배포 rules.json은
     * `rules_to_json(cfg)`이라 그 나무와 같다)의 사본에서 점으로 이은 경로의 값을 바꾸고, 바꾼 결과를 같은 검증(rules_from_dict)에
     * 넣는다. 경로가 없거나 중간이 객체가 아니면 SpecException "override path not found: '<경로>'"(Python repr). 덮어쓰기는
     * 적힌 순서대로다. 벡터의 config_overrides는 규칙 설정에만 적용한다(vectors.py:45-46의 가정).
     */
    fun applyOverrides(rulesTree: Any?, overrides: Map<String, Any?>): RulesConfig {
        val plain = deepCopy(rulesTree)
        for ((dotted, value) in overrides) {
            val parts = dotted.split(".")
            var node: Any? = plain
            for ((depth, part) in parts.withIndex()) {
                if (node !is MutableMap<*, *> || !node.containsKey(part)) {
                    throw SpecException("override path not found: ${PyRepr.str(dotted)}")
                }
                if (depth == parts.size - 1) {
                    @Suppress("UNCHECKED_CAST")
                    (node as MutableMap<String, Any?>)[part] = deepCopy(value)
                } else {
                    node = node[part]
                }
            }
        }
        return SpecLoader.rulesFromTree(plain)
    }

    /** JSON 나무의 깊은 사본(객체는 LinkedHashMap으로 키 순서를 지킨다). */
    private fun deepCopy(v: Any?): Any? = when (v) {
        is Map<*, *> -> {
            val m = LinkedHashMap<String, Any?>()
            for ((k, x) in v) m[k as String] = deepCopy(x)
            m
        }
        is List<*> -> ArrayList<Any?>(v.map { deepCopy(it) })
        else -> v
    }

    // -----------------------------------------------------------------------------------------------------------------
    // Kotlin 값 → 벡터의 JSON 나무 (vectors.py:412-458; dataclasses.asdict: 정수는 Long, 실수는 Double, 열거형은 wire)
    // -----------------------------------------------------------------------------------------------------------------

    private fun signalsJson(s: FrameSignals): Map<String, Any?> = linkedMapOf(
        "t_ms" to s.tMs,
        "face" to s.face,
        "ear_l" to s.earL,
        "ear_r" to s.earR,
        "eye_w_l" to s.eyeWL,
        "eye_w_r" to s.eyeWR,
        "blink_l" to s.blinkL,
        "blink_r" to s.blinkR,
        "eye_look" to ArrayList<Any?>(s.eyeLook),
        "jaw_open" to s.jawOpen,
        "yaw" to s.yaw,
        "pitch" to s.pitch,
        "roll" to s.roll,
        "iris_h_l" to s.irisHL,
        "iris_h_r" to s.irisHR,
        "iris_v_l" to s.irisVL,
        "iris_v_r" to s.irisVR,
    )

    private fun stateJson(st: FrameState): Map<String, Any?> = linkedMapOf(
        "t_ms" to st.tMs,
        "face" to st.face,
        "closure" to st.closure,
        "closure_valid" to st.closureValid,
        "provisional" to st.provisional,
        "yaw_rel" to st.yawRel,
        "pitch_rel" to st.pitchRel,
        "pose_valid" to st.poseValid,
        "gaze_h_rel" to st.gazeHRel,
        "gaze_v_rel" to st.gazeVRel,
        "jaw_rel" to st.jawRel,
    )

    private fun baselineJson(b: Baseline): Map<String, Any?> = linkedMapOf(
        "e_l" to b.eL,
        "e_r" to b.eR,
        "b_l" to b.bL,
        "b_r" to b.bR,
        "r0" to ArrayList<Any?>(b.r0),
        "jaw0" to b.jaw0,
        "gaze_h0" to b.gazeH0,
        "gaze_v0" to b.gazeV0,
        "t_complete_ms" to b.tCompleteMs,
        "status" to b.status.wire,
    )

    private fun runJson(tMs: Long, r: ClosureRun): Map<String, Any?> = linkedMapOf(
        "t_ms" to tMs,
        "start_ms" to r.startMs,
        "end_ms" to r.endMs,
        "duration_ms" to r.durationMs,
        "amplitude" to r.amplitude,
        "peak_closing_velocity" to r.peakClosingVelocity,
        "peak_opening_velocity" to r.peakOpeningVelocity,
        "velocity_valid" to r.velocityValid,
        "truncated" to r.truncated,
    )

    private fun stepJson(tMs: Long, s: StepFeatures): Map<String, Any?> = linkedMapOf(
        "t_ms" to tMs,
        "k" to s.k.toLong(),
        "t_end_ms" to s.tEndMs,
        "full_lookback" to s.fullLookback,
        "values" to ArrayList<Any?>(s.values),
    )

    /** vectors.py:457-458 `_event_to_json`: {"t_ms", DriverStateEvent의 필드 여덟}. */
    fun eventJson(tMs: Long?, e: DriverStateEvent): Map<String, Any?> {
        val m = LinkedHashMap<String, Any?>()
        if (tMs != null) m["t_ms"] = tMs
        m["type"] = e.type.wire
        m["severity"] = e.severity.toLong()
        m["t_start_ms"] = e.tStartMs
        m["duration_ms"] = e.durationMs
        m["confidence"] = e.confidence
        m["source"] = e.source.wire
        m["phase"] = e.phase.wire
        m["provisional"] = e.provisional
        return m
    }
}
