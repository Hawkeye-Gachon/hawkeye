package hawkeye.drowsy.core

import hawkeye.drowsy.core.internal.TimeCells
import hawkeye.drowsy.core.spec.SpecException
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorFile
import hawkeye.drowsy.core.testkit.VectorModel
import hawkeye.drowsy.core.testkit.VectorRunner
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * DriverPipeline(SPEC 7.2의 1~9단계)의 벡터 실행기 경로. 실행기는 시험 고정물 VectorRunner(vectors.py `_run_pipeline`)다.
 *
 * - E5의 파이프라인 경로: runs 묶음 15개.
 * - E6: steps 묶음 15개.
 * - E4의 파이프라인 경로: baseline 묶음 9개.
 * - E7의 벡터 쪽: rules 묶음 52개(config_overrides, 대체 모델 포함).
 * 판정은 SPEC 16장과 엄격 층(계획 9.0절)이고 counters 다섯은 정확히 같아야 한다. ONNX 모델을 쓰는 두 벡터(model_v0_inputs,
 * e2e_250s)는 :onnx 시험(AllVectorsTest)이 102개 전체와 함께 돌린다.
 * 그리고 C-errors의 DriverPipeline 생성자 다섯 경우를 생성자로 본다(모델이 3개가 아닌 값을 돌려주는 셋은 ModelCallErrorsTest).
 */
class PipelineVectorsTest {

    private val ctx: VectorRunner.Context by lazy { VectorRunner.Context.shipped() }

    @Test
    fun runsBundlePipelinePath() {
        val r = runBundle("runs", 15)
        println("PipelineVectorsTest.runsBundlePipelinePath: ${r.passLine}")
        println("PipelineVectorsTest.runsBundlePipelinePath: runs ${r.times.summary()}")
        println("PipelineVectorsTest.runsBundlePipelinePath: strict-layer notes (above 1e-9, not failures): ${r.notes.size} ${r.notes.take(20)}")
        assertEquals(emptyList(), r.failed)
    }

    @Test
    fun stepsBundlePipelinePath() {
        val r = runBundle("steps", 15)
        println("PipelineVectorsTest.stepsBundlePipelinePath: ${r.passLine}")
        println("PipelineVectorsTest.stepsBundlePipelinePath: steps compared ${r.stepRows}, steps with all 17 values bit-identical ${r.stepRowsBitIdentical}")
        println("PipelineVectorsTest.stepsBundlePipelinePath: largest |kotlin - python| per feature: ${r.featureSummary()}")
        println("PipelineVectorsTest.stepsBundlePipelinePath: runs ${r.times.summary()}")
        println("PipelineVectorsTest.stepsBundlePipelinePath: strict-layer notes (above 1e-9, not failures): ${r.notes.size} ${r.notes.take(20)}")
        assertEquals(emptyList(), r.failed)
    }

    @Test
    fun baselineBundlePipelinePath() {
        val r = runBundle("baseline", 9)
        println("PipelineVectorsTest.baselineBundlePipelinePath: ${r.passLine}")
        for (b in r.baselines) println("PipelineVectorsTest.baselineBundlePipelinePath: $b")
        println("PipelineVectorsTest.baselineBundlePipelinePath: strict-layer notes (above 1e-9, not failures): ${r.notes.size} ${r.notes.take(20)}")
        assertEquals(emptyList(), r.failed)
        assertEquals(6, r.baselines.size)
    }

    /** E7의 벡터 쪽: rules 묶음 52개. 비교한 이벤트·경고 수와 시각 칸, confidence 칸을 센다. */
    @Test
    fun rulesBundlePipelinePath() {
        val r = runBundle("rules", 52)
        println("PipelineVectorsTest.rulesBundlePipelinePath: ${r.passLine}")
        println("PipelineVectorsTest.rulesBundlePipelinePath: events compared ${r.events.rows}, alerts compared ${r.alerts.rows}")
        println("PipelineVectorsTest.rulesBundlePipelinePath: events ${r.events.summary()}")
        println("PipelineVectorsTest.rulesBundlePipelinePath: alerts ${r.alerts.summary()}")
        println("PipelineVectorsTest.rulesBundlePipelinePath: vectors with config_overrides ${r.withOverrides}, with a stub model ${r.withStub}, steps compared ${r.stepRows}, step_probs rows ${r.stepProbRows}, probs rows ${r.probRows}")
        println("PipelineVectorsTest.rulesBundlePipelinePath: largest share of a tolerance used per section: ${r.toleranceUse}")
        println("PipelineVectorsTest.rulesBundlePipelinePath: strict-layer notes (above 1e-9, not failures): ${r.notes.size} ${r.notes.take(20)}")
        assertEquals(emptyList(), r.failed)
    }

    /** C-errors의 생성자 경우(FORMATS.md 2.7절, pipeline.py:90-93, 105-111): 같은 순서, 같은 메시지로 거부한다. */
    @Test
    fun constructorCasesOfCErrors() {
        val data = TestData.readyJson("components/C-errors.json")
        var n = 0
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            if (case["component"] != "DriverPipeline") continue
            val setup = TestData.obj(case["setup"])
            if (!setup.containsKey("kwargs")) continue // 모델 경우(ModelCallErrorsTest)
            val kwargs = TestData.obj(setup["kwargs"])
            val err = TestData.obj(TestData.obj(case["out"])["error"] ?: throw AssertionError("${case["case"]}: no error expected?"))
            assertEquals("ValueError", err["type"])
            val e = assertFailsWith<IllegalArgumentException>(case["case"] as String) {
                DriverPipeline(
                    spec = SpecLoader.pipelineSpecFromTree(setup["pipeline"]),
                    rules = SpecLoader.rulesFromTree(setup["rules"]),
                    baseline = kwargs["baseline"]?.let { VectorFile.baseline(it, "baseline") },
                    featureStartMs = kwargs["feature_start_ms"] as Long?,
                    model = null,
                )
            }
            assertEquals(err["message"], e.message, case["case"] as String)
            n++
        }
        println("PipelineVectorsTest.constructorCasesOfCErrors: $n DriverPipeline constructor cases refused with the Python message")
        assertEquals(5, n)
    }

    /**
     * config_overrides(spec.py:969-990 `apply_overrides`): 없는 경로의 메시지와 바꾼 값. 기대값은 바꾸지 않은 참조로 한 번
     * 돌려 적은 것이다(Python 3.11.16, drowsy/spec.py, config/rules.yaml; 작업 폴더 s4/apply_overrides_python.py와 .out).
     */
    @Test
    fun configOverridesAsThePythonReference() {
        val tree = TestData.parseFile(File(TestData.export, "rules.json"))
        for ((path, message) in listOf(
            "vats.nope" to "override path not found: 'vats.nope'",
            "vats.reset_enabled.x" to "override path not found: 'vats.reset_enabled.x'",
            "nope" to "override path not found: 'nope'",
            "" to "override path not found: ''",
            "events.sleep.repeat_ms." to "override path not found: 'events.sleep.repeat_ms.'",
        )) {
            val e = assertFailsWith<SpecException>(path) { VectorRunner.applyOverrides(tree, mapOf(path to 1L)) }
            assertEquals(message, e.message, path)
        }
        val r1 = VectorRunner.applyOverrides(tree, mapOf("vats.reset_enabled" to true))
        assertEquals(true, r1.vats.resetEnabled)
        val r2 = VectorRunner.applyOverrides(tree, mapOf("drowsiness.source" to "either", "drowsiness.model.refractory_ms" to 600000L))
        assertEquals(DrowsySource.EITHER, r2.drowsiness.source)
        assertEquals(600000L, r2.drowsiness.model.refractoryMs)
        // 바꾼 결과도 같은 검증을 거친다(spec.py:986-987): Python의 메시지 그대로.
        val e = assertFailsWith<SpecException> { VectorRunner.applyOverrides(tree, mapOf("drowsiness.source" to "nope")) }
        assertEquals("drowsiness.source must be one of ('model', 'perclos', 'either')", e.message)
        // 원본 나무는 바뀌지 않는다(사본에서 바꾼다).
        assertEquals(SpecLoader.rulesFromTree(tree), SpecLoader.parseRules(File(TestData.export, "rules.json").readBytes()))
        println("PipelineVectorsTest.configOverridesAsThePythonReference: 5 missing paths refused with the Python message, 3 overrides applied, 1 invalid value refused with the Python message")
    }

    private class Sections {
        var rows = 0
        var timeCells = 0
        var timeExact = 0
        var confidenceCells = 0
        var confidenceExact = 0
        var confidenceModel = 0

        fun add(expected: Any?, actual: Any?) {
            val e = expected as? List<*> ?: return
            val a = actual as? List<*> ?: return
            rows += e.size
            for (i in 0 until minOf(e.size, a.size)) {
                val em = e[i] as? Map<*, *> ?: continue
                val am = a[i] as? Map<*, *> ?: continue
                for (key in listOf("t_start_ms", "duration_ms")) {
                    timeCells++
                    if (em[key] == am[key]) timeExact++
                }
                if (em["source"] == "model") {
                    confidenceModel++
                } else {
                    confidenceCells++
                    val ec = TestData.num(em["confidence"])
                    val ac = am["confidence"] as? Double
                    if (ac != null && ac == ec) confidenceExact++
                }
            }
        }

        fun summary(): String =
            "time cells (t_start_ms, duration_ms) exactly equal $timeExact/$timeCells; confidence (source != model) exactly equal " +
                "$confidenceExact/$confidenceCells; confidence with source model (1e-4) $confidenceModel"
    }

    private class BundleResult(val bundle: String, val total: Int) {
        val failed = ArrayList<String>()
        val notes = ArrayList<String>()
        val baselines = ArrayList<String>()
        val times = TimeCells()
        val events = Sections()
        val alerts = Sections()
        val featureMax = DoubleArray(N_FEATURES)
        val toleranceUse = java.util.TreeMap<String, Double>()
        var stepRows = 0
        var stepRowsBitIdentical = 0
        var stepProbRows = 0
        var probRows = 0
        var frames = 0
        var accepted = 0L
        var withOverrides = 0
        var withStub = 0
        val passLine: String
            get() = "${total - failed.size}/$total $bundle vectors pass through DriverPipeline.update (SPEC 16 and strict layer, " +
                "every expected section), frames fed $frames, accepted $accepted"

        fun featureSummary(): String = featureMax.withIndex().joinToString(", ") { (i, d) -> "#$i ${FEATURE_NAMES[i]} $d" }
    }

    private fun runBundle(bundle: String, count: Int): BundleResult {
        val vectors = TestData.vectorFiles().map { it to VectorFile.read(it.readBytes()) }.filter { it.second.bundle == bundle }
        assertEquals(count, vectors.size, "$bundle vectors")
        val r = BundleResult(bundle, vectors.size)
        for ((file, v) in vectors) {
            assertTrue(v.model !is VectorModel.Onnx, "${v.name}: ONNX vectors run in :onnx")
            if (v.configOverrides.isNotEmpty()) r.withOverrides++
            if (v.model is VectorModel.Stub) r.withStub++
            val o = VectorRunner.verify(file.nameWithoutExtension, v, ctx, null)
            if (!o.passed) {
                r.failed.add("${v.name}: stage0 ${o.stageZero} problems ${o.problems} ${o.result?.problems} ${o.result?.strict?.failures}")
                continue
            }
            val actual = o.actual!!
            val result = o.result!!
            r.frames += v.frames.size
            r.accepted += (TestData.obj(actual["counters"])["frames_accepted"] as Long)
            r.notes.addAll(result.strict.notes.map { "${v.name}: $it" })
            for ((s, x) in result.toleranceUse.maxRatio) if (x > (r.toleranceUse[s] ?: -1.0)) r.toleranceUse[s] = x
            if (v.expected.containsKey("runs")) r.times.add(v.expected["runs"], actual["runs"])
            if (v.expected.containsKey("steps")) countSteps(r, v.expected["steps"], actual["steps"])
            if (v.expected.containsKey("events")) r.events.add(v.expected["events"], actual["events"])
            if (v.expected.containsKey("alerts")) r.alerts.add(v.expected["alerts"], actual["alerts"])
            if (v.expected.containsKey("step_probs")) r.stepProbRows += (v.expected["step_probs"] as List<*>).size
            if (v.expected.containsKey("probs")) r.probRows += (v.expected["probs"] as List<*>).size
            if (v.expected.containsKey("baseline")) {
                val eb = v.expected["baseline"] as Map<*, *>?
                val ab = actual["baseline"] as Map<*, *>?
                if (eb != null) {
                    r.baselines.add(
                        "${v.name}: t_complete_ms ${ab?.get("t_complete_ms")} status ${ab?.get("status")} first used at t_ms ${ab?.get("t_ms")} " +
                            "(python ${eb["t_complete_ms"]} ${eb["status"]} ${eb["t_ms"]})",
                    )
                }
            }
        }
        return r
    }

    private fun countSteps(r: BundleResult, expected: Any?, actual: Any?) {
        val e = expected as List<*>
        val a = actual as List<*>
        for (i in 0 until minOf(e.size, a.size)) {
            val ev = VectorFile.list((e[i] as Map<*, *>)["values"], "values").map { VectorFile.num(it, "v") }
            @Suppress("UNCHECKED_CAST")
            val av = (a[i] as Map<*, *>)["values"] as List<Double>
            if (ev.size != av.size) continue
            r.stepRows++
            var bits = true
            for (f in ev.indices) {
                val d = Math.abs(av[f] - ev[f])
                if (d > r.featureMax[f] || d.isNaN()) r.featureMax[f] = if (d.isNaN()) Double.POSITIVE_INFINITY else d
                if (av[f].toRawBits() != ev[f].toRawBits()) bits = false
            }
            if (bits) r.stepRowsBitIdentical++
        }
    }
}
