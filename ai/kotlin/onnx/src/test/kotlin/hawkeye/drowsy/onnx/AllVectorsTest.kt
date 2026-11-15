package hawkeye.drowsy.onnx

import hawkeye.drowsy.core.testkit.DiffData
import hawkeye.drowsy.core.testkit.VectorFile
import hawkeye.drowsy.core.testkit.VectorRunner
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 벡터 102개 전체를 한 시험 작업(:onnx:test)에서 돌린다(단계 4 끝의 조건). 실행기는 시험 고정물 VectorRunner(vectors.py
 * `compute_expected`, `verify_vector`)이고, ONNX 모델을 쓰는 두 벡터(model_v0_inputs, e2e_250s)에는 데스크톱 ONNX Runtime
 * 1.28.0의 OnnxDrowsinessModel을 준다. 판정은 실행기의 0단계, SPEC 16장 판정과 엄격 층(계획 9.0절)이다.
 *
 * E12 (계획 9.1절): e2e_250s를 DriverPipeline과 ONNX로. 여덟 항목(alerts, baseline, counters, events, model_inputs, probs,
 * step_probs, steps) 모두 엄격 층으로 통과하고, probs 12개(k 239~250), step_probs 190개(178개 null), model_calls = 12.
 * 항목별 가장 큰 차이와 계수를 적는다.
 */
class AllVectorsTest {

    private val ctx = VectorRunner.Context.shipped()

    /** 연 모델을 모아 두었다가 벡터 하나가 끝나면 닫는다. */
    private class Opener : VectorRunner.OnnxOpener {
        val opened = ArrayList<OnnxDrowsinessModel>()
        override fun open(file: java.io.File, windowSteps: Int, nFeatures: Int): OnnxDrowsinessModel =
            OnnxDrowsinessModel.open(file.readBytes(), windowSteps, nFeatures).also { opened.add(it) }

        fun closeAll() {
            for (m in opened) m.close()
            opened.clear()
        }
    }

    @Test
    fun allVectorsPassInOneTask() {
        val files = DiffData.vectorFiles()
        val byBundle = java.util.TreeMap<String, IntArray>() // bundle -> [passed, total]
        val failed = ArrayList<String>()
        var frames = 0
        var onnxVectors = 0
        var notes = 0
        for (f in files) {
            val v = VectorFile.read(f.readBytes())
            val opener = Opener()
            val o = try {
                VectorRunner.verify(f.nameWithoutExtension, v, ctx, opener)
            } finally {
                if (opener.opened.isNotEmpty()) onnxVectors++
                opener.closeAll()
            }
            val cell = byBundle.getOrPut(v.bundle) { IntArray(2) }
            cell[1]++
            frames += v.frames.size
            if (o.passed) {
                cell[0]++
                notes += o.result!!.strict.notes.size
            } else {
                failed.add("${v.name}: stage0 ${o.stageZero} problems ${o.problems} ${o.result?.problems} ${o.result?.strict?.failures}")
            }
        }
        val passed = byBundle.values.sumOf { it[0] }
        val summary = "AllVectorsTest.allVectorsPassInOneTask: $passed/${files.size} vectors pass (stage 0, SPEC 16 and strict layer) " +
            "${byBundle.entries.joinToString(", ") { "${it.key} ${it.value[0]}/${it.value[1]}" }}; frames $frames; vectors run with " +
            "ONNX Runtime ${OnnxDrowsinessModel.runtimeVersion()} $onnxVectors; strict-layer notes (above 1e-9, not failures) $notes"
        println(summary)
        for (x in failed) println("AllVectorsTest.allVectorsPassInOneTask: FAILED $x")
        assertEquals(102, files.size)
        assertEquals(emptyList(), failed)
    }

    /** E12: e2e_250s를 DriverPipeline과 ONNX로. */
    @Test
    fun e2eThroughDriverPipelineAndOnnx() {
        val file = java.io.File(DiffData.vectors, "e2e_250s.json")
        val v = VectorFile.read(file.readBytes())
        val opener = Opener()
        val o = try {
            VectorRunner.verify(file.nameWithoutExtension, v, ctx, opener)
        } finally {
            opener.closeAll()
        }
        assertEquals(emptyList(), o.stageZero)
        assertEquals(emptyList(), o.problems)
        val result = o.result!!
        val actual = o.actual!!
        val sections = v.expected.keys.sorted()
        println("AllVectorsTest.e2eThroughDriverPipelineAndOnnx: items ${sections.size} $sections, frames ${v.frames.size}")
        for (s in sections) {
            val d = MaxDiff()
            d.walk(v.expected[s], actual[s])
            println("AllVectorsTest.e2eThroughDriverPipelineAndOnnx: $s: numbers compared ${d.numbers}, largest |kotlin - python| ${d.max}, at ${d.at ?: "-"}")
        }
        val counters = DiffData.obj(actual["counters"])
        val stepProbs = DiffData.list(actual["step_probs"])
        val probs = DiffData.list(actual["probs"])
        val probKs = probs.map { DiffData.obj(it)["k"] as Long }
        val nullStepProbs = stepProbs.count { DiffData.obj(it)["probs"] == null }
        println(
            "AllVectorsTest.e2eThroughDriverPipelineAndOnnx: counters $counters; probs ${probs.size} (k ${probKs.firstOrNull()}..${probKs.lastOrNull()}); " +
                "step_probs ${stepProbs.size} ($nullStepProbs null); events ${DiffData.list(actual["events"]).size}, alerts ${DiffData.list(actual["alerts"]).size}, " +
                "steps ${DiffData.list(actual["steps"]).size}, model_inputs ${DiffData.list(actual["model_inputs"]).size}",
        )
        println(
            "AllVectorsTest.e2eThroughDriverPipelineAndOnnx: SPEC 16 problems ${result.problems.size} ${result.problems}, strict-layer failures " +
                "${result.strict.failures.size} ${result.strict.failures}, notes ${result.strict.notes.size} ${result.strict.notes.take(10)}, " +
                "largest share of a tolerance used ${result.toleranceUse.maxRatio}",
        )
        assertTrue(result.passed, "${result.problems} ${result.strict.failures}")
        assertEquals(8, sections.size)
        assertEquals(12, probs.size)
        assertEquals((239L..250L).toList(), probKs)
        assertEquals(190, stepProbs.size)
        assertEquals(178, nullStepProbs)
        assertEquals(12L, counters["model_calls"])
    }

    /** 수 잎의 가장 큰 |a − e|. 정수와 실수 모두 센다(구조가 다른 곳은 비교기가 이미 문제로 적는다). */
    private class MaxDiff {
        var max = 0.0
        var at: String? = null
        var numbers = 0

        fun walk(e: Any?, a: Any?, path: String = "") {
            when (e) {
                is Map<*, *> -> if (a is Map<*, *>) for (k in e.keys) walk(e[k], a[k], "$path.$k")
                is List<*> -> if (a is List<*> && a.size == e.size) for (i in e.indices) walk(e[i], a[i], "$path[$i]")
                is Long, is Double -> {
                    val ev = (e as Number).toDouble()
                    val av = (a as? Number)?.toDouble() ?: Double.NaN
                    numbers++
                    val d = Math.abs(av - ev)
                    if (d.isNaN() || d > max) {
                        max = if (d.isNaN()) Double.POSITIVE_INFINITY else d
                        at = path
                    }
                }
                else -> Unit
            }
        }
    }
}
