package hawkeye.drowsy.core

import hawkeye.drowsy.core.internal.rotationFromEuler
import hawkeye.drowsy.core.spec.PipelineSpec
import hawkeye.drowsy.core.spec.RulesConfig
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * 모델 창의 순서와 밀기, 유한하지 않은 창의 추론 건너뛰기(SPEC 16.1 끝 목록, 계획 9.3절의 "유한하지 않은 창의 추론
 * 건너뛰기" 행, E7). 벡터는 첫 창만 보고 그 뒤 확률은 1e-4로 비교하므로 창을 잘못 밀어도(차이 약 6e-8) 통과한다.
 *
 * tests/test_pipeline.py의 세 시험을 옮겼다. 기대값은 그 Python 시험에 적힌 값이다(그 시험들은 바꾸지 않은 참조에서
 * 통과한다: 작업 폴더 f_s4/pytest_ported_tests.out).
 * - test_timeline_first_model_call_at_239s (271-285행), test_model_window_is_120_by_17_oldest_first (288-300행):
 *   `timeline` 픽스처(245초, 스스로 보정, 100 ms 간격, 첫 t_ms 1, varied_blinks).
 * - test_non_finite_window_skips_inference (1185-1198행).
 * Python 시험은 config/feature_spec.json과 config/rules.yaml을 쓴다. 여기서는 export/의 두 파일을 쓴다. 두 쌍이 같은
 * 값이라는 것(pipeline_hash 650495e5..., pipeline 블록 ==, RulesConfig ==)은 같은 Python으로 확인했다(f_s4 보고).
 * 합성 프레임 `raw`, `colmajor`, `synthetic_baseline`, `StubModel`, `run`, `varied_blinks`는 그 파일의 도우미를 그대로 옮겼다.
 */
class DriverPipelineWindowTest {

    private val spec: PipelineSpec =
        SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes()).pipeline
    private val rules: RulesConfig = SpecLoader.parseRules(File(TestData.export, "rules.json").readBytes())

    /** test_pipeline.py:119-128 StubModel: 받은 창을 기록하고 정해진 확률을 돌려준다. */
    private class StubModel(val probs: List<Double> = listOf(0.2, 0.3, 0.5)) : ModelFn {
        val windows = ArrayList<List<List<Double>>>()
        override fun predict(window: List<List<Double>>): List<Double> {
            windows.add(window.map { it.toList() })
            return probs
        }
    }

    /** test_pipeline.py:56-64 colmajor: R = Ry·Rx·Rz를 열 우선 4x4로. M[i][j] = m[4·j + i]. */
    private fun colmajor(yaw: Double): List<Double> {
        val r = rotationFromEuler(yaw, 0.0, 0.0)
        val m = DoubleArray(16)
        for (i in 0 until 3) for (j in 0 until 3) m[4 * j + i] = r[3 * i + j]
        m[15] = 1.0
        return m.toList()
    }

    /** test_pipeline.py:67-100 raw (closed, face만; level과 yaw는 이 시험들이 쓰지 않는다). */
    private fun raw(tMs: Long, closed: Boolean): RawFrame {
        val lm = LinkedHashMap<Int, Point>()
        for (index in spec.landmarks.requiredIndices()) lm[index] = Point(0.5, 0.5)
        val opening = if (closed) 2.0 else 12.0
        val blink = if (closed) 0.7 else 0.1
        for ((ear, cx) in listOf(spec.landmarks.earR to 250.0, spec.landmarks.earL to 390.0)) {
            val cy = 200.0
            val points = listOf(
                cx - 20.0 to cy,
                cx - 7.0 to cy - opening / 2.0,
                cx + 7.0 to cy - opening / 2.0,
                cx + 20.0 to cy,
                cx + 7.0 to cy + opening / 2.0,
                cx - 7.0 to cy + opening / 2.0,
            )
            for ((index, p) in ear.zip(points)) lm[index] = Point(p.first / W.toDouble(), p.second / H.toDouble())
        }
        val bs = LinkedHashMap<String, Double>()
        for (name in spec.blendshapes.requiredNames()) bs[name] = 0.0
        bs[spec.blendshapes.eyeBlinkL] = blink
        bs[spec.blendshapes.eyeBlinkR] = blink
        return RawFrame(tMs = tMs, face = true, w = W, h = H, lm = lm, bs = bs, m = colmajor(0.0))
    }

    /** test_pipeline.py:103-116 synthetic_baseline. */
    private fun syntheticBaseline(tCompleteMs: Long = 0L) = Baseline(
        eL = 0.3, eR = 0.3, bL = 0.1, bR = 0.1, r0 = IDENTITY_R0, jaw0 = 0.0, gazeH0 = 0.0, gazeV0 = 0.0,
        tCompleteMs = tCompleteMs, status = BaselineStatus.OK,
    )

    /** test_pipeline.py:131-160 run: 세션 시간 목록대로 프레임을 넣고 (세션 시간, 출력) 목록을 돌려준다(속도 null). */
    private fun run(pipe: DriverPipeline, sessionTimes: LongProgression, originMs: Long = 1L, closed: (Long) -> Boolean = { false }):
        List<Pair<Long, PipelineOutput>> {
        val outputs = ArrayList<Pair<Long, PipelineOutput>>()
        for (s in sessionTimes) {
            val out = assertNotNull(pipe.update(raw(originMs + s, closed(s)), null), "frame at session time $s dropped")
            outputs.add(s to out)
        }
        return outputs
    }

    private fun allSteps(outputs: List<Pair<Long, PipelineOutput>>): List<StepFeatures> = outputs.flatMap { it.second.steps }

    /** test_pipeline.py:174-176 varied_blinks: 7초마다 감는다. 길이는 100~500 ms로 돌아가며 바뀐다. */
    private fun variedBlinks(s: Long): Boolean = (s / 1000) % 7 == 0L && s % 1000 < 100 * (1 + (s / 7000) % 5)

    /** test_pipeline.py:215-220 timeline 픽스처: 245초 세션(T_b = 60초). 첫 t_ms는 1이다. */
    private fun timeline(): Triple<DriverPipeline, StubModel, List<Pair<Long, PipelineOutput>>> {
        val model = StubModel()
        val pipe = DriverPipeline(spec = spec, rules = rules, baseline = null, featureStartMs = null, model = model)
        val outputs = run(pipe, 0L..245000L step 100L, originMs = 1L, closed = ::variedBlinks)
        return Triple(pipe, model, outputs)
    }

    @Test
    fun timelineFirstModelCallAt239sAndModelWindowIsOldestFirst() {
        val (pipe, model, outputs) = timeline()
        // test_timeline_first_model_call_at_239s
        val withProbs = outputs.filter { it.second.probs != null }
        val (firstS, firstOut) = withProbs[0]
        assertEquals(239000L, firstS)
        assertEquals(239, firstOut.steps.last().k)
        assertEquals(ClassProbs(alert = 0.2, lowVigilant = 0.3, drowsy = 0.5), firstOut.probs)
        assertEquals(listOf(239000L, 240000L, 241000L, 242000L, 243000L, 244000L, 245000L), withProbs.map { it.first })
        assertEquals(7L, pipe.modelCalls)
        assertEquals(185L, pipe.stepsEmitted)
        assertEquals(2451L, pipe.framesAccepted)
        assertEquals(0L, pipe.droppedFrames)
        assertEquals(0L, pipe.nonfiniteWindows)

        // test_model_window_is_120_by_17_oldest_first
        val steps = allSteps(outputs).associateBy { it.k }
        assertEquals(7, model.windows.size)
        val first = model.windows[0]
        assertEquals(120, first.size)
        assertTrue(first.all { it.size == N_FEATURES })
        // features[0]이 가장 오래된 스텝(k = 120), features[119]가 가장 최근 스텝(k = 239)이다.
        assertEquals((120 until 240).map { steps.getValue(it).values }, first)
        // 마지막 호출의 창은 k = 126~245: 창을 밀 때 가장 오래된 행을 뺀다.
        assertEquals((126 until 246).map { steps.getValue(it).values }, model.windows.last())
        // 스텝마다 값이 달라서 순서가 뒤집히면 위 비교가 실패한다.
        assertTrue(first.map { it[F_CLOSURE_MEAN_10] }.toSet().size > 5)
        assertNotEquals(first, first.reversed())
        println(
            "DriverPipelineWindowTest.timelineFirstModelCallAt239sAndModelWindowIsOldestFirst: frames ${outputs.size}, " +
                "model calls ${model.windows.size}, first window k 120..239, last window k 126..245, as the Python tests",
        )
    }

    @Test
    fun nonFiniteWindowSkipsInference() {
        val model = StubModel()
        val pipe = DriverPipeline(spec = spec, rules = rules, baseline = syntheticBaseline(), featureStartMs = 0L, model = model)
        run(pipe, 0L..179000L step 100L)
        assertEquals(1L, pipe.modelCalls)
        // 정상 입력에서는 생기지 않는 값이다. 방어 동작만 보려고 창의 가장 최근 스텝을 NaN으로 바꿔 놓는다.
        pipe.replaceNewestWindowRowForTest(List(N_FEATURES) { Double.NaN })
        val later = run(pipe, 179100L..181000L step 100L)
        // 창에 유한하지 않은 값이 있는 동안은 추론을 건너뛰고 센다. 스텝과 규칙은 그대로 돈다.
        assertEquals(listOf(180, 181), allSteps(later).map { it.k })
        assertTrue(later.all { it.second.probs == null })
        assertEquals(1L, pipe.modelCalls)
        assertEquals(2L, pipe.nonfiniteWindows)
        assertEquals(1, model.windows.size)
        println(
            "DriverPipelineWindowTest.nonFiniteWindowSkipsInference: model calls ${pipe.modelCalls}, " +
                "nonfinite windows ${pipe.nonfiniteWindows}, steps after the NaN row ${allSteps(later).map { it.k }}, as the Python test",
        )
    }

    private companion object {
        const val W = 640
        const val H = 480
    }
}
