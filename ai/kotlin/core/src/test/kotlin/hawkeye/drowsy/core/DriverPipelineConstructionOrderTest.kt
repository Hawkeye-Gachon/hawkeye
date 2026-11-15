package hawkeye.drowsy.core

import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * DriverPipeline 생성 검사 셋이 다른 객체를 만들기 전에 오는지 본다(pipeline.py:90-93, 105-111: 원본은 검사 둘 뒤에
 * global_baseline을, 검사 셋 뒤에 ClosureRunDetector를 만든다).
 *
 * 검사를 거치지 않은 spec 객체(run_detector.velocity_lookback_samples = -1)를 쓴다. 그런 spec에서 Kotlin 검출기는
 * DoubleArray(-1)로 NegativeArraySizeException을 내므로, 검출기를 검사보다 먼저 만들면 Python의 ValueError 대신 그 예외가
 * 나온다. 기대값은 바꾸지 않은 Python 참조를 같은 입력으로 돌려 적은 것이다(Python 3.11.16, drowsy/pipeline.py
 * fcf25763..., 작업 폴더 f_s4/construction_order_python.py와 .out). 그 실행에서 검사가 모두 통과하는 다섯째 경우는
 * Python이 파이프라인을 만들지만 Kotlin 검출기는 음수 길이를 받지 않는다. 그것은 이 시험의 대상이 아니다(검사를 거친
 * spec에서는 velocity_lookback_samples >= 1, spec.py:656).
 */
class DriverPipelineConstructionOrderTest {

    @Test
    fun constructionChecksComeBeforeTheDetectorIsBuilt() {
        val spec0 = SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes()).pipeline
        val rules = SpecLoader.parseRules(File(TestData.export, "rules.json").readBytes())
        val bad = spec0.copy(runDetector = spec0.runDetector.copy(velocityLookbackSamples = -1))
        val otherClock = rules.copy(clock = rules.clock.copy(resetGapMs = rules.clock.resetGapMs + 1))
        val b = Baseline(
            eL = 0.3, eR = 0.3, bL = 0.1, bR = 0.1, r0 = listOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            jaw0 = 0.05, gazeH0 = 0.0, gazeV0 = 0.0, tCompleteMs = 60000L, status = BaselineStatus.OK,
        )
        // (경우, rules, baseline, featureStartMs, Python의 ValueError 메시지)
        val cases = listOf(
            Triple("clock_mismatch", Triple(otherClock, null as Baseline?, null as Long?), "rules.clock must equal spec.clock"),
            Triple("feature_start_without_baseline", Triple(rules, null, 0L), "feature_start_ms requires an injected baseline"),
            Triple("feature_start_negative", Triple(rules, b, -1000L), "feature_start_ms must be a non-negative multiple of clock.step_ms"),
            Triple("feature_start_not_multiple", Triple(rules, b, 60500L), "feature_start_ms must be a non-negative multiple of clock.step_ms"),
        )
        for ((name, args, message) in cases) {
            val (r, baseline, featureStart) = args
            val e = assertFailsWith<IllegalArgumentException>(name) {
                DriverPipeline(spec = bad, rules = r, baseline = baseline, featureStartMs = featureStart, model = null)
            }
            assertEquals(message, e.message, name)
        }
        println("DriverPipelineConstructionOrderTest: ${cases.size} cases with velocity_lookback_samples -1 refused by the construction checks with the Python message")
    }
}
