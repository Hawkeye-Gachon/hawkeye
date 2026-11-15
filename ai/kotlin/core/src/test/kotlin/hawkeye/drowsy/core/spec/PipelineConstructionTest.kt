package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorFile
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * DriverPipeline 생성 검사 셋 (pipeline.py:90-93, 105-111; 계획 5.7절, 5.10절 R20)을 C-errors 자료의 생성자 경우로 본다
 * (FORMATS.md 2.7절: `component == "DriverPipeline"`이고 setup = {pipeline, rules, kwargs}). Python의 ValueError는
 * IllegalArgumentException이고 메시지는 같아야 한다. 파이프라인 자체는 단계 3에 생기므로 여기서는 검사 함수만 부른다.
 */
class PipelineConstructionTest {

    @Test
    fun constructorCasesOfCErrors() {
        val data = TestData.readyJson("components/C-errors.json")
        var n = 0
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            if (case["component"] != "DriverPipeline") continue
            val setup = TestData.obj(case["setup"])
            if (!setup.containsKey("kwargs")) continue // 모델 경우(단계 3 이후)
            val spec = SpecLoader.pipelineSpecFromTree(setup["pipeline"])
            val rules = SpecLoader.rulesFromTree(setup["rules"])
            val kwargs = TestData.obj(setup["kwargs"])
            val baseline = kwargs["baseline"]?.let { VectorFile.baseline(it, "baseline") }
            val featureStart = kwargs["feature_start_ms"] as Long?
            val out = TestData.obj(case["out"])
            val err = TestData.obj(out["error"] ?: throw AssertionError("${case["case"]}: no error expected?"))
            assertEquals("ValueError", err["type"])
            val e = assertFailsWith<IllegalArgumentException>(case["case"] as String) {
                SpecValidation.checkPipelineConstruction(spec, rules, baseline, featureStart)
            }
            assertEquals(err["message"], e.message, case["case"] as String)
            n++
        }
        println("PipelineConstructionTest: $n DriverPipeline constructor cases refused with the Python message")
        assertEquals(5, n)
    }

    /** 통과하는 경우의 시작 시각: 기준선 없음 0, 기준선의 t_complete_ms, feature_start_ms가 있으면 그것. */
    @Test
    fun featureStartOfAcceptedArguments() {
        val spec = SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes()).pipeline
        val rules = SpecLoader.parseRules(File(TestData.export, "rules.json").readBytes())
        assertEquals(0L, SpecValidation.checkPipelineConstruction(spec, rules, null, null))
        val vector = TestData.vectorFiles().map { VectorFile.read(it.readBytes()) }.first { it.baseline != null }
        val b = vector.baseline!!
        assertEquals(b.tCompleteMs, SpecValidation.checkPipelineConstruction(spec, rules, b, null))
        assertEquals(3000L, SpecValidation.checkPipelineConstruction(spec, rules, b, 3000L))
    }
}
