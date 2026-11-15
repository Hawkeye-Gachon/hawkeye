package hawkeye.drowsy.core.vectors

import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorFile
import hawkeye.drowsy.core.testkit.VectorModel
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * E2의 벡터 읽기 (계획 9.1절 E2). 벡터 102개를 시험 고정물의 벡터 모형으로 읽고, 실행기의 0단계(계획 9.0절)를 모두 통과하는지,
 * 프레임 수와 가장 큰 t_ms가 그대로인지 본다. 통과 기준의 수(102개, 프레임 20,327개, 가장 큰 t_ms 1731465423289)는
 * 계획 9.1절 E2의 것이다.
 */
class VectorReadTest {

    @Test
    fun allVectorsReadAndPassStageZero() {
        val files = TestData.vectorFiles()
        val spec = SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes())
        val rulesSha = sha256Hex(File(TestData.export, "rules.json").readBytes())

        var frames = 0L
        var faceFrames = 0L
        var maxT = Long.MIN_VALUE
        var maxTVector = ""
        var over32Bit = 0L
        var modelWindows = 0
        val perBundle = java.util.TreeMap<String, Int>()
        val stageZero = ArrayList<String>()
        for (f in files) {
            val v = VectorFile.read(f.readBytes())
            perBundle.merge(v.bundle, 1, Int::plus)
            for (p in v.stageZeroProblems(f.name.removeSuffix(".json"), spec.pipelineHash, rulesSha)) stageZero.add("${f.name}: $p")
            for (fr in v.frames) {
                frames++
                if (fr.face) {
                    faceFrames++
                    assertEquals(16, fr.m.size, "${f.name} t_ms ${fr.tMs}: matrix")
                } else {
                    assertTrue(fr.lm.isEmpty() && fr.bs.isEmpty() && fr.m.isEmpty(), "${f.name} t_ms ${fr.tMs}: no-face frame")
                }
                if (fr.tMs > Int.MAX_VALUE) over32Bit++
                if (fr.tMs > maxT) {
                    maxT = fr.tMs
                    maxTVector = v.name
                }
            }
            v.modelInputs?.let { windows ->
                for (w in windows) {
                    assertEquals(spec.windowSteps, w.size, "${f.name}: model input rows")
                    for (row in w) assertEquals(spec.features.size, row.size, "${f.name}: model input columns")
                }
                modelWindows += windows.size
            }
            val m = v.model
            if (m is VectorModel.Onnx) {
                assertEquals(sha256Hex(File(TestData.export, m.file).readBytes()), m.sha256, "${f.name}: model sha256")
            }
        }
        println(
            "VectorReadTest: ${files.size} vectors $perBundle, $frames frames ($faceFrames with a face), " +
                "largest t_ms $maxT in $maxTVector, $over32Bit frames with t_ms above 2^31-1, $modelWindows model windows, " +
                "stage-0 problems ${stageZero.size}",
        )
        assertEquals(emptyList(), stageZero)
        assertEquals(102, files.size)
        assertEquals(20327L, frames)
        assertEquals(1731465423289L, maxT)
    }

    /** 0단계가 실제로 거부하는지: 해시, 이름, 느슨한 허용 오차 값, 빠진 허용 오차 키(벡터 하나를 고친 사본으로). */
    @Test
    fun stageZeroRefusesChangedVectors() {
        val f = File(TestData.vectors, "runs_basic_blink.json").takeIf { it.isFile } ?: TestData.vectorFiles().first { it.name.startsWith("runs_") }
        val tree = TestData.obj(TestData.parseFile(f))
        val v = VectorFile.fromTree(tree)
        val stem = f.name.removeSuffix(".json")
        assertEquals(emptyList(), v.stageZeroProblems(stem, v.pipelineHash, v.rulesHash))
        assertEquals(1, v.stageZeroProblems(stem, v.pipelineHash + "0", v.rulesHash).size)
        assertEquals(1, v.stageZeroProblems(stem, v.pipelineHash, v.rulesHash.uppercase()).size)
        assertEquals(1, v.stageZeroProblems(stem + "_x", v.pipelineHash, v.rulesHash).size)

        val loose = LinkedHashMap(tree)
        val tol = LinkedHashMap(TestData.obj(tree["tolerances"]))
        val runs = LinkedHashMap(TestData.obj(tol["runs"]))
        runs["time_abs_ms"] = 2L
        tol["runs"] = runs
        loose["tolerances"] = tol
        val p1 = VectorFile.fromTree(loose).stageZeroProblems(stem, v.pipelineHash, v.rulesHash)
        assertEquals(1, p1.size, p1.toString())
        runs.remove("rel")
        val p2 = VectorFile.fromTree(loose).stageZeroProblems(stem, v.pipelineHash, v.rulesHash)
        assertEquals(1, p2.size, p2.toString())
    }

    /** vectors.py:609-612의 속도 규칙: t_ms 이하인 마지막 항목, 없으면 null. */
    @Test
    fun speedRule() {
        val f = TestData.vectorFiles().first { file ->
            TestData.list(TestData.obj(TestData.parseFile(file))["speed"]).size >= 2
        }
        val v = VectorFile.read(f.readBytes())
        // 바로 셈한 기대값: t 이하인 항목 가운데 마지막 것.
        fun direct(t: Long): Double? = v.speed.lastOrNull { it.first <= t }?.second
        val probes = ArrayList<Long>()
        for ((t, _) in v.speed) probes.addAll(listOf(t - 1, t, t + 1))
        probes.add(Long.MAX_VALUE)
        for (t in probes) assertEquals(direct(t), v.speedAt(t), "t_ms $t")
        assertEquals(null, v.speedAt(v.speed[0].first - 1))
    }
}
