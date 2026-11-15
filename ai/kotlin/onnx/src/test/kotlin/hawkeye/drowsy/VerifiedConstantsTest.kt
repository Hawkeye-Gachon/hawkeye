package hawkeye.drowsy

import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testkit.DiffData
import hawkeye.drowsy.core.testkit.VectorFile
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * VerifiedConstants(리드 판정 9)를 tests/vectors와 export/의 번들 파일에서 다시 계산한다. 상수가 낡았으면(벡터나 번들 파일이
 * 바뀌었으면) 실패한다. 같은 시험 작업의 AllVectorsTest가 벡터 102개를 이 번들 파일로 통과시킨다.
 * 그리고 JAR에 든 번들 파일(processResources가 복사한 것)이 export/의 파일과 바이트로 같은지 본다.
 */
class VerifiedConstantsTest {

    private val bundled = listOf("drowsiness_v0.onnx", "feature_spec_v0.json", "rules.json")

    private fun resource(name: String): ByteArray? =
        DrowsinessMonitor::class.java.getResourceAsStream("/hawkeye/drowsy/bundle/$name")?.use { it.readBytes() }

    @Test
    fun constantsEqualValuesRecomputedFromVectorsAndShippedFiles() {
        val files = DiffData.vectorFiles()
        // 벡터 묶음 지문(계획 3.5절; 이음 규칙은 VerifiedConstants.VECTOR_SET의 가정).
        val lines = files.map { "${it.name} ${sha256Hex(it.readBytes())}" }
        val vectorSet = sha256Hex(lines.joinToString("\n").toByteArray(Charsets.UTF_8))
        val pipelineHashes = HashSet<String>()
        val rulesHashes = HashSet<String>()
        for (f in files) {
            val v = VectorFile.read(f.readBytes())
            pipelineHashes.add(v.pipelineHash)
            rulesHashes.add(v.rulesHash)
        }
        val specBytes = File(DiffData.export, "feature_spec_v0.json").readBytes()
        val rulesBytes = File(DiffData.export, "rules.json").readBytes()
        val specHash = SpecLoader.parseFeatureSpec(specBytes).pipelineHash
        val rulesSha = sha256Hex(rulesBytes)
        val bits = DiffData.constantBits(specBytes, rulesBytes)
        println(
            "VerifiedConstantsTest: recomputed vector count ${files.size}, vector set $vectorSet, pipeline_hash of vectors $pipelineHashes " +
                "and of feature_spec_v0.json $specHash, rules_hash of vectors $rulesHashes and sha256(rules.json) $rulesSha, constant bits $bits",
        )
        assertEquals(VerifiedConstants.VECTOR_COUNT, files.size, "vector count")
        assertEquals(VerifiedConstants.VECTOR_SET, vectorSet, "vector set fingerprint")
        assertEquals(setOf(VerifiedConstants.PIPELINE_HASH), pipelineHashes, "pipeline_hash of every vector")
        assertEquals(VerifiedConstants.PIPELINE_HASH, specHash, "pipeline_hash of feature_spec_v0.json")
        assertEquals(setOf(VerifiedConstants.RULES_HASH), rulesHashes, "rules_hash of every vector")
        assertEquals(VerifiedConstants.RULES_HASH, rulesSha, "sha256 of rules.json")
        assertEquals(VerifiedConstants.CONSTANT_BITS, bits, "constant bits of feature_spec_v0.json and rules.json")
        // 번들(클래스 경로)의 두 설정 파일에서도 같은 값이다.
        val bSpec = assertNotNull(resource("feature_spec_v0.json"))
        val bRules = assertNotNull(resource("rules.json"))
        assertEquals(VerifiedConstants.CONSTANT_BITS, DiffData.constantBits(bSpec, bRules), "constant bits of the bundled files")
        assertEquals(VerifiedConstants.RULES_HASH, sha256Hex(bRules), "sha256 of the bundled rules.json")
        assertEquals(VerifiedConstants.PIPELINE_HASH, SpecLoader.parseFeatureSpec(bSpec).pipelineHash, "pipeline_hash of the bundled spec")
        assertEquals(
            listOf(VerifiedConstants.PIPELINE_HASH, VerifiedConstants.RULES_HASH, VerifiedConstants.VECTOR_SET, VerifiedConstants.CONSTANT_BITS),
            listOf(VerifiedConstants.VALUES.pipelineHash, VerifiedConstants.VALUES.rulesHash, VerifiedConstants.VALUES.vectorSet, VerifiedConstants.VALUES.constantBits),
        )
        assertEquals(VerifiedConstants.VECTOR_COUNT, VerifiedConstants.VALUES.vectorCount)
    }

    /** bundle: 번들 자원 셋이 export/의 파일과 sha256으로 같고, 번들 폴더에 다른 파일이 없다. */
    @Test
    fun bundledResourcesEqualExportFiles() {
        for (name in bundled) {
            val res = assertNotNull(resource(name), "resource hawkeye/drowsy/bundle/$name")
            val src = File(DiffData.export, name).readBytes()
            assertEquals(sha256Hex(src), sha256Hex(res), "$name: sha256 of the resource and of export/$name")
            assertEquals(src.size, res.size, "$name: bytes")
            println("VerifiedConstantsTest.bundledResourcesEqualExportFiles: $name ${res.size} bytes sha256 ${sha256Hex(res)} = export/$name")
        }
        // 시험의 클래스 경로에서 번들 폴더는 build/resources/main의 폴더다(file: URL). 그 안의 이름이 정확히 셋이다.
        val dir = DrowsinessMonitor::class.java.getResource("/hawkeye/drowsy/bundle/")
        assertNotNull(dir)
        if (dir.protocol == "file") {
            val names = (File(dir.toURI()).list() ?: emptyArray()).sorted()
            println("VerifiedConstantsTest.bundledResourcesEqualExportFiles: bundle folder ${dir.protocol} entries $names")
            assertEquals(bundled.sorted(), names)
        } else {
            println("VerifiedConstantsTest.bundledResourcesEqualExportFiles: bundle folder is ${dir.protocol}, entry list not read here")
        }
    }
}
