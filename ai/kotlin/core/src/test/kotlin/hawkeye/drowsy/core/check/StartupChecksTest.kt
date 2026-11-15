package hawkeye.drowsy.core.check

import hawkeye.drowsy.core.spec.ConstantBits
import hawkeye.drowsy.core.spec.MiniJson
import hawkeye.drowsy.core.testdata.TestData
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 시작 검사(계획 5.11절)를 변조표 자료로 본다 (계획 9.4절, E10의 단계 1 몫: 메타데이터 맵으로 되는 경우;
 * FORMATS.md 2.3절 tamper/manifest.json). 기대 코드는 계획 9.4절의 표다(Kotlin 시작 검사는 새 코드이고 Python에는 같은 것이
 * 없다, manifest의 expected_codes_source). 파일과 사실(sha256, 메타데이터)은 Python 내보내기가 만들었다.
 *
 * 여기서 돌리지 않는 경우: ONNX 파일 자체를 읽어야 하는 경우(SC1~SC3, SD3)는 :onnx의
 * hawkeye.drowsy.onnx.OnnxStartupChecksTest가, SD4(공개 길. openReplay는 리드 판정 7로 빠졌고 그 자리의 공개 입구
 * DrowsinessMonitor.open)와 SD5(`DriveSession.create`)는 :onnx의 hawkeye.drowsy.DrowsinessMonitorOpenTest가 본다. 모델 파일이 기본 파일과 바이트가 같은 경우는 메타데이터를 manifest의 base_metadata(Python이
 * onnx 1.23.2로 읽은 값)로, 입출력 모양을 ONNX 계약(SPEC 14.1: features [1, 120, 17], probs [1, 3])으로 준다.
 */
class StartupChecksTest {

    private val baseIo = ModelIo("features", listOf(1L, 120L, 17L), "probs", listOf(1L, 3L))

    private fun repoFile(path: String): File =
        if (path.startsWith("export/")) File(TestData.export, path.removePrefix("export/")) else File(TestData.repo, path)

    @Suppress("UNCHECKED_CAST")
    private fun strings(m: Any?): Map<String, String> = (TestData.obj(m) as Map<String, String>)

    @Test
    fun tamperTableMetadataCases() {
        val manifest = TestData.readyJson("tamper/manifest.json")
        val base = TestData.obj(manifest["base"])
        val baseMetadata = strings(manifest["base_metadata"])
        val verifiedJson = TestData.obj(manifest["verified"])
        val baseFiles = HashMap<String, File>()
        for (role in listOf("task", "onnx", "feature_spec", "rules")) {
            val b = TestData.obj(base[role])
            val f = repoFile(b["path"] as String)
            val bytes = f.readBytes()
            assertEquals(b["sha256"], sha256Hex(bytes), "base $role")
            baseFiles[role] = f
        }
        val baseOnnxSha = TestData.obj(base["onnx"])["sha256"] as String
        val baseBits = ConstantBits.fingerprint(
            TestData.obj(MiniJson.parse(baseFiles.getValue("feature_spec").readBytes()))["pipeline"],
            MiniJson.parse(baseFiles.getValue("rules").readBytes()),
        )

        var run = 0
        val notRun = ArrayList<String>()
        val bad = ArrayList<String>()
        for (c in TestData.list(manifest["cases"])) {
            val case = TestData.obj(c)
            val id = case["id"] as String
            val files = TestData.obj(case["files"])
            val missing = TestData.list(case["missing"]).map { it as String }.toSet()
            val expectedCodes = TestData.list(case["expected_codes"]).map { it as String }.toSet()
            // 파일 바이트: 경우의 파일(READY 검사) 또는 기본 파일
            fun bytesOf(role: String): ByteArray? {
                if (role in missing) return null
                val rel = files[role] as String?
                return if (rel != null) TestData.readyBytes(rel) else baseFiles.getValue(role).readBytes()
            }
            val onnx = bytesOf("onnx")
            if (onnx != null && sha256Hex(onnx) != baseOnnxSha) {
                notRun.add("$id (ONNX file changed: needs ONNX Runtime, hawkeye.drowsy.onnx.OnnxStartupChecksTest)")
                continue
            }
            if (id.startsWith("SD4") || id.startsWith("SD5")) {
                notRun.add("$id (public entry or DriveSession.create: hawkeye.drowsy.DrowsinessMonitorOpenTest)")
                continue
            }
            val modelFileName = (files["onnx"] as String?)?.substringAfterLast('/') ?: "drowsiness_v0.onnx"
            val specFileName = (files["feature_spec"] as String?)?.substringAfterLast('/') ?: "feature_spec_v0.json"
            val specBytes = bytesOf("feature_spec")
            val rulesBytes = bytesOf("rules")
            val task = bytesOf("task")
            val metadata = when {
                onnx == null -> null
                case["metadata"] != null -> strings(case["metadata"])
                else -> baseMetadata
            }
            val override = case["verified_override"]?.let { TestData.obj(it) }
            val bits = if (override != null && override["constant_bits"] == "compute from the files in use") {
                ConstantBits.fingerprint(TestData.obj(MiniJson.parse(specBytes!!))["pipeline"], MiniJson.parse(rulesBytes!!))
            } else {
                baseBits
            }
            val verified = VerifiedValues(
                pipelineHash = verifiedJson["pipeline_hash"] as String,
                rulesHash = (override?.get("rules_hash") ?: verifiedJson["rules_hash"]) as String,
                vectorSet = "",
                vectorCount = 0,
                constantBits = bits,
            )
            // 사실과 대조: 쓰는 파일의 sha256 (facts.sha256_in_use)
            val facts = TestData.obj(case["facts"])
            facts["sha256_in_use"]?.let { inUse ->
                val m = TestData.obj(inUse)
                for ((role, bytes) in listOf("task" to task, "onnx" to onnx, "feature_spec" to specBytes, "rules" to rulesBytes)) {
                    if (m.containsKey(role)) assertEquals(m[role], bytes?.let { sha256Hex(it) }, "$id: sha256 of $role")
                }
            }
            val result = StartupChecks.run(
                StartupInput(
                    modelFileName = modelFileName,
                    modelFileSha256 = onnx?.let { sha256Hex(it) },
                    modelSha256Pin = null,
                    modelMetadata = metadata,
                    modelIo = if (onnx == null) null else baseIo,
                    specFileName = specFileName,
                    specBytes = specBytes,
                    rulesBytes = rulesBytes,
                    taskSha256 = task?.let { sha256Hex(it) },
                    verified = verified,
                ),
            )
            run++
            val codes = result.problems.map { it.check.name }.toSet()
            if (codes != expectedCodes) bad.add("$id: expected $expectedCodes, got $codes ${result.problems.map { it.message }}")
            if (result.passed != expectedCodes.isEmpty()) bad.add("$id: passed=${result.passed}")
            // 메시지에 기대값과 실제값이 있다(값이 있는 비교만)
            for (p in result.problems) {
                if (p.expected != null) assertTrue(p.message.contains(p.expected!!), "$id: ${p.message}")
                if (p.actual != null) assertTrue(p.message.contains(p.actual!!), "$id: ${p.message}")
            }
        }
        println("StartupChecksTest: ${run} tamper cases run, codes exactly as plan 9.4 in ${run - bad.size}; not run here: $notRun")
        assertEquals(emptyList(), bad)
        assertEquals(TestData.list(manifest["cases"]).size, run + notRun.size)
    }

    /** 진짜 파일: 문제 0, 읽은 spec과 rules가 결과에 있다. 모델 고정값이 다르면 MODEL_FILE_VS_PIN 하나. */
    @Test
    fun realFilesPassAndThePinIsCompared() {
        val manifest = TestData.readyJson("tamper/manifest.json")
        val metadata = strings(manifest["base_metadata"])
        val spec = File(TestData.export, "feature_spec_v0.json").readBytes()
        val rules = File(TestData.export, "rules.json").readBytes()
        val onnxSha = sha256Hex(File(TestData.export, "drowsiness_v0.onnx").readBytes())
        val taskSha = TestData.obj(TestData.obj(manifest["base"])["task"])["sha256"] as String
        val verified = VerifiedValues(
            pipelineHash = TestData.obj(manifest["verified"])["pipeline_hash"] as String,
            rulesHash = sha256Hex(rules),
            vectorSet = "",
            vectorCount = 0,
            constantBits = ConstantBits.fingerprint(TestData.obj(MiniJson.parse(spec))["pipeline"], MiniJson.parse(rules)),
        )
        fun input(pin: String?, io: ModelIo?) = StartupInput("drowsiness_v0.onnx", onnxSha, pin, metadata, io,
            "feature_spec_v0.json", spec, rules, taskSha, verified)
        val ok = StartupChecks.run(input(null, baseIo))
        assertEquals(emptyList(), ok.problems)
        assertTrue(ok.passed && ok.spec != null && ok.rules != null)
        assertEquals(emptyList(), StartupChecks.run(input(onnxSha, baseIo)).problems)
        val pinned = StartupChecks.run(input("0".repeat(64), baseIo))
        assertEquals(listOf(StartupCheck.MODEL_FILE_VS_PIN), pinned.problems.map { it.check })
        val unreadable = StartupChecks.run(input(null, null))
        assertEquals(listOf(StartupCheck.MODEL_UNREADABLE), unreadable.problems.map { it.check })
        val wrongIo = StartupChecks.run(input(null, ModelIo("x", listOf(1L, 60L, 17L), "probs", listOf(1L, 4L))))
        assertEquals(listOf(StartupCheck.MODEL_IO_CONTRACT, StartupCheck.MODEL_IO_CONTRACT), wrongIo.problems.map { it.check })
    }

    /** 상수 비트 지문: 줄의 모양, 수 잎의 수, 문서 순서. 바이트가 달라도(CRLF) 같은 값이면 같은 지문이다. */
    @Test
    fun constantBitsLines() {
        val specTree = TestData.obj(MiniJson.parse(File(TestData.export, "feature_spec_v0.json").readBytes()))
        val rulesTree = MiniJson.parse(File(TestData.export, "rules.json").readBytes())
        val lines = ConstantBits.lines(specTree["pipeline"], rulesTree)
        assertTrue("pipeline.landmarks.ear_r[0] = 33" in lines, lines.take(3).toString())
        assertTrue("pipeline.clock.step_ms = 1000" in lines)
        assertTrue("rules.forward_zone.yaw_abs_max_deg = 4037000000000000" in lines) // 23.0
        assertTrue(lines.none { it.contains("model_sha256") || it.contains("eye_aux_enabled") }, "strings and booleans are not numbers")
        assertEquals(lines.size, lines.toSet().size, "paths are unique")
        val crlf = File(TestData.export, "rules.json").readText(Charsets.UTF_8).replace("\n", "\r\n").toByteArray(Charsets.UTF_8)
        assertEquals(ConstantBits.fingerprint(specTree["pipeline"], rulesTree), ConstantBits.fingerprint(specTree["pipeline"], MiniJson.parse(crlf)))
        println("StartupChecksTest: constant bits ${lines.size} numeric leaves, fingerprint ${ConstantBits.fingerprint(specTree["pipeline"], rulesTree)}")
    }
}
