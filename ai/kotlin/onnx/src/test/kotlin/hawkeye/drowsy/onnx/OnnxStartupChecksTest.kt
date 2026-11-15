package hawkeye.drowsy.onnx

import hawkeye.drowsy.core.check.ModelIo
import hawkeye.drowsy.core.check.StartupCheck
import hawkeye.drowsy.core.check.StartupChecks
import hawkeye.drowsy.core.check.StartupInput
import hawkeye.drowsy.core.check.VerifiedValues
import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.testkit.DiffData
import org.junit.jupiter.api.Assumptions
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * E10 (계획 9.1절, 9.4절): 시작 검사를 진짜 파일과 변조표의 모든 경우로 본다. 단계 1은 모델 파일이 기본 파일과 같은 경우만
 * 메타데이터 맵(manifest의 base_metadata)으로 돌렸다. 여기서는 ONNX Runtime이 **쓰는 모델 파일 바이트에서** 메타데이터와
 * 입출력 모양을 읽는다(OnnxDrowsinessModel.inspect, 계획 5.8절의 시작 순서). 그래서 SC1, SC2, SC3, SD3도 돈다.
 *
 * 기대 코드는 계획 9.4절의 표다(manifest의 expected_codes, expected_codes_source "plan 9.4"; 시작 검사는 새 코드라 Python에
 * 같은 것이 없다). 파일과 사실(sha256, onnx 1.23.2와 onnxruntime 1.31.1로 읽은 메타데이터와 모양)은 Python 내보내기가
 * 만들었다(FORMATS.md 2.3절). 변조 파일은 모두 자료에 있어 시험 안에서 만든 변조는 없다.
 *
 * 여기서 돌리지 않는 경우: SD4(공개 길. openReplay는 리드 판정 7로 빠졌고, 그 자리의 공개 입구 DrowsinessMonitor.open은
 * hawkeye.drowsy.DrowsinessMonitorOpenTest가 본다), SD5(`DriveSession.create`: 같은 DrowsinessMonitorOpenTest). 셈에 이름과
 * 까닭을 적는다.
 */
class OnnxStartupChecksTest {

    private fun strings(m: Any?): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        for ((k, v) in DiffData.obj(m)) out[k] = v as String
        return out
    }

    /** 모델 바이트 → (메타데이터, 입출력). ONNX Runtime이 읽지 못하면 (null, null) = MODEL_UNREADABLE (계획 5.8절). */
    private fun readModel(bytes: ByteArray): Pair<Map<String, String>?, ModelIo?> = try {
        val info = OnnxDrowsinessModel.inspect(bytes)
        info.metadata to info.io
    } catch (e: IllegalArgumentException) {
        null to null
    }

    @Test
    fun tamperTableAllCasesWithOnnxRuntime() {
        Assumptions.assumeTrue(DiffData.readySmall() != null, "no READY_small.txt: differential data (small) not exported")
        val manifest = DiffData.readyJson("tamper/manifest.json")
        val base = DiffData.obj(manifest["base"])
        val verifiedJson = DiffData.obj(manifest["verified"])
        val baseFiles = HashMap<String, File>()
        for (role in listOf("task", "onnx", "feature_spec", "rules")) {
            val b = DiffData.obj(base[role])
            val f = DiffData.repoFile(b["path"] as String)
            assertEquals(b["sha256"], sha256Hex(f.readBytes()), "base $role")
            baseFiles[role] = f
        }
        val baseBits = DiffData.constantBits(baseFiles.getValue("feature_spec").readBytes(), baseFiles.getValue("rules").readBytes())

        var run = 0
        val notRun = ArrayList<String>()
        val bad = ArrayList<String>()
        val accepted = ArrayList<String>()
        val perCase = ArrayList<String>()
        for (c in DiffData.list(manifest["cases"])) {
            val case = DiffData.obj(c)
            val id = case["id"] as String
            if (id == "SD4") {
                notRun.add("SD4 (public path: DrowsinessMonitorOpenTest)")
                continue
            }
            if (id == "SD5") {
                notRun.add("SD5 (DriveSession.create: DrowsinessMonitorOpenTest)")
                continue
            }
            val files = DiffData.obj(case["files"])
            val missing = DiffData.list(case["missing"]).map { it as String }.toSet()
            val expectedCodes = DiffData.list(case["expected_codes"]).map { it as String }.toSet()
            fun bytesOf(role: String): ByteArray? {
                if (role in missing) return null
                val rel = files[role] as String?
                return if (rel != null) DiffData.readyBytes(rel) else baseFiles.getValue(role).readBytes()
            }
            val onnx = bytesOf("onnx")
            val specBytes = bytesOf("feature_spec")
            val rulesBytes = bytesOf("rules")
            val task = bytesOf("task")
            val modelFileName = (files["onnx"] as String?)?.substringAfterLast('/') ?: "drowsiness_v0.onnx"
            val specFileName = (files["feature_spec"] as String?)?.substringAfterLast('/') ?: "feature_spec_v0.json"

            // 메타데이터: 경우가 맵을 주면 그것(SB*: 시작 검사 함수가 맵을 받는다), 아니면 쓰는 모델 파일에서 ONNX Runtime으로 읽는다.
            val (fileMetadata, fileIo) = if (onnx == null) null to null else readModel(onnx)
            val metadata = when {
                onnx == null -> null
                case["metadata"] != null -> strings(case["metadata"])
                else -> fileMetadata
            }
            // 사실과 대조: Python이 같은 파일에서 읽은 메타데이터와 모양(FORMATS.md 2.3절 facts)
            val facts = DiffData.obj(case["facts"])
            if (facts.containsKey("metadata")) assertEquals(strings(facts["metadata"]), fileMetadata, "$id: metadata read from the file")
            if (facts.containsKey("inputs")) {
                val input = DiffData.list(DiffData.list(facts["inputs"])[0])
                val output = DiffData.list(DiffData.list(facts["outputs"])[0])
                assertEquals(
                    ModelIo(input[0] as String, DiffData.list(input[1]).map { it as Long }, output[0] as String, DiffData.list(output[1]).map { it as Long }),
                    fileIo,
                    "$id: io read from the file",
                )
            }
            if (facts["onnxruntime_loads"] == false) assertNull(fileIo, "$id: onnxruntime 1.31.1 could not load it, 1.28.0 must not either")

            val override = case["verified_override"]?.let { DiffData.obj(it) }
            val bits = if (override != null && override["constant_bits"] == "compute from the files in use") {
                DiffData.constantBits(specBytes!!, rulesBytes!!)
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
            DiffData.obj(facts["sha256_in_use"]).let { inUse ->
                for ((role, bytes) in listOf("task" to task, "onnx" to onnx, "feature_spec" to specBytes, "rules" to rulesBytes)) {
                    if (inUse.containsKey(role)) assertEquals(inUse[role], bytes?.let { sha256Hex(it) }, "$id: sha256 of $role")
                }
            }
            fun check(pin: String?) = StartupChecks.run(
                StartupInput(
                    modelFileName = modelFileName,
                    modelFileSha256 = onnx?.let { sha256Hex(it) },
                    modelSha256Pin = pin,
                    modelMetadata = metadata,
                    modelIo = if (onnx == null) null else fileIo,
                    specFileName = specFileName,
                    specBytes = specBytes,
                    rulesBytes = rulesBytes,
                    taskSha256 = task?.let { sha256Hex(it) },
                    verified = verified,
                ),
            )
            val result = check(null)
            run++
            val codes = result.problems.map { it.check.name }.toSet()
            if (codes != expectedCodes) bad.add("$id: expected $expectedCodes, got $codes ${result.problems.map { it.message }}")
            if (result.passed != expectedCodes.isEmpty()) bad.add("$id: passed=${result.passed}")
            for (p in result.problems) {
                if (p.expected != null) assertTrue(p.message.contains(p.expected!!), "$id: ${p.message}")
                if (p.actual != null) assertTrue(p.message.contains(p.actual!!), "$id: ${p.message}")
            }
            perCase.add("$id=${codes.sorted()}")
            // SD1: 빠진 파일이 문제의 subject다(manifest의 note).
            if (id.startsWith("SD1_")) {
                val name = mapOf(
                    "task" to "face_landmarker.task", "onnx" to modelFileName, "feature_spec" to specFileName, "rules" to "rules.json",
                ).getValue(missing.single())
                assertEquals(listOf(name), result.problems.map { it.subject }, "$id subject")
            }
            // 받아들여진 변조: 시작 검사가 문제를 하나도 내지 않은 경우(SB11의 표시 값, K12 (가)의 SC2)
            if (result.passed) accepted.add(id)
            // SC2 (K12 (가)): 가중치 한 바이트는 고정값이 없으면 받아들여지고, 기본 파일의 sha256을 고정값으로 주면 MODEL_FILE_VS_PIN 하나다.
            if (id == "SC2") {
                val pinned = check(DiffData.obj(base["onnx"])["sha256"] as String)
                assertEquals(listOf(StartupCheck.MODEL_FILE_VS_PIN), pinned.problems.map { it.check }, "SC2 with the base pin")
            }
            // SD3: 모양이 계약과 다른 모델은 보고만 하고, 어댑터는 그 파일로 열리지 않는다(demo.py:285-288).
            if (id == "SD3") {
                assertFailsWith<IllegalArgumentException> { OnnxDrowsinessModel.open(onnx!!, 120, 17) }
                assertTrue("H5_WINDOW_STEPS" !in codes)
            }
        }
        println("OnnxStartupChecksTest: ${DiffData.list(manifest["cases"]).size} cases in the table; $run run with metadata and io read by ONNX Runtime ${OnnxDrowsinessModel.runtimeVersion()}, codes exactly as plan 9.4 in ${run - bad.size}")
        println("OnnxStartupChecksTest: accepted tamperings (no problem by design): $accepted")
        println("OnnxStartupChecksTest: not run: $notRun")
        println("OnnxStartupChecksTest: codes per case: $perCase")
        assertEquals(emptyList(), bad)
        assertEquals(DiffData.list(manifest["cases"]).size, run + notRun.size)
    }

    /** 진짜 파일 넷(번들 그대로): 메타데이터와 입출력을 ONNX Runtime으로 모델 파일에서 읽어도 문제 0이다. */
    @Test
    fun realFilesPassWithMetadataFromTheModelFile() {
        Assumptions.assumeTrue(DiffData.readySmall() != null, "no READY_small.txt: differential data (small) not exported")
        val manifest = DiffData.readyJson("tamper/manifest.json")
        val onnx = File(DiffData.export, "drowsiness_v0.onnx").readBytes()
        val spec = File(DiffData.export, "feature_spec_v0.json").readBytes()
        val rules = File(DiffData.export, "rules.json").readBytes()
        val taskSha = DiffData.obj(DiffData.obj(manifest["base"])["task"])["sha256"] as String
        val info = OnnxDrowsinessModel.inspect(onnx)
        val result = StartupChecks.run(
            StartupInput(
                modelFileName = "drowsiness_v0.onnx",
                modelFileSha256 = sha256Hex(onnx),
                modelSha256Pin = sha256Hex(onnx),
                modelMetadata = info.metadata,
                modelIo = info.io,
                specFileName = "feature_spec_v0.json",
                specBytes = spec,
                rulesBytes = rules,
                taskSha256 = taskSha,
                verified = VerifiedValues(
                    pipelineHash = DiffData.obj(manifest["verified"])["pipeline_hash"] as String,
                    rulesHash = sha256Hex(rules),
                    vectorSet = "",
                    vectorCount = 0,
                    constantBits = DiffData.constantBits(spec, rules),
                ),
            ),
        )
        assertEquals(emptyList(), result.problems)
        assertTrue(result.passed)
    }
}
