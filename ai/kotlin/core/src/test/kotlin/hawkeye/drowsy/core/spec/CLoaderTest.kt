package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.testdata.TestData
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * C-loader 자료로 SpecLoader와 SpecValidation을 본다 (계획 5.10절, 9.2.6절; FORMATS.md 2.7절; 리드 판정 2).
 * 문서마다 Python(`spec.pipeline_spec_from_dict`, `rules_from_dict`, `feature_spec_from_dict`)이 받았는지와 거부 메시지가 있다.
 * Kotlin이 같은 글(`text`)을 읽어 같은 판정과 같은 메시지를 내야 한다.
 *
 * 허용하는 차이(그 밖의 차이는 모두 실패):
 * - 리드 판정 2: `flags`가 붙은 문서에서만, "Python 받음, Kotlin 거부" 또는 "둘 다 거부, 다른 메시지". 표지마다 센다.
 *   그리고 Kotlin의 거부 메시지가 그 표지의 원인이어야 한다(다른 까닭으로 거부하면 실패):
 *   - `int_outside_int64`: MiniJson의 "JSON: integer N does not fit in 64 bits at offset K"이고 N은 Long 밖이다.
 *   - `int_outside_int32`만: "<where>: integer N is outside the 32-bit range of this field"이고 `<where>`는 그 문서의
 *     바꾼 자리(`target.path`), N은 Long 안 Int 밖이다.
 *   - `accepted_changed_string`: 차이를 허용하지 않는다(그 문자열은 Python처럼 검사하지 않는다, 리드 판정 2).
 *   가정: Int 범위 밖(R22)과 Long 범위 밖(계획 5.10절) 정수의 처리는 소유자 확인을 기다린다.
 * - 계획 5.10절: Kotlin은 pipeline_hash를 다시 계산하지 않는다(SPEC 15장). 그래서 Python이 "pipeline_hash mismatch"로만
 *   거부하는 문서를 Kotlin은 받는다. 그런 문서는 따로 세고, 그 문서의 pipeline_hash 문자열이 검증 값과 달라 시작 검사의
 *   H3_SPEC_VS_VERIFIED가 잡는다는 것을 확인한다.
 * "Python 거부, Kotlin 받음"은 위 둘째 경우 말고는 어디서든 실패다. 표지가 붙은 문서도 건너뛰지 않고 비교한다.
 */
class CLoaderTest {

    private fun judge(target: String, text: String): String? = try {
        val bytes = text.toByteArray(Charsets.UTF_8)
        when (target) {
            "pipeline" -> SpecLoader.parsePipelineSpec(bytes)
            "rules" -> SpecLoader.parseRules(bytes)
            "feature_spec" -> SpecLoader.parseFeatureSpec(bytes)
            else -> throw AssertionError("unknown target $target")
        }
        null
    } catch (e: SpecException) {
        e.message ?: ""
    }

    /**
     * 표지가 붙은 문서에서 Kotlin의 거부 메시지가 그 표지의 원인인지 본다. 원인이면 그 이름, 아니면 null.
     * where는 바꾼 잎의 자리(`target.path`, 예: `pipeline.landmarks.corner_outer_r`)다.
     */
    private fun flaggedCause(flags: List<String>, where: String, ktMessage: String): String? {
        if ("int_outside_int64" in flags) {
            val m = INT64_MESSAGE.matchEntire(ktMessage) ?: return null
            val n = java.math.BigInteger(m.groupValues[1])
            val inLong = n.bitLength() < 64
            return if (inLong) null else "int_outside_int64 (MiniJson 64-bit message)"
        }
        if ("int_outside_int32" in flags) {
            val m = INT32_MESSAGE.matchEntire(ktMessage) ?: return null
            if (m.groupValues[1] != where) return null
            val n = m.groupValues[2].toLong()
            val inInt = n >= Int.MIN_VALUE && n <= Int.MAX_VALUE
            return if (inInt) null else "int_outside_int32 (32-bit range message at the changed leaf)"
        }
        return null // accepted_changed_string, 또는 표지 없음: 차이를 허용하지 않는다
    }

    @Test
    fun everyDocumentJudgedAsPythonDoes() {
        val data = TestData.readyJson("components/C-loader.json")
        assertTrue(data.containsKey("depends_on"), "C-loader.json has no depends_on")
        val cases = TestData.list(data["cases"])
        val verifiedPipelineHash = SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes()).pipelineHash

        var sameVerdictAndMessage = 0
        var accepted = 0
        var refused = 0
        val perTarget = java.util.TreeMap<String, Int>()
        val flaggedTotal = java.util.TreeMap<String, Int>()
        val flaggedSame = java.util.TreeMap<String, Int>()
        val flaggedPyAcceptsKtRefuses = java.util.TreeMap<String, Int>()
        val flaggedOtherMessage = java.util.TreeMap<String, Int>()
        var hashNotRecomputed = 0
        var flaggedDiffDocs = 0
        val perCause = java.util.TreeMap<String, Int>()
        val failures = ArrayList<String>()
        val examples = ArrayList<String>()

        for ((i, c) in cases.withIndex()) {
            val case = TestData.obj(c)
            val target = case["target"] as String
            val text = case["text"] as String
            val pyAccepted = case["accepted"] as Boolean
            val pyError = case["error"]?.let { TestData.obj(it) }
            val pyMessage = pyError?.get("message") as String?
            if (pyError != null) assertEquals("SpecError", pyError["type"], "case $i")
            assertEquals(pyAccepted, pyError == null, "case $i: accepted and error disagree")
            val flags = TestData.list(case["flags"]).map { it as String }
            perTarget.merge(target, 1, Int::plus)
            for (f in flags) flaggedTotal.merge(f, 1, Int::plus)

            val ktMessage = judge(target, text)
            val ktAccepted = ktMessage == null
            if (ktAccepted) accepted++ else refused++
            val label = "case $i $target ${case["path"]} '${case["change"]}' flags=$flags"
            when {
                ktAccepted == pyAccepted && ktMessage == pyMessage -> {
                    sameVerdictAndMessage++
                    for (f in flags) flaggedSame.merge(f, 1, Int::plus)
                }
                !pyAccepted && ktAccepted && flags.isEmpty() && target == "feature_spec" &&
                    pyMessage!!.startsWith("pipeline_hash mismatch: ") -> {
                    // 계획 5.10절: pipeline_hash를 다시 계산하지 않는다. 시작 검사가 잡는지 확인한다.
                    val spec = SpecLoader.parseFeatureSpec(text.toByteArray(Charsets.UTF_8))
                    if (spec.pipelineHash == verifiedPipelineHash) {
                        failures.add("$label: pipeline_hash equals the verified value, H3 would not refuse it")
                    } else {
                        hashNotRecomputed++
                        examples.add("$label: Python '$pyMessage'; Kotlin accepts (H3_SPEC_VS_VERIFIED refuses '${spec.pipelineHash}')")
                    }
                }
                !ktAccepted && flaggedCause(flags, "$target.${case["path"]}", ktMessage!!) != null -> {
                    val cause = flaggedCause(flags, "$target.${case["path"]}", ktMessage)!!
                    flaggedDiffDocs++
                    perCause.merge(cause, 1, Int::plus)
                    for (f in flags) {
                        if (pyAccepted) flaggedPyAcceptsKtRefuses.merge(f, 1, Int::plus) else flaggedOtherMessage.merge(f, 1, Int::plus)
                    }
                    if (examples.size < 40) {
                        examples.add(if (pyAccepted) "$label: Python accepts; Kotlin '$ktMessage'" else "$label: Python '$pyMessage'; Kotlin '$ktMessage'")
                    }
                }
                else -> failures.add("$label: Python accepted=$pyAccepted '$pyMessage'; Kotlin accepted=$ktAccepted '$ktMessage'")
            }
        }

        println("CLoaderTest: ${cases.size} documents $perTarget; Kotlin accepted $accepted, refused $refused")
        println("CLoaderTest: same verdict and same message $sameVerdictAndMessage")
        println("CLoaderTest: flagged documents per flag $flaggedTotal")
        println("CLoaderTest:   of them same verdict and message $flaggedSame")
        println("CLoaderTest:   Python accepts, Kotlin refuses $flaggedPyAcceptsKtRefuses")
        println("CLoaderTest:   both refuse, other message $flaggedOtherMessage")
        println("CLoaderTest:   Kotlin refusal per cause $perCause")
        println("CLoaderTest: pipeline_hash not recomputed (plan 5.10; Python refuses, Kotlin accepts, H3 refuses) $hashNotRecomputed")
        println("CLoaderTest: unexplained differences ${failures.size}")
        for (e in examples.take(12)) println("CLoaderTest example: $e")
        assertEquals(emptyList(), failures.take(30))
        // 표지마다: 같음 + 두 허용 방향 = 그 표지의 문서 수(건너뛴 문서가 없다)
        for ((flag, total) in flaggedTotal) {
            val sum = (flaggedSame[flag] ?: 0) + (flaggedPyAcceptsKtRefuses[flag] ?: 0) + (flaggedOtherMessage[flag] ?: 0)
            assertEquals(total, sum, "flag $flag")
        }
        assertEquals(cases.size, sameVerdictAndMessage + hashNotRecomputed + flaggedDiffDocs, "every document is counted once")
    }

    private companion object {
        /** MiniJson.kt: 정수 글자가 Long 밖. */
        val INT64_MESSAGE = Regex("""JSON: integer (-?[0-9]+) does not fit in 64 bits at offset [0-9]+""")
        /** SpecLoader.kt `int()`: Int 칸의 정수가 Int 밖(R22). */
        val INT32_MESSAGE = Regex("""(.+): integer (-?[0-9]+) is outside the 32-bit range of this field""")
    }
}
