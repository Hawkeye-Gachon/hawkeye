package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.testdata.TestData
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * 메시지 안의 Python 글자가 참조 Python 3.11과 같은지 본다 (계획 5.10절: 메시지는 Python의 문장 그대로).
 *
 * 기대값은 모두 Python 3.11.16(unicodedata 14.0.0)이 만든 것이다. 만든 스크립트: 작업 폴더
 * work/kotlin/impl/f_s1/gen_py_printable.py.
 * - spec_messages_expected.jsonl: `spec.pipeline_spec_from_dict(json.loads(text))`의 메시지. 모르는 키의 정렬(spec.py:526
 *   `sorted(str(key) ...)`, 코드 포인트 순)과, JDK 17의 유니코드 표에 없는 새 글자의 repr.
 * - py_printable_expected.json: 모든 코드 포인트 0..0x10FFFF의 `repr(chr(cp))`를 "\n"으로 이은 UTF-8의 sha256.
 */
class PyMessageTest {

    @Test
    fun loaderMessagesEqualPythonForUnicodeKeysAndCharacters() {
        val lines = javaClass.getResource("/spec_messages_expected.jsonl")!!.readText(Charsets.UTF_8).lines().filter { it.isNotBlank() }
        val failures = ArrayList<String>()
        for (line in lines) {
            val row = TestData.obj(MiniJson.parse(line))
            assertEquals("pipeline", row["target"])
            val got = try {
                SpecLoader.parsePipelineSpec((row["text"] as String).toByteArray(Charsets.UTF_8))
                null
            } catch (e: SpecException) {
                e.message
            }
            if (got != row["message"]) failures.add("${row["label"]}: Python ${PyRepr.value(row["message"])}, Kotlin ${PyRepr.value(got)}")
        }
        println("PyMessageTest: ${lines.size} loader messages compared, ${failures.size} differ")
        assertEquals(emptyList(), failures)
        assertEquals(23, lines.size)
    }

    @Test
    fun reprOfEveryCodePointEqualsPython() {
        val expected = TestData.obj(MiniJson.parse(javaClass.getResource("/py_printable_expected.json")!!.readText(Charsets.UTF_8)))
        val planes = TestData.list(expected["repr_sha256_per_plane"])
        val all = StringBuilder()
        val differing = ArrayList<String>()
        var nonPrintable = 0
        for (plane in 0 until 17) {
            val sb = StringBuilder()
            for (cp in plane * 0x10000 until (plane + 1) * 0x10000) {
                if (cp > plane * 0x10000) sb.append('\n')
                // 한 코드 포인트의 문자열(대리 글자 하나도 한 글자의 문자열이다, Python chr(cp)와 같다)
                val r = PyRepr.str(String(Character.toChars(cp)))
                if (r.startsWith("'\\") && r != "'\\\\'") nonPrintable++
                sb.append(r)
            }
            val text = sb.toString()
            if (plane > 0) all.append('\n')
            all.append(text)
            val h = TestData.sha256Hex(text.toByteArray(Charsets.UTF_8))
            if (h != planes[plane]) differing.add("plane $plane")
        }
        // 이스케이프하는 글자 수에는 \t, \n, \r가 들어간다(isprintable이 False인 글자 수와 같다)
        println("PyMessageTest: repr of 0x110000 code points; escaped $nonPrintable; planes differing from Python: $differing")
        assertEquals(emptyList(), differing)
        assertEquals(expected["repr_sha256_all"], TestData.sha256Hex(all.toString().toByteArray(Charsets.UTF_8)))
        assertEquals((expected["non_printable"] as Long).toInt(), nonPrintable)
        assertEquals((expected["ranges"] as Long).toInt(), PyPrintable.RANGE_COUNT)
    }

    /** compareCodePoints: Python `str` 비교의 순서(코드 포인트 순, 앞이 같으면 짧은 쪽이 먼저; Python 3.11에서 `a < b`로 확인한 것과 같다). */
    @Test
    fun unknownKeysAreSortedByCodePoint() {
        assertEquals(-1, SpecLoader.compareCodePoints("\uE000", "\uD83D\uDE00"))
        assertEquals(1, SpecLoader.compareCodePoints("\uD800\uDC00", "\uFFFF"))
        assertEquals(0, SpecLoader.compareCodePoints("a\uD83D\uDE00", "a\uD83D\uDE00"))
        assertEquals(-1, SpecLoader.compareCodePoints("ab", "abc"))
        assertEquals(1, SpecLoader.compareCodePoints("\uD800", "\uD7FF"))
        assertEquals(-1, SpecLoader.compareCodePoints("\uD800", "\uE000"))
    }
}
