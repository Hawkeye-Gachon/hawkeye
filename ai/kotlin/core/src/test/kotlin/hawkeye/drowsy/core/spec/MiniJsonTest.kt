package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.testdata.TestData
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * MiniJson(계획 5.10절)의 시험. 문법과 거부 규칙은 계획이 정한 읽기 동작이다(Python 기대값이 아니다).
 * 수의 비트는 Python이 쓴 숫자 표(`kotlin/.diffdata/numbers/canary.json`: 같은 double을 10진과 `float.hex()`로 적은 것)로
 * 본다. 그 자료가 없으면 그 시험은 건너뛴다.
 */
class MiniJsonTest {

    private fun parse(s: String): Any? = MiniJson.parse(s)

    @Test
    fun valuesAndTheirKotlinTypes() {
        val v = TestData.obj(parse("""{"i": 250, "f": 250.0, "e": 1e2, "n": -0, "z": -0.0, "t": true, "u": false, "x": null, "s": "a", "l": [1, 2.5]}"""))
        assertEquals(250L, v["i"])
        assertEquals(250.0, v["f"])
        assertEquals(100.0, v["e"])
        assertEquals(0L, v["n"])
        assertEquals((-0.0).toRawBits(), (v["z"] as Double).toRawBits())
        assertEquals(true, v["t"])
        assertEquals(false, v["u"])
        assertTrue(v.containsKey("x"))
        assertNull(v["x"])
        assertEquals("a", v["s"])
        assertEquals(listOf<Any?>(1L, 2.5), v["l"])
        assertEquals(listOf("i", "f", "e", "n", "z", "t", "u", "x", "s", "l"), v.keys.toList(), "file key order is kept")
    }

    @Test
    fun integersAreLongAndOverflowIsRejected() {
        assertEquals(Long.MAX_VALUE, parse("9223372036854775807"))
        assertEquals(Long.MIN_VALUE, parse("-9223372036854775808"))
        assertEquals(1731465423289L, parse("1731465423289"))
        assertFailsWith<SpecException> { parse("9223372036854775808") }
        assertFailsWith<SpecException> { parse("-9223372036854775809") }
        assertFailsWith<SpecException> { parse("4294967596000000000000") }
    }

    @Test
    fun pythonNonFiniteLiteralsAreReadAsDoubles() {
        assertTrue((parse("NaN") as Double).isNaN())
        assertEquals(Double.POSITIVE_INFINITY, parse("Infinity"))
        assertEquals(Double.NEGATIVE_INFINITY, parse("[-Infinity]").let { TestData.list(it)[0] })
        assertFailsWith<SpecException> { parse("-NaN") }
        assertFailsWith<SpecException> { parse("nan") }
        assertFailsWith<SpecException> { parse("Inf") }
    }

    @Test
    fun stringsAndEscapes() {
        assertEquals("a\"b\\c/d\b\u000C\n\r\t", parse(""""a\"b\\c\/d\b\f\n\r\t""""))
        assertEquals("😀", parse(""""😀""""))
        assertEquals("졸음", parse(""""졸음""""))
        assertEquals("졸음", MiniJson.parse("\"졸음\"".toByteArray(Charsets.UTF_8)))
        assertFailsWith<SpecException> { parse("\"a\u0001b\"") }
        assertFailsWith<SpecException> { parse(""""\x"""") }
        assertFailsWith<SpecException> { parse(""""\u12"""") }
        assertFailsWith<SpecException> { parse(""""abc""") }
    }

    /**
     * `\u` 뒤의 네 글자는 ASCII 16진 숫자(0-9, a-f, A-F)만 받는다. 기대 판정은 Python 3.11 `json.loads`가 낸 것이다
     * (2026-11-15, `.venv` Python 3.11.16에서 잼). `Character.digit`은 아라비아-인도 숫자, 데바나가리 숫자, 전각 숫자·글자도 받으므로
     * 쓰지 않는다.
     */
    @Test
    fun unicodeEscapeAcceptsOnlyAsciiHexDigits() {
        // Python: 'A', 'J', 'J', '\uffff', '\uffff'
        assertEquals("A", parse(""""\u0041""""))
        assertEquals("J", parse(""""\u004a""""))
        assertEquals("J", parse(""""\u004A""""))
        assertEquals("\uFFFF", parse(""""\uFFFF""""))
        assertEquals("\uFFFF", parse(""""\uffff""""))
        // Python: JSONDecodeError "Invalid \uXXXX escape"
        val rejected = listOf(
            "\"\\u\u0660\u0660\u0664\u0661\"",            // 아라비아-인도 숫자 ٠٠٤١
            "\"\\u\uFF10\uFF10\uFF14\uFF21\"",            // 전각 ００４Ａ
            "\"\\u\u0966\u0966\u0966\u0966\"",            // 데바나가리 숫자 ००००
            "\"\\u004\u0661\"",                           // 마지막 한 자리만 아라비아-인도 숫자
            "{\"source\": \"perclos\\u\u0660\u0660\u0660\u0660\"}",
            "\"\\u00g1\"",
            "\"\\u+041\"",
            "\"\\u 041\"",
        )
        for (bad in rejected) {
            assertFailsWith<SpecException>(bad) { parse(bad) }
            assertFailsWith<SpecException>(bad) { MiniJson.parse(bad.toByteArray(Charsets.UTF_8)) }
        }
    }

    @Test
    fun strictnessRules() {
        // 중복 키, BOM, 남은 글자 (계획 5.10절)
        val dup = assertFailsWith<SpecException> { parse("""{"a": 1, "b": 2, "a": 3}""") }
        assertTrue(dup.message!!.contains("duplicate key \"a\""), dup.message)
        parse("""{"a": {"a": 1}}""")
        assertFailsWith<SpecException> { parse("﻿{}") }
        assertFailsWith<SpecException> { MiniJson.parse(byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte(), '{'.code.toByte(), '}'.code.toByte())) }
        assertFailsWith<SpecException> { parse("{} {}") }
        assertFailsWith<SpecException> { parse("1 2") }
        // 잘못된 UTF-8 바이트
        assertFailsWith<SpecException> { MiniJson.parse(byteArrayOf('"'.code.toByte(), 0xC3.toByte(), '"'.code.toByte())) }
        // 수 문법
        for (bad in listOf("01", "1.", ".5", "+1", "-", "1e", "1e+", "0x10", "1.5.", "--1")) {
            assertFailsWith<SpecException>(bad) { parse(bad) }
        }
        // 그 밖의 문법
        for (bad in listOf("", " ", "[1,]", "{\"a\":1,}", "{a:1}", "[1 2]", "{\"a\" 1}", "tru", "nul", "'a'", "[", "{")) {
            assertFailsWith<SpecException>(bad) { parse(bad) }
        }
        // JSON의 공백 넷만 받는다
        assertEquals(listOf<Any?>(1L), parse(" \t\r\n[ 1 ]\n"))
        assertFailsWith<SpecException> { parse(" [1]") }
    }

    @Test
    fun nestingDepthIsLimited() {
        val ok = "[".repeat(MiniJson.MAX_DEPTH) + "]".repeat(MiniJson.MAX_DEPTH)
        parse(ok)
        val deep = "[".repeat(MiniJson.MAX_DEPTH + 1) + "]".repeat(MiniJson.MAX_DEPTH + 1)
        assertFailsWith<SpecException> { parse(deep) }
    }

    @Test
    fun exceptionIsAnIllegalArgumentException() {
        val e: IllegalArgumentException = assertFailsWith<SpecException> { parse("[") }
        assertTrue(e.message!!.startsWith("JSON: "))
    }

    @Test
    fun everyVectorFileParses() {
        val files = TestData.vectorFiles()
        assertEquals(102, files.size, "tests/vectors/*.json (README.md: 102 vectors)")
        var frames = 0L
        for (f in files) {
            val root = TestData.obj(MiniJson.parse(f.readBytes()))
            assertEquals(f.name.removeSuffix(".json"), root["name"], f.name)
            frames += TestData.list(root["frames"]).size
        }
        println("MiniJsonTest: ${files.size} vector files parsed, $frames frames")
    }

    @Test
    fun exportedConfigFilesParse() {
        val spec = TestData.obj(TestData.parseFile(java.io.File(TestData.export, "feature_spec_v0.json")))
        assertTrue(spec["pipeline_hash"] is String)
        val rules = TestData.obj(TestData.parseFile(java.io.File(TestData.export, "rules.json")))
        assertEquals(7, TestData.list(rules["alert_priority"]).size)
    }

    /** 10진 글자를 MiniJson으로 읽은 double의 비트 = Python `float.hex()` 글자를 `Double.parseDouble`로 읽은 비트. */
    @Test
    fun decimalNumbersMatchPythonHexBits() {
        val data = TestData.readyJson("numbers/canary.json")
        val dec = TestData.list(data["f64"])
        val hex = TestData.list(data["f64_hex"])
        assertEquals((data["count_f64"] as Long).toInt(), dec.size)
        assertEquals(dec.size, hex.size)
        var mismatches = 0
        val first = ArrayList<String>()
        for (i in dec.indices) {
            val a = TestData.num(dec[i])
            val b = java.lang.Double.parseDouble(hex[i] as String)
            if (a.toRawBits() != b.toRawBits()) {
                mismatches++
                if (first.size < 20) first.add("[$i] ${hex[i]}: decimal gave ${java.lang.Double.toHexString(a)}")
            }
        }
        val ints = TestData.list(data["i64"])
        assertEquals((data["count_i64"] as Long).toInt(), ints.size)
        for (v in ints) assertTrue(v is Long, "i64 element $v is not read as Long")
        println("MiniJsonTest canary: ${dec.size} doubles, ${ints.size} integers, $mismatches bit mismatches")
        assertEquals(0, mismatches, first.joinToString("\n"))
    }
}
