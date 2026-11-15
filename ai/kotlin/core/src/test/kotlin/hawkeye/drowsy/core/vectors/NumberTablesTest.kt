package hawkeye.drowsy.core.vectors

import hawkeye.drowsy.core.spec.MiniJson
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.Hkdiff1
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * E2의 숫자 표 (계획 9.1절 E2, 9.2.2절 끝; FORMATS.md 2.1절). Python이 같은 double을 세 번 적었다: canary.bin(2진 f64),
 * canary.json의 `f64`(Python `repr` 10진 글자), `f64_hex`(`float.hex()`). Kotlin은 2진을 읽고, 10진 글자를 실행기의 JSON
 * 읽기(MiniJson)로, 16진 글자를 `Double.parseDouble`로 읽어 세 비트(`doubleToRawLongBits`)가 모두 같은지 본다.
 * 정수 1,011개도 2진(i64)과 10진(JSON 정수 → Long)이 같아야 한다.
 */
class NumberTablesTest {

    @Test
    fun binaryDecimalAndHexGiveTheSameBits() {
        val binBytes = TestData.readyBytes("numbers/canary.bin")
        val bin = Hkdiff1.read(binBytes)
        TestData.checkReference(bin.header, "numbers/canary.bin")
        val json = TestData.readyJson("numbers/canary.json")

        val nF = (json["count_f64"] as Long).toInt()
        val nI = (json["count_i64"] as Long).toInt()
        assertEquals(json["count_f64"], bin.header["count_f64"])
        assertEquals(json["count_i64"], bin.header["count_i64"])

        val binF = bin.f64Bits("f64")
        val dec = TestData.list(json["f64"])
        val hex = TestData.list(json["f64_hex"])
        assertEquals(nF, binF.size)
        assertEquals(nF, dec.size)
        assertEquals(nF, hex.size)
        var decMismatch = 0
        var hexMismatch = 0
        var intLiterals = 0
        val first = ArrayList<String>()
        for (i in 0 until nF) {
            val d = dec[i]
            val decBits = when (d) {
                is Double -> d.toRawBits()
                is Long -> { intLiterals++; d.toDouble().toRawBits() } // Python json은 실수를 언제나 소수점이나 지수로 쓴다
                else -> throw AssertionError("f64[$i] is not a number: $d")
            }
            val hexBits = java.lang.Double.parseDouble(hex[i] as String).toRawBits()
            if (decBits != binF[i]) {
                decMismatch++
                if (first.size < 20) first.add("f64[$i] bin=${java.lang.Long.toHexString(binF[i])} dec=${java.lang.Long.toHexString(decBits)}")
            }
            if (hexBits != binF[i]) {
                hexMismatch++
                if (first.size < 20) first.add("f64[$i] bin=${java.lang.Long.toHexString(binF[i])} hex=${hex[i]}")
            }
        }

        val binI = bin.i64("i64")
        val ints = TestData.list(json["i64"])
        assertEquals(nI, binI.size)
        assertEquals(nI, ints.size)
        var intMismatch = 0
        for (i in 0 until nI) {
            val v = ints[i]
            if (v !is Long || v != binI[i]) {
                intMismatch++
                if (first.size < 20) first.add("i64[$i] bin=${binI[i]} json=$v")
            }
        }

        // 고정한 까다로운 값(FORMATS.md 2.1절): 머리의 fixed_f64가 그 인덱스다.
        val fixed = TestData.list(bin.header["fixed_f64"]).map { (it as Long).toInt() }
        val fixedI = TestData.list(bin.header["fixed_i64"]).map { (it as Long).toInt() }
        println(
            "NumberTablesTest: f64 $nF (fixed ${fixed.size}), i64 $nI (fixed ${fixedI.size}); " +
                "decimal-vs-binary bit mismatches $decMismatch, hex-vs-binary $hexMismatch, i64 mismatches $intMismatch, " +
                "decimal integer literals $intLiterals",
        )
        assertEquals(0, decMismatch + hexMismatch + intMismatch, first.joinToString("\n"))
        assertEquals(0, intLiterals, "a float written as an integer literal")
    }

    /** 고정값 몇 개를 비트로 직접 본다(-0.0의 부호, 가장 작은 비정규수, 2^53 + 1의 반올림, 1731465123457). */
    @Test
    fun awkwardFixedValues() {
        val json = TestData.readyJson("numbers/canary.json")
        val dec = TestData.list(json["f64"])
        val asBits = dec.map { (it as Double).toRawBits() }.toSet()
        for (v in listOf(0.1, -0.0, 0.0, 5e-324, Double.MAX_VALUE, 0.30000000000000004, 0.4, 0.6, 16.49993,
            java.lang.Double.MIN_NORMAL, 9007199254740992.0, 1731465123457.0)) {
            assertTrue(v.toRawBits() in asBits, "fixed value ${java.lang.Double.toHexString(v)} missing")
        }
        assertEquals((-0.0).toRawBits(), (MiniJson.parse("-0.0") as Double).toRawBits())
        val ints = TestData.list(json["i64"]).toSet()
        for (v in listOf(0L, 2147483647L, 2147483648L, -2147483649L, 9007199254740993L, 1731465123457L,
            Long.MAX_VALUE, Long.MIN_VALUE)) {
            assertTrue(v in ints, "fixed integer $v missing")
        }
    }
}
