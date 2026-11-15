package hawkeye.drowsy.core

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToLong
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * 이식이 기대는 Kotlin 의미를 첫 빌드에서 사실로 바꾸는 시험(계획 5.0절 R2, R4, R6, R7, R9, R22, 가정 A4).
 *
 * 기대값은 바꾸지 않은 Python 참조 함수(`drowsy.blink._round_half_up`, `drowsy.calibrate.median`,
 * `nearest_rank_quantile`, `_clip01`)를 CPython 3.11로 돌린 값이다. 그 스크립트와 출력은 단계 0 작업 폴더의
 * `semantics_expected.py`, `semantics_expected.log`에 있다. 여기의 이식 함수는 시험용 사본이고 라이브러리 코드가 아니다.
 *
 * 실수의 같음은 비트로 본다(`toRawBits`). `-0.0`과 `0.0`을 가르기 위해서다.
 */
class KotlinSemanticsTest {

    // ----- 시험용 이식 사본 (Python 문장 그대로) -----

    /** `blink.py:337-339` `int(math.floor(x + 0.5))`. R4. */
    private fun roundHalfUp(x: Double): Long = floor(x + 0.5).toLong()

    /** R7의 비교자: `<`만 쓴다. `-0.0`과 `0.0`, NaN은 같은 것으로 본다. */
    private val pythonOrder = Comparator<Double> { a, b -> if (a < b) -1 else if (b < a) 1 else 0 }

    /** `calibrate.py` `median`. */
    private fun median(values: List<Double>): Double {
        val ordered = values.sortedWith(pythonOrder)
        val n = ordered.size
        val mid = n / 2
        if (n % 2 == 1) return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0
    }

    /** `calibrate.py` `nearest_rank_quantile`. R5. */
    private fun nearestRankQuantile(values: List<Double>, q: Double): Double {
        val n = values.size
        val ordered = values.sortedWith(pythonOrder)
        var rank = ceil(q * n).toInt()
        if (rank < 1) rank = 1
        if (rank > n) rank = n
        return ordered[rank - 1]
    }

    /** `calibrate.py:391-396` `_clip01`. R6. */
    private fun clip01(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x > 1.0) return 1.0
        return x
    }

    /** `calibrate.py:192-193`처럼 `if a > b: a = b`로 쓴 최솟값. R6. */
    private fun ifMin(a: Double, b: Double): Double {
        var r = a
        if (r > b) r = b
        return r
    }

    private fun assertBits(expected: Double, actual: Double, message: String) {
        assertEquals(expected.toRawBits(), actual.toRawBits(), "$message: expected $expected, actual $actual")
    }

    private fun signBits(values: List<Double>): List<Boolean> = values.map { it.toRawBits() < 0L }

    // ----- R4: 반올림 -----

    @Test
    fun roundHalfUpMatchesPython() {
        // (x, Python _round_half_up(x))
        val cases = listOf(
            0.49999999999999994 to 1L, 2.5 to 3L, -2.5 to -2L, -0.5 to 0L, 1.5 to 2L, -1.5 to -1L,
            0.5 to 1L, -2.6 to -3L, 4503599627370497.0 to 4503599627370498L, -0.0 to 0L,
        )
        for ((x, expected) in cases) {
            assertEquals(expected, roundHalfUp(x), "floor(x + 0.5) for x=$x")
        }
    }

    @Test
    fun otherRoundingFunctionsDifferFromPython() {
        // Math.round와 roundToLong은 x + 0.5를 double로 더하지 않는다: 0.49999999999999994에서 0 (Python 1).
        assertEquals(0L, Math.round(0.49999999999999994))
        assertEquals(0L, 0.49999999999999994.roundToLong())
        // 2^52 + 1에서 x + 0.5는 짝수 쪽으로 올라가 ...498이 된다. Math.round는 ...497 (Python 4503599627370498).
        assertEquals(4503599627370497L, Math.round(4503599627370497.0))
        // kotlin.math.round는 짝수 쪽 반올림이다: 2.5 → 2.0 (Python 3).
        assertBits(2.0, kotlin.math.round(2.5), "kotlin.math.round(2.5)")
        // (x + 0.5).toLong()은 0 쪽으로 자른다: -2.6 → -2 (Python -3). 음이 아닌 x에서는 floor와 같다.
        assertEquals(-2L, (-2.6 + 0.5).toLong())
        assertEquals(roundHalfUp(2.6), (2.6 + 0.5).toLong())
    }

    // ----- R6: 최솟값·최댓값·자르기 -----

    @Test
    fun ifClipMatchesPython() {
        assertBits(-0.0, clip01(-0.0), "clip01(-0.0)")
        assertBits(0.0, clip01(0.0), "clip01(0.0)")
        assertTrue(clip01(Double.NaN).isNaN(), "clip01(NaN)")
        assertBits(0.0, clip01(-1e-300), "clip01(-1e-300)")
        assertBits(1.0, clip01(1.0000000000000002), "clip01(1.0000000000000002)")
        assertBits(0.5, clip01(0.5), "clip01(0.5)")
    }

    @Test
    fun ifMinMatchesPythonAndMinOfDoesNot() {
        // Python if_min 결과: (0.0, -0.0) → 0.0, (-0.0, 0.0) → -0.0, (nan, 1.0) → nan, (1.0, nan) → 1.0
        assertBits(0.0, ifMin(0.0, -0.0), "ifMin(0.0, -0.0)")
        assertBits(-0.0, ifMin(-0.0, 0.0), "ifMin(-0.0, 0.0)")
        assertTrue(ifMin(Double.NaN, 1.0).isNaN(), "ifMin(NaN, 1.0)")
        assertBits(1.0, ifMin(1.0, Double.NaN), "ifMin(1.0, NaN)")

        // minOf/maxOf(Math.min/max)는 -0.0을 0.0보다 작게 보고 NaN을 퍼뜨린다. 그래서 Python과 다르다.
        assertBits(-0.0, minOf(0.0, -0.0), "minOf(0.0, -0.0)")
        assertTrue(minOf(1.0, Double.NaN).isNaN(), "minOf(1.0, NaN)")
        assertBits(0.0, maxOf(-0.0, 0.0), "maxOf(-0.0, 0.0)")
        assertBits(0.0, maxOf(0.0, -0.0), "maxOf(0.0, -0.0)")
    }

    @Test
    fun coerceInOnTheseInputsEqualsIfClip() {
        // 잰 사실: Double.coerceIn은 기본형 비교를 쓰므로 이 입력들에서는 if 자르기와 같은 값을 낸다.
        // R6은 그래도 coerceIn을 쓰지 않는다(일곱 모듈에 없는 것을 넣지 않는다, R21).
        assertBits(-0.0, (-0.0).coerceIn(0.0, 1.0), "coerceIn(-0.0)")
        assertTrue(Double.NaN.coerceIn(0.0, 1.0).isNaN(), "coerceIn(NaN)")
        assertBits(1.0, 1.0000000000000002.coerceIn(0.0, 1.0), "coerceIn(1.0000000000000002)")
    }

    // ----- R7: 정렬, 중앙값, 분위수 -----

    @Test
    fun stableLessThanSortKeepsSignedZeroArrivalOrderLikePython() {
        val values = listOf(2.0, 0.0, -0.0, 1.0, -0.0, 0.0, -1.0)
        // Python sorted(values) = [-1.0, 0.0, -0.0, -0.0, 0.0, 1.0, 2.0]
        val expected = listOf(-1.0, 0.0, -0.0, -0.0, 0.0, 1.0, 2.0)
        val actual = values.sortedWith(pythonOrder)
        assertEquals(expected.map { it.toRawBits() }, actual.map { it.toRawBits() })
        // 원본은 바뀌지 않는다(복사본 정렬).
        assertEquals(listOf(2.0, 0.0, -0.0, 1.0, -0.0, 0.0, -1.0).map { it.toRawBits() }, values.map { it.toRawBits() })
    }

    @Test
    fun otherSortsPutNegativeZeroFirst() {
        val values = listOf(2.0, 0.0, -0.0, 1.0, -0.0, 0.0, -1.0)
        val pythonSigns = listOf(true, false, true, true, false, false, false)

        val array = values.toDoubleArray()
        array.sort()
        assertEquals(listOf(true, true, true, false, false, false, false), signBits(array.toList()))
        assertNotEquals(pythonSigns, signBits(array.toList()))

        // List<Double>.sorted()는 Double.compareTo(전순서)를 쓴다.
        assertEquals(listOf(true, true, true, false, false, false, false), signBits(values.sorted()))
    }

    @Test
    fun sortedWithIsStableForObjects() {
        data class Item(val key: Double, val arrival: Int)
        val items = listOf(Item(1.0, 0), Item(0.0, 1), Item(1.0, 2), Item(-0.0, 3), Item(0.0, 4), Item(1.0, 5))
        val sorted = items.sortedWith { a, b -> if (a.key < b.key) -1 else if (b.key < a.key) 1 else 0 }
        assertEquals(listOf(1, 3, 4, 0, 2, 5), sorted.map { it.arrival })
    }

    @Test
    fun medianAndQuantileMatchPythonIncludingZeroSign() {
        // (값, Python median)
        val medians = listOf(
            listOf(0.0, -0.0) to 0.0,
            listOf(-0.0, 0.0) to 0.0,
            listOf(-0.0, 0.0, 0.0) to 0.0,
            listOf(0.0, -0.0, -0.0) to -0.0,
            listOf(0.0, -0.0, 0.0, -0.0) to 0.0,
            listOf(-0.0, 0.0, -0.0, 0.0) to 0.0,
            listOf(1.0, -0.0, 0.0, -1.0) to 0.0,
            listOf(0.0, -0.0, 0.0) to -0.0,
        )
        for ((values, expected) in medians) {
            assertBits(expected, median(values), "median($values)")
        }
        // (값, q, Python nearest_rank_quantile)
        val quantiles = listOf(
            Triple(listOf(0.0, -0.0, 0.0), 0.5, -0.0),
            Triple(listOf(-0.0, 0.0), 0.5, -0.0),
            Triple(listOf(0.0, -0.0), 0.5, 0.0),
            Triple(listOf(3.0, 1.0, 2.0), 0.1, 1.0),
            Triple(listOf(3.0, 1.0, 2.0), 0.9, 3.0),
        )
        for ((values, q, expected) in quantiles) {
            assertBits(expected, nearestRankQuantile(values, q), "quantile($values, $q)")
        }
        // DoubleArray.sort()로 만든 중앙값은 [0.0, -0.0, 0.0]에서 Python과 부호가 다르다.
        val array = doubleArrayOf(0.0, -0.0, 0.0)
        array.sort()
        assertBits(0.0, array[1], "DoubleArray.sort median of [0.0, -0.0, 0.0]")
    }

    // ----- 비교 연산자: 정적 형에 따라 의미가 바뀐다 (A4 "박싱된 비교") -----

    @Test
    fun comparisonSemanticsDependOnStaticType() {
        val pz = 0.0
        val nz = -0.0
        val nan = Double.NaN
        // Double로 알려진 값: IEEE 754
        assertTrue(nz == pz)
        assertFalse(nz < pz)
        assertFalse(nan == nan)
        assertFalse(nan < 1.0 || nan > 1.0)

        // Double?로 알려진 값도 IEEE 754다.
        val nzN: Double? = -0.0
        val pzN: Double? = 0.0
        assertTrue(nzN == pzN)

        // Any나 Comparable로 알려진 값: equals/compareTo(전순서). -0.0 < 0.0, NaN == NaN.
        val nzAny: Any = -0.0
        val pzAny: Any = 0.0
        assertFalse(nzAny == pzAny)
        val nzCmp: Comparable<Double> = -0.0
        assertTrue(nzCmp < 0.0)
        val nanAny: Any = Double.NaN
        assertTrue(nanAny == Double.NaN as Any)
    }

    // ----- R9: data class와 목록의 같음 -----

    @Test
    fun dataClassAndListEquality() {
        data class WithArray(val xs: DoubleArray)
        data class WithList(val xs: List<Double>)
        data class WithDouble(val x: Double)

        // 배열을 든 data class는 참조로 비교한다.
        assertNotEquals(WithArray(doubleArrayOf(1.0)), WithArray(doubleArrayOf(1.0)))
        // 목록을 든 data class는 내용으로 비교한다.
        assertEquals(WithList(listOf(1.0, 2.0)), WithList(listOf(1.0, 2.0)))

        // data class의 Double 필드와 List<Double>은 Double.equals(전순서) 의미다: -0.0 ≠ 0.0, NaN = NaN.
        // 그래서 출력 비교는 전체 객체 ==가 아니라 필드별 기본형 비교로 한다(E8의 "-0.0 == 0.0").
        assertNotEquals(WithDouble(-0.0), WithDouble(0.0))
        assertEquals(WithDouble(Double.NaN), WithDouble(Double.NaN))
        assertNotEquals(listOf(-0.0), listOf(0.0))
        assertEquals(listOf(Double.NaN), listOf(Double.NaN))
        assertTrue(WithDouble(-0.0).x == WithDouble(0.0).x)
    }

    // ----- R2, R3, R22: 정수 나눗셈과 줄이기 -----

    @Test
    fun integerDivisionAndNarrowing() {
        val faceMs = 300_000L
        val deadlineMs = 600_000L
        // Long / Long은 정수 나눗셈이다(조용히 틀린다). Python 300000 / 600000 = 0.5.
        assertEquals(0L, faceMs / deadlineMs)
        assertBits(0.5, faceMs.toDouble() / deadlineMs.toDouble(), "toDouble division")
        // Kotlin /는 0 쪽으로 자르고 Python //는 내림이다: -7 / 2 = -3, Python -7 // 2 = -4.
        assertEquals(-3L, -7L / 2L)
        assertEquals(-4L, Math.floorDiv(-7L, 2L))
        // Long.toInt()는 조용히 줄인다: 4294967596 → 300 (R22: 범위를 검사해야 한다).
        assertEquals(300, 4294967596L.toInt())
    }
}
