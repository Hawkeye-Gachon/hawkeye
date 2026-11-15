package hawkeye.drowsy.core.vectors

import hawkeye.drowsy.core.spec.MiniJson
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorComparator
import hawkeye.drowsy.core.testkit.VectorFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * E2의 비교기 표 (계획 9.1절 E2; FORMATS.md 2.2절). 행마다 Python `drowsy.vectors._compare`를 한 번 부른 결과
 * (`python_ok`, `python_problems`)가 있다. Kotlin 비교기(SPEC 16장 층)가 같은 판정과 같은 문장을 내야 한다.
 * 유일한 예외는 `nan_row`(실제 값이나 기대값에 NaN이 있는 행)다: Python은 통과시키고 Kotlin은 실패시킨다(계획 9.0절).
 * 그 행들은 따로 세어 적는다. 파일의 depends_on(drowsy/vectors.py)이 저장소와 다르면 실패한다(리드 판정 1).
 */
class ComparatorTableTest {

    /** FORMATS.md 2.2절의 형 있는 값 V → JSON 나무의 값. */
    private fun decode(v: Any?): Any? {
        val m = TestData.obj(v)
        return when (m["t"]) {
            "f" -> when (val h = m["hex"] as String) {
                "nan" -> Double.NaN
                "inf" -> Double.POSITIVE_INFINITY
                "-inf" -> Double.NEGATIVE_INFINITY
                else -> java.lang.Double.parseDouble(h)
            }
            "i" -> m["v"] as Long
            "b" -> m["v"] as Boolean
            "n" -> null
            "s" -> m["v"] as String
            "l" -> TestData.list(m["v"]).map { decode(it) }
            "d" -> LinkedHashMap<String, Any?>().also { out -> for ((k, x) in TestData.obj(m["v"])) out[k] = decode(x) }
            else -> throw AssertionError("unknown value type ${m["t"]}")
        }
    }

    /** 형과 비트까지 같은지(Double은 raw bits, -0.0과 0.0을 가른다). */
    private fun sameTyped(a: Any?, b: Any?): Boolean = when {
        a == null || b == null -> a == null && b == null
        a is Double && b is Double -> a.toRawBits() == b.toRawBits()
        a is List<*> && b is List<*> -> a.size == b.size && a.indices.all { sameTyped(a[it], b[it]) }
        a is Map<*, *> && b is Map<*, *> -> a.keys.toList() == b.keys.toList() && a.keys.all { sameTyped(a[it], b[it]) }
        else -> a.javaClass == b.javaClass && a == b
    }

    private fun hasNaN(v: Any?): Boolean = when (v) {
        is Double -> v.isNaN()
        is List<*> -> v.any { hasNaN(it) }
        is Map<*, *> -> v.values.any { hasNaN(it) }
        else -> false
    }

    /** 리드 판정 1: depends_on의 sha256이 저장소와 다르거나 파일이 없으면 실패한다(복사본 나무로 확인). */
    @Test
    fun dependsOnMismatchIsRefused() {
        val data = TestData.readyJson("numbers/comparator.json")
        TestData.checkDependsOn(data, "comparator.json")
        val changed = LinkedHashMap(data)
        changed["depends_on"] = mapOf("drowsy/vectors.py" to "0".repeat(64))
        kotlin.test.assertFailsWith<AssertionError> { TestData.checkDependsOn(changed, "changed") }
        changed["depends_on"] = mapOf("drowsy/no_such_module.py" to "0".repeat(64))
        kotlin.test.assertFailsWith<AssertionError> { TestData.checkDependsOn(changed, "missing") }
        changed["depends_on"] = emptyMap<String, Any?>()
        kotlin.test.assertFailsWith<AssertionError> { TestData.checkDependsOn(changed, "empty") }
    }

    @Test
    fun kotlinComparatorMatchesPythonExceptNaNRows() {
        val data = TestData.readyJson("numbers/comparator.json")
        assertTrue(data.containsKey("depends_on"), "comparator.json has no depends_on")
        val rows = TestData.list(data["rows"])

        var same = 0
        var sameMessages = 0
        var nanRows = 0
        var nanRowsKotlinFails = 0
        var jsonTextChecked = 0
        val bad = ArrayList<String>()
        val tolerancesSeen = LinkedHashMap<String, Map<String, Any?>>()
        for (r in rows) {
            val row = TestData.obj(r)
            val id = row["id"]
            val section = row["section"] as String
            val key = row["key"] as String
            val tolerance = TestData.obj(row["tolerance"])
            tolerancesSeen.putIfAbsent(section, tolerance)
            val expected = decode(row["expected"])
            val actual = decode(row["actual"])

            // expected_json / actual_json: 벡터 파일이 쓰는 모양의 JSON 글자. 있으면 MiniJson으로 읽은 값이 형과 비트까지 같아야 한다.
            for ((text, value) in listOf(row["expected_json"] to expected, row["actual_json"] to actual)) {
                if (text != null) {
                    jsonTextChecked++
                    val parsed = MiniJson.parse(text as String)
                    if (!sameTyped(parsed, value)) bad.add("row $id: JSON text $text reads as $parsed")
                }
            }

            val problems = ArrayList<String>()
            VectorComparator.compare(section, key, expected, actual, tolerance, "expected.$section", problems)
            val kotlinOk = problems.isEmpty()
            val pythonOk = row["python_ok"] as Boolean
            val pythonProblems = TestData.list(row["python_problems"]).map { it as String }
            val nanRow = row["nan_row"] as Boolean
            if (nanRow != (hasNaN(expected) || hasNaN(actual))) bad.add("row $id: nan_row flag $nanRow does not match the values")
            if (nanRow) {
                nanRows++
                // 계획 9.0절: Python은 NaN을 통과시키고(python_ok), Kotlin은 실패시킨다.
                if (pythonOk && !kotlinOk) nanRowsKotlinFails++ else bad.add("row $id (NaN): python_ok=$pythonOk kotlin_ok=$kotlinOk $problems")
                continue
            }
            if (kotlinOk == pythonOk) same++ else bad.add("row $id $section/$key: python_ok=$pythonOk kotlin_ok=$kotlinOk $problems")
            if (problems == pythonProblems) sameMessages++ else bad.add("row $id: python $pythonProblems kotlin $problems")
        }

        // 표의 허용 오차(Python TOLERANCES에서 왔다)가 Kotlin 벡터 모형의 표와 키·값 모두 같다.
        for ((section, tol) in tolerancesSeen) {
            val table = VectorFile.TOLERANCES[section] ?: emptyMap()
            assertEquals(table.keys, tol.keys, "tolerance keys of $section")
            for ((k, v) in table) assertEquals(v, (tol[k] as Number).toDouble(), "tolerance $section.$k")
        }
        assertEquals(VectorFile.TOLERANCES.keys + "counters", tolerancesSeen.keys.toSet(), "sections covered")

        println(
            "ComparatorTableTest: ${rows.size} rows; verdict equal to Python ${same}/${rows.size - nanRows}, " +
                "messages equal $sameMessages/${rows.size - nanRows}; NaN rows $nanRows (Python passes, Kotlin fails: " +
                "$nanRowsKotlinFails); JSON texts read back $jsonTextChecked; differences ${bad.size}",
        )
        assertEquals(emptyList(), bad.take(30))
        assertEquals(rows.size - nanRows, same)
        assertEquals(rows.size - nanRows, sameMessages)
        assertEquals(nanRows, nanRowsKotlinFails)
    }
}
