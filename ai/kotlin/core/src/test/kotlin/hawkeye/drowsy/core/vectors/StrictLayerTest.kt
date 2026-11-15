package hawkeye.drowsy.core.vectors

import hawkeye.drowsy.core.spec.MiniJson
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorComparator
import hawkeye.drowsy.core.testkit.VectorFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 엄격 층(계획 9.0절 표)의 규칙 시험. 이 층은 Python에 없는 Kotlin 실행기의 규칙이라 기대값은 계획의 표다.
 * 손으로 만든 작은 나무에서, SPEC 16장 층은 통과하지만 엄격 층은 실패해야 하는 경우와 그 반대를 본다.
 */
class StrictLayerTest {

    private fun tree(json: String): Any? = MiniJson.parse(json)

    private fun run(section: String, e: String, a: String): Pair<List<String>, VectorComparator.StrictResult> {
        val problems = ArrayList<String>()
        val tol: Map<String, Any?> = VectorFile.TOLERANCES[section] ?: emptyMap()
        VectorComparator.compare(section, section, tree(e), tree(a), tol, "expected.$section", problems)
        val strict = VectorComparator.StrictResult()
        VectorComparator.strictLayer(section, tree(e), tree(a), "expected.$section", strict)
        return problems to strict
    }

    private fun event(source: String, tStart: Long, duration: Long, confidence: String) =
        """[{"t_ms": 5000, "type": "drowsy", "severity": 2, "t_start_ms": $tStart, "duration_ms": $duration, """ +
            """"confidence": $confidence, "source": "$source", "phase": "onset", "provisional": false}]"""

    @Test
    fun eventTimesAreExactUnlessSourceIsClosure() {
        // source face: ±1 ms는 SPEC 16장에서 통과, 엄격 층에서 실패
        val (p1, s1) = run("events", event("face", 1000, 3000, "1.0"), event("face", 1001, 2999, "1.0"))
        assertEquals(emptyList(), p1)
        assertEquals(2, s1.failures.size, s1.failures.toString())
        assertTrue(s1.closureTiming.isEmpty())
        // source closure: 역시 실패지만 걸침 판정 목록에도 적힌다
        val (p2, s2) = run("alerts", event("closure", 1000, 3000, "1.0"), event("closure", 1000, 3001, "1.0"))
        assertEquals(emptyList(), p2)
        assertEquals(1, s2.failures.size)
        assertEquals(1, s2.closureTiming.size)
        // 같으면 아무것도 없다
        val (p3, s3) = run("events", event("perclos", 1000, 3000, "0.25"), event("perclos", 1000, 3000, "0.25"))
        assertEquals(emptyList(), p3)
        assertTrue(s3.failures.isEmpty() && s3.notes.isEmpty())
    }

    @Test
    fun confidenceIsExactUnlessSourceIsModel() {
        // 1 ms 차이의 confidence(60초 보정에서 1.7e-5)는 1e-4 안이라 SPEC 16장은 통과한다
        val (p1, s1) = run("events", event("face", 0, 60000, "0.5"), event("face", 0, 60000, "0.500017"))
        assertEquals(emptyList(), p1)
        assertEquals(1, s1.failures.size)
        val (p2, s2) = run("events", event("model", 0, 60000, "0.5"), event("model", 0, 60000, "0.500017"))
        assertEquals(emptyList(), p2)
        assertTrue(s2.failures.isEmpty())
        // -0.0과 0.0은 수로 같다
        val (_, s3) = run("events", event("eye", 0, 1, "0.0"), event("eye", 0, 1, "-0.0"))
        assertTrue(s3.failures.isEmpty())
    }

    @Test
    fun runTimesAreExact() {
        val e = """[{"t_ms": 10, "start_ms": 100, "end_ms": 400, "duration_ms": 300, "amplitude": 0.9, "peak_closing_velocity": 5.0, "peak_opening_velocity": -4.0, "velocity_valid": true, "truncated": false}]"""
        val a = e.replace("\"end_ms\": 400", "\"end_ms\": 401").replace("\"duration_ms\": 300", "\"duration_ms\": 301")
        val (p, s) = run("runs", e, a)
        assertEquals(emptyList(), p)
        assertEquals(2, s.failures.size)
        assertEquals(2, s.closureTiming.size)
    }

    @Test
    fun exactStepFeaturesAndNumericNotes() {
        val values = (0 until 17).map { 0.25 + it }
        fun step(vs: List<Double>) = """[{"t_ms": 1, "k": 3, "t_end_ms": 4000, "full_lookback": true, "values": [${vs.joinToString(", ") { it.toString() }}]}]"""
        val e = step(values)
        for (j in 0 until 17) {
            val changed = values.toMutableList()
            changed[j] = Math.nextUp(changed[j]) // 1 ulp: SPEC 16장(1e-5)은 통과
            val (p, s) = run("steps", e, step(changed))
            assertEquals(emptyList(), p, "feature #$j")
            if (j in VectorComparator.EXACT_STEP_FEATURES) {
                assertEquals(1, s.failures.size, "feature #$j must be exact")
            } else {
                assertTrue(s.failures.isEmpty(), "feature #$j is not exact")
                assertTrue(s.notes.isEmpty(), "1 ulp is below 1e-9")
            }
        }
        // #2에 1e-7 차이: SPEC 16장 통과, 엄격 층은 실패가 아니라 "기대보다 큰 수치 차이"
        val changed = values.toMutableList()
        changed[2] = changed[2] + 1e-7
        val (p, s) = run("steps", e, step(changed))
        assertEquals(emptyList(), p)
        assertTrue(s.failures.isEmpty())
        assertEquals(1, s.notes.size)
    }

    @Test
    fun nanAndInfinityFailTheSpecLayer() {
        for (bad in listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY)) {
            val problems = ArrayList<String>()
            VectorComparator.compare("closure", "closure", 0.5, bad, VectorFile.TOLERANCES.getValue("closure"), "expected.closure", problems)
            assertEquals(1, problems.size, "actual $bad")
            val p2 = ArrayList<String>()
            VectorComparator.compare("closure", "closure", bad, bad, VectorFile.TOLERANCES.getValue("closure"), "expected.closure", p2)
            assertEquals(1, p2.size, "expected and actual $bad")
        }
    }

    /**
     * 허용 오차에 rel > 0이 있는 항목(steps, model_inputs 1e-5; runs 1e-4)에서도 무한대·NaN은 실패다(계획 9.0절).
     * `abs + rel·|e|`는 e가 무한대이면 무한대라서, 유한성을 따로 보지 않으면 무한대 기대값이 어떤 실제 값이든 통과시킨다.
     * 엄격 층도 같은 값을 실패로 적는다.
     */
    @Test
    fun nanAndInfinityFailInSectionsWithRelativeTolerance() {
        val values = (0 until 17).map { 0.25 + it }
        fun step(j: Int, v: String) = """[{"t_ms": 1, "k": 3, "t_end_ms": 4000, "full_lookback": true, "values": [""" +
            values.indices.joinToString(", ") { if (it == j) v else values[it].toString() } + "]}]"
        fun window(j: Int, v: String) = """[{"t_ms": 1, "k": 119, "features": [[""" +
            values.indices.joinToString(", ") { if (it == j) v else values[it].toString() } + "]]}]"
        fun runRow(v: String) = """[{"t_ms": 10, "start_ms": 100, "end_ms": 400, "duration_ms": 300, "amplitude": $v, """ +
            """"peak_closing_velocity": 5.0, "peak_opening_velocity": -4.0, "velocity_valid": true, "truncated": false}]"""
        val nonFinite = listOf("Infinity", "-Infinity", "NaN")
        val pairs = ArrayList<Pair<String, String>>() // (expected, actual)
        for (e in nonFinite) for (a in listOf("0.5", "Infinity", "-Infinity", "NaN")) pairs.add(e to a)
        for (a in nonFinite) pairs.add("0.5" to a)
        var checked = 0
        for ((e, a) in pairs) {
            for (j in listOf(2, 0)) { // #2는 엄격 층의 "그 밖의 실수", #0은 정확히 같아야 하는 특징
                for ((section, make) in listOf<Pair<String, (Int, String) -> String>>("steps" to ::step, "model_inputs" to ::window)) {
                    val (p, s) = run(section, make(j, e), make(j, a))
                    assertEquals(1, p.size, "$section #$j expected $e actual $a: $p")
                    assertEquals(1, s.failures.size, "$section #$j strict layer, expected $e actual $a: ${s.failures}")
                    checked++
                }
            }
            val (p, s) = run("runs", runRow(e), runRow(a))
            assertEquals(1, p.size, "runs expected $e actual $a: $p")
            assertEquals(1, s.failures.size, "runs strict layer, expected $e actual $a: ${s.failures}")
            checked++
        }
        assertEquals(75, checked)
    }

    /**
     * 계획 9.0절: 스텝 특징 #0, #1, #4, #9, #10, #11, #12, #14, #15, #16은 정확히 같다. expected.model_inputs의 features
     * `[120][17]`(원시 스텝 특징, drowsy/vectors.py:27, 34)도 같은 특징이라 각 행의 자리가 특징 번호다.
     */
    @Test
    fun exactStepFeaturesInModelInputs() {
        val values = (0 until 17).map { 0.25 + it }
        fun window(rows: List<List<Double>>) = """[{"t_ms": 1, "k": 119, "features": [""" +
            rows.joinToString(", ") { r -> r.joinToString(", ", "[", "]") { it.toString() } } + "]}]"
        val e = window(listOf(values, values))
        for (j in 0 until 17) {
            val changed = values.toMutableList()
            changed[j] = Math.nextUp(changed[j]) // 1 ulp: SPEC 16장(1e-5)은 통과
            val (p, s) = run("model_inputs", e, window(listOf(values, changed)))
            assertEquals(emptyList(), p, "feature #$j")
            if (j in VectorComparator.EXACT_STEP_FEATURES) {
                assertEquals(1, s.failures.size, "model_inputs feature #$j must be exact")
                assertTrue(s.failures[0].startsWith("expected.model_inputs[0].features[1][$j]: step feature #$j"), s.failures[0])
            } else {
                assertTrue(s.failures.isEmpty(), "model_inputs feature #$j is not exact")
                assertTrue(s.notes.isEmpty(), "1 ulp is below 1e-9")
            }
        }
    }

    @Test
    fun problemsStopAtTwenty() {
        val e = (0 until 30).map { 1L }
        val a = (0 until 30).map { 2L }
        val problems = ArrayList<String>()
        VectorComparator.compare("counters", "counters", e, a, emptyMap(), "expected.counters", problems)
        assertEquals(VectorComparator.MAX_PROBLEMS, problems.size)
        assertTrue(TestData.list(e).size == 30)
    }
}
