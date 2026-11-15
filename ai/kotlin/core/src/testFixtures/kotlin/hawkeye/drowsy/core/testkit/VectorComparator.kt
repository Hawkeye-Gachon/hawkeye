package hawkeye.drowsy.core.testkit

import hawkeye.drowsy.core.spec.PyRepr

/**
 * 벡터 비교기 (계획 9.0절). 두 층이다.
 *
 * 1. SPEC 16장 판정: drowsy/vectors.py:667-710 `_compare`를 그대로 옮겼다(같은 순서, 같은 형 규칙, 같은 메시지, 문제 20개
 *    한도). 다른 곳은 한 곳뿐이다: 실수는 `!(abs(a − e) <= abs + rel·abs(e))`이면 실패라서 **NaN과 무한대도 실패**다.
 *    Python은 실제 값이 NaN이면 비교가 False라 통과시킨다(vectors.py:709, 계획 9.0절). 기대값이 NaN이어도 Kotlin은 실패다.
 * 2. 엄격 층([strictLayer]): SPEC 16장의 허용 오차가 가릴 수 있는 이산 계산의 실수를 잡는다(계획 9.0절 표).
 *
 * 값은 JSON 나무(MiniJson의 형): null, Boolean, Long(Python int), Double(Python float), String, List, Map.
 * Python의 `bool`은 `int`의 하위형이다. 그래서 기대값이 정수일 때 실제 불리언은 따로 거부한다(vectors.py:697).
 */
object VectorComparator {

    /** vectors.py:664 `_MAX_PROBLEMS`. */
    const val MAX_PROBLEMS: Int = 20

    private val ANGLE_KEYS = setOf("yaw", "pitch", "roll")
    private val TIME_KEYS: Map<String, Set<String>> = mapOf(
        "runs" to setOf("start_ms", "end_ms", "duration_ms"),
        "events" to setOf("t_start_ms", "duration_ms"),
        "alerts" to setOf("t_start_ms", "duration_ms"),
    )

    /** 허용 오차를 얼마나 썼는지(항목별 가장 큰 |a − e| / 허용 오차). 허용 오차가 0인 비교는 넣지 않는다. */
    class ToleranceUse {
        val maxRatio: MutableMap<String, Double> = LinkedHashMap()
        internal fun note(section: String, diff: Double, tol: Double) {
            if (!(tol > 0.0)) return
            val r = diff / tol
            val old = maxRatio[section]
            if (old == null || r > old) maxRatio[section] = r
        }
    }

    /**
     * vectors.py:667-710 `_compare`. key는 가장 가까운 필드 이름이다. problems에 Python과 같은 문장으로 적는다.
     * tolerance의 값은 JSON 그대로(Long 또는 Double)다: 메시지의 `abs`, `rel` 표기가 Python과 같아야 한다.
     */
    fun compare(
        section: String,
        key: String,
        expected: Any?,
        actual: Any?,
        tolerance: Map<String, Any?>,
        path: String,
        problems: MutableList<String>,
        use: ToleranceUse? = null,
    ) {
        if (problems.size >= MAX_PROBLEMS) return
        when (expected) {
            is Map<*, *> -> {
                if (actual !is Map<*, *> || expected.keys != actual.keys) {
                    problems.add("$path: field names differ")
                    return
                }
                for (name in expected.keys) {
                    name as String
                    compare(section, name, expected[name], actual[name], tolerance, "$path.$name", problems, use)
                }
            }
            is List<*> -> {
                if (actual !is List<*> || expected.size != actual.size) {
                    val actualLen = if (actual is List<*>) actual.size.toString() else "no list"
                    problems.add("$path: expected ${expected.size} items, got $actualLen")
                    return
                }
                for (index in expected.indices) {
                    compare(section, key, expected[index], actual[index], tolerance, "$path[$index]", problems, use)
                }
            }
            null, is Boolean, is String -> {
                // Python: type(actual) is not type(expected) or actual != expected
                val sameType = when (expected) {
                    null -> actual == null
                    is Boolean -> actual is Boolean
                    else -> actual is String
                }
                if (!sameType || actual != expected) {
                    problems.add("$path: expected ${PyRepr.value(expected)}, got ${PyRepr.value(actual)}")
                }
            }
            is Long -> {
                var limit = 0L
                if (key in (TIME_KEYS[section] ?: emptySet())) {
                    limit = intOf(tolerance["time_abs_ms"] ?: throw IllegalArgumentException("$path: no time_abs_ms tolerance"))
                }
                // Python: isinstance(actual, bool) or not isinstance(actual, int) or abs(actual - expected) > limit
                val ok = actual is Long && absDiffAtMost(actual, expected, limit)
                if (ok && limit > 0 && use != null) use.note(section, Math.abs((actual as Long) - expected).toDouble(), limit.toDouble())
                if (!ok) {
                    problems.add("$path: expected ${PyRepr.value(expected)} (tolerance $limit), got ${PyRepr.value(actual)}")
                }
            }
            is Double -> {
                val absTol: Any?
                val relTol: Any?
                if (section == "signals" && key in ANGLE_KEYS) {
                    absTol = tolerance["angle_abs_deg"] ?: throw IllegalArgumentException("$path: no angle_abs_deg tolerance")
                    relTol = 0.0
                } else if (section == "events" || section == "alerts") {
                    absTol = tolerance["confidence_abs"] ?: throw IllegalArgumentException("$path: no confidence_abs tolerance")
                    relTol = 0.0
                } else {
                    absTol = tolerance["abs"] ?: 0.0
                    relTol = tolerance["rel"] ?: 0.0
                }
                val a: Double? = when (actual) {
                    is Double -> actual
                    is Long -> actual.toDouble()
                    else -> null // Boolean, null, String, List, Map
                }
                val tol = numOf(absTol) + numOf(relTol) * Math.abs(expected)
                // Kotlin: NaN과 무한대도 실패(계획 9.0절). Python은 `abs(a - e) > tol`이라 NaN을 통과시킨다.
                // 유한성을 먼저 따로 본다: e가 무한대이고 rel > 0이면 tol도 무한대라 `<=`만으로는 어떤 a든 통과한다.
                val ok = a != null && a.isFinite() && expected.isFinite() && Math.abs(a - expected) <= tol
                if (ok && use != null) use.note(section, Math.abs(a!! - expected), tol)
                if (!ok) {
                    problems.add(
                        "$path: expected ${PyRepr.value(expected)} (abs ${PyRepr.value(absTol)}, rel ${PyRepr.value(relTol)}), " +
                            "got ${PyRepr.value(actual)}",
                    )
                }
            }
            else -> throw IllegalArgumentException("$path: unsupported expected value ${expected.javaClass.name}")
        }
    }

    /** |a − e| <= limit, Python 정수처럼 넘침 없이. */
    private fun absDiffAtMost(a: Long, e: Long, limit: Long): Boolean {
        val d = a - e
        val overflow = ((a xor e) and (a xor d)) < 0
        if (overflow || d == Long.MIN_VALUE) return false // |a − e| >= 2^63 > limit
        return Math.abs(d) <= limit
    }

    /** Python `int(x)`: 정수는 그대로, 실수는 0 쪽으로 자른다. */
    private fun intOf(v: Any?): Long = when (v) {
        is Long -> v
        is Double -> v.toLong()
        else -> throw IllegalArgumentException("tolerance value $v is not a number")
    }

    private fun numOf(v: Any?): Double = when (v) {
        is Long -> v.toDouble()
        is Double -> v
        else -> throw IllegalArgumentException("tolerance value $v is not a number")
    }

    // -----------------------------------------------------------------------------------------------------------------
    // 엄격 층 (계획 9.0절)
    // -----------------------------------------------------------------------------------------------------------------

    /** 엄격 층에서 정확히 같아야 하는 스텝 특징의 번호: #0, #1, #4, #9, #10, #11, #12, #14, #15, #16 (정수 ms의 몫, 개수, ms). */
    val EXACT_STEP_FEATURES: Set<Int> = setOf(0, 1, 4, 9, 10, 11, 12, 14, 15, 16)

    /** "기대보다 큰 수치 차이"의 경계: abs 1e-9 + rel 1e-9 (계획 9.0절 표 마지막 줄). */
    const val NOTE_ABS: Double = 1e-9
    const val NOTE_REL: Double = 1e-9

    /** 엄격 층의 결과. failures는 실패, closureTiming은 그 가운데 걸침 판정(계획 9.2.5절)으로 볼 것, notes는 보고만. */
    class StrictResult {
        val failures: MutableList<String> = ArrayList()
        val closureTiming: MutableList<String> = ArrayList()
        val notes: MutableList<String> = ArrayList()
    }

    /**
     * 한 항목(section)의 기대 나무와 실제 나무에 엄격 층을 적용한다(계획 9.0절 표).
     * - events·alerts의 t_start_ms, duration_ms: source != closure이면 정확히 같다. closure이면 정확히 같아야 하되,
     *   다르면 걸침 판정으로 볼 목록(closureTiming)에도 적는다. 검토 목록에 사람이 넣지 않는 한 실패다.
     * - runs의 start_ms, end_ms, duration_ms: closure와 같다.
     * - confidence: source != model이면 정확히 같다(정수 ms의 몫). model이면 SPEC 16장(1e-4)만.
     * - steps의 values[#0, #1, #4, #9, #10, #11, #12, #14, #15, #16]: 정확히 같다. model_inputs의 features 각 행(17개)도 같은
     *   스텝 특징이라 같은 번호가 정확히 같다.
     * - 그 밖의 실수: abs 1e-9 + rel 1e-9를 넘으면 notes에 숫자와 함께 적는다(실패가 아니다, 소유자가 판단한다).
     * - 기대값이나 실제 값이 NaN 또는 무한대인 실수는 어디서나 실패다(계획 9.0절).
     * "정확히 같다"는 수로 같음(`==`, -0.0 == 0.0)이다. NaN은 같지 않다. 구조(키 집합, 길이)가 다른 곳은 SPEC 16장 층이
     * 이미 실패로 적으므로 여기서는 건너뛴다.
     */
    fun strictLayer(section: String, expected: Any?, actual: Any?, path: String, out: StrictResult) {
        walk(section, section, expected, actual, path, null, -1, out)
    }

    private fun walk(
        section: String,
        key: String,
        e: Any?,
        a: Any?,
        path: String,
        source: String?,
        valueIndex: Int,
        out: StrictResult,
    ) {
        when (e) {
            is Map<*, *> -> {
                if (a !is Map<*, *> || e.keys != a.keys) return
                val src = if ((section == "events" || section == "alerts") && e["source"] is String) e["source"] as String else source
                for (name in e.keys) {
                    name as String
                    walk(section, name, e[name], a[name], "$path.$name", src, -1, out)
                }
            }
            is List<*> -> {
                if (a !is List<*> || e.size != a.size) return
                for (i in e.indices) {
                    // 원소의 자리: values의 자리, features는 바깥 목록(행)과 안쪽 목록(특징 번호) 모두 자리를 넘기고
                    // 실수 칸은 가장 안쪽 목록의 자리를 받는다.
                    walk(section, key, e[i], a[i], "$path[$i]", source, if (key == "values" || key == "features") i else -1, out)
                }
            }
            is Long -> {
                if (a !is Long || a == e) return
                val eventTime = (section == "events" || section == "alerts") && (key == "t_start_ms" || key == "duration_ms")
                val runTime = section == "runs" && (key == "start_ms" || key == "end_ms" || key == "duration_ms")
                if (eventTime && source != "closure") {
                    out.failures.add("$path: expected exactly $e (source $source), got $a")
                } else if (eventTime || runTime) {
                    val msg = "$path: expected exactly $e (closure timing, plan 9.2.5 straddle review), got $a"
                    out.failures.add(msg)
                    out.closureTiming.add(msg)
                }
            }
            is Double -> {
                val av: Double = when (a) {
                    is Double -> a
                    is Long -> a.toDouble()
                    else -> return
                }
                val confidence = (section == "events" || section == "alerts") && key == "confidence"
                // 스텝 특징: steps의 values[17], 그리고 model_inputs의 features[120][17] 각 행(원시 스텝 특징, 행의 자리가 특징 번호)
                val stepFeature = (section == "steps" && key == "values") || (section == "model_inputs" && key == "features")
                val exactStep = stepFeature && valueIndex in EXACT_STEP_FEATURES
                if (!e.isFinite() || !av.isFinite()) {
                    // 계획 9.0절: NaN과 무한대는 실패다(정확히 같아야 하는 값에서 inf == inf여도, 허용 오차가 무한대가 되어도).
                    out.failures.add("$path: expected ${hex(e)}, got ${hex(av)} (NaN and infinity fail, plan 9.0)")
                } else if ((confidence && source != "model") || exactStep) {
                    if (!(av == e)) {
                        val what = if (confidence) "confidence (source $source)" else "step feature #$valueIndex"
                        out.failures.add("$path: $what expected exactly ${hex(e)}, got ${hex(av)}")
                    }
                } else if (!(Math.abs(av - e) <= NOTE_ABS + NOTE_REL * Math.abs(e))) {
                    out.notes.add("$path: difference ${Math.abs(av - e)} is above 1e-9 + 1e-9*|e| (expected ${hex(e)}, got ${hex(av)})")
                }
            }
            else -> Unit
        }
    }

    private fun hex(x: Double): String = java.lang.Double.toHexString(x)

    // -----------------------------------------------------------------------------------------------------------------
    // 벡터 하나
    // -----------------------------------------------------------------------------------------------------------------

    /** 벡터 하나의 비교 결과. passed = SPEC 16장 문제 0이고 엄격 층 실패 0. */
    class Result(
        val problems: List<String>,
        val strict: StrictResult,
        val toleranceUse: ToleranceUse,
    ) {
        val passed: Boolean get() = problems.isEmpty() && strict.failures.isEmpty()
    }

    /**
     * 벡터의 expected 전체를 실제 출력(같은 모양의 JSON 나무, 항목 → 값)과 비교한다. vectors.py:760-771의 순서를 따른다:
     * expected에 있는 항목마다, 실제 출력에 그 항목이 없으면 문제, 허용 오차 키가 표와 다르면 문제, 아니면 `_compare`.
     * 그다음 엄격 층. 0단계([VectorFile.stageZeroProblems])는 부르는 쪽이 먼저 한다.
     */
    fun compareVector(vector: VectorFile, actual: Map<String, Any?>): Result {
        val problems = ArrayList<String>()
        val strict = StrictResult()
        val use = ToleranceUse()
        if (vector.expected.isEmpty()) problems.add("expected is empty")
        for ((section, stored) in vector.expected) {
            if (!actual.containsKey(section)) {
                problems.add("expected.$section: not an output of bundle '${vector.bundle}'")
                continue
            }
            val tolerance = vector.tolerances[section] ?: emptyMap()
            val table = VectorFile.TOLERANCES[section]
            if (table != null && tolerance.keys != table.keys) {
                problems.add("tolerances.$section: expected the keys ${table.keys.sorted()}")
                continue
            }
            compare(section, section, stored, actual[section], tolerance, "expected.$section", problems, use)
            strictLayer(section, stored, actual[section], "expected.$section", strict)
        }
        return Result(problems, strict, use)
    }
}
