package hawkeye.drowsy.core.internal

/**
 * 보고용 셈(계획 9.1절 E3, E4의 "필드별 가장 큰 차이(절대값, 허용 오차 대비), 1e-9 안에 든 프레임 수").
 * 판정은 VectorComparator가 한다. 이것은 숫자만 모은다.
 */
internal class FieldStats(private val tolerance: (field: String) -> Double) {
    val maxAbs: MutableMap<String, Double> = java.util.TreeMap()
    var rows = 0
        private set
    var rowsWithin1e9 = 0
        private set

    /** 행(프레임) 하나: 기대 객체와 실제 객체의 실수 잎을 필드 이름으로 모은다. 목록 안의 실수는 그 필드 이름으로 센다. */
    fun row(expected: Map<*, *>, actual: Map<*, *>) {
        rows++
        var worst = 0.0
        for ((k, e) in expected) {
            val name = k as String
            worst = maxOf0(worst, leaf(name, e, actual[name]))
        }
        if (worst <= 1e-9) rowsWithin1e9++
    }

    private fun leaf(name: String, e: Any?, a: Any?): Double {
        if (e is List<*> && a is List<*> && e.size == a.size) {
            var w = 0.0
            for (i in e.indices) w = maxOf0(w, leaf(name, e[i], a[i]))
            return w
        }
        val ed = (e as? Double) ?: return 0.0
        val ad = when (a) {
            is Double -> a
            is Long -> a.toDouble()
            else -> return Double.POSITIVE_INFINITY
        }
        val d = Math.abs(ad - ed)
        val dd = if (d.isNaN()) Double.POSITIVE_INFINITY else d
        val old = maxAbs[name]
        if (old == null || dd > old) maxAbs[name] = dd
        return dd
    }

    private fun maxOf0(a: Double, b: Double): Double = if (b > a) b else a

    /** "field max_abs (ratio of tolerance)" 목록. */
    fun summary(): String = maxAbs.entries.joinToString(", ") { (k, v) ->
        val tol = tolerance(k)
        "$k ${fmt(v)} (${fmt(v / tol)} of $tol)"
    }

    private fun fmt(x: Double): String = String.format(java.util.Locale.ROOT, "%.3e", x)
}
