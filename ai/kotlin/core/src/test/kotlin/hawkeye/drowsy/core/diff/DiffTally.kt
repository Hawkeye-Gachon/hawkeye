package hawkeye.drowsy.core.diff

/**
 * 차등 비교의 수 세기 (리드 판정 10: 세션, 프레임, 비교한 값, 차이, 원인별 알려진 차이, 허용 오차 칸마다 가장 큰
 * |kotlin - python|). 세션 하나가 자기 Tally를 채우고 묶음 끝에서 합친다. 스레드에 안전하지 않다(합치기는 한 스레드에서).
 *
 * 비교 규칙(계획 9.2.1절, 9.2.4절):
 * - STATE 모드: 모든 수는 수로 같다(`==`, -0.0 == 0.0; 부호만 다른 0은 센다). 둘 다 NaN이면 같다고 보고 센다
 *   (FORMATS.md 2.7절의 "both NaN"; S-odd 입력에 NaN이 있다).
 * - RAW·SIG 모드: 칸마다 9.2.4절의 좁은 허용 오차(Tol). Tol이 없는 칸은 STATE와 같이 수로 같다.
 * - 정수, 참거짓, 열거형, 개수는 언제나 정확히 같다.
 */
internal class DiffTally {
    /** 허용 오차: |k - p| <= abs + rel * |p|. */
    class Tol(val abs: Double, val rel: Double) {
        override fun toString(): String = if (rel == 0.0) "abs $abs" else if (abs == 0.0) "rel $rel" else "abs $abs + rel $rel"
    }

    /** 칸 하나의 기록. maxAbs는 두 값이 모두 유한할 때의 |k - p| 가운데 가장 큰 것. */
    class Field(val name: String, val tol: Tol?) {
        var compared = 0L
        var differences = 0L
        var maxAbs = 0.0
        var maxWhere = ""
        var bitEqual = 0L
    }

    val fields = LinkedHashMap<String, Field>()
    var sessions = 0
    val modeRuns = LinkedHashMap<String, Int>()
    val framesFed = LinkedHashMap<String, Long>()
    var compared = 0L
    var differences = 0L
    var signOnlyZeros = 0L
    var bothNaN = 0L
    val samples = ArrayList<String>()
    val known = LinkedHashMap<String, Int>()
    val knownSessions = ArrayList<String>()
    val notes = LinkedHashMap<String, Long>()

    fun field(name: String, tol: Tol?): Field = fields.getOrPut(name) { Field(name, tol) }

    fun difference(text: String) {
        differences++
        if (samples.size < 30) samples.add(text)
    }

    fun note(key: String, add: Long = 1) {
        notes[key] = (notes[key] ?: 0L) + add
    }

    fun knownDifference(cause: String, sid: String, detail: String) {
        known[cause] = (known[cause] ?: 0) + 1
        knownSessions.add("$sid: $detail")
    }

    fun merge(o: DiffTally) {
        sessions += o.sessions
        for ((k, v) in o.modeRuns) modeRuns[k] = (modeRuns[k] ?: 0) + v
        for ((k, v) in o.framesFed) framesFed[k] = (framesFed[k] ?: 0L) + v
        compared += o.compared
        differences += o.differences
        signOnlyZeros += o.signOnlyZeros
        bothNaN += o.bothNaN
        for (s in o.samples) if (samples.size < 30) samples.add(s)
        for ((k, v) in o.known) known[k] = (known[k] ?: 0) + v
        knownSessions.addAll(o.knownSessions)
        for ((k, v) in o.notes) notes[k] = (notes[k] ?: 0L) + v
        for ((name, f) in o.fields) {
            val mine = field(name, f.tol)
            mine.compared += f.compared
            mine.differences += f.differences
            mine.bitEqual += f.bitEqual
            if (f.maxAbs > mine.maxAbs) {
                mine.maxAbs = f.maxAbs
                mine.maxWhere = f.maxWhere
            }
        }
    }
}

/**
 * 세션 하나, 모드 하나의 비교기. where는 지금 비교하는 자리(세션, 모드, 행, t_ms)다. 차이는 Tally에 적고, 구조가 어긋난
 * 차이(개수, 받아들임, 기준선 유무)는 [Structural]로 그 세션의 비교를 멈춘다(그 뒤의 행은 어긋난 짝끼리 비교하게 된다).
 */
internal class DiffCmp(private val t: DiffTally, private val mode: String) {
    var where: String = ""
    private val exactAll = mode == "STATE"

    class Structural(message: String) : RuntimeException(message)

    fun f(name: String, tol: DiffTally.Tol? = null): DiffTally.Field = t.field("$mode.$name", if (exactAll) null else tol)

    private fun hex(x: Double): String = java.lang.Double.toHexString(x)

    fun num(fd: DiffTally.Field, k: Double, p: Double) {
        t.compared++
        fd.compared++
        if (java.lang.Double.doubleToRawLongBits(k) == java.lang.Double.doubleToRawLongBits(p)) {
            fd.bitEqual++
            if (k.isNaN()) t.bothNaN++
            return
        }
        if (k.isNaN() && p.isNaN()) {
            t.bothNaN++
            return
        }
        val d = Math.abs(k - p)
        if (!d.isNaN() && !d.isInfinite() && d > fd.maxAbs) {
            fd.maxAbs = d
            fd.maxWhere = "$where kotlin ${hex(k)} python ${hex(p)}"
        }
        if (k == p) {
            // 값은 같고 비트만 다르다: 부호만 다른 0.
            t.signOnlyZeros++
            return
        }
        val tol = fd.tol
        val ok = tol != null && d <= tol.abs + tol.rel * Math.abs(p)
        if (!ok) {
            fd.differences++
            t.difference("$where ${fd.name}: kotlin ${hex(k)} ($k) python ${hex(p)} ($p) |diff| $d rule ${tol ?: "=="}")
        }
    }

    fun exact(fd: DiffTally.Field, k: Long, p: Long) {
        t.compared++
        fd.compared++
        if (k == p) {
            fd.bitEqual++
            return
        }
        fd.differences++
        t.difference("$where ${fd.name}: kotlin $k python $p")
    }

    fun exact(fd: DiffTally.Field, k: Boolean, p: Boolean) = exact(fd, if (k) 1L else 0L, if (p) 1L else 0L)

    /** 개수나 짝이 어긋남: 차이로 적고 이 세션의 비교를 멈춘다. */
    fun structural(fd: DiffTally.Field, k: Any?, p: Any?): Nothing {
        t.compared++
        fd.compared++
        fd.differences++
        val text = "$where ${fd.name}: kotlin $k python $p (structural; the rest of this session is not compared)"
        t.difference(text)
        throw Structural(text)
    }

    fun count(fd: DiffTally.Field, k: Int, p: Int) {
        if (k != p) structural(fd, k, p)
        t.compared++
        fd.compared++
        fd.bitEqual++
    }
}
