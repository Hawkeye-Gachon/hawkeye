package hawkeye.drowsy.core.diff

import org.junit.jupiter.api.Assumptions
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 리드 판정 10: 차등 비교 하나. kotlin/.diffdata/의 묶음 여덟(S-quick, S-odd, S-edge, F-stat, F-inj, F-cal, F-calib, G-sig)을
 * 세션 파일마다 파일이 적은 모든 모드(RAW, SIG, STATE)로 Kotlin에 다시 넣어 바꾸지 않은 Python 참조의 출력과 비교한다
 * (계획 9.2절의 규칙, 비교는 [DiffReplay]). S-full(12,000세션)은 돌리지 않는다.
 *
 * 알려진 차이는 판정 8(e)의 원인별로 세어 적고, 그 밖의 차이가 하나라도 있으면 실패한다. READY_corpora.txt가 없으면
 * (자료를 내보내지 않은 PC) 건너뛴다. 자료가 다른 참조에서 만들어졌으면 실패한다.
 *
 * RAW·SIG의 이산 출력이 다르면 계획 9.2.5절의 걸침 판정이 필요하다. 걸침 검토 파일은 없다(처음에는 비어 있다). 그래서
 * RAW·SIG의 이산 차이는 모두 실패로 센다.
 */
class DifferentialCorporaTest {

    private fun runCorpus(name: String, modes: List<String>) {
        val ready = DiffCorpus.readyCorpora()
        Assumptions.assumeTrue(ready != null, "no READY_corpora.txt: differential corpora not exported")
        val corpus = DiffCorpus.load(name, ready!!)
        val threads = minOf(4, Runtime.getRuntime().availableProcessors())
        val pool = Executors.newFixedThreadPool(threads)
        val total = DiffTally()
        val sessionModes = LinkedHashMap<String, Int>()
        val started = System.nanoTime()
        try {
            val futures = corpus.entries.map { e ->
                pool.submit(Callable {
                    val s = DiffCorpus.read(e)
                    val t = DiffTally()
                    t.sessions = 1
                    if (e.exception != (s.exception != null)) throw AssertionError("${e.path}: index.json and header disagree on the exception")
                    for (m in s.modes) DiffReplay.run(s, m, t)
                    s.modes.joinToString("+") to t
                })
            }
            for (f in futures) {
                val (m, t) = f.get()
                sessionModes[m] = (sessionModes[m] ?: 0) + 1
                total.merge(t)
            }
        } finally {
            pool.shutdownNow()
        }
        val seconds = (System.nanoTime() - started) / 1e9

        // 이 묶음에서 실제로 비교한 칸만 적는다(SIG의 신호 칸처럼 입력이라 비교하지 않는 칸은 빠진다).
        val tolFields = total.fields.values.filter { it.tol != null && it.compared > 0 }
        val maxLine = tolFields.joinToString("; ") { "${it.name} (${it.tol}) ${it.maxAbs}" }
        val nonZeroExact = total.fields.values.filter { it.tol == null && it.maxAbs > 0.0 }
        println(
            "D-$name: sessions ${total.sessions} (modes $sessionModes), mode runs ${total.modeRuns}, frames fed per mode ${total.framesFed}, " +
                "compared values ${total.compared}, differences ${total.differences}, known differences per cause ${knownPerCause(total)}, " +
                "sign-only zeros ${total.signOnlyZeros}, both NaN ${total.bothNaN}, notes ${total.notes}, threads $threads, " +
                "wall ${"%.1f".format(seconds)} s",
        )
        println("D-$name largest |kotlin - python| per tolerance field: ${if (tolFields.isEmpty()) "(none: STATE only, every value compared with ==)" else maxLine}")
        for (fd in tolFields.filter { it.maxAbs > 0.0 }) println("D-$name   ${fd.name}: ${fd.maxAbs} at ${fd.maxWhere}")
        println("D-$name exact fields with a value difference: ${nonZeroExact.map { "${it.name} ${it.maxAbs} at ${it.maxWhere}" }}")
        println("D-$name fields: " + total.fields.values.filter { it.compared > 0 }.joinToString("; ") { "${it.name} compared ${it.compared} bit-equal ${it.bitEqual} differences ${it.differences}" })
        for (k in total.knownSessions) println("D-$name known difference: $k")
        for (d in total.samples) println("D-$name DIFFERENCE: $d")

        assertEquals(corpus.entries.size, total.sessions)
        assertEquals(modes.toSet(), total.modeRuns.keys, "$name: modes run")
        for (m in modes) assertEquals(corpus.entries.size, total.modeRuns[m], "$name: every session runs in $m")
        assertTrue(total.compared > 0, "$name: nothing compared")
        val allowed = setOf(DiffReplay.CAUSE_ROUND_HALF_UP, DiffReplay.CAUSE_NAN_BASELINE)
        assertTrue(allowed.containsAll(total.known.keys), "$name: unknown cause ${total.known.keys}")
        // 판정 8(e): 참조가 예외를 낸 세션은 모두 알려진 원인으로 설명되어야 한다(조용히 건너뛰지 않는다).
        assertEquals(corpus.entries.count { it.exception }, total.known[DiffReplay.CAUSE_ROUND_HALF_UP] ?: 0, "$name: raising sessions explained")
        assertEquals(0L, total.differences, "$name: ${total.differences} differences outside the known causes: ${total.samples.take(10)}")
    }

    private fun knownPerCause(t: DiffTally): Map<String, Int> =
        linkedMapOf(
            DiffReplay.CAUSE_ROUND_HALF_UP to (t.known[DiffReplay.CAUSE_ROUND_HALF_UP] ?: 0),
            DiffReplay.CAUSE_NAN_BASELINE to (t.known[DiffReplay.CAUSE_NAN_BASELINE] ?: 0),
        )

    @Test
    fun sQuick() = runCorpus("S-quick", listOf("STATE"))

    @Test
    fun sOdd() = runCorpus("S-odd", listOf("STATE"))

    @Test
    fun sEdge() = runCorpus("S-edge", listOf("STATE"))

    @Test
    fun fStat() = runCorpus("F-stat", listOf("RAW", "SIG", "STATE"))

    @Test
    fun fInj() = runCorpus("F-inj", listOf("RAW", "SIG", "STATE"))

    @Test
    fun fCal() = runCorpus("F-cal", listOf("RAW", "SIG", "STATE"))

    @Test
    fun fCalib() = runCorpus("F-calib", listOf("RAW", "SIG", "STATE"))

    @Test
    fun gSig() = runCorpus("G-sig", listOf("SIG", "STATE"))
}
