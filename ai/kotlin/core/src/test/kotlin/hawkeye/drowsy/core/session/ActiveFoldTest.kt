package hawkeye.drowsy.core.session

import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.Phase
import hawkeye.drowsy.core.Source
import hawkeye.drowsy.core.testdata.TestData
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * 진행 중 이벤트 접기(계획 6.2절, K10 (가))의 시험.
 *
 * 기대값: `kotlin/.diffdata/active_expected/<벡터>.json`. Python 내보내기(`kotlin/difftools/small.py` `active`)가
 * 벡터의 `expected.events`(바꾸지 않은 참조 구현의 출력)를 같은 접기로 걸어 쓴 것이고, 내보낼 때
 * `demo.event_spans`와 맞춰 보았다(FORMATS.md 2.5절, 4.1절). 이 자료가 없으면 그 시험은 건너뛰고, 벡터의 이벤트만으로
 * 볼 수 있는 불변식 시험은 언제나 돈다.
 */
class ActiveFoldTest {

    private class Group(val tMs: Long, val events: List<DriverStateEvent>)

    /** 벡터의 expected.events를 t_ms로 묶는다(파일 순서). 같은 프레임의 이벤트는 이어져 있어야 한다. */
    private fun groups(vector: Map<String, Any?>): List<Group>? {
        val expected = TestData.obj(vector["expected"])
        if (!expected.containsKey("events")) return null
        val out = ArrayList<Group>()
        val seen = HashSet<Long>()
        var cur: MutableList<DriverStateEvent>? = null
        var curT = 0L
        for (ev in TestData.list(expected["events"])) {
            val m = TestData.obj(ev)
            val t = m["t_ms"] as Long
            if (cur == null || t != curT) {
                assertTrue(seen.add(t), "events of frame t_ms=$t are not contiguous")
                if (cur != null) out.add(Group(curT, cur))
                cur = ArrayList()
                curT = t
            }
            cur.add(TestData.event(m))
        }
        if (cur != null) out.add(Group(curT, cur))
        return out
    }

    private fun vectorsWithEvents(): List<Pair<File, Map<String, Any?>>> =
        TestData.vectorFiles().map { it to TestData.obj(TestData.parseFile(it)) }
            .filter { TestData.obj(it.second["expected"]).containsKey("events") }

    private fun ev(type: EventType, phase: Phase, tStart: Long, source: Source = Source.CLOSURE, prov: Boolean = false) =
        DriverStateEvent(
            type = type, severity = 3, tStartMs = tStart, durationMs = 0L, confidence = 1.0, source = source,
            phase = phase, provisional = prov,
        )

    // ----- Python 기대값과 비교 -----

    @Test
    fun foldMatchesPythonActiveExpectedForEveryVectorWithEvents() {
        TestData.requireReadySmall()
        val index = TestData.readyJson("active_expected/index.json")
        val listed = TestData.list(index["vectors"]).map { TestData.obj(it)["vector"] as String }.sorted()
        val withEvents = vectorsWithEvents()
        assertEquals(withEvents.map { it.first.name.removeSuffix(".json") }.sorted(), listed,
            "active_expected/index.json lists exactly the vectors that have expected.events")

        var framesCompared = 0
        var entriesCompared = 0
        for ((file, vector) in withEvents) {
            val name = file.name.removeSuffix(".json")
            val exp = TestData.readyJson("active_expected/$name.json")
            assertEquals(name, exp["vector"])
            assertEquals(exp["vector_sha256"], TestData.sha256Hex(file.readBytes()), "$name: vector file sha256")
            val priority = TestData.list(exp["alert_priority"]).map { EventType.fromWire(it as String) }
            val fold = ActiveFold(priority)
            val gs = groups(vector)!!
            val frames = TestData.list(exp["frames"])
            assertEquals(frames.size, gs.size, "$name: number of frames with events")
            for ((i, g) in gs.withIndex()) {
                val f = TestData.obj(frames[i])
                assertEquals(f["t_ms"], g.tMs, "$name: frame $i t_ms")
                val got = fold.apply(g.events, g.tMs)
                val want = TestData.list(f["active"]).map { TestData.obj(it) }
                assertEquals(want.size, got.size, "$name: frame t_ms=${g.tMs} active size; got ${got.map { it.type }}")
                for ((j, w) in want.withIndex()) {
                    val a = got[j]
                    val where = "$name: frame t_ms=${g.tMs} active[$j]"
                    assertEquals(w["type"], a.type.wire, "$where.type")
                    assertEquals(w["severity"], a.severity.toLong(), "$where.severity")
                    assertEquals(w["t_start_ms"], a.tStartMs, "$where.t_start_ms")
                    assertEquals(w["source"], a.source.wire, "$where.source")
                    assertEquals(w["provisional"], a.provisional, "$where.provisional")
                    assertEquals(w["onset_frame_t_ms"], a.onsetFrameTMs, "$where.onset_frame_t_ms")
                    assertEquals(6, w.size, "$where: entry keys ${w.keys}")
                    entriesCompared++
                }
                framesCompared++
            }
        }
        println("ActiveFoldTest: ${withEvents.size} vectors, $framesCompared frames with events, $entriesCompared active entries compared")
    }

    // ----- 벡터 이벤트만으로 보는 불변식(기대값이 필요 없다) -----

    @Test
    fun invariantsHoldOnEveryVectorEventStream() {
        val priority = TestData.shippedAlertPriority()
        var vectors = 0
        var frames = 0
        for ((file, vector) in vectorsWithEvents()) {
            // 지금 벡터의 config_overrides는 alert_priority를 바꾸지 않는다. 바꾸는 벡터가 생기면 이 시험을 고쳐야 한다.
            val overrides = TestData.obj(vector["config_overrides"] ?: emptyMap<String, Any?>())
            assertFalse(overrides.keys.any { it.startsWith("alert_priority") }, "${file.name} overrides alert_priority")
            val fold = ActiveFold(priority)
            val latestOnset = HashMap<EventType, Pair<DriverStateEvent, Long>>()
            for (g in groups(vector)!!) {
                val before = fold.active
                val got = fold.apply(g.events, g.tMs)
                // SLEEP이 있으면 MICROSLEEP이 없다.
                if (got.any { it.type == EventType.SLEEP }) {
                    assertFalse(got.any { it.type == EventType.MICROSLEEP }, "${file.name} t=${g.tMs}: sleep and microsleep")
                }
                // 우선순위 순서이고 종류마다 하나.
                val idx = got.map { priority.indexOf(it.type) }
                assertEquals(idx.sorted(), idx, "${file.name} t=${g.tMs}: not in alert_priority order")
                assertEquals(idx.toSet().size, idx.size)
                // 목록의 항목은 그 종류의 가장 최근 onset이다.
                for (e in g.events) if (e.phase == Phase.ONSET) latestOnset[e.type] = e to g.tMs
                for (a in got) {
                    val (o, t) = latestOnset[a.type] ?: fail("${file.name}: ${a.type} active without onset")
                    assertEquals(ActiveEvent(o.type, o.severity, o.tStartMs, o.source, o.provisional, t), a)
                }
                assertTrue(before !== got || g.events.isEmpty())
                frames++
            }
            // 처음부터 다시 접으면 같은 결과(이벤트 목록 전체로 다시 만든 값 = 차례로 만든 값).
            val again = ActiveFold(priority)
            var last: List<ActiveEvent> = emptyList()
            for (g in groups(vector)!!) last = again.apply(g.events, g.tMs)
            assertEquals(fold.active, last)
            vectors++
        }
        assertTrue(vectors > 0)
        println("ActiveFoldTest invariants: $vectors vectors, $frames frames with events")
    }

    // ----- 접기 규칙 자체 (계획 6.2절의 의사 코드) -----

    @Test
    fun sleepOnsetClearsMicrosleep() {
        val fold = ActiveFold(TestData.shippedAlertPriority())
        fold.apply(listOf(ev(EventType.MICROSLEEP, Phase.ONSET, 100)), 1100)
        assertEquals(listOf(EventType.MICROSLEEP), fold.active.map { it.type })
        fold.apply(listOf(ev(EventType.SLEEP, Phase.ONSET, 100)), 3100)
        assertEquals(listOf(EventType.SLEEP), fold.active.map { it.type })
        assertEquals(3100L, fold.active[0].onsetFrameTMs)
        fold.apply(listOf(ev(EventType.SLEEP, Phase.END, 100)), 5000)
        assertTrue(fold.active.isEmpty())
    }

    @Test
    fun sleepEndAlsoClearsMicrosleep() {
        val fold = ActiveFold(TestData.shippedAlertPriority())
        fold.apply(listOf(ev(EventType.MICROSLEEP, Phase.ONSET, 100)), 1100)
        fold.apply(listOf(ev(EventType.SLEEP, Phase.END, 50)), 1200)
        assertTrue(fold.active.isEmpty())
    }

    @Test
    fun endRemovesOnlyItsType() {
        val fold = ActiveFold(TestData.shippedAlertPriority())
        fold.apply(
            listOf(ev(EventType.CALIBRATING, Phase.ONSET, 1, Source.FACE), ev(EventType.VATS, Phase.ONSET, 2, Source.HEAD_POSE)),
            10,
        )
        fold.apply(listOf(ev(EventType.CALIBRATING, Phase.END, 1, Source.FACE)), 20)
        assertEquals(listOf(EventType.VATS), fold.active.map { it.type })
        fold.apply(listOf(ev(EventType.DROWSY, Phase.END, 5, Source.PERCLOS)), 30)
        assertEquals(listOf(EventType.VATS), fold.active.map { it.type })
    }

    @Test
    fun secondOnsetReplacesEntryAndOrderFollowsPriority() {
        val priority = TestData.shippedAlertPriority()
        val fold = ActiveFold(priority)
        fold.apply(listOf(ev(EventType.CALIBRATING, Phase.ONSET, 1, Source.FACE, prov = true)), 1)
        fold.apply(listOf(ev(EventType.SLEEP, Phase.ONSET, 5)), 6)
        fold.apply(listOf(ev(EventType.CALIBRATING, Phase.ONSET, 7, Source.FACE)), 8)
        val types = fold.active.map { it.type }
        assertEquals(priority.filter { it == EventType.SLEEP || it == EventType.CALIBRATING }, types)
        val cal = fold.active.first { it.type == EventType.CALIBRATING }
        assertEquals(7L, cal.tStartMs)
        assertFalse(cal.provisional)
        assertEquals(8L, cal.onsetFrameTMs)
    }

    @Test
    fun noEventsKeepsTheSameList() {
        val fold = ActiveFold(TestData.shippedAlertPriority())
        val a = fold.apply(listOf(ev(EventType.VATS, Phase.ONSET, 2, Source.HEAD_POSE)), 10)
        assertSame(a, fold.apply(emptyList(), 20))
        fold.reset()
        assertTrue(fold.active.isEmpty())
    }

    @Test
    fun activeListIsReadOnly() {
        val fold = ActiveFold(TestData.shippedAlertPriority())
        val a = fold.apply(listOf(ev(EventType.VATS, Phase.ONSET, 2, Source.HEAD_POSE)), 10)
        @Suppress("UNCHECKED_CAST")
        val asMutable = a as MutableList<ActiveEvent>
        assertFailsWith<UnsupportedOperationException> { asMutable.clear() }
    }

    @Test
    fun priorityMustListEveryTypeOnce() {
        val p = TestData.shippedAlertPriority()
        assertFailsWith<IllegalArgumentException> { ActiveFold(p.drop(1)) }
        assertFailsWith<IllegalArgumentException> { ActiveFold(p.drop(1) + p[2]) }
        ActiveFold(p)
    }
}
