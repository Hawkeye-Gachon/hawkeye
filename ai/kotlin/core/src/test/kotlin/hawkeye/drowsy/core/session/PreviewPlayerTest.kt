package hawkeye.drowsy.core.session

import hawkeye.drowsy.core.ClassProbs
import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.DrowsyInternalApi
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.spec.SpecException
import hawkeye.drowsy.core.testdata.TestData
import java.io.ByteArrayInputStream
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * 전달 0 재생기(PreviewPlayer)의 시험.
 *
 * 내용의 기대값: `kotlin/.diffdata/replay0/<name>.json`. Python 내보내기(`kotlin/difftools/small.py` `replay0`)가 합성 각본
 * 세션을 바꾸지 않은 참조 `DriverPipeline`에 돌려 프레임마다 쓴 출력, gaze_zone, baseline, 계수, 속도, 접기 결과다
 * (FORMATS.md 2.6절). 이 자료가 없으면 그 시험은 건너뛴다.
 *
 * 시각 규칙(언제 기록을 내는지, 처음부터 다시 재생)은 계획 5.8절 끝의 규칙이다. 그 시험은 손으로 만든 작은 재생 자료로도 돈다.
 *
 * 수는 비트로 비교한다(`toRawBits`). 재생기는 파일의 글자를 그대로 double로 읽을 뿐이므로 비트가 같아야 한다.
 */
@OptIn(DrowsyInternalApi::class)
class PreviewPlayerTest {

    private val replayNames = listOf("replay0_main", "replay0_gap")

    private fun replayFile(name: String) = File(TestData.diffData, "replay0/$name.json")

    private fun bitsEq(expected: Any?, actual: Double, where: String) {
        val e = TestData.num(expected)
        assertEquals(e.toRawBits(), actual.toRawBits(), "$where: expected $e, got $actual")
    }

    private fun assertEvent(w: Map<String, Any?>, a: DriverStateEvent, where: String) {
        assertEquals(8, w.size, "$where keys ${w.keys}")
        assertEquals(w["type"], a.type.wire, "$where.type")
        assertEquals(w["severity"], a.severity.toLong(), "$where.severity")
        assertEquals(w["t_start_ms"], a.tStartMs, "$where.t_start_ms")
        assertEquals(w["duration_ms"], a.durationMs, "$where.duration_ms")
        bitsEq(w["confidence"], a.confidence, "$where.confidence")
        assertEquals(w["source"], a.source.wire, "$where.source")
        assertEquals(w["phase"], a.phase.wire, "$where.phase")
        assertEquals(w["provisional"], a.provisional, "$where.provisional")
    }

    private fun assertProbs(w: Any?, a: ClassProbs?, where: String) {
        if (w == null) {
            assertNull(a, where)
            return
        }
        assertNotNull(a, where)
        val l = TestData.list(w)
        assertEquals(3, l.size, where)
        bitsEq(l[0], a.alert, "$where[0]")
        bitsEq(l[1], a.lowVigilant, "$where[1]")
        bitsEq(l[2], a.drowsy, "$where[2]")
    }

    /** DriverState 하나를 재생 자료의 프레임 하나와 필드마다 비교한다. */
    private fun assertStateMatches(f: Map<String, Any?>, s: DriverState, firstTMs: Long, where: String) {
        val out = TestData.obj(f["output"])
        assertEquals(f["t_ms"], s.tMs, "$where.t_ms")
        assertEquals((f["t_ms"] as Long) - firstTMs, s.sessionMs, "$where sessionMs")
        val st = TestData.obj(out["state"])
        assertEquals(11, st.size)
        val o = s.output.state
        assertEquals(st["t_ms"], o.tMs)
        assertEquals(st["face"], o.face, "$where.face")
        bitsEq(st["closure"], o.closure, "$where.closure")
        assertEquals(st["closure_valid"], o.closureValid, "$where.closure_valid")
        assertEquals(st["provisional"], o.provisional, "$where.provisional")
        bitsEq(st["yaw_rel"], o.yawRel, "$where.yaw_rel")
        bitsEq(st["pitch_rel"], o.pitchRel, "$where.pitch_rel")
        assertEquals(st["pose_valid"], o.poseValid, "$where.pose_valid")
        bitsEq(st["gaze_h_rel"], o.gazeHRel, "$where.gaze_h_rel")
        bitsEq(st["gaze_v_rel"], o.gazeVRel, "$where.gaze_v_rel")
        bitsEq(st["jaw_rel"], o.jawRel, "$where.jaw_rel")

        val runs = TestData.list(out["runs"])
        assertEquals(runs.size, s.output.runs.size, "$where.runs")
        for ((j, rv) in runs.withIndex()) {
            val r = TestData.obj(rv)
            val a = s.output.runs[j]
            assertEquals(r["start_ms"], a.startMs)
            assertEquals(r["end_ms"], a.endMs)
            assertEquals(r["duration_ms"], a.durationMs)
            bitsEq(r["amplitude"], a.amplitude, "$where.runs[$j].amplitude")
            bitsEq(r["peak_closing_velocity"], a.peakClosingVelocity, "$where.runs[$j].pcv")
            bitsEq(r["peak_opening_velocity"], a.peakOpeningVelocity, "$where.runs[$j].pov")
            assertEquals(r["velocity_valid"], a.velocityValid)
            assertEquals(r["truncated"], a.truncated)
        }
        val steps = TestData.list(out["steps"])
        assertEquals(steps.size, s.output.steps.size, "$where.steps")
        for ((j, sv) in steps.withIndex()) {
            val e = TestData.obj(sv)
            val a = s.output.steps[j]
            assertEquals(e["k"], a.k.toLong())
            assertEquals(e["t_end_ms"], a.tEndMs)
            assertEquals(e["full_lookback"], a.fullLookback)
            val vals = TestData.list(e["values"])
            assertEquals(17, vals.size)
            assertEquals(17, a.values.size)
            for (q in 0 until 17) bitsEq(vals[q], a.values[q], "$where.steps[$j].values[$q]")
        }
        val sp = TestData.list(out["step_probs"])
        assertEquals(sp.size, s.output.stepProbs.size)
        for ((j, p) in sp.withIndex()) assertProbs(p, s.output.stepProbs[j], "$where.step_probs[$j]")
        assertProbs(out["probs"], s.output.probs, "$where.probs")

        val events = TestData.list(out["events"])
        assertEquals(events.size, s.events.size, "$where.events")
        for ((j, ev) in events.withIndex()) assertEvent(TestData.obj(ev), s.events[j], "$where.events[$j]")
        val alerts = TestData.list(out["alerts"])
        assertEquals(alerts.size, s.alerts.size, "$where.alerts")
        for ((j, ev) in alerts.withIndex()) assertEvent(TestData.obj(ev), s.alerts[j], "$where.alerts[$j]")
        if (alerts.isEmpty()) assertNull(s.primaryAlert) else assertSame(s.alerts[0], s.primaryAlert)

        assertEquals(f["gaze_zone"], s.gazeZone.wire, "$where.gaze_zone")
        val bl = f["baseline"]
        if (bl == null) {
            assertNull(s.baseline, "$where.baseline")
            assertEquals(CalibrationStatus.CALIBRATING, s.calibration)
        } else {
            val b = TestData.obj(bl)
            val a = assertNotNull(s.baseline, "$where.baseline")
            bitsEq(b["e_l"], a.eL, "$where.baseline.e_l")
            bitsEq(b["e_r"], a.eR, "$where.baseline.e_r")
            bitsEq(b["b_l"], a.bL, "$where.baseline.b_l")
            bitsEq(b["b_r"], a.bR, "$where.baseline.b_r")
            val r0 = TestData.list(b["r0"])
            assertEquals(9, r0.size)
            for (q in 0 until 9) bitsEq(r0[q], a.r0[q], "$where.baseline.r0[$q]")
            bitsEq(b["jaw0"], a.jaw0, "$where.baseline.jaw0")
            bitsEq(b["gaze_h0"], a.gazeH0, "$where.baseline.gaze_h0")
            bitsEq(b["gaze_v0"], a.gazeV0, "$where.baseline.gaze_v0")
            assertEquals(b["t_complete_ms"], a.tCompleteMs)
            assertEquals(b["status"], a.status.wire)
            assertEquals(a.status.name, s.calibration.name, "$where calibration")
        }
        val c = TestData.obj(f["counters"])
        assertEquals(c["frames_accepted"], s.counters.framesAccepted)
        assertEquals(c["dropped_frames"], s.counters.droppedFrames)
        assertEquals(c["steps_emitted"], s.counters.stepsEmitted)
        assertEquals(c["model_calls"], s.counters.modelCalls)
        assertEquals(c["nonfinite_windows"], s.counters.nonfiniteWindows)
        val speed = f["speed_kmh"]
        if (speed == null) assertNull(s.speedKmh, "$where.speed_kmh") else bitsEq(speed, s.speedKmh!!, "$where.speed_kmh")

        // active: Python이 같은 접기로 쓴 값(FORMATS.md 4.1절)과 비교한다.
        val act = TestData.list(f["active"])
        assertEquals(act.size, s.active.size, "$where.active ${s.active.map { it.type }}")
        for ((j, av) in act.withIndex()) {
            val w = TestData.obj(av)
            val a = s.active[j]
            assertEquals(w["type"], a.type.wire, "$where.active[$j].type")
            assertEquals(w["severity"], a.severity.toLong())
            assertEquals(w["t_start_ms"], a.tStartMs)
            assertEquals(w["source"], a.source.wire)
            assertEquals(w["provisional"], a.provisional)
            assertEquals(w["onset_frame_t_ms"], a.onsetFrameTMs)
        }
        for (t in EventType.entries) assertEquals(s.active.any { it.type == t }, s.isActive(t))
    }

    private class Loaded(val data: Map<String, Any?>, val player: PreviewPlayer, val frames: List<Map<String, Any?>>) {
        val withOutput: List<Map<String, Any?>> = frames.filter { it["output"] != null }
        val firstTMs: Long = withOutput[0]["t_ms"] as Long
        fun sessionMs(i: Int): Long = (withOutput[i]["t_ms"] as Long) - firstTMs
    }

    private fun load(name: String): Loaded {
        val data = TestData.readyJson("replay0/$name.json")
        val player = replayFile(name).inputStream().use { PreviewPlayer.read(it, TestData.shippedAlertPriority()) }
        return Loaded(data, player, TestData.list(data["frames"]).map { TestData.obj(it) })
    }

    @Test
    fun replayDataIsFromTheShippedRulesAndSpec() {
        TestData.requireReadySmall()
        val rulesHash = TestData.sha256Hex(File(TestData.export, "rules.json").readBytes())
        val spec = TestData.obj(TestData.parseFile(File(TestData.export, "feature_spec_v0.json")))
        for (name in replayNames) {
            val l = load(name)
            val session = TestData.obj(l.data["session"])
            assertEquals("hawkeye-replay0", l.data["format"])
            assertEquals(1L, l.data["version"])
            assertEquals(name, session["name"])
            assertEquals(name, l.player.name)
            assertEquals(rulesHash, session["rules_hash"], "$name: rules_hash = sha256(export/rules.json)")
            assertEquals(rulesHash, l.player.rulesHash)
            assertEquals(spec["pipeline_hash"], session["pipeline_hash"], "$name: pipeline_hash")
            assertEquals(l.withOutput.size, l.player.recordCount)
        }
    }

    @Test
    fun everyRecordReplaysFieldByFieldWhenFramesArriveOnTime() {
        TestData.requireReadySmall()
        var total = 0
        val typesSeen = HashSet<EventType>()
        for (name in replayNames) {
            val l = load(name)
            val origin = 5_000_000L // 모니터 시계의 원점은 아무 값이나 된다
            var prevActive: List<ActiveEvent>? = null
            for (i in l.withOutput.indices) {
                val s = l.player.onAcceptedFrame(origin + l.sessionMs(i))
                    ?: throw AssertionError("$name: no state for record $i at its own session time")
                assertStateMatches(l.withOutput[i], s, l.firstTMs, "$name[$i]")
                // 불변식(계획 6.2절): 보정하는 세션에서 CALIBRATING == provisional, SLEEP이면 MICROSLEEP 없음,
                // active는 이벤트가 있는 프레임에서만 바뀐다.
                assertEquals(s.calibration == CalibrationStatus.CALIBRATING, s.output.state.provisional, "$name[$i]")
                if (s.isActive(EventType.SLEEP)) assertFalse(s.isActive(EventType.MICROSLEEP))
                if (prevActive != null && s.events.isEmpty()) assertSame(prevActive, s.active, "$name[$i]")
                prevActive = s.active
                for (e in s.events) typesSeen.add(e.type)
                total++
            }
            assertEquals(1L, l.player.loopsStarted)
        }
        assertEquals(EventType.entries.toSet(), typesSeen, "every event type occurs in the replay data")
        println("PreviewPlayerTest: $total records compared field by field")
    }

    @Test
    fun recordsAreNeverSkippedAndAtMostOnePerFrame() {
        TestData.requireReadySmall()
        val l = load("replay0_main")
        // 빠른 프레임(33 ms): 각 기록은 그 세션 시간 이상인 첫 프레임에서 나온다.
        var t = 0L
        var next = 0
        var calls = 0
        while (next < l.withOutput.size) {
            val s = l.player.onAcceptedFrame(t)
            calls++
            val due = t >= l.sessionMs(next)
            if (due) {
                assertNotNull(s, "record $next due at $t")
                assertEquals(l.withOutput[next]["t_ms"], s.tMs)
                next++
            } else {
                assertNull(s, "record $next not due at $t")
            }
            t += 33
        }
        // 느린 프레임(500 ms): 기록이 밀리지만 하나씩 차례로 나오고 건너뛰지 않는다.
        val slow = load("replay0_main")
        var tt = 0L
        for (i in 0 until 50) {
            val s = slow.player.onAcceptedFrame(tt)
            assertNotNull(s)
            assertEquals(slow.withOutput[i]["t_ms"], s.tMs, "slow frame $i")
            tt += 500
        }
        println("PreviewPlayerTest: 33 ms cadence, ${l.withOutput.size} records in $calls frames")
    }

    @Test
    fun replayStartsOverAfterTheLastRecordLikeANewDrive() {
        TestData.requireReadySmall()
        val l = load("replay0_gap")
        val first = ArrayList<DriverState>()
        for (i in l.withOutput.indices) first.add(l.player.onAcceptedFrame(100L + l.sessionMs(i))!!)
        assertTrue(first.last().active.isNotEmpty() || first.any { it.active.isNotEmpty() })
        // 다음 프레임은 새 차례의 원점이 되고 첫 기록을 바로 낸다.
        val restartAt = 999_999L
        val again = l.player.onAcceptedFrame(restartAt)!!
        assertEquals(2L, l.player.loopsStarted)
        assertEquals(0L, again.sessionMs)
        assertStateMatches(l.withOutput[0], again, l.firstTMs, "loop2[0]")
        for (i in 1 until l.withOutput.size) {
            assertNull(l.player.onAcceptedFrame(restartAt + l.sessionMs(i) - 1))
            val s = l.player.onAcceptedFrame(restartAt + l.sessionMs(i))!!
            assertStateMatches(l.withOutput[i], s, l.firstTMs, "loop2[$i]")
            assertEquals(first[i].active, s.active, "loop2[$i] active")
        }
    }

    // ----- 손으로 만든 작은 재생 자료: 시각 규칙과 형식 검사 (Python 기대값이 필요 없는 것) -----

    private fun stateJson(t: Long, provisional: Boolean) =
        """{"t_ms":$t,"face":true,"closure":0.25,"closure_valid":true,"provisional":$provisional,"yaw_rel":-0.0,""" +
            """"pitch_rel":1.5,"pose_valid":false,"gaze_h_rel":0.0,"gaze_v_rel":0,"jaw_rel":0.0}"""

    private fun eventJson(type: String, phase: String, tStart: Long) =
        """{"type":"$type","severity":3,"t_start_ms":$tStart,"duration_ms":0,"confidence":1.0,"source":"closure",""" +
            """"phase":"$phase","provisional":false}"""

    private fun frameJson(t: Long, events: String = "", output: Boolean = true, speed: String = "null") =
        """{"t_ms":$t,"speed_kmh":$speed,"output":""" +
            (if (!output) "null" else """{"state":${stateJson(t, true)},"runs":[],"steps":[],"step_probs":[],""" +
                """"probs":null,"events":[$events],"alerts":[]}""") +
            ""","gaze_zone":"unknown","baseline":null,"counters":{"frames_accepted":1,"dropped_frames":0,""" +
            """"steps_emitted":0,"model_calls":0,"nonfinite_windows":0},"active":[]}"""

    private fun replayJson(frames: List<String>, format: String = "hawkeye-replay0", version: String = "1") =
        """{"format":"$format","version":$version,"session":{"name":"hand","rules_hash":"x"},""" +
            """"frames":[${frames.joinToString(",")}]}"""

    private fun player(json: String) =
        PreviewPlayer.read(ByteArrayInputStream(json.toByteArray(Charsets.UTF_8)), TestData.shippedAlertPriority())

    @Test
    fun handMadeReplayTimingDroppedFramesAndLoop() {
        val p = player(
            replayJson(
                listOf(
                    frameJson(1000, eventJson("microsleep", "onset", 900), speed = "20.0"),
                    frameJson(1050, output = false),
                    frameJson(1100),
                    frameJson(1300, eventJson("sleep", "onset", 900)),
                ),
            ),
        )
        assertEquals(3, p.recordCount) // 버린 프레임의 기록은 빠진다
        val s0 = p.onAcceptedFrame(7)!!
        assertEquals(0L, s0.sessionMs)
        assertEquals(1000L, s0.tMs)
        assertEquals(20.0, s0.speedKmh)
        assertEquals(listOf(EventType.MICROSLEEP), s0.active.map { it.type })
        assertEquals(0.0.toRawBits(), s0.output.state.gazeVRel.toRawBits()) // 정수 글자 0 → 0.0
        assertEquals((-0.0).toRawBits(), s0.output.state.yawRel.toRawBits())
        assertNull(p.onAcceptedFrame(7 + 99))
        val s1 = p.onAcceptedFrame(7 + 100)!!
        assertEquals(100L, s1.sessionMs)
        assertSame(s0.active, s1.active)
        assertNull(s1.speedKmh)
        // 늦게 와도 하나만 낸다.
        val s2 = p.onAcceptedFrame(7 + 10_000)!!
        assertEquals(300L, s2.sessionMs)
        assertEquals(listOf(EventType.SLEEP), s2.active.map { it.type })
        assertEquals(1L, p.loopsStarted)
        // 끝나면 새 차례: 접기도 비우고 다시 접는다.
        val r0 = p.onAcceptedFrame(50_000)!!
        assertEquals(2L, p.loopsStarted)
        assertEquals(0L, r0.sessionMs)
        assertEquals(listOf(EventType.MICROSLEEP), r0.active.map { it.type })
        assertNull(p.onAcceptedFrame(50_050))
    }

    @Test
    fun handMadeReplayRejectsOtherFormats() {
        val ok = listOf(frameJson(1000))
        player(replayJson(ok))
        assertFailsWith<IllegalArgumentException> { player(replayJson(ok, format = "hawkeye-replay1")) }
        assertFailsWith<IllegalArgumentException> { player(replayJson(ok, version = "2")) }
        assertFailsWith<IllegalArgumentException> { player(replayJson(ok, version = "1.0")) }
        assertFailsWith<IllegalArgumentException> { player(replayJson(emptyList())) }
        assertFailsWith<IllegalArgumentException> { player(replayJson(listOf(frameJson(1000, output = false)))) }
        // 시각이 늘지 않음
        assertFailsWith<IllegalArgumentException> { player(replayJson(listOf(frameJson(1000), frameJson(1000)))) }
        // 상태 객체에 모르는 키, 모르는 이벤트 종류, 출력과 프레임의 시각이 다름
        assertFailsWith<IllegalArgumentException> {
            player(replayJson(listOf(frameJson(1000).replace("\"jaw_rel\":0.0", "\"jaw_rel\":0.0,\"x\":1"))))
        }
        assertFailsWith<IllegalArgumentException> { player(replayJson(listOf(frameJson(1000, eventJson("nap", "onset", 1))))) }
        assertFailsWith<IllegalArgumentException> {
            player(replayJson(listOf(frameJson(1000).replaceFirst("{\"t_ms\":1000", "{\"t_ms\":999"))))
        }
        // JSON 문법 오류는 SpecException(IllegalArgumentException을 잇는다)
        assertFailsWith<SpecException> { player(replayJson(ok) + "x") }
    }
}
