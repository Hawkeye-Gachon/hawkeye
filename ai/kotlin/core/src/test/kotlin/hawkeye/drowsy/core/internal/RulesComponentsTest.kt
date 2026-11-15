package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.ClassProbs
import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.GazeZone
import hawkeye.drowsy.core.StepFeatures
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * E7 (계획 9.1절)의 부품 자료: 규칙 엔진 쪽 다섯 부품을 Python 호출 기록에 다시 넣는다(FORMATS.md 2.7절). 모두 수로 같아야
 * 한다(`==`; -0.0 == 0.0). 기대값은 모두 바꾸지 않은 참조 구현이 낸 것이다(kotlin/.diffdata/components/).
 *
 * - C-hysteresis: StepHysteresis(onset_steps, end_steps, refractory_ms)의 update/reset 흐름 → 호출마다 (active, start_ms).
 * - C-engine: RuleEngine(rules)의 on_step/update 흐름 → 호출마다 이벤트 목록, update 뒤의 gaze_zone.
 * - C-policy: AlertPolicy(rules).decide(events, speed, t) 흐름 → 호출마다 경고 목록.
 * - C-errors의 RuleEngine 흐름 6개: 증가하지 않는 t는 Python과 같은 메시지로 거부되고, 흐름은 이어지며 뒤의 출력도 맞다(거부가
 *   상태를 바꾸지 않았다는 증거). 그 흐름에는 이벤트가 없으므로, 이벤트를 내는 C-engine 흐름에 거부될 호출을 끼워 넣은
 *   시험도 둔다(ruleEngineRefusalsInsideEventStreams).
 * - C-func의 ratio(rules._ratio) 기록.
 * 거부된 호출 하나에서 멈추지 않고 모든 차이를 센다. 차이는 처음 20개까지 메시지에 적는다.
 */
class RulesComponentsTest {

    private class Diffs(val what: String) {
        val list = ArrayList<String>()
        var count = 0
        fun add(msg: String) {
            count++
            if (list.size < 20) list.add(msg)
        }
        fun assertNone() = assertEquals(0, count, "$what: $count differences, first: $list")
    }

    @Test
    fun hysteresisRecords() {
        val data = TestData.readyJson("components/C-hysteresis.json")
        val cases = TestData.list(data["cases"])
        val d = Diffs("C-hysteresis")
        var calls = 0
        var resets = 0
        var activeTrue = 0
        var noValue = 0
        var kSkips = 0
        for ((ci, c) in cases.withIndex()) {
            val case = TestData.obj(c)
            val args = TestData.list(case["args"])
            val h = StepHysteresis(
                onsetSteps = VectorFile.int(args[0], "args[0]"),
                endSteps = VectorFile.int(args[1], "args[1]"),
                refractoryMs = VectorFile.long(args[2], "args[2]"),
            )
            val callList = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            assertEquals(callList.size, outs.size, "case $ci: calls and out differ in length")
            var lastK: Long? = null
            for (i in callList.indices) {
                val call = TestData.list(callList[i])
                when (call[0]) {
                    "r" -> {
                        h.reset()
                        resets++
                        lastK = null
                    }
                    "u" -> {
                        val k = VectorFile.long(call[1], "k")
                        if (lastK != null && k != lastK + 1) kSkips++
                        lastK = k
                        if (call[3] != true) noValue++
                        h.update(
                            k = VectorFile.int(call[1], "k"),
                            tEndMs = VectorFile.long(call[2], "t_end_ms"),
                            hasValue = VectorFile.bool(call[3], "has_value"),
                            onsetHit = VectorFile.bool(call[4], "onset_hit"),
                            endHit = VectorFile.bool(call[5], "end_hit"),
                        )
                    }
                    else -> throw AssertionError("case $ci call $i: unknown call ${call[0]}")
                }
                calls++
                val o = TestData.list(outs[i])
                val eActive = VectorFile.bool(o[0], "active")
                val eStart = VectorFile.long(o[1], "start_ms")
                if (h.active) activeTrue++
                if (h.active != eActive || h.startMs != eStart) {
                    d.add("case $ci call $i ${call}: expected ($eActive, $eStart), got (${h.active}, ${h.startMs})")
                }
            }
        }
        println(
            "RulesComponentsTest.hysteresisRecords: ${cases.size} cases, $calls calls ($resets resets, $noValue steps without a value, " +
                "$kSkips skipped k), active after $activeTrue calls, differences ${d.count}",
        )
        d.assertNone()
    }

    @Test
    fun engineRecords() {
        val data = TestData.readyJson("components/C-engine.json")
        val cases = TestData.list(data["cases"])
        val d = Diffs("C-engine")
        var stepCalls = 0
        var frameCalls = 0
        var events = 0
        val bySource = java.util.TreeMap<String, Int>()
        val byType = java.util.TreeMap<String, Int>()
        val zones = java.util.TreeMap<String, Int>()
        for ((ci, c) in cases.withIndex()) {
            val case = TestData.obj(c)
            val rules = SpecLoader.rulesFromTree(case["rules"])
            bySource.merge(rules.drowsiness.source.wire, 1, Int::plus)
            val engine = RuleEngine(rules)
            val callList = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            assertEquals(callList.size, outs.size, "case $ci: calls and out differ in length")
            for (i in callList.indices) {
                val call = TestData.list(callList[i])
                val o = TestData.list(outs[i])
                val got: List<DriverStateEvent>
                var zone: GazeZone? = null
                when (call[0]) {
                    "s" -> {
                        got = engine.onStep(step(call[1], "case $ci call $i step"), probs(call[2], "case $ci call $i probs"))
                        stepCalls++
                    }
                    "f" -> {
                        got = engine.update(JsonTrees.frameState(call[1], "case $ci call $i state"))
                        zone = engine.gazeZone
                        frameCalls++
                    }
                    else -> throw AssertionError("case $ci call $i: unknown call ${call[0]}")
                }
                val expected = TestData.list(o[0]).map { TestData.event(TestData.obj(it)) }
                events += expected.size
                for (e in expected) byType.merge(e.type.wire, 1, Int::plus)
                compareEvents("case $ci call $i", expected, got, d)
                val eZone = o[1] as String?
                if (zone != null) zones.merge(zone.wire, 1, Int::plus)
                if ((zone?.wire) != eZone) d.add("case $ci call $i: gaze_zone expected $eZone, got ${zone?.wire}")
            }
        }
        println(
            "RulesComponentsTest.engineRecords: ${cases.size} cases (drowsiness.source $bySource), $stepCalls on_step calls, " +
                "$frameCalls update calls, $events events $byType, gaze_zone after update $zones, differences ${d.count}",
        )
        d.assertNone()
    }

    @Test
    fun policyRecords() {
        val data = TestData.readyJson("components/C-policy.json")
        val cases = TestData.list(data["cases"])
        val d = Diffs("C-policy")
        var calls = 0
        var eventsIn = 0
        var alerts = 0
        var reissued = 0
        var nullSpeed = 0
        var negativeSpeed = 0
        val bySource = java.util.TreeMap<String, Int>()
        for ((ci, c) in cases.withIndex()) {
            val case = TestData.obj(c)
            val src = (case["source"] as String).substringBefore(" ")
            bySource.merge(src, 1, Int::plus)
            val policy = AlertPolicy(SpecLoader.rulesFromTree(case["rules"]))
            val callList = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            assertEquals(callList.size, outs.size, "case $ci: calls and out differ in length")
            for (i in callList.indices) {
                val call = TestData.list(callList[i])
                val evs = TestData.list(call[0]).map { TestData.event(TestData.obj(it)) }
                val speed = call[1]?.let { TestData.num(it) }
                if (speed == null) nullSpeed++ else if (speed < 0.0) negativeSpeed++
                val t = VectorFile.long(call[2], "t_ms")
                eventsIn += evs.size
                val got = policy.decide(evs, speed, t)
                val expected = TestData.list(outs[i]).map { TestData.event(TestData.obj(it)) }
                alerts += expected.size
                for (e in expected) if (evs.none { it == e }) reissued++ // 이 호출의 이벤트가 아닌 경고 = 미뤄 둔 것이나 되풀이의 사본
                compareEvents("case $ci call $i", expected, got, d)
                calls++
            }
        }
        println(
            "RulesComponentsTest.policyRecords: ${cases.size} cases $bySource, $calls decide calls ($nullSpeed with speed null, " +
                "$negativeSpeed with a negative speed), $eventsIn events in, $alerts alerts out ($reissued of them reissued copies), " +
                "differences ${d.count}",
        )
        d.assertNone()
    }

    /** C-errors의 RuleEngine 흐름: 증가하지 않는 t는 같은 메시지로 거부되고(IllegalArgumentException), 흐름은 이어지며 뒤의 출력도 맞다. */
    @Test
    fun ruleEngineStreamsOfCErrors() {
        val data = TestData.readyJson("components/C-errors.json")
        val d = Diffs("C-errors RuleEngine")
        var streams = 0
        var calls = 0
        var refused = 0
        var events = 0
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            if (case["component"] != "RuleEngine") continue
            streams++
            val name = case["case"] as String
            val engine = RuleEngine(SpecLoader.rulesFromTree(TestData.obj(case["setup"])["rules"]))
            val callList = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            assertEquals(callList.size, outs.size, "$name: calls and out differ in length")
            for (i in callList.indices) {
                calls++
                val o = TestData.obj(outs[i])
                val state = JsonTrees.frameState(callList[i], "$name call $i")
                val err = o["error"]
                if (err != null) {
                    val e = TestData.obj(err)
                    assertEquals("ValueError", e["type"])
                    try {
                        engine.update(state)
                        d.add("$name call $i: expected ${e["type"]} '${e["message"]}', got no error")
                    } catch (x: IllegalArgumentException) {
                        refused++
                        if (x.message != e["message"]) d.add("$name call $i: message '${x.message}' != '${e["message"]}'")
                    }
                    continue
                }
                val got = try {
                    engine.update(state)
                } catch (x: IllegalArgumentException) {
                    d.add("$name call $i: unexpected ${x.message}")
                    continue
                }
                val expected = TestData.list(o["ok"]).map { TestData.event(TestData.obj(it)) }
                events += expected.size
                compareEvents("$name call $i", expected, got, d)
            }
        }
        println(
            "RulesComponentsTest.ruleEngineStreamsOfCErrors: $streams streams, $calls calls, $refused refused with the Python message, " +
                "$events events after or between refusals compared, differences ${d.count}",
        )
        assertEquals(6, streams)
        assertTrue(refused > 0)
        d.assertNone()
    }

    /**
     * 계획 9.2.6절 C-errors "거부된 호출 뒤에도 흐름을 이어 가고 뒤의 출력이 맞아야 한다". C-errors의 RuleEngine 흐름 6개는
     * 이벤트를 내지 않아서(위 시험) 이것을 약하게만 보인다. 그래서 이벤트를 내는 C-engine 흐름 120개에 거부될 update를 끼워
     * 넣고, 끼운 호출이 C-errors의 Python 메시지로 거부되며 gaze_zone을 바꾸지 않고, 기록된 모든 출력(이벤트, gaze_zone)이
     * 그대로 나오는지 본다.
     *
     * 끼우는 규칙(Python 쪽과 같다): 이미 한 "f" 호출 수를 nF, 그 마지막 t_ms를 lastT라 할 때, "f" 호출 앞에서 nF >= 1이고
     * nF % 17 == 3이면 그 호출의 상태를 t_ms = lastT - D[j % 3]로 바꿔 먼저 넣는다(D = 0, 1, 500; j는 파일 전체에서 끼운 수).
     * 기대값은 C-engine.json(끼우지 않은 Python 실행)과 C-errors.json의 메시지다. 같은 규칙으로 바꾸지 않은 Python RuleEngine에
     * 끼워 넣어도 기록된 출력이 모두 그대로 나오고 메시지가 같다는 것은 Python으로 확인했다(작업 폴더
     * f_s4/engine_refusal_injection_python.py와 .out: 120경우, 9889번 거부, 이벤트 2400, 차이 0).
     */
    @Test
    fun ruleEngineRefusalsInsideEventStreams() {
        var message: String? = null
        for (c in TestData.list(TestData.readyJson("components/C-errors.json")["cases"])) {
            val case = TestData.obj(c)
            if (case["component"] != "RuleEngine") continue
            val err = TestData.list(case["out"]).map { TestData.obj(it)["error"] }.firstOrNull { it != null } ?: continue
            assertEquals("ValueError", TestData.obj(err)["type"])
            message = TestData.obj(err)["message"] as String
            break
        }
        assertTrue(message != null, "C-errors has no RuleEngine refusal")
        val cases = TestData.list(TestData.readyJson("components/C-engine.json")["cases"])
        val d = Diffs("C-engine with injected refusals")
        val gaps = longArrayOf(0L, 1L, 500L)
        var j = 0
        var calls = 0
        var refused = 0
        var events = 0
        var eventsAfter = 0
        var casesWithInjection = 0
        for ((ci, c) in cases.withIndex()) {
            val case = TestData.obj(c)
            val engine = RuleEngine(SpecLoader.rulesFromTree(case["rules"]))
            val callList = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            assertEquals(callList.size, outs.size, "case $ci: calls and out differ in length")
            var nF = 0
            var lastT = 0L
            var zoneBefore: GazeZone? = null
            var injectedHere = 0
            for (i in callList.indices) {
                val call = TestData.list(callList[i])
                if (call[0] == "f" && nF >= 1 && nF % 17 == 3) {
                    val bad = JsonTrees.frameState(call[1], "case $ci call $i state").copy(tMs = lastT - gaps[j % 3])
                    j++
                    injectedHere++
                    calls++
                    try {
                        engine.update(bad)
                        d.add("case $ci before call $i: t_ms ${bad.tMs} after $lastT was not refused")
                    } catch (x: IllegalArgumentException) {
                        refused++
                        if (x.message != message) d.add("case $ci before call $i: message '${x.message}' != '$message'")
                    }
                    if (engine.gazeZone != zoneBefore) d.add("case $ci before call $i: gaze_zone ${engine.gazeZone} after the refusal, $zoneBefore before")
                }
                calls++
                val o = TestData.list(outs[i])
                val got: List<DriverStateEvent>
                var zone: GazeZone? = null
                when (call[0]) {
                    "s" -> got = engine.onStep(step(call[1], "case $ci call $i step"), probs(call[2], "case $ci call $i probs"))
                    "f" -> {
                        val state = JsonTrees.frameState(call[1], "case $ci call $i state")
                        got = engine.update(state)
                        zone = engine.gazeZone
                        nF++
                        lastT = state.tMs
                        zoneBefore = zone
                    }
                    else -> throw AssertionError("case $ci call $i: unknown call ${call[0]}")
                }
                val expected = TestData.list(o[0]).map { TestData.event(TestData.obj(it)) }
                events += expected.size
                if (injectedHere > 0) eventsAfter += expected.size
                compareEvents("case $ci call $i", expected, got, d)
                val eZone = o[1] as String?
                if ((zone?.wire) != eZone) d.add("case $ci call $i: gaze_zone expected $eZone, got ${zone?.wire}")
            }
            if (injectedHere > 0) casesWithInjection++
        }
        println(
            "RulesComponentsTest.ruleEngineRefusalsInsideEventStreams: ${cases.size} C-engine cases, $casesWithInjection with injected refusals, " +
                "$calls calls, $refused refused with the C-errors message, $events events compared, " +
                "$eventsAfter of them after the first refusal of their case, differences ${d.count}",
        )
        assertEquals(cases.size, casesWithInjection)
        assertEquals(j, refused)
        assertTrue(eventsAfter > 0)
        d.assertNone()
    }

    /**
     * C-errors의 41경우가 모두 어느 시험에 닿는지 센다(E7). 이 클래스는 RuleEngine 흐름과, 다른 시험에 없던 함수 경우 둘
     * (nearest_rank_quantile, median의 빈 입력)을 직접 본다. 나머지는 SignalsTest(신호 함수 7), BaselineTest·ClosureRunDetectorTest·
     * StepAggregatorTest(흐름 각 6), PipelineVectorsTest·PipelineConstructionTest(생성자 5), ModelCallErrorsTest(모델 3)가 본다.
     */
    @Test
    fun everyCaseOfCErrorsHasAnOwnerAndTheEmptyInputCasesAreRefused() {
        val data = TestData.readyJson("components/C-errors.json")
        val byComponent = java.util.TreeMap<String, Int>()
        var emptyCases = 0
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            val component = case["component"] as String
            val setup = TestData.obj(case["setup"])
            val key = when {
                component == "DriverPipeline" && setup.containsKey("model_returns") -> "DriverPipeline model (ModelCallErrorsTest)"
                component == "DriverPipeline" -> "DriverPipeline constructor (PipelineVectorsTest, PipelineConstructionTest)"
                component == "function" && setup["fn"] in setOf("nearest_rank_quantile", "median") -> "function calibrate (here)"
                component == "function" -> "function signals (SignalsTest)"
                else -> "$component stream"
            }
            byComponent.merge(key, 1, Int::plus)
            if (key != "function calibrate (here)") continue
            val err = TestData.obj(TestData.obj(case["out"])["error"])
            assertEquals("ValueError", err["type"])
            val args = TestData.list(setup["args"])
            val e = kotlin.test.assertFailsWith<IllegalArgumentException>(case["case"] as String) {
                when (setup["fn"]) {
                    "nearest_rank_quantile" -> nearestRankQuantile(JsonTrees.doubles(args[0], "values"), VectorFile.num(args[1], "q"))
                    else -> median(JsonTrees.doubles(args[0], "values"))
                }
            }
            assertEquals(err["message"], e.message, case["case"] as String)
            emptyCases++
        }
        println("RulesComponentsTest.everyCaseOfCErrorsHasAnOwnerAndTheEmptyInputCasesAreRefused: ${byComponent.values.sum()} cases $byComponent; empty-input cases refused with the Python message $emptyCases")
        assertEquals(41, byComponent.values.sum())
        assertEquals(2, emptyCases)
    }

    /** C-func의 ratio(rules._ratio, fn 이름 `ratio`): 분모가 0 이하이면 0.0, 아니면 float(n) / float(d). 정확히 같아야 한다(compare exact). */
    @Test
    fun ratioRecords() {
        val data = TestData.readyJson("components/C-func.json")
        var n = 0
        var nonPositive = 0
        val d = Diffs("C-func ratio")
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            if (case["fn"] != "ratio") continue
            n++
            assertEquals("exact", case["compare"])
            val args = TestData.list(case["args"])
            val num = VectorFile.long(args[0], "num")
            val den = VectorFile.long(args[1], "den")
            if (den <= 0L) nonPositive++
            val expected = TestData.num(case["result"])
            val got = ratio(num, den)
            if (!(got == expected) || got.toRawBits() != expected.toRawBits()) {
                d.add("ratio($num, $den): expected ${java.lang.Double.toHexString(expected)}, got ${java.lang.Double.toHexString(got)}")
            }
        }
        println("RulesComponentsTest.ratioRecords: $n ratio cases ($nonPositive with a denominator <= 0), bit-identical to Python in ${n - d.count}, differences ${d.count}")
        assertEquals(112, n)
        d.assertNone()
    }

    // ----- 도우미 -----

    /** 이벤트 목록을 필드마다 비교한다. 정수와 열거형은 정확히, confidence는 수로 같음(`==`). 순서도 같아야 한다. */
    private fun compareEvents(where: String, expected: List<DriverStateEvent>, got: List<DriverStateEvent>, d: Diffs) {
        if (expected.size != got.size) {
            d.add("$where: expected ${expected.size} events $expected, got ${got.size} $got")
            return
        }
        for (j in expected.indices) {
            val e = expected[j]
            val a = got[j]
            val same = e.type == a.type && e.severity == a.severity && e.tStartMs == a.tStartMs && e.durationMs == a.durationMs &&
                e.confidence == a.confidence && e.source == a.source && e.phase == a.phase && e.provisional == a.provisional
            if (!same) d.add("$where[$j]: expected $e, got $a")
        }
    }

    private fun step(v: Any?, where: String): StepFeatures {
        val m = VectorFile.obj(v, where)
        return StepFeatures(
            k = VectorFile.int(m["k"], "$where.k"),
            tEndMs = VectorFile.long(m["t_end_ms"], "$where.t_end_ms"),
            fullLookback = VectorFile.bool(m["full_lookback"], "$where.full_lookback"),
            values = JsonTrees.doubles(m["values"], "$where.values"),
        )
    }

    private fun probs(v: Any?, where: String): ClassProbs? {
        if (v == null) return null
        val p = JsonTrees.doubles(v, where)
        require(p.size == 3) { "$where must hold 3 values" }
        return ClassProbs(alert = p[0], lowVigilant = p[1], drowsy = p[2])
    }
}
