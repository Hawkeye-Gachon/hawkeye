package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.ClosureRun
import hawkeye.drowsy.core.spec.RunDetectorSpec
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorComparator
import hawkeye.drowsy.core.testkit.VectorFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * E5 (계획 9.1절)의 검출기만 넣는 경로와 C-detector. 파이프라인 경로는 hawkeye.drowsy.core.PipelineVectorsTest에 있다.
 *
 * - runs 묶음 15개: tests/vectors/README.md "runs"대로 expected.closure의 (t_ms, closure, closure_valid)를
 *   ClosureRunDetector(pipeline.run_detector, 진입 0.5 / 이탈 0.3)에 차례로 넣어 expected.runs와 비교한다. SPEC 16장 판정과
 *   엄격 층, 그리고 이 경로에서는 시각 칸이 모두 정확히 같아야 한다(계획 9.1절 E5).
 * - C-detector (FORMATS.md 2.7절): 400경우의 update·reset 흐름을 새 검출기에 다시 넣고 호출마다 구간, ongoing_ms,
 *   run_start_ms, state를 비교한다. 값은 모두 수로 같아야 한다(차이 0).
 * - C-errors의 ClosureRunDetector 흐름 6개: 증가하지 않는 t는 Python과 같은 메시지로 거부되고 흐름은 이어지며 뒤의 출력도
 *   맞다(거부가 상태를 바꾸지 않았다는 증거, 계획 9.2.6절). E7의 자료지만 이 단계의 부품이라 여기서 다시 넣는다.
 * 기대값은 모두 바꾸지 않은 Python 참조에서 왔다(벡터 파일, 내보낸 부품 기록).
 */
class ClosureRunDetectorTest {

    private val spec = JsonTrees.shippedSpec.pipeline

    @Test
    fun runsBundleDetectorOnly() {
        val vectors = TestData.vectorFiles().map { it to VectorFile.read(it.readBytes()) }.filter { it.second.bundle == "runs" }
        assertEquals(15, vectors.size, "runs vectors")
        val failed = ArrayList<String>()
        val notes = ArrayList<String>()
        val times = TimeCells()
        var runCount = 0
        var frames = 0
        for ((file, v) in vectors) {
            val stage0 = v.stageZeroProblems(file.nameWithoutExtension, JsonTrees.shippedSpec.pipelineHash, JsonTrees.shippedRulesSha256)
            if (stage0.isNotEmpty()) {
                failed.add("${v.name}: constants differ: $stage0")
                continue
            }
            val detector = ClosureRunDetector(spec.runDetector)
            val runs = ArrayList<Any?>()
            for ((i, item) in VectorFile.list(v.expected["closure"], "expected.closure").withIndex()) {
                val st = VectorFile.obj(item, "expected.closure[$i]")
                val t = VectorFile.long(st["t_ms"], "expected.closure[$i].t_ms")
                val run = detector.update(t, VectorFile.num(st["closure"], "closure"), VectorFile.bool(st["closure_valid"], "closure_valid"))
                if (run != null) runs.add(JsonTrees.run(t, run))
                frames++
            }
            val tol = v.tolerances["runs"] ?: emptyMap()
            val problems = ArrayList<String>()
            val strict = VectorComparator.StrictResult()
            VectorComparator.compare("runs", "runs", v.expected["runs"], runs, tol, "expected.runs", problems)
            VectorComparator.strictLayer("runs", v.expected["runs"], runs, "expected.runs", strict)
            times.add(v.expected["runs"], runs)
            runCount += runs.size
            if (problems.isNotEmpty() || strict.failures.isNotEmpty()) failed.add("${v.name}: $problems ${strict.failures}")
            notes.addAll(strict.notes.map { "${v.name}: $it" })
        }
        println(
            "ClosureRunDetectorTest.runsBundleDetectorOnly: ${vectors.size - failed.size}/${vectors.size} runs vectors pass " +
                "(detector only, SPEC 16 and strict layer), frames $frames, runs $runCount",
        )
        println("ClosureRunDetectorTest.runsBundleDetectorOnly: ${times.summary()}")
        println("ClosureRunDetectorTest.runsBundleDetectorOnly: strict-layer notes (above 1e-9, not failures): ${notes.size} ${notes.take(20)}")
        assertEquals(emptyList(), failed)
        assertEquals(times.cells, times.exact, "every time cell exactly equal on the detector-only path")
    }

    @Test
    fun componentDetectorRecords() {
        val data = TestData.readyJson("components/C-detector.json")
        assertEquals("ClosureRunDetector", data["component"])
        val cases = TestData.list(data["cases"])
        val bad = ArrayList<String>()
        var calls = 0
        var resets = 0
        var runs = 0
        var truncated = 0
        var timeCells = 0
        val cfgs = HashSet<RunDetectorSpec>()
        for ((ci, c) in cases.withIndex()) {
            val case = TestData.obj(c)
            val cfg = TestData.obj(case["cfg"])
            // components.py:297: ClosureRunDetector(RunDetectorSpec(**cfg)), 검사 없이 만든 설정이다.
            val settings = RunDetectorSpec(
                enterThr = VectorFile.num(cfg["enter_thr"], "enter_thr"),
                exitThr = VectorFile.num(cfg["exit_thr"], "exit_thr"),
                minDurationMs = VectorFile.long(cfg["min_duration_ms"], "min_duration_ms"),
                maxDtMs = VectorFile.long(cfg["max_dt_ms"], "max_dt_ms"),
                holdMs = VectorFile.long(cfg["hold_ms"], "hold_ms"),
                velocityLookbackSamples = VectorFile.int(cfg["velocity_lookback_samples"], "velocity_lookback_samples"),
            )
            cfgs.add(settings)
            val detector = ClosureRunDetector(settings)
            val inputs = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            assertEquals(inputs.size, outs.size, "case $ci")
            for (k in inputs.indices) {
                val call = TestData.list(inputs[k])
                calls++
                var got: ClosureRun? = null
                if (call[0] == "r") {
                    resets++
                    detector.reset()
                } else {
                    got = detector.update(
                        VectorFile.long(call[1], "t"), VectorFile.num(call[2], "c"), VectorFile.bool(call[3], "valid"),
                    )
                }
                val o = TestData.list(outs[k])
                val want = JsonTrees.runOrNull(o[0], "case $ci out $k")
                if (want != null) {
                    runs++
                    timeCells += 3
                    if (want.truncated) truncated++
                }
                val where = "case $ci call $k"
                bad.addAll(runProblems(where, want, got))
                if (o[1] != detector.ongoingMs()) bad.add("$where ongoing_ms: expected ${o[1]}, got ${detector.ongoingMs()}")
                if (o[2] != detector.runStartMs()) bad.add("$where run_start_ms: expected ${o[2]}, got ${detector.runStartMs()}")
                if (o[3] != stateName(detector.state)) bad.add("$where state: expected ${o[3]}, got ${detector.state}")
            }
        }
        println(
            "ClosureRunDetectorTest.componentDetectorRecords: C-detector ${cases.size} cases (${cfgs.size} settings), $calls calls " +
                "($resets resets), $runs runs ($truncated truncated), time cells $timeCells, differences ${bad.size}",
        )
        assertEquals(emptyList(), bad.take(20))
        assertEquals(0, bad.size)
    }

    @Test
    fun componentErrorStreams() {
        val data = TestData.readyJson("components/C-errors.json")
        var streams = 0
        var refused = 0
        var calls = 0
        val bad = ArrayList<String>()
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            if (case["component"] != "ClosureRunDetector") continue
            streams++
            val name = case["case"] as String
            val pipeline = SpecLoader.pipelineSpecFromTree(TestData.obj(case["setup"])["pipeline"])
            val detector = ClosureRunDetector(pipeline.runDetector)
            val inputs = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            for (k in inputs.indices) {
                calls++
                val call = TestData.list(inputs[k])
                val o = TestData.obj(outs[k])
                val result = runCatching {
                    detector.update(VectorFile.long(call[0], "t"), VectorFile.num(call[1], "c"), VectorFile.bool(call[2], "valid"))
                }
                if (o.containsKey("error")) {
                    refused++
                    val err = TestData.obj(o["error"])
                    val e = result.exceptionOrNull()
                    if (e !is IllegalArgumentException || err["type"] != "ValueError" || err["message"] != e.message) {
                        bad.add("$name call $k: expected $err, got ${result.getOrNull() ?: e}")
                    }
                    continue
                }
                if (result.isFailure) {
                    bad.add("$name call $k: unexpected ${result.exceptionOrNull()}")
                    continue
                }
                val ok = TestData.list(o["ok"])
                val where = "$name call $k"
                bad.addAll(runProblems(where, JsonTrees.runOrNull(ok[0], where), result.getOrNull()))
                if (ok[1] != detector.ongoingMs()) bad.add("$where ongoing_ms: expected ${ok[1]}, got ${detector.ongoingMs()}")
                if (ok[2] != detector.runStartMs()) bad.add("$where run_start_ms: expected ${ok[2]}, got ${detector.runStartMs()}")
                if (ok[3] != stateName(detector.state)) bad.add("$where state: expected ${ok[3]}, got ${detector.state}")
            }
        }
        println("ClosureRunDetectorTest.componentErrorStreams: $streams streams, $calls calls, $refused refused with the Python message, problems ${bad.size}")
        assertEquals(6, streams)
        assertTrue(refused > 0)
        assertEquals(emptyList(), bad.take(20))
    }

    companion object {
        /** blink.py:110-112의 문자열. */
        internal fun stateName(s: RunState): String = when (s) {
            RunState.OPEN -> "open"
            RunState.CLOSED -> "closed"
            RunState.HOLD -> "hold"
        }

        /** 구간 둘이 같은지: 시각은 같은 정수, 실수는 수로 같음(FORMATS.md 2.7절), 플래그는 같다. */
        fun runProblems(where: String, want: ClosureRun?, got: ClosureRun?): List<String> {
            if (want == null) return if (got == null) emptyList() else listOf("$where: expected no run, got $got")
            if (got == null) return listOf("$where: expected $want, got no run")
            val out = ArrayList<String>()
            if (got.startMs != want.startMs) out.add("$where start_ms: expected ${want.startMs}, got ${got.startMs}")
            if (got.endMs != want.endMs) out.add("$where end_ms: expected ${want.endMs}, got ${got.endMs}")
            if (got.durationMs != want.durationMs) out.add("$where duration_ms: expected ${want.durationMs}, got ${got.durationMs}")
            if (!JsonTrees.numericallyEqual(got.amplitude, want.amplitude)) out.add("$where amplitude: expected ${want.amplitude}, got ${got.amplitude}")
            if (!JsonTrees.numericallyEqual(got.peakClosingVelocity, want.peakClosingVelocity)) {
                out.add("$where peak_closing_velocity: expected ${want.peakClosingVelocity}, got ${got.peakClosingVelocity}")
            }
            if (!JsonTrees.numericallyEqual(got.peakOpeningVelocity, want.peakOpeningVelocity)) {
                out.add("$where peak_opening_velocity: expected ${want.peakOpeningVelocity}, got ${got.peakOpeningVelocity}")
            }
            if (got.velocityValid != want.velocityValid) out.add("$where velocity_valid: expected ${want.velocityValid}, got ${got.velocityValid}")
            if (got.truncated != want.truncated) out.add("$where truncated: expected ${want.truncated}, got ${got.truncated}")
            return out
        }
    }
}

/**
 * 보고용 셈(계획 9.1절 E5 "정확히 같은 시각 칸의 수, ±1 ms를 쓴 수"): runs 항목의 start_ms, end_ms, duration_ms 칸을 센다.
 * 판정은 VectorComparator(SPEC 16장 층과 엄격 층)가 한다. 길이가 다른 목록은 앞에서부터 짝이 있는 것만 센다.
 */
internal class TimeCells {
    var cells = 0
        private set
    var exact = 0
        private set
    var withinOneMs = 0
        private set

    fun add(expected: Any?, actual: Any?) {
        val e = expected as? List<*> ?: return
        val a = actual as? List<*> ?: return
        for (i in 0 until minOf(e.size, a.size)) {
            val em = e[i] as? Map<*, *> ?: continue
            val am = a[i] as? Map<*, *> ?: continue
            for (key in listOf("start_ms", "end_ms", "duration_ms")) {
                val ev = em[key] as? Long ?: continue
                cells++
                val av = am[key] as? Long
                if (av == ev) {
                    exact++
                } else if (av != null && Math.abs(av - ev) <= 1L) {
                    withinOneMs++
                }
            }
        }
    }

    fun summary(): String = "time cells (start_ms, end_ms, duration_ms) exactly equal $exact/$cells, used the +-1 ms tolerance $withinOneMs"
}
