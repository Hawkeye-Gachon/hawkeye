package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.BaselineDiagnostics
import hawkeye.drowsy.core.spec.ClosureSpec
import hawkeye.drowsy.core.spec.PipelineSpec
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorComparator
import hawkeye.drowsy.core.testkit.VectorFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * E4 (계획 9.1절), 부품 경로: `baseline` 묶음 9개를 `DriverPipeline` 없이 SPEC 16장 판정과 엄격 층으로 비교한다
 * (tests/vectors/README.md "baseline": 프레임마다 signals.compute → BaselineEstimator.update → 완성 전에는 전역 상수 기준선과
 * provisional = true, 완성된 프레임부터는 그 기준선과 provisional = false로 frame_state). 파이프라인 경로는 DriverPipeline이
 * 생기는 단계 3~4의 일이다. 그리고 C-baseline 기록(80경우), C-errors의 BaselineEstimator 흐름(거부 뒤에도 출력이 맞다),
 * C-func의 calibrate 함수 기록과 순위 표, |yaw_rel| 정확히 40도의 closureValid 단위 시험(계획 5.3절, 9.3절).
 */
class BaselineTest {

    private val spec = JsonTrees.shippedSpec.pipeline

    @Test
    fun baselineBundleComponentPath() {
        val vectors = TestData.vectorFiles().map { it to VectorFile.read(it.readBytes()) }.filter { it.second.bundle == "baseline" }
        assertEquals(9, vectors.size, "baseline vectors")
        val stats = FieldStats { 1e-6 }
        val failed = ArrayList<String>()
        val notes = ArrayList<String>()
        val baselines = ArrayList<String>()
        var rows = 0
        for ((file, v) in vectors) {
            val stage0 = v.stageZeroProblems(file.nameWithoutExtension, JsonTrees.shippedSpec.pipelineHash, JsonTrees.shippedRulesSha256)
            if (stage0.isNotEmpty()) {
                failed.add("${v.name}: constants differ: $stage0")
                continue
            }
            if (v.configOverrides.isNotEmpty()) failed.add("${v.name}: config_overrides are not expected in the baseline bundle")
            val run = ComponentRun(spec, v.baseline)
            for (frame in v.frames) run.feed(frame)
            val actual = mapOf("closure" to run.closure, "baseline" to run.firstBaseline)
            // 부품 경로가 내는 두 항목만 비교한다(counters의 steps_emitted 등은 파이프라인 경로의 일이다).
            val problems = ArrayList<String>()
            val strict = VectorComparator.StrictResult()
            for (section in listOf("closure", "baseline")) {
                if (!v.expected.containsKey(section)) continue
                val tol = v.tolerances[section] ?: emptyMap()
                VectorComparator.compare(section, section, v.expected[section], actual[section], tol, "expected.$section", problems)
                VectorComparator.strictLayer(section, v.expected[section], actual[section], "expected.$section", strict)
            }
            // 받아들인 프레임 수와 버린 프레임 수는 이 실행기(pipeline.py:159-170의 받아들임)가 센 것과 같아야 한다.
            val counters = TestData.obj(v.expected["counters"])
            if (counters["frames_accepted"] != run.accepted || counters["dropped_frames"] != run.dropped) {
                problems.add("counters: expected ${counters["frames_accepted"]}/${counters["dropped_frames"]}, got ${run.accepted}/${run.dropped}")
            }
            if (problems.isNotEmpty() || strict.failures.isNotEmpty()) failed.add("${v.name}: $problems ${strict.failures}")
            notes.addAll(strict.notes.map { "${v.name}: $it" })
            val expectedClosure = VectorFile.list(v.expected["closure"], "expected.closure")
            for (i in expectedClosure.indices) {
                if (i < run.closure.size) stats.row(expectedClosure[i] as Map<*, *>, run.closure[i])
            }
            rows += run.closure.size
            if (v.expected.containsKey("baseline")) {
                val eb = TestData.obj(v.expected["baseline"])
                val ab = run.firstBaseline
                stats.row(eb, ab ?: emptyMap<String, Any?>())
                baselines.add(
                    "${v.name}: t_complete_ms ${ab?.get("t_complete_ms")} status ${ab?.get("status")} first used at t_ms ${ab?.get("t_ms")} " +
                        "(python ${eb["t_complete_ms"]} ${eb["status"]} ${eb["t_ms"]})",
                )
            }
        }
        println("BaselineTest.baselineBundleComponentPath: ${vectors.size - failed.size}/${vectors.size} vectors pass (closure, baseline; SPEC 16 and strict layer), closure rows $rows")
        println("BaselineTest.baselineBundleComponentPath: rows with every field within 1e-9: ${stats.rowsWithin1e9}/${stats.rows} (closure rows and the 6 baselines)")
        println("BaselineTest.baselineBundleComponentPath: max |kotlin - python| per field (fraction of 1e-6): ${stats.summary()}")
        for (b in baselines) println("BaselineTest.baselineBundleComponentPath: $b")
        println("BaselineTest.baselineBundleComponentPath: strict-layer notes (above 1e-9, not failures): ${notes.size} ${notes.take(20)}")
        assertEquals(emptyList(), failed)
        assertEquals(6, baselines.size)
    }

    /**
     * 벡터 README의 부품 경로. 받아들임(버림 검사)은 pipeline.py:159-170 그대로이고, 기준선과 상태는 pipeline.py:175-182 그대로다.
     * 첫 기준선의 t_ms는 vectors.py:619-620처럼 기준선을 처음 가진 프레임의 t_ms다.
     */
    private class ComponentRun(private val spec: PipelineSpec, private val injected: Baseline?) {
        private val estimator = if (injected == null) BaselineEstimator(spec) else null
        private val global = globalBaseline(spec)
        private var baseline: Baseline? = injected
        private var hasFrame = false
        private var prevT = 0L
        var accepted = 0L
        var dropped = 0L
        val closure = ArrayList<Map<String, Any?>>()
        var firstBaseline: Map<String, Any?>? = null

        fun feed(frame: hawkeye.drowsy.core.RawFrame) {
            if (hasFrame && frame.tMs <= prevT) {
                dropped++
                return
            }
            hasFrame = true
            prevT = frame.tMs
            accepted++
            val sig = compute(frame, spec)
            if (baseline == null && estimator != null) {
                val completed = estimator.update(sig)
                if (completed != null) baseline = completed
            }
            val b = baseline
            val state = if (b == null) frameState(sig, global, spec, true) else frameState(sig, b, spec, false)
            if (injected == null && firstBaseline == null && b != null) {
                val m = LinkedHashMap<String, Any?>()
                m["t_ms"] = frame.tMs
                m.putAll(JsonTrees.baseline(b))
                firstBaseline = m
            }
            closure.add(JsonTrees.state(state))
        }
    }

    /**
     * C-baseline (FORMATS.md 2.7절): 경우마다 새 BaselineEstimator(spec)에 FrameSignals를 차례로 넣고 호출마다 기준선 또는
     * null을 비교한다. 끝의 diagnostics도 비교한다. r0까지 모두 수로 같아야 한다(FORMATS.md 2.7절: "compare" 키가 없는
     * 경우는 수로 같음, 계획 9.2.6절). 계획 9.2.4절의 r0 절대 1e-12는 RAW·SIG 세션의 규칙이라 여기에 쓰지 않는다.
     * r0은 sin, cos를 지나므로(rotation_from_euler) 비트가 같은 원소 수도 센다. 기기에서 이것이 너무 엄격하면 시험의 허용
     * 오차를 넓히지 않고 내보내기가 그 경우에 "compare"를 적어야 한다.
     */
    @Test
    fun componentBaselineRecords() {
        val data = TestData.readyJson("components/C-baseline.json")
        val cases = TestData.list(data["cases"])
        val bad = ArrayList<String>()
        var calls = 0
        var completed = 0
        val statuses = java.util.TreeMap<String, Int>()
        var r0Bits = 0
        var r0Values = 0
        var r0Worst = 0.0
        for ((ci, c) in cases.withIndex()) {
            val case = TestData.obj(c)
            val pipeline = SpecLoader.pipelineSpecFromTree(case["pipeline"])
            val est = BaselineEstimator(pipeline)
            val inputs = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            assertEquals(inputs.size, outs.size, "case $ci")
            for (k in inputs.indices) {
                calls++
                val got = est.update(JsonTrees.frameSignals(inputs[k], "case $ci call $k"))
                val want = outs[k]
                if (want == null) {
                    if (got != null) bad.add("case $ci call $k: expected null, got $got")
                    continue
                }
                if (got == null) {
                    bad.add("case $ci call $k: expected a baseline, got null")
                    continue
                }
                completed++
                val w = TestData.obj(want)
                statuses[w["status"] as String] = (statuses[w["status"] as String] ?: 0) + 1
                val a = JsonTrees.baseline(got)
                for ((key, ev) in w) {
                    val av = a[key]
                    if (key == "r0") {
                        val e = JsonTrees.doubles(ev, "r0")
                        @Suppress("UNCHECKED_CAST")
                        val x = av as List<Double>
                        for (j in e.indices) {
                            r0Values++
                            if (x[j].toRawBits() == e[j].toRawBits()) r0Bits++
                            val d = Math.abs(x[j] - e[j])
                            if (d > r0Worst) r0Worst = d
                            if (!JsonTrees.numericallyEqual(x[j], e[j])) bad.add("case $ci call $k r0[$j]: expected ${e[j]}, got ${x[j]}")
                        }
                    } else if (!sameValue(ev, av)) {
                        bad.add("case $ci call $k $key: expected $ev, got $av")
                    }
                }
            }
            bad.addAll(diagnosticsProblems("case $ci", case["diagnostics"], est.diagnostics))
        }
        println("BaselineTest.componentBaselineRecords: ${cases.size} cases, $calls update calls, $completed baselines $statuses, problems ${bad.size}")
        println("BaselineTest.componentBaselineRecords: r0 elements bit-identical $r0Bits/$r0Values, largest |d| $r0Worst")
        assertEquals(emptyList(), bad.take(20))
    }

    /** C-errors의 BaselineEstimator 흐름: 증가하지 않는 t는 같은 메시지로 거부되고, 흐름은 이어지며 뒤의 출력도 맞다. */
    @Test
    fun componentErrorStreams() {
        val data = TestData.readyJson("components/C-errors.json")
        var streams = 0
        var refused = 0
        var calls = 0
        val bad = ArrayList<String>()
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            if (case["component"] != "BaselineEstimator") continue
            streams++
            val name = case["case"] as String
            val est = BaselineEstimator(SpecLoader.pipelineSpecFromTree(TestData.obj(case["setup"])["pipeline"]))
            val inputs = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            for (k in inputs.indices) {
                calls++
                val o = TestData.obj(outs[k])
                val result = runCatching { est.update(JsonTrees.frameSignals(inputs[k], "$name call $k")) }
                if (o.containsKey("error")) {
                    refused++
                    val err = TestData.obj(o["error"])
                    val e = result.exceptionOrNull()
                    if (e !is IllegalArgumentException || err["type"] != "ValueError" || err["message"] != e.message) {
                        bad.add("$name call $k: expected $err, got ${result.getOrNull() ?: e}")
                    }
                } else {
                    val want = o["ok"]
                    if (result.isFailure) {
                        bad.add("$name call $k: unexpected ${result.exceptionOrNull()}")
                        continue
                    }
                    val got = result.getOrNull()
                    if (want == null) {
                        if (got != null) bad.add("$name call $k: expected null, got $got")
                    } else if (got == null) {
                        bad.add("$name call $k: expected a baseline, got null")
                    } else {
                        val a = JsonTrees.baseline(got)
                        for ((key, ev) in TestData.obj(want)) {
                            // r0도 수로 같아야 한다(FORMATS.md 2.7절, 계획 9.2.6절). sameValue가 원소마다 수로 비교한다.
                            if (!sameValue(ev, a[key])) {
                                bad.add("$name call $k $key: expected $ev, got ${a[key]}")
                            }
                        }
                    }
                }
            }
        }
        println("BaselineTest.componentErrorStreams: $streams streams, $calls calls, $refused refused with the Python message, problems ${bad.size}")
        assertEquals(6, streams)
        assertEquals(emptyList(), bad.take(20))
    }

    /** C-func의 calibrate 함수 기록: nearest_rank_quantile, median, eye_closure (모두 exact), 빈 입력 오류, 순위 표. */
    @Test
    fun componentFunctionRecords() {
        val data = TestData.readyJson("components/C-func.json")
        val counts = java.util.TreeMap<String, Int>()
        val bad = ArrayList<String>()
        var negZeroResults = 0
        for ((i, c) in TestData.list(data["cases"]).withIndex()) {
            val case = TestData.obj(c)
            val fn = case["fn"] as String
            if (fn !in CALIBRATE_FNS) continue
            counts[fn] = (counts[fn] ?: 0) + 1
            val args = TestData.list(case["args"])
            val error = case["error"]?.let { TestData.obj(it) }
            val result = runCatching {
                when (fn) {
                    "nearest_rank_quantile" -> nearestRankQuantile(JsonTrees.doubles(args[0], "args"), VectorFile.num(args[1], "q"))
                    "median" -> median(JsonTrees.doubles(args[0], "args"))
                    else -> {
                        val cfg = TestData.obj(args[4])
                        eyeClosure(
                            ear = VectorFile.num(args[0], "ear"), blink = VectorFile.num(args[1], "blink"),
                            e = VectorFile.num(args[2], "e"), b = VectorFile.num(args[3], "b"),
                            cfg = closureCfg(
                                rho = VectorFile.num(cfg["rho"], "rho"), bClosed = VectorFile.num(cfg["b_closed"], "b_closed"),
                                bSpanMin = VectorFile.num(cfg["b_span_min"], "b_span_min"), w = VectorFile.num(cfg["w"], "w"),
                            ),
                        )
                    }
                }
            }
            if (error != null) {
                val e = result.exceptionOrNull()
                if (e !is IllegalArgumentException || error["type"] != "ValueError" || error["message"] != e.message) {
                    bad.add("case $i $fn: expected $error, got ${result.getOrNull() ?: e}")
                }
                continue
            }
            if (result.isFailure) {
                bad.add("case $i $fn: unexpected ${result.exceptionOrNull()}")
                continue
            }
            val got = result.getOrThrow()
            val want = VectorFile.num(case["result"], "result")
            // 수로 같음. 그리고 0의 부호도 Python과 같아야 한다(R7: 안정 정렬이 -0.0과 0.0의 도착 순서를 지킨다).
            if (!JsonTrees.numericallyEqual(got, want) || (want == 0.0 && got.toRawBits() != want.toRawBits())) {
                bad.add("case $i $fn: expected ${java.lang.Double.toHexString(want)}, got ${java.lang.Double.toHexString(got)}")
            }
            if (want == 0.0 && want.toRawBits() != 0L) negZeroResults++
        }
        // 순위 표: rank[i][n-1]은 값 0.0, 1.0, ..., n−1 가운데 Python이 고른 것의 1부터 센 순위다.
        val table = TestData.obj(data["rank_table"])
        val qs = JsonTrees.doubles(table["qs"], "qs")
        val nMax = (table["n_max"] as Long).toInt()
        val ranks = TestData.list(table["rank"])
        var cells = 0
        for (qi in qs.indices) {
            val row = TestData.list(ranks[qi])
            assertEquals(nMax, row.size)
            for (n in 1..nMax) {
                cells++
                val values = List(n) { (n - 1 - it).toDouble() } // 거꾸로 넣어 정렬을 지나게 한다
                val picked = nearestRankQuantile(values, qs[qi])
                val rank = picked.toLong() + 1
                if (rank != row[n - 1]) bad.add("rank table q=${qs[qi]} n=$n: expected ${row[n - 1]}, got $rank")
            }
        }
        println("BaselineTest.componentFunctionRecords: C-func calibrate cases per function $counts (results -0.0: $negZeroResults), rank table cells $cells, problems ${bad.size}")
        assertEquals(CALIBRATE_FNS, counts.keys)
        assertEquals(emptyList(), bad.take(20))
    }

    /**
     * |yaw_rel| 정확히 40도 (계획 5.3절, 9.3절, SPEC 16.1 끝 목록 `docs/SPEC.md:2453-2454`). 기대 판정은 옮긴 Python 단위
     * 테스트 tests/test_calibrate.py:715-724: ±39.9와 정확히 ±40.0은 유효('이하'), ±40.1과 75.0은 무효.
     * 거기 더해 test_calibrate.py:701-703(쓸 수 있는 눈이 없으면 무효)과 얼굴 없음.
     */
    @Test
    fun closureValidAtExactly40Degrees() {
        val cfg = spec.closure
        assertEquals(40.0, cfg.validMaxAbsYawDeg)
        for (yaw in listOf(39.9, -39.9, 40.0, -40.0)) assertTrue(closureValid(true, true, yaw, cfg), "yaw_rel $yaw")
        for (yaw in listOf(40.1, -40.1, 75.0)) assertFalse(closureValid(true, true, yaw, cfg), "yaw_rel $yaw")
        assertFalse(closureValid(true, false, 10.0, cfg))
        assertFalse(closureValid(false, true, 0.0, cfg))
    }

    /**
     * tests/test_calibrate.py:715-724를 frameState로 옮긴 것. 입력은 그 시험 그대로다: make_baseline(jaw0 0.05, gaze_h0 0.1,
     * gaze_v0 −0.1; R0 = rotation_from_euler(0, 0, 0), status ok, t_complete_ms 60000), 감긴 두 눈(ear 0.0, blink 0.9),
     * pitch 0, roll 0. 기대값은 그 Python 시험의 판정이다: ±39.9와 ±40.0은 유효하고 closure 1.0이며 |yaw_rel|은 정확히 40.0,
     * ±40.1과 75.0은 무효이고 closure 0.0이며 yaw_rel은 입력과 1e-9 안.
     * 가정(A2): pitch 0, roll 0에서는 JVM StrictMath도 yaw_rel 40.0을 정확히 돌려준다(f_s2에서 잼). pitch −7.5, roll 3.25로는
     * JVM이 0x1.4000000000001p5, CPython이 0x1.4p+5를 내어 판정이 갈린다(Calibrate.kt closureValid의 주석). 그 입력은 이
     * 시험에 없다.
     */
    @Test
    fun frameStateAtExactly40DegreesAsInThePythonTest() {
        val baseline = Baseline(
            eL = 0.30, eR = 0.30, bL = 0.20, bR = 0.20, r0 = rotationFromEuler(0.0, 0.0, 0.0),
            jaw0 = 0.05, gazeH0 = 0.1, gazeV0 = -0.1, tCompleteMs = 60000L, status = hawkeye.drowsy.core.BaselineStatus.OK,
        )
        fun closed(yaw: Double) = FrameSignals(
            tMs = 1L, face = true, earL = 0.0, earR = 0.0, eyeWL = 40.0, eyeWR = 40.0, blinkL = 0.9, blinkR = 0.9,
            eyeLook = List(8) { 0.0 }, jawOpen = 0.05, yaw = yaw, pitch = 0.0, roll = 0.0,
        )
        for (yaw in listOf(39.9, -39.9)) {
            assertTrue(frameState(closed(yaw), baseline, spec, false).closureValid, "yaw $yaw")
        }
        for (yaw in listOf(40.0, -40.0)) {
            val state = frameState(closed(yaw), baseline, spec, false)
            assertEquals(40.0, Math.abs(state.yawRel), "yaw $yaw: |yaw_rel| ${java.lang.Double.toHexString(state.yawRel)}")
            assertEquals(40.0, spec.closure.validMaxAbsYawDeg)
            assertTrue(state.closureValid, "yaw $yaw")
            assertEquals(1.0, state.closure, "yaw $yaw")
        }
        for (yaw in listOf(40.1, -40.1, 75.0)) {
            val state = frameState(closed(yaw), baseline, spec, false)
            assertFalse(state.closureValid, "yaw $yaw")
            assertEquals(0.0, state.closure, "yaw $yaw")
            assertTrue(Math.abs(state.yawRel - yaw) <= 1e-9, "yaw $yaw: yaw_rel ${state.yawRel}")
        }
    }

    private fun closureCfg(rho: Double, bClosed: Double, bSpanMin: Double, w: Double): ClosureSpec =
        spec.closure.copy(rho = rho, bClosed = bClosed, bSpanMin = bSpanMin, w = w)

    private fun diagnosticsProblems(where: String, want: Any?, got: BaselineDiagnostics?): List<String> {
        if (want == null) return if (got == null) emptyList() else listOf("$where: diagnostics expected null, got $got")
        if (got == null) return listOf("$where: diagnostics expected, got null")
        val w = TestData.obj(want)
        val out = ArrayList<String>()
        if (w["n_face_frames"] != got.nFaceFrames.toLong()) out.add("$where: n_face_frames ${w["n_face_frames"]} vs ${got.nFaceFrames}")
        if (!sameValue(w["face_ratio"], got.faceRatio)) out.add("$where: face_ratio ${w["face_ratio"]} vs ${got.faceRatio}")
        if (!sameValue(w["blink_l_quantiles"], got.blinkLQuantiles)) out.add("$where: blink_l_quantiles ${w["blink_l_quantiles"]} vs ${got.blinkLQuantiles}")
        if (!sameValue(w["blink_r_quantiles"], got.blinkRQuantiles)) out.add("$where: blink_r_quantiles ${w["blink_r_quantiles"]} vs ${got.blinkRQuantiles}")
        return out
    }

    /** 자료의 JSON 값과 Kotlin 값이 같은지: 정수는 같은 정수, 실수는 수로 같음, 문자열과 목록은 원소마다. */
    private fun sameValue(e: Any?, a: Any?): Boolean = when (e) {
        null -> a == null
        is Long -> a is Long && a == e
        is Double -> (a is Double && JsonTrees.numericallyEqual(a, e))
        is String -> a == e
        is Boolean -> a == e
        is List<*> -> a is List<*> && a.size == e.size && e.indices.all { sameValue(e[it], a[it]) }
        else -> false
    }

    companion object {
        val CALIBRATE_FNS: Set<String> = sortedSetOf("nearest_rank_quantile", "median", "eye_closure")
    }
}
