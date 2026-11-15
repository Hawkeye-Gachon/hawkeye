package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.StepFeatures
import hawkeye.drowsy.core.spec.NodSpec
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * E6 (계획 9.1절)의 부품 자료: C-steps(StepAggregator 60경우, ThresholdEventCounter 200경우, NodDetector 200경우)와
 * C-func의 분위수 순위 표(n = 1..2000). 스텝 묶음 15개의 파이프라인 경로는 hawkeye.drowsy.core.PipelineVectorsTest에 있다.
 * 그리고 C-errors의 StepAggregator 흐름 6개(증가하지 않는 t의 거부와 그 뒤의 출력; E7의 자료지만 이 단계의 부품이다)와
 * start_ms 생성 검사. 값은 모두 수로 같아야 한다(FORMATS.md 2.7절: `==`, 그리고 둘 다 NaN). 기대값은 바꾸지 않은 Python
 * 참조가 낸 기록이다.
 */
class StepAggregatorTest {

    @Test
    fun componentStepRecords() {
        val data = TestData.readyJson("components/C-steps.json")
        val cases = TestData.list(data["cases"])
        val kinds = java.util.TreeMap<String, Int>()
        val bad = ArrayList<String>()
        var aggCalls = 0
        var aggSteps = 0
        var aggRuns = 0
        var counterCalls = 0
        var counted = 0
        var nodCalls = 0
        var nods = 0
        var startAfterZero = 0
        val featureMax = DoubleArray(17)
        for ((ci, c) in cases.withIndex()) {
            val case = TestData.obj(c)
            val kind = case["kind"] as String
            kinds[kind] = (kinds[kind] ?: 0) + 1
            val inputs = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            assertEquals(inputs.size, outs.size, "case $ci")
            when (kind) {
                "aggregator" -> {
                    // components.py:332, 348: StepAggregator(pipeline_spec_from_dict(pb), start_ms, origin)
                    val pipeline = SpecLoader.pipelineSpecFromTree(case["pipeline"])
                    val startMs = VectorFile.long(case["start_ms"], "start_ms")
                    if (startMs > 0) startAfterZero++
                    val agg = StepAggregator(pipeline, startMs, VectorFile.long(case["origin_ms"], "origin_ms"))
                    for (k in inputs.indices) {
                        aggCalls++
                        val call = TestData.list(inputs[k])
                        val state = JsonTrees.frameState(call[0], "case $ci call $k state")
                        val run = JsonTrees.runOrNull(call[1], "case $ci call $k run")
                        if (run != null) aggRuns++
                        val got = agg.update(state, run, VectorFile.long(call[2], "ongoing_ms"))
                        val want = TestData.list(outs[k])
                        aggSteps += want.size
                        bad.addAll(stepProblems("case $ci call $k", want, got, featureMax))
                    }
                }
                "counter" -> {
                    val a = TestData.obj(case["args"])
                    // components.py:368: ThresholdEventCounter(**args)
                    val counter = ThresholdEventCounter(
                        enterThr = VectorFile.num(a["enter_thr"], "enter_thr"),
                        exitThr = VectorFile.num(a["exit_thr"], "exit_thr"),
                        minDurationMs = VectorFile.long(a["min_duration_ms"], "min_duration_ms"),
                        maxDurationMs = VectorFile.long(a["max_duration_ms"], "max_duration_ms"),
                        countAtEnd = VectorFile.bool(a["count_at_end"], "count_at_end"),
                        cancelGapMs = VectorFile.long(a["cancel_gap_ms"], "cancel_gap_ms"),
                    )
                    for (k in inputs.indices) {
                        counterCalls++
                        val call = TestData.list(inputs[k])
                        var got = false
                        if (call[0] == "r") {
                            counter.reset()
                        } else {
                            got = counter.update(VectorFile.long(call[1], "t"), VectorFile.num(call[2], "v"), VectorFile.bool(call[3], "valid"))
                        }
                        val o = TestData.list(outs[k])
                        if (o[0] == true) counted++
                        if (o[0] != got || o[1] != counter.active) {
                            bad.add("case $ci call $k: expected $o, got [$got, ${counter.active}]")
                        }
                    }
                }
                "nod" -> {
                    val cfg = TestData.obj(case["cfg"])
                    // components.py:389: NodDetector(NodSpec(**cfg)), 검사 없이 만든 설정이다.
                    val nod = NodDetector(
                        NodSpec(
                            enterDeg = VectorFile.num(cfg["enter_deg"], "enter_deg"),
                            exitDeg = VectorFile.num(cfg["exit_deg"], "exit_deg"),
                            minDurationMs = VectorFile.long(cfg["min_duration_ms"], "min_duration_ms"),
                            maxDurationMs = VectorFile.long(cfg["max_duration_ms"], "max_duration_ms"),
                            emaTauMs = VectorFile.long(cfg["ema_tau_ms"], "ema_tau_ms"),
                            emaMaxDtMs = VectorFile.long(cfg["ema_max_dt_ms"], "ema_max_dt_ms"),
                            cancelGapMs = VectorFile.long(cfg["cancel_gap_ms"], "cancel_gap_ms"),
                        ),
                    )
                    for (k in inputs.indices) {
                        nodCalls++
                        val call = TestData.list(inputs[k])
                        var got = false
                        if (call[0] == "r") {
                            nod.reset()
                        } else {
                            got = nod.update(VectorFile.long(call[1], "t"), VectorFile.num(call[2], "p"), VectorFile.bool(call[3], "valid"))
                        }
                        val o = TestData.list(outs[k])
                        if (o[0] == true) nods++
                        if (o[0] != got || !JsonTrees.numericallyEqual(nod.mean, VectorFile.num(o[1], "mean"))) {
                            bad.add("case $ci call $k: expected $o, got [$got, ${nod.mean}]")
                        }
                    }
                }
                else -> bad.add("case $ci: unknown kind $kind")
            }
        }
        println("StepAggregatorTest.componentStepRecords: C-steps cases per kind $kinds")
        println(
            "StepAggregatorTest.componentStepRecords: aggregator $aggCalls calls ($aggRuns with a run, $startAfterZero cases with start_ms > 0), " +
                "$aggSteps steps; counter $counterCalls calls ($counted counted); nod $nodCalls calls ($nods counted); differences ${bad.size}",
        )
        println("StepAggregatorTest.componentStepRecords: largest |kotlin - python| per feature #0..#16: ${featureMax.joinToString(", ")}")
        assertEquals(emptyList(), bad.take(20))
        assertEquals(0, bad.size)
    }

    /**
     * C-func의 분위수 순위 표 (계획 9.1절 E6, 9.3절 "분위수 순위의 자르기"): rank[i][n-1]은 값 0.0, 1.0, ..., n−1 가운데
     * Python nearest_rank_quantile이 고른 것의 1부터 센 순위다(FORMATS.md 2.7절). 정확히 같아야 한다(`floor(q·n) + 1`로 틀린
     * 구현은 벡터 102개를 모두 통과한다).
     */
    @Test
    fun quantileRankTable() {
        val data = TestData.readyJson("components/C-func.json")
        val table = TestData.obj(data["rank_table"])
        val qs = JsonTrees.doubles(table["qs"], "qs")
        val nMax = (table["n_max"] as Long).toInt()
        val ranks = TestData.list(table["rank"])
        assertEquals(qs.size, ranks.size)
        var cells = 0
        var integerQn = 0
        val bad = ArrayList<String>()
        for (qi in qs.indices) {
            val row = TestData.list(ranks[qi])
            assertEquals(nMax, row.size)
            for (n in 1..nMax) {
                cells++
                val qn = qs[qi] * n.toDouble()
                if (qn == Math.floor(qn)) integerQn++
                val values = List(n) { (n - 1 - it).toDouble() } // 거꾸로 넣어 정렬을 지나게 한다
                val rank = nearestRankQuantile(values, qs[qi]).toLong() + 1
                if (rank != row[n - 1]) bad.add("q=${qs[qi]} n=$n: expected ${row[n - 1]}, got $rank")
            }
        }
        println("StepAggregatorTest.quantileRankTable: qs $qs, n = 1..$nMax, cells $cells (q*n an integer in $integerQn), differences ${bad.size}")
        assertEquals(emptyList(), bad.take(20))
        assertEquals(0, bad.size)
    }

    @Test
    fun componentErrorStreams() {
        val data = TestData.readyJson("components/C-errors.json")
        var streams = 0
        var refused = 0
        var calls = 0
        var steps = 0
        val bad = ArrayList<String>()
        val featureMax = DoubleArray(17)
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            if (case["component"] != "StepAggregator") continue
            streams++
            val name = case["case"] as String
            val setup = TestData.obj(case["setup"])
            val agg = StepAggregator(
                SpecLoader.pipelineSpecFromTree(setup["pipeline"]),
                VectorFile.long(setup["start_ms"], "start_ms"),
                VectorFile.long(setup["origin_ms"], "origin_ms"),
            )
            val inputs = TestData.list(case["calls"])
            val outs = TestData.list(case["out"])
            for (k in inputs.indices) {
                calls++
                val call = TestData.list(inputs[k])
                val o = TestData.obj(outs[k])
                val result = runCatching {
                    agg.update(
                        JsonTrees.frameState(call[0], "$name call $k"),
                        JsonTrees.runOrNull(call[1], "$name call $k run"),
                        VectorFile.long(call[2], "ongoing_ms"),
                    )
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
                val want = TestData.list(o["ok"])
                steps += want.size
                bad.addAll(stepProblems("$name call $k", want, result.getOrThrow(), featureMax))
            }
        }
        println("StepAggregatorTest.componentErrorStreams: $streams streams, $calls calls, $refused refused with the Python message, $steps steps, problems ${bad.size}")
        assertEquals(6, streams)
        assertTrue(refused > 0)
        assertEquals(emptyList(), bad.take(20))
    }

    /**
     * Python 단위 테스트 tests/test_step_features.py:559-563 `test_start_must_be_on_step_boundary`를 옮겼다:
     * start_ms 60500과 -1000(origin 0)은 ValueError다. Python 테스트는 예외 종류만 보므로 여기서도 종류만 본다
     * (ValueError → IllegalArgumentException, K21).
     */
    @Test
    fun startMustBeOnStepBoundary() {
        val pipeline = JsonTrees.shippedSpec.pipeline
        assertFailsWith<IllegalArgumentException> { StepAggregator(spec = pipeline, startMs = 60500L, originMs = 0L) }
        assertFailsWith<IllegalArgumentException> { StepAggregator(spec = pipeline, startMs = -1000L, originMs = 0L) }
    }

    /**
     * 메시지와 그 밖의 경계 값. 입력과 판정은 바꾸지 않은 참조를 돌려 기록한 결과다(2026-11-15, Python 3.11.16,
     * drowsy/step_features.py sha256 b69fbe26...dbc9, 출고 export/feature_spec_v0.json의 pipeline 블록). 그 실행은
     * `StepAggregator(spec, start_ms, origin_ms)`를 아래 여덟 쌍으로 불렀고, 60500, -1000, -1, 1, 500, 1500(origin 0)은
     * 모두 ValueError "start_ms must be a non-negative multiple of clock.step_ms"를 냈고, (0, 0)과
     * (60000, 1731465123457)은 받아들였다. 받아들이는 쪽은 내보낸 C-steps 자료(start_ms 0과 0보다 큰 값 40개, 큰 origin)가
     * componentStepRecords에서도 본다.
     */
    @Test
    fun startMsVerdictsAndMessageAsRecordedFromPython() {
        val pipeline = JsonTrees.shippedSpec.pipeline
        for (bad in listOf(60500L, -1000L, -1L, 1L, 500L, 1500L)) {
            val e = assertFailsWith<IllegalArgumentException>("start_ms $bad") {
                StepAggregator(spec = pipeline, startMs = bad, originMs = 0L)
            }
            assertEquals("start_ms must be a non-negative multiple of clock.step_ms", e.message)
        }
        StepAggregator(spec = pipeline, startMs = 0L, originMs = 0L)
        StepAggregator(spec = pipeline, startMs = 60000L, originMs = 1731465123457L)
    }

    companion object {
        /** 스텝 목록 둘이 같은지(k, t_end_ms, full_lookback은 같은 값, values 17개는 수로 같음). featureMax에 특징별 가장 큰 |d|. */
        fun stepProblems(where: String, want: List<Any?>, got: List<StepFeatures>, featureMax: DoubleArray): List<String> {
            if (want.size != got.size) return listOf("$where: expected ${want.size} steps, got ${got.size}")
            val out = ArrayList<String>()
            for (j in want.indices) {
                val w = TestData.obj(want[j])
                val g = got[j]
                if (w["k"] != g.k.toLong()) out.add("$where step $j k: expected ${w["k"]}, got ${g.k}")
                if (w["t_end_ms"] != g.tEndMs) out.add("$where step $j t_end_ms: expected ${w["t_end_ms"]}, got ${g.tEndMs}")
                if (w["full_lookback"] != g.fullLookback) out.add("$where step $j full_lookback: expected ${w["full_lookback"]}, got ${g.fullLookback}")
                val values = JsonTrees.doubles(w["values"], "$where step $j values")
                if (values.size != g.values.size) {
                    out.add("$where step $j: expected ${values.size} values, got ${g.values.size}")
                    continue
                }
                for (f in values.indices) {
                    val d = Math.abs(g.values[f] - values[f])
                    if (d > featureMax[f] || d.isNaN()) featureMax[f] = if (d.isNaN()) Double.POSITIVE_INFINITY else d
                    if (!JsonTrees.numericallyEqual(g.values[f], values[f])) {
                        out.add("$where step $j #$f: expected ${java.lang.Double.toHexString(values[f])}, got ${java.lang.Double.toHexString(g.values[f])}")
                    }
                }
            }
            return out
        }
    }
}
