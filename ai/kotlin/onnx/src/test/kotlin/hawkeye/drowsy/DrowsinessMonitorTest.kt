package hawkeye.drowsy

import hawkeye.drowsy.core.ModelFn
import hawkeye.drowsy.core.RawFrame
import hawkeye.drowsy.core.session.CalibrationStatus
import hawkeye.drowsy.core.session.DriverState
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testkit.DiffData
import hawkeye.drowsy.core.testkit.VectorComparator
import hawkeye.drowsy.core.testkit.VectorFile
import hawkeye.drowsy.core.testkit.VectorRunner
import org.junit.jupiter.api.Assumptions
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * 입구 DrowsinessMonitor의 시험 (리드 판정 9). 기대값은 참조 구현에서만 온다:
 * - e2e_250s 벡터(tests/vectors, 바꾸지 않은 Python 참조가 만든 출력)를 벡터 비교기(SPEC 16장 판정과 엄격 층)로 본다.
 * - active는 Python 내보내기의 active_expected/e2e_250s.json(FORMATS.md 2.5절)과 본다. 자료가 없으면 그 시험은 건너뛴다.
 *
 * 벡터 프레임의 lm·bs·m은 Python의 double이다. 입구는 MediaPipe처럼 FloatArray를 받으므로 이 시험은 그 값을 `toFloat()`로
 * 바꿔 넣는다. 그래서 E12(DriverPipeline에 double을 그대로 넣는 길)와 달리 마지막 자리가 다를 수 있고, 판정은 벡터의 허용
 * 오차와 엄격 층이 정한다. 같은 float32 반올림을 한 프레임을 바꾸지 않은 Python 참조로 돌린 결과도 SPEC 16장 문제 0이고 이벤트
 * 시각이 모두 같았다(작업 폴더 s5/e2e_float32_python.out, 진단일 뿐 기대값으로 쓰지 않는다).
 */
class DrowsinessMonitorTest {

    private val mediapipeSha: String =
        SpecLoader.parseFeatureSpec(File(DiffData.export, "feature_spec_v0.json").readBytes()).pipeline.mediapipe.modelSha256

    /** 벡터 프레임 하나를 입구의 인자로 바꿔 넣는다. 얼굴 없는 프레임은 updateNoFace. */
    private fun feed(monitor: DrowsinessMonitor, frame: RawFrame, speedKmh: Double?): DriverState? {
        if (!frame.face) return monitor.updateNoFace(frame.tMs, frame.w, frame.h, speedKmh)
        val n = 478 // 얼굴 랜드마크 모델의 점 수(홍채 468..477 포함). 벡터에는 필요한 44점만 있다.
        val x = FloatArray(n)
        val y = FloatArray(n)
        for ((i, p) in frame.lm) {
            x[i] = p.x.toFloat()
            y[i] = p.y.toFloat()
        }
        val names = ArrayList<String>(frame.bs.size)
        val scores = FloatArray(frame.bs.size)
        for ((j, e) in frame.bs.entries.withIndex()) {
            names.add(e.key)
            scores[j] = e.value.toFloat()
        }
        val matrix = FloatArray(frame.m.size) { frame.m[it].toFloat() }
        return monitor.update(frame.tMs, frame.w, frame.h, x, y, names, scores, matrix, speedKmh)
    }

    private class Run(val states: List<Pair<RawFrame, DriverState?>>, val actual: Map<String, Any?>)

    /**
     * 벡터 하나(기본은 e2e_250s)를 입구로 돌리고, 벡터 실행기와 같은 모양의 출력 나무를 만든다(VectorRunner.Collector).
     * repeatEvery가 있으면 번호 % repeatEvery == repeatEvery - 1인 프레임 바로 뒤에 같은 프레임을 한 번 더 넣는다(tMs가 늘지
     * 않으므로 파이프라인이 버린다). 그 반복 호출은 null이어야 하고 lastState를 바꾸지 않아야 한다.
     */
    private fun runE2e(
        monitorOf: (tap: (ModelFn) -> ModelFn) -> DrowsinessMonitor,
        v: VectorFile,
        beforeFrame: (Int) -> Unit = {},
        repeatEvery: Int? = null,
    ): Run {
        val collector = VectorRunner.Collector(baselineInjected = false)
        val states = ArrayList<Pair<RawFrame, DriverState?>>()
        monitorOf { m -> collector.recording(m) }.use { monitor ->
            for ((i, frame) in v.frames.withIndex()) {
                beforeFrame(i)
                val st = feed(monitor, frame, v.speedAt(frame.tMs))
                states.add(frame to st)
                if (st != null) collector.add(st.tMs, st.output, st.baseline)
                if (repeatEvery != null && i % repeatEvery == repeatEvery - 1) {
                    val before = monitor.lastState
                    val again = feed(monitor, frame, v.speedAt(frame.tMs))
                    assertEquals(null, again, "frame $i repeated (t=${frame.tMs}): the pipeline drops it")
                    assertSame(before, monitor.lastState, "frame $i repeated: lastState unchanged")
                }
            }
            val c = monitor.counters
            val lastRepeated = repeatEvery != null && (v.frames.size - 1) % repeatEvery == repeatEvery - 1
            if (!lastRepeated) assertEquals(c, monitor.lastState!!.counters, "counters of the last state = monitor counters")
            return Run(
                states,
                collector.result(c.framesAccepted, c.droppedFrames, c.stepsEmitted, c.modelCalls, c.nonfiniteWindows),
            )
        }
    }

    private fun e2eVector(): VectorFile = checkedVector("e2e_250s")

    private fun checkedVector(name: String): VectorFile {
        val v = DiffData.vector(name)
        val ctx = VectorRunner.Context.shipped()
        assertEquals(emptyList(), v.stageZeroProblems(name, ctx.featureSpec.pipelineHash, ctx.rulesSha256))
        return v
    }

    /** 비교기 판정과 항목별 가장 큰 |kotlin − python|을 적고, 통과해야 한다. */
    private fun assertVectorEqual(label: String, v: VectorFile, actual: Map<String, Any?>) {
        val result = VectorComparator.compareVector(v, actual)
        for (s in v.expected.keys.sorted()) {
            val d = MaxDiff()
            d.walk(v.expected[s], actual[s])
            println("$label: $s: numbers compared ${d.numbers}, largest |kotlin - python| ${d.max}, at ${d.at ?: "-"}")
        }
        println(
            "$label: SPEC 16 problems ${result.problems.size} ${result.problems}; strict-layer failures ${result.strict.failures.size} " +
                "${result.strict.failures}; strict-layer notes (above 1e-9, not failures) ${result.strict.notes.size} " +
                "${result.strict.notes.take(5)}; largest share of a tolerance used ${result.toleranceUse.maxRatio}",
        )
        assertTrue(result.passed, "${result.problems} ${result.strict.failures}")
    }

    /** E13-entry: e2e_250s를 DrowsinessMonitor.update로 넣으면 벡터의 여덟 항목이 모두 같다(SPEC 16장 판정과 엄격 층). */
    @Test
    fun e2eThroughUpdateEqualsTheVector() {
        val v = e2eVector()
        val run = runE2e({ tap -> DrowsinessMonitor.openBundled(mediapipeSha, tap) }, v)
        assertVectorEqual("DrowsinessMonitorTest.e2eThroughUpdateEqualsTheVector", v, run.actual)
        assertEquals(8, v.expected.size)

        // DriverState의 파생 값(계획 6.2절): sessionMs, speedKmh, calibration과 provisional, primaryAlert, isActive.
        val accepted = run.states.mapNotNull { it.second }
        val first = accepted.first().tMs
        var alerts = 0
        for ((frame, st) in run.states) {
            st!!
            assertEquals(frame.tMs, st.tMs)
            assertEquals(st.tMs - first, st.sessionMs)
            assertEquals(v.speedAt(frame.tMs), st.speedKmh)
            assertEquals(st.calibration == CalibrationStatus.CALIBRATING, st.output.state.provisional, "t=${st.tMs}")
            assertEquals(st.alerts.firstOrNull(), st.primaryAlert)
            for (a in st.active) assertTrue(st.isActive(a.type))
            alerts += st.alerts.size
        }
        println(
            "DrowsinessMonitorTest.e2eThroughUpdateEqualsTheVector: frames ${v.frames.size}, states ${accepted.size}, " +
                "face frames ${v.frames.count { it.face }}, alerts $alerts, last counters ${accepted.last().counters}",
        )
    }

    /** E13-entry: 같은 실행의 DriverState.active가 Python 내보내기의 active_expected/e2e_250s.json과 같다. */
    @Test
    fun e2eActiveEqualsExporterData() {
        Assumptions.assumeTrue(DiffData.readySmall() != null, "no READY_small.txt: differential data (small) not exported")
        val v = e2eVector()
        val exp = DiffData.readyJson("active_expected/e2e_250s.json")
        assertEquals("e2e_250s", exp["vector"])
        assertEquals(exp["vector_sha256"], hawkeye.drowsy.core.check.sha256Hex(File(DiffData.vectors, "e2e_250s.json").readBytes()))
        val rules = SpecLoader.parseRules(File(DiffData.export, "rules.json").readBytes())
        assertEquals(DiffData.list(exp["alert_priority"]), rules.alertPriority.map { it.wire })

        val run = runE2e({ tap -> DrowsinessMonitor.openBundled(mediapipeSha, tap) }, v)
        val byT = HashMap<Long, List<Map<String, Any?>>>()
        for (f in DiffData.list(exp["frames"])) {
            val m = DiffData.obj(f)
            byT[m["t_ms"] as Long] = DiffData.list(m["active"]).map { DiffData.obj(it) }
        }
        var framesWithEvents = 0
        var entries = 0
        var unchanged = 0
        var previous: List<hawkeye.drowsy.core.session.ActiveEvent> = emptyList()
        for ((_, st) in run.states) {
            st!!
            val want = byT[st.tMs]
            if (want == null) {
                // 이벤트가 없는 프레임에서는 active가 바뀌지 않는다(같은 목록 객체).
                assertTrue(st.events.isEmpty(), "t=${st.tMs}: events but no active_expected frame")
                assertSame(previous, st.active, "t=${st.tMs}: active changed without events")
                unchanged++
            } else {
                assertTrue(st.events.isNotEmpty(), "t=${st.tMs}: active_expected frame without events")
                assertEquals(want.size, st.active.size, "t=${st.tMs}: active ${st.active.map { it.type }}")
                for ((j, w) in want.withIndex()) {
                    val a = st.active[j]
                    val where = "t=${st.tMs} active[$j]"
                    assertEquals(6, w.size, "$where keys ${w.keys}")
                    assertEquals(w["type"], a.type.wire, "$where.type")
                    assertEquals(w["severity"], a.severity.toLong(), "$where.severity")
                    assertEquals(w["t_start_ms"], a.tStartMs, "$where.t_start_ms")
                    assertEquals(w["source"], a.source.wire, "$where.source")
                    assertEquals(w["provisional"], a.provisional, "$where.provisional")
                    assertEquals(w["onset_frame_t_ms"], a.onsetFrameTMs, "$where.onset_frame_t_ms")
                    entries++
                }
                framesWithEvents++
            }
            previous = st.active
        }
        assertEquals(byT.size, framesWithEvents, "every active_expected frame was met")
        println(
            "DrowsinessMonitorTest.e2eActiveEqualsExporterData: ${byT.size} frames with events in active_expected, $framesWithEvents compared, " +
                "$entries active entries equal, $unchanged frames without events kept the same list; active after the last frame " +
                "${run.states.last().second!!.active.map { it.type.wire }}",
        )
    }

    /**
     * 겹친 호출: 모델 함수 안에서(첫 추론, 스텝 239) 막아 둔 사이에 다른 스레드가 update와 updateNoFace를 부르면
     * IllegalStateException("update must be called one at a time")이고, 그 호출은 상태를 바꾸지 않는다: 막힌 실행이 끝까지
     * 돌아도 출력이 벡터와 같다.
     */
    @Test
    fun overlappingCallIsRefusedAndChangesNothing() {
        val v = e2eVector()
        val inside = CountDownLatch(1)
        val release = CountDownLatch(1)
        val calls = AtomicInteger(0)
        var monitorRef: DrowsinessMonitor? = null
        var run: Run? = null
        var failure: Throwable? = null
        val worker = Thread {
            try {
                run = runE2e(
                    { tap ->
                        DrowsinessMonitor.openBundled(mediapipeSha) { m ->
                            val inner = tap(m)
                            ModelFn { w ->
                                if (calls.getAndIncrement() == 0) {
                                    inside.countDown()
                                    check(release.await(60, TimeUnit.SECONDS)) { "not released" }
                                }
                                inner.predict(w)
                            }
                        }.also { monitorRef = it }
                    },
                    v,
                )
            } catch (t: Throwable) {
                failure = t
                inside.countDown()
            }
        }
        worker.start()
        assertTrue(inside.await(120, TimeUnit.SECONDS), "the model was not called")
        val monitor = monitorRef!!
        val lastBefore = monitor.lastState
        val countersBefore = monitor.counters
        val later = 10_000_000L
        val e1 = assertFailsWith<IllegalStateException> { monitor.updateNoFace(later, 640, 480, 80.0) }
        val face = v.frames.first { it.face }
        val e2 = assertFailsWith<IllegalStateException> { feed(monitor, RawFrame(later + 1, true, face.w, face.h, face.lm, face.bs, face.m), 80.0) }
        assertEquals("update must be called one at a time", e1.message)
        assertEquals("update must be called one at a time", e2.message)
        assertSame(lastBefore, monitor.lastState)
        assertEquals(countersBefore, monitor.counters)
        release.countDown()
        worker.join(120_000)
        failure?.let { throw it }
        assertVectorEqual("DrowsinessMonitorTest.overlappingCallIsRefusedAndChangesNothing", v, run!!.actual)
        println(
            "DrowsinessMonitorTest.overlappingCallIsRefusedAndChangesNothing: 2 overlapping calls refused with '${e1.message}' while the " +
                "first model call (counters then $countersBefore) was blocked; model calls ${calls.get()}",
        )
    }

    /** 닫은 뒤: close는 두 번 불러도 되고, update와 updateNoFace는 IllegalStateException. lastState와 counters는 읽힌다. */
    @Test
    fun updateAfterCloseIsRefused() {
        val v = e2eVector()
        val monitor = DrowsinessMonitor.open(mediapipeSha)
        var last: DriverState? = null
        for (frame in v.frames.take(30)) last = feed(monitor, frame, v.speedAt(frame.tMs)) ?: last
        assertNotNull(last)
        val counters = monitor.counters
        monitor.close()
        monitor.close()
        val next = v.frames[30]
        val e1 = assertFailsWith<IllegalStateException> { feed(monitor, next, 80.0) }
        val e2 = assertFailsWith<IllegalStateException> { monitor.updateNoFace(next.tMs, next.w, next.h, 80.0) }
        assertEquals("the DrowsinessMonitor is closed", e1.message)
        assertEquals("the DrowsinessMonitor is closed", e2.message)
        assertSame(last, monitor.lastState)
        assertEquals(counters, monitor.counters)
        println("DrowsinessMonitorTest.updateAfterCloseIsRefused: 30 frames, close twice, then update and updateNoFace refused with '${e1.message}'; counters $counters")
    }

    /**
     * 얼굴 없는 프레임(updateNoFace)과 버린 프레임(늘지 않는 tMs → null)을 입구로 넣어 참조와 본다.
     *
     * 1. 벡터를 그대로 넣으면 벡터의 모든 항목이 같다(SPEC 16장 판정과 엄격 층). 두 벡터 모두 기준선을 넣지 않고 모델을 부를
     *    만큼 길지 않다(70 s, 183 s; 첫 모델 호출은 239 s).
     * 2. 50번째 프레임마다 같은 프레임을 한 번 더 넣으면, 반복 호출은 null이고 lastState를 바꾸지 않으며, counters를 뺀 모든
     *    항목이 여전히 벡터와 같고, counters는 바꾸지 않은 Python 참조가 같은 입력(float32로 반올림한 프레임, 같은 반복)으로 낸
     *    값이다: 작업 폴더 f_s5/noface_dup_python.py와 .out(vectors._run_pipeline, Python 3.11.16). 그 실행에서 counters 밖의
     *    항목은 SPEC 16장 문제 0이었다.
     */
    @Test
    fun noFaceAndDroppedFramesThroughTheEntryEqualTheReference() {
        // 이름 → (반복한 프레임 수, Python 참조의 counters: frames_accepted, dropped_frames, steps_emitted, model_calls, nonfinite_windows)
        val repeatedReference = mapOf(
            "rules_calibrating_provisional" to listOf(8L, 423L, 8L, 10L, 0L, 0L),
            "baseline_fallback" to listOf(22L, 1101L, 22L, 3L, 0L, 0L),
        )
        for ((name, ref) in repeatedReference) {
            val v = checkedVector(name)
            assertTrue(v.baseline == null, "$name: no injected baseline")
            val noFace = v.frames.count { !it.face }

            val plain = runE2e({ tap -> DrowsinessMonitor.openBundled(mediapipeSha, tap) }, v)
            assertVectorEqual("DrowsinessMonitorTest.noFaceAndDroppedFrames $name", v, plain.actual)
            assertTrue(plain.states.all { it.second != null })
            for ((frame, st) in plain.states) assertEquals(frame.face, st!!.output.state.face, "$name t=${frame.tMs}")

            val repeated = runE2e({ tap -> DrowsinessMonitor.openBundled(mediapipeSha, tap) }, v, repeatEvery = 50)
            val withoutCounters = VectorFile(
                name = v.name, bundle = v.bundle, description = v.description, pipelineHash = v.pipelineHash,
                rulesHash = v.rulesHash, tolerances = v.tolerances, baseline = v.baseline, configOverrides = v.configOverrides,
                model = v.model, frames = v.frames, speed = v.speed, modelInputs = v.modelInputs,
                expected = v.expected.filterKeys { it != "counters" }, root = v.root,
            )
            assertVectorEqual("DrowsinessMonitorTest.noFaceAndDroppedFrames $name repeated", withoutCounters, repeated.actual)
            val c = DiffData.obj(repeated.actual["counters"])
            assertEquals(
                ref.drop(1),
                listOf(c["frames_accepted"], c["dropped_frames"], c["steps_emitted"], c["model_calls"], c["nonfinite_windows"]),
                "$name repeated: counters as the Python reference",
            )
            assertEquals(ref[0], (v.frames.size / 50).toLong(), "$name: frames repeated")
            println(
                "DrowsinessMonitorTest.noFaceAndDroppedFramesThroughTheEntryEqualTheReference: $name frames ${v.frames.size} " +
                    "(no face $noFace); plain run equals the vector; ${ref[0]} repeated frames gave null, counters $c as the Python reference",
            )
        }
    }

    /**
     * 처리 오류에서 멈춘다(계획 5.8절 K8, 참조 drowsy/live.py:1527-1528, 1624-1631: 프레임 하나를 처리하다 난 오류는 종류를
     * 가리지 않고 세션을 끝낸다). 모델 함수가 값 2개를 돌려주면 파이프라인이 거부한다(pipeline.py:241-242).
     *
     * 기대값은 바꾸지 않은 참조의 기록된 실행에서 온다: 작업 폴더 f_s5/live_failstop_python.py와 .out(Python 3.11.16,
     * 진짜 LiveLoop.frames와 LiveSession.process_raw; MediaPipe 자리만 벡터의 RawFrame을 넘긴다; e2e_250s 프레임을 float32로
     * 반올림, 속도 None). 그 실행: 프레임 1434(t_ms 239001)에서 ValueError "model must return 3 probabilities (alert,
     * low_vigilant, drowsy)", 처리한 프레임 1434, 그때 파이프라인 계수 frames_accepted 1435, dropped 0, steps 179,
     * model_calls 0, nonfinite 0, 그 뒤 68 프레임은 처리하지 않았다.
     *
     * 입구에서: 그 프레임의 update가 같은 메시지의 IllegalArgumentException을 그대로 던지고, ONNX 세션이 닫히며, 남은 68
     * 프레임과 updateNoFace는 모두 IllegalStateException(cause = 그 오류)이고 lastState와 counters가 그대로다.
     */
    @Test
    fun processingErrorStopsTheMonitorAsTheReference() {
        val v = e2eVector()
        var onnx: ModelFn? = null
        val monitor = DrowsinessMonitor.openBundled(mediapipeSha) { m ->
            onnx = m
            ModelFn { listOf(0.25, 0.5) }
        }
        monitor.use {
            var processed = 0
            var failedAt = -1
            var error: Throwable? = null
            for ((i, frame) in v.frames.withIndex()) {
                try {
                    assertNotNull(feed(monitor, frame, null), "t=${frame.tMs}")
                    processed++
                } catch (e: IllegalArgumentException) {
                    failedAt = i
                    error = e
                    break
                }
            }
            assertEquals(1434, failedAt)
            assertEquals(239001L, v.frames[failedAt].tMs)
            assertEquals(1434, processed)
            assertEquals("model must return 3 probabilities (alert, low_vigilant, drowsy)", error!!.message)
            val c = monitor.counters
            assertEquals(listOf(1435L, 0L, 179L, 0L, 0L), listOf(c.framesAccepted, c.droppedFrames, c.stepsEmitted, c.modelCalls, c.nonfiniteWindows))
            val last = monitor.lastState!!
            assertEquals(v.frames[failedAt - 1].tMs, last.tMs)

            // ONNX 세션이 닫혔다(K8: 해제를 끝낸 뒤 알린다).
            val zero = List(120) { List(17) { 0.0 } }
            val closedModel = assertFailsWith<IllegalStateException> { onnx!!.predict(zero) }
            assertEquals("the model is closed", closedModel.message)

            var refused = 0
            for (frame in v.frames.drop(failedAt + 1)) {
                val e = assertFailsWith<IllegalStateException> { feed(monitor, frame, null) }
                assertSame(error, e.cause)
                assertEquals("the DrowsinessMonitor stopped after a processing error: ${error}", e.message)
                refused++
            }
            val e2 = assertFailsWith<IllegalStateException> { monitor.updateNoFace(10_000_000L, 640, 480, null) }
            assertSame(error, e2.cause)
            assertEquals(68, refused)
            assertSame(last, monitor.lastState)
            assertEquals(c, monitor.counters)
            println(
                "DrowsinessMonitorTest.processingErrorStopsTheMonitorAsTheReference: frame $failedAt t=${v.frames[failedAt].tMs} threw " +
                    "'${error.message}' after $processed states; counters $c; ONNX session closed; the next $refused frames and " +
                    "updateNoFace refused with IllegalStateException '${e2.message}'",
            )
        }
    }

    /**
     * close()가 진행 중인 update와 겹치면 그 update가 끝나기를 기다린 뒤 닫는다(계획 5.8절 close: 하던 프레임을 끝내고 ONNX
     * 세션을 닫는다). 첫 모델 호출(프레임 1434) 안에서 막아 둔 사이에 다른 스레드가 close()를 부르면, close는 돌아오지 않고
     * 기다린다. 풀어 주면 그 update는 확률을 가진 상태를 돌려주고(모델 호출 1), close가 돌아오며, 다음 프레임은
     * "the DrowsinessMonitor is closed"로 거부된다.
     */
    @Test
    fun closeDuringAnUpdateWaitsForItToFinish() {
        val v = e2eVector()
        val inside = CountDownLatch(1)
        val release = CountDownLatch(1)
        val calls = AtomicInteger(0)
        var monitorRef: DrowsinessMonitor? = null
        val results = ArrayList<DriverState?>()
        var refusal: Throwable? = null
        var failure: Throwable? = null
        val worker = Thread {
            try {
                val monitor = DrowsinessMonitor.openBundled(mediapipeSha) { m ->
                    ModelFn { w ->
                        if (calls.getAndIncrement() == 0) {
                            inside.countDown()
                            check(release.await(60, TimeUnit.SECONDS)) { "not released" }
                        }
                        m.predict(w)
                    }
                }
                monitorRef = monitor
                for (frame in v.frames) {
                    try {
                        results.add(feed(monitor, frame, v.speedAt(frame.tMs)))
                    } catch (e: IllegalStateException) {
                        refusal = e
                        break
                    }
                }
            } catch (t: Throwable) {
                failure = t
                inside.countDown()
            }
        }
        worker.start()
        assertTrue(inside.await(120, TimeUnit.SECONDS), "the model was not called")
        failure?.let { throw it }
        val monitor = monitorRef!!
        var closeError: Throwable? = null
        val closer = Thread {
            try {
                monitor.close()
            } catch (t: Throwable) {
                closeError = t
            }
        }
        closer.start()
        closer.join(500)
        val waited = closer.isAlive
        release.countDown()
        closer.join(120_000)
        worker.join(120_000)
        failure?.let { throw it }
        closeError?.let { throw it }
        assertTrue(waited, "close() returned while an update was in progress")
        assertEquals(1435, results.size, "frames 0..1434 returned states")
        val inFlight = results.last()!!
        assertEquals(239001L, inFlight.tMs)
        assertNotNull(inFlight.output.probs, "the update in progress finished its model call")
        assertEquals(1L, inFlight.counters.modelCalls)
        assertEquals("the DrowsinessMonitor is closed", refusal?.message)
        println(
            "DrowsinessMonitorTest.closeDuringAnUpdateWaitsForItToFinish: close() waited while the update at t=239001 was blocked in the " +
                "first model call; that update then returned probs ${inFlight.output.probs} (counters ${inFlight.counters}); the next frame " +
                "was refused with '${refusal?.message}'",
        )
    }

    /** 수 잎의 가장 큰 |a − e| (AllVectorsTest와 같은 셈). */
    private class MaxDiff {
        var max = 0.0
        var at: String? = null
        var numbers = 0

        fun walk(e: Any?, a: Any?, path: String = "") {
            when (e) {
                is Map<*, *> -> if (a is Map<*, *>) for (k in e.keys) walk(e[k], a[k], "$path.$k")
                is List<*> -> if (a is List<*> && a.size == e.size) for (i in e.indices) walk(e[i], a[i], "$path[$i]")
                is Long, is Double -> {
                    val ev = (e as Number).toDouble()
                    val av = (a as? Number)?.toDouble() ?: Double.NaN
                    numbers++
                    val d = Math.abs(av - ev)
                    if (d.isNaN() || d > max) {
                        max = if (d.isNaN()) Double.POSITIVE_INFINITY else d
                        at = path
                    }
                }
                else -> Unit
            }
        }
    }
}
