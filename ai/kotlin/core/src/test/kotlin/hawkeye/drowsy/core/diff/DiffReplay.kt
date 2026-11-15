package hawkeye.drowsy.core.diff

import hawkeye.drowsy.core.ClosureRun
import hawkeye.drowsy.core.DriverPipeline
import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.DrowsyInternalApi
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.ModelFn
import hawkeye.drowsy.core.N_FEATURES
import hawkeye.drowsy.core.PipelineOutput
import hawkeye.drowsy.core.Source
import hawkeye.drowsy.core.diff.DiffTally.Tol
import hawkeye.drowsy.core.internal.FrameSignals
import hawkeye.drowsy.core.internal.compute
import hawkeye.drowsy.core.session.ActiveEvent
import hawkeye.drowsy.core.session.ActiveFold
import hawkeye.drowsy.core.session.DriveSession
import hawkeye.drowsy.core.session.DriverState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * 세션 파일 하나를 한 모드로 Kotlin에 다시 넣고 파일의 참조 출력과 비교한다 (계획 9.2.1절, 9.2.4절; FORMATS.md 1.5절).
 *
 * - RAW: 열에서 RawFrame을 다시 만들어 `DriveSession.feed`로 넣는다(리드 판정 10: 자료가 허락하는 곳은 DriveSession을
 *   지난다. `active`가 DriverState에서 나온다). 신호는 `signals.compute`를 따로 불러 sig 열과 비교한다(참조 내보내기가
 *   같은 일을 한다, hk.py run_session). 파이프라인은 기준선 주입 생성자로 만든다(세션 머리의 baseline_injected).
 * - SIG: 신호 열로 `DriverPipeline.updateFromSignals`. `active`는 같은 ActiveFold로 접는다.
 * - STATE: 받아들인 프레임에는 다음 상태 행을, 버려진 프레임에는 직전 행을 그 t_ms로 넣는다(FORMATS.md 1.5절).
 *   스스로 보정한 세션은 bl_frame 행에서 bl_* 기준선을 `updateFromState`에 준다.
 *
 * 파일의 대체 모델(stub_probs)은 호출 번호 i에 `min(i, count - 1)` 행을 돌려준다. 모델에 넘어온 창을 기록해
 * mw_sha(f64 LE 바이트의 sha256), mwfull과 비교한다.
 *
 * 알려진 차이(리드 판정 8(e)). 두 원인만 센다. 그 앞의 모든 행은 그대로 비교한다. 그 밖의 모든 차이는 실패다.
 * - [CAUSE_ROUND_HALF_UP]: 참조가 `drowsy/blink.py:339 _round_half_up`에서 ValueError를 낸 프레임에서 Kotlin이 예외를 내지
 *   않는 것(판정 4, 소유자 결정). [isRoundHalfUpRaise]의 조건이 모두 맞을 때만.
 * - [CAUSE_NAN_BASELINE]: 참조가 예외를 내지 않은 프레임에서 Kotlin의 기준선 추정기 정렬이 NaN 표본 때문에 던지는 것
 *   (판정 8(b)). [isNanBaselineThrow]의 조건이 모두 맞을 때만. 다른 예외는 모두 차이다.
 */
@OptIn(DrowsyInternalApi::class)
internal object DiffReplay {
    // 가정(A2): JVM StrictMath와 CPython의 atan2, hypot, sin, cos는 마지막 몇 자리만 다르다. 그래서 RAW·SIG의 이 좁은
    // 허용 오차(계획 9.2.4절)가 맞다. 넘으면 차이로 적는다(STATE에서는 DiffCmp가 모든 값을 수로 같게 본다).
    private val ABS9 = Tol(1e-9, 0.0)
    private val ABS12 = Tol(1e-12, 0.0)
    private val REL9 = Tol(0.0, 1e-9)
    private val STEP = Tol(1e-9, 1e-9)

    // 계획 9.2.4절: 스텝 특징 가운데 개수(#9 long_closure_count_60, #12 yawn_count_60, #14 nod_count_60)와
    // ms(#10 max_closure_ms_60, #11 max_closure_ms_10)는 RAW·SIG에서도 정확히 같다.
    private val EXACT_STEP_FEATURES = setOf(9, 10, 11, 12, 14)

    const val CAUSE_ROUND_HALF_UP = "roundHalfUp of a non-finite value (reference raises, Kotlin does not; rulings 4, 8(e))"
    const val CAUSE_NAN_BASELINE = "NaN in baseline samples (rulings 8(b), 8(e))"

    /** 파일의 대체 모델. 창을 기록한다. */
    private class StubModel(private val table: DoubleArray, private val count: Int) : ModelFn {
        val windows = ArrayList<List<List<Double>>>()
        override fun predict(window: List<List<Double>>): List<Double> {
            val row = minOf(windows.size, count - 1)
            windows.add(window)
            return listOf(table[row * 3], table[row * 3 + 1], table[row * 3 + 2])
        }
    }

    private fun windowSha(w: List<List<Double>>): ByteArray {
        val buf = ByteBuffer.allocate(w.size * N_FEATURES * 8).order(ByteOrder.LITTLE_ENDIAN)
        for (row in w) for (v in row) buf.putDouble(v)
        return MessageDigest.getInstance("SHA-256").digest(buf.array())
    }

    /** java.util.TimSort가 비교자의 모순을 보았을 때의 메시지(JDK 17 TimSort.mergeLo/mergeHi). */
    const val TIMSORT_MESSAGE = "Comparison method violates its general contract!"

    /**
     * 참조의 예외(FORMATS.md 1.6절의 exception)가 판정 4의 roundHalfUp 차이인지.
     *
     * 가정: 아래 조건이 모두 맞으면 참조는 유한하지 않은 감김 값이 감김 검출기의 보간(blink.py:296-305 → 339)에 닿아
     * 예외를 냈고, Kotlin roundHalfUp(판정 4)은 같은 자리에서 예외를 내지 않는다. 조건:
     * - 종류 ValueError, 자리 `drowsy/blink.py:339 _round_half_up`, 메시지 `cannot convert float NaN to integer`;
     * - 그 프레임까지 넣은 상태 행(fedStates) 가운데 closure_valid인데 closure가 유한하지 않은 행이 있다(검출기는 유효한
     *   행의 closure만 읽는다, blink.py:204-215). 다른 칸(yaw_rel, gaze, jaw)의 유한하지 않은 값만으로는 세지 않는다.
     */
    fun isRoundHalfUpRaise(ex: Map<String, Any?>, fedStates: List<FrameState>): Boolean =
        ex["type"] == "ValueError" && ex["where"] == "drowsy/blink.py:339 _round_half_up" &&
            ex["message"] == "cannot convert float NaN to integer" &&
            fedStates.any { it.closureValid && !it.closure.isFinite() }

    /** 기준선 추정기가 표본으로 모으는 칸(BaselineEstimator.collect, calibrate.py:216-228)에 유한하지 않은 값이 있는지. */
    private fun nonFiniteSample(s: FrameSignals, minEyeWidthPx: Double): Boolean {
        if (!s.face) return false
        if (s.eyeWL >= minEyeWidthPx && !s.earL.isFinite()) return true
        if (s.eyeWR >= minEyeWidthPx && !s.earR.isFinite()) return true
        if (!s.blinkL.isFinite() || !s.blinkR.isFinite() || !s.jawOpen.isFinite()) return true
        if (!s.yaw.isFinite() || !s.pitch.isFinite() || !s.roll.isFinite()) return true
        return s.eyeLook.any { !it.isFinite() }
    }

    /**
     * 참조가 예외를 내지 않은 프레임에서 Kotlin이 던진 것이 판정 8(b)의 차이(NaN이 든 기준선 표본 목록의 정렬)인지.
     *
     * 가정: 아래 조건이 모두 맞으면 그 예외는 NaN 표본 때문에 `<`만 쓰는 비교자가 모순되어 TimSort가 던진 것이다(Calibrate.kt의
     * NaN 가정, 판정 8(b)). Python의 sorted는 같은 입력에서 예외를 내지 않는다. 조건:
     * - IllegalArgumentException이고 메시지가 [TIMSORT_MESSAGE];
     * - 쌓기 기록에서 java.util.TimSort가 BaselineEstimator.tryComplete 위에 있다(정렬은 추정기의 마감에서만 돈다);
     * - 이 프레임 전에 추정기에 들어간 신호(sampledBefore: 받아들여진 프레임만, 이 프레임 제외. 추정기는 마감 검사 뒤에
     *   이 프레임을 모은다)의 표본 칸에 유한하지 않은 값이 있다;
     * - Kotlin도 참조도 이 프레임 전에는 기준선이 없었다.
     */
    fun isNanBaselineThrow(
        raised: Throwable,
        sampledBefore: List<FrameSignals>,
        minEyeWidthPx: Double,
        kotlinHadBaseline: Boolean,
        pythonHadBaseline: Boolean,
    ): Boolean {
        if (raised !is IllegalArgumentException || raised.message != TIMSORT_MESSAGE) return false
        if (kotlinHadBaseline || pythonHadBaseline) return false
        val st = raised.stackTrace
        val sortAt = st.indexOfFirst { it.className == "java.util.TimSort" }
        val estimatorAt = st.indexOfFirst {
            it.className == "hawkeye.drowsy.core.internal.BaselineEstimator" && it.methodName == "tryComplete"
        }
        if (sortAt < 0 || estimatorAt < 0 || estimatorAt < sortAt) return false
        return sampledBefore.any { nonFiniteSample(it, minEyeWidthPx) }
    }

    fun run(s: DiffSession, mode: String, t: DiffTally) {
        t.modeRuns[mode] = (t.modeRuns[mode] ?: 0) + 1
        t.framesFed[mode] = (t.framesFed[mode] ?: 0L) + s.n
        if (mode != "STATE" && s.labels[mode] == "near") t.note("$mode sessions labelled near")
        val c = DiffCmp(t, mode)
        c.where = "${s.sid} $mode"
        try {
            replay(s, mode, t, c)
        } catch (e: DiffCmp.Structural) {
            // 이미 차이로 적었다.
        }
    }

    private fun replay(s: DiffSession, mode: String, t: DiffTally, c: DiffCmp) {
        val fAccepted = c.f("fed.accepted")
        val fCtor = c.f("constructor_error")
        val fRaise = c.f("exception")

        val model = s.stubProbs?.let { StubModel(it, s.stubCount) }
        val pipeline: DriverPipeline
        try {
            pipeline = DriverPipeline(
                spec = s.spec,
                rules = s.rules,
                baseline = s.baselineInjected,
                featureStartMs = s.featureStartMs,
                model = model,
            )
        } catch (e: IllegalArgumentException) {
            val ce = s.constructorError ?: c.structural(fCtor, "IllegalArgumentException: ${e.message}", null)
            if (ce["type"] != "ValueError" || ce["message"] != e.message) c.structural(fCtor, e.message, ce)
            c.count(fCtor, 1, 1)
            return
        }
        if (s.constructorError != null) c.structural(fCtor, null, s.constructorError)

        val drive = if (mode == "RAW") DriveSession(pipeline = pipeline, alertPriority = s.rules.alertPriority) else null
        val fold = if (drive == null) ActiveFold(s.rules.alertPriority) else null
        var active: List<ActiveEvent> = emptyList()

        val raw = if (mode == "RAW") s.rawFrames() else null
        val sig = if (mode == "RAW" || mode == "SIG") s.signals() else null
        val states = if (mode == "STATE") s.states() else null

        // 입력에 유한하지 않은 신호가 있는 세션의 수(보고용. 알려진 차이의 판정은 isNanBaselineThrow가 한다).
        if (sig != null) {
            val nonFiniteSig = sig.any { x ->
                !(x.earL.isFinite() && x.earR.isFinite() && x.eyeWL.isFinite() && x.eyeWR.isFinite() && x.blinkL.isFinite() &&
                    x.blinkR.isFinite() && x.jawOpen.isFinite() && x.yaw.isFinite() && x.pitch.isFinite() && x.roll.isFinite() &&
                    x.eyeLook.all { it.isFinite() })
            }
            t.note("$mode sessions with a non-finite signal input", if (nonFiniteSig) 1 else 0)
        }

        // 칸 (계획 9.2.4절의 규칙; STATE에서는 DiffCmp가 모두 수로 같게 본다)
        val fClosure = c.f("state.closure", ABS9)
        val fYawRel = c.f("state.yaw_rel", ABS9)
        val fPitchRel = c.f("state.pitch_rel", ABS9)
        val fGazeH = c.f("state.gaze_h_rel")
        val fGazeV = c.f("state.gaze_v_rel")
        val fJaw = c.f("state.jaw_rel")
        val fFlags = c.f("state.flags")
        val fZone = c.f("extras.gaze_zone")
        val fMask = c.f("extras.active_mask")
        val fOngoing = c.f("extras.ongoing_feature_ms")
        val fSessionMs = c.f("drive.session_ms")
        val fSpeedNull = c.f("drive.speed_kmh_null")
        val fSpeed = c.f("drive.speed_kmh")
        val fDriveCounters = c.f("drive.counters")
        val fBlHas = c.f("baseline.present")
        val fBlFrame = c.f("baseline.bl_frame")
        val fBlE = c.f("baseline.e_l_e_r", ABS9)
        val fBlB = c.f("baseline.b_l_b_r_jaw0_gaze0")
        val fBlR0 = c.f("baseline.r0", ABS12)
        val fBlT = c.f("baseline.t_complete_ms")
        val fBlStatus = c.f("baseline.status")
        val fDgN = c.f("diagnostics.n_face_frames")
        val fDgQ = c.f("diagnostics.quantile_count")
        val fDgV = c.f("diagnostics.face_ratio_and_quantiles")
        val fRunN = c.f("runs.count")
        val fRunT = c.f("runs.times")
        val fRunAmp = c.f("runs.amplitude", ABS9)
        val fRunVel = c.f("runs.velocities", REL9)
        val fRunFlags = c.f("runs.flags")
        val fStN = c.f("steps.count")
        val fStK = c.f("steps.k_t_end_full")
        val fStExact = c.f("steps.values_exact(#9,#10,#11,#12,#14)")
        val fStOther = c.f("steps.values_other", STEP)
        val fStProbNull = c.f("steps.prob_null")
        val fStProbs = c.f("steps.probs")
        val fMwN = c.f("model.calls_count")
        val fMwK = c.f("model.k")
        val fMwSha = c.f("model.window_sha256")
        val fMwFullExact = c.f("model.window_values_exact")
        val fMwFullOther = c.f("model.window_values_other", STEP)
        val fEvN = c.f("events.count")
        val fEvD = c.f("events.discrete")
        val fEvConf = c.f("events.confidence", ABS12)
        val fAlN = c.f("alerts.count")
        val fAlD = c.f("alerts.discrete")
        val fAlConf = c.f("alerts.confidence", ABS12)
        val fCounters = c.f("counters")
        val fActiveEnd = c.f("active_at_end")
        val fRows = c.f("accepted_rows")

        // mwfull 행: 호출 번호 → 행
        val fullRowOfCall = HashMap<Int, Int>()
        for ((r, call) in s.mwfullCall.withIndex()) fullRowOfCall[call.toInt()] = r

        var runPtr = 0
        var stPtr = 0
        var mwPtr = 0
        var evPtr = 0
        var alPtr = 0
        var j = 0
        var firstAcceptedT: Long? = null
        var hadBaseline = s.baselineInjected != null
        val stateFor = ArrayList<FrameState>()
        val injected = s.baselineInjected != null
        // 추정기에 들어간 신호(받아들여진 프레임, RAW·SIG). isNanBaselineThrow의 입력이다.
        val sampled = ArrayList<FrameSignals>()

        for (i in 0 until s.n) {
            c.where = "${s.sid} $mode fed $i row $j t_ms ${s.tMs[i]}"
            val acc = s.accepted[i]
            val speed = s.speedAt(i)
            if (mode == "STATE") {
                // 가정: 버려질 프레임(accepted 0)은 받아들임 단계(t_ms가 직전 이하)에서 상태를 읽기 전에 버려지므로
                // 어떤 상태를 넣어도 출력이 같다(FORMATS.md 1.5절 "feed any state with that t_ms"). 그래서 직전 행을 그
                // t_ms로 넣는다(첫 프레임은 언제나 받아들여진다).
                val st = if (acc != 0) states!![j] else stateFor.last().copy(tMs = s.tMs[i])
                stateFor.add(st)
            }
            val stepsEmittedBefore = pipeline.stepsEmitted
            val modelCallsBefore = pipeline.modelCalls
            val nonfiniteBefore = pipeline.nonfiniteWindows
            val callsBefore = model?.windows?.size ?: 0
            val activeBefore = active
            val kotlinHadBaseline = pipeline.baseline != null

            val out: PipelineOutput?
            var raised: Throwable? = null
            var o: PipelineOutput? = null
            var ds: DriverState? = null
            try {
                o = when (mode) {
                    "RAW" -> {
                        val d = drive!!.feed(raw!![i], speed)
                        ds = d
                        d?.output
                    }
                    "SIG" -> pipeline.updateFromSignals(sig!![i], speed)
                    else -> {
                        val bl = if (acc != 0 && !injected && s.blFrame != null && j == s.blFrame) s.blBaseline else null
                        pipeline.updateFromState(stateFor[i], bl, speed)
                    }
                }
            } catch (e: Throwable) {
                raised = e
            }
            out = o

            // ----- 신호 (RAW만: SIG에서는 입력이다). signals.compute는 상태가 없으므로 넣은 모든 프레임에서 비교한다
            // (계획 9.2.5절, E9; FORMATS.md 1.3절: 신호 열은 넣은 프레임마다 있다). 버려진 프레임, 예외 프레임도.
            if (mode == "RAW") compareSignals(c, compute(raw!![i], s.spec), sig!![i])

            if (acc == 2) {
                // 참조가 이 프레임에서 예외를 냈다(FORMATS.md 1.6절). 이 행의 Kotlin 출력은 비교하지 않는다.
                val ex = s.exception ?: c.structural(fRaise, "row marked raised", "no exception in the header")
                if ((ex["fed_index"] as Long).toInt() != i) c.structural(fRaise, i, ex["fed_index"])
                if (raised != null) {
                    val same = ex["type"] == "ValueError" && raised is IllegalArgumentException && raised.message == ex["message"]
                    if (!same) c.structural(fRaise, raised.toString(), ex)
                    c.count(fRaise, 1, 1)
                } else {
                    // 넣은 상태 행: STATE는 넣은 그대로, RAW·SIG는 참조가 이 프레임까지 계산한 행(이 프레임의 행 포함).
                    val fedStates = if (mode == "STATE") stateFor else s.states().subList(0, minOf(j + 1, s.nRows))
                    if (!isRoundHalfUpRaise(ex, fedStates)) c.structural(fRaise, "no exception", ex)
                    // 가정: 계수의 자리. 받아들임(accept)은 예외 전에 끝나므로 frames_accepted, dropped_frames는 이 프레임 뒤의
                    // Kotlin 값과 같아야 한다. 스텝 쪽 계수는 참조가 어느 검출기에서 멈췄는지에 따라 이 프레임 앞(파이프라인의
                    // 검출기, pipeline.py:186) 또는 뒤(RuleEngine의 검출기, rules.py:518; pipeline.py:194의 스텝 계수 뒤)의 값이다.
                    // 둘 가운데 어느 것과도 같지 않으면 차이다.
                    val cnt = s.counters
                    c.exact(fCounters, pipeline.framesAccepted, cnt["frames_accepted"] as Long)
                    c.exact(fCounters, pipeline.droppedFrames, cnt["dropped_frames"] as Long)
                    val before = listOf(stepsEmittedBefore, modelCallsBefore, nonfiniteBefore)
                    val after = listOf(pipeline.stepsEmitted, pipeline.modelCalls, pipeline.nonfiniteWindows)
                    val py = listOf(cnt["steps_emitted"] as Long, cnt["model_calls"] as Long, cnt["nonfinite_windows"] as Long)
                    val site = when (py) {
                        before -> "the pipeline's detector"
                        after -> "the rule engine's detector"
                        else -> c.structural(fCounters, "before $before after $after", py)
                    }
                    t.note("raise site: $site")
                    c.count(fCounters, 1, 1)
                    // active_at_end은 마지막으로 끝난 프레임 뒤의 목록이다.
                    compareActiveEnd(c, fActiveEnd, s, activeBefore)
                    c.count(fRows, j + 1, s.nAcceptedRows)
                    t.knownDifference(
                        CAUSE_ROUND_HALF_UP, s.sid,
                        "fed $i t_ms ${s.tMs[i]}: reference ${ex["type"]} '${ex["message"]}' at ${ex["where"]}; Kotlin did not raise; " +
                            "rows before compared; reference stopped inside $site",
                    )
                }
                return
            }

            if (raised != null) {
                val pythonHadBaseline = !injected && s.blFrame != null && s.blFrame < j
                if (mode != "STATE" &&
                    isNanBaselineThrow(raised, sampled, s.spec.signals.minEyeWidthPx, kotlinHadBaseline, pythonHadBaseline)
                ) {
                    t.knownDifference(
                        CAUSE_NAN_BASELINE, s.sid,
                        "$mode fed $i t_ms ${s.tMs[i]}: Kotlin raised $raised in the baseline estimator's sort; reference did not raise; " +
                            "rows before compared",
                    )
                    return
                }
                c.structural(fRaise, raised.toString(), "no exception")
            }
            if ((out == null) != (acc == 0)) c.structural(fAccepted, if (out == null) 0 else 1, acc)
            c.count(fAccepted, 1, 1)
            if (out == null) continue
            if (sig != null) sampled.add(sig[i])
            // RAW: 아래의 gaze_zone, 기준선, 속도, 계수는 DriveSession이 돌려준 DriverState에서 읽는다.
            val dState = if (mode == "RAW") ds ?: c.structural(fAccepted, "no DriverState", "accepted") else null

            // ----- 상태 (RAW, SIG; STATE에서는 입력이다) -----
            val st = out.state
            if (mode != "STATE") {
                c.exact(c.f("state.t_ms"), st.tMs, s.tMs[i])
                c.num(fClosure, st.closure, s.closure[j])
                c.num(fYawRel, st.yawRel, s.yawRel[j])
                c.num(fPitchRel, st.pitchRel, s.pitchRel[j])
                c.num(fGazeH, st.gazeHRel, s.gazeHRel[j])
                c.num(fGazeV, st.gazeVRel, s.gazeVRel[j])
                c.num(fJaw, st.jawRel, s.jawRel[j])
                val flags = (if (st.face) 1 else 0) or (if (st.closureValid) 2 else 0) or (if (st.provisional) 4 else 0) or
                    (if (st.poseValid) 8 else 0)
                c.exact(fFlags, flags.toLong(), s.stateFlags[j].toLong())
            }

            // ----- 덧붙임 -----
            val zone = if (dState != null) dState.gazeZone else pipeline.gazeZone
            c.exact(fZone, zone.ordinal.toLong(), s.zones[s.gazeZone[j]].ordinal.toLong())
            if (dState != null) {
                active = dState.active
                // DriverState.speedKmh는 넣은 속도 그대로다(계획 6.2절).
                val sp = dState.speedKmh
                c.exact(fSpeedNull, sp == null, speed == null)
                if (sp != null && speed != null) c.num(fSpeed, sp, speed)
            }
            if (fold != null) active = fold.apply(out.events, st.tMs)
            var mask = 0
            for (a in active) mask = mask or (1 shl s.eventTypes.indexOf(a.type))
            c.exact(fMask, mask.toLong(), s.activeMask[j].toLong())
            c.exact(fOngoing, pipeline.featureDetectorOngoingMsForTest(), s.ongoingFeatureMs[j])
            if (firstAcceptedT == null) firstAcceptedT = s.tMs[i]
            if (dState != null) c.exact(fSessionMs, dState.sessionMs, s.tMs[i] - firstAcceptedT!!)

            // ----- 기준선 (스스로 보정하는 세션: RAW, SIG에서 Kotlin이 확정한다) -----
            if (!injected) {
                val expectHas = s.blFrame != null && j >= s.blFrame
                val kb = if (dState != null) dState.baseline else pipeline.baseline
                if ((kb != null) != expectHas) c.structural(fBlHas, kb != null, expectHas)
                c.count(fBlHas, 1, 1)
                if (kb != null && !hadBaseline) {
                    hadBaseline = true
                    c.exact(fBlFrame, j.toLong(), s.blFrame!!.toLong())
                    if (mode != "STATE") {
                        val v = s.blF64!!
                        c.num(fBlE, kb.eL, v[0])
                        c.num(fBlE, kb.eR, v[1])
                        c.num(fBlB, kb.bL, v[2])
                        c.num(fBlB, kb.bR, v[3])
                        for (q in 0 until 9) c.num(fBlR0, kb.r0[q], v[4 + q])
                        c.num(fBlB, kb.jaw0, v[13])
                        c.num(fBlB, kb.gazeH0, v[14])
                        c.num(fBlB, kb.gazeV0, v[15])
                        c.exact(fBlT, kb.tCompleteMs, s.blTCompleteMs!!)
                        c.exact(fBlStatus, kb.status.ordinal.toLong(), s.blStatus!!.ordinal.toLong())
                        val d = pipeline.baselineDiagnostics
                        val dn = s.dgNFace
                        if ((d != null) != (dn != null)) c.structural(fDgN, d, dn)
                        if (d != null && dn != null) {
                            val dv = s.dgF64!!
                            val q = (dv.size - 1) / 2
                            c.count(fDgQ, d.blinkLQuantiles.size, q)
                            c.count(fDgQ, d.blinkRQuantiles.size, q)
                            c.exact(fDgN, d.nFaceFrames.toLong(), dn)
                            c.num(fDgV, d.faceRatio, dv[0])
                            for (x in 0 until q) c.num(fDgV, d.blinkLQuantiles[x], dv[1 + x])
                            for (x in 0 until q) c.num(fDgV, d.blinkRQuantiles[x], dv[1 + q + x])
                        }
                    }
                }
            }

            // ----- 감김 구간 -----
            var nr = 0
            while (runPtr + nr < s.runFrame.size && s.runFrame[runPtr + nr] == j.toLong()) nr++
            c.count(fRunN, out.runs.size, nr)
            for ((q, run) in out.runs.withIndex()) compareRun(c, s, runPtr + q, run, fRunT, fRunAmp, fRunVel, fRunFlags)
            runPtr += nr

            // ----- 스텝과 스텝 확률 -----
            var ns = 0
            while (stPtr + ns < s.stFrame.size && s.stFrame[stPtr + ns] == j.toLong()) ns++
            c.count(fStN, out.steps.size, ns)
            for ((q, step) in out.steps.withIndex()) {
                val e = stPtr + q
                c.exact(fStK, step.k.toLong(), s.stK[e])
                c.exact(fStK, step.tEndMs, s.stTEndMs[e])
                c.exact(fStK, step.fullLookback, s.stFull[e] == 1)
                for (x in 0 until N_FEATURES) {
                    c.num(if (x in EXACT_STEP_FEATURES) fStExact else fStOther, step.values[x], s.stValues[e * N_FEATURES + x])
                }
                val pr = out.stepProbs[q]
                c.exact(fStProbNull, pr == null, s.stProbNull[e] == 1)
                if (pr != null && s.stProbNull[e] == 0) {
                    c.num(fStProbs, pr.alert, s.stProbs[3 * e])
                    c.num(fStProbs, pr.lowVigilant, s.stProbs[3 * e + 1])
                    c.num(fStProbs, pr.drowsy, s.stProbs[3 * e + 2])
                }
            }
            // probs는 마지막 스텝의 확률이다(FORMATS.md 1.3절 끝의 유도 규칙).
            val lastProbs = if (out.steps.isEmpty()) null else out.stepProbs.last()
            if (out.probs != lastProbs) c.structural(fStProbs, out.probs, lastProbs)

            // ----- 모델 호출 (확률이 있는 스텝과 차례로 짝이 맞는다, hk.py run_session) -----
            val ksWithProbs = out.steps.indices.filter { out.stepProbs[it] != null }.map { out.steps[it].k.toLong() }
            val newCalls = (model?.windows?.size ?: 0) - callsBefore
            var nm = 0
            while (mwPtr + nm < s.mwFrame.size && s.mwFrame[mwPtr + nm] == j.toLong()) nm++
            c.count(fMwN, newCalls, nm)
            c.count(fMwN, ksWithProbs.size, nm)
            for (q in 0 until nm) {
                val e = mwPtr + q
                val call = callsBefore + q
                val w = model!!.windows[call]
                c.exact(fMwK, ksWithProbs[q], s.mwK[e])
                c.count(fMwN, w.size, s.windowSteps)
                val sha = windowSha(w)
                var shaSame = true
                for (b in 0 until 32) if ((sha[b].toInt() and 0xFF) != s.mwSha[e * 32 + b]) shaSame = false
                if (mode == "STATE") {
                    c.exact(fMwSha, shaSame, true)
                } else {
                    // RAW·SIG: 창의 값은 스텝 값이라 9.2.4절의 스텝 허용 오차 안에서만 같다. sha가 같은 호출 수를 센다.
                    t.note("$mode model windows with the same sha256", if (shaSame) 1 else 0)
                    t.note("$mode model windows", 1)
                }
                val fr = fullRowOfCall[call]
                if (fr != null) {
                    val base = fr * s.windowSteps * N_FEATURES
                    for (r in 0 until s.windowSteps) for (x in 0 until N_FEATURES) {
                        c.num(if (x in EXACT_STEP_FEATURES) fMwFullExact else fMwFullOther, w[r][x], s.mwfull[base + r * N_FEATURES + x])
                    }
                }
            }
            if (dState != null) {
                // DriverState의 계수(이 프레임 뒤의 값). 참조 열에서: 받아들인 수 = 행 수, 버린 수 = 넣은 수 - 받아들인 수
                // (예외 프레임은 세션을 끝내므로 앞에 없다), 스텝 수 = 이 행까지의 스텝 행, 모델 호출 수 = 이 행까지의 mw 행.
                // nonfinite_windows는 프레임마다의 참조 열이 없어 파이프라인의 값과 비교한다(세션 끝에서 참조와 비교한다).
                val dc = dState.counters
                c.exact(fDriveCounters, dc.framesAccepted, (j + 1).toLong())
                c.exact(fDriveCounters, dc.droppedFrames, (i - j).toLong())
                c.exact(fDriveCounters, dc.stepsEmitted, (stPtr + ns).toLong())
                c.exact(fDriveCounters, dc.modelCalls, (mwPtr + nm).toLong())
                c.exact(fDriveCounters, dc.nonfiniteWindows, pipeline.nonfiniteWindows)
            }
            mwPtr += nm
            stPtr += ns

            // ----- 이벤트와 경고 (순서까지) -----
            evPtr = compareEvents(c, s, s.events, evPtr, j, out.events, fEvN, fEvD, fEvConf)
            alPtr = compareEvents(c, s, s.alerts, alPtr, j, out.alerts, fAlN, fAlD, fAlConf)
            j++
        }

        // ----- 세션 끝 -----
        c.where = "${s.sid} $mode end"
        c.count(fRows, j, s.nAcceptedRows)
        c.count(fRunN, runPtr, s.runFrame.size)
        c.count(fStN, stPtr, s.stFrame.size)
        c.count(fMwN, mwPtr, s.mwFrame.size)
        c.count(fEvN, evPtr, s.events.frame.size)
        c.count(fAlN, alPtr, s.alerts.frame.size)
        if (s.exception != null) c.structural(fRaise, "no exception", s.exception)
        val cnt = s.counters
        c.exact(fCounters, pipeline.framesAccepted, cnt["frames_accepted"] as Long)
        c.exact(fCounters, pipeline.droppedFrames, cnt["dropped_frames"] as Long)
        c.exact(fCounters, pipeline.stepsEmitted, cnt["steps_emitted"] as Long)
        c.exact(fCounters, pipeline.modelCalls, cnt["model_calls"] as Long)
        c.exact(fCounters, pipeline.nonfiniteWindows, cnt["nonfinite_windows"] as Long)
        if (drive != null) {
            val dc = drive.counters
            c.exact(fCounters, dc.framesAccepted, cnt["frames_accepted"] as Long)
            c.exact(fCounters, dc.droppedFrames, cnt["dropped_frames"] as Long)
            c.exact(fCounters, dc.stepsEmitted, cnt["steps_emitted"] as Long)
            c.exact(fCounters, dc.modelCalls, cnt["model_calls"] as Long)
            c.exact(fCounters, dc.nonfiniteWindows, cnt["nonfinite_windows"] as Long)
        }
        compareActiveEnd(c, fActiveEnd, s, active)
        if (!injected && s.blFrame == null) {
            if (pipeline.baseline != null) c.structural(fBlHas, true, false)
        }
    }

    /** RAW: Kotlin signals.compute(k)와 참조의 신호 열(p), 계획 9.2.4절의 칸별 규칙. */
    private fun compareSignals(c: DiffCmp, k: FrameSignals, p: FrameSignals) {
        c.exact(c.f("signals.face"), k.face, p.face)
        c.num(c.f("signals.blink_l"), k.blinkL, p.blinkL)
        c.num(c.f("signals.blink_r"), k.blinkR, p.blinkR)
        c.num(c.f("signals.jaw_open"), k.jawOpen, p.jawOpen)
        val fEyeLook = c.f("signals.eye_look")
        for (q in 0 until 8) c.num(fEyeLook, k.eyeLook[q], p.eyeLook[q])
        c.num(c.f("signals.ear_l", ABS9), k.earL, p.earL)
        c.num(c.f("signals.ear_r", ABS9), k.earR, p.earR)
        c.num(c.f("signals.eye_w_l", ABS9), k.eyeWL, p.eyeWL)
        c.num(c.f("signals.eye_w_r", ABS9), k.eyeWR, p.eyeWR)
        c.num(c.f("signals.yaw", ABS9), k.yaw, p.yaw)
        c.num(c.f("signals.pitch", ABS9), k.pitch, p.pitch)
        c.num(c.f("signals.roll", ABS9), k.roll, p.roll)
        val fIris = c.f("signals.iris", ABS9)
        val fIrisNull = c.f("signals.iris_null")
        val ki = listOf(k.irisHL, k.irisHR, k.irisVL, k.irisVR)
        val pi = listOf(p.irisHL, p.irisHR, p.irisVL, p.irisVR)
        for (q in 0 until 4) {
            c.exact(fIrisNull, ki[q] == null, pi[q] == null)
            val kv = ki[q]
            val pv = pi[q]
            if (kv != null && pv != null) c.num(fIris, kv, pv)
        }
    }

    private fun compareRun(
        c: DiffCmp, s: DiffSession, e: Int, run: ClosureRun,
        fT: DiffTally.Field, fAmp: DiffTally.Field, fVel: DiffTally.Field, fFlags: DiffTally.Field,
    ) {
        c.exact(fT, run.startMs, s.runTimes[3 * e])
        c.exact(fT, run.endMs, s.runTimes[3 * e + 1])
        c.exact(fT, run.durationMs, s.runTimes[3 * e + 2])
        c.num(fAmp, run.amplitude, s.runValues[3 * e])
        c.num(fVel, run.peakClosingVelocity, s.runValues[3 * e + 1])
        c.num(fVel, run.peakOpeningVelocity, s.runValues[3 * e + 2])
        val flags = (if (run.velocityValid) 1 else 0) or (if (run.truncated) 2 else 0)
        c.exact(fFlags, flags.toLong(), s.runFlags[e].toLong())
    }

    private fun compareEvents(
        c: DiffCmp, s: DiffSession, ex: DiffSession.Events, ptr: Int, row: Int, got: List<DriverStateEvent>,
        fN: DiffTally.Field, fD: DiffTally.Field, fConf: DiffTally.Field,
    ): Int {
        var n = 0
        while (ptr + n < ex.frame.size && ex.frame[ptr + n] == row.toLong()) n++
        c.count(fN, got.size, n)
        for ((q, ev) in got.withIndex()) {
            val e = ptr + q
            c.exact(fD, ev.type.ordinal.toLong(), s.eventTypes[ex.type[e]].ordinal.toLong())
            c.exact(fD, ev.severity.toLong(), ex.severity[e].toLong())
            c.exact(fD, ev.tStartMs, ex.times[2 * e])
            c.exact(fD, ev.durationMs, ex.times[2 * e + 1])
            c.exact(fD, ev.source.ordinal.toLong(), s.sources[ex.source[e]].ordinal.toLong())
            c.exact(fD, ev.phase.ordinal.toLong(), s.phases[ex.phase[e]].ordinal.toLong())
            c.exact(fD, ev.provisional, ex.provisional[e] == 1)
            c.num(fConf, ev.confidence, ex.confidence[e])
        }
        return ptr + n
    }

    /** 머리의 active_at_end(FORMATS.md 4.1절의 항목, alert_priority 순서)와 마지막 목록. */
    private fun compareActiveEnd(c: DiffCmp, fd: DiffTally.Field, s: DiffSession, got: List<ActiveEvent>) {
        val exp = s.activeAtEnd
        c.count(fd, got.size, exp.size)
        for ((q, a) in got.withIndex()) {
            val m = exp[q]
            c.exact(fd, a.type.ordinal.toLong(), EventType.fromWire(m["type"] as String).ordinal.toLong())
            c.exact(fd, a.severity.toLong(), m["severity"] as Long)
            c.exact(fd, a.tStartMs, m["t_start_ms"] as Long)
            c.exact(fd, a.source.ordinal.toLong(), Source.fromWire(m["source"] as String).ordinal.toLong())
            c.exact(fd, a.provisional, m["provisional"] as Boolean)
            c.exact(fd, a.onsetFrameTMs, m["onset_frame_t_ms"] as Long)
        }
    }
}
