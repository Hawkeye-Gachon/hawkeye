package hawkeye.drowsy.core.diff

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.BaselineStatus
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.GazeZone
import hawkeye.drowsy.core.Phase
import hawkeye.drowsy.core.Point
import hawkeye.drowsy.core.RawFrame
import hawkeye.drowsy.core.Source
import hawkeye.drowsy.core.internal.FrameSignals
import hawkeye.drowsy.core.spec.PipelineSpec
import hawkeye.drowsy.core.spec.RulesConfig
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testkit.Hkdiff1
import hawkeye.drowsy.core.testkit.VectorFile

/**
 * HKDIFF1 세션 파일 하나 (kotlin/difftools/FORMATS.md 1.2~1.6절): 세 모드의 입력(RAW, SIG, STATE)과 참조 출력 열.
 * 열 이름과 뜻은 FORMATS.md 1.3절 그대로다. 없는 묶음(RAW 입력, 신호, 기준선, 대체 모델)은 null이다.
 * 모든 기대값은 파일에서 온다(바꾸지 않은 Python 참조가 낸 값).
 */
internal class DiffSession(val h: Hkdiff1) {
    val header: Map<String, Any?> = h.header
    val sid: String = header["session_id"] as String
    val modes: List<String> = VectorFile.list(header["modes"], "modes").map { it as String }
    val labels: Map<String, Any?> = VectorFile.obj(header["labels"], "labels")

    val spec: PipelineSpec = SpecLoader.parsePipelineSpec((header["pipeline_json"] as String).toByteArray(Charsets.UTF_8))
    val rules: RulesConfig = SpecLoader.parseRules((header["rules_json"] as String).toByteArray(Charsets.UTF_8))
    val windowSteps: Int = VectorFile.int(header["window_steps"], "window_steps")

    // ----- 열거형 표 (FORMATS.md 0.4절, 머리의 enums) -----
    private val enums = VectorFile.obj(header["enums"], "enums")
    val eventTypes: List<EventType> = VectorFile.list(enums["event_types"], "event_types").map { EventType.fromWire(it as String) }
    val phases: List<Phase> = VectorFile.list(enums["phases"], "phases").map { Phase.fromWire(it as String) }
    val sources: List<Source> = VectorFile.list(enums["sources"], "sources").map { Source.fromWire(it as String) }
    val statuses: List<BaselineStatus> =
        VectorFile.list(enums["baseline_statuses"], "baseline_statuses").map { BaselineStatus.fromWire(it as String) }
    val zones: List<GazeZone> = VectorFile.list(enums["zones"], "zones").map { GazeZone.fromWire(it as String) }

    // ----- 넣은 프레임 (fed) -----
    val tMs: LongArray = h.i64("t_ms")
    val n: Int = tMs.size
    val accepted: IntArray = h.u8("accepted")
    private val speed = h.f64("speed")
    private val speedNull = h.u8("speed_null")
    fun speedAt(i: Int): Double? = if (speedNull[i] == 1) null else speed[i]

    // ----- 생성자 인자 -----
    val baselineInjected: Baseline? = header["baseline_injected"]?.let { baselineFromHex(it) }
    val featureStartMs: Long? = header["feature_start_ms"] as Long?
    val constructorError: Map<String, Any?>? = header["constructor_error"]?.let { VectorFile.obj(it, "constructor_error") }
    val exception: Map<String, Any?>? = header["exception"]?.let { VectorFile.obj(it, "exception") }
    val stubProbs: DoubleArray? = if (header["model"] != null) h.f64("stub_probs") else null
    val stubCount: Int = header["model"]?.let { VectorFile.int(VectorFile.obj(it, "model")["count"], "model.count") } ?: 0

    private fun baselineFromHex(v: Any?): Baseline {
        val m = VectorFile.obj(v, "baseline")
        fun x(key: String) = java.lang.Double.parseDouble(m[key] as String)
        return Baseline(
            eL = x("e_l"), eR = x("e_r"), bL = x("b_l"), bR = x("b_r"),
            r0 = VectorFile.list(m["r0"], "r0").map { java.lang.Double.parseDouble(it as String) },
            jaw0 = x("jaw0"), gazeH0 = x("gaze_h0"), gazeV0 = x("gaze_v0"),
            tCompleteMs = VectorFile.long(m["t_complete_ms"], "t_complete_ms"),
            status = BaselineStatus.fromWire(m["status"] as String),
        )
    }

    // ----- RAW 입력 (FORMATS.md 1.5절) -----
    private val landmarkIndices = VectorFile.list(header["landmark_indices"], "landmark_indices").map { VectorFile.int(it, "i") }
    private val blendshapeNames = VectorFile.list(header["blendshape_names"], "blendshape_names").map { it as String }

    fun rawFrames(): List<RawFrame> {
        val face = h.u8("face")
        val w = h.i32("w")
        val hh = h.i32("h")
        val lm = h.f64("lm")
        val bs = h.f64("bs")
        val m = h.f64("m")
        val out = ArrayList<RawFrame>(n)
        var f = 0
        for (i in 0 until n) {
            if (face[i] == 1) {
                val lmMap = LinkedHashMap<Int, Point>()
                for ((j, idx) in landmarkIndices.withIndex()) {
                    val o = (f * landmarkIndices.size + j) * 2
                    lmMap[idx] = Point(lm[o], lm[o + 1])
                }
                val bsMap = LinkedHashMap<String, Double>()
                for ((j, name) in blendshapeNames.withIndex()) bsMap[name] = bs[f * blendshapeNames.size + j]
                val mat = (0 until 16).map { m[f * 16 + it] }
                out.add(RawFrame(tMs = tMs[i], face = true, w = w[i], h = hh[i], lm = lmMap, bs = bsMap, m = mat))
                f++
            } else {
                out.add(RawFrame(tMs = tMs[i], face = false, w = w[i], h = hh[i], lm = emptyMap(), bs = emptyMap(), m = emptyList()))
            }
        }
        return out
    }

    // ----- 신호 (SIG 입력, RAW 기대값; 넣은 프레임마다) -----
    val hasSignals: Boolean = h.columns.containsKey("sig_face")

    fun signals(): List<FrameSignals> {
        val face = h.u8("sig_face")
        val earL = h.f64("ear_l"); val earR = h.f64("ear_r"); val eyeWL = h.f64("eye_w_l"); val eyeWR = h.f64("eye_w_r")
        val blinkL = h.f64("blink_l"); val blinkR = h.f64("blink_r"); val jaw = h.f64("jaw_open")
        val yaw = h.f64("yaw"); val pitch = h.f64("pitch"); val roll = h.f64("roll")
        val eyeLook = h.f64("eye_look")
        val iris = h.f64("iris")
        val irisNull = h.u8("iris_null")
        fun irisAt(i: Int, k: Int): Double? = if (irisNull[i * 4 + k] == 1) null else iris[i * 4 + k]
        return (0 until n).map { i ->
            FrameSignals(
                tMs = tMs[i], face = face[i] == 1, earL = earL[i], earR = earR[i], eyeWL = eyeWL[i], eyeWR = eyeWR[i],
                blinkL = blinkL[i], blinkR = blinkR[i], eyeLook = (0 until 8).map { eyeLook[i * 8 + it] }, jawOpen = jaw[i],
                yaw = yaw[i], pitch = pitch[i], roll = roll[i],
                irisHL = irisAt(i, 0), irisHR = irisAt(i, 1), irisVL = irisAt(i, 2), irisVR = irisAt(i, 3),
            )
        }
    }

    // ----- 상태 (받아들인 행마다; STATE 입력, RAW·SIG 기대값) -----
    val closure: DoubleArray = h.f64("closure")
    val yawRel: DoubleArray = h.f64("yaw_rel")
    val pitchRel: DoubleArray = h.f64("pitch_rel")
    val gazeHRel: DoubleArray = h.f64("gaze_h_rel")
    val gazeVRel: DoubleArray = h.f64("gaze_v_rel")
    val jawRel: DoubleArray = h.f64("jaw_rel")
    val stateFlags: IntArray = h.u8("state_flags")
    val nRows: Int = stateFlags.size

    /** 받아들인 행 j의 FrameState. tMs = 받아들인 j번째 넣은 프레임의 t_ms (FORMATS.md 1.3절 끝). */
    fun states(): List<FrameState> {
        val acceptedT = (0 until n).filter { accepted[it] != 0 }.map { tMs[it] }
        return (0 until nRows).map { j ->
            val fl = stateFlags[j]
            FrameState(
                tMs = acceptedT[j], face = fl and 1 != 0, closure = closure[j], closureValid = fl and 2 != 0,
                provisional = fl and 4 != 0, yawRel = yawRel[j], pitchRel = pitchRel[j], poseValid = fl and 8 != 0,
                gazeHRel = gazeHRel[j], gazeVRel = gazeVRel[j], jawRel = jawRel[j],
            )
        }
    }

    // ----- 덧붙임 -----
    val gazeZone: IntArray = h.u8("gaze_zone")
    val activeMask: IntArray = h.u8("active_mask")
    val ongoingFeatureMs: LongArray = h.i64("ongoing_feature_ms")

    // ----- 기준선 (스스로 보정한 세션만) -----
    val blFrame: Int? = if (h.columns.containsKey("bl_frame")) h.u32("bl_frame")[0].toInt() else null
    val blF64: DoubleArray? = if (blFrame != null) h.f64("bl_f64") else null
    val blTCompleteMs: Long? = if (blFrame != null) h.i64("bl_t_complete_ms")[0] else null
    val blStatus: BaselineStatus? = if (blFrame != null) statuses[h.u8("bl_status")[0]] else null
    val dgNFace: Long? = if (h.columns.containsKey("dg_n_face")) h.i64("dg_n_face")[0] else null
    val dgF64: DoubleArray? = if (dgNFace != null) h.f64("dg_f64") else null

    /** bl_f64 열(e_l, e_r, b_l, b_r, r0[0..8], jaw0, gaze_h0, gaze_v0)과 bl_t_complete_ms, bl_status로 만든 기준선. */
    val blBaseline: Baseline? = blF64?.let { v ->
        Baseline(
            eL = v[0], eR = v[1], bL = v[2], bR = v[3], r0 = (4 until 13).map { v[it] }, jaw0 = v[13], gazeH0 = v[14],
            gazeV0 = v[15], tCompleteMs = blTCompleteMs!!, status = blStatus!!,
        )
    }

    // ----- 출력 열 -----
    val runFrame: LongArray = h.u32("run_frame")
    val runTimes: LongArray = h.i64("run_times")
    val runValues: DoubleArray = h.f64("run_values")
    val runFlags: IntArray = h.u8("run_flags")

    val stFrame: LongArray = h.u32("st_frame")
    val stK: LongArray = h.i64("st_k")
    val stTEndMs: LongArray = h.i64("st_t_end_ms")
    val stFull: IntArray = h.u8("st_full")
    val stValues: DoubleArray = h.f64("st_values")
    val stProbNull: IntArray = h.u8("st_prob_null")
    val stProbs: DoubleArray = h.f64("st_probs")

    val mwFrame: LongArray = h.u32("mw_frame")
    val mwK: LongArray = h.i64("mw_k")
    val mwSha: IntArray = h.u8("mw_sha")
    val mwfullCall: LongArray = h.u32("mwfull_call")
    val mwfull: DoubleArray = h.f64("mwfull")

    class Events(h: Hkdiff1, p: String) {
        val frame: LongArray = h.u32("${p}_frame")
        val type: IntArray = h.u8("${p}_type")
        val severity: IntArray = h.u8("${p}_severity")
        val times: LongArray = h.i64("${p}_times")
        val confidence: DoubleArray = h.f64("${p}_confidence")
        val source: IntArray = h.u8("${p}_source")
        val phase: IntArray = h.u8("${p}_phase")
        val provisional: IntArray = h.u8("${p}_provisional")
    }

    val events = Events(h, "ev")
    val alerts = Events(h, "al")

    val counters: Map<String, Any?> = VectorFile.obj(header["counters"], "counters")
    val activeAtEnd: List<Map<String, Any?>> = VectorFile.list(header["active_at_end"], "active_at_end").map { VectorFile.obj(it, "active") }
    val nAcceptedRows: Int = VectorFile.int(header["n_accepted_rows"], "n_accepted_rows")
}
