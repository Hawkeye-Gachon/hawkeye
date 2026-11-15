package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.ClosureRun
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.Point
import hawkeye.drowsy.core.StepFeatures
import hawkeye.drowsy.core.spec.FeatureSpec
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorFile
import java.io.File

// 시험 도우미: Kotlin 값 ↔ 벡터·자료 파일의 JSON 나무(Python 필드 이름).
// 출력 쪽은 drowsy/vectors.py:433-441 `_signals_to_json`, `_state_to_json`, 412-415 `baseline_to_json`과 같은 키와 형이다
// (dataclasses.asdict: 정수는 Long, 실수는 Double, None은 null, 튜플은 목록). 비교는 VectorComparator가 한다.
internal object JsonTrees {

    /** export/feature_spec_v0.json (배포 spec). 벡터의 0단계와 실행기가 쓴다. */
    val shippedSpec: FeatureSpec by lazy { SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes()) }

    /** export/rules.json 바이트의 sha256 (벡터의 rules_hash와 비교한다). */
    val shippedRulesSha256: String by lazy { TestData.sha256Hex(File(TestData.export, "rules.json").readBytes()) }

    /** vectors.py:433-436 `_signals_to_json`: FrameSignals의 필드 전부, 필드 순서대로. */
    fun signals(s: FrameSignals): Map<String, Any?> = linkedMapOf(
        "t_ms" to s.tMs,
        "face" to s.face,
        "ear_l" to s.earL,
        "ear_r" to s.earR,
        "eye_w_l" to s.eyeWL,
        "eye_w_r" to s.eyeWR,
        "blink_l" to s.blinkL,
        "blink_r" to s.blinkR,
        "eye_look" to ArrayList<Any?>(s.eyeLook),
        "jaw_open" to s.jawOpen,
        "yaw" to s.yaw,
        "pitch" to s.pitch,
        "roll" to s.roll,
        "iris_h_l" to s.irisHL,
        "iris_h_r" to s.irisHR,
        "iris_v_l" to s.irisVL,
        "iris_v_r" to s.irisVR,
    )

    /** vectors.py:439-440 `_state_to_json`. */
    fun state(st: FrameState): Map<String, Any?> = linkedMapOf(
        "t_ms" to st.tMs,
        "face" to st.face,
        "closure" to st.closure,
        "closure_valid" to st.closureValid,
        "provisional" to st.provisional,
        "yaw_rel" to st.yawRel,
        "pitch_rel" to st.pitchRel,
        "pose_valid" to st.poseValid,
        "gaze_h_rel" to st.gazeHRel,
        "gaze_v_rel" to st.gazeVRel,
        "jaw_rel" to st.jawRel,
    )

    /** vectors.py:412-415 `baseline_to_json`. */
    fun baseline(b: Baseline): LinkedHashMap<String, Any?> = linkedMapOf(
        "e_l" to b.eL,
        "e_r" to b.eR,
        "b_l" to b.bL,
        "b_r" to b.bR,
        "r0" to ArrayList<Any?>(b.r0),
        "jaw0" to b.jaw0,
        "gaze_h0" to b.gazeH0,
        "gaze_v0" to b.gazeV0,
        "t_complete_ms" to b.tCompleteMs,
        "status" to b.status.wire,
    )

    /** vectors.py:443-444 `_run_to_json`: {"t_ms", ClosureRun의 필드}. tMs가 null이면 t_ms를 넣지 않는다. */
    fun run(tMs: Long?, r: ClosureRun): Map<String, Any?> {
        val m = LinkedHashMap<String, Any?>()
        if (tMs != null) m["t_ms"] = tMs
        m["start_ms"] = r.startMs
        m["end_ms"] = r.endMs
        m["duration_ms"] = r.durationMs
        m["amplitude"] = r.amplitude
        m["peak_closing_velocity"] = r.peakClosingVelocity
        m["peak_opening_velocity"] = r.peakOpeningVelocity
        m["velocity_valid"] = r.velocityValid
        m["truncated"] = r.truncated
        return m
    }

    /** vectors.py:447-454 `_step_to_json`: {"t_ms", k, t_end_ms, full_lookback, values}. tMs가 null이면 t_ms를 넣지 않는다. */
    fun step(tMs: Long?, s: StepFeatures): Map<String, Any?> {
        val m = LinkedHashMap<String, Any?>()
        if (tMs != null) m["t_ms"] = tMs
        m["k"] = s.k.toLong()
        m["t_end_ms"] = s.tEndMs
        m["full_lookback"] = s.fullLookback
        m["values"] = ArrayList<Any?>(s.values)
        return m
    }

    /** 자료 파일의 FrameState 객체(Python 필드 이름, FORMATS.md 2.7절) → FrameState. */
    fun frameState(v: Any?, where: String): FrameState {
        val m = VectorFile.obj(v, where)
        fun d(key: String): Double = VectorFile.num(m[key], "$where.$key")
        fun b(key: String): Boolean = VectorFile.bool(m[key], "$where.$key")
        return FrameState(
            tMs = VectorFile.long(m["t_ms"], "$where.t_ms"),
            face = b("face"),
            closure = d("closure"),
            closureValid = b("closure_valid"),
            provisional = b("provisional"),
            yawRel = d("yaw_rel"),
            pitchRel = d("pitch_rel"),
            poseValid = b("pose_valid"),
            gazeHRel = d("gaze_h_rel"),
            gazeVRel = d("gaze_v_rel"),
            jawRel = d("jaw_rel"),
        )
    }

    /**
     * 부품 자료의 감김 구간(FORMATS.md 2.7절 C-detector): null 또는 [start_ms, end_ms, duration_ms, amplitude,
     * peak_closing_velocity, peak_opening_velocity, velocity_valid, truncated].
     */
    fun runOrNull(v: Any?, where: String): ClosureRun? {
        if (v == null) return null
        val l = VectorFile.list(v, where)
        require(l.size == 8) { "$where must have 8 fields" }
        return ClosureRun(
            startMs = VectorFile.long(l[0], "$where[0]"),
            endMs = VectorFile.long(l[1], "$where[1]"),
            durationMs = VectorFile.long(l[2], "$where[2]"),
            amplitude = VectorFile.num(l[3], "$where[3]"),
            peakClosingVelocity = VectorFile.num(l[4], "$where[4]"),
            peakOpeningVelocity = VectorFile.num(l[5], "$where[5]"),
            velocityValid = VectorFile.bool(l[6], "$where[6]"),
            truncated = VectorFile.bool(l[7], "$where[7]"),
        )
    }

    /** 자료 파일의 FrameSignals 객체(Python 필드 이름, FORMATS.md 2.7절) → FrameSignals. */
    fun frameSignals(v: Any?, where: String): FrameSignals {
        val m = VectorFile.obj(v, where)
        fun d(key: String): Double = VectorFile.num(m[key], "$where.$key")
        fun opt(key: String): Double? = m[key]?.let { VectorFile.num(it, "$where.$key") }
        return FrameSignals(
            tMs = VectorFile.long(m["t_ms"], "$where.t_ms"),
            face = VectorFile.bool(m["face"], "$where.face"),
            earL = d("ear_l"),
            earR = d("ear_r"),
            eyeWL = d("eye_w_l"),
            eyeWR = d("eye_w_r"),
            blinkL = d("blink_l"),
            blinkR = d("blink_r"),
            eyeLook = VectorFile.list(m["eye_look"], "$where.eye_look").mapIndexed { i, x -> VectorFile.num(x, "$where.eye_look[$i]") },
            jawOpen = d("jaw_open"),
            yaw = d("yaw"),
            pitch = d("pitch"),
            roll = d("roll"),
            irisHL = opt("iris_h_l"),
            irisHR = opt("iris_h_r"),
            irisVL = opt("iris_v_l"),
            irisVR = opt("iris_v_r"),
        )
    }

    fun doubles(v: Any?, where: String): List<Double> =
        VectorFile.list(v, where).mapIndexed { i, x -> VectorFile.num(x, "$where[$i]") }

    fun point(v: Any?, where: String): Point {
        val p = doubles(v, where)
        require(p.size == 2) { "$where must be [x, y]" }
        return Point(p[0], p[1])
    }

    /** 수로 같음(`==`, -0.0 == 0.0), 그리고 둘 다 NaN (FORMATS.md 2.7절). */
    fun numericallyEqual(a: Double, e: Double): Boolean = a == e || (a.isNaN() && e.isNaN())
}
