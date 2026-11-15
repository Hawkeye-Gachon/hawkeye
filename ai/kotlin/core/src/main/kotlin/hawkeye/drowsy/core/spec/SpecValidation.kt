package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.CLASS_ORDER
import hawkeye.drowsy.core.DrowsySource
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.FEATURE_NAMES
import hawkeye.drowsy.core.N_FEATURES

/**
 * 설정의 범위 검사 (계획 5.10절의 CK, RD, P, F, R). 원본: drowsy/spec.py:634-817, 883-922.
 *
 * 원본 순서 그대로 모두 옮긴다(R8): Python에서는 0 나눗셈이 예외지만 Kotlin에서는 조용한 무한대이므로, 검사가 빠지면
 * Python이 거부하는 설정으로 조용히 다른 값을 낸다. 메시지는 Python의 영어 문장 그대로다(값의 표기는 PyRepr).
 * Python의 연쇄 비교 `a < b < c`는 `a < b && b < c`다. 정수의 `%`, `//`는 Python의 floor 규칙(Math.floorMod,
 * Math.floorDiv)으로 옮긴다. 검사 순서상 그 자리의 피연산자는 언제나 양수라 결과는 같지만 뜻을 그대로 둔다.
 *
 * 상태가 없다(R19).
 */
internal object SpecValidation {

    /** Python `TRANSFORMS` (spec.py:50-52). */
    val TRANSFORMS: List<String> = listOf("none", "log1p")

    /** Python `STANDARDIZATION_IN_GRAPH` (spec.py:54). */
    const val STANDARDIZATION_IN_GRAPH: String = "in_graph"

    /** spec.py:604-606 `_check`. */
    private fun check(condition: Boolean, message: String) {
        if (!condition) throw SpecException(message)
    }

    /** spec.py:634-636 `_validate_clock` (CK1~CK2). 두 번 부른다: pipeline.clock, rules의 clock. */
    fun validateClock(clock: ClockSpec, where: String) {
        check(clock.stepMs > 0, "$where.step_ms must be > 0")
        check(0 < clock.maxHoldMs && clock.maxHoldMs < clock.resetGapMs, "$where: need 0 < max_hold_ms < reset_gap_ms")
    }

    /**
     * spec.py:639-656 `_validate_run_detector` (RD1~RD7). `clock`은 그 검출기와 함께 쓰이는 시간 규칙(같은 파일의
     * clock 블록)이다. 두 번 부른다: pipeline.run_detector, rules의 closure.detector.
     */
    fun validateRunDetector(det: RunDetectorSpec, clock: ClockSpec, where: String) {
        check(
            0.0 < det.exitThr && det.exitThr < det.enterThr && det.enterThr <= 1.0,
            "$where: need 0 < exit_thr < enter_thr <= 1",
        )
        check(det.minDurationMs >= 0, "$where.min_duration_ms must be >= 0")
        check(det.maxDtMs > 0, "$where.max_dt_ms must be > 0")
        check(det.holdMs >= 0, "$where.hold_ms must be >= 0")
        check(det.maxDtMs <= det.holdMs, "$where: need max_dt_ms <= hold_ms")
        check(
            det.holdMs < clock.resetGapMs,
            "$where: need hold_ms < clock.reset_gap_ms (hold_ms is ${det.holdMs}, reset_gap_ms is ${clock.resetGapMs})",
        )
        check(det.velocityLookbackSamples >= 1, "$where.velocity_lookback_samples must be >= 1")
    }

    /** spec.py:659-789 `_validate_pipeline` (P1~P48). */
    fun validatePipeline(p: PipelineSpec) {
        val lm = p.landmarks
        check(lm.earR.size == 6 && lm.earL.size == 6, "pipeline.landmarks: EAR sets need 6 points each")
        check(lm.irisR.size == 5 && lm.irisL.size == 5, "pipeline.landmarks: iris sets need 5 points each")
        for ((name, contour) in listOf("contour_r" to lm.contourR, "contour_l" to lm.contourL)) {
            check(contour.size == 16 && HashSet(contour).size == 16, "pipeline.landmarks.$name: need 16 distinct points")
        }
        check(HashSet(lm.contourR).containsAll(lm.earR), "pipeline.landmarks: ear_r must be a subset of contour_r")
        check(HashSet(lm.contourL).containsAll(lm.earL), "pipeline.landmarks: ear_l must be a subset of contour_l")
        // EAR의 p1, p4가 눈 꼬리다. 시선 대용치와 눈 폭이 같은 두 점을 쓴다. Python은 두 집합의 같음이다.
        check(
            setOf(lm.cornerOuterR, lm.cornerInnerR) == setOf(lm.earR[0], lm.earR[3]),
            "pipeline.landmarks: right eye corners must be p1 and p4 of ear_r",
        )
        check(
            setOf(lm.cornerOuterL, lm.cornerInnerL) == setOf(lm.earL[0], lm.earL[3]),
            "pipeline.landmarks: left eye corners must be p1 and p4 of ear_l",
        )
        check(lm.faceWidth.size == 2, "pipeline.landmarks.face_width: need 2 points")
        val required = lm.requiredIndices()
        check(required.size == 44, "pipeline.landmarks: expected 44 distinct indices, got ${required.size}")
        check(required[0] >= 0 && required[required.size - 1] <= 477, "pipeline.landmarks: indices must be within 0..477")

        val bs = p.blendshapes
        check(bs.order.size == 52 && HashSet(bs.order).size == 52, "pipeline.blendshapes.order: need 52 distinct names")
        check(bs.eyeLook.size == 8, "pipeline.blendshapes.eye_look: need 8 names")
        val names = bs.requiredNames()
        check(HashSet(names).size == 11, "pipeline.blendshapes: the 11 used names must be distinct")
        check(HashSet(bs.order).containsAll(names), "pipeline.blendshapes: used names must appear in order")

        val clock = p.clock
        validateClock(clock, "pipeline.clock")

        check(p.signals.minEyeWidthPx > 0.0, "pipeline.signals.min_eye_width_px must be > 0")

        val base = p.baseline
        check(
            0 < base.durationMs && base.durationMs <= base.maxDurationMs,
            "pipeline.baseline: need 0 < duration_ms <= max_duration_ms",
        )
        check(base.extendStepMs > 0, "pipeline.baseline.extend_step_ms must be > 0")
        // T_b가 스텝 경계에 놓여야 "T_b 이후에 시작하는 스텝"이 정확히 정해진다.
        check(
            Math.floorMod(base.durationMs, clock.stepMs) == 0L && Math.floorMod(base.extendStepMs, clock.stepMs) == 0L,
            "pipeline.baseline: duration_ms and extend_step_ms must be multiples of clock.step_ms",
        )
        // max_duration_ms - duration_ms: 위에서 0 < duration_ms <= max_duration_ms라 Long 안에서 넘치지 않는다.
        check(
            Math.floorMod(base.maxDurationMs - base.durationMs, base.extendStepMs) == 0L,
            "pipeline.baseline: max_duration_ms must be duration_ms plus a whole number of extensions",
        )
        check(base.minFaceFrames >= 1, "pipeline.baseline.min_face_frames must be >= 1")
        check(0.0 <= base.minFaceRatio && base.minFaceRatio <= 1.0, "pipeline.baseline.min_face_ratio must be within 0..1")
        check(0.0 < base.eMin && base.eMin < base.eMax, "pipeline.baseline: need 0 < e_min < e_max")
        val quantiles = ArrayList<Double>()
        quantiles.add(base.earQuantile)
        quantiles.add(base.blinkQuantile)
        quantiles.addAll(base.blinkRecordQuantiles)
        var allInRange = true
        for (q in quantiles) {
            if (!(0.0 < q && q <= 1.0)) {
                allInRange = false
                break
            }
        }
        check(allInRange, "pipeline.baseline: quantiles must be within (0, 1]")
        check(base.offAxisYawDeg > 0.0, "pipeline.baseline.off_axis_yaw_deg must be > 0")

        val clo = p.closure
        check(0.0 < clo.rho && clo.rho < 1.0, "pipeline.closure.rho must be within (0, 1)")
        check(0.0 < clo.bClosed && clo.bClosed <= 1.0, "pipeline.closure.b_closed must be within (0, 1]")
        check(clo.bSpanMin > 0.0, "pipeline.closure.b_span_min must be > 0")
        check(0.0 <= clo.w && clo.w <= 1.0, "pipeline.closure.w must be within 0..1")
        check(
            0.0 < clo.farEyeRampDeg && clo.farEyeRampDeg <= clo.farEyeZeroYawDeg,
            "pipeline.closure: need 0 < far_eye_ramp_deg <= far_eye_zero_yaw_deg",
        )
        check(clo.validMaxAbsYawDeg > 0.0, "pipeline.closure.valid_max_abs_yaw_deg must be > 0")

        validateRunDetector(p.runDetector, clock, "pipeline.run_detector")

        val sf = p.stepFeatures
        check(sf.names == FEATURE_NAMES, "pipeline.step_features.names must equal drowsy.types.FEATURE_NAMES")
        check(sf.windowSteps >= 1, "pipeline.step_features.window_steps must be >= 1")
        check(
            0 < sf.lookbackShortMs && sf.lookbackShortMs < sf.lookbackLongMs,
            "pipeline.step_features: need 0 < lookback_short_ms < lookback_long_ms",
        )
        check(
            Math.floorMod(sf.lookbackShortMs, clock.stepMs) == 0L && Math.floorMod(sf.lookbackLongMs, clock.stepMs) == 0L,
            "pipeline.step_features: windows must be multiples of clock.step_ms",
        )
        check(0.0 < sf.perclosThr && sf.perclosThr <= 1.0, "pipeline.step_features.perclos_thr must be within (0, 1]")
        for ((name, value) in listOf(
            "perclos_min_valid_ms_60" to sf.perclosMinValidMs60,
            "perclos_min_valid_ms_10" to sf.perclosMinValidMs10,
            "blink_rate_min_valid_ms" to sf.blinkRateMinValidMs,
            "blink_duration_cap_ms" to sf.blinkDurationCapMs,
            "long_closure_min_ms" to sf.longClosureMinMs,
            "max_closure_cap_ms" to sf.maxClosureCapMs,
            "pitch_std_min_face_ms" to sf.pitchStdMinFaceMs,
        )) {
            check(value > 0, "pipeline.step_features.$name must be > 0")
        }

        check(0.0 < p.yawn.exitThr && p.yawn.exitThr < p.yawn.enterThr, "pipeline.yawn: need 0 < exit_thr < enter_thr")
        check(p.yawn.countAtMs > 0 && p.yawn.cancelGapMs > 0, "pipeline.yawn: times must be > 0")
        check(0.0 < p.nod.exitDeg && p.nod.exitDeg < p.nod.enterDeg, "pipeline.nod: need 0 < exit_deg < enter_deg")
        check(
            0 < p.nod.minDurationMs && p.nod.minDurationMs <= p.nod.maxDurationMs,
            "pipeline.nod: need 0 < min_duration_ms <= max_duration_ms",
        )
        check(
            p.nod.emaTauMs > 0 && p.nod.emaMaxDtMs > 0 && p.nod.cancelGapMs > 0,
            "pipeline.nod: times must be > 0",
        )
        // 초기화를 일으키는 끊김은 언제나 진행 중인 하품·끄덕임 판정을 취소하는 끊김이기도 해야 한다(spec.py:759-762).
        check(
            p.yawn.cancelGapMs < clock.resetGapMs,
            "pipeline.yawn: need cancel_gap_ms < clock.reset_gap_ms " +
                "(cancel_gap_ms is ${p.yawn.cancelGapMs}, reset_gap_ms is ${clock.resetGapMs})",
        )
        check(
            p.nod.cancelGapMs < clock.resetGapMs,
            "pipeline.nod: need cancel_gap_ms < clock.reset_gap_ms " +
                "(cancel_gap_ms is ${p.nod.cancelGapMs}, reset_gap_ms is ${clock.resetGapMs})",
        )

        val glob = p.globalConstants
        check(base.eMin <= glob.e && glob.e <= base.eMax, "pipeline.global_constants.e must be within baseline e_min..e_max")
        check(0.0 <= glob.b && glob.b < clo.bClosed, "pipeline.global_constants.b must be within [0, closure.b_closed)")
        check(glob.r0.size == 9, "pipeline.global_constants.r0: need 9 values (row-major 3x3)")

        val sha = p.mediapipe.modelSha256
        // Python `len(sha) == 64`는 코드 포인트 수다. 16진 글자만 받으므로 UTF-16 길이와 판정이 같다.
        var allHex = true
        for (ch in sha) {
            if ("0123456789abcdef".indexOf(ch) < 0) {
                allHex = false
                break
            }
        }
        check(sha.length == 64 && allHex, "pipeline.mediapipe.model_sha256 must be 64 lowercase hex characters")
        check(p.mediapipe.numFaces >= 1, "pipeline.mediapipe.num_faces must be >= 1")
    }

    /** spec.py:792-817 `_validate_feature_spec` (F1~F13). pipeline 검사가 먼저 끝나 있어야 한다. */
    fun validateFeatureSpec(spec: FeatureSpec) {
        val pipe = spec.pipeline
        check(spec.specVersion >= 0, "spec_version must be >= 0")
        // R22: step_sec * 1000은 Long으로 계산한다(Int 곱은 넘칠 수 있다).
        check(spec.stepSec.toLong() * 1000L == pipe.clock.stepMs, "step_sec must match pipeline.clock.step_ms")
        check(
            spec.windowSteps == pipe.stepFeatures.windowSteps,
            "window_steps must match pipeline.step_features.window_steps",
        )
        check(spec.classOrder == CLASS_ORDER, "class_order must be ${PyRepr.value(CLASS_ORDER)}")
        check(spec.features.size == N_FEATURES, "features: expected $N_FEATURES entries, got ${spec.features.size}")

        val std = spec.standardization
        check(
            std.location == STANDARDIZATION_IN_GRAPH,
            "standardization.location must be ${PyRepr.value(STANDARDIZATION_IN_GRAPH)}",
        )
        check(std.zClip > 0.0, "standardization.z_clip must be > 0")
        check(std.stdFloor > 0.0, "standardization.std_floor must be > 0")
        check(std.temperature > 0.0, "standardization.temperature must be > 0")

        val windowsSec = listOf(
            Math.floorDiv(pipe.stepFeatures.lookbackLongMs, 1000L),
            Math.floorDiv(pipe.stepFeatures.lookbackShortMs, 1000L),
        )
        for (i in spec.features.indices) {
            val feat = spec.features[i]
            val where = "features[$i]"
            check(feat.index == i, "$where.index must be $i")
            check(
                feat.name == FEATURE_NAMES[i],
                "$where.name must be ${PyRepr.value(FEATURE_NAMES[i])}, got ${PyRepr.value(feat.name)}",
            )
            check(feat.windowSec.toLong() in windowsSec, "$where.window_sec must be one of ${PyRepr.tuple(windowsSec)}")
            check(feat.name.endsWith("_${feat.windowSec}"), "$where: name suffix must match window_sec")
            check(feat.transform in TRANSFORMS, "$where.transform must be one of ${PyRepr.tuple(TRANSFORMS)}")
            check(feat.std >= std.stdFloor, "$where.std must be >= standardization.std_floor")
        }
    }

    /** spec.py:883-922 `_validate_rules` (R1~R19). source와 alert_priority는 아직 문자열이다(RulesDraft). */
    fun validateRules(cfg: RulesDraft) {
        validateClock(cfg.clock, "clock")

        val fz = cfg.forwardZone
        check(fz.yawAbsMaxDeg > 0.0, "forward_zone.yaw_abs_max_deg must be > 0")
        check(fz.pitchUpMaxDeg > 0.0 && fz.pitchDownMaxDeg > 0.0, "forward_zone pitch limits must be > 0")
        check(fz.gazeHAbsMax > 0.0 && fz.gazeVAbsMax > 0.0, "forward_zone gaze limits must be > 0")
        check(fz.debounceMs >= 0 && fz.faceLossCreditMs >= 0, "forward_zone times must be >= 0")

        check(cfg.gazeAwayLong.onsetMs > 0, "gaze_away_long.onset_ms must be > 0")
        check(
            0 < cfg.vats.cumulativeMs && cfg.vats.cumulativeMs <= cfg.vats.windowMs,
            "vats: need 0 < cumulative_ms <= window_ms",
        )
        check(cfg.vats.resetForwardMs > 0, "vats.reset_forward_ms must be > 0")

        validateRunDetector(cfg.closure.detector, cfg.clock, "closure.detector")
        check(
            0 < cfg.closure.microsleepMs && cfg.closure.microsleepMs <= cfg.closure.sleepMs,
            "closure: need 0 < microsleep_ms <= sleep_ms",
        )

        val sources = DrowsySource.entries.map { it.wire }
        check(cfg.drowsinessSource in sources, "drowsiness.source must be one of ${PyRepr.tuple(sources)}")
        val model = cfg.drowsinessModel
        check(
            0.0 <= model.endP && model.endP < model.onsetP && model.onsetP <= 1.0,
            "drowsiness.model: need 0 <= end_p < onset_p <= 1",
        )
        val perclos = cfg.drowsinessPerclos
        check(
            0.0 < perclos.endRatio && perclos.endRatio < perclos.onsetRatio && perclos.onsetRatio <= 1.0,
            "drowsiness.perclos: need 0 < end_ratio < onset_ratio <= 1",
        )
        // spec.py:906-908: model, perclos 순서로 같은 두 검사.
        for ((name, onsetSteps, endSteps, refractoryMs) in listOf(
            Quad("model", model.onsetSteps, model.endSteps, model.refractoryMs),
            Quad("perclos", perclos.onsetSteps, perclos.endSteps, perclos.refractoryMs),
        )) {
            check(onsetSteps >= 1 && endSteps >= 1, "drowsiness.$name: step counts must be >= 1")
            check(refractoryMs >= 0, "drowsiness.$name.refractory_ms must be >= 0")
        }

        check(cfg.unableToAnalyze.windowMs > 0, "unable_to_analyze.window_ms must be > 0")
        check(
            0.0 < cfg.unableToAnalyze.minFaceRatio && cfg.unableToAnalyze.minFaceRatio <= 1.0,
            "unable_to_analyze.min_face_ratio must be within (0, 1]",
        )

        // EVENT_TYPES 순서 = EventType 선언 순서.
        for (type in EventType.entries) {
            val policy = cfg.events.get(type)
            val name = type.wire
            check(0 <= policy.severity && policy.severity <= 3, "events.$name.severity must be within 0..3")
            check(policy.cooldownMs >= 0 && policy.repeatMs >= 0, "events.$name: times must be >= 0")
            check(policy.minSpeedKmh >= 0.0, "events.$name.min_speed_kmh must be >= 0")
        }

        // Python `sorted(cfg.alert_priority) == sorted(EVENT_TYPES)`: 문자열 목록의 정렬한 사본을 비교한다.
        val given = cfg.alertPriority.sortedWith { a, b -> a.compareTo(b) }
        val all = EventType.entries.map { it.wire }.sortedWith { a, b -> a.compareTo(b) }
        check(given == all, "alert_priority must list every event type exactly once")
    }

    /**
     * pipeline.py:90-91 (계획 5.10절 R20): 파이프라인을 만들 때 `rules.clock == spec.clock`. Python은 ValueError다.
     * 메시지는 Python 그대로다. [checkPipelineConstruction]의 첫 검사다.
     */
    fun requireClockMatches(spec: PipelineSpec, rules: RulesConfig) {
        if (rules.clock != spec.clock) throw IllegalArgumentException("rules.clock must equal spec.clock")
    }

    /**
     * DriverPipeline 생성 검사 셋 (pipeline.py:90-93, 105-111, 계획 5.7절): 순서와 메시지는 Python 그대로이고 Python의
     * ValueError는 IllegalArgumentException이다. 특징 누적을 시작할 세션 시간을 돌려준다(기준선이 없으면 0,
     * pipeline.py:103 `_feature_start_ms = 0`). DriverPipeline(단계 3)이 생성자에서 부른다.
     */
    fun checkPipelineConstruction(spec: PipelineSpec, rules: RulesConfig, baseline: Baseline?, featureStartMs: Long?): Long {
        requireClockMatches(spec, rules)
        if (baseline == null && featureStartMs != null) {
            throw IllegalArgumentException("feature_start_ms requires an injected baseline")
        }
        var start = 0L
        if (baseline != null) {
            start = baseline.tCompleteMs
            if (featureStartMs != null) start = featureStartMs
            if (start < 0 || Math.floorMod(start, spec.clock.stepMs) != 0L) {
                throw IllegalArgumentException("feature_start_ms must be a non-negative multiple of clock.step_ms")
            }
        }
        return start
    }

    private data class Quad(val name: String, val onsetSteps: Int, val endSteps: Int, val refractoryMs: Long)
}
