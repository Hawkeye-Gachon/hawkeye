package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.DrowsySource
import hawkeye.drowsy.core.EventType
import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode

/**
 * 설정 파일 바이트 → 설정 객체 (계획 4.1절, 5.10절). 원본: drowsy/spec.py:516-588(구조 검사 S1~S9), 820-861
 * (`pipeline_spec_from_dict`, `feature_spec_from_dict`), 925-947(`rules_from_dict`).
 *
 * 순서는 Python 그대로다: 먼저 JSON 나무 전체를 자료형으로 바꾸고(구조 검사, 필드 선언 순서로), 그다음 범위 검사
 * (SpecValidation, 원본 순서)를 한다. 오류는 모두 SpecException이고 메시지는 Python의 영어 문장 그대로다.
 * 메시지 안의 값은 Python `repr`로 적는다([PyRepr]).
 *
 * Python과 다른 곳(계획이 정한 것):
 * - `pipeline_hash`를 다시 계산하지 않는다(SPEC 15장, 계획 5.10절). Python은 맞지 않으면 "pipeline_hash mismatch"로
 *   거부하지만 Kotlin은 문자열을 그대로 둔다. 시작 검사의 H2(파일 바이트 해시)와 H3(문자열 비교)가 대신 잡는다.
 * - JSON 읽기(MiniJson)는 중복 키, BOM, 남은 글자, Long을 넘는 정수 글자, 512단계보다 깊은 중첩을 거부한다.
 * - `Int` 칸의 정수가 Int 범위 밖이면 거부한다(R22). Python `int`는 한계가 없다.
 *   가정: 이 두 경우(Int 범위 밖, Long 범위 밖)에는 Python이 받는 문서를 Kotlin이 거부하거나 다른 문장으로 거부할 수 있다.
 *   C-loader 자료에서 `int_outside_int32`, `int_outside_int64` 표지가 붙은 문서만 그렇다. 소유자 확인을 기다린다(리드 판정 2).
 * - Python이 검사하지 않는 문자열(`pipeline.pose.*` 등)은 Kotlin도 검사하지 않고 String으로 둔다(리드 판정 2).
 *
 * 상태가 없다(R19).
 */
object SpecLoader {

    /** feature_spec_v{N}.json 바이트. pipeline_hash는 다시 계산하지 않고 문자열로 둔다(SPEC 15장). */
    @JvmStatic
    @Throws(SpecException::class)
    fun parseFeatureSpec(bytes: ByteArray): FeatureSpec = featureSpecFromTree(MiniJson.parse(bytes))

    /** rules.json 바이트. */
    @JvmStatic
    @Throws(SpecException::class)
    fun parseRules(bytes: ByteArray): RulesConfig = rulesFromTree(MiniJson.parse(bytes))

    /** feature_spec 바이트를 읽고 JSON 나무도 함께 돌려준다(상수 비트 지문이 그 나무를 쓴다, 계획 5.11절). */
    internal fun readFeatureSpec(bytes: ByteArray): Pair<FeatureSpec, Map<String, Any?>> {
        val tree = MiniJson.parse(bytes)
        val spec = featureSpecFromTree(tree)
        @Suppress("UNCHECKED_CAST")
        return spec to (tree as Map<String, Any?>)
    }

    /** rules.json 바이트를 읽고 JSON 나무도 함께 돌려준다. */
    internal fun readRules(bytes: ByteArray): Pair<RulesConfig, Map<String, Any?>> {
        val tree = MiniJson.parse(bytes)
        val rules = rulesFromTree(tree)
        @Suppress("UNCHECKED_CAST")
        return rules to (tree as Map<String, Any?>)
    }

    /** `pipeline` 블록 하나의 바이트 (spec.py:820-824 `pipeline_spec_from_dict`). 시험과 차등 비교 자료가 쓴다. */
    internal fun parsePipelineSpec(bytes: ByteArray): PipelineSpec = pipelineSpecFromTree(MiniJson.parse(bytes))

    /** spec.py:820-824 `pipeline_spec_from_dict`: 구조 검사(strict_float=True) 뒤 범위 검사. */
    internal fun pipelineSpecFromTree(block: Any?): PipelineSpec {
        val spec = Convert(strictFloat = true).pipelineSpec(block, "pipeline")
        SpecValidation.validatePipeline(spec)
        return spec
    }

    /**
     * spec.py:832-843 `feature_spec_from_dict`: 구조 검사(strict_float=True), pipeline 범위 검사, feature_spec 범위 검사.
     * Python의 마지막 단계(pipeline_hash 다시 계산과 비교, spec.py:837-842)는 하지 않는다(계획 5.10절, SPEC 15장).
     */
    internal fun featureSpecFromTree(data: Any?): FeatureSpec {
        val spec = Convert(strictFloat = true).featureSpec(data, "feature_spec")
        SpecValidation.validatePipeline(spec.pipeline)
        SpecValidation.validateFeatureSpec(spec)
        return spec
    }

    /**
     * spec.py:925-929 `rules_from_dict`: 구조 검사(strict_float=False: 실수 칸의 정수 글자를 실수로 바꾼다) 뒤 범위 검사.
     * drowsiness.source와 alert_priority는 검사가 끝난 뒤에 열거형으로 바꾼다(계획 5.10절).
     */
    internal fun rulesFromTree(data: Any?): RulesConfig {
        val draft = Convert(strictFloat = false).rulesDraft(data, "rules")
        SpecValidation.validateRules(draft)
        return RulesConfig(
            clock = draft.clock,
            forwardZone = draft.forwardZone,
            gazeAwayLong = draft.gazeAwayLong,
            vats = draft.vats,
            closure = draft.closure,
            drowsiness = DrowsinessRules(
                source = DrowsySource.fromWire(draft.drowsinessSource),
                model = draft.drowsinessModel,
                perclos = draft.drowsinessPerclos,
            ),
            unableToAnalyze = draft.unableToAnalyze,
            events = draft.events,
            alertPriority = java.util.Collections.unmodifiableList(draft.alertPriority.map { EventType.fromWire(it) }),
        )
    }

    /**
     * Python `str`의 비교(코드 포인트를 앞에서부터 비교하고, 앞이 같으면 짧은 쪽이 먼저). 홀로 선 대리 글자는 그 값 하나의
     * 코드 포인트다(`codePointAt`이 그렇게 돌려준다; MiniJson의 `\ud800`은 Python json.loads처럼 홀로 선 대리 글자다).
     */
    internal fun compareCodePoints(a: String, b: String): Int {
        var i = 0
        var j = 0
        while (i < a.length && j < b.length) {
            val ca = a.codePointAt(i)
            val cb = b.codePointAt(j)
            if (ca != cb) return if (ca < cb) -1 else 1
            i += Character.charCount(ca)
            j += Character.charCount(cb)
        }
        val restA = i < a.length
        val restB = j < b.length
        return if (restA == restB) 0 else if (restA) 1 else -1
    }

    // -----------------------------------------------------------------------------------------------------------------
    // 구조 검사: spec.py:516-588 `build_dataclass`, `_convert`
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 한 번의 변환. strictFloat는 Python의 `strict_float`다(feature_spec은 true, rules는 false).
     * 각 자료형의 필드는 Python dataclass의 선언 순서로 바꾼다. 이름 붙인 인자는 적힌 순서로 계산되므로(Kotlin 규칙)
     * 생성자 호출의 인자 순서가 곧 검사 순서다.
     */
    private class Convert(private val strictFloat: Boolean) {

        /** spec.py:522-531: mapping인지, 모르는 키(정렬해서), 빠진 키(필드 순서로). */
        fun mapping(data: Any?, where: String, names: List<String>): Map<*, *> {
            if (data !is Map<*, *>) throw SpecException("$where: expected a mapping, got ${PyRepr.typeName(data)}")
            val unknown = ArrayList<String>()
            for (key in data.keys) {
                if (key !in names) unknown.add(key as String)
            }
            if (unknown.isNotEmpty()) {
                // Python `sorted(str(key) ...)`: 코드 포인트 순. String.compareTo는 UTF-16 단위 순이라 BMP 밖 글자
                // (첫 단위 0xD800..0xDBFF)를 U+E000..U+FFFF 앞에 둔다. 그래서 코드 포인트로 비교한다.
                val sorted = unknown.sortedWith { a, b -> compareCodePoints(a, b) }
                throw SpecException("$where: unknown keys ${PyRepr.value(sorted)}")
            }
            val missing = ArrayList<String>()
            for (name in names) {
                if (!data.containsKey(name)) missing.add(name)
            }
            if (missing.isNotEmpty()) throw SpecException("$where: missing keys ${PyRepr.value(missing)}")
            return data
        }

        /** spec.py:538-541. Boolean만 받는다. */
        fun bool(value: Any?, where: String): Boolean {
            if (value !is Boolean) throw SpecException("$where: expected bool, got ${PyRepr.value(value)}")
            return value
        }

        /** spec.py:542-545. 정수 글자(Long)만 받는다. 250.0도 true도 아니다. */
        fun long(value: Any?, where: String): Long {
            if (value !is Long) throw SpecException("$where: expected int, got ${PyRepr.value(value)}")
            return value
        }

        /**
         * spec.py:542-545에 더해 R22: Int 칸은 Int 범위를 검사해 줄인다. `Long.toInt()`로 조용히 줄이지 않는다.
         * 이 문장은 Kotlin의 것이다(Python에는 정수 한계가 없다).
         */
        fun int(value: Any?, where: String): Int {
            val v = long(value, where)
            if (v < Int.MIN_VALUE || v > Int.MAX_VALUE) {
                throw SpecException("$where: integer $v is outside the 32-bit range of this field")
            }
            return v.toInt()
        }

        /** spec.py:546-555. */
        fun float(value: Any?, where: String): Double {
            var v = value
            if (v is Long && !strictFloat) {
                // Python float(int): 가장 가까운 double. Long.toDouble()도 가장 가까운 값으로 바꾼다(JLS 5.1.2).
                v = v.toDouble()
            }
            if (v is Long) {
                throw SpecException(
                    "$where: expected float, got integer ${PyRepr.value(v)} (write ${PyRepr.float(v.toDouble())})",
                )
            }
            if (v !is Double) throw SpecException("$where: expected float, got ${PyRepr.value(v)}")
            if (v.isNaN() || v.isInfinite()) {
                throw SpecException("$where: expected a finite float, got ${PyRepr.value(v)}")
            }
            return v
        }

        /** spec.py:556-559. */
        fun str(value: Any?, where: String): String {
            if (value !is String) throw SpecException("$where: expected str, got ${PyRepr.value(value)}")
            return value
        }

        /** spec.py:572-576: tuple 칸. 원소의 자리는 `where[i]`. */
        fun <T> list(value: Any?, where: String, element: (Any?, String) -> T): List<T> {
            if (value !is List<*>) throw SpecException("$where: expected a list, got ${PyRepr.value(value)}")
            val out = ArrayList<T>(value.size)
            for (i in value.indices) out.add(element(value[i], "$where[$i]"))
            return java.util.Collections.unmodifiableList(out)
        }

        fun intList(value: Any?, where: String): List<Int> = list(value, where) { v, w -> int(v, w) }
        fun floatList(value: Any?, where: String): List<Double> = list(value, where) { v, w -> float(v, w) }
        fun strList(value: Any?, where: String): List<String> = list(value, where) { v, w -> str(v, w) }

        // ----- PipelineSpec (spec.py:69-279) -----

        fun pipelineSpec(data: Any?, where: String): PipelineSpec {
            val m = mapping(data, where, PIPELINE_FIELDS)
            return PipelineSpec(
                landmarks = landmarks(m["landmarks"], "$where.landmarks"),
                blendshapes = blendshapes(m["blendshapes"], "$where.blendshapes"),
                pose = pose(m["pose"], "$where.pose"),
                clock = clock(m["clock"], "$where.clock"),
                signals = signals(m["signals"], "$where.signals"),
                baseline = baseline(m["baseline"], "$where.baseline"),
                closure = closure(m["closure"], "$where.closure"),
                runDetector = runDetector(m["run_detector"], "$where.run_detector"),
                stepFeatures = stepFeatures(m["step_features"], "$where.step_features"),
                yawn = yawn(m["yawn"], "$where.yawn"),
                nod = nod(m["nod"], "$where.nod"),
                globalConstants = globalConstants(m["global_constants"], "$where.global_constants"),
                mediapipe = mediapipe(m["mediapipe"], "$where.mediapipe"),
            )
        }

        fun landmarks(data: Any?, where: String): LandmarkSpec {
            val m = mapping(data, where, LANDMARK_FIELDS)
            return LandmarkSpec(
                earR = intList(m["ear_r"], "$where.ear_r"),
                earL = intList(m["ear_l"], "$where.ear_l"),
                irisR = intList(m["iris_r"], "$where.iris_r"),
                irisL = intList(m["iris_l"], "$where.iris_l"),
                cornerOuterR = int(m["corner_outer_r"], "$where.corner_outer_r"),
                cornerInnerR = int(m["corner_inner_r"], "$where.corner_inner_r"),
                cornerOuterL = int(m["corner_outer_l"], "$where.corner_outer_l"),
                cornerInnerL = int(m["corner_inner_l"], "$where.corner_inner_l"),
                contourR = intList(m["contour_r"], "$where.contour_r"),
                contourL = intList(m["contour_l"], "$where.contour_l"),
                faceWidth = intList(m["face_width"], "$where.face_width"),
            )
        }

        fun blendshapes(data: Any?, where: String): BlendshapeSpec {
            val m = mapping(data, where, BLENDSHAPE_FIELDS)
            return BlendshapeSpec(
                eyeBlinkL = str(m["eye_blink_l"], "$where.eye_blink_l"),
                eyeBlinkR = str(m["eye_blink_r"], "$where.eye_blink_r"),
                eyeLook = strList(m["eye_look"], "$where.eye_look"),
                jawOpen = str(m["jaw_open"], "$where.jaw_open"),
                order = strList(m["order"], "$where.order"),
            )
        }

        fun pose(data: Any?, where: String): PoseSpec {
            val m = mapping(data, where, POSE_FIELDS)
            return PoseSpec(
                order = str(m["order"], "$where.order"),
                yawPos = str(m["yaw_pos"], "$where.yaw_pos"),
                pitchPos = str(m["pitch_pos"], "$where.pitch_pos"),
                rollPos = str(m["roll_pos"], "$where.roll_pos"),
                matrixLayout = str(m["matrix_layout"], "$where.matrix_layout"),
                angleUnit = str(m["angle_unit"], "$where.angle_unit"),
            )
        }

        fun clock(data: Any?, where: String): ClockSpec {
            val m = mapping(data, where, CLOCK_FIELDS)
            return ClockSpec(
                stepMs = long(m["step_ms"], "$where.step_ms"),
                maxHoldMs = long(m["max_hold_ms"], "$where.max_hold_ms"),
                resetGapMs = long(m["reset_gap_ms"], "$where.reset_gap_ms"),
            )
        }

        fun signals(data: Any?, where: String): SignalSpec {
            val m = mapping(data, where, SIGNAL_FIELDS)
            return SignalSpec(minEyeWidthPx = float(m["min_eye_width_px"], "$where.min_eye_width_px"))
        }

        fun baseline(data: Any?, where: String): BaselineSpec {
            val m = mapping(data, where, BASELINE_FIELDS)
            return BaselineSpec(
                durationMs = long(m["duration_ms"], "$where.duration_ms"),
                extendStepMs = long(m["extend_step_ms"], "$where.extend_step_ms"),
                maxDurationMs = long(m["max_duration_ms"], "$where.max_duration_ms"),
                minFaceFrames = int(m["min_face_frames"], "$where.min_face_frames"),
                minFaceRatio = float(m["min_face_ratio"], "$where.min_face_ratio"),
                eMin = float(m["e_min"], "$where.e_min"),
                eMax = float(m["e_max"], "$where.e_max"),
                earQuantile = float(m["ear_quantile"], "$where.ear_quantile"),
                blinkQuantile = float(m["blink_quantile"], "$where.blink_quantile"),
                blinkRecordQuantiles = floatList(m["blink_record_quantiles"], "$where.blink_record_quantiles"),
                offAxisYawDeg = float(m["off_axis_yaw_deg"], "$where.off_axis_yaw_deg"),
            )
        }

        fun closure(data: Any?, where: String): ClosureSpec {
            val m = mapping(data, where, CLOSURE_FIELDS)
            return ClosureSpec(
                rho = float(m["rho"], "$where.rho"),
                bClosed = float(m["b_closed"], "$where.b_closed"),
                bSpanMin = float(m["b_span_min"], "$where.b_span_min"),
                w = float(m["w"], "$where.w"),
                farEyeZeroYawDeg = float(m["far_eye_zero_yaw_deg"], "$where.far_eye_zero_yaw_deg"),
                farEyeRampDeg = float(m["far_eye_ramp_deg"], "$where.far_eye_ramp_deg"),
                validMaxAbsYawDeg = float(m["valid_max_abs_yaw_deg"], "$where.valid_max_abs_yaw_deg"),
            )
        }

        fun runDetector(data: Any?, where: String): RunDetectorSpec {
            val m = mapping(data, where, RUN_DETECTOR_FIELDS)
            return RunDetectorSpec(
                enterThr = float(m["enter_thr"], "$where.enter_thr"),
                exitThr = float(m["exit_thr"], "$where.exit_thr"),
                minDurationMs = long(m["min_duration_ms"], "$where.min_duration_ms"),
                maxDtMs = long(m["max_dt_ms"], "$where.max_dt_ms"),
                holdMs = long(m["hold_ms"], "$where.hold_ms"),
                velocityLookbackSamples = int(m["velocity_lookback_samples"], "$where.velocity_lookback_samples"),
            )
        }

        fun stepFeatures(data: Any?, where: String): StepFeatureSpec {
            val m = mapping(data, where, STEP_FEATURE_FIELDS)
            return StepFeatureSpec(
                names = strList(m["names"], "$where.names"),
                windowSteps = int(m["window_steps"], "$where.window_steps"),
                lookbackLongMs = long(m["lookback_long_ms"], "$where.lookback_long_ms"),
                lookbackShortMs = long(m["lookback_short_ms"], "$where.lookback_short_ms"),
                perclosThr = float(m["perclos_thr"], "$where.perclos_thr"),
                perclosMinValidMs60 = long(m["perclos_min_valid_ms_60"], "$where.perclos_min_valid_ms_60"),
                perclosMinValidMs10 = long(m["perclos_min_valid_ms_10"], "$where.perclos_min_valid_ms_10"),
                blinkRateMinValidMs = long(m["blink_rate_min_valid_ms"], "$where.blink_rate_min_valid_ms"),
                blinkDurationCapMs = long(m["blink_duration_cap_ms"], "$where.blink_duration_cap_ms"),
                longClosureMinMs = long(m["long_closure_min_ms"], "$where.long_closure_min_ms"),
                maxClosureCapMs = long(m["max_closure_cap_ms"], "$where.max_closure_cap_ms"),
                pitchStdMinFaceMs = long(m["pitch_std_min_face_ms"], "$where.pitch_std_min_face_ms"),
            )
        }

        fun yawn(data: Any?, where: String): YawnSpec {
            val m = mapping(data, where, YAWN_FIELDS)
            return YawnSpec(
                enterThr = float(m["enter_thr"], "$where.enter_thr"),
                exitThr = float(m["exit_thr"], "$where.exit_thr"),
                countAtMs = long(m["count_at_ms"], "$where.count_at_ms"),
                cancelGapMs = long(m["cancel_gap_ms"], "$where.cancel_gap_ms"),
            )
        }

        fun nod(data: Any?, where: String): NodSpec {
            val m = mapping(data, where, NOD_FIELDS)
            return NodSpec(
                enterDeg = float(m["enter_deg"], "$where.enter_deg"),
                exitDeg = float(m["exit_deg"], "$where.exit_deg"),
                minDurationMs = long(m["min_duration_ms"], "$where.min_duration_ms"),
                maxDurationMs = long(m["max_duration_ms"], "$where.max_duration_ms"),
                emaTauMs = long(m["ema_tau_ms"], "$where.ema_tau_ms"),
                emaMaxDtMs = long(m["ema_max_dt_ms"], "$where.ema_max_dt_ms"),
                cancelGapMs = long(m["cancel_gap_ms"], "$where.cancel_gap_ms"),
            )
        }

        fun globalConstants(data: Any?, where: String): GlobalConstants {
            val m = mapping(data, where, GLOBAL_CONSTANT_FIELDS)
            return GlobalConstants(
                e = float(m["e"], "$where.e"),
                b = float(m["b"], "$where.b"),
                jaw0 = float(m["jaw0"], "$where.jaw0"),
                gazeH0 = float(m["gaze_h0"], "$where.gaze_h0"),
                gazeV0 = float(m["gaze_v0"], "$where.gaze_v0"),
                r0 = floatList(m["r0"], "$where.r0"),
            )
        }

        fun mediapipe(data: Any?, where: String): MediaPipeSpec {
            val m = mapping(data, where, MEDIAPIPE_FIELDS)
            return MediaPipeSpec(
                modelSha256 = str(m["model_sha256"], "$where.model_sha256"),
                numFaces = int(m["num_faces"], "$where.num_faces"),
            )
        }

        // ----- FeatureSpec (spec.py:287-344) -----

        fun featureSpec(data: Any?, where: String): FeatureSpec {
            val m = mapping(data, where, FEATURE_SPEC_FIELDS)
            return FeatureSpec(
                specVersion = int(m["spec_version"], "$where.spec_version"),
                stepSec = int(m["step_sec"], "$where.step_sec"),
                windowSteps = int(m["window_steps"], "$where.window_steps"),
                classOrder = strList(m["class_order"], "$where.class_order"),
                features = list(m["features"], "$where.features") { v, w -> featureDef(v, w) },
                standardization = standardization(m["standardization"], "$where.standardization"),
                pipeline = pipelineSpec(m["pipeline"], "$where.pipeline"),
                pipelineHash = str(m["pipeline_hash"], "$where.pipeline_hash"),
            )
        }

        fun featureDef(data: Any?, where: String): FeatureDef {
            val m = mapping(data, where, FEATURE_DEF_FIELDS)
            return FeatureDef(
                index = int(m["index"], "$where.index"),
                name = str(m["name"], "$where.name"),
                windowSec = int(m["window_sec"], "$where.window_sec"),
                unit = str(m["unit"], "$where.unit"),
                formula = str(m["formula"], "$where.formula"),
                undefinedValue = float(m["undefined_value"], "$where.undefined_value"),
                transform = str(m["transform"], "$where.transform"),
                mean = float(m["mean"], "$where.mean"),
                std = float(m["std"], "$where.std"),
            )
        }

        fun standardization(data: Any?, where: String): Standardization {
            val m = mapping(data, where, STANDARDIZATION_FIELDS)
            return Standardization(
                location = str(m["location"], "$where.location"),
                zClip = float(m["z_clip"], "$where.z_clip"),
                stdFloor = float(m["std_floor"], "$where.std_floor"),
                temperature = float(m["temperature"], "$where.temperature"),
            )
        }

        // ----- RulesConfig (spec.py:352-506) -----

        fun rulesDraft(data: Any?, where: String): RulesDraft {
            val m = mapping(data, where, RULES_FIELDS)
            val clock = clock(m["clock"], "$where.clock")
            val forwardZone = forwardZone(m["forward_zone"], "$where.forward_zone")
            val gazeAwayLong = gazeAwayLong(m["gaze_away_long"], "$where.gaze_away_long")
            val vats = vats(m["vats"], "$where.vats")
            val closure = closureRules(m["closure"], "$where.closure")
            // drowsiness 블록 (spec.py:424-433): source는 문자열 그대로 둔다(검사 뒤에 열거형).
            val dw = "$where.drowsiness"
            val d = mapping(m["drowsiness"], dw, DROWSINESS_FIELDS)
            val source = str(d["source"], "$dw.source")
            val model = modelDrowsy(d["model"], "$dw.model")
            val perclos = perclosDrowsy(d["perclos"], "$dw.perclos")
            val unable = unableToAnalyze(m["unable_to_analyze"], "$where.unable_to_analyze")
            val events = eventPolicies(m["events"], "$where.events")
            val alertPriority = strList(m["alert_priority"], "$where.alert_priority")
            return RulesDraft(
                clock = clock,
                forwardZone = forwardZone,
                gazeAwayLong = gazeAwayLong,
                vats = vats,
                closure = closure,
                drowsinessSource = source,
                drowsinessModel = model,
                drowsinessPerclos = perclos,
                unableToAnalyze = unable,
                events = events,
                alertPriority = alertPriority,
            )
        }

        fun forwardZone(data: Any?, where: String): ForwardZoneRules {
            val m = mapping(data, where, FORWARD_ZONE_FIELDS)
            return ForwardZoneRules(
                yawAbsMaxDeg = float(m["yaw_abs_max_deg"], "$where.yaw_abs_max_deg"),
                pitchUpMaxDeg = float(m["pitch_up_max_deg"], "$where.pitch_up_max_deg"),
                pitchDownMaxDeg = float(m["pitch_down_max_deg"], "$where.pitch_down_max_deg"),
                eyeAuxEnabled = bool(m["eye_aux_enabled"], "$where.eye_aux_enabled"),
                gazeHAbsMax = float(m["gaze_h_abs_max"], "$where.gaze_h_abs_max"),
                gazeVAbsMax = float(m["gaze_v_abs_max"], "$where.gaze_v_abs_max"),
                debounceMs = long(m["debounce_ms"], "$where.debounce_ms"),
                faceLossCreditMs = long(m["face_loss_credit_ms"], "$where.face_loss_credit_ms"),
            )
        }

        fun gazeAwayLong(data: Any?, where: String): GazeAwayLongRules {
            val m = mapping(data, where, GAZE_AWAY_LONG_FIELDS)
            return GazeAwayLongRules(onsetMs = long(m["onset_ms"], "$where.onset_ms"))
        }

        fun vats(data: Any?, where: String): VatsRules {
            val m = mapping(data, where, VATS_FIELDS)
            return VatsRules(
                windowMs = long(m["window_ms"], "$where.window_ms"),
                cumulativeMs = long(m["cumulative_ms"], "$where.cumulative_ms"),
                resetEnabled = bool(m["reset_enabled"], "$where.reset_enabled"),
                resetForwardMs = long(m["reset_forward_ms"], "$where.reset_forward_ms"),
            )
        }

        fun closureRules(data: Any?, where: String): ClosureRules {
            val m = mapping(data, where, CLOSURE_RULES_FIELDS)
            return ClosureRules(
                detector = runDetector(m["detector"], "$where.detector"),
                microsleepMs = long(m["microsleep_ms"], "$where.microsleep_ms"),
                sleepMs = long(m["sleep_ms"], "$where.sleep_ms"),
            )
        }

        fun modelDrowsy(data: Any?, where: String): ModelDrowsyRules {
            val m = mapping(data, where, MODEL_DROWSY_FIELDS)
            return ModelDrowsyRules(
                onsetP = float(m["onset_p"], "$where.onset_p"),
                onsetSteps = int(m["onset_steps"], "$where.onset_steps"),
                endP = float(m["end_p"], "$where.end_p"),
                endSteps = int(m["end_steps"], "$where.end_steps"),
                refractoryMs = long(m["refractory_ms"], "$where.refractory_ms"),
            )
        }

        fun perclosDrowsy(data: Any?, where: String): PerclosDrowsyRules {
            val m = mapping(data, where, PERCLOS_DROWSY_FIELDS)
            return PerclosDrowsyRules(
                onsetRatio = float(m["onset_ratio"], "$where.onset_ratio"),
                onsetSteps = int(m["onset_steps"], "$where.onset_steps"),
                endRatio = float(m["end_ratio"], "$where.end_ratio"),
                endSteps = int(m["end_steps"], "$where.end_steps"),
                refractoryMs = long(m["refractory_ms"], "$where.refractory_ms"),
            )
        }

        fun unableToAnalyze(data: Any?, where: String): UnableToAnalyzeRules {
            val m = mapping(data, where, UNABLE_FIELDS)
            return UnableToAnalyzeRules(
                windowMs = long(m["window_ms"], "$where.window_ms"),
                minFaceRatio = float(m["min_face_ratio"], "$where.min_face_ratio"),
            )
        }

        fun eventPolicies(data: Any?, where: String): EventPolicies {
            val m = mapping(data, where, EVENT_POLICIES_FIELDS)
            return EventPolicies(
                calibrating = eventPolicy(m["calibrating"], "$where.calibrating"),
                unableToAnalyze = eventPolicy(m["unable_to_analyze"], "$where.unable_to_analyze"),
                gazeAwayLong = eventPolicy(m["gaze_away_long"], "$where.gaze_away_long"),
                vats = eventPolicy(m["vats"], "$where.vats"),
                drowsy = eventPolicy(m["drowsy"], "$where.drowsy"),
                microsleep = eventPolicy(m["microsleep"], "$where.microsleep"),
                sleep = eventPolicy(m["sleep"], "$where.sleep"),
            )
        }

        fun eventPolicy(data: Any?, where: String): EventPolicy {
            val m = mapping(data, where, EVENT_POLICY_FIELDS)
            return EventPolicy(
                severity = int(m["severity"], "$where.severity"),
                cooldownMs = long(m["cooldown_ms"], "$where.cooldown_ms"),
                repeatMs = long(m["repeat_ms"], "$where.repeat_ms"),
                minSpeedKmh = float(m["min_speed_kmh"], "$where.min_speed_kmh"),
            )
        }
    }

    // Python dataclass의 필드 이름, 선언 순서 (spec.py:69-506). 빠진 키 메시지가 이 순서로 적힌다.
    private val PIPELINE_FIELDS = listOf(
        "landmarks", "blendshapes", "pose", "clock", "signals", "baseline", "closure", "run_detector",
        "step_features", "yawn", "nod", "global_constants", "mediapipe",
    )
    private val LANDMARK_FIELDS = listOf(
        "ear_r", "ear_l", "iris_r", "iris_l", "corner_outer_r", "corner_inner_r", "corner_outer_l", "corner_inner_l",
        "contour_r", "contour_l", "face_width",
    )
    private val BLENDSHAPE_FIELDS = listOf("eye_blink_l", "eye_blink_r", "eye_look", "jaw_open", "order")
    private val POSE_FIELDS = listOf("order", "yaw_pos", "pitch_pos", "roll_pos", "matrix_layout", "angle_unit")
    private val CLOCK_FIELDS = listOf("step_ms", "max_hold_ms", "reset_gap_ms")
    private val SIGNAL_FIELDS = listOf("min_eye_width_px")
    private val BASELINE_FIELDS = listOf(
        "duration_ms", "extend_step_ms", "max_duration_ms", "min_face_frames", "min_face_ratio", "e_min", "e_max",
        "ear_quantile", "blink_quantile", "blink_record_quantiles", "off_axis_yaw_deg",
    )
    private val CLOSURE_FIELDS = listOf(
        "rho", "b_closed", "b_span_min", "w", "far_eye_zero_yaw_deg", "far_eye_ramp_deg", "valid_max_abs_yaw_deg",
    )
    private val RUN_DETECTOR_FIELDS = listOf(
        "enter_thr", "exit_thr", "min_duration_ms", "max_dt_ms", "hold_ms", "velocity_lookback_samples",
    )
    private val STEP_FEATURE_FIELDS = listOf(
        "names", "window_steps", "lookback_long_ms", "lookback_short_ms", "perclos_thr", "perclos_min_valid_ms_60",
        "perclos_min_valid_ms_10", "blink_rate_min_valid_ms", "blink_duration_cap_ms", "long_closure_min_ms",
        "max_closure_cap_ms", "pitch_std_min_face_ms",
    )
    private val YAWN_FIELDS = listOf("enter_thr", "exit_thr", "count_at_ms", "cancel_gap_ms")
    private val NOD_FIELDS = listOf(
        "enter_deg", "exit_deg", "min_duration_ms", "max_duration_ms", "ema_tau_ms", "ema_max_dt_ms", "cancel_gap_ms",
    )
    private val GLOBAL_CONSTANT_FIELDS = listOf("e", "b", "jaw0", "gaze_h0", "gaze_v0", "r0")
    private val MEDIAPIPE_FIELDS = listOf("model_sha256", "num_faces")
    private val FEATURE_SPEC_FIELDS = listOf(
        "spec_version", "step_sec", "window_steps", "class_order", "features", "standardization", "pipeline",
        "pipeline_hash",
    )
    private val FEATURE_DEF_FIELDS = listOf(
        "index", "name", "window_sec", "unit", "formula", "undefined_value", "transform", "mean", "std",
    )
    private val STANDARDIZATION_FIELDS = listOf("location", "z_clip", "std_floor", "temperature")
    private val RULES_FIELDS = listOf(
        "clock", "forward_zone", "gaze_away_long", "vats", "closure", "drowsiness", "unable_to_analyze", "events",
        "alert_priority",
    )
    private val FORWARD_ZONE_FIELDS = listOf(
        "yaw_abs_max_deg", "pitch_up_max_deg", "pitch_down_max_deg", "eye_aux_enabled", "gaze_h_abs_max",
        "gaze_v_abs_max", "debounce_ms", "face_loss_credit_ms",
    )
    private val GAZE_AWAY_LONG_FIELDS = listOf("onset_ms")
    private val VATS_FIELDS = listOf("window_ms", "cumulative_ms", "reset_enabled", "reset_forward_ms")
    private val CLOSURE_RULES_FIELDS = listOf("detector", "microsleep_ms", "sleep_ms")
    private val DROWSINESS_FIELDS = listOf("source", "model", "perclos")
    private val MODEL_DROWSY_FIELDS = listOf("onset_p", "onset_steps", "end_p", "end_steps", "refractory_ms")
    private val PERCLOS_DROWSY_FIELDS = listOf("onset_ratio", "onset_steps", "end_ratio", "end_steps", "refractory_ms")
    private val UNABLE_FIELDS = listOf("window_ms", "min_face_ratio")
    private val EVENT_POLICIES_FIELDS = listOf(
        "calibrating", "unable_to_analyze", "gaze_away_long", "vats", "drowsy", "microsleep", "sleep",
    )
    private val EVENT_POLICY_FIELDS = listOf("severity", "cooldown_ms", "repeat_ms", "min_speed_kmh")
}

/**
 * 검사 전의 규칙 설정. drowsiness.source와 alert_priority가 아직 문자열이다(계획 5.10절: Python과 같은 자리에서
 * 같은 문장으로 검사한 뒤에 열거형으로 바꾼다). 나머지 필드는 RulesConfig와 같다.
 */
internal class RulesDraft(
    val clock: ClockSpec,
    val forwardZone: ForwardZoneRules,
    val gazeAwayLong: GazeAwayLongRules,
    val vats: VatsRules,
    val closure: ClosureRules,
    val drowsinessSource: String,
    val drowsinessModel: ModelDrowsyRules,
    val drowsinessPerclos: PerclosDrowsyRules,
    val unableToAnalyze: UnableToAnalyzeRules,
    val events: EventPolicies,
    val alertPriority: List<String>,
)

/**
 * 오류 메시지 안의 값을 Python `repr`과 같은 글자로 쓴다(메시지가 Python과 같아야 한다, 계획 4.1절, 5.10절).
 * 값은 MiniJson이 만드는 형이다: null, Boolean, Long, Double, String, List, Map.
 */
internal object PyRepr {

    /** Python `type(x).__name__` (spec.py:523). */
    fun typeName(v: Any?): String = when (v) {
        null -> "NoneType"
        is Boolean -> "bool"
        is Long, is Int -> "int"
        is Double -> "float"
        is String -> "str"
        is List<*> -> "list"
        is Map<*, *> -> "dict"
        else -> v.javaClass.simpleName
    }

    fun value(v: Any?): String = when (v) {
        null -> "None"
        is Boolean -> if (v) "True" else "False"
        is Long, is Int -> v.toString()
        is Double -> float(v)
        is String -> str(v)
        is List<*> -> v.joinToString(", ", "[", "]") { value(it) }
        is Map<*, *> -> v.entries.joinToString(", ", "{", "}") { value(it.key) + ": " + value(it.value) }
        else -> throw IllegalArgumentException("no Python repr for ${v.javaClass.name}")
    }

    /** Python tuple의 repr: `('none', 'log1p')`, `(60, 10)`. 원소 하나이면 `(x,)`. */
    fun tuple(items: List<Any?>): String =
        if (items.size == 1) "(" + value(items[0]) + ",)" else items.joinToString(", ", "(", ")") { value(it) }

    /**
     * Python `repr(float)`: 정확히 왕복하는 가장 짧은 10진 숫자(같은 길이가 둘이면 가까운 쪽), 소수점 위치가 -4 이하나
     * 16 초과이면 지수 표기(`1e-05`, `1e+16`), 아니면 고정 표기(끝이 정수이면 `.0`). CPython `float_repr_style = 'short'`.
     * 왕복 여부는 BigDecimal로 정확히 판정한다(이웃 double과의 가운데 점, 가운데에서는 짝수 가수 규칙). 그래서 플랫폼의
     * 10진 쓰기·읽기 구현에 기대지 않는다.
     */
    fun float(x: Double): String {
        if (x.isNaN()) return "nan"
        if (x == Double.POSITIVE_INFINITY) return "inf"
        if (x == Double.NEGATIVE_INFINITY) return "-inf"
        if (x == 0.0) return if (x.toRawBits() < 0L) "-0.0" else "0.0"
        val ax = Math.abs(x)
        val a = BigDecimal(ax)
        val down = Math.nextDown(ax)
        val lowMid = a.add(BigDecimal(down)).divide(TWO)
        val highMid = if (ax == Double.MAX_VALUE) {
            a.add(BigDecimal(Math.ulp(ax)).divide(TWO))
        } else {
            a.add(BigDecimal(Math.nextUp(ax))).divide(TWO)
        }
        val evenSignificand = (ax.toRawBits() and 1L) == 0L
        fun roundTrips(c: BigDecimal): Boolean {
            val lo = c.compareTo(lowMid)
            val hi = c.compareTo(highMid)
            if (lo > 0 && hi < 0) return true
            return (lo == 0 || hi == 0) && evenSignificand
        }
        var chosen: BigDecimal? = null
        for (p in 1..17) {
            val near = a.round(MathContext(p, RoundingMode.HALF_EVEN))
            if (roundTrips(near)) {
                chosen = near
                break
            }
            val other = if (near > a) a.round(MathContext(p, RoundingMode.DOWN)) else a.round(MathContext(p, RoundingMode.UP))
            if (roundTrips(other)) {
                chosen = other
                break
            }
        }
        val c = (chosen ?: a).stripTrailingZeros()
        val digits = c.unscaledValue().toString()
        val decpt = digits.length - c.scale()
        val body = if (decpt <= -4 || decpt > 16) {
            val exp = decpt - 1
            val mant = if (digits.length == 1) digits else digits[0] + "." + digits.substring(1)
            val e = Math.abs(exp).toString().padStart(2, '0')
            mant + "e" + (if (exp < 0) "-" else "+") + e
        } else if (decpt <= 0) {
            "0." + "0".repeat(-decpt) + digits
        } else if (decpt < digits.length) {
            digits.substring(0, decpt) + "." + digits.substring(decpt)
        } else {
            digits + "0".repeat(decpt - digits.length) + ".0"
        }
        return if (x < 0) "-$body" else body
    }

    private val TWO = BigDecimal(2)

    /**
     * Python `repr(str)`: 작은따옴표로 감싼다. 글자에 작은따옴표가 있고 큰따옴표가 없으면 큰따옴표로 감싼다.
     * `\\`, 감싼 따옴표, `\t`, `\n`, `\r`를 이스케이프하고, 인쇄할 수 없는 글자는 `\xNN`, `\uNNNN`, `\UNNNNNNNN`으로 쓴다.
     * 인쇄할 수 없는 글자의 판정은 참조 Python의 `str.isprintable` 표([PyPrintable], Python 3.11 / unicodedata 14.0.0에서
     * 만든 것)다. JVM·ART의 `Character.getType`은 유니코드 판이 달라(JDK 17은 13.0) 새 글자에서 다르므로 쓰지 않는다.
     * 설정 파일의 메시지에만 쓰인다.
     */
    fun str(s: String): String {
        val quote = if (s.indexOf('\'') >= 0 && s.indexOf('"') < 0) '"' else '\''
        val sb = StringBuilder(s.length + 2)
        sb.append(quote)
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            i += Character.charCount(cp)
            when {
                cp == quote.code || cp == '\\'.code -> sb.append('\\').appendCodePoint(cp)
                cp == '\t'.code -> sb.append("\\t")
                cp == '\n'.code -> sb.append("\\n")
                cp == '\r'.code -> sb.append("\\r")
                cp < 0x20 || cp == 0x7F -> sb.append("\\x").append(hex(cp, 2))
                cp < 0x7F -> sb.appendCodePoint(cp)
                PyPrintable.isPrintable(cp) -> sb.appendCodePoint(cp)
                cp <= 0xFF -> sb.append("\\x").append(hex(cp, 2))
                cp <= 0xFFFF -> sb.append("\\u").append(hex(cp, 4))
                else -> sb.append("\\U").append(hex(cp, 8))
            }
        }
        sb.append(quote)
        return sb.toString()
    }

    private fun hex(v: Int, width: Int): String = Integer.toHexString(v).padStart(width, '0')
}
