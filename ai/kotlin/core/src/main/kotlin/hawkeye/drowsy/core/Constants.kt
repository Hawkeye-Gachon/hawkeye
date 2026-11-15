package hawkeye.drowsy.core

import java.util.Collections

// 원본: drowsy/types.py:28-138, drowsy/rules.py:89-91. SPEC 3.9절.
// 문자열 상수는 wire 문자열을 가진 열거형이다(K21 (가)). wire는 Python과 벡터, rules.json에 적힌 문자열 그대로다.
// name.lowercase()로 만들지 않는다. 선언 순서는 Python 튜플(EVENT_TYPES 등)의 순서와 같다.
// 이 파일의 상수는 조정하는 임계값이 아니라 이식의 정체다. Python에서도 코드 상수이고, 읽기 단계가 JSON을 이것과
// 비교한다(spec.py:726, 800, 813).

/** 이벤트 종류. 선언 순서 = `EVENT_TYPES` (types.py:35-43). */
enum class EventType(val wire: String) {
    CALIBRATING("calibrating"),
    UNABLE_TO_ANALYZE("unable_to_analyze"),
    GAZE_AWAY_LONG("gaze_away_long"),
    VATS("vats"),
    DROWSY("drowsy"),
    MICROSLEEP("microsleep"),
    SLEEP("sleep");

    internal companion object {
        /** wire 문자열 → 열거형. 모르는 문자열은 IllegalArgumentException. */
        fun fromWire(wire: String): EventType = entries.firstOrNull { it.wire == wire }
            ?: throw IllegalArgumentException("unknown event type: \"$wire\"")
    }
}

/** 이벤트 단계. 선언 순서 = `PHASES` (types.py:46-48). */
enum class Phase(val wire: String) {
    ONSET("onset"),
    END("end");

    internal companion object {
        fun fromWire(wire: String): Phase = entries.firstOrNull { it.wire == wire }
            ?: throw IllegalArgumentException("unknown phase: \"$wire\"")
    }
}

/** 이벤트 출처. 선언 순서 = `SOURCES` (types.py:51-64). */
enum class Source(val wire: String) {
    FACE("face"),
    HEAD_POSE("head_pose"),
    EYE("eye"),
    MODEL("model"),
    PERCLOS("perclos"),
    CLOSURE("closure");

    internal companion object {
        fun fromWire(wire: String): Source = entries.firstOrNull { it.wire == wire }
            ?: throw IllegalArgumentException("unknown source: \"$wire\"")
    }
}

/** 규칙 설정 `drowsiness.source`의 값. 선언 순서 = `DROWSY_SOURCES` (types.py:67-70). */
enum class DrowsySource(val wire: String) {
    MODEL("model"),
    PERCLOS("perclos"),
    EITHER("either");

    internal companion object {
        fun fromWire(wire: String): DrowsySource = entries.firstOrNull { it.wire == wire }
            ?: throw IllegalArgumentException("unknown drowsiness source: \"$wire\"")
    }
}

/** 기준선 상태. 선언 순서 = `BASELINE_STATUSES` (types.py:76-79). */
enum class BaselineStatus(val wire: String) {
    OK("ok"),
    OFF_AXIS("off_axis"),
    FALLBACK_GLOBAL("fallback_global");

    internal companion object {
        fun fromWire(wire: String): BaselineStatus = entries.firstOrNull { it.wire == wire }
            ?: throw IllegalArgumentException("unknown baseline status: \"$wire\"")
    }
}

/** 전방 주시 표지 (rules.py:89-91 `ZONE_IN`, `ZONE_OUT`, `ZONE_UNKNOWN`). 표시용이고 벡터에 없다(SPEC 12.10). */
enum class GazeZone(val wire: String) {
    IN("in"),
    OUT("out"),
    UNKNOWN("unknown");

    internal companion object {
        fun fromWire(wire: String): GazeZone = entries.firstOrNull { it.wire == wire }
            ?: throw IllegalArgumentException("unknown gaze zone: \"$wire\"")
    }
}

// ---------------------------------------------------------------------------
// 모델 출력 클래스 순서 (types.py:82-85). probs[CLASS_DROWSY]가 p_drowsy다.
// ---------------------------------------------------------------------------

/** alert, low_vigilant, drowsy. 읽기 전용 목록이다. */
val CLASS_ORDER: List<String> = Collections.unmodifiableList(listOf("alert", "low_vigilant", "drowsy"))
const val CLASS_ALERT: Int = 0
const val CLASS_LOW_VIGILANT: Int = 1
const val CLASS_DROWSY: Int = 2

// ---------------------------------------------------------------------------
// 스텝 특징의 고정 순서 (types.py:88-124). feature_spec.json의 features와 같아야 하며 읽기 단계가 검사한다.
// ---------------------------------------------------------------------------

/** 스텝 특징 17개의 이름, 순서 고정. 읽기 전용 목록이다. */
val FEATURE_NAMES: List<String> = Collections.unmodifiableList(
    listOf(
        "perclos_60",
        "perclos_10",
        "closure_mean_60",
        "closure_mean_10",
        "blink_rate_60",
        "blink_dur_mean_60",
        "blink_dur_std_60",
        "closing_vel_mean_60",
        "opening_vel_mean_60",
        "long_closure_count_60",
        "max_closure_ms_60",
        "max_closure_ms_10",
        "yawn_count_60",
        "pitch_std_60",
        "nod_count_60",
        "face_missing_ratio_60",
        "face_missing_ratio_10",
    ),
)
const val N_FEATURES: Int = 17
const val F_PERCLOS_60: Int = 0
const val F_PERCLOS_10: Int = 1
const val F_CLOSURE_MEAN_60: Int = 2
const val F_CLOSURE_MEAN_10: Int = 3
const val F_BLINK_RATE_60: Int = 4
const val F_BLINK_DUR_MEAN_60: Int = 5
const val F_BLINK_DUR_STD_60: Int = 6
const val F_CLOSING_VEL_MEAN_60: Int = 7
const val F_OPENING_VEL_MEAN_60: Int = 8
const val F_LONG_CLOSURE_COUNT_60: Int = 9
const val F_MAX_CLOSURE_MS_60: Int = 10
const val F_MAX_CLOSURE_MS_10: Int = 11
const val F_YAWN_COUNT_60: Int = 12
const val F_PITCH_STD_60: Int = 13
const val F_NOD_COUNT_60: Int = 14
const val F_FACE_MISSING_RATIO_60: Int = 15
const val F_FACE_MISSING_RATIO_10: Int = 16

// ---------------------------------------------------------------------------
// FrameSignals.eye_look의 고정 순서와 위치 (types.py:127-135). 이식 안에서만 쓴다(계획 5.1절).
// ---------------------------------------------------------------------------

internal val EYE_LOOK_ORDER: List<String> = Collections.unmodifiableList(
    listOf("in_l", "out_l", "up_l", "down_l", "in_r", "out_r", "up_r", "down_r"),
)
internal const val EL_IN_L: Int = 0
internal const val EL_OUT_L: Int = 1
internal const val EL_UP_L: Int = 2
internal const val EL_DOWN_L: Int = 3
internal const val EL_IN_R: Int = 4
internal const val EL_OUT_R: Int = 5
internal const val EL_UP_R: Int = 6
internal const val EL_DOWN_R: Int = 7

/** 단위 회전(행 우선 9개, types.py:138). 전역 상수 기준선의 R0. 이식 안에서만 쓴다(계획 5.1절). */
internal val IDENTITY_R0: List<Double> = Collections.unmodifiableList(
    listOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
