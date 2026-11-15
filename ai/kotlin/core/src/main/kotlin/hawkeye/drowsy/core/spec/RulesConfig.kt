package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.DrowsySource
import hawkeye.drowsy.core.EventType

// 원본: drowsy/spec.py:352-506 (config/rules.yaml, 앱에는 export/rules.json). 계획 4.1절의 선언 그대로다.
// 필드 순서는 Python과 같다. 열거형 칸(drowsiness.source, alert_priority)은 읽을 때 문자열로 두고 Python과 같은 자리
// (spec.py:900, 919-922)에서 같은 문장으로 검사한 뒤에 열거형으로 바꾼다(계획 5.10절, SpecLoader).

/** 전방 주시 영역 (spec.py:352-367). */
data class ForwardZoneRules(
    val yawAbsMaxDeg: Double,
    val pitchUpMaxDeg: Double,
    val pitchDownMaxDeg: Double,
    val eyeAuxEnabled: Boolean,
    val gazeHAbsMax: Double,
    val gazeVAbsMax: Double,
    val debounceMs: Long,
    val faceLossCreditMs: Long,
)

/** 장시간 시선 이탈 (spec.py:370-374). */
data class GazeAwayLongRules(val onsetMs: Long)

/** VATS (spec.py:377-384). */
data class VatsRules(val windowMs: Long, val cumulativeMs: Long, val resetEnabled: Boolean, val resetForwardMs: Long)

/** 눈 감김 규칙 (spec.py:387-396). detector는 규칙용 검출기(진입 0.7 / 이탈 0.5)다. */
data class ClosureRules(val detector: RunDetectorSpec, val microsleepMs: Long, val sleepMs: Long)

/** 모델 졸음 판정 (spec.py:399-407). */
data class ModelDrowsyRules(
    val onsetP: Double,
    val onsetSteps: Int,
    val endP: Double,
    val endSteps: Int,
    val refractoryMs: Long,
)

/** PERCLOS 졸음 판정 (spec.py:410-421). */
data class PerclosDrowsyRules(
    val onsetRatio: Double,
    val onsetSteps: Int,
    val endRatio: Double,
    val endSteps: Int,
    val refractoryMs: Long,
)

/** 졸음 판정원 (spec.py:424-433). */
data class DrowsinessRules(val source: DrowsySource, val model: ModelDrowsyRules, val perclos: PerclosDrowsyRules)

/** 분석 불가 (spec.py:436-441). */
data class UnableToAnalyzeRules(val windowMs: Long, val minFaceRatio: Double)

/** 이벤트 종류 하나의 심각도와 경고 정책 (spec.py:444-454). */
data class EventPolicy(val severity: Int, val cooldownMs: Long, val repeatMs: Long, val minSpeedKmh: Double)

/** 이벤트 종류별 정책 (spec.py:457-485). */
data class EventPolicies(
    val calibrating: EventPolicy,
    val unableToAnalyze: EventPolicy,
    val gazeAwayLong: EventPolicy,
    val vats: EventPolicy,
    val drowsy: EventPolicy,
    val microsleep: EventPolicy,
    val sleep: EventPolicy,
) {
    /** 종류로 정책을 찾는다 (spec.py:469-485). 열거형이라 모르는 종류가 없다. */
    fun get(type: EventType): EventPolicy = when (type) {
        EventType.CALIBRATING -> calibrating
        EventType.UNABLE_TO_ANALYZE -> unableToAnalyze
        EventType.GAZE_AWAY_LONG -> gazeAwayLong
        EventType.VATS -> vats
        EventType.DROWSY -> drowsy
        EventType.MICROSLEEP -> microsleep
        EventType.SLEEP -> sleep
    }
}

/**
 * 규칙 엔진과 경고 결정의 설정 전체 (spec.py:488-506).
 * clock은 pipeline.clock과 같아야 한다(파이프라인을 만들 때 검사한다, pipeline.py:90-91).
 * alertPriority는 높은 순위부터이고 모든 종류가 한 번씩 들어 있다(읽을 때 검사한다).
 */
data class RulesConfig(
    val clock: ClockSpec,
    val forwardZone: ForwardZoneRules,
    val gazeAwayLong: GazeAwayLongRules,
    val vats: VatsRules,
    val closure: ClosureRules,
    val drowsiness: DrowsinessRules,
    val unableToAnalyze: UnableToAnalyzeRules,
    val events: EventPolicies,
    val alertPriority: List<EventType>,
)
