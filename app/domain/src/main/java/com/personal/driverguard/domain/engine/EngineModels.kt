package com.personal.driverguard.domain.engine

import com.personal.driverguard.domain.analysis.temporal.ClosedEyeRatio
import com.personal.driverguard.domain.analysis.temporal.EyeState
import com.personal.driverguard.domain.analysis.temporal.EyeThresholds
import com.personal.driverguard.entity.AlertKind
import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.CautionCause
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.DrowsinessLevel
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.QualityFlag

/**
 * 판정 규칙 ID. 이벤트 레코드의 rules 에 남겨 어떤 규칙으로 판정했는지 재현할 수 있게 한다.
 * 임계값은 모두 AnalysisConfig 에 있다.
 */
object JudgmentRules {
    /** 양쪽 눈 감김이 eyeClosureEventMs 이상 연속 → 주의 */
    const val EYE_SUSTAINED = "EYE_SUSTAINED"

    /** 눈 감김 비율 ≥ closedRatioDrowsyThreshold (유효 관측률 충족 시) → 졸음 의심 */
    const val CLOSED_RATIO = "CLOSED_RATIO"

    /** repeatWindowMs 안에 지속 눈 감김 repeatedClosureCount 회 이상 → 졸음 의심 */
    const val REPEATED_CLOSURE = "REPEATED_CLOSURE"

    /** repeatWindowMs 안에 입 벌림 yawnSupportCount 회 이상 + 지속 눈 감김 1회 이상 → 졸음 의심 */
    const val YAWN_WITH_CLOSURE = "YAWN_WITH_CLOSURE"

    /** 졸음 의심 중 입 벌림 반복이 함께 관찰됨(상향) */
    const val SUPPORT_YAWN = "SUPPORT_YAWN"

    /** 졸음 의심 중 고개 숙임 반복이 함께 관찰됨(실험, 설정으로 켰을 때만) */
    const val SUPPORT_NOD = "SUPPORT_NOD"

    /** 기준 대비 yaw·pitch 이탈이 headAwayEventMs 이상 지속 → 시각적 부주의 의심 */
    const val HEAD_AWAY = "HEAD_AWAY"

    /** 입술 안쪽 벌림 비율이 mouthOpenEventMs 이상 지속 → 지속적 입 벌림(하품 의심) */
    const val MOUTH_SUSTAINED = "MOUTH_SUSTAINED"

    /** 고개 숙임 nodRepeatCount 회 이상(실험) */
    const val NOD_REPEATED = "NOD_REPEATED"

    /** 눈 지표를 쓸 수 없는 상태가 unavailableNoticeMs 이상 지속 */
    const val UNAVAILABLE = "UNAVAILABLE"
}

/** 엔진이 내보내는 이벤트 레코드 변경. 시각은 단조 시계. key 는 엔진 안에서만 유일하다. */
sealed interface EventChange {
    val key: Long

    data class Opened(
        override val key: Long,
        val type: EventType,
        val candidateStartMs: Long,
        val confirmedAtMs: Long,
        val level: AlertLevel,
        val rules: List<String>,
        val metrics: String,
    ) : EventChange

    data class Updated(
        override val key: Long,
        val level: AlertLevel,
        val rules: List<String>,
        val metrics: String,
    ) : EventChange

    data class Closed(
        override val key: Long,
        val endMs: Long,
        val durationMs: Long,
        val reason: EventEndReason,
        val metrics: String,
    ) : EventChange
}

/** 엔진 안에서 이벤트 레코드의 열림·갱신·닫힘을 모아 둔다. 프레임마다 같은 이벤트를 새로 만들지 않는다. */
class EventLedger {
    private var nextKey = 1L
    private val pending = mutableListOf<EventChange>()
    private val openTypes = mutableMapOf<Long, EventType>()

    val openCount: Int get() = openTypes.size

    fun open(type: EventType, candidateStartMs: Long, confirmedAtMs: Long, level: AlertLevel, rules: List<String>, metrics: String): Long {
        val key = nextKey++
        openTypes[key] = type
        pending += EventChange.Opened(key, type, candidateStartMs, confirmedAtMs, level, rules, metrics)
        return key
    }

    fun update(key: Long, level: AlertLevel, rules: List<String>, metrics: String) {
        if (key in openTypes) pending += EventChange.Updated(key, level, rules, metrics)
    }

    fun close(key: Long, endMs: Long, durationMs: Long, reason: EventEndReason, metrics: String) {
        if (openTypes.remove(key) != null) pending += EventChange.Closed(key, endMs, durationMs, reason, metrics)
    }

    fun drain(): List<EventChange> = pending.toList().also { pending.clear() }
}

/** 개발 화면용 지표 묶음 */
data class EngineSnapshot(
    val timestampMs: Long,
    val faceQuality: QualityFlag,
    val eyeQuality: QualityFlag,
    val mouthQuality: QualityFlag,
    val poseQuality: QualityFlag,
    val faceCount: Int,
    val leftEar: Double?,
    val rightEar: Double?,
    val filteredLeftEar: Double?,
    val filteredRightEar: Double?,
    val thresholds: EyeThresholds?,
    val eyeState: EyeState,
    val eyeClosedMs: Long,
    val closedRatio: ClosedEyeRatio,
    val mouthRatio: Double?,
    val mouthOpenThreshold: Double?,
    val mouthOpenMs: Long,
    val yawDeg: Double?,
    val pitchDeg: Double?,
    val rollDeg: Double?,
    val headAwayMs: Long,
    val nodCount: Int,
    val closureCount: Int,
    val yawnCount: Int,
    val unavailableMs: Long,
    val drowsinessRules: Set<String>,
)

data class EngineOutput(
    val display: DisplayState,
    val cautionCause: CautionCause?,
    val drowsiness: DrowsinessLevel,
    val distraction: Boolean,
    /** 화면에 보여 줄 분석 불가 사유(눈 지표 품질) */
    val unavailableReason: QualityFlag?,
    val changes: List<EventChange>,
    /** 이번 갱신에서 엔진이 요청한 경고. 반복 제한은 AlertPolicy 가 한다. */
    val alertRequests: Set<AlertKind>,
    val snapshot: EngineSnapshot?,
)
