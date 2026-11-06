package com.personal.driverguard.entity

data class TripVo(
    val id: Long,
    val startedAtMs: Long,
    val endedAtMs: Long?,
    val durationMs: Long?,
    /** 위치 기록이 없으면 null */
    val distanceMeters: Double?,
    val mode: AnalysisMode,
    /** 눈 지표를 판정에 쓸 수 있었던 시간 */
    val validObservationMs: Long?,
    /** 분석 불가(얼굴 미검출·지표 무효·프레임 공백) 시간. 휴식·앱 중단 시간은 빠진다. */
    val unavailableMs: Long?,
    /** AnalysisConfig.versionTag() */
    val configVersion: String,
    /** 모델 파일 이름과 SHA-256 앞부분. 시연이면 null */
    val modelInfo: String?,
    /** 캘리브레이션 기준값 요약. 시연이면 시연용 기준값 */
    val calibrationSummary: String?,
    /** 앱이 비정상 종료되어 다음 실행 때 정리된 여정 */
    val endedAbnormally: Boolean,
)

data class TripListItemVo(
    val trip: TripVo,
    val eventCount: Int,
)

/**
 * 저장되는 이벤트·판정. 시각은 모두 벽시계(epoch ms).
 * 후보 시작(조건이 처음 관측됨) → 확인(지속 조건 충족) → 종료 를 구분해 남긴다.
 */
data class DriverEventVo(
    val id: Long = 0,
    val tripId: Long,
    val type: EventType,
    val level: AlertLevel,
    val candidateStartedAtMs: Long,
    val confirmedAtMs: Long,
    val endedAtMs: Long?,
    /** 유효 관측으로 확인된 지속 시간. 공백은 넣지 않는다. */
    val durationMs: Long?,
    val endReason: EventEndReason?,
    /** 판정에 적용된 규칙 ID 목록(쉼표 구분). 관측 이벤트면 해당 지표 규칙. */
    val rules: String,
    /** 관련 지표 요약(예: "minEAR=0.12, ratio=0.21") */
    val metrics: String,
    val latitude: Double?,
    val longitude: Double?,
    val locationAccuracyMeters: Float?,
    /** 위치 측정 시각(벽시계) */
    val locationAtMs: Long?,
    val source: AnalysisMode,
)

/** 사용자가 직접 기록한 휴식 */
data class RestVo(
    val id: Long,
    val tripId: Long,
    val startedAtMs: Long,
    val endedAtMs: Long?,
    val durationMs: Long?,
)

/** 분석 중단 구간(휴식, 앱 백그라운드, 카메라 오류) */
data class InterruptionVo(
    val id: Long,
    val tripId: Long,
    val reason: InterruptionReason,
    val startedAtMs: Long,
    val endedAtMs: Long?,
    val durationMs: Long?,
)

data class TripDetailVo(
    val trip: TripVo,
    val events: List<DriverEventVo>,
    val rests: List<RestVo>,
    val interruptions: List<InterruptionVo>,
)

data class EventTypeSummary(
    val type: EventType,
    val count: Int,
    /** 지속 시간을 아는 이벤트들의 합 */
    val totalDurationMs: Long,
    /** 비정상 종료 등으로 지속 시간을 모르는 이벤트 수 */
    val unknownDurationCount: Int,
)
