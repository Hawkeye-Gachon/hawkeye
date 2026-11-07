package com.personal.driverguard.data.remote

import kotlinx.serialization.Serializable

/*
 * DriverGuard 백엔드(backend/) 요청·응답 형식. 백엔드의 XxxDto 와 필드 이름을 그대로 맞춘다.
 * 시각은 ISO-8601 문자열이다. 날짜·시각은 오프셋을 포함한 문자열(예: 2026-11-07T14:30:00+09:00)이고,
 * 날짜만 있는 값은 yyyy-MM-dd 다. 백엔드는 서울 시간(Asia/Seoul)으로 응답한다.
 * 백엔드가 응답 필드를 늘려도 앱이 깨지지 않도록 Json 은 모르는 키를 무시한다(HawkeyeApi 참고).
 */

// ── 공통 오류 ──
@Serializable
data class ApiErrorBody(
    val status: Int = 0,
    val error: String? = null,
    val code: String? = null,
    val message: String? = null,
    val path: String? = null,
    val fieldErrors: List<FieldError>? = null,
) {
    @Serializable
    data class FieldError(val field: String, val message: String)
}

// ── 인증 ──
@Serializable
data class SignupRequest(val email: String, val name: String, val password: String, val ageConfirmed: Boolean)

@Serializable
data class SignupResponse(
    val userId: Long,
    val email: String,
    val name: String,
    val grantType: String,
    val accessToken: String,
    val accessTokenExpiresIn: Long,
    val refreshToken: String,
    val termsAgreementRequired: Boolean,
)

@Serializable
data class LoginRequest(val email: String, val password: String)

@Serializable
data class RefreshTokenRequest(val refreshToken: String)

@Serializable
data class TokenResponse(
    val grantType: String,
    val accessToken: String,
    /** 초 단위 */
    val accessTokenExpiresIn: Long,
    val refreshToken: String,
    val termsAgreementRequired: Boolean,
)

@Serializable
data class WithdrawRequest(val password: String)

// ── 사용자 ──
@Serializable
data class MeResponse(
    val userId: Long,
    val email: String,
    val name: String,
    /** ACTIVE / WITHDRAWN */
    val status: String,
    val createdAt: String? = null,
    val termsAgreementRequired: Boolean,
)

// ── 약관 ──
@Serializable
data class TermsSummary(
    val termsId: Long,
    val code: String,
    val title: String,
    val version: Int,
    val required: Boolean,
    val effectiveAt: String? = null,
)

@Serializable
data class TermsListResponse(val terms: List<TermsSummary>)

@Serializable
data class TermsDetail(
    val termsId: Long,
    val code: String,
    val title: String,
    val version: Int,
    val required: Boolean,
    val effectiveAt: String? = null,
    /** 마크다운 */
    val content: String,
)

@Serializable
data class AgreeRequest(val agreements: List<AgreeItem>)

@Serializable
data class AgreeItem(val termsId: Long, val agreed: Boolean)

@Serializable
data class AgreementsResponse(val termsAgreementRequired: Boolean, val agreements: List<AgreementItem>)

@Serializable
data class AgreementItem(
    val termsId: Long,
    val code: String,
    val title: String,
    val version: Int,
    val required: Boolean,
    val agreed: Boolean? = null,
    val agreedAt: String? = null,
)

// ── 쉼터 ──
object PlaceType {
    const val REST_AREA = "REST_AREA"
    const val DROWSY_SHELTER = "DROWSY_SHELTER"
}

@Serializable
data class NearbyRequest(
    val lat: Double,
    val lng: Double,
    /** 진행 방향(북쪽 0도, 시계 방향). 모르면 null */
    val heading: Double? = null,
    /** m/s. 백엔드는 2.8m/s 이상일 때만 heading 을 쓴다 */
    val speed: Double? = null,
    /** null 이면 휴게소·졸음쉼터 모두 */
    val type: String? = null,
)

@Serializable
data class NearbyResponse(
    val directionApplied: Boolean,
    /** 고속도로 위인지. false 면 place 가 null(일반 도로라 안내하지 않음), null 이면 확인 못함 */
    val onHighway: Boolean? = null,
    /** ROAD / UNAVAILABLE */
    val distanceSource: String? = null,
    val place: NearestPlace? = null,
)

@Serializable
data class NearestPlace(
    val placeId: Long,
    /** REST_AREA / DROWSY_SHELTER */
    val type: String,
    val name: String,
    val latitude: Double,
    val longitude: Double,
    val roadDistanceMeters: Int,
    val durationSeconds: Int,
)

// ── 주행 ──
object EventType {
    const val INATTENTION = "INATTENTION"
    const val DROWSINESS = "DROWSINESS"
}

object NavMode {
    const val WAYPOINT = "WAYPOINT"
    const val DESTINATION = "DESTINATION"
}

object TriggerType {
    const val DROWSINESS = "DROWSINESS"
    const val INATTENTION_REPEATED = "INATTENTION_REPEATED"
}

@Serializable
data class StartDriveRequest(val clientDriveId: String, val startedAt: String)

@Serializable
data class StartDriveResponse(
    val driveId: Long,
    /** IN_PROGRESS / ENDED / ABANDONED */
    val status: String,
    val startedAt: String,
    val abandonedDriveId: Long? = null,
)

@Serializable
data class EndDriveRequest(
    val endedAt: String,
    val distanceMeters: Int,
    val events: List<EventRequest>,
    /** 단순화한 경로 [[위도, 경도], ...] */
    val route: List<List<Double>>,
    val waypoints: List<WaypointRequest>,
)

@Serializable
data class EventRequest(
    val type: String,
    val detectedAt: String,
    val latitude: Double? = null,
    val longitude: Double? = null,
)

@Serializable
data class WaypointRequest(
    val placeId: Long,
    val mode: String,
    val trigger: String,
    /** NaviHelper 성공 콜백을 받은 시각 */
    val addedAt: String,
    val visitedAt: String? = null,
)

@Serializable
data class DriveReport(
    val driveId: Long,
    val status: String,
    val startedAt: String,
    val endedAt: String? = null,
    val observedSeconds: Int? = null,
    val distanceMeters: Int? = null,
    val inattentionCount: Int? = null,
    val drowsinessCount: Int? = null,
    val route: Route? = null,
    val events: List<EventItem> = emptyList(),
    val waypoints: List<WaypointItem> = emptyList(),
    /** 종료 업로드 응답에만 있다 */
    val skipped: Skipped? = null,
) {
    @Serializable
    data class Route(val start: Point? = null, val end: Point? = null, val points: List<List<Double>> = emptyList())

    @Serializable
    data class Point(val latitude: Double, val longitude: Double)

    @Serializable
    data class EventItem(val type: String, val detectedAt: String, val latitude: Double? = null, val longitude: Double? = null)

    @Serializable
    data class WaypointItem(
        val waypointId: Long,
        val placeId: Long,
        val placeName: String? = null,
        val placeType: String? = null,
        val mode: String,
        val trigger: String,
        val addedAt: String,
        val visited: Boolean = false,
        val visitedAt: String? = null,
    )

    @Serializable
    data class Skipped(val events: Int, val waypoints: Int)
}

@Serializable
data class DriveListItem(
    val driveId: Long,
    val startedAt: String,
    val endedAt: String? = null,
    val observedSeconds: Int? = null,
    val distanceMeters: Int? = null,
    val inattentionCount: Int? = null,
    val drowsinessCount: Int? = null,
)

@Serializable
data class DriveListResponse(val drives: List<DriveListItem>, val nextCursor: String? = null)

// ── 이전 기록 ──
@Serializable
data class CalendarResponse(val month: String, val driveDates: List<String>)

@Serializable
data class ReportSummary(
    val driveCount: Int,
    val observedSeconds: Int,
    val distanceMeters: Int,
    val inattentionCount: Int,
    val drowsinessCount: Int,
)

@Serializable
data class HourItem(
    val hour: Int,
    /** 0 이면 '관찰 기록 없음', 0 보다 크고 횟수가 0 이면 '0회' */
    val observedSeconds: Int,
    val inattentionCount: Int,
    val drowsinessCount: Int,
)

@Serializable
data class DayItem(
    val date: String,
    /** MONDAY ~ SUNDAY */
    val dayOfWeek: String,
    val driveCount: Int,
    val observedSeconds: Int,
    val inattentionCount: Int,
    val drowsinessCount: Int,
)

@Serializable
data class Peak(val inattention: PeakItem, val drowsiness: PeakItem)

/** 가장 많이 감지된 시간대. 같은 횟수면 여러 개, 감지가 없으면 hours = [], count = 0 */
@Serializable
data class PeakItem(val hours: List<Int>, val count: Int)

@Serializable
data class DailyReport(
    val date: String,
    val summary: ReportSummary,
    /** 항상 0~23시 24개 */
    val hourly: List<HourItem>,
    val peak: Peak,
    val drives: List<DriveListItem>,
)

@Serializable
data class WeeklyReport(
    val weekStart: String,
    val weekEnd: String,
    val summary: ReportSummary,
    /** 항상 월~일 7개 */
    val daily: List<DayItem>,
    /** 한 주 동안 같은 시간대 합산 24개 */
    val hourly: List<HourItem>,
    val peak: Peak,
)
