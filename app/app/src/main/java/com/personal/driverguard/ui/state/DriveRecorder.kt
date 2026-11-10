package com.personal.driverguard.ui.state

import com.personal.driverguard.data.remote.EndDriveRequest
import com.personal.driverguard.data.remote.EventRequest
import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.WaypointRequest
import com.personal.driverguard.domain.engine.DistanceAccumulator
import com.personal.driverguard.entity.LocationFix
import kotlinx.serialization.Serializable
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.temporal.ChronoUnit
import java.util.UUID

/** 백엔드가 쓰는 시간대. 보내는 시각은 모두 이 시간대의 오프셋을 붙인 ISO 문자열이다. */
val SEOUL: ZoneId = ZoneId.of("Asia/Seoul")

fun nowIso(): String = OffsetDateTime.now(SEOUL).truncatedTo(ChronoUnit.SECONDS).toString()

/**
 * 한 번의 관찰(주행)을 앱 안에 기록한다. 백엔드 설계에 따라 주행 중에는 서버와 통신하지 않고,
 * 감지 기록·단순화한 경로·이동 거리·경유지를 모았다가 종료할 때 [toEndRequest] 로 한 번에 올린다.
 *
 * 스레드 안전하지 않다. ViewModel 이 메인 스레드에서만 부른다.
 */
class DriveRecorder(
    val clientDriveId: String = UUID.randomUUID().toString(),
    val startedAt: String = nowIso(),
    /** elapsedRealtime 기준 시작 시각(경과 시간 표시용) */
    val startedMonotonicMs: Long,
) {
    /** 서버가 준 주행 ID. 시작 등록에 실패했으면 null 이고 종료할 때 다시 등록한다. */
    var driveId: Long? = null

    private val events = mutableListOf<EventRequest>()
    /** 추가한 쉼터. 위치가 쉼터 가까이에서 멈추면 [visitedAt] 을 채운다. */
    private class Waypoint(val request: WaypointRequest, val latitude: Double?, val longitude: Double?, var visitedAt: String? = null)
    private val waypoints = mutableListOf<Waypoint>()
    private val route = mutableListOf<List<Double>>()
    private val distance = DistanceAccumulator()

    var lastFix: GeoFix? = null
        private set

    val inattentionCount: Int get() = events.count { it.type == EventType.INATTENTION }
    val drowsinessCount: Int get() = events.count { it.type == EventType.DROWSINESS }

    fun onLocation(fix: GeoFix, at: String = nowIso()) {
        val previous = lastFix
        lastFix = fix
        checkArrival(fix, previous, at)
        distance.add(LocationFix(fix.latitude, fix.longitude, fix.accuracyMeters, fix.monotonicMs))
        val accurate = fix.accuracyMeters?.let { it <= MAX_ROUTE_ACCURACY_M } ?: false
        if (!accurate) return
        val last = route.lastOrNull()
        if (last == null || DistanceAccumulator.haversineMeters(last[0], last[1], fix.latitude, fix.longitude) >= ROUTE_STEP_M) {
            route += listOf(fix.latitude, fix.longitude)
        }
    }

    /** 감지 1건. 위치는 가장 최근 위치를 붙인다(없으면 비운다). */
    fun recordEvent(type: String, at: String = nowIso()) {
        val fix = lastFix
        events += EventRequest(type, at, fix?.latitude, fix?.longitude)
    }

    /** 차량 내비가 쉼터 추가를 확인했을 때. 쉼터 좌표가 있으면 도착(정차)을 감지해 방문 시각을 남긴다. */
    fun recordWaypoint(placeId: Long, mode: String, trigger: String, at: String = nowIso(), latitude: Double? = null, longitude: Double? = null) {
        waypoints += Waypoint(WaypointRequest(placeId, mode, trigger, at), latitude, longitude)
    }

    /** 아직 들르지 않은 쉼터 중 가장 먼저 추가한 곳(모의 주행이 찾아갈 곳) */
    fun pendingStop(): Pair<Double, Double>? =
        waypoints.firstOrNull { it.visitedAt == null && it.latitude != null && it.longitude != null }?.let { it.latitude!! to it.longitude!! }

    /** 쉼터 [ARRIVAL_RADIUS_M] 안에서 거의 멈췄으면(속도 [STOPPED_MPS] 이하) 그 쉼터를 방문으로 기록한다. */
    private fun checkArrival(fix: GeoFix, previous: GeoFix?, at: String) {
        if (waypoints.none { it.visitedAt == null }) return
        val speed = fix.speedMps?.toDouble() ?: previous?.let {
            val sec = (fix.monotonicMs - it.monotonicMs) / 1000.0
            if (sec > 0) DistanceAccumulator.haversineMeters(it.latitude, it.longitude, fix.latitude, fix.longitude) / sec else null
        } ?: return
        if (speed > STOPPED_MPS) return
        waypoints.filter { it.visitedAt == null && it.latitude != null && it.longitude != null }
            .filter { DistanceAccumulator.haversineMeters(it.latitude!!, it.longitude!!, fix.latitude, fix.longitude) <= ARRIVAL_RADIUS_M }
            .forEach { it.visitedAt = at }
    }

    fun toEndRequest(endedAt: String = nowIso()): EndDriveRequest = EndDriveRequest(
        endedAt = endedAt,
        distanceMeters = distance.distanceOrNull()?.toInt() ?: 0,
        events = events.takeLast(MAX_ITEMS).toList(),
        route = thin(route, MAX_ITEMS),
        waypoints = waypoints.takeLast(MAX_WAYPOINTS).map { it.request.copy(visitedAt = it.visitedAt) },
    )

    companion object {
        /** 경로 점 사이 최소 간격 */
        const val ROUTE_STEP_M = 25.0

        /** 쉼터 도착으로 보는 거리 */
        const val ARRIVAL_RADIUS_M = 300.0

        /** 정차로 보는 속도(약 시속 7km) */
        const val STOPPED_MPS = 2.0

        /** 경로에 넣을 위치의 최대 오차 */
        const val MAX_ROUTE_ACCURACY_M = 50f

        /** 백엔드 제한: events·route 최대 2,000, waypoints 최대 20 */
        const val MAX_ITEMS = 2_000
        const val MAX_WAYPOINTS = 20

        /** 점이 너무 많으면 처음·끝을 남기고 고르게 솎아 낸다 */
        fun thin(points: List<List<Double>>, max: Int): List<List<Double>> {
            if (points.size <= max) return points.toList()
            val step = (points.size - 1).toDouble() / (max - 1)
            return List(max) { i -> points[(i * step).toInt().coerceAtMost(points.lastIndex)] }
        }
    }
}

/** 종료 업로드에 실패한 주행. 다음 실행 때 다시 올린다. */
@Serializable
data class PendingUpload(
    val clientDriveId: String,
    val startedAt: String,
    val driveId: Long?,
    val body: EndDriveRequest,
)
