package com.personal.driverguard.ui.state

import android.os.SystemClock
import com.personal.driverguard.data.remote.NearestPlace
import com.personal.driverguard.data.remote.PlaceType
import com.personal.driverguard.domain.engine.DistanceAccumulator
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * 테스트용 모의 주행. 실내·정차 상태로 시험하면 위치가 움직이지 않아 리포트 지도와 거리가 비므로,
 * debug 빌드에서는 관찰 중 실제 GPS 대신 이 경로를 따라 1초마다 위치를 찍는다(HawkeyeViewModel.startDrive).
 * 정식 배포 전에는 [DebugFlags.DUMMY_ROUTE] 를 끄거나 지운다.
 *
 * 경로: 경부고속도로 양재IC → 기흥IC 부근을 대략 따라가는 좌표(시험용 근사치, 실제 도로와 정확히 맞지 않음).
 */
object DummyRoute {
    /** 꺾이는 지점(위도, 경도) */
    private val anchors = listOf(
        37.4687 to 127.0390, // 양재IC 부근
        37.4310 to 127.0560,
        37.4045 to 127.0640, // 판교 부근
        37.3720 to 127.0790,
        37.3446 to 127.0960, // 신갈 부근
        37.3050 to 127.1080,
        37.2747 to 127.1157, // 기흥IC 부근
    )

    /** 구간마다 나눌 점 수(총 점 = 구간 수 × 이 값 + 1) */
    private const val STEPS_PER_SEGMENT = 8

    fun points(): List<List<Double>> {
        val out = mutableListOf<List<Double>>()
        anchors.zipWithNext().forEachIndexed { seg, (a, b) ->
            for (i in 0 until STEPS_PER_SEGMENT) {
                val t = i.toDouble() / STEPS_PER_SEGMENT
                // 직선만 이어지지 않게 구간 가운데를 살짝 휘게 한다
                val bend = sin(t * PI) * 0.0012 * (if (seg % 2 == 0) 1 else -1)
                out += listOf(a.first + (b.first - a.first) * t, a.second + (b.second - a.second) * t + bend)
            }
        }
        out += listOf(anchors.last().first, anchors.last().second)
        return out.map { p -> p.map { (it * 1e6).roundToInt() / 1e6 } }
    }

    fun lengthMeters(points: List<List<Double>>): Int =
        points.zipWithNext().sumOf { (a, b) -> DistanceAccumulator.haversineMeters(a[0], a[1], b[0], b[1]) }.roundToInt()

    /** 모의 주행 속도: 시속 100km */
    const val DRIVE_SPEED_MPS = 27.8f

    /** 모의 위치를 내보내는 간격(실제 GPS 와 같은 1초) */
    const val DRIVE_INTERVAL_MS = 1_000L

    /** 쉼터에 들렀을 때 멈춰 있는 시간 */
    const val REST_MS = 20_000L

    /**
     * 모의 주행: 경로를 따라 [DRIVE_SPEED_MPS] 로 달리며 [DRIVE_INTERVAL_MS] 마다 위치를 내보낸다.
     * 시간에 맞는 속도로 움직이므로 서버의 거리·속도 검사를 통과한다. 경로 끝(기흥IC)에 닿으면 그 자리에 멈춰 있는다.
     * [restStop] 이 쉼터를 주면 그쪽으로 곧장 가서 [REST_MS] 동안 멈췄다가(속도 0) 떠났던 지점으로 돌아와 경로를 이어 간다.
     */
    fun drive(now: () -> Long = { SystemClock.elapsedRealtime() }, restStop: () -> Pair<Double, Double>? = { null }): Flow<GeoFix> = flow {
        val route = points()
        val legs = route.zipWithNext().map { (a, b) -> DistanceAccumulator.haversineMeters(a[0], a[1], b[0], b[1]) }
        val step = DRIVE_SPEED_MPS * DRIVE_INTERVAL_MS / 1000.0
        val visited = mutableSetOf<Pair<Double, Double>>()
        var travelled = 0.0
        while (true) {
            val here = positionAt(route, legs, travelled, now())
            val stop = restStop()?.takeIf { it !in visited }
            if (stop == null) {
                emit(here)
                delay(DRIVE_INTERVAL_MS)
                travelled += step
                continue
            }
            // 쉼터로 갔다가 쉬고, 떠났던 지점으로 돌아온다
            var lat = here.latitude; var lng = here.longitude
            suspend fun moveTo(tLat: Double, tLng: Double) {
                while (true) {
                    val left = DistanceAccumulator.haversineMeters(lat, lng, tLat, tLng)
                    val f = if (left <= step) 1.0 else step / left
                    val head = bearing(listOf(lat, lng), listOf(tLat, tLng))
                    lat += (tLat - lat) * f; lng += (tLng - lng) * f
                    delay(DRIVE_INTERVAL_MS)
                    emit(GeoFix(lat, lng, 5f, head, if (f >= 1.0) 0f else DRIVE_SPEED_MPS, now()))
                    if (f >= 1.0) return
                }
            }
            moveTo(stop.first, stop.second)
            repeat((REST_MS / DRIVE_INTERVAL_MS).toInt()) {
                delay(DRIVE_INTERVAL_MS)
                emit(GeoFix(stop.first, stop.second, 5f, null, 0f, now()))
            }
            visited += stop
            moveTo(here.latitude, here.longitude)
        }
    }

    /** 경로 시작에서 [meters] 만큼 간 지점 */
    internal fun positionAt(route: List<List<Double>>, legs: List<Double>, meters: Double, at: Long): GeoFix {
        var left = meters
        legs.forEachIndexed { i, len ->
            if (left <= len) {
                val a = route[i]; val b = route[i + 1]
                val t = if (len == 0.0) 0.0 else left / len
                return GeoFix(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 5f, bearing(a, b), DRIVE_SPEED_MPS, at)
            }
            left -= len
        }
        val a = route[route.lastIndex - 1]; val b = route.last()
        return GeoFix(b[0], b[1], 5f, bearing(a, b), 0f, at)
    }

    private fun bearing(a: List<Double>, b: List<Double>): Float {
        val p1 = Math.toRadians(a[0]); val p2 = Math.toRadians(b[0]); val dl = Math.toRadians(b[1] - a[1])
        val y = sin(dl) * cos(p2)
        val x = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl)
        return ((Math.toDegrees(atan2(y, x)) + 360) % 360).toFloat()
    }

    /** 쉼터 조회용 더미 위치: 판교 부근, 부산 방향(남쪽)으로 시속 100km 주행 중 */
    fun fix() = GeoFix(37.4045, 127.0640, 10f, 175f, 27.8f, SystemClock.elapsedRealtime())

    /** 서버가 쉼터를 찾지 못했을 때 쓰는 더미 쉼터(시험용 근사 좌표) */
    fun restPlace() = NearestPlace(
        placeId = 9001, type = PlaceType.REST_AREA, name = "죽전휴게소",
        latitude = 37.3290, longitude = 127.1050, roadDistanceMeters = 8_400, durationSeconds = 330,
    )
}
