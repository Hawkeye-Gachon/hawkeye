package com.personal.driverguard.ui.state

import android.os.SystemClock
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * 모의 주행. [route] 를 따라 [speedMps] 로 달리며 [intervalMs] 마다 위치를 내보낸다(실제 GPS 처럼 1초 간격).
 *
 * 경유지 쉼터가 생기면([fixes] 의 pendingStop) 그쪽으로 가서 멈춘다.
 *  - [arriveWithinMs] 가 있으면(시연) 그 시간 안에 도착하도록 빨리 감는다. 위치 시각은 같은 속도로 달린 것처럼 앞당겨
 *    거리·속도 계산이 어긋나지 않게 하고, 경로가 곧은 선으로 끊기지 않게 중간 위치도 촘촘히 보낸다.
 *  - 없으면 같은 속도로 달려간다(서버 시험용).
 *  - 쉼터에서는 [restMs] 동안 멈춰 있다(null 이면 [resume] 을 부를 때까지). 그다음 경로로 돌아와 이어 달린다.
 * 경로 끝(부산)에 닿으면 그 자리에 멈춰 있는다.
 */
class DriveSimulator(
    private val route: DemoRoute,
    private val speedMps: Double = DEMO_SPEED_MPS,
    private val arriveWithinMs: Long? = null,
    private val restMs: Long? = null,
    private val intervalMs: Long = 1_000L,
    private val clock: () -> Long = { SystemClock.elapsedRealtime() },
    startMeters: Double = 0.0,
) {
    var travelledMeters: Double = startMeters
        private set

    /** 빨리 감은 만큼 앞당긴 위치 시각 */
    private var skewMs = 0L
    private val resumes = Channel<Unit>(Channel.CONFLATED)

    /** 쉼터에서 다시 출발한다 */
    fun resume() {
        resumes.trySend(Unit)
    }

    fun fixes(pendingStop: () -> LatLng? = { null }): Flow<GeoFix> = flow {
        val visited = mutableSetOf<LatLng>()
        val cruiseStep = speedMps * intervalMs / 1000.0
        while (true) {
            val stop = pendingStop()?.takeIf { it !in visited }
            if (stop == null) {
                val atEnd = travelledMeters >= route.lengthMeters
                emit(fix(route.positionAt(travelledMeters), route.bearingAt(travelledMeters), if (atEnd) 0.0 else speedMps))
                delay(intervalMs)
                travelledMeters = min(travelledMeters + cruiseStep, route.lengthMeters)
                continue
            }
            // 1) 쉼터 진입 지점까지 경로를 따라가고, 마지막에 쉼터로 들어간다
            val start = travelledMeters
            val exitMeters = max(route.project(stop.lat, stop.lng).first, start)
            val exit = route.positionAt(exitMeters)
            val along = exitMeters - start
            val hop = DemoRoute.dist(exit, stop)
            val total = along + hop
            fun at(d: Double): LatLng = if (d <= along) route.positionAt(start + d) else DemoRoute.lerp(exit, stop, if (hop == 0.0) 1.0 else (d - along) / hop)
            val tickStep = arriveWithinMs?.let { total / max(1L, it / intervalMs) } ?: cruiseStep
            var d = 0.0
            var prev = at(0.0)
            while (d < total) {
                delay(intervalMs)
                val target = min(total, d + tickStep)
                // 빨리 감기: 한 번에 멀리 가면 중간 위치를 촘촘히 보내고, 시각은 같은 속도로 달린 만큼 앞당긴다
                val parts = max(1, ceil((target - d) / MAX_FIX_SPACING_M).toInt())
                val stepPerPart = (target - d) / parts
                repeat(parts) { k ->
                    d = if (k == parts - 1) target else d + stepPerPart
                    // 이번 틱의 실제 1초는 빼고(k == 0), 달린 거리만큼의 시간을 더한다
                    if (arriveWithinMs != null) skewMs += (stepPerPart / speedMps * 1000).toLong() - if (k == 0) intervalMs else 0
                    val p = at(d)
                    emit(fix(p, DemoRoute.bearing(prev, p), if (d >= total) 0.0 else speedMps))
                    prev = p
                }
            }
            travelledMeters = exitMeters
            // 2) 쉼터에서 멈춰 있기
            resumes.tryReceive()
            if (restMs != null) {
                repeat((restMs / intervalMs).toInt()) { delay(intervalMs); emit(fix(stop, null, 0.0)) }
            } else {
                while (resumes.tryReceive().isFailure) { delay(intervalMs); emit(fix(stop, null, 0.0)) }
            }
            visited += stop
            // 3) 경로로 돌아오기
            var back = 0.0
            prev = stop
            while (back < hop) {
                delay(intervalMs)
                back = min(hop, back + cruiseStep)
                val p = DemoRoute.lerp(stop, exit, back / hop)
                emit(fix(p, DemoRoute.bearing(prev, p), speedMps))
                prev = p
            }
        }
    }

    private fun fix(p: LatLng, bearing: Float?, speed: Double) =
        GeoFix(p.lat, p.lng, FIX_ACCURACY_M, bearing, speed.toFloat(), clock() + skewMs)

    companion object {
        /** 시연 속도: 시속 80km */
        const val DEMO_SPEED_MPS = 80_000.0 / 3600

        /** 시연: 경유지를 추가하면 이 시간 뒤 도착 */
        const val DEMO_ARRIVE_MS = 6_000L

        /** 서버 시험용(debug): 시속 100km, 쉼터에서 20초 정차 */
        const val TEST_SPEED_MPS = 100_000.0 / 3600
        const val TEST_REST_MS = 20_000L

        private const val FIX_ACCURACY_M = 5f
        private const val MAX_FIX_SPACING_M = 150.0
    }
}
