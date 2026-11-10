package com.personal.driverguard.ui.state

import com.personal.driverguard.domain.engine.DistanceAccumulator
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DummyRouteTest {
    @Test
    fun routeIsAbout23Km() {
        val route = DummyRoute.points()
        assertEquals(49, route.size)
        assertTrue("양재→기흥 약 23km: ${DummyRoute.lengthMeters(route)}", DummyRoute.lengthMeters(route) in 20_000..27_000)
    }

    /** 1초마다 시속 100km 만큼 움직여야 서버의 속도 검사(거리 ÷ 시간)를 통과한다 */
    @Test
    fun drivesAtHighwaySpeedOncePerSecond() = runTest {
        var clock = 0L
        val fixes = DummyRoute.drive(now = { clock.also { clock += DummyRoute.DRIVE_INTERVAL_MS } }).take(60).toList()
        val rec = DriveRecorder(startedMonotonicMs = 0)
        fixes.forEach(rec::onLocation)
        fixes.zipWithNext().forEach { (a, b) ->
            val mps = DistanceAccumulator.haversineMeters(a.latitude, a.longitude, b.latitude, b.longitude) / ((b.monotonicMs - a.monotonicMs) / 1000.0)
            assertTrue("초속 $mps", mps in 25.0..30.0)
        }
        val meters = rec.toEndRequest().distanceMeters
        assertTrue("59초 × 27.8m ≈ 1,640m: $meters", meters in 1_500..1_750)
        assertTrue(rec.toEndRequest().route.size > 30)
    }

    /** 쉼터를 추가하면 그쪽으로 가서 멈추고, 주행 기록이 방문으로 남긴다 */
    @Test
    fun detoursToRestStopAndIsRecordedAsVisited() = runTest {
        var clock = 0L
        val rec = DriveRecorder(startedMonotonicMs = 0)
        val place = DummyRoute.restPlace()
        rec.recordWaypoint(place.placeId, "WAYPOINT", "DROWSINESS", latitude = place.latitude, longitude = place.longitude)
        val fixes = DummyRoute.drive(now = { clock.also { clock += DummyRoute.DRIVE_INTERVAL_MS } }, restStop = { rec.pendingStop() })
            .take(1_000).toList()
        fixes.forEach { rec.onLocation(it) }
        assertTrue("방문 기록", rec.toEndRequest().waypoints.single().visitedAt != null)
        fixes.zipWithNext().forEach { (a, b) ->
            val mps = DistanceAccumulator.haversineMeters(a.latitude, a.longitude, b.latitude, b.longitude) / ((b.monotonicMs - a.monotonicMs) / 1000.0)
            assertTrue("초속 $mps", mps <= 30.0)
        }
        assertTrue("멈춰 있던 시간", fixes.count { it.latitude == place.latitude && it.longitude == place.longitude } >= 20)
    }

    @Test
    fun stopsAtRouteEnd() {
        val route = DummyRoute.points()
        val legs = route.zipWithNext().map { (a, b) -> DistanceAccumulator.haversineMeters(a[0], a[1], b[0], b[1]) }
        val end = DummyRoute.positionAt(route, legs, 1_000_000.0, 0)
        assertEquals(route.last()[0], end.latitude, 1e-9)
        assertEquals(0f, end.speedMps)
    }
}
