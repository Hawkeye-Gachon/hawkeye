package com.personal.driverguard.ui.state

import com.personal.driverguard.domain.engine.DistanceAccumulator
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DriveSimulatorTest {
    private val route = DemoTestData.route

    @Test
    fun routeIsGyeongbuSeoulToBusan() {
        assertTrue("양재IC→구서IC 약 416km: ${route.lengthMeters}", route.lengthMeters in 405_000.0..425_000.0)
        assertTrue("서울에서 출발", route.points.first().lat > 37.4)
        assertTrue("부산에 도착", route.points.last().lat < 35.3)
        assertEquals("첫 쉼터", "기흥휴게소", route.nextRest(0.0)?.name)
    }

    /** 시속 80km로 1초마다 경로를 따라간다 */
    @Test
    fun drivesAt80KmhOncePerSecond() = runTest {
        var clock = 0L
        val sim = DriveSimulator(route, clock = { clock.also { clock += 1_000 } })
        val fixes = sim.fixes().take(30).toList()
        fixes.zipWithNext().forEach { (a, b) ->
            val mps = DistanceAccumulator.haversineMeters(a.latitude, a.longitude, b.latitude, b.longitude) / ((b.monotonicMs - a.monotonicMs) / 1000.0)
            assertTrue("초속 $mps", mps in 20.0..24.0)
        }
        assertTrue(fixes.all { it.speedMps!! > 22f && it.bearingDeg != null })
    }

    /** 시연: 경유지를 넣으면 6초 안에 쉼터에 도착해 멈추고, 주행 기록이 방문으로 남긴다. 위치 시각은 시속 80km로 달린 만큼 앞당겨진다. */
    @Test
    fun demoArrivesAtRestStopWithinSixSeconds() = runTest {
        val rest = route.nextRest(0.0)!!
        val rec = DriveRecorder(startedMonotonicMs = 0)
        val sim = DriveSimulator(route, arriveWithinMs = DriveSimulator.DEMO_ARRIVE_MS, clock = { testScheduler.currentTime })
        rec.recordWaypoint(rest.placeId, "WAYPOINT", "DROWSINESS", latitude = rest.lat, longitude = rest.lng)
        val arrivedFix = sim.fixes { rec.pendingStop()?.let { LatLng(it.first, it.second) } }.first { f ->
            rec.onLocation(f)
            rec.toEndRequest().waypoints.single().visitedAt != null
        }
        assertEquals(0f, arrivedFix.speedMps)
        assertTrue("6초 안에 도착: ${testScheduler.currentTime}ms", testScheduler.currentTime <= 6_000)
        assertEquals(rest.lat, arrivedFix.latitude, 1e-9)
        val km = rec.toEndRequest().distanceMeters / 1000.0
        assertTrue("빨리 감은 거리도 기록: ${km}km", km in 24.0..30.0)
        assertTrue("경로가 곧은 선으로 끊기지 않음", rec.toEndRequest().route.size > 100)
        assertTrue("위치 시각은 시속 80km 기준(약 20분): ${arrivedFix.monotonicMs}", arrivedFix.monotonicMs in 1_000_000L..1_400_000L)
        assertNotNull(rec.takeArrivals().singleOrNull())
    }

    /** 쉼터에서는 resume 을 부를 때까지 멈춰 있다가 경로로 돌아와 이어 달린다 */
    @Test
    fun staysAtRestStopUntilResumed() = runTest {
        val rest = route.nextRest(0.0)!!
        val sim = DriveSimulator(route, arriveWithinMs = 2_000, clock = { testScheduler.currentTime })
        var stop: LatLng? = LatLng(rest.lat, rest.lng)
        var stoppedFixes = 0
        val last = sim.fixes { stop }.first { f ->
            if (f.speedMps == 0f && f.latitude == rest.lat) {
                stoppedFixes++
                if (stoppedFixes == 5) { stop = null; sim.resume() }
            }
            stoppedFixes >= 5 && f.speedMps!! > 20f && f.latitude < rest.lat - 0.001
        }
        assertTrue("다시 경로를 따라 남쪽으로 달림", last.speedMps!! > 20f)
        assertEquals(5, stoppedFixes)
    }
}
