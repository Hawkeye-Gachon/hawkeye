package com.personal.driverguard.ui.state

import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.NavMode
import com.personal.driverguard.data.remote.TriggerType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DriveRecorderTest {

    private fun fix(lat: Double, lng: Double, t: Long, acc: Float = 5f) = GeoFix(lat, lng, acc, 90f, 20f, t)

    @Test
    fun countsEventsAndAttachesLatestLocation() {
        val rec = DriveRecorder(startedMonotonicMs = 0)
        rec.recordEvent(EventType.INATTENTION, "2026-11-07T10:00:00+09:00")
        rec.onLocation(fix(37.0, 127.0, 1_000))
        rec.recordEvent(EventType.DROWSINESS, "2026-11-07T10:01:00+09:00")
        assertEquals(1, rec.inattentionCount)
        assertEquals(1, rec.drowsinessCount)
        val body = rec.toEndRequest("2026-11-07T10:30:00+09:00")
        assertNull("위치를 받기 전 감지는 위치를 비운다", body.events[0].latitude)
        assertEquals(37.0, body.events[1].latitude!!, 0.0)
    }

    @Test
    fun routeKeepsPointsAtLeast25mApartAndSkipsInaccurateFixes() {
        val rec = DriveRecorder(startedMonotonicMs = 0)
        rec.onLocation(fix(37.0, 127.0, 1_000))
        rec.onLocation(fix(37.00005, 127.0, 2_000)) // 약 5.6m → 버림
        rec.onLocation(fix(37.0005, 127.0, 3_000)) // 약 56m → 남김
        rec.onLocation(fix(37.002, 127.0, 4_000, acc = 120f)) // 오차 큼 → 경로에서 뺌
        val route = rec.toEndRequest().route
        assertEquals(listOf(listOf(37.0, 127.0), listOf(37.0005, 127.0)), route)
    }

    @Test
    fun distanceIsZeroWithoutLocation() {
        val body = DriveRecorder(startedMonotonicMs = 0).toEndRequest()
        assertEquals(0, body.distanceMeters)
        assertTrue(body.route.isEmpty() && body.events.isEmpty() && body.waypoints.isEmpty())
    }

    @Test
    fun distanceAccumulatesDrivenMeters() {
        val rec = DriveRecorder(startedMonotonicMs = 0)
        // 1초마다 약 22m(시속 80km)씩 북쪽으로
        for (i in 0..10) rec.onLocation(fix(37.0 + i * 0.0002, 127.0, i * 1_000L))
        val m = rec.toEndRequest().distanceMeters
        assertTrue("약 222m 여야 함: $m", m in 200..240)
    }

    @Test
    fun waypointIsRecordedWithModeAndTrigger() {
        val rec = DriveRecorder(startedMonotonicMs = 0)
        rec.recordWaypoint(41, NavMode.WAYPOINT, TriggerType.DROWSINESS, "2026-11-07T10:05:00+09:00")
        val wp = rec.toEndRequest().waypoints.single()
        assertEquals(41L, wp.placeId)
        assertEquals("WAYPOINT", wp.mode)
        assertNull(wp.visitedAt)
    }

    @Test
    fun waypointIsVisitedWhenStoppedNearby() {
        val rec = DriveRecorder(startedMonotonicMs = 0)
        rec.recordWaypoint(41, NavMode.WAYPOINT, TriggerType.DROWSINESS, "2026-11-07T10:05:00+09:00", 37.3290, 127.1050)
        // 가까이 지나가기만 하면(시속 72km) 방문이 아니다
        rec.onLocation(GeoFix(37.3292, 127.1050, 5f, 180f, 20f, 1_000), "2026-11-07T10:10:00+09:00")
        assertNull(rec.toEndRequest().waypoints.single().visitedAt)
        // 멀리서 멈춘 것도 방문이 아니다(약 1.1km)
        rec.onLocation(GeoFix(37.3390, 127.1050, 5f, null, 0f, 2_000), "2026-11-07T10:11:00+09:00")
        assertNull(rec.toEndRequest().waypoints.single().visitedAt)
        // 쉼터 300m 안에서 멈추면 방문, 시각은 처음 멈춘 때
        rec.onLocation(GeoFix(37.3295, 127.1050, 5f, null, 0.3f, 3_000), "2026-11-07T10:20:00+09:00")
        rec.onLocation(GeoFix(37.3295, 127.1050, 5f, null, 0f, 4_000), "2026-11-07T10:21:00+09:00")
        assertEquals("2026-11-07T10:20:00+09:00", rec.toEndRequest().waypoints.single().visitedAt)
        assertNull(rec.pendingStop())
    }

    @Test
    fun stoppedIsDerivedFromMovementWhenSpeedIsMissing() {
        val rec = DriveRecorder(startedMonotonicMs = 0)
        rec.recordWaypoint(41, NavMode.WAYPOINT, TriggerType.DROWSINESS, "2026-11-07T10:05:00+09:00", 37.3290, 127.1050)
        rec.onLocation(GeoFix(37.3291, 127.1050, 5f, null, null, 0), "2026-11-07T10:20:00+09:00")
        rec.onLocation(GeoFix(37.32911, 127.1050, 5f, null, null, 5_000), "2026-11-07T10:20:05+09:00")
        assertEquals("2026-11-07T10:20:05+09:00", rec.toEndRequest().waypoints.single().visitedAt)
    }

    @Test
    fun thinKeepsEndpointsAndLimit() {
        val pts = List(5_000) { listOf(it.toDouble(), 0.0) }
        val out = DriveRecorder.thin(pts, 2_000)
        assertEquals(2_000, out.size)
        assertEquals(0.0, out.first()[0], 0.0)
        assertEquals(4_999.0, out.last()[0], 0.0)
    }

    @Test
    fun timestampsAreSeoulOffset() {
        assertTrue(nowIso().endsWith("+09:00"))
    }
}
