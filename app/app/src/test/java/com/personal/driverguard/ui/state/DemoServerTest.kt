package com.personal.driverguard.ui.state

import com.personal.driverguard.data.remote.EndDriveRequest
import com.personal.driverguard.data.remote.EventRequest
import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.NearbyRequest
import com.personal.driverguard.data.remote.WaypointRequest
import com.personal.driverguard.platform.VoiceGuide
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DemoServerTest {
    private val route = DemoTestData.route
    private val server = DemoServer { route }

    @Test
    fun nearbyReturnsNextRestAheadOnGyeongbu() {
        val here = route.positionAt(5_000.0)
        val r = server.nearby(NearbyRequest(here.lat, here.lng, 180.0, 22.0))
        assertEquals(true, r.onHighway)
        assertEquals("기흥휴게소", r.place?.name)
        assertTrue("도로 거리 약 22km: ${r.place?.roadDistanceMeters}", r.place!!.roadDistanceMeters in 20_000..24_000)
        // 기흥을 지나면 다음 쉼터
        val after = route.positionAt(30_000.0)
        assertEquals("오산 졸음쉼터", server.nearby(NearbyRequest(after.lat, after.lng)).place?.name)
    }

    @Test
    fun nearbyOffHighwayHasNoPlace() {
        val r = server.nearby(NearbyRequest(37.5665, 126.9780)) // 서울시청
        assertEquals(false, r.onHighway)
        assertNull(r.place)
    }

    @Test
    fun endReportCarriesRestNameAndVisit() {
        val rest = route.nextRest(0.0)!!
        val body = EndDriveRequest(
            endedAt = "2026-11-14T10:30:00+09:00", distanceMeters = 27_000,
            events = listOf(EventRequest(EventType.DROWSINESS, "2026-11-14T10:05:00+09:00", 37.4, 127.05)),
            route = listOf(listOf(37.46, 127.04), listOf(37.24, 127.10)),
            waypoints = listOf(WaypointRequest(rest.placeId, "WAYPOINT", "DROWSINESS", "2026-11-14T10:05:40+09:00", "2026-11-14T10:20:00+09:00")),
        )
        val r = server.endReport(9_000_001, body)
        assertEquals(1, r.drowsinessCount)
        assertEquals("기흥휴게소", r.waypoints.single().placeName)
        assertTrue(r.waypoints.single().visited)
        assertTrue("27km 는 시속 80km로 약 20분: ${r.observedSeconds}", r.observedSeconds!! in 1_150..1_300)
    }

    @Test
    fun yesNoUnderstandsShortKoreanAnswers() {
        listOf("응", "네", "그래 추가해줘", "좋아요").forEach { assertEquals(it, true, VoiceGuide.yesNo(it)) }
        listOf("아니", "괜찮아요", "나중에").forEach { assertEquals(it, false, VoiceGuide.yesNo(it)) }
        assertNull(VoiceGuide.yesNo("음"))
        assertFalse(VoiceGuide.yesNo("아니 됐어")!!)
    }

    @Test
    fun josaFollowsFinalConsonant() {
        assertEquals("를", HawkeyeViewModel.josaReul("기흥휴게소"))
        assertEquals("을", HawkeyeViewModel.josaReul("망향휴게소 졸음쉼터 입장"))
        assertEquals("로", HawkeyeViewModel.josaRo("기흥휴게소"))
        assertEquals("로", HawkeyeViewModel.josaRo("서울"))
        assertEquals("으로", HawkeyeViewModel.josaRo("휴게소 입구점"))
    }
}
