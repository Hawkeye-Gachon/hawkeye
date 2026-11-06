package com.personal.driverguard.domain.engine

import com.personal.driverguard.domain.closed
import com.personal.driverguard.domain.engine
import com.personal.driverguard.domain.feed
import com.personal.driverguard.domain.obs
import com.personal.driverguard.domain.opened
import com.personal.driverguard.entity.AlertKind
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.QualityFlag
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** 기준 EAR 0.30 → 닫힘 0.21 미만, 열림 0.255 초과(기본 설정) */
class EyeClosureTest {

    private val closedEar = 0.08

    @Test
    fun `짧은 깜빡임은 지속 눈 감김 이벤트를 만들지 않는다`() {
        val e = engine()
        val out = e.feed(0, 10_000) { t -> if (t % 1_000 < 150) obs(t, closedEar, closedEar) else obs(t) }
        assertTrue(out.opened(EventType.EYE_CLOSURE).isEmpty())
        assertTrue(out.none { AlertKind.EYE_CAUTION in it.alertRequests })
        assertEquals(DisplayState.OBSERVING, out.last().display)
    }

    @Test
    fun `연속 눈 감김은 임계 시간 뒤 하나의 이벤트로 활성화된다`() {
        val e = engine()
        e.feed(0, 1_000) { obs(it) }
        val out = e.feed(1_000, 4_000) { obs(it, closedEar, closedEar) }
        val opened = out.opened(EventType.EYE_CLOSURE)
        assertEquals(1, opened.size)
        // 중앙값 필터(3프레임) 지연 + 한 프레임 이내
        val firstClosed = opened.single().candidateStartMs
        assertTrue(firstClosed in 1_000..1_100)
        assertTrue(opened.single().confirmedAtMs - firstClosed in 1_500..1_533)
        assertEquals(1, out.count { AlertKind.EYE_CAUTION in it.alertRequests })
        assertEquals(DisplayState.CAUTION, out.last().display)
    }

    @Test
    fun `히스테리시스 중간 구간은 감김을 유지하고 열림 임계값을 넘으면 정상 복귀로 끝난다`() {
        val e = engine()
        e.feed(0, 1_000) { obs(it) }
        val closed = e.feed(1_000, 3_000) { obs(it, closedEar, closedEar) }
        // 0.23 은 닫힘(0.21)과 열림(0.255) 사이
        val middle = e.feed(3_000, 4_000) { obs(it, 0.23, 0.23) }
        assertTrue(middle.closed().isEmpty())
        val reopened = e.feed(4_000, 4_500) { obs(it) }
        val end = reopened.closed().single()
        assertEquals(EventEndReason.RECOVERED, end.reason)
        val start = closed.opened(EventType.EYE_CLOSURE).single().candidateStartMs
        // 필터 지연 때문에 열림 판단은 한두 프레임 늦다
        assertTrue(end.durationMs in (4_000 - start)..(4_000 - start + 100))
    }

    @Test
    fun `한쪽 눈 추적 실패는 양쪽 눈 감김이 되지 않는다`() {
        val e = engine()
        e.feed(0, 1_000) { obs(it) }
        val out = e.feed(1_000, 5_000) { obs(it, left = closedEar, right = null) }
        assertTrue(out.opened(EventType.EYE_CLOSURE).isEmpty())
        // 분석 불가로 처리된다
        assertEquals(1, out.opened(EventType.ANALYSIS_UNAVAILABLE).size)
        assertEquals(QualityFlag.LANDMARKS_INVALID, out.last().unavailableReason)
        assertEquals(DisplayState.UNAVAILABLE, out.last().display)
    }

    @Test
    fun `윙크(한쪽만 감음)는 눈 감김이 아니다`() {
        val e = engine()
        val out = e.feed(0, 5_000) { obs(it, left = closedEar, right = 0.30) }
        assertTrue(out.opened(EventType.EYE_CLOSURE).isEmpty())
    }

    @Test
    fun `프레임 속도가 달라도 같은 시간 기준으로 작동한다`() {
        for (period in listOf(100L, 66L, 33L)) {
            val e = engine()
            e.feed(0, 1_000, period) { obs(it) }
            val out = e.feed(1_000, 2_400, period) { obs(it, closedEar, closedEar) }
            assertTrue("period=$period 1.4초 감김은 이벤트가 아님", out.opened(EventType.EYE_CLOSURE).isEmpty())
            val more = e.feed(2_400, 3_500, period) { obs(it, closedEar, closedEar) }
            val opened = more.opened(EventType.EYE_CLOSURE).single()
            assertTrue("period=$period", opened.confirmedAtMs - opened.candidateStartMs in 1_500 until 1_500 + period)
        }
    }

    @Test
    fun `긴 공백은 눈 감김 지속 시간에 들어가지 않는다`() {
        val e = engine()
        e.feed(0, 1_000) { obs(it) }
        val a = e.feed(1_000, 2_000) { obs(it, closedEar, closedEar) }
        // 1초 동안 관측 없음(틱만 옴)
        val gap = e.feed(2_000, 3_000) { null }
        val b = e.feed(3_000, 4_200) { obs(it, closedEar, closedEar) }
        assertTrue((a + gap + b).opened(EventType.EYE_CLOSURE).isEmpty())
        // 공백 뒤 새 관측으로 다시 1.5초를 채워야 이벤트가 된다
        val c = e.feed(4_200, 5_000) { obs(it, closedEar, closedEar) }
        val opened = c.opened(EventType.EYE_CLOSURE).single()
        assertTrue(opened.candidateStartMs >= 3_000)
    }

    @Test
    fun `확인된 눈 감김이 공백으로 끊기면 정상 복귀가 아닌 공백 사유로 닫힌다`() {
        val e = engine()
        e.feed(0, 500) { obs(it) }
        e.feed(500, 2_500) { obs(it, closedEar, closedEar) }
        val gap = e.feed(2_500, 3_500, 100) { null }
        val end = gap.closed().first()
        assertEquals(EventEndReason.OBSERVATION_GAP, end.reason)
    }

    @Test
    fun `분석 불가는 눈 감김으로 바뀌지 않는다`() {
        val e = engine()
        e.feed(0, 1_000) { obs(it) }
        val out = e.feed(1_000, 6_000) { com.personal.driverguard.entity.FaceObservation.notFound(it) }
        assertTrue(out.opened(EventType.EYE_CLOSURE).isEmpty())
        assertEquals(DisplayState.UNAVAILABLE, out.last().display)
        assertEquals(QualityFlag.FACE_NOT_FOUND, out.last().unavailableReason)
        // 3초 지연 뒤 한 번만 기록
        val unavailable = out.opened(EventType.ANALYSIS_UNAVAILABLE).single()
        assertTrue(unavailable.confirmedAtMs - unavailable.candidateStartMs in 3_000..3_033)
    }
}
