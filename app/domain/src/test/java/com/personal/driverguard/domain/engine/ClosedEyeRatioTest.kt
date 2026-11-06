package com.personal.driverguard.domain.engine

import com.personal.driverguard.domain.analysis.temporal.ClosedEyeRatioWindow
import com.personal.driverguard.domain.analysis.temporal.RatioStatus
import com.personal.driverguard.domain.engine
import com.personal.driverguard.domain.feed
import com.personal.driverguard.domain.obs
import com.personal.driverguard.domain.opened
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.FaceObservation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ClosedEyeRatioTest {

    @Test
    fun `창 경계에 걸친 구간은 경계에서 잘라 계산한다`() {
        val w = ClosedEyeRatioWindow(windowMs = 60_000, minCoverage = 0.8)
        w.reset(0)
        w.add(0, 30_000, closed = true)
        w.add(30_000, 70_000, closed = false)
        val r = w.evaluate(70_000)
        // 창 [10s, 70s]: 감김 20초 + 뜸 40초
        assertEquals(60_000, r.validMs)
        assertEquals(20_000, r.closedMs)
        assertEquals(1.0 / 3, r.ratio!!, 1e-9)
        assertEquals(RatioStatus.AVAILABLE, r.status)
    }

    @Test
    fun `창 길이만큼 쌓이기 전에는 측정 중이다`() {
        val w = ClosedEyeRatioWindow(60_000, 0.8)
        w.reset(0)
        w.add(0, 30_000, closed = false)
        val r = w.evaluate(30_000)
        assertEquals(RatioStatus.MEASURING, r.status)
        assertEquals(30_000, r.windowUsedMs)
    }

    @Test
    fun `유효 관측이 없으면 비율이 없다`() {
        val w = ClosedEyeRatioWindow(60_000, 0.8)
        w.reset(0)
        assertNull(w.evaluate(70_000).ratio)
    }

    @Test
    fun `분석 불가 구간은 분자와 유효 분모에서 빠지고 관측률이 낮으면 판정을 보류한다`() {
        val e = engine()
        // 40초 정상 관측, 이어서 25초 얼굴 미검출
        e.feed(0, 40_000) { obs(it) }
        val out = e.feed(40_000, 65_000) { FaceObservation.notFound(it) }
        val ratio = out.last().snapshot!!.closedRatio
        assertTrue(ratio.validMs in 34_500..35_500) // 창 [약 5s, 65s] 중 유효한 것은 약 5~40초
        assertEquals(0.0, ratio.ratio!!, 1e-9)
        assertEquals(RatioStatus.LOW_COVERAGE, ratio.status)
    }

    /** 1초 감김 / 1초 뜸 반복(지속 이벤트 기준 1.5초 미만) → 눈 감김 비율 약 0.5 */
    private fun halfClosed(t: Long) = if ((t / 1_000) % 2 == 0L) obs(t, 0.08, 0.08) else obs(t)

    @Test
    fun `관측률이 충분하고 눈 감김 비율이 기준 이상이면 졸음 의심이다`() {
        val e = engine()
        val out = e.feed(0, 61_000, 50) { halfClosed(it) }
        assertTrue(out.opened(EventType.EYE_CLOSURE).isEmpty())
        val drowsy = out.opened(EventType.DROWSINESS_SUSPECTED).single()
        assertEquals(listOf(JudgmentRules.CLOSED_RATIO), drowsy.rules)
        assertEquals(DisplayState.REST_RECOMMENDED, out.last().display)
    }

    @Test
    fun `관측률이 낮으면 눈 감김 비율이 높아도 판정하지 않는다`() {
        val e = engine()
        // 1초 관측 / 1초 미검출 반복 → 관측률 약 0.5
        val out = e.feed(0, 90_000, 50) { t -> if ((t / 2_000) % 2 == 0L) halfClosed(t) else FaceObservation.notFound(t) }
        assertTrue(out.opened(EventType.DROWSINESS_SUSPECTED).isEmpty())
        assertEquals(RatioStatus.LOW_COVERAGE, out.last().snapshot!!.closedRatio.status)
    }

    @Test
    fun `측정 중에는 판정하지 않는다`() {
        val e = engine(AnalysisConfig())
        val out = e.feed(0, 50_000, 50) { halfClosed(it) }
        assertTrue(out.opened(EventType.DROWSINESS_SUSPECTED).isEmpty())
        assertEquals(RatioStatus.MEASURING, out.last().snapshot!!.closedRatio.status)
    }
}
