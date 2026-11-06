package com.personal.driverguard.domain.engine

import com.personal.driverguard.domain.closed
import com.personal.driverguard.domain.engine
import com.personal.driverguard.domain.feed
import com.personal.driverguard.domain.obs
import com.personal.driverguard.domain.opened
import com.personal.driverguard.entity.AlertKind
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.CautionCause
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.DrowsinessLevel
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.QualityFlag
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class JudgmentTest {

    @Test
    fun `짧은 고개 돌림과 지속적인 이탈을 구분한다`() {
        val e = engine()
        e.feed(0, 1_000) { obs(it) }
        // 미러 확인처럼 1초 돌림
        val brief = e.feed(1_000, 2_000) { obs(it, yaw = 40.0) } + e.feed(2_000, 4_000) { obs(it) }
        assertTrue(brief.opened(EventType.HEAD_AWAY).isEmpty())
        assertTrue(brief.opened(EventType.DISTRACTION_SUSPECTED).isEmpty())
        // 3초 유지
        val sustained = e.feed(4_000, 7_000) { obs(it, yaw = 40.0) }
        assertEquals(1, sustained.opened(EventType.HEAD_AWAY).size)
        assertEquals(1, sustained.opened(EventType.DISTRACTION_SUSPECTED).size)
        assertEquals(DisplayState.CAUTION, sustained.last().display)
        assertEquals(CautionCause.HEAD_AWAY, sustained.last().cautionCause)
        assertTrue(sustained.last().distraction)
    }

    @Test
    fun `고개 각도 진입과 복귀 기준이 분리돼 경계 근처 흔들림에 끊기지 않는다`() {
        val e = engine()
        e.feed(0, 500) { obs(it) }
        // 진입 25도 초과, 복귀 20도(25×0.8) 미만
        val out = e.feed(500, 4_000) { t -> obs(t, yaw = if ((t / 100) % 2 == 0L) 27.0 else 22.0) }
        assertEquals(1, out.opened(EventType.HEAD_AWAY).size)
        assertTrue(out.closed().isEmpty())
    }

    @Test
    fun `고개를 크게 돌리면 눈 지표만 무효가 되고 자세는 계속 판정한다`() {
        val e = engine()
        e.feed(0, 500) { obs(it) }
        val out = e.feed(500, 3_500) { obs(it, yaw = 45.0) }
        val snap = out.last().snapshot!!
        assertEquals(QualityFlag.EXCESSIVE_HEAD_ROTATION, snap.eyeQuality)
        assertEquals(QualityFlag.VALID, snap.poseQuality)
        assertEquals(1, out.opened(EventType.HEAD_AWAY).size)
        assertTrue(out.opened(EventType.EYE_CLOSURE).isEmpty())
    }

    @Test
    fun `반복적인 지속 눈 감김은 졸음 의심과 휴식 권고가 된다`() {
        val e = engine()
        val out = e.feed(0, 30_000) { t -> if ((t % 6_000) in 1_000 until 3_000) obs(t, 0.08, 0.08) else obs(t) }
        assertEquals(5, out.opened(EventType.EYE_CLOSURE).size)
        val drowsy = out.opened(EventType.DROWSINESS_SUSPECTED)
        assertEquals(1, drowsy.size)
        assertTrue(JudgmentRules.REPEATED_CLOSURE in drowsy.single().rules)
        assertEquals(DisplayState.REST_RECOMMENDED, out.last().display)
        assertTrue(out.any { AlertKind.REST in it.alertRequests })
    }

    @Test
    fun `입 벌림 한 번은 휴식 권고가 되지 않는다`() {
        val e = engine()
        val out = e.feed(0, 10_000) { t -> obs(t, mouth = if (t in 1_000 until 4_000) 0.6 else 0.05) }
        assertEquals(1, out.opened(EventType.MOUTH_OPEN).size)
        assertTrue(out.opened(EventType.DROWSINESS_SUSPECTED).isEmpty())
        assertEquals(DisplayState.OBSERVING, out.last().display)
    }

    @Test
    fun `입 벌림 반복과 눈 감김이 함께 있으면 졸음 의심이고 보조 징후로 표시된다`() {
        val e = engine()
        val closure = e.feed(0, 5_000) { t -> if (t in 1_000 until 3_000) obs(t, 0.08, 0.08) else obs(t) }
        assertEquals(1, closure.opened(EventType.EYE_CLOSURE).size)
        val yawns = e.feed(5_000, 25_000) { t ->
            obs(t, mouth = if (t in 6_000 until 9_000 || t in 16_000 until 19_000) 0.6 else 0.05)
        }
        assertEquals(2, yawns.opened(EventType.MOUTH_OPEN).size)
        val drowsy = yawns.opened(EventType.DROWSINESS_SUSPECTED).single()
        assertTrue(JudgmentRules.YAWN_WITH_CLOSURE in drowsy.rules)
        assertEquals(DrowsinessLevel.SUSPECTED_WITH_SUPPORT, yawns.last().drowsiness)
    }

    @Test
    fun `입 벌림 재발생 최소 간격 안의 벌림은 새 이벤트로 세지 않는다`() {
        val e = engine()
        val out = e.feed(0, 12_000) { t -> obs(t, mouth = if (t in 1_000 until 4_000 || t in 5_000 until 8_000) 0.6 else 0.05) }
        assertEquals(1, out.opened(EventType.MOUTH_OPEN).size)
    }

    @Test
    fun `졸음 의심은 유지 시간이 지나면 끝난다`() {
        val config = AnalysisConfig(drowsinessHoldMs = 5_000)
        val e = engine(config)
        val during = e.feed(0, 15_000) { t -> if ((t % 6_000) in 1_000 until 3_000) obs(t, 0.08, 0.08) else obs(t) }
        assertEquals(DisplayState.REST_RECOMMENDED, during.last().display)
        val after = e.feed(15_000, 25_000) { obs(it) }
        val key = during.opened(EventType.DROWSINESS_SUSPECTED).single().key
        val end = after.closed().single { it.key == key }
        assertEquals(EventEndReason.HOLD_EXPIRED, end.reason)
        assertFalse(after.last().display == DisplayState.REST_RECOMMENDED)
    }

    @Test
    fun `반복 끄덕임은 실험 이벤트로만 기록하고 기본 설정에서는 졸음 판정에 쓰지 않는다`() {
        val e = engine()
        // 15도 숙였다가 0.5초 뒤 복귀를 2초마다
        val out = e.feed(0, 10_000) { t -> obs(t, pitch = if ((t % 2_000) in 500 until 1_000) 15.0 else 0.0) }
        assertEquals(1, out.opened(EventType.HEAD_NOD).size)
        assertTrue(out.opened(EventType.DROWSINESS_SUSPECTED).isEmpty())
    }

    @Test
    fun `휴식 시작은 열린 이벤트를 휴식 사유로 닫고 재개 전 관측은 버린다`() {
        val e = engine()
        e.feed(0, 500) { obs(it) }
        e.feed(500, 3_000) { obs(it, 0.08, 0.08) }
        val paused = e.pause(3_000, EventEndReason.REST_STARTED)
        assertEquals(EventEndReason.REST_STARTED, paused.changes.filterIsInstance<EventChange.Closed>().single().reason)
        assertEquals(DisplayState.PAUSED, paused.display)
        e.resume(10_000)
        // 재개 이전에 촬영된(지연 도착한) 관측
        val late = e.onObservation(obs(9_000, 0.08, 0.08))
        assertTrue(late.changes.isEmpty())
        val after = e.feed(10_000, 11_000) { obs(it) }
        assertEquals(DisplayState.OBSERVING, after.last().display)
    }

    @Test
    fun `주행 종료 시 활성 이벤트를 모두 닫는다`() {
        val e = engine()
        e.feed(0, 500) { obs(it) }
        e.feed(500, 4_000) { obs(it, 0.08, 0.08, yaw = 0.0) }
        val out = e.finish(4_000)
        val closed = out.changes.filterIsInstance<EventChange.Closed>()
        assertTrue(closed.isNotEmpty())
        assertTrue(closed.all { it.reason == EventEndReason.TRIP_ENDED })
    }

    @Test
    fun `캘리브레이션이 없으면 상대 지표를 판정하지 않는다`() {
        val e = com.personal.driverguard.domain.engine.DriverStateEngine(AnalysisConfig(), baseline = null, startMs = 0)
        val out = e.feed(0, 5_000) { obs(it, 0.08, 0.08, yaw = 40.0) }
        assertTrue(out.opened(EventType.EYE_CLOSURE).isEmpty())
        assertTrue(out.opened(EventType.HEAD_AWAY).isEmpty())
        assertEquals(QualityFlag.CALIBRATION_REQUIRED, out.last().unavailableReason)
    }
}
