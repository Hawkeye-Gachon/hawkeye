package com.personal.driverguard.domain.session

import com.personal.driverguard.domain.FakeTime
import com.personal.driverguard.domain.FakeTripRepository
import com.personal.driverguard.domain.demo.DemoScenario
import com.personal.driverguard.domain.demo.DemoScript
import com.personal.driverguard.domain.engine.AlertPolicy
import com.personal.driverguard.domain.obs
import com.personal.driverguard.entity.AlertKind
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.LocationFix
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TripRecorderTest {

    private val time = FakeTime()
    private val repo = FakeTripRepository()

    private fun recorder(mode: AnalysisMode = AnalysisMode.REAL, config: AnalysisConfig = AnalysisConfig()) =
        TripRecorder(repo, time, config, mode, DemoScript.BASELINE, if (mode == AnalysisMode.REAL) "face_landmarker.task#64184e22" else null)

    /** [durationMs] 동안 [periodMs] 간격으로 관측을 넣고, 재생하기로 한 경고를 모은다 */
    private suspend fun TripRecorder.run(token: Long, durationMs: Long, periodMs: Long = 50, make: (Long) -> FaceObservation): List<AlertKind> {
        val alerts = mutableListOf<AlertKind>()
        val end = time.now + durationMs
        while (time.now < end) {
            onObservation(token, make(time.now))?.alert?.let { alerts += it }
            time.now += periodMs
        }
        return alerts
    }

    @Test
    fun `상태가 진동해도 경고는 반복 제한을 지키고 이벤트는 모두 기록된다`() = runTest {
        val r = recorder()
        val token = r.start()
        val t0 = time.now
        // 2.5초 이탈 / 1초 정면을 60초 동안 반복
        val alerts = r.run(token, 60_000) { t -> obs(t, yaw = if ((t - t0) % 3_500 < 2_500) 40.0 else 0.0) }
        val headAlerts = alerts.count { it == AlertKind.HEAD_CAUTION }
        val headEvents = repo.eventsOf(EventType.HEAD_AWAY).size
        assertTrue("경고 $headAlerts 회", headAlerts in 1..3)
        assertTrue("이벤트 $headEvents 건", headEvents >= 15)
    }

    @Test
    fun `높은 단계 경고는 낮은 단계의 반복 제한에 막히지 않는다`() {
        val policy = AlertPolicy(AnalysisConfig())
        assertEquals(AlertKind.EYE_CAUTION, policy.select(setOf(AlertKind.EYE_CAUTION), 0))
        assertNull(policy.select(setOf(AlertKind.EYE_CAUTION), 5_000))
        assertEquals(AlertKind.REST, policy.select(setOf(AlertKind.EYE_CAUTION, AlertKind.REST), 6_000))
        assertNull(policy.select(setOf(AlertKind.REST), 30_000))
        assertEquals(AlertKind.REST, policy.select(setOf(AlertKind.REST), 66_000))
    }

    @Test
    fun `주행 종료 시 활성 이벤트를 종료 사유와 함께 닫는다`() = runTest {
        val r = recorder()
        val token = r.start()
        r.run(token, 500) { obs(it) }
        r.run(token, 3_000) { obs(it, 0.08, 0.08) }
        r.finish()
        val closure = repo.eventsOf(EventType.EYE_CLOSURE).single()
        assertEquals(EventEndReason.TRIP_ENDED, closure.endReason)
        assertNotNull(closure.endedAtMs)
        assertTrue(repo.events.values.all { it.endedAtMs != null })
        val trip = repo.trips.single()
        assertNotNull(trip.endedAtMs)
        assertTrue(trip.validObservationMs!! in 3_000..3_600)
    }

    @Test
    fun `이전 세션의 지연 콜백은 무시한다`() = runTest {
        val r = recorder()
        val first = r.start()
        r.run(first, 1_000) { obs(it) }
        r.startRest()
        time.now += 10_000
        val second = r.endRest()!!
        assertTrue(second != first)
        // 휴식 전 토큰으로 늦게 도착한 눈 감김 결과
        val late = r.run(first, 3_000) { obs(it, 0.08, 0.08) }
        assertTrue(late.isEmpty())
        assertTrue(repo.eventsOf(EventType.EYE_CLOSURE).isEmpty())
        assertNull(r.onObservation(first, obs(time.now)))
        assertNotNull(r.onObservation(second, obs(time.now + 50)))
    }

    @Test
    fun `오래되거나 부정확한 위치는 이벤트 위치로 쓰지 않는다`() = runTest {
        val r = recorder()
        val token = r.start()
        // 60초 전 위치
        r.onLocation(LocationFix(37.5, 127.0, 5f, time.now))
        time.now += 60_000
        r.run(token, 500) { obs(it) }
        r.run(token, 2_000) { obs(it, 0.08, 0.08) }
        val stale = repo.eventsOf(EventType.EYE_CLOSURE).single()
        assertNull(stale.latitude)

        r.run(token, 1_000) { obs(it) }
        r.onLocation(LocationFix(37.6, 127.1, 500f, time.now))
        r.run(token, 2_000) { obs(it, 0.08, 0.08) }
        assertNull(repo.eventsOf(EventType.EYE_CLOSURE).maxBy { it.confirmedAtMs }.latitude)

        r.run(token, 1_000) { obs(it) }
        r.onLocation(LocationFix(37.7, 127.2, 8f, time.now))
        r.run(token, 2_000) { obs(it, 0.08, 0.08) }
        val fresh = repo.eventsOf(EventType.EYE_CLOSURE).maxBy { it.confirmedAtMs }
        assertEquals(37.7, fresh.latitude!!, 1e-9)
        assertNotNull(fresh.locationAtMs)
    }

    @Test
    fun `시연 데이터는 실제 데이터와 구분해 저장한다`() = runTest {
        val r = recorder(AnalysisMode.DEMO)
        val token = r.start()
        r.run(token, 4_000) { t -> DemoScript.observationAt(DemoScenario.SUSTAINED_CLOSURE, t - 1_000_000L, t)!! }
        r.finish()
        val trip = repo.trips.single()
        assertEquals(AnalysisMode.DEMO, trip.mode)
        assertNull(trip.modelInfo)
        assertTrue(repo.events.isNotEmpty())
        assertTrue(repo.events.values.all { it.source == AnalysisMode.DEMO })
    }

    @Test
    fun `앱이 화면을 벗어나면 중단 구간을 기록하고 그동안 관측은 버린다`() = runTest {
        val r = recorder()
        val token = r.start()
        r.run(token, 1_000) { obs(it) }
        r.interrupt(com.personal.driverguard.entity.InterruptionReason.APP_BACKGROUND)
        assertNull(r.onObservation(token, obs(time.now)))
        time.now += 5_000
        val next = r.resumeAfterInterruption()!!
        r.run(next, 1_000) { obs(it) }
        r.finish()
        val interruption = repo.interruptions.values.single()
        assertEquals(5_000L, interruption.durationMs)
        // 중단 시간은 관측 시간·분석 불가 시간 어디에도 들어가지 않는다
        val trip = repo.trips.single()
        assertTrue(trip.unavailableMs!! < 500)
    }

    @Test
    fun `경고 반복 제한 중에도 이벤트는 기록된다`() = runTest {
        val r = recorder()
        val token = r.start()
        val alerts = mutableListOf<AlertKind>()
        repeat(3) {
            alerts += r.run(token, 1_000) { obs(it) }
            alerts += r.run(token, 2_000) { obs(it, 0.08, 0.08) }
        }
        assertEquals(3, repo.eventsOf(EventType.EYE_CLOSURE).size)
        // 9초 동안 세 번 → 주의 경고는 한 번(반복 제한 20초)만, 세 번째에 졸음 의심 휴식 권고
        assertEquals(1, alerts.count { it == AlertKind.EYE_CAUTION })
        assertTrue(AlertKind.REST in alerts)
    }
}
