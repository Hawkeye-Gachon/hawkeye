package com.personal.driverguard.domain.session

import com.personal.driverguard.domain.engine.AlertPolicy
import com.personal.driverguard.domain.engine.DistanceAccumulator
import com.personal.driverguard.domain.engine.DriverStateEngine
import com.personal.driverguard.domain.engine.EngineOutput
import com.personal.driverguard.domain.engine.EngineSnapshot
import com.personal.driverguard.domain.engine.EventChange
import com.personal.driverguard.domain.platform.TimeSource
import com.personal.driverguard.domain.repository.ITripRepository
import com.personal.driverguard.entity.AlertKind
import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.entity.CalibrationBaseline
import com.personal.driverguard.entity.CautionCause
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.DriverEventVo
import com.personal.driverguard.entity.DrowsinessLevel
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.InterruptionReason
import com.personal.driverguard.entity.LocationFix
import com.personal.driverguard.entity.QualityFlag
import kotlin.math.abs

/**
 * 분석 세션 식별자. 카메라 분석은 비동기라 결과가 늦게 도착할 수 있다.
 * 프레임을 제출할 때의 토큰을 결과에 붙여 두고, 분석을 멈추거나(휴식·백그라운드) 여정이 바뀌면 토큰을 바꿔
 * 이전 세션의 지연 콜백이 새 구간에 섞이지 않게 한다.
 */
class AnalysisSessionGate {
    @Volatile
    var current: Long = 0L
        private set

    @Volatile
    var open: Boolean = false
        private set

    fun openNew(): Long {
        current += 1
        open = true
        return current
    }

    fun close() {
        open = false
    }

    fun accepts(token: Long): Boolean = open && token == current
}

/** 이벤트에 붙일 위치. 오래됐거나 정확도가 낮은 위치는 쓰지 않는다. */
class LocationGate(private val maxAgeMs: Long, private val maxAccuracyM: Double) {
    private var last: LocationFix? = null

    fun offer(fix: LocationFix) {
        last = fix
    }

    fun forEvent(eventMonotonicMs: Long): LocationFix? {
        val fix = last ?: return null
        val accuracy = fix.accuracyMeters ?: return null
        if (accuracy > maxAccuracyM) return null
        if (abs(eventMonotonicMs - fix.monotonicMs) > maxAgeMs) return null
        return fix
    }
}

data class SessionUpdate(
    val display: DisplayState,
    val cautionCause: CautionCause?,
    val drowsiness: DrowsinessLevel,
    val distraction: Boolean,
    val unavailableReason: QualityFlag?,
    /** 이번 입력으로 새로 재생할 경고. 대부분 null 이다. */
    val alert: AlertKind?,
    val snapshot: EngineSnapshot?,
    val isResting: Boolean,
)

/**
 * 주행 시작부터 종료까지 하나의 여정을 기록한다.
 * 상태 엔진 → 이벤트 저장, 경고 정책, 위치, 분석 중단·휴식 기록을 묶는다.
 *
 * 경고 반복 제한과 이벤트 기록은 분리돼 있다. 경고가 반복 제한으로 재생되지 않아도 이벤트는 저장된다.
 * 스레드 안전하지 않다. 모든 메서드를 한 코루틴에서 순서대로 호출한다.
 */
class TripRecorder(
    private val repository: ITripRepository,
    private val time: TimeSource,
    private val config: AnalysisConfig,
    private val mode: AnalysisMode,
    private val baseline: CalibrationBaseline?,
    private val modelInfo: String?,
) {
    private class StoredEvent(
        val dbId: Long,
        val confirmedAtMs: Long,
        var level: AlertLevel,
        var rules: List<String>,
        var metrics: String,
    )

    private lateinit var engine: DriverStateEngine
    private val alertPolicy = AlertPolicy(config)
    private val distance = DistanceAccumulator()
    private val locationGate = LocationGate(config.maxLocationAgeMs, config.maxLocationAccuracyM)
    val sessionGate = AnalysisSessionGate()

    private val events = mutableMapOf<Long, StoredEvent>()
    private var startMs = 0L
    private var restId: Long? = null
    private var restStartMs = 0L
    private var interruptionId: Long? = null
    private var interruptionStartMs = 0L

    var tripId: Long = 0L
        private set
    var isFinished = false
        private set
    val isResting: Boolean get() = restId != null
    val isInterrupted: Boolean get() = interruptionId != null

    /** 여정을 만들고 첫 분석 세션 토큰을 돌려준다 */
    suspend fun start(): Long {
        startMs = time.monotonicMs()
        engine = DriverStateEngine(config, baseline, startMs)
        tripId = repository.createTrip(
            startedAtMs = time.toWallClock(startMs),
            mode = mode,
            configVersion = config.versionTag(),
            modelInfo = modelInfo,
            calibrationSummary = baseline?.summary(),
        )
        return sessionGate.openNew()
    }

    /** [token] 은 프레임을 제출할 때의 세션 토큰. 현재 세션이 아니면 버리고 null 을 돌려준다. */
    suspend fun onObservation(token: Long, observation: FaceObservation): SessionUpdate? {
        if (isFinished || !sessionGate.accepts(token)) return null
        return handle(engine.onObservation(observation), observation.timestampMs)
    }

    suspend fun onTick(nowMs: Long = time.monotonicMs()): SessionUpdate? {
        if (isFinished) return null
        return handle(engine.onTick(nowMs), nowMs)
    }

    fun onLocation(fix: LocationFix) {
        if (isFinished) return
        locationGate.offer(fix)
        distance.add(fix)
    }

    /** 앱이 화면에서 벗어나거나 카메라 오류로 분석이 끊겼다. 기록하고 분석을 멈춘다. */
    suspend fun interrupt(reason: InterruptionReason) {
        if (isFinished || isResting || isInterrupted) return
        val now = time.monotonicMs()
        sessionGate.close()
        persist(engine.pause(now, EventEndReason.APP_PAUSED).changes)
        interruptionStartMs = now
        interruptionId = repository.insertInterruption(tripId, reason, time.toWallClock(now))
    }

    /** 중단됐던 분석을 다시 시작한다. 새 세션 토큰을 돌려준다. */
    suspend fun resumeAfterInterruption(): Long? {
        val id = interruptionId ?: return null
        val now = time.monotonicMs()
        repository.endInterruption(id, time.toWallClock(now), now - interruptionStartMs)
        interruptionId = null
        engine.resume(now)
        return sessionGate.openNew()
    }

    /** 사용자가 휴식을 시작했다. 진행 중인 징후·판정을 닫고 분석 판단을 멈춘다. */
    suspend fun startRest() {
        if (isFinished || isResting) return
        if (isInterrupted) resumeAfterInterruption()
        val now = time.monotonicMs()
        sessionGate.close()
        persist(engine.pause(now, EventEndReason.REST_STARTED).changes)
        alertPolicy.reset()
        restStartMs = now
        restId = repository.insertRest(tripId, time.toWallClock(now))
    }

    /** 휴식을 끝내고 새 세션 토큰을 돌려준다. */
    suspend fun endRest(): Long? {
        val id = restId ?: return null
        val now = time.monotonicMs()
        repository.endRest(id, time.toWallClock(now), now - restStartMs)
        restId = null
        engine.resume(now)
        return sessionGate.openNew()
    }

    /** 여정을 끝낸다. 진행 중인 이벤트·휴식·중단 기록은 종료 시각으로 닫는다. */
    suspend fun finish(): Long {
        if (isFinished) return tripId
        sessionGate.close()
        if (isResting) endRest()
        val now = time.monotonicMs()
        interruptionId?.let { repository.endInterruption(it, time.toWallClock(now), now - interruptionStartMs) }
        interruptionId = null
        persist(engine.finish(now).changes)
        repository.finishTrip(
            tripId = tripId,
            endedAtMs = time.toWallClock(now),
            durationMs = now - startMs,
            distanceMeters = distance.distanceOrNull(),
            validObservationMs = engine.validObservedMs,
            unavailableMs = engine.invalidObservedMs + engine.gapMs,
        )
        isFinished = true
        return tripId
    }

    private suspend fun handle(output: EngineOutput, nowMs: Long): SessionUpdate {
        persist(output.changes)
        return SessionUpdate(
            display = output.display,
            cautionCause = output.cautionCause,
            drowsiness = output.drowsiness,
            distraction = output.distraction,
            unavailableReason = output.unavailableReason,
            alert = if (isResting || isInterrupted) null else alertPolicy.select(output.alertRequests, nowMs),
            snapshot = output.snapshot,
            isResting = isResting,
        )
    }

    private suspend fun persist(changes: List<EventChange>) {
        for (change in changes) {
            when (change) {
                is EventChange.Opened -> {
                    val fix = locationGate.forEvent(change.confirmedAtMs)
                    val id = repository.insertEvent(
                        DriverEventVo(
                            tripId = tripId,
                            type = change.type,
                            level = change.level,
                            candidateStartedAtMs = time.toWallClock(change.candidateStartMs),
                            confirmedAtMs = time.toWallClock(change.confirmedAtMs),
                            endedAtMs = null,
                            durationMs = null,
                            endReason = null,
                            rules = change.rules.joinToString(","),
                            metrics = change.metrics,
                            latitude = fix?.latitude,
                            longitude = fix?.longitude,
                            locationAccuracyMeters = fix?.accuracyMeters,
                            locationAtMs = fix?.let { time.toWallClock(it.monotonicMs) },
                            source = mode,
                        ),
                    )
                    events[change.key] = StoredEvent(id, change.confirmedAtMs, change.level, change.rules, change.metrics)
                }
                is EventChange.Updated -> {
                    val stored = events[change.key] ?: continue
                    stored.level = change.level
                    stored.rules = change.rules
                    stored.metrics = change.metrics
                    repository.updateEvent(stored.dbId, stored.level, stored.rules.joinToString(","), stored.metrics, null, null, null)
                }
                is EventChange.Closed -> {
                    val stored = events.remove(change.key) ?: continue
                    if (change.metrics.isNotEmpty()) stored.metrics = change.metrics
                    repository.updateEvent(
                        eventId = stored.dbId,
                        level = stored.level,
                        rules = stored.rules.joinToString(","),
                        metrics = stored.metrics,
                        endedAtMs = time.toWallClock(change.endMs),
                        durationMs = change.durationMs,
                        endReason = change.reason,
                    )
                }
            }
        }
    }
}
