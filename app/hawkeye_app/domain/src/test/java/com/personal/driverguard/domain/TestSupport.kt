package com.personal.driverguard.domain

import com.personal.driverguard.domain.analysis.geometry.PoseMath
import com.personal.driverguard.domain.demo.DemoScript
import com.personal.driverguard.domain.engine.DriverStateEngine
import com.personal.driverguard.domain.engine.EngineOutput
import com.personal.driverguard.domain.engine.EventChange
import com.personal.driverguard.domain.platform.TimeSource
import com.personal.driverguard.domain.repository.ITripRepository
import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.entity.DriverEventVo
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.InterruptionReason
import com.personal.driverguard.entity.InterruptionVo
import com.personal.driverguard.entity.QualityFlag
import com.personal.driverguard.entity.RestVo
import com.personal.driverguard.entity.TripDetailVo
import com.personal.driverguard.entity.TripListItemVo
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf

class FakeTime(var now: Long = 1_000_000L) : TimeSource {
    override fun monotonicMs(): Long = now
    override fun wallClockMs(): Long = now + WALL_OFFSET

    companion object {
        const val WALL_OFFSET = 1_700_000_000_000L
    }
}

val BASE = DemoScript.BASELINE

/** 기준값(눈 0.30, 입 0.05, 정면) 대비 합성 관측 */
fun obs(
    t: Long,
    left: Double? = 0.30,
    right: Double? = 0.30,
    mouth: Double = 0.05,
    yaw: Double = 0.0,
    pitch: Double = 0.0,
    eyeQuality: QualityFlag = if (left != null && right != null) QualityFlag.VALID else QualityFlag.LANDMARKS_INVALID,
) = FaceObservation(
    timestampMs = t,
    faceCount = 1,
    faceQuality = QualityFlag.VALID,
    leftEar = left,
    rightEar = right,
    eyeQuality = eyeQuality,
    leftEyeValid = left != null,
    rightEyeValid = right != null,
    mouthOpenRatio = mouth,
    mouthQuality = QualityFlag.VALID,
    pose = PoseMath.poseOf(yaw, pitch),
    poseQuality = QualityFlag.VALID,
    faceWidthRatio = 0.35,
)

/** [from, to) 를 [periodMs] 간격 관측으로 채워 엔진에 넣고 모든 출력을 돌려준다. */
fun DriverStateEngine.feed(
    from: Long,
    to: Long,
    periodMs: Long = 33,
    make: (Long) -> FaceObservation?,
): List<EngineOutput> {
    val out = mutableListOf<EngineOutput>()
    var t = from
    while (t < to) {
        make(t)?.let { out += onObservation(it) } ?: out.add(onTick(t))
        t += periodMs
    }
    return out
}

fun List<EngineOutput>.opened(type: EventType) = flatMap { it.changes }.filterIsInstance<EventChange.Opened>().filter { it.type == type }

fun List<EngineOutput>.closed() = flatMap { it.changes }.filterIsInstance<EventChange.Closed>()

fun engine(config: AnalysisConfig = AnalysisConfig(), start: Long = 0L) = DriverStateEngine(config, BASE, start)

/** 저장 내용을 메모리에 보관하는 저장소 */
class FakeTripRepository : ITripRepository {
    data class Trip(
        val id: Long,
        val mode: AnalysisMode,
        val modelInfo: String?,
        var endedAtMs: Long? = null,
        var validObservationMs: Long? = null,
        var unavailableMs: Long? = null,
    )

    val trips = mutableListOf<Trip>()
    val events = mutableMapOf<Long, DriverEventVo>()
    val rests = mutableMapOf<Long, RestVo>()
    val interruptions = mutableMapOf<Long, InterruptionVo>()
    private var nextId = 1L

    override suspend fun createTrip(startedAtMs: Long, mode: AnalysisMode, configVersion: String, modelInfo: String?, calibrationSummary: String?): Long {
        val id = nextId++
        trips += Trip(id, mode, modelInfo)
        return id
    }

    override suspend fun finishTrip(tripId: Long, endedAtMs: Long, durationMs: Long, distanceMeters: Double?, validObservationMs: Long, unavailableMs: Long) {
        trips.first { it.id == tripId }.apply {
            this.endedAtMs = endedAtMs
            this.validObservationMs = validObservationMs
            this.unavailableMs = unavailableMs
        }
    }

    override suspend fun insertEvent(event: DriverEventVo): Long {
        val id = nextId++
        events[id] = event.copy(id = id)
        return id
    }

    override suspend fun updateEvent(eventId: Long, level: AlertLevel, rules: String, metrics: String, endedAtMs: Long?, durationMs: Long?, endReason: EventEndReason?) {
        events[eventId] = events.getValue(eventId).copy(level = level, rules = rules, metrics = metrics, endedAtMs = endedAtMs, durationMs = durationMs, endReason = endReason)
    }

    override suspend fun insertRest(tripId: Long, startedAtMs: Long): Long {
        val id = nextId++
        rests[id] = RestVo(id, tripId, startedAtMs, null, null)
        return id
    }

    override suspend fun endRest(restId: Long, endedAtMs: Long, durationMs: Long) {
        rests[restId] = rests.getValue(restId).copy(endedAtMs = endedAtMs, durationMs = durationMs)
    }

    override suspend fun insertInterruption(tripId: Long, reason: InterruptionReason, startedAtMs: Long): Long {
        val id = nextId++
        interruptions[id] = InterruptionVo(id, tripId, reason, startedAtMs, null, null)
        return id
    }

    override suspend fun endInterruption(interruptionId: Long, endedAtMs: Long, durationMs: Long) {
        interruptions[interruptionId] = interruptions.getValue(interruptionId).copy(endedAtMs = endedAtMs, durationMs = durationMs)
    }

    override suspend fun closeUnfinishedTrips(startedBeforeMs: Long): Int = 0

    override fun observeTrips(): Flow<List<TripListItemVo>> = flowOf(emptyList())

    override fun observeTripDetail(tripId: Long): Flow<TripDetailVo?> = flowOf(null)

    fun eventsOf(type: EventType) = events.values.filter { it.type == type }
}
