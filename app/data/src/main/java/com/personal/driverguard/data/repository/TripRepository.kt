package com.personal.driverguard.data.repository

import android.content.Context
import com.personal.driverguard.data.db.DriverGuardDatabase
import com.personal.driverguard.data.db.EventEntity
import com.personal.driverguard.data.db.InterruptionEntity
import com.personal.driverguard.data.db.RestEntity
import com.personal.driverguard.data.db.TripDao
import com.personal.driverguard.data.db.TripEntity
import com.personal.driverguard.domain.repository.ITripRepository
import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.entity.DriverEventVo
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.InterruptionReason
import com.personal.driverguard.entity.InterruptionVo
import com.personal.driverguard.entity.RestVo
import com.personal.driverguard.entity.TripDetailVo
import com.personal.driverguard.entity.TripListItemVo
import com.personal.driverguard.entity.TripVo
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map

class TripRepository(private val dao: TripDao) : ITripRepository {

    companion object {
        /** Room 을 data 모듈 밖으로 드러내지 않고 저장소를 만든다. */
        fun create(context: Context): ITripRepository = TripRepository(DriverGuardDatabase.create(context).tripDao())
    }

    override suspend fun createTrip(
        startedAtMs: Long,
        mode: AnalysisMode,
        configVersion: String,
        modelInfo: String?,
        calibrationSummary: String?,
    ): Long = dao.insertTrip(
        TripEntity(
            startedAtMs = startedAtMs,
            mode = mode.name,
            configVersion = configVersion,
            modelInfo = modelInfo,
            calibrationSummary = calibrationSummary,
        ),
    )

    override suspend fun finishTrip(
        tripId: Long,
        endedAtMs: Long,
        durationMs: Long,
        distanceMeters: Double?,
        validObservationMs: Long,
        unavailableMs: Long,
    ) = dao.finishTrip(tripId, endedAtMs, durationMs, distanceMeters, validObservationMs, unavailableMs, abnormal = false)

    override suspend fun insertEvent(event: DriverEventVo): Long = dao.insertEvent(event.toEntity())

    override suspend fun updateEvent(
        eventId: Long,
        level: AlertLevel,
        rules: String,
        metrics: String,
        endedAtMs: Long?,
        durationMs: Long?,
        endReason: EventEndReason?,
    ) = dao.updateEvent(eventId, level.name, rules, metrics, endedAtMs, durationMs, endReason?.name)

    override suspend fun insertRest(tripId: Long, startedAtMs: Long): Long =
        dao.insertRest(RestEntity(tripId = tripId, startedAtMs = startedAtMs))

    override suspend fun endRest(restId: Long, endedAtMs: Long, durationMs: Long) =
        dao.endRest(restId, endedAtMs, durationMs)

    override suspend fun insertInterruption(tripId: Long, reason: InterruptionReason, startedAtMs: Long): Long =
        dao.insertInterruption(InterruptionEntity(tripId = tripId, reason = reason.name, startedAtMs = startedAtMs))

    override suspend fun endInterruption(interruptionId: Long, endedAtMs: Long, durationMs: Long) =
        dao.endInterruption(interruptionId, endedAtMs, durationMs)

    /**
     * 끝나지 않은 여정을 마지막 기록 시각으로 닫고 비정상 종료로 표시한다.
     * 진행 중이던 이벤트의 종료 시각은 알 수 없으므로 비워 두고 종료 사유만 남긴다(리포트에 '알 수 없음'으로 표시).
     */
    override suspend fun closeUnfinishedTrips(startedBeforeMs: Long): Int {
        val trips = dao.unfinishedTrips(startedBeforeMs)
        for (trip in trips) {
            val end = maxOf(dao.lastRecordedAt(trip.id) ?: trip.startedAtMs, trip.startedAtMs)
            dao.markOpenEvents(trip.id, EventEndReason.ABNORMAL_TERMINATION.name)
            dao.finishTrip(trip.id, end, end - trip.startedAtMs, null, null, null, abnormal = true)
        }
        return trips.size
    }

    override fun observeTrips(): Flow<List<TripListItemVo>> =
        dao.observeTrips().map { rows -> rows.map { TripListItemVo(it.trip.toVo(), it.eventCount) } }

    override fun observeTripDetail(tripId: Long): Flow<TripDetailVo?> =
        combine(
            dao.observeTrip(tripId),
            dao.observeEvents(tripId),
            dao.observeRests(tripId),
            dao.observeInterruptions(tripId),
        ) { trip, events, rests, interruptions ->
            trip?.let {
                TripDetailVo(
                    it.toVo(),
                    events.map { e -> e.toVo() },
                    rests.map { r -> r.toVo() },
                    interruptions.map { i -> i.toVo() },
                )
            }
        }
}

private fun TripEntity.toVo() = TripVo(
    id = id,
    startedAtMs = startedAtMs,
    endedAtMs = endedAtMs,
    durationMs = durationMs,
    distanceMeters = distanceMeters,
    mode = AnalysisMode.valueOf(mode),
    validObservationMs = validObservationMs,
    unavailableMs = unavailableMs,
    configVersion = configVersion,
    modelInfo = modelInfo,
    calibrationSummary = calibrationSummary,
    endedAbnormally = endedAbnormally,
)

private fun DriverEventVo.toEntity() = EventEntity(
    tripId = tripId,
    type = type.name,
    level = level.name,
    candidateStartedAtMs = candidateStartedAtMs,
    confirmedAtMs = confirmedAtMs,
    endedAtMs = endedAtMs,
    durationMs = durationMs,
    endReason = endReason?.name,
    rules = rules,
    metrics = metrics,
    latitude = latitude,
    longitude = longitude,
    locationAccuracyMeters = locationAccuracyMeters,
    locationAtMs = locationAtMs,
    source = source.name,
)

private fun EventEntity.toVo() = DriverEventVo(
    id = id,
    tripId = tripId,
    type = EventType.valueOf(type),
    level = AlertLevel.valueOf(level),
    candidateStartedAtMs = candidateStartedAtMs,
    confirmedAtMs = confirmedAtMs,
    endedAtMs = endedAtMs,
    durationMs = durationMs,
    endReason = endReason?.let(EventEndReason::valueOf),
    rules = rules,
    metrics = metrics,
    latitude = latitude,
    longitude = longitude,
    locationAccuracyMeters = locationAccuracyMeters,
    locationAtMs = locationAtMs,
    source = AnalysisMode.valueOf(source),
)

private fun RestEntity.toVo() = RestVo(id, tripId, startedAtMs, endedAtMs, durationMs)

private fun InterruptionEntity.toVo() =
    InterruptionVo(id, tripId, InterruptionReason.valueOf(reason), startedAtMs, endedAtMs, durationMs)
