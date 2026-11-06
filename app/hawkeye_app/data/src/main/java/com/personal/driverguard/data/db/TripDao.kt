package com.personal.driverguard.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface TripDao {
    @Insert
    suspend fun insertTrip(trip: TripEntity): Long

    @Query(
        "UPDATE trips SET endedAtMs = :endedAtMs, durationMs = :durationMs, distanceMeters = :distanceMeters, " +
            "validObservationMs = :validObservationMs, unavailableMs = :unavailableMs, endedAbnormally = :abnormal WHERE id = :tripId",
    )
    suspend fun finishTrip(
        tripId: Long,
        endedAtMs: Long,
        durationMs: Long?,
        distanceMeters: Double?,
        validObservationMs: Long?,
        unavailableMs: Long?,
        abnormal: Boolean,
    )

    @Insert
    suspend fun insertEvent(event: EventEntity): Long

    @Query(
        "UPDATE events SET level = :level, rules = :rules, metrics = :metrics, endedAtMs = :endedAtMs, " +
            "durationMs = :durationMs, endReason = :endReason WHERE id = :eventId",
    )
    suspend fun updateEvent(eventId: Long, level: String, rules: String, metrics: String, endedAtMs: Long?, durationMs: Long?, endReason: String?)

    /** 비정상 종료로 열린 채 남은 이벤트를 닫는다(지속 시간은 알 수 없다) */
    @Query("UPDATE events SET endReason = :endReason WHERE tripId = :tripId AND endedAtMs IS NULL")
    suspend fun markOpenEvents(tripId: Long, endReason: String)

    @Insert
    suspend fun insertRest(rest: RestEntity): Long

    @Query("UPDATE rests SET endedAtMs = :endedAtMs, durationMs = :durationMs WHERE id = :restId")
    suspend fun endRest(restId: Long, endedAtMs: Long, durationMs: Long)

    @Insert
    suspend fun insertInterruption(interruption: InterruptionEntity): Long

    @Query("UPDATE interruptions SET endedAtMs = :endedAtMs, durationMs = :durationMs WHERE id = :id")
    suspend fun endInterruption(id: Long, endedAtMs: Long, durationMs: Long)

    @Query("SELECT * FROM trips WHERE endedAtMs IS NULL AND startedAtMs < :startedBeforeMs")
    suspend fun unfinishedTrips(startedBeforeMs: Long): List<TripEntity>

    /** 여정에서 마지막으로 기록된 시각(이벤트·휴식·중단의 시작/종료 중 가장 늦은 것) */
    @Query(
        "SELECT MAX(t) FROM (" +
            "SELECT MAX(COALESCE(endedAtMs, confirmedAtMs)) AS t FROM events WHERE tripId = :tripId " +
            "UNION ALL SELECT MAX(COALESCE(endedAtMs, startedAtMs)) AS t FROM rests WHERE tripId = :tripId " +
            "UNION ALL SELECT MAX(COALESCE(endedAtMs, startedAtMs)) AS t FROM interruptions WHERE tripId = :tripId)",
    )
    suspend fun lastRecordedAt(tripId: Long): Long?

    @Query(
        "SELECT trips.*, (SELECT COUNT(*) FROM events WHERE events.tripId = trips.id) AS eventCount " +
            "FROM trips ORDER BY startedAtMs DESC",
    )
    fun observeTrips(): Flow<List<TripWithEventCount>>

    @Query("SELECT * FROM trips WHERE id = :tripId")
    fun observeTrip(tripId: Long): Flow<TripEntity?>

    @Query("SELECT * FROM events WHERE tripId = :tripId ORDER BY confirmedAtMs")
    fun observeEvents(tripId: Long): Flow<List<EventEntity>>

    @Query("SELECT * FROM rests WHERE tripId = :tripId ORDER BY startedAtMs")
    fun observeRests(tripId: Long): Flow<List<RestEntity>>

    @Query("SELECT * FROM interruptions WHERE tripId = :tripId ORDER BY startedAtMs")
    fun observeInterruptions(tripId: Long): Flow<List<InterruptionEntity>>
}
