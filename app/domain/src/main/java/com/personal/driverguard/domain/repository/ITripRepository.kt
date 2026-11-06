package com.personal.driverguard.domain.repository

import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.entity.DriverEventVo
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.InterruptionReason
import com.personal.driverguard.entity.TripDetailVo
import com.personal.driverguard.entity.TripListItemVo
import kotlinx.coroutines.flow.Flow

/** 여정·이벤트·휴식·분석 중단 저장소. 시각 인자는 모두 벽시계(epoch ms) 기준이다. */
interface ITripRepository {
    suspend fun createTrip(
        startedAtMs: Long,
        mode: AnalysisMode,
        configVersion: String,
        modelInfo: String?,
        calibrationSummary: String?,
    ): Long

    suspend fun finishTrip(
        tripId: Long,
        endedAtMs: Long,
        durationMs: Long,
        distanceMeters: Double?,
        validObservationMs: Long,
        unavailableMs: Long,
    )

    suspend fun insertEvent(event: DriverEventVo): Long

    suspend fun updateEvent(
        eventId: Long,
        level: AlertLevel,
        rules: String,
        metrics: String,
        endedAtMs: Long?,
        durationMs: Long?,
        endReason: EventEndReason?,
    )

    suspend fun insertRest(tripId: Long, startedAtMs: Long): Long

    suspend fun endRest(restId: Long, endedAtMs: Long, durationMs: Long)

    suspend fun insertInterruption(tripId: Long, reason: InterruptionReason, startedAtMs: Long): Long

    suspend fun endInterruption(interruptionId: Long, endedAtMs: Long, durationMs: Long)

    /**
     * 앱이 비정상 종료되어 끝나지 않은 여정 중 [startedBeforeMs] 이전에 시작된 것을 정리한다.
     * 이번 실행에서 시작한 여정을 닫지 않도록 프로세스 시작 시각을 넘긴다. 정리한 여정 수를 돌려준다.
     */
    suspend fun closeUnfinishedTrips(startedBeforeMs: Long): Int

    fun observeTrips(): Flow<List<TripListItemVo>>

    fun observeTripDetail(tripId: Long): Flow<TripDetailVo?>
}
