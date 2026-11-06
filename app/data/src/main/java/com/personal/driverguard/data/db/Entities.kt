package com.personal.driverguard.data.db

import androidx.room.ColumnInfo
import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/*
 * 저장하는 것은 수치 요약뿐이다. 원본 영상, 얼굴 이미지, 프레임별 전체 랜드마크는 저장하지 않는다.
 * 시각 컬럼은 모두 벽시계(epoch ms). 지속 시간은 단조 시계로 계산한 값이다.
 */

@Entity(tableName = "trips")
data class TripEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val startedAtMs: Long,
    val endedAtMs: Long? = null,
    val durationMs: Long? = null,
    val distanceMeters: Double? = null,
    /** AnalysisMode.name */
    val mode: String,
    val validObservationMs: Long? = null,
    val unavailableMs: Long? = null,
    val configVersion: String,
    val modelInfo: String?,
    val calibrationSummary: String?,
    val endedAbnormally: Boolean = false,
)

@Entity(
    tableName = "events",
    foreignKeys = [ForeignKey(TripEntity::class, ["id"], ["tripId"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("tripId")],
)
data class EventEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val tripId: Long,
    /** EventType.name */
    val type: String,
    /** AlertLevel.name */
    val level: String,
    val candidateStartedAtMs: Long,
    val confirmedAtMs: Long,
    val endedAtMs: Long?,
    val durationMs: Long?,
    /** EventEndReason.name */
    val endReason: String?,
    val rules: String,
    val metrics: String,
    val latitude: Double?,
    val longitude: Double?,
    val locationAccuracyMeters: Float?,
    val locationAtMs: Long?,
    /** AnalysisMode.name */
    val source: String,
)

@Entity(
    tableName = "rests",
    foreignKeys = [ForeignKey(TripEntity::class, ["id"], ["tripId"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("tripId")],
)
data class RestEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val tripId: Long,
    val startedAtMs: Long,
    val endedAtMs: Long? = null,
    val durationMs: Long? = null,
)

@Entity(
    tableName = "interruptions",
    foreignKeys = [ForeignKey(TripEntity::class, ["id"], ["tripId"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("tripId")],
)
data class InterruptionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val tripId: Long,
    /** InterruptionReason.name */
    val reason: String,
    val startedAtMs: Long,
    val endedAtMs: Long? = null,
    val durationMs: Long? = null,
)

data class TripWithEventCount(
    @Embedded val trip: TripEntity,
    @ColumnInfo(name = "eventCount") val eventCount: Int,
)
