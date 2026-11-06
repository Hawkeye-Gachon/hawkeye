package com.personal.driverguard.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * 버전 2: 분석 방식이 MediaPipe 기반으로 바뀌며 이벤트 생명주기·중단 기록 컬럼이 생겼다.
 * 시제품이라 이전 버전(ONNX 기반) 기록은 이전하지 않고 지운다.
 */
@Database(
    entities = [TripEntity::class, EventEntity::class, RestEntity::class, InterruptionEntity::class],
    version = 2,
    exportSchema = true,
)
abstract class DriverGuardDatabase : RoomDatabase() {
    abstract fun tripDao(): TripDao

    companion object {
        fun create(context: Context): DriverGuardDatabase =
            Room.databaseBuilder(context.applicationContext, DriverGuardDatabase::class.java, "driverguard.db")
                .fallbackToDestructiveMigrationFrom(dropAllTables = true, 1)
                .build()
    }
}
