package com.personal.driverguard.domain.engine

import com.personal.driverguard.entity.LocationFix
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 위치 점들로 이동 거리를 누적한다. 정확도가 낮은 점, 정지 중 흔들림, 비현실적인 점프는 버린다.
 */
class DistanceAccumulator(
    private val maxAccuracyMeters: Float = 30f,
    private val minStepMeters: Double = 5.0,
    private val maxSpeedMps: Double = 70.0,
) {
    var totalMeters: Double = 0.0
        private set

    /** 쓸 만한 위치를 한 번이라도 받았는지. 아니면 거리를 '기록 없음'으로 본다. */
    var hasData: Boolean = false
        private set

    private var last: LocationFix? = null

    fun add(fix: LocationFix) {
        val accuracy = fix.accuracyMeters
        if (accuracy == null || accuracy > maxAccuracyMeters) return
        hasData = true
        val previous = last
        if (previous == null) {
            last = fix
            return
        }
        val seconds = (fix.monotonicMs - previous.monotonicMs) / 1000.0
        if (seconds <= 0) return
        val meters = haversineMeters(previous.latitude, previous.longitude, fix.latitude, fix.longitude)
        if (meters / seconds > maxSpeedMps) return
        if (meters < max(minStepMeters, accuracy.toDouble())) return
        totalMeters += meters
        last = fix
    }

    fun distanceOrNull(): Double? = if (hasData) totalMeters else null

    companion object {
        private const val EARTH_RADIUS_M = 6_371_000.0

        fun haversineMeters(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val dLat = Math.toRadians(lat2 - lat1)
            val dLon = Math.toRadians(lon2 - lon1)
            val a = sin(dLat / 2).pow(2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon / 2).pow(2)
            return 2 * EARTH_RADIUS_M * asin(sqrt(a))
        }
    }
}
