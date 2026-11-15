package com.personal.driverguard.platform.pleos

import ai.pleos.playground.location.fused.FusedLocation
import ai.pleos.playground.location.fused.listener.FusedLocationListener
import ai.pleos.playground.location.fused.model.LocationData
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.personal.driverguard.ui.state.GeoFix
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

/**
 * Pleos FusedLocation(차량 GPS·주행 정보를 합친 위치)로 관찰 중 위치를 받는다.
 * 지원하지 않는 기기면 [isAvailable] 이 false 이고, 그때는 일반 Android 위치를 쓴다(HawkeyeViewModel).
 * 속도는 주지 않으므로 주행 기록이 앞 위치와 비교해 계산한다. 정확도도 주지 않아 차량 위치 기준 오차로 둔다.
 */
class PleosFusedLocation(context: Context) {
    private val appContext = context.applicationContext
    private var fused: FusedLocation? = null

    fun isAvailable(): Boolean = runCatching {
        val f = fused ?: FusedLocation(appContext).also { fused = it }
        f.isSupported().also { if (it) f.initialize() }
    }.onFailure { Log.w(TAG, "FusedLocation 을 쓸 수 없습니다", it) }.getOrDefault(false)

    fun updates(): Flow<GeoFix> = callbackFlow {
        val f = fused ?: FusedLocation(appContext).also { fused = it; it.initialize() }
        val listener = object : FusedLocationListener {
            override fun onChanged(data: LocationData?) { data?.let { trySend(it.toFix()) } }
            override fun onError(e: Exception) { Log.w(TAG, "위치 오류: ${e.message}") }
        }
        runCatching {
            f.getLastKnownLocation(listener)
            f.registerFusedLocationCallback(listener)
        }.onFailure { close(it) }
        awaitClose { runCatching { f.unregisterFusedLocationCallback(listener) } }
    }

    fun release() {
        runCatching { fused?.release() }
        fused = null
    }

    private fun LocationData.toFix() = GeoFix(
        latitude = latitude,
        longitude = longitude,
        accuracyMeters = ASSUMED_ACCURACY_M,
        bearingDeg = heading,
        speedMps = null,
        monotonicMs = SystemClock.elapsedRealtime(),
    )

    private companion object {
        const val TAG = "PleosFusedLocation"
        const val ASSUMED_ACCURACY_M = 10f
    }
}
