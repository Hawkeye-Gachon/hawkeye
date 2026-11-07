package com.personal.driverguard.ui.state

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Looper
import androidx.core.content.ContextCompat
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.emptyFlow

/** 관찰 중 받은 위치 한 점. 쉼터 조회에는 진행 방향·속도도 쓴다. */
data class GeoFix(
    val latitude: Double,
    val longitude: Double,
    val accuracyMeters: Float?,
    /** 진행 방향(북쪽 0도, 시계 방향). 모르면 null */
    val bearingDeg: Float?,
    /** m/s. 모르면 null */
    val speedMps: Float?,
    /** elapsedRealtime 기준 측정 시각 */
    val monotonicMs: Long,
)

/**
 * LocationManager 로 위치를 받는다(Play 서비스 불필요). 권한이 없으면 아무것도 내보내지 않는다.
 * Pleos FusedLocation SDK 는 아직 '준비 중'이라 일반 Android 위치를 쓴다.
 */
class LocationTracker(private val context: Context) {

    fun hasPermission(): Boolean = listOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        .any { ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED }

    @SuppressLint("MissingPermission")
    fun updates(): Flow<GeoFix> {
        val manager = context.getSystemService(LocationManager::class.java) ?: return emptyFlow()
        if (!hasPermission()) return emptyFlow()
        return callbackFlow {
            val listener = object : LocationListener {
                override fun onLocationChanged(location: Location) {
                    trySend(location.toFix())
                }

                @Deprecated("Deprecated in Java")
                override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit
                override fun onProviderEnabled(provider: String) = Unit
                override fun onProviderDisabled(provider: String) = Unit
            }
            val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
                .filter { runCatching { manager.isProviderEnabled(it) }.getOrDefault(false) }
            providers.forEach { runCatching { manager.requestLocationUpdates(it, INTERVAL_MS, 0f, listener, Looper.getMainLooper()) } }
            // 마지막으로 알려진 위치로 바로 시작한다(첫 GPS 고정까지 기다리지 않도록)
            providers.mapNotNull { runCatching { manager.getLastKnownLocation(it) }.getOrNull() }
                .maxByOrNull { it.elapsedRealtimeNanos }
                ?.let { trySend(it.toFix()) }
            awaitClose { manager.removeUpdates(listener) }
        }
    }

    private fun Location.toFix() = GeoFix(
        latitude = latitude,
        longitude = longitude,
        accuracyMeters = if (hasAccuracy()) accuracy else null,
        bearingDeg = if (hasBearing()) bearing else null,
        speedMps = if (hasSpeed()) speed else null,
        monotonicMs = elapsedRealtimeNanos / 1_000_000,
    )

    private companion object {
        const val INTERVAL_MS = 1_000L
    }
}
