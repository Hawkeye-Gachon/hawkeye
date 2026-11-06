package com.personal.driverguard.platform.android

import android.Manifest
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.Uri
import android.os.Bundle
import android.os.Looper
import android.os.SystemClock
import androidx.core.content.ContextCompat
import com.personal.driverguard.R
import com.personal.driverguard.domain.platform.ILocationSource
import com.personal.driverguard.domain.platform.IRestAreaGuide
import com.personal.driverguard.domain.platform.TimeSource
import com.personal.driverguard.entity.LocationFix
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

/*
 * 일반 Android 플랫폼 구현.
 * Pleos Connect 에서 차량 위치·지도·음성 출력을 별도 SDK 로 제공한다면 domain.platform 인터페이스로 새 구현을 만들어
 * PlatformLocator 에서 교체한다. 현재 Pleos 전용 연동 코드는 없다.
 */

object AndroidTimeSource : TimeSource {
    override fun monotonicMs(): Long = SystemClock.elapsedRealtime()
    override fun wallClockMs(): Long = System.currentTimeMillis()
}

/** LocationManager 기반 위치(Play 서비스 불필요). 권한이 없으면 아무것도 내보내지 않는다. */
class AndroidLocationSource(private val context: Context) : ILocationSource {

    override fun hasPermission(): Boolean =
        granted(Manifest.permission.ACCESS_FINE_LOCATION) || granted(Manifest.permission.ACCESS_COARSE_LOCATION)

    @SuppressLint("MissingPermission")
    @Suppress("OVERRIDE_DEPRECATION")
    override fun locationUpdates(): Flow<LocationFix> = callbackFlow {
        val manager = context.getSystemService(LocationManager::class.java)
        if (!hasPermission() || manager == null) {
            close()
            return@callbackFlow
        }
        // API 29 이하에서는 LocationListener 의 나머지 메서드가 추상 메서드라 람다 대신 object 로 구현한다
        val listener = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                trySend(location.toFix())
            }
            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit
            override fun onProviderEnabled(provider: String) = Unit
            override fun onProviderDisabled(provider: String) = Unit
        }
        val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
            .filter { runCatching { manager.isProviderEnabled(it) }.getOrDefault(false) }
        providers.forEach { manager.requestLocationUpdates(it, UPDATE_INTERVAL_MS, 0f, listener, Looper.getMainLooper()) }
        awaitClose { manager.removeUpdates(listener) }
    }

    private fun granted(permission: String) =
        ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED

    private fun Location.toFix() = LocationFix(
        latitude = latitude,
        longitude = longitude,
        accuracyMeters = if (hasAccuracy()) accuracy else null,
        monotonicMs = elapsedRealtimeNanos / 1_000_000,
    )

    private companion object {
        const val UPDATE_INTERVAL_MS = 2_000L
    }
}

/**
 * 설치된 지도 앱에 '휴게소' 검색을 넘긴다. 앱이 장소를 고르거나 추천하지 않는다.
 * 실제 휴식 장소 검색·길 안내 API(지도 SDK, Pleos 내비게이션 등)가 확인되면 이 인터페이스로 교체한다.
 */
class AndroidRestAreaGuide(private val context: Context) : IRestAreaGuide {

    private fun searchIntent() = Intent(
        Intent.ACTION_VIEW,
        Uri.parse("geo:0,0?q=" + Uri.encode(context.getString(R.string.rest_area_search_query))),
    ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

    override fun isAvailable(): Boolean = searchIntent().resolveActivity(context.packageManager) != null

    override fun openRestAreaSearch(): Boolean = try {
        context.startActivity(searchIntent())
        true
    } catch (_: ActivityNotFoundException) {
        false
    }
}
