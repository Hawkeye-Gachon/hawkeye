package com.personal.driverguard

import android.content.Context
import com.personal.driverguard.data.remote.HawkeyeApi
import com.personal.driverguard.data.remote.TokenStore
import com.personal.driverguard.data.repository.SettingsRepository
import com.personal.driverguard.data.repository.TripRepository
import com.personal.driverguard.domain.repository.ISettingsRepository
import com.personal.driverguard.domain.repository.ITripRepository

/** 수동 DI. 저장소 구현체를 한 곳에서 만들어 앱 전체가 같은 인스턴스를 쓰게 한다. */
object RepositoryLocator {

    @Volatile
    private var tripRepository: ITripRepository? = null

    @Volatile
    private var settingsRepository: ISettingsRepository? = null

    @Volatile
    private var api: HawkeyeApi? = null

    /** DriverGuard 백엔드 클라이언트. 주소는 BuildConfig.API_BASE_URL(app/build.gradle.kts 참고). */
    fun provideApi(context: Context): HawkeyeApi =
        api ?: synchronized(this) {
            api ?: HawkeyeApi(BuildConfig.API_BASE_URL, TokenStore.create(context), debugLogging = BuildConfig.DEBUG).also { api = it }
        }

    fun provideTripRepository(context: Context): ITripRepository =
        tripRepository ?: synchronized(this) {
            tripRepository ?: TripRepository.create(context).also { tripRepository = it }
        }

    fun provideSettingsRepository(context: Context): ISettingsRepository =
        settingsRepository ?: synchronized(this) {
            settingsRepository ?: SettingsRepository(context).also { settingsRepository = it }
        }
}
