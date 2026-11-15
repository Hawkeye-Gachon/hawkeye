package com.personal.driverguard

import android.content.Context
import com.personal.driverguard.data.remote.HawkeyeApi
import com.personal.driverguard.data.remote.TokenStore

/** 수동 DI. 백엔드 클라이언트를 한 곳에서 만들어 앱 전체가 같은 인스턴스를 쓰게 한다. */
object RepositoryLocator {

    @Volatile
    private var api: HawkeyeApi? = null

    /** Hawkeye 백엔드 클라이언트. 주소는 BuildConfig.API_BASE_URL(app/build.gradle.kts 참고). */
    fun provideApi(context: Context): HawkeyeApi =
        api ?: synchronized(this) {
            api ?: HawkeyeApi(BuildConfig.API_BASE_URL, TokenStore.create(context), debugLogging = BuildConfig.DEBUG).also { api = it }
        }
}
