package com.personal.driverguard

import android.app.Application
import androidx.lifecycle.ViewModelProvider
import kotlinx.coroutines.launch

class App : Application() {

    val viewModelFactory: ViewModelProvider.Factory by lazy { ViewModelFactory.create(this) }

    override fun onCreate() {
        super.onCreate()
        val processStartedAt = PlatformLocator.timeSource.wallClockMs()
        PlatformLocator.appScope.launch {
            RepositoryLocator.provideTripRepository(this@App).closeUnfinishedTrips(startedBeforeMs = processStartedAt)
        }
    }
}
