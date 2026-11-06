package com.personal.driverguard

import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.createSavedStateHandle
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.personal.driverguard.views.calibration.CalibrationViewModel
import com.personal.driverguard.views.devsettings.DevViewModel
import com.personal.driverguard.views.driving.DrivingViewModel
import com.personal.driverguard.views.home.HomeViewModel
import com.personal.driverguard.views.tripdetail.TripDetailViewModel
import com.personal.driverguard.views.trips.TripListViewModel

/** 화면별 ViewModel 에 저장소·플랫폼 구현을 주입한다. 새 화면을 추가하면 여기에 initializer 를 등록한다. */
object ViewModelFactory {

    fun create(app: App): ViewModelProvider.Factory = viewModelFactory {
        val platform = PlatformLocator
        initializer {
            HomeViewModel(platform.provideModelAssets(app))
        }
        initializer {
            CalibrationViewModel(
                context = app,
                settingsRepository = RepositoryLocator.provideSettingsRepository(app),
                calibrationStore = platform.calibrationStore,
                stats = platform.analysisStats,
            )
        }
        initializer {
            DrivingViewModel(
                context = app,
                tripRepository = RepositoryLocator.provideTripRepository(app),
                settingsRepository = RepositoryLocator.provideSettingsRepository(app),
                modelAssets = platform.provideModelAssets(app),
                calibrationStore = platform.calibrationStore,
                time = platform.timeSource,
                locationSource = platform.provideLocationSource(app),
                alertOutput = platform.createAlertOutput(app),
                restAreaGuide = platform.provideRestAreaGuide(app),
                stats = platform.analysisStats,
                appScope = platform.appScope,
            )
        }
        initializer {
            TripListViewModel(RepositoryLocator.provideTripRepository(app))
        }
        initializer {
            TripDetailViewModel(createSavedStateHandle(), RepositoryLocator.provideTripRepository(app))
        }
        initializer {
            DevViewModel(
                context = app,
                settingsRepository = RepositoryLocator.provideSettingsRepository(app),
                modelAssets = platform.provideModelAssets(app),
                calibrationStore = platform.calibrationStore,
                statsHolder = platform.analysisStats,
                metricLog = platform.provideMetricLog(app),
            )
        }
    }
}
