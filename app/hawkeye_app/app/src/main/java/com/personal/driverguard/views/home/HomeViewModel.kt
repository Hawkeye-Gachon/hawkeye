package com.personal.driverguard.views.home

import androidx.lifecycle.viewModelScope
import com.personal.driverguard.analysis.ModelAssetStatus
import com.personal.driverguard.runtime.ModelAssetHolder
import com.personal.driverguard.views.base.BaseViewModel
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

sealed interface HomeEvent {
    /** 실제 분석: 카메라 위치 확인·캘리브레이션부터 */
    data object Calibrate : HomeEvent
}

class HomeViewModel(
    private val modelAssets: ModelAssetHolder,
) : BaseViewModel<HomeEvent>() {

    /** null 이면 확인 중 */
    val modelStatus: StateFlow<ModelAssetStatus?> = modelAssets.status

    init {
        viewModelScope.launch { modelAssets.ensureChecked() }
    }

    /** 분석은 카메라 권한과 모델이 있어야 시작할 수 있다. */
    fun start(cameraGranted: Boolean) {
        if (cameraGranted && modelStatus.value is ModelAssetStatus.Ready) sendEvent(HomeEvent.Calibrate)
    }
}
