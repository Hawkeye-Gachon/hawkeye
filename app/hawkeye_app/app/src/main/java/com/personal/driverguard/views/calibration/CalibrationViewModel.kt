package com.personal.driverguard.views.calibration

import android.content.Context
import androidx.lifecycle.viewModelScope
import com.personal.driverguard.domain.analysis.calibration.CalibrationFailure
import com.personal.driverguard.domain.analysis.calibration.CalibrationManager
import com.personal.driverguard.domain.analysis.calibration.CalibrationState
import com.personal.driverguard.domain.analysis.calibration.ClosedEyeCalibration
import com.personal.driverguard.domain.repository.ISettingsRepository
import com.personal.driverguard.entity.CalibrationBaseline
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.QualityFlag
import com.personal.driverguard.runtime.AnalysisStatsHolder
import com.personal.driverguard.runtime.CalibrationStore
import com.personal.driverguard.runtime.LiveAnalysisSession
import com.personal.driverguard.views.base.BaseViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class CalibrationStep {
    /** 분석기 준비 */
    PREPARING,

    /** 평소 자세 기준값 측정 */
    COLLECTING,
    FAILED,

    /** 기준값 측정 완료. 선택적으로 눈 감김 보정 가능 */
    DONE,

    /** 선택: 눈 감김 보정 측정 중 */
    CLOSED_EYES,

    /** 분석기를 준비하지 못함(모델·OpenCV) */
    ERROR,
}

data class CalibrationUiState(
    val step: CalibrationStep = CalibrationStep.PREPARING,
    val progress: Double = 0.0,
    val issue: QualityFlag? = null,
    val failure: CalibrationFailure? = null,
    val baseline: CalibrationBaseline? = null,
    val closedEyeFailure: CalibrationFailure? = null,
    val error: String? = null,
    val cameraError: String? = null,
)

sealed interface CalibrationEvent {
    data object StartDriving : CalibrationEvent
}

/**
 * 주행 전 정차 상태에서 카메라 위치와 평소 운전 자세를 확인한다.
 * 기준값은 세션별로 측정해 메모리에만 두고 이번 주행에만 쓴다.
 */
class CalibrationViewModel(
    private val context: Context,
    private val settingsRepository: ISettingsRepository,
    private val calibrationStore: CalibrationStore,
    private val stats: AnalysisStatsHolder,
) : BaseViewModel<CalibrationEvent>() {

    private val config = settingsRepository.config.value
    private val _ui = MutableStateFlow(CalibrationUiState())
    val ui: StateFlow<CalibrationUiState> = _ui.asStateFlow()

    private val _session = MutableStateFlow<LiveAnalysisSession?>(null)
    val session: StateFlow<LiveAnalysisSession?> = _session.asStateFlow()

    private val observations = Channel<FaceObservation>(Channel.CONFLATED)
    private val manager = CalibrationManager(config)
    private var closedEye: ClosedEyeCalibration? = null

    init {
        viewModelScope.launch {
            val created = try {
                withContext(Dispatchers.Default) {
                    LiveAnalysisSession(context, config, stats) { frame -> observations.trySend(frame.observation) }
                }
            } catch (e: Throwable) {
                _ui.update { it.copy(step = CalibrationStep.ERROR, error = e.message ?: e.javaClass.simpleName) }
                return@launch
            }
            _session.value = created
            _ui.update { it.copy(step = CalibrationStep.COLLECTING) }
            for (obs in observations) handle(obs)
        }
    }

    private fun handle(obs: FaceObservation) {
        when (_ui.value.step) {
            CalibrationStep.COLLECTING -> when (val s = manager.add(obs)) {
                is CalibrationState.Collecting -> _ui.update { it.copy(progress = s.progress, issue = s.lastIssue) }
                is CalibrationState.Completed -> _ui.update { it.copy(step = CalibrationStep.DONE, progress = 1.0, baseline = s.baseline, issue = null) }
                is CalibrationState.Failed -> _ui.update { it.copy(step = CalibrationStep.FAILED, failure = s.reason, issue = s.dominantIssue) }
            }
            CalibrationStep.CLOSED_EYES -> {
                val c = closedEye ?: return
                val progress = c.add(obs)
                when (val r = c.result) {
                    null -> _ui.update { it.copy(progress = progress) }
                    is CalibrationState.Completed -> _ui.update { it.copy(step = CalibrationStep.DONE, baseline = r.baseline, closedEyeFailure = null) }
                    is CalibrationState.Failed -> _ui.update { it.copy(step = CalibrationStep.DONE, closedEyeFailure = r.reason) }
                    else -> Unit
                }
            }
            else -> Unit
        }
    }

    fun retry() {
        manager.reset()
        closedEye = null
        _ui.update { CalibrationUiState(step = CalibrationStep.COLLECTING, cameraError = it.cameraError) }
    }

    /** 선택적 눈 감김 보정. 정차 상태에서만 안내한다. */
    fun startClosedEyeCalibration() {
        val baseline = _ui.value.baseline ?: return
        closedEye = ClosedEyeCalibration(config, baseline)
        _ui.update { it.copy(step = CalibrationStep.CLOSED_EYES, progress = 0.0, closedEyeFailure = null) }
    }

    fun confirm() {
        val baseline = _ui.value.baseline ?: return
        calibrationStore.set(baseline)
        sendEvent(CalibrationEvent.StartDriving)
    }

    fun onCameraError(message: String) {
        _ui.update { it.copy(cameraError = message) }
    }

    override fun onCleared() {
        observations.close()
        _session.value?.close()
    }
}
