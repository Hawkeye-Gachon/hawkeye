package com.personal.driverguard.views.driving

import android.content.Context
import androidx.lifecycle.viewModelScope
import com.personal.driverguard.analysis.ModelAssetStatus
import com.personal.driverguard.domain.platform.IAlertOutput
import com.personal.driverguard.domain.platform.ILocationSource
import com.personal.driverguard.domain.platform.IRestAreaGuide
import com.personal.driverguard.domain.platform.TimeSource
import com.personal.driverguard.domain.repository.ISettingsRepository
import com.personal.driverguard.domain.repository.ITripRepository
import com.personal.driverguard.domain.session.SessionUpdate
import com.personal.driverguard.domain.session.TripRecorder
import com.personal.driverguard.entity.AnalysisMode
import com.personal.driverguard.entity.CautionCause
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.DrowsinessLevel
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.InterruptionReason
import com.personal.driverguard.entity.LocationFix
import com.personal.driverguard.entity.QualityFlag
import com.personal.driverguard.runtime.AnalysisStatsHolder
import com.personal.driverguard.runtime.CalibrationStore
import com.personal.driverguard.runtime.LiveAnalysisSession
import com.personal.driverguard.runtime.ModelAssetHolder
import com.personal.driverguard.views.base.BaseViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

enum class DrivingPhase { STARTING, RUNNING, FINISHING, FAILED }

data class DrivingUiState(
    val phase: DrivingPhase = DrivingPhase.STARTING,
    val display: DisplayState = DisplayState.PREPARING,
    val cautionCause: CautionCause? = null,
    val drowsiness: DrowsinessLevel = DrowsinessLevel.NONE,
    val distraction: Boolean = false,
    val unavailableReason: QualityFlag? = null,
    val isResting: Boolean = false,
    /** elapsedRealtime 기준 주행 시작 시각 */
    val startedAtMs: Long? = null,
    val startError: String? = null,
    val cameraError: String? = null,
    val locationEnabled: Boolean = false,
    /** 기본 주행 화면에서는 카메라 영상을 숨긴다. 사용자가 켤 때만 보인다. */
    val showCamera: Boolean = false,
)

sealed interface DrivingEvent {
    data class Finished(val tripId: Long) : DrivingEvent

    data object Exit : DrivingEvent
}

/**
 * 한 번의 주행 세션. 카메라+MediaPipe/OpenCV 관측과 시간·위치·휴식 입력을 하나의 채널로 모아
 * 한 코루틴에서 순서대로 [TripRecorder] 에 넘긴다.
 * 분석은 캘리브레이션 기준값과 모델이 있어야 시작한다.
 */
class DrivingViewModel(
    private val context: Context,
    private val tripRepository: ITripRepository,
    private val settingsRepository: ISettingsRepository,
    private val modelAssets: ModelAssetHolder,
    private val calibrationStore: CalibrationStore,
    private val time: TimeSource,
    private val locationSource: ILocationSource,
    private val alertOutput: IAlertOutput,
    private val restAreaGuide: IRestAreaGuide,
    private val stats: AnalysisStatsHolder,
    private val appScope: CoroutineScope,
) : BaseViewModel<DrivingEvent>() {

    private val config = settingsRepository.config.value

    private val _ui = MutableStateFlow(DrivingUiState())
    val ui: StateFlow<DrivingUiState> = _ui.asStateFlow()

    val restAreaGuideAvailable: Boolean = restAreaGuide.isAvailable()
    val voiceAvailable: Boolean get() = alertOutput.voiceAvailable

    private val _session = MutableStateFlow<LiveAnalysisSession?>(null)
    val session: StateFlow<LiveAnalysisSession?> = _session.asStateFlow()

    private sealed interface Input {
        data class Frame(val token: Long, val observation: FaceObservation) : Input
        data object Tick : Input
        data class Location(val fix: LocationFix) : Input
        data object StartRest : Input
        data object EndRest : Input
        data class Visible(val visible: Boolean) : Input
        data class CameraFailed(val message: String) : Input
        data object Finish : Input
    }

    private val inputs = Channel<Input>(Channel.UNLIMITED)
    private val recorderLock = Mutex()
    private var recorder: TripRecorder? = null

    init {
        viewModelScope.launch(Dispatchers.Default) { runSession() }
    }

    fun toggleRest() {
        inputs.trySend(if (_ui.value.isResting) Input.EndRest else Input.StartRest)
    }

    fun requestFinish() {
        if (recorder == null) sendEvent(DrivingEvent.Exit) else inputs.trySend(Input.Finish)
    }

    fun openRestAreaSearch() {
        restAreaGuide.openRestAreaSearch()
    }

    /** 화면이 보이는지(ON_START/ON_STOP). 백그라운드 분석은 하지 않고 끊긴 구간을 기록한다. */
    fun onVisibilityChanged(visible: Boolean) {
        inputs.trySend(Input.Visible(visible))
    }

    fun onCameraError(message: String) {
        _ui.update { it.copy(cameraError = message) }
        inputs.trySend(Input.CameraFailed(message))
    }

    /** 운전자 카메라 화면 보이기/숨기기. 분석은 화면 표시와 상관없이 계속된다. */
    fun toggleCamera() {
        _ui.update { it.copy(showCamera = !it.showCamera) }
    }

    private suspend fun runSession() {
        val prepared = try {
            prepare()
        } catch (e: Exception) {
            _ui.update { it.copy(phase = DrivingPhase.FAILED, startError = e.message ?: e.javaClass.simpleName) }
            return
        }
        val recorder = prepared
        val token = recorderLock.withLock {
            recorder.start().also { this.recorder = recorder }
        }
        applyToken(token)
        _ui.update {
            it.copy(phase = DrivingPhase.RUNNING, startedAtMs = time.monotonicMs(), locationEnabled = locationSource.hasPermission())
        }

        coroutineScope {
            launch {
                while (isActive) {
                    delay(TICK_MS)
                    inputs.send(Input.Tick)
                }
            }
            launch {
                locationSource.locationUpdates().catch { }.collect { inputs.send(Input.Location(it)) }
            }
            for (input in inputs) {
                val finished = recorderLock.withLock { handle(recorder, input) }
                if (finished) break
            }
            coroutineContext.cancelChildren()
        }
    }

    /** 카메라 분석을 준비하고 기록기를 만든다. 필요한 것이 없으면 예외로 알린다. */
    private suspend fun prepare(): TripRecorder {
        val baseline = calibrationStore.baseline.value ?: error("캘리브레이션 기준값이 없습니다. 정차 상태에서 카메라 위치 확인부터 해 주세요.")
        val model = modelAssets.ensureChecked() as? ModelAssetStatus.Ready ?: error("얼굴 특징점 모델 파일을 찾을 수 없습니다.")
        val session = withContext(Dispatchers.Default) {
            LiveAnalysisSession(context, config, stats) { frame -> inputs.trySend(Input.Frame(frame.token, frame.observation)) }
        }
        _session.value = session
        return TripRecorder(tripRepository, time, config, AnalysisMode.REAL, baseline, model.info)
    }

    private fun applyToken(token: Long?) {
        token ?: return
        _session.value?.token = token
    }

    /** 입력 하나를 처리한다. 여정이 끝났으면 true. */
    private suspend fun handle(recorder: TripRecorder, input: Input): Boolean {
        when (input) {
            is Input.Frame -> recorder.onObservation(input.token, input.observation)?.let(::apply)
            Input.Tick -> recorder.onTick()?.let(::apply)
            is Input.Location -> recorder.onLocation(input.fix)
            Input.StartRest -> {
                alertOutput.cancelAll()
                recorder.startRest()
                _ui.update { it.copy(isResting = true, display = DisplayState.PAUSED) }
            }
            Input.EndRest -> {
                applyToken(recorder.endRest())
                _ui.update { it.copy(isResting = false, display = DisplayState.PREPARING) }
            }
            is Input.Visible -> if (input.visible) {
                applyToken(recorder.resumeAfterInterruption())
            } else {
                alertOutput.cancelAll()
                recorder.interrupt(InterruptionReason.APP_BACKGROUND)
            }
            is Input.CameraFailed -> recorder.interrupt(InterruptionReason.CAMERA_ERROR)
            Input.Finish -> {
                _ui.update { it.copy(phase = DrivingPhase.FINISHING) }
                alertOutput.cancelAll()
                val id = recorder.finish()
                calibrationStore.clear()
                sendEvent(DrivingEvent.Finished(id))
                return true
            }
        }
        return false
    }

    private fun apply(update: SessionUpdate) {
        _ui.update {
            it.copy(
                display = update.display,
                cautionCause = update.cautionCause,
                drowsiness = update.drowsiness,
                distraction = update.distraction,
                unavailableReason = update.unavailableReason,
                isResting = update.isResting,
            )
        }
        update.alert?.let(alertOutput::play)
    }

    override fun onCleared() {
        val session = _session.value
        _session.value = null
        alertOutput.release()
        appScope.launch(Dispatchers.Default) {
            session?.close()
            // 사용자가 '주행 종료'를 누르지 않고 화면이 사라진 경우에도 여정을 닫는다
            recorderLock.withLock {
                recorder?.takeIf { !it.isFinished }?.finish()
            }
            calibrationStore.clear()
        }
    }

    private companion object {
        const val TICK_MS = 250L

    }
}
