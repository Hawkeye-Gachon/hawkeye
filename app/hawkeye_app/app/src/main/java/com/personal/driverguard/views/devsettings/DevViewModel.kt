package com.personal.driverguard.views.devsettings

import android.content.Context
import android.os.SystemClock
import androidx.lifecycle.viewModelScope
import com.personal.driverguard.analysis.ModelAssetStatus
import com.personal.driverguard.domain.analysis.calibration.CalibrationManager
import com.personal.driverguard.domain.analysis.calibration.CalibrationState
import com.personal.driverguard.domain.engine.DriverStateEngine
import com.personal.driverguard.domain.engine.EngineSnapshot
import com.personal.driverguard.domain.repository.ISettingsRepository
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.runtime.AnalysisStats
import com.personal.driverguard.runtime.AnalysisStatsHolder
import com.personal.driverguard.runtime.CalibrationStore
import com.personal.driverguard.runtime.LiveAnalysisSession
import com.personal.driverguard.runtime.MetricLog
import com.personal.driverguard.runtime.ModelAssetHolder
import com.personal.driverguard.views.base.BaseViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface DevEvent {
    data object Saved : DevEvent
    data object Reset : DevEvent
    data class LogStopped(val path: String) : DevEvent
    data class LogsDeleted(val count: Int) : DevEvent
}

data class LiveState(
    val starting: Boolean = false,
    val error: String? = null,
    val snapshot: EngineSnapshot? = null,
    val display: DisplayState = DisplayState.PREPARING,
    val calibrating: Boolean = false,
    val calibrationMessage: String? = null,
    val hasBaseline: Boolean = false,
    val logging: Boolean = false,
    val logFiles: Int = 0,
)

/**
 * 개발 화면. 정차 상태 전용이며 기록을 남기지 않는 실시간 분석(카메라·랜드마크·지표), 임계값 편집,
 * 모델·처리량 확인, 명시적으로 시작하는 수치 로그를 제공한다.
 */
class DevViewModel(
    private val context: Context,
    private val settingsRepository: ISettingsRepository,
    modelAssets: ModelAssetHolder,
    private val calibrationStore: CalibrationStore,
    private val statsHolder: AnalysisStatsHolder,
    private val metricLog: MetricLog,
) : BaseViewModel<DevEvent>() {

    val modelStatus: StateFlow<ModelAssetStatus?> = modelAssets.status
    val stats: StateFlow<AnalysisStats> = statsHolder.stats
    val config: StateFlow<AnalysisConfig> = settingsRepository.config

    private val _draft = MutableStateFlow(toDraft(settingsRepository.config.value))
    val draft: StateFlow<Map<String, String>> = _draft.asStateFlow()
    private val _invalidFields = MutableStateFlow<Set<String>>(emptySet())
    val invalidFields: StateFlow<Set<String>> = _invalidFields.asStateFlow()
    private val _formErrors = MutableStateFlow<List<String>>(emptyList())
    val formErrors: StateFlow<List<String>> = _formErrors.asStateFlow()

    private val _live = MutableStateFlow(LiveState(hasBaseline = calibrationStore.baseline.value != null, logFiles = metricLog.files().size))
    val live: StateFlow<LiveState> = _live.asStateFlow()
    private val _session = MutableStateFlow<LiveAnalysisSession?>(null)
    val session: StateFlow<LiveAnalysisSession?> = _session.asStateFlow()

    private val observations = Channel<FaceObservation>(Channel.UNLIMITED)
    private var engine: DriverStateEngine? = null
    private var calibration: CalibrationManager? = null
    private var lastPublishMs = 0L

    init {
        viewModelScope.launch { modelAssets.ensureChecked() }
    }

    /** 실시간 탭을 처음 열 때 분석기를 만든다(카메라 권한이 있을 때만 화면이 부른다) */
    fun startLive() {
        if (_session.value != null || _live.value.starting) return
        _live.update { it.copy(starting = true, error = null) }
        viewModelScope.launch(Dispatchers.Default) {
            val cfg = settingsRepository.config.value
            val created = try {
                LiveAnalysisSession(context, cfg, statsHolder) { observations.trySend(it.observation) }
            } catch (e: Throwable) {
                _live.update { it.copy(starting = false, error = e.message ?: e.javaClass.simpleName) }
                return@launch
            }
            engine = DriverStateEngine(cfg, calibrationStore.baseline.value, SystemClock.elapsedRealtime())
            _session.value = created
            _live.update { it.copy(starting = false) }
            for (obs in observations) consume(obs, cfg)
        }
    }

    private suspend fun consume(obs: FaceObservation, cfg: AnalysisConfig) {
        calibration?.let { manager ->
            when (val s = manager.add(obs)) {
                is CalibrationState.Completed -> {
                    calibrationStore.set(s.baseline)
                    calibration = null
                    engine = DriverStateEngine(cfg, s.baseline, obs.timestampMs)
                    _live.update { it.copy(calibrating = false, hasBaseline = true, calibrationMessage = s.baseline.summary()) }
                }
                is CalibrationState.Failed -> {
                    calibration = null
                    _live.update { it.copy(calibrating = false, calibrationMessage = "실패: ${s.reason}" + (s.dominantIssue?.let { f -> " ($f)" } ?: "")) }
                }
                is CalibrationState.Collecting -> _live.update { it.copy(calibrationMessage = "측정 중 ${(s.progress * 100).toInt()}%" + (s.lastIssue?.let { f -> " · $f" } ?: "")) }
            }
        }
        val output = engine?.onObservation(obs) ?: return
        val snapshot = output.snapshot ?: return
        withContext(Dispatchers.IO) { metricLog.append(snapshot) }
        if (obs.timestampMs - lastPublishMs >= cfg.uiRefreshMs) {
            lastPublishMs = obs.timestampMs
            _live.update { it.copy(snapshot = snapshot, display = output.display) }
        }
    }

    /** 기록 없이 기준값만 다시 측정한다(개발용). 결과는 다음 실제 주행에도 쓰인다. */
    fun recalibrate() {
        calibration = CalibrationManager(settingsRepository.config.value)
        _live.update { it.copy(calibrating = true, calibrationMessage = null) }
    }

    fun toggleLog() {
        viewModelScope.launch(Dispatchers.IO) {
            if (metricLog.isRecording) {
                val path = metricLog.currentFile?.absolutePath.orEmpty()
                metricLog.stop()
                _live.update { it.copy(logging = false, logFiles = metricLog.files().size) }
                sendEvent(DevEvent.LogStopped(path))
            } else {
                metricLog.start()
                _live.update { it.copy(logging = true, logFiles = metricLog.files().size) }
            }
        }
    }

    fun deleteLogs() {
        viewModelScope.launch(Dispatchers.IO) {
            val count = metricLog.deleteAll()
            _live.update { it.copy(logging = false, logFiles = 0) }
            sendEvent(DevEvent.LogsDeleted(count))
        }
    }

    fun onFieldChange(key: String, value: String) {
        _draft.update { it + (key to value) }
        _invalidFields.update { it - key }
    }

    fun save() {
        var cfg = settingsRepository.config.value
        val invalid = mutableSetOf<String>()
        for (field in allConfigFields) {
            val updated = field.write(cfg, _draft.value[field.key].orEmpty())
            if (updated == null) invalid += field.key else cfg = updated
        }
        _invalidFields.value = invalid
        if (invalid.isNotEmpty()) {
            _formErrors.value = emptyList()
            return
        }
        val errors = settingsRepository.update(cfg)
        _formErrors.value = errors
        if (errors.isEmpty()) sendEvent(DevEvent.Saved)
    }

    fun resetToDefaults() {
        settingsRepository.resetToDefaults()
        _draft.value = toDraft(settingsRepository.config.value)
        _invalidFields.value = emptySet()
        _formErrors.value = emptyList()
        sendEvent(DevEvent.Reset)
    }

    fun onCameraError(message: String) {
        _live.update { it.copy(error = message) }
    }

    override fun onCleared() {
        observations.close()
        metricLog.stop()
        _session.value?.close()
    }

    private fun toDraft(config: AnalysisConfig) = allConfigFields.associate { it.key to it.read(config) }
}
