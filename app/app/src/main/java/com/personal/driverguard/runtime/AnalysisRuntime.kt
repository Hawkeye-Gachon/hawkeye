package com.personal.driverguard.runtime

import android.content.Context
import com.personal.driverguard.analysis.AnalyzedFrame
import com.personal.driverguard.analysis.FaceModelAsset
import com.personal.driverguard.analysis.ModelAssetStatus
import com.personal.driverguard.entity.CalibrationBaseline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** 모델 파일 확인 결과(null 이면 확인 중). 한 번 확인하면 앱이 끝날 때까지 재사용한다. */
class ModelAssetHolder(private val context: Context) {
    private val mutex = Mutex()
    private val _status = MutableStateFlow<ModelAssetStatus?>(null)
    val status: StateFlow<ModelAssetStatus?> = _status.asStateFlow()

    suspend fun ensureChecked(): ModelAssetStatus = mutex.withLock {
        _status.value ?: withContext(Dispatchers.IO) { FaceModelAsset.check(context) }.also { _status.value = it }
    }
}

/**
 * 이번 주행에 쓸 캘리브레이션 기준값. 세션별 측정이 기본이라 저장하지 않고 메모리에만 둔다.
 * 실제 분석 주행이 끝나면 비워서 다음 주행에서 카메라 위치가 바뀌었을 수 있는 기준값을 재사용하지 않는다.
 */
class CalibrationStore {
    private val _baseline = MutableStateFlow<CalibrationBaseline?>(null)
    val baseline: StateFlow<CalibrationBaseline?> = _baseline.asStateFlow()

    fun set(baseline: CalibrationBaseline) {
        _baseline.value = baseline
    }

    fun clear() {
        _baseline.value = null
    }
}

data class AnalysisStats(
    /** 카메라에서 받은 프레임 */
    val cameraFrames: Long = 0,
    /** 분석 제출 빈도 제한으로 건너뛴 프레임 */
    val throttled: Long = 0,
    /** 이전 추론이 끝나지 않아 버린 프레임 */
    val droppedBusy: Long = 0,
    /** 결과를 받은 프레임 */
    val analyzed: Long = 0,
    val errors: Long = 0,
    val lastInferenceMs: Double? = null,
    val avgInferenceMs: Double? = null,
    val lastPostMs: Double? = null,
    /** 결과 도착 간격으로 계산한 처리량(지수 이동 평균) */
    val throughputFps: Double? = null,
    val lastError: String? = null,
    val startedAtMs: Long? = null,
)

/** 실제 분석에서 측정한 처리량·지연. 개발 화면에 보여 준다. */
class AnalysisStatsHolder {
    private val _stats = MutableStateFlow(AnalysisStats())
    val stats: StateFlow<AnalysisStats> = _stats.asStateFlow()
    private var lastResultMs: Long? = null

    fun reset(nowMs: Long) {
        lastResultMs = null
        _stats.value = AnalysisStats(startedAtMs = nowMs)
    }

    fun cameraFrame() = _stats.update { it.copy(cameraFrames = it.cameraFrames + 1) }

    fun throttled() = _stats.update { it.copy(throttled = it.throttled + 1) }

    fun droppedBusy() = _stats.update { it.copy(droppedBusy = it.droppedBusy + 1) }

    fun conversionError(message: String?) = _stats.update { it.copy(errors = it.errors + 1, lastError = message) }

    fun record(frame: AnalyzedFrame, nowMs: Long) {
        val previous = lastResultMs
        lastResultMs = nowMs
        _stats.update { s ->
            val t = frame.timings
            val instantFps = previous?.let { p -> (nowMs - p).takeIf { it > 0 }?.let { 1000.0 / it } }
            s.copy(
                analyzed = s.analyzed + 1,
                errors = s.errors + if (frame.error != null) 1 else 0,
                lastError = frame.error ?: s.lastError,
                lastInferenceMs = t?.inferenceMs ?: s.lastInferenceMs,
                avgInferenceMs = t?.inferenceMs?.let { v -> s.avgInferenceMs?.let { it * 0.9 + v * 0.1 } ?: v } ?: s.avgInferenceMs,
                lastPostMs = t?.postMs ?: s.lastPostMs,
                throughputFps = instantFps?.let { v -> s.throughputFps?.let { it * 0.9 + v * 0.1 } ?: v } ?: s.throughputFps,
            )
        }
    }
}
