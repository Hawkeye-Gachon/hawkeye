package com.personal.driverguard.runtime

import android.content.Context
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import com.personal.driverguard.analysis.AnalyzedFrame
import com.personal.driverguard.analysis.FaceAnalysisPipeline
import com.personal.driverguard.analysis.FaceOverlay
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.platform.android.camera.CameraFrameSource
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.Closeable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * 카메라 프레임 공급 + MediaPipe·OpenCV 파이프라인을 한 화면 수명 동안 묶는다.
 * 캘리브레이션·주행·개발 화면이 각자 하나씩 만들고 ViewModel 이 사라질 때 닫는다.
 *
 * 생성자가 모델·네이티브 라이브러리를 준비하지 못하면 예외를 던진다. 호출자는 이를 오류로 보여 주고 시연으로 바꾸지 않는다.
 */
class LiveAnalysisSession(
    context: Context,
    config: AnalysisConfig,
    private val stats: AnalysisStatsHolder,
    private val onFrame: (AnalyzedFrame) -> Unit,
) : Closeable {

    /** 분석 프레임 변환 전용 단일 스레드(메인 스레드 밖) */
    val executor: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "camera-frames") }

    /** 프레임을 제출할 때 붙일 분석 세션 토큰. 주행 화면이 바꾼다. */
    @Volatile
    var token: Long = 0L

    private val _overlay = MutableStateFlow<FaceOverlay?>(null)

    /** 최근 프레임의 오버레이 좌표(화면 미리보기 위에 그린다) */
    val overlay: StateFlow<FaceOverlay?> = _overlay.asStateFlow()

    private val pipeline = FaceAnalysisPipeline(context, config, onFrame = { frame ->
        stats.record(frame, SystemClock.elapsedRealtime())
        _overlay.value = frame.overlay
        onFrame(frame)
    })

    val analyzer: ImageAnalysis.Analyzer = CameraFrameSource(pipeline, config.targetAnalysisFps, { token }, stats)

    init {
        stats.reset(SystemClock.elapsedRealtime())
    }

    /** 진행 중인 프레임 변환이 끝난 뒤 같은 스레드에서 파이프라인을 닫는다. 늦게 도착한 콜백은 파이프라인이 버린다. */
    override fun close() {
        _overlay.value = null
        executor.execute { pipeline.close() }
        executor.shutdown()
        runCatching { executor.awaitTermination(2, TimeUnit.SECONDS) }
    }
}
