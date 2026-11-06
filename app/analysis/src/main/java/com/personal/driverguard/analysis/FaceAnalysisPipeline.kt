package com.personal.driverguard.analysis

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import com.personal.driverguard.analysis.landmark.FaceLandmarkExtractor
import com.personal.driverguard.analysis.landmark.LandmarkResult
import com.personal.driverguard.analysis.landmark.MediaPipeFaceLandmarkExtractor
import com.personal.driverguard.analysis.pose.OpenCvHeadPoseEstimator
import com.personal.driverguard.domain.analysis.geometry.Bounds
import com.personal.driverguard.domain.analysis.geometry.LandmarkPoints
import com.personal.driverguard.domain.analysis.quality.FrameMeasurement
import com.personal.driverguard.domain.analysis.quality.ObservationQualityEvaluator
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.FaceObservation
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/** 개발 화면·캘리브레이션 화면 오버레이용 좌표. 분석 이미지(비미러) 픽셀 기준이며 미러링은 화면에서만 한다. */
class FaceOverlay(
    val imageWidth: Int,
    val imageHeight: Int,
    /** 얼굴 하나일 때 478점 x0,y0,x1,y1,… 없으면 null */
    val points: FloatArray?,
    /** 재투영한 자세 축: 원점·x·y·z 끝점(8개). 자세 실패면 null */
    val axes: FloatArray?,
    val faceCount: Int,
)

class FrameTimings(
    /** 제출부터 MediaPipe 결과 도착까지 */
    val inferenceMs: Double,
    /** 결과 도착 후 기하 계산·solvePnP·품질 평가 */
    val postMs: Double,
)

/** 한 프레임의 분석 결과. [token] 은 제출 시점의 분석 세션 토큰이다. */
class AnalyzedFrame(
    val token: Long,
    val observation: FaceObservation,
    val overlay: FaceOverlay?,
    val timings: FrameTimings?,
    val error: String? = null,
)

/**
 * 카메라 프레임(회전 보정된 비미러 Bitmap) → MediaPipe 랜드마크 → 픽셀 좌표 → OpenCV 자세 → 품질 평가 → [FaceObservation].
 *
 * 버퍼 소유권: 호출자가 만든 Bitmap 을 [submit] 하면 결과 콜백이 끝날 때까지 파이프라인이 빌려 쓴다.
 * 한 번에 한 프레임만 받는다(바쁘면 false). 호출자는 [isBusy] 가 false 일 때만 같은 Bitmap 을 다시 채운다.
 * 결과 콜백은 MediaPipe 내부 스레드에서 온다. 닫힌 뒤 도착한 지연 콜백은 버린다.
 */
class FaceAnalysisPipeline(
    context: Context,
    config: AnalysisConfig,
    private val onFrame: (AnalyzedFrame) -> Unit,
    extractorFactory: (Context, (LandmarkResult) -> Unit, (RuntimeException) -> Unit) -> FaceLandmarkExtractor =
        { c, r, e -> MediaPipeFaceLandmarkExtractor(c, r, e) },
) : Closeable {

    private class Pending(val bitmap: Bitmap, val timestampMs: Long, val token: Long, val submittedNanos: Long)

    private val evaluator = ObservationQualityEvaluator(config)
    private val pose = OpenCvHeadPoseEstimator()
    private val busy = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)

    @Volatile
    private var pending: Pending? = null
    private var lastTimestampMs = Long.MIN_VALUE
    private val extractor: FaceLandmarkExtractor = extractorFactory(context.applicationContext, ::onLandmarks, ::onError)

    val isBusy: Boolean get() = busy.get()

    /**
     * 추론을 제출한다. 바쁘거나 닫혔으면 false(호출자가 프레임을 버린다).
     * 타임스탬프가 이전 이하면 1ms 올려 MediaPipe 의 단조 증가 조건을 지킨다.
     */
    fun submit(bitmap: Bitmap, timestampMs: Long, token: Long): Boolean {
        if (closed.get() || !busy.compareAndSet(false, true)) return false
        val ts = if (timestampMs <= lastTimestampMs) lastTimestampMs + 1 else timestampMs
        lastTimestampMs = ts
        pending = Pending(bitmap, ts, token, SystemClock.elapsedRealtimeNanos())
        return try {
            extractor.detectAsync(bitmap, ts)
            true
        } catch (e: RuntimeException) {
            pending = null
            busy.set(false)
            onFrame(AnalyzedFrame(token, FaceObservation.notFound(ts, flag = com.personal.driverguard.entity.QualityFlag.ANALYSIS_ERROR), null, null, e.message))
            false
        }
    }

    private fun onLandmarks(result: LandmarkResult) {
        val p = pending
        if (closed.get() || p == null || p.timestampMs != result.timestampMs) return
        val arrived = SystemClock.elapsedRealtimeNanos()
        try {
            val faceCount = result.faces.size
            val points = result.faces.singleOrNull()?.let { LandmarkPoints.fromNormalized(it, result.imageWidth, result.imageHeight) }
            val estimate = points?.takeIf { it.allFinite() }?.let(pose::estimate)
            val luma = points?.takeIf { it.allFinite() }?.let { faceLuma(p.bitmap, it.bounds()) }
            val observation = evaluator.evaluate(
                FrameMeasurement(
                    timestampMs = result.timestampMs,
                    faceCount = faceCount,
                    landmarks = points,
                    pose = estimate?.pose,
                    faceLuma = luma?.first,
                    faceLumaStdDev = luma?.second,
                ),
            )
            val overlay = FaceOverlay(result.imageWidth, result.imageHeight, result.faces.singleOrNull(), estimate?.axes, faceCount)
            val done = SystemClock.elapsedRealtimeNanos()
            release()
            onFrame(AnalyzedFrame(p.token, observation, overlay, FrameTimings((arrived - p.submittedNanos) / 1e6, (done - arrived) / 1e6)))
        } catch (e: Exception) {
            release()
            onFrame(AnalyzedFrame(p.token, FaceObservation.notFound(result.timestampMs, flag = com.personal.driverguard.entity.QualityFlag.ANALYSIS_ERROR), null, null, e.message))
        }
    }

    private fun onError(e: RuntimeException) {
        val p = pending ?: return
        release()
        if (!closed.get()) {
            onFrame(AnalyzedFrame(p.token, FaceObservation.notFound(p.timestampMs, flag = com.personal.driverguard.entity.QualityFlag.ANALYSIS_ERROR), null, null, e.message))
        }
    }

    private fun release() {
        pending = null
        busy.set(false)
    }

    /**
     * 얼굴 경계 안을 16×16 격자로 표본 추출한 밝기(BT.601 luma) 평균·표준편차.
     * 밝기만으로 선글라스·가림을 판별하지 않는다. 극단적인 저조도·저대비만 거른다.
     */
    private fun faceLuma(bitmap: Bitmap, b: Bounds): Pair<Double, Double>? {
        val left = b.left.toInt().coerceIn(0, bitmap.width - 1)
        val right = b.right.toInt().coerceIn(0, bitmap.width - 1)
        val top = b.top.toInt().coerceIn(0, bitmap.height - 1)
        val bottom = b.bottom.toInt().coerceIn(0, bitmap.height - 1)
        if (right <= left || bottom <= top) return null
        var sum = 0.0
        var sumSq = 0.0
        var n = 0
        for (iy in 0 until GRID) {
            val y = top + (bottom - top) * iy / (GRID - 1)
            for (ix in 0 until GRID) {
                val x = left + (right - left) * ix / (GRID - 1)
                val c = bitmap.getPixel(x, y)
                val l = 0.299 * Color.red(c) + 0.587 * Color.green(c) + 0.114 * Color.blue(c)
                sum += l
                sumSq += l * l
                n++
            }
        }
        val mean = sum / n
        return mean to sqrt((sumSq / n - mean * mean).coerceAtLeast(0.0))
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        // MediaPipe 를 먼저 닫아 콜백을 멈춘 뒤 OpenCV Mat 을 해제한다
        runCatching { extractor.close() }
        pose.close()
        release()
    }

    private companion object {
        const val GRID = 16
    }
}
