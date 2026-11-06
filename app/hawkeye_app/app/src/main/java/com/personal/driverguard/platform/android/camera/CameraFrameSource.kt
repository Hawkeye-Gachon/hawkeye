package com.personal.driverguard.platform.android.camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.personal.driverguard.analysis.FaceAnalysisPipeline
import com.personal.driverguard.runtime.AnalysisStatsHolder
import java.nio.ByteBuffer

/**
 * CameraX 분석 프레임 공급(CameraFrameSource). 분석 전용 단일 스레드 executor 에서 호출된다.
 *
 * - 백프레셔: ImageAnalysis 는 STRATEGY_KEEP_ONLY_LATEST 로 적체를 막고, 여기서는 [targetFps] 를 넘는 프레임과
 *   이전 추론이 끝나지 않았을 때의 프레임을 버린다. 카메라 FPS·분석 제출 빈도·화면 갱신 빈도는 각각 따로다.
 * - 버퍼 소유권: ImageProxy 의 RGBA 버퍼를 재사용하는 앱 소유 Bitmap 으로 '복사'한 뒤 finally 에서 바로 닫는다.
 *   MediaPipe 는 복사본을 쓰므로 추론이 끝나기 전에 원본 버퍼를 닫아도 안전하다. 복사본은 추론이 끝나기 전에는 다시 채우지 않는다.
 * - rowStride·pixelStride 를 고려해 행 단위로 복사한다(행 끝에 여분 바이트가 있을 수 있다).
 * - 회전: rotationDegrees 로 정방향을 만든다. 미러링은 하지 않는다(모델 입력은 비미러, 거울상은 화면 표시에서만).
 * - 타임스탬프: 프레임을 받은 시점의 elapsedRealtime. 센서 타임스탬프는 기기마다 시간 기준이 달라 쓰지 않는다.
 */
class CameraFrameSource(
    private val pipeline: FaceAnalysisPipeline,
    targetFps: Int,
    private val token: () -> Long,
    private val stats: AnalysisStatsHolder,
) : ImageAnalysis.Analyzer {

    private val minIntervalMs = 1_000L / targetFps.coerceAtLeast(1)
    private var lastSubmittedMs = Long.MIN_VALUE / 2

    // 재사용 버퍼(크기·회전이 바뀔 때만 새로 만든다)
    private var packed: ByteBuffer? = null
    private var source: Bitmap? = null
    private var upright: Bitmap? = null
    private val canvasMatrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    override fun analyze(image: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        stats.cameraFrame()
        try {
            if (now - lastSubmittedMs < minIntervalMs) {
                stats.throttled()
                return
            }
            if (pipeline.isBusy) {
                stats.droppedBusy()
                return
            }
            val frame = toUpright(image)
            lastSubmittedMs = now
            if (!pipeline.submit(frame, now, token())) stats.droppedBusy()
        } catch (e: Exception) {
            stats.conversionError("프레임 변환 실패: ${e.message}")
        } finally {
            image.close()
        }
    }

    private fun toUpright(image: ImageProxy): Bitmap {
        val plane = image.planes[0]
        val width = image.width
        val height = image.height
        val rowBytes = width * 4
        val src = source?.takeIf { it.width == width && it.height == height }
            ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { source = it }

        val buffer = plane.buffer.apply { rewind() }
        if (plane.pixelStride == 4 && plane.rowStride == rowBytes) {
            src.copyPixelsFromBuffer(buffer)
        } else {
            val out = packed?.takeIf { it.capacity() == rowBytes * height } ?: ByteBuffer.allocateDirect(rowBytes * height).also { packed = it }
            out.clear()
            val row = ByteArray(rowBytes)
            for (y in 0 until height) {
                buffer.position(y * plane.rowStride)
                if (plane.pixelStride == 4) {
                    buffer.get(row, 0, rowBytes)
                } else {
                    for (x in 0 until width) buffer.position(y * plane.rowStride + x * plane.pixelStride).let { buffer.get(row, x * 4, 4) }
                }
                out.put(row)
            }
            out.rewind()
            src.copyPixelsFromBuffer(out)
        }

        val rotation = image.imageInfo.rotationDegrees
        if (rotation % 360 == 0) return src
        val swap = rotation % 180 != 0
        val outW = if (swap) height else width
        val outH = if (swap) width else height
        val dst = upright?.takeIf { it.width == outW && it.height == outH }
            ?: Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888).also { upright = it }
        canvasMatrix.reset()
        canvasMatrix.postTranslate(-width / 2f, -height / 2f)
        canvasMatrix.postRotate(rotation.toFloat())
        canvasMatrix.postTranslate(outW / 2f, outH / 2f)
        Canvas(dst).drawBitmap(src, canvasMatrix, paint)
        return dst
    }
}
