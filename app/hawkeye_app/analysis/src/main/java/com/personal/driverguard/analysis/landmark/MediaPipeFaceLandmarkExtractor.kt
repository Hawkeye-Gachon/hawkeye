package com.personal.driverguard.analysis.landmark

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.io.Closeable

/**
 * MediaPipe 추론 결과. [faces] 는 얼굴마다 정규화 좌표 x0,y0,x1,y1,… (0~1, 회전 보정된 분석 이미지 기준).
 * 정규화 z 와 visibility/presence 는 담지 않는다. Face Landmarker 는 랜드마크별 신뢰도를 채워 주지 않는다.
 */
class LandmarkResult(
    /** detectAsync 에 넘긴 타임스탬프(프레임을 받은 단조 시계 시각). 콜백 도착 시각이 아니다. */
    val timestampMs: Long,
    val faces: List<FloatArray>,
    val imageWidth: Int,
    val imageHeight: Int,
)

/** 얼굴 특징점 추출(FaceLandmarkExtractor). 구현을 바꿔 끼울 수 있게 인터페이스로 둔다. */
interface FaceLandmarkExtractor : Closeable {
    /**
     * 비동기 추론을 제출한다. 결과·오류는 생성 시 넘긴 콜백으로 온다.
     * [timestampMs] 는 이전 제출보다 커야 한다(MediaPipe LIVE_STREAM 요구사항).
     * 호출자는 콜백이 오기 전까지 [bitmap] 을 바꾸거나 재활용하면 안 된다.
     */
    fun detectAsync(bitmap: Bitmap, timestampMs: Long)
}

/**
 * MediaPipe Tasks Vision Face Landmarker, LIVE_STREAM 모드.
 * 모델은 앱 assets/models/face_landmarker.task (공식 배포본, Apache 2.0)에서 읽는다. 네트워크를 쓰지 않는다.
 *
 * - 인스턴스는 한 번 만들어 재사용한다(프레임마다 만들지 않는다).
 * - numFaces 를 2 로 둬 두 번째 얼굴이 보이는지 확인한다. 여러 얼굴이면 운전자를 고르지 않는다(품질 평가에서 MULTIPLE_FACES).
 * - 아래 신뢰도 값은 MediaPipe 내부 검출·추적 임계값이며, 결과에 랜드마크별 신뢰도로 나오지 않는다.
 * - 이전 결과를 처리 중이면 MediaPipe 가 새 프레임을 건너뛸 수 있다. 그래서 호출자는 한 번에 한 프레임만 제출한다.
 */
class MediaPipeFaceLandmarkExtractor(
    context: Context,
    private val onResult: (LandmarkResult) -> Unit,
    private val onError: (RuntimeException) -> Unit,
    modelAssetPath: String = MODEL_ASSET_PATH,
) : FaceLandmarkExtractor {

    private val landmarker: FaceLandmarker = FaceLandmarker.createFromOptions(
        context,
        FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelAssetPath).setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(MAX_FACES)
            .setMinFaceDetectionConfidence(MIN_DETECTION)
            .setMinFacePresenceConfidence(MIN_PRESENCE)
            .setMinTrackingConfidence(MIN_TRACKING)
            .setOutputFaceBlendshapes(false)
            .setOutputFacialTransformationMatrixes(false)
            .setResultListener { result, input -> onResult(result.toLandmarkResult(input.width, input.height)) }
            .setErrorListener { onError(it) }
            .build(),
    )

    override fun detectAsync(bitmap: Bitmap, timestampMs: Long) {
        landmarker.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
    }

    override fun close() {
        landmarker.close()
    }

    private fun FaceLandmarkerResult.toLandmarkResult(width: Int, height: Int): LandmarkResult {
        val faces = faceLandmarks().map { face ->
            FloatArray(face.size * 2).also { out ->
                face.forEachIndexed { i, lm ->
                    out[i * 2] = lm.x()
                    out[i * 2 + 1] = lm.y()
                }
            }
        }
        return LandmarkResult(timestampMs(), faces, width, height)
    }

    companion object {
        const val MODEL_ASSET_PATH = "models/face_landmarker.task"
        private const val MAX_FACES = 2
        private const val MIN_DETECTION = 0.5f
        private const val MIN_PRESENCE = 0.5f
        private const val MIN_TRACKING = 0.5f
    }
}
