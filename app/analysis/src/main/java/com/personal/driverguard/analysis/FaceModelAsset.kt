package com.personal.driverguard.analysis

import android.content.Context
import com.personal.driverguard.analysis.landmark.MediaPipeFaceLandmarkExtractor
import java.io.FileNotFoundException
import java.security.MessageDigest

sealed interface ModelAssetStatus {
    data class Ready(val fileName: String, val sizeBytes: Long, val sha256: String) : ModelAssetStatus {
        /** 여정에 남길 모델 식별 정보 */
        val info: String get() = "$fileName sha256:${sha256.take(12)}"
    }

    data object Missing : ModelAssetStatus

    data class Failed(val message: String) : ModelAssetStatus
}

/**
 * 앱에 포함된 Face Landmarker 모델 파일을 확인하고 SHA-256 을 계산한다.
 * 출처: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
 */
object FaceModelAsset {
    /** README 에 기록한 공식 배포본의 SHA-256 */
    const val EXPECTED_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

    fun check(context: Context): ModelAssetStatus = try {
        val digest = MessageDigest.getInstance("SHA-256")
        var size = 0L
        context.assets.open(MediaPipeFaceLandmarkExtractor.MODEL_ASSET_PATH).use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
                size += n
            }
        }
        val hex = digest.digest().joinToString("") { "%02x".format(it) }
        ModelAssetStatus.Ready(MediaPipeFaceLandmarkExtractor.MODEL_ASSET_PATH.substringAfterLast('/'), size, hex)
    } catch (_: FileNotFoundException) {
        ModelAssetStatus.Missing
    } catch (e: Exception) {
        ModelAssetStatus.Failed(e.message ?: e.javaClass.simpleName)
    }
}
