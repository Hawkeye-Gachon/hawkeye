package com.personal.driverguard.domain.analysis.quality

import com.personal.driverguard.domain.analysis.geometry.FaceLandmarkIndices
import com.personal.driverguard.domain.analysis.geometry.LandmarkGeometry
import com.personal.driverguard.domain.analysis.geometry.LandmarkPoints
import com.personal.driverguard.domain.analysis.geometry.PoseMath
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.CalibrationBaseline
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.HeadPose
import com.personal.driverguard.entity.QualityFlag
import kotlin.math.abs

/**
 * 추론 직후의 한 프레임 측정값. 분석 파이프라인(:analysis)이 MediaPipe·OpenCV 결과로 채운다.
 * [landmarks] 는 얼굴이 정확히 하나일 때만 채운다(여러 얼굴이면 운전자를 특정하지 않는다).
 */
data class FrameMeasurement(
    val timestampMs: Long,
    val faceCount: Int,
    val landmarks: LandmarkPoints?,
    /** OpenCV solvePnP 결과. 실패하면 null */
    val pose: HeadPose?,
    /** 얼굴 영역 평균 밝기·표준편차(0~255). 측정하지 못했으면 null */
    val faceLuma: Double?,
    val faceLumaStdDev: Double?,
    val error: Boolean = false,
)

/** 캘리브레이션 기준 대비로 다시 매긴 지표 품질과 상대 자세 */
data class RelativeView(
    val eyeQuality: QualityFlag,
    val mouthQuality: QualityFlag,
    val poseQuality: QualityFlag,
    val yawDeg: Double?,
    val pitchDeg: Double?,
    val rollDeg: Double?,
)

/**
 * 지표별 분석 가능 여부를 판단한다.
 * 1) [evaluate]: 얼굴 전체(미검출·복수·크기·잘림·좌표·조도)와 눈·입·자세 각각의 기하학적 유효성
 * 2) [relativeTo]: 캘리브레이션 기준 대비 회전이 커서 눈·입 비율이 왜곡되는 경우 그 지표만 무효
 *
 * 밝기만으로 선글라스 착용이나 눈 가림을 판별하지 않는다. 가림은 신뢰성 있게 구분할 수 없어
 * 랜드마크가 계속 추정되는 한 EAR 이 틀릴 수 있다(README 의 한계 참고).
 */
class ObservationQualityEvaluator(private val config: AnalysisConfig) {

    fun evaluate(m: FrameMeasurement): FaceObservation {
        val ts = m.timestampMs
        if (m.error) return FaceObservation.notFound(ts, m.faceCount, QualityFlag.ANALYSIS_ERROR)
        if (m.faceCount == 0) return FaceObservation.notFound(ts, 0)
        if (m.faceCount > 1) return FaceObservation.notFound(ts, m.faceCount, QualityFlag.MULTIPLE_FACES)
        val points = m.landmarks ?: return FaceObservation.notFound(ts, m.faceCount, QualityFlag.LANDMARKS_INVALID)
        if (points.size < FaceLandmarkIndices.LANDMARK_COUNT - 10 || !points.allFinite()) {
            return FaceObservation.notFound(ts, 1, QualityFlag.LANDMARKS_INVALID)
        }

        val bounds = points.bounds()
        val widthRatio = bounds.width / points.imageWidth
        val outside = LandmarkGeometry.outsideRatio(bounds, points.imageWidth, points.imageHeight)
        val faceQuality = when {
            widthRatio < config.minFaceWidthRatio || outside > config.maxFaceOutsideRatio -> QualityFlag.FACE_TOO_SMALL
            m.faceLuma != null && m.faceLumaStdDev != null &&
                (m.faceLuma < config.minFaceLuma || m.faceLumaStdDev < config.minFaceLumaStdDev) -> QualityFlag.LOW_LIGHT_OR_LOW_CONTRAST
            else -> QualityFlag.VALID
        }

        val right = LandmarkGeometry.eyeAspectRatio(points, FaceLandmarkIndices.RIGHT_EYE_EAR, config.minEyeWidthPx)
        val left = LandmarkGeometry.eyeAspectRatio(points, FaceLandmarkIndices.LEFT_EYE_EAR, config.minEyeWidthPx)
        val mouth = LandmarkGeometry.mouthOpenRatio(
            points,
            FaceLandmarkIndices.MOUTH_INNER_VERTICAL_PAIRS,
            FaceLandmarkIndices.MOUTH_RIGHT_CORNER,
            FaceLandmarkIndices.MOUTH_LEFT_CORNER,
            config.minMouthWidthPx,
        )
        val leftValid = left.value != null
        val rightValid = right.value != null
        val eyeQuality = when {
            !faceQuality.isValid -> faceQuality
            leftValid && rightValid -> QualityFlag.VALID
            else -> QualityFlag.LANDMARKS_INVALID
        }
        val mouthQuality = when {
            !faceQuality.isValid -> faceQuality
            mouth.value != null -> QualityFlag.VALID
            else -> QualityFlag.LANDMARKS_INVALID
        }
        val pose = m.pose?.takeIf { it.rotation.isFinite() && it.reprojectionErrorPx.isFinite() }
        val poseQuality = when {
            !faceQuality.isValid -> faceQuality
            pose == null -> QualityFlag.LANDMARKS_INVALID
            pose.reprojectionErrorPx / bounds.width > config.maxReprojectionErrorRatio -> QualityFlag.LANDMARKS_INVALID
            else -> QualityFlag.VALID
        }

        return FaceObservation(
            timestampMs = ts,
            faceCount = 1,
            faceQuality = faceQuality,
            leftEar = left.value,
            rightEar = right.value,
            eyeQuality = eyeQuality,
            leftEyeValid = leftValid,
            rightEyeValid = rightValid,
            mouthOpenRatio = mouth.value,
            mouthQuality = mouthQuality,
            pose = pose,
            poseQuality = poseQuality,
            faceWidthRatio = widthRatio,
            faceLuma = m.faceLuma,
            faceLumaStdDev = m.faceLumaStdDev,
        )
    }

    /** 기준 자세 대비 상대 회전을 구하고, 회전이 크면 눈·입 지표만 무효로 바꾼다. */
    fun relativeTo(obs: FaceObservation, baseline: CalibrationBaseline?): RelativeView {
        if (baseline == null) {
            val required = { q: QualityFlag -> if (q.isValid) QualityFlag.CALIBRATION_REQUIRED else q }
            return RelativeView(required(obs.eyeQuality), required(obs.mouthQuality), required(obs.poseQuality), null, null, null)
        }
        val pose = obs.pose?.takeIf { obs.poseQuality.isValid }
        val rel = pose?.let { PoseMath.euler(PoseMath.relative(baseline.poseRotation, it.rotation)) }
        // 자세를 모르면 회전 왜곡 여부를 확인할 수 없지만, 눈·입 자체는 기하학적으로 유효하므로 그대로 둔다
        val rotated = rel != null && (abs(rel.yawDeg) > config.metricMaxYawDeg || abs(rel.pitchDeg) > config.metricMaxPitchDeg)
        fun gate(q: QualityFlag) = if (q.isValid && rotated) QualityFlag.EXCESSIVE_HEAD_ROTATION else q
        return RelativeView(
            eyeQuality = gate(obs.eyeQuality),
            mouthQuality = gate(obs.mouthQuality),
            poseQuality = if (rel != null) QualityFlag.VALID else obs.poseQuality,
            yawDeg = rel?.yawDeg,
            pitchDeg = rel?.pitchDeg,
            rollDeg = rel?.rollDeg,
        )
    }
}
