package com.personal.driverguard.entity

import kotlin.math.abs

/**
 * 관측 품질. VALID 가 아니면 해당 지표를 판정에 쓰지 않는다.
 * 얼굴 전체와 눈·입·머리 자세의 품질은 따로 매긴다(예: 자세는 추정되지만 눈 비율은 왜곡된 경우 눈만 무효).
 * MediaPipe Face Landmarker 는 랜드마크별 신뢰도를 주지 않으므로 여기 값은 모두 기하학적 보조 조건으로 판단한 것이다.
 */
enum class QualityFlag {
    VALID,
    FACE_NOT_FOUND,

    /** 얼굴이 둘 이상 보여 운전자를 특정할 수 없음 */
    MULTIPLE_FACES,

    /** 얼굴이 너무 작거나 이미지 밖으로 잘림 */
    FACE_TOO_SMALL,

    /** 좌표가 유한하지 않거나 분모(눈·입 너비)가 너무 작음, solvePnP 실패·재투영 오차 과다 */
    LANDMARKS_INVALID,

    /** 기준 자세 대비 회전이 커서 해당 지표를 계산할 수 없음 */
    EXCESSIVE_HEAD_ROTATION,
    LOW_LIGHT_OR_LOW_CONTRAST,

    /** 프레임 사이 간격이 연속 관측 기준을 넘음 */
    FRAME_GAP,

    /** 캘리브레이션 기준값이 없어 상대 지표를 계산할 수 없음 */
    CALIBRATION_REQUIRED,

    /** 프레임 변환·추론 중 오류 */
    ANALYSIS_ERROR,
    ;

    val isValid: Boolean get() = this == VALID
}

/** 3x3 행렬(행 우선). 회전 행렬 표현에 쓴다. */
class Mat3(val m: DoubleArray) {
    init {
        require(m.size == 9) { "Mat3 는 9개 원소가 필요합니다" }
    }

    operator fun get(r: Int, c: Int): Double = m[r * 3 + c]

    operator fun times(o: Mat3): Mat3 = Mat3(
        DoubleArray(9) { i ->
            val r = i / 3
            val c = i % 3
            this[r, 0] * o[0, c] + this[r, 1] * o[1, c] + this[r, 2] * o[2, c]
        },
    )

    fun transpose(): Mat3 = Mat3(DoubleArray(9) { i -> this[i % 3, i / 3] })

    fun isFinite(): Boolean = m.all { it.isFinite() }

    fun approxEquals(o: Mat3, eps: Double = 1e-9): Boolean = m.indices.all { abs(m[it] - o.m[it]) <= eps }

    override fun equals(other: Any?): Boolean = other is Mat3 && m.contentEquals(other.m)

    override fun hashCode(): Int = m.contentHashCode()

    override fun toString(): String = m.joinToString(prefix = "Mat3[", postfix = "]") { "%.4f".format(it) }

    companion object {
        val IDENTITY = Mat3(doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    }
}

/**
 * 머리 자세. [rotation] 은 머리 좌표계(운전자 기준 x=왼쪽, y=위, z=얼굴 앞쪽)를 '카메라를 정면으로 보는 자세' 기준으로 표현한 회전이다.
 * 각도 규약(도, Y-X-Z 내재 회전 순서, R = Ry(yaw)·Rx(pitch)·Rz(roll)):
 *  - yaw   + : 운전자 기준 왼쪽으로 고개를 돌림
 *  - pitch + : 고개를 숙임(턱이 아래로)
 *  - roll  + : 운전자 기준 오른쪽 어깨 쪽으로 기울임
 * 근사 카메라 내부 파라미터로 계산하므로 절대 각도는 정확하지 않다. 판정에는 캘리브레이션 기준 대비 상대 회전을 쓴다.
 */
data class HeadPose(
    val rotation: Mat3,
    val yawDeg: Double,
    val pitchDeg: Double,
    val rollDeg: Double,
    /** 평균 재투영 오차(픽셀) */
    val reprojectionErrorPx: Double,
)

/**
 * 한 프레임의 얼굴 관측. 분석 파이프라인이 만들고, 시간 누적 엔진이 소비한다. 순수 수치만 담고 이미지·랜드마크 전체는 담지 않는다.
 * [timestampMs] 는 프레임을 받은 시점의 단조 시계(elapsedRealtime) 값이다. 추론 결과가 도착한 시각이 아니다.
 */
data class FaceObservation(
    val timestampMs: Long,
    val faceCount: Int,
    /** 얼굴 전체 품질(미검출·복수 얼굴·작음·좌표 이상·조도) */
    val faceQuality: QualityFlag,
    val leftEar: Double? = null,
    val rightEar: Double? = null,
    /** 양쪽 눈 모두 유효할 때만 VALID. 한쪽만 유효하면 보수적으로 무효 처리한다. */
    val eyeQuality: QualityFlag = faceQuality,
    val leftEyeValid: Boolean = false,
    val rightEyeValid: Boolean = false,
    val mouthOpenRatio: Double? = null,
    val mouthQuality: QualityFlag = faceQuality,
    val pose: HeadPose? = null,
    val poseQuality: QualityFlag = faceQuality,
    /** 얼굴 경계 너비 / 이미지 너비 */
    val faceWidthRatio: Double? = null,
    val faceLuma: Double? = null,
    val faceLumaStdDev: Double? = null,
) {
    companion object {
        fun notFound(timestampMs: Long, faceCount: Int = 0, flag: QualityFlag = QualityFlag.FACE_NOT_FOUND) =
            FaceObservation(timestampMs, faceCount, flag)
    }
}

/** 캘리브레이션 기준값. 세션마다 새로 측정하는 것이 기본이다. */
data class CalibrationBaseline(
    val leftEarOpen: Double,
    val rightEarOpen: Double,
    /** 선택적인 눈 감김 캘리브레이션 결과. 없으면 눈 뜬 기준만으로 상대 임계값을 만든다(충분히 보정된 것이 아님). */
    val leftEarClosed: Double? = null,
    val rightEarClosed: Double? = null,
    val mouthClosedRatio: Double,
    /** 기준 자세 회전(중앙값 자세에 가장 가까운 샘플) */
    val poseRotation: Mat3,
    val yawDeg: Double,
    val pitchDeg: Double,
    val rollDeg: Double,
    val faceWidthRatio: Double,
    val sampleCount: Int,
    /** 유효 샘플 EAR 사분위 범위(평균 두 눈) */
    val earIqr: Double,
    val yawIqrDeg: Double,
    val pitchIqrDeg: Double,
) {
    val hasClosedEyeCalibration: Boolean get() = leftEarClosed != null && rightEarClosed != null

    fun summary(): String = buildString {
        append("EAR L%.3f R%.3f".format(leftEarOpen, rightEarOpen))
        if (hasClosedEyeCalibration) append(" (감음 L%.3f R%.3f)".format(leftEarClosed, rightEarClosed))
        append(", 입 %.3f".format(mouthClosedRatio))
        append(", 자세 y%.1f p%.1f r%.1f".format(yawDeg, pitchDeg, rollDeg))
        append(", 얼굴 %.2f, n=%d".format(faceWidthRatio, sampleCount))
    }
}

/** 위치 한 점. [monotonicMs] 는 elapsedRealtime 기준 측정 시각. */
data class LocationFix(
    val latitude: Double,
    val longitude: Double,
    val accuracyMeters: Float?,
    val monotonicMs: Long,
)
