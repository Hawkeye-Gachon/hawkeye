package com.personal.driverguard.domain.analysis.geometry

import com.personal.driverguard.entity.HeadPose
import com.personal.driverguard.entity.Mat3
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * 카메라 내부 파라미터. 회전 보정된 분석 이미지(픽셀) 기준이다.
 * 분석 이미지를 회전·리사이즈·크롭하면 이 값도 같은 변환으로 다시 만들어야 한다.
 */
data class CameraIntrinsics(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    /** 실제 렌즈 보정값이 아니라 근사치인지 */
    val approximate: Boolean,
) {
    companion object {
        /**
         * 개발용 근사: 초점거리 = 긴 변 길이, 주점 = 이미지 중심, 왜곡 0.
         * 실제 전면 카메라 화각(대략 70~80°)과 다를 수 있어 절대 각도 정확도를 보장하지 않는다.
         */
        fun approximate(imageWidth: Int, imageHeight: Int): CameraIntrinsics {
            val f = maxOf(imageWidth, imageHeight).toDouble()
            return CameraIntrinsics(f, f, imageWidth / 2.0, imageHeight / 2.0, approximate = true)
        }
    }
}

/**
 * 머리 자세 회전 규약.
 *
 * 좌표계
 *  - 카메라(OpenCV): x = 이미지 오른쪽, y = 이미지 아래, z = 카메라 앞(운전자 쪽). 비미러 이미지 기준.
 *  - 얼굴 모델([CanonicalFaceModel]): x = 운전자 왼쪽, y = 위, z = 얼굴 앞.
 *  solvePnP 는 R_cm(모델 → 카메라)을 준다. 운전자가 카메라를 정면으로 볼 때 R_cm = F = diag(1, −1, −1)
 *  (운전자 왼쪽은 비미러 이미지의 오른쪽, 위는 −y, 얼굴 앞은 카메라 쪽 −z).
 *  머리 회전 H 는 R_cm = F·H 로 정의해 정면에서 H = I 가 되게 한다.
 *
 * 오일러각: H = Ry(yaw)·Rx(pitch)·Rz(roll) (머리 좌표계 내재 회전 Y→X→Z)
 *  - yaw   = atan2(H02, H22)   + : 운전자 기준 왼쪽으로 돌림
 *  - pitch = asin(−H12)        + : 고개 숙임
 *  - roll  = atan2(H10, H11)   + : 운전자 기준 오른쪽 어깨 쪽 기울임
 * 판정에는 기준 자세 대비 상대 회전 H_rel = H_baseᵀ·H 를 먼저 구한 뒤 각도로 해석한다(경계각 문제 회피).
 */
object PoseMath {
    val CAMERA_FROM_FRONTAL = Mat3(doubleArrayOf(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0))

    data class Euler(val yawDeg: Double, val pitchDeg: Double, val rollDeg: Double)

    /** solvePnP 의 R_cm 에서 머리 회전 H 를 구한다. F 는 대칭 직교행렬이라 F⁻¹ = F. */
    fun headFromCamera(rotationCameraFromModel: Mat3): Mat3 = CAMERA_FROM_FRONTAL * rotationCameraFromModel

    fun cameraFromHead(head: Mat3): Mat3 = CAMERA_FROM_FRONTAL * head

    fun euler(h: Mat3): Euler {
        val pitch = asin((-h[1, 2]).coerceIn(-1.0, 1.0))
        val yaw = atan2(h[0, 2], h[2, 2])
        val roll = atan2(h[1, 0], h[1, 1])
        return Euler(Math.toDegrees(yaw), Math.toDegrees(pitch), Math.toDegrees(roll))
    }

    fun fromEuler(yawDeg: Double, pitchDeg: Double, rollDeg: Double): Mat3 {
        val a = Math.toRadians(yawDeg)
        val b = Math.toRadians(pitchDeg)
        val c = Math.toRadians(rollDeg)
        val ry = Mat3(doubleArrayOf(cos(a), 0.0, sin(a), 0.0, 1.0, 0.0, -sin(a), 0.0, cos(a)))
        val rx = Mat3(doubleArrayOf(1.0, 0.0, 0.0, 0.0, cos(b), -sin(b), 0.0, sin(b), cos(b)))
        val rz = Mat3(doubleArrayOf(cos(c), -sin(c), 0.0, sin(c), cos(c), 0.0, 0.0, 0.0, 1.0))
        return ry * rx * rz
    }

    /** 기준 자세 대비 상대 회전 */
    fun relative(base: Mat3, current: Mat3): Mat3 = base.transpose() * current

    fun headPose(head: Mat3, reprojectionErrorPx: Double): HeadPose {
        val e = euler(head)
        return HeadPose(head, e.yawDeg, e.pitchDeg, e.rollDeg, reprojectionErrorPx)
    }

    /** 시연·테스트용: 각도에서 HeadPose 를 만든다 */
    fun poseOf(yawDeg: Double, pitchDeg: Double = 0.0, rollDeg: Double = 0.0): HeadPose =
        HeadPose(fromEuler(yawDeg, pitchDeg, rollDeg), yawDeg, pitchDeg, rollDeg, 0.0)

    /** 회전 행렬 R 을 3D 점에 적용 */
    fun apply(r: Mat3, p: Point3): Point3 = Point3(
        r[0, 0] * p.x + r[0, 1] * p.y + r[0, 2] * p.z,
        r[1, 0] * p.x + r[1, 1] * p.y + r[1, 2] * p.z,
        r[2, 0] * p.x + r[2, 1] * p.y + r[2, 2] * p.z,
    )

    /** 핀홀 투영(왜곡 0). 테스트와 축 표시 검증에 쓴다. */
    fun project(intrinsics: CameraIntrinsics, rotation: Mat3, translation: Point3, p: Point3): Point2 {
        val c = apply(rotation, p)
        val x = c.x + translation.x
        val y = c.y + translation.y
        val z = c.z + translation.z
        return Point2(intrinsics.fx * x / z + intrinsics.cx, intrinsics.fy * y / z + intrinsics.cy)
    }
}
