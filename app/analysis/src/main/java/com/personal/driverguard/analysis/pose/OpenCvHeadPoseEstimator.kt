package com.personal.driverguard.analysis.pose

import com.personal.driverguard.domain.analysis.geometry.CameraIntrinsics
import com.personal.driverguard.domain.analysis.geometry.CanonicalFaceModel
import com.personal.driverguard.domain.analysis.geometry.FaceLandmarkIndices
import com.personal.driverguard.domain.analysis.geometry.LandmarkPoints
import com.personal.driverguard.domain.analysis.geometry.PoseMath
import com.personal.driverguard.entity.HeadPose
import com.personal.driverguard.entity.Mat3
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import java.io.Closeable
import kotlin.math.abs
import kotlin.math.hypot

/** 자세 추정 결과와 개발 화면용 재투영 축(코끝 원점, 모델 x·y·z 축 끝점, 픽셀 좌표 8개 = 원점·x·y·z) */
class PoseEstimate(val pose: HeadPose, val axes: FloatArray, val intrinsics: CameraIntrinsics)

/**
 * OpenCV solvePnP 기반 머리 자세 추정(HeadPoseEstimator).
 *
 * 1) 얼굴의 안정적인 2D 특징점([FaceLandmarkIndices.POSE_POINTS])을 분석 이미지 픽셀 좌표로 고른다.
 * 2) 같은 인덱스의 3D 기준 좌표([CanonicalFaceModel], cm)를 대응시킨다.
 * 3) 내부 파라미터는 근사치([CameraIntrinsics.approximate]), 왜곡 계수는 0 으로 둔다(개발용 근사).
 * 4) solvePnP(ITERATIVE) → rvec·tvec, 5) Rodrigues → 회전 행렬 R_cm,
 * 6) PoseMath 규약으로 머리 회전·오일러각, 7) 판정은 엔진이 캘리브레이션 기준 대비 상대 회전으로 한다.
 *
 * solvePnP 실패, 얼굴이 카메라 뒤(tvec.z ≤ 0)이거나 카메라를 등진 회전, 유한하지 않은 값은 null 을 돌려준다.
 * 재투영 오차의 허용 범위는 얼굴 크기에 맞춰 ObservationQualityEvaluator 가 판단한다.
 *
 * Mat 은 한 번 만들어 재사용한다. 한 스레드(MediaPipe 결과 콜백 스레드)에서만 호출해야 한다.
 */
class OpenCvHeadPoseEstimator : Closeable {

    // 프로퍼티는 선언 순서대로 초기화되므로, Mat 을 만들기 전에 네이티브 라이브러리부터 불러온다
    init {
        check(loaded) { "OpenCV 네이티브 라이브러리를 불러오지 못했습니다" }
    }

    private val objectPoints = MatOfPoint3f(*CanonicalFaceModel.posePoints().map { Point3(it.x, it.y, it.z) }.toTypedArray())
    private val axisPoints: MatOfPoint3f
    private val imagePoints = MatOfPoint2f()
    private val cameraMatrix = Mat(3, 3, CvType.CV_64F)
    private val distortion = MatOfDouble(0.0, 0.0, 0.0, 0.0)
    private val rvec = Mat()
    private val tvec = Mat()
    private val rotation = Mat()
    private val projected = MatOfPoint2f()
    private val projectedAxes = MatOfPoint2f()
    private val buffer = arrayOfNulls<Point>(FaceLandmarkIndices.POSE_POINTS.size)
    private var closed = false

    init {
        val nose = CanonicalFaceModel.POINTS.getValue(FaceLandmarkIndices.NOSE_TIP)
        axisPoints = MatOfPoint3f(
            Point3(nose.x, nose.y, nose.z),
            Point3(nose.x + AXIS_CM, nose.y, nose.z),
            Point3(nose.x, nose.y + AXIS_CM, nose.z),
            Point3(nose.x, nose.y, nose.z + AXIS_CM),
        )
    }

    fun estimate(points: LandmarkPoints): PoseEstimate? {
        if (closed) return null
        FaceLandmarkIndices.POSE_POINTS.forEachIndexed { i, index ->
            val p = points[index]
            if (!p.isFinite) return null
            buffer[i] = Point(p.x, p.y)
        }
        imagePoints.fromArray(*buffer.requireNoNulls())
        val k = CameraIntrinsics.approximate(points.imageWidth, points.imageHeight)
        cameraMatrix.put(0, 0, k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)

        val ok = try {
            Calib3d.solvePnP(objectPoints, imagePoints, cameraMatrix, distortion, rvec, tvec, false, Calib3d.SOLVEPNP_ITERATIVE)
        } catch (_: Exception) {
            false
        }
        if (!ok) return null
        val tz = tvec.get(2, 0)?.firstOrNull() ?: return null
        if (!(tz > 0)) return null

        Calib3d.Rodrigues(rvec, rotation)
        val r = DoubleArray(9)
        rotation.get(0, 0, r)
        val cameraFromModel = Mat3(r)
        if (!cameraFromModel.isFinite()) return null
        val head = PoseMath.headFromCamera(cameraFromModel)
        // 얼굴 앞쪽(z)이 카메라를 등지면 잘못된 해로 본다
        if (head[2, 2] <= 0.0) return null

        Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distortion, projected)
        val reprojected = projected.toArray()
        val observed = imagePoints.toArray()
        var error = 0.0
        for (i in reprojected.indices) error += hypot(reprojected[i].x - observed[i].x, reprojected[i].y - observed[i].y)
        error /= reprojected.size

        Calib3d.projectPoints(axisPoints, rvec, tvec, cameraMatrix, distortion, projectedAxes)
        val axes = projectedAxes.toArray().flatMap { listOf(it.x.toFloat(), it.y.toFloat()) }.toFloatArray()

        val pose = PoseMath.headPose(head, error)
        if (!pose.yawDeg.isFinite() || abs(pose.yawDeg) > 90 || abs(pose.pitchDeg) > 90) return null
        return PoseEstimate(pose, axes, k)
    }

    override fun close() {
        if (closed) return
        closed = true
        listOf(objectPoints, axisPoints, imagePoints, cameraMatrix, distortion, rvec, tvec, rotation, projected, projectedAxes)
            .forEach { it.release() }
    }

    companion object {
        /** 개발 화면 축 길이(cm) */
        private const val AXIS_CM = 5.0

        /** OpenCVLoader.initLocal 은 앱 APK 에 들어 있는 libopencv_java4.so 를 불러온다(OpenCV Manager 불필요) */
        val loaded: Boolean by lazy { OpenCVLoader.initLocal() }
    }
}
