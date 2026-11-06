package com.personal.driverguard.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.mediapipe.tasks.components.containers.Connection
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.personal.driverguard.analysis.pose.OpenCvHeadPoseEstimator
import com.personal.driverguard.domain.analysis.geometry.CameraIntrinsics
import com.personal.driverguard.domain.analysis.geometry.CanonicalFaceModel
import com.personal.driverguard.domain.analysis.geometry.FaceLandmarkIndices
import com.personal.driverguard.domain.analysis.geometry.LandmarkPoints
import com.personal.driverguard.domain.analysis.geometry.Point3
import com.personal.driverguard.domain.analysis.geometry.PoseMath
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** OpenCV·MediaPipe 네이티브가 필요한 검증. 기기·에뮬레이터에서 실행한다. */
@RunWith(AndroidJUnit4::class)
class PoseAndTopologyDeviceTest {

    private fun project(yaw: Double, pitch: Double, roll: Double, width: Int = 480, height: Int = 640): LandmarkPoints {
        val k = CameraIntrinsics.approximate(width, height)
        val r = PoseMath.cameraFromHead(PoseMath.fromEuler(yaw, pitch, roll))
        val t = Point3(1.5, -2.0, 55.0)
        val nose = CanonicalFaceModel.POINTS.getValue(FaceLandmarkIndices.NOSE_TIP)
        val xs = DoubleArray(FaceLandmarkIndices.LANDMARK_COUNT)
        val ys = DoubleArray(FaceLandmarkIndices.LANDMARK_COUNT)
        for (i in xs.indices) {
            val p = PoseMath.project(k, r, t, CanonicalFaceModel.POINTS[i] ?: nose)
            xs[i] = p.x
            ys[i] = p.y
        }
        return LandmarkPoints(width, height, xs, ys)
    }

    @Test
    fun solvePnPRecoversSyntheticPoseWithExpectedSigns() {
        val estimator = OpenCvHeadPoseEstimator()
        try {
            for ((y, p, r) in listOf(Triple(0.0, 0.0, 0.0), Triple(20.0, 0.0, 0.0), Triple(-25.0, 10.0, 0.0), Triple(0.0, 15.0, -8.0), Triple(30.0, -12.0, 6.0))) {
                val est = estimator.estimate(project(y, p, r))
                assertNotNull("($y,$p,$r)", est)
                val pose = est!!.pose
                assertEquals("yaw ($y,$p,$r)", y, pose.yawDeg, 1.0)
                assertEquals("pitch ($y,$p,$r)", p, pose.pitchDeg, 1.0)
                assertEquals("roll ($y,$p,$r)", r, pose.rollDeg, 1.0)
                assertTrue(pose.reprojectionErrorPx < 0.5)
            }
        } finally {
            estimator.close()
        }
    }

    @Test
    fun landmarkIndicesBelongToOfficialConnectionSets() {
        fun ids(set: Set<Connection>) = set.flatMap { listOf(it.start(), it.end()) }.toSet()
        val left = ids(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
        val right = ids(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
        val lips = ids(FaceLandmarker.FACE_LANDMARKS_LIPS)
        assertTrue(FaceLandmarkIndices.LEFT_EYE_EAR.all { it in left })
        assertTrue(FaceLandmarkIndices.RIGHT_EYE_EAR.all { it in right })
        val mouth = FaceLandmarkIndices.MOUTH_INNER_VERTICAL_PAIRS.flatMap { listOf(it.first, it.second) } +
            listOf(FaceLandmarkIndices.MOUTH_LEFT_CORNER, FaceLandmarkIndices.MOUTH_RIGHT_CORNER)
        assertTrue(mouth.all { it in lips })
    }
}
