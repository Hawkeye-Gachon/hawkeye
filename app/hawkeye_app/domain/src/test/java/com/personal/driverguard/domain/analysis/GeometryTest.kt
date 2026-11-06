package com.personal.driverguard.domain.analysis

import com.personal.driverguard.domain.analysis.geometry.CameraIntrinsics
import com.personal.driverguard.domain.analysis.geometry.CanonicalFaceModel
import com.personal.driverguard.domain.analysis.geometry.FaceLandmarkIndices
import com.personal.driverguard.domain.analysis.geometry.LandmarkGeometry
import com.personal.driverguard.domain.analysis.geometry.LandmarkPoints
import com.personal.driverguard.domain.analysis.geometry.Point3
import com.personal.driverguard.domain.analysis.geometry.PoseMath
import com.personal.driverguard.entity.Mat3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class GeometryTest {

    private val minWidth = 6.0

    /** 정면 얼굴을 투영한 478점(필요한 점만 실제 값, 나머지는 코끝) */
    private fun projectedFace(
        yaw: Double = 0.0,
        pitch: Double = 0.0,
        roll: Double = 0.0,
        width: Int = 640,
        height: Int = 480,
        distanceCm: Double = 60.0,
    ): LandmarkPoints {
        val k = CameraIntrinsics.approximate(width, height)
        val r = PoseMath.cameraFromHead(PoseMath.fromEuler(yaw, pitch, roll))
        val t = Point3(0.0, 0.0, distanceCm)
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
    fun `EAR 인덱스는 공식 기준 얼굴에서 눈꼬리와 위·아래 눈꺼풀에 대응한다`() {
        for (eye in listOf(FaceLandmarkIndices.RIGHT_EYE_EAR, FaceLandmarkIndices.LEFT_EYE_EAR)) {
            val p = eye.map { CanonicalFaceModel.POINTS.getValue(it) }
            // p2↔p6, p3↔p5 는 같은 x 의 위·아래 점
            assertEquals(p[1].x, p[5].x, 1e-6)
            assertEquals(p[2].x, p[4].x, 1e-6)
            assertTrue(p[1].y > p[5].y && p[2].y > p[4].y)
            // p1·p4 는 눈의 양 끝(가로로 가장 바깥)
            val xs = p.map { it.x }
            assertEquals(xs.min(), minOf(p[0].x, p[3].x), 1e-6)
            assertEquals(xs.max(), maxOf(p[0].x, p[3].x), 1e-6)
        }
        // 오른쪽 눈은 운전자 기준 오른쪽(모델 x 음수), 왼쪽 눈은 양수
        assertTrue(FaceLandmarkIndices.RIGHT_EYE_EAR.all { CanonicalFaceModel.POINTS.getValue(it).x < 0 })
        assertTrue(FaceLandmarkIndices.LEFT_EYE_EAR.all { CanonicalFaceModel.POINTS.getValue(it).x > 0 })
        // 입술 안쪽 점 쌍은 위가 아래보다 높다
        for ((top, bottom) in FaceLandmarkIndices.MOUTH_INNER_VERTICAL_PAIRS) {
            assertTrue(CanonicalFaceModel.POINTS.getValue(top).y > CanonicalFaceModel.POINTS.getValue(bottom).y)
        }
    }

    @Test
    fun `EAR 과 입 벌림 비율은 얼굴 크기(거리)에 따라 바뀌지 않는다`() {
        val near = projectedFace(distanceCm = 40.0)
        val far = projectedFace(distanceCm = 80.0)
        val earNear = LandmarkGeometry.eyeAspectRatio(near, FaceLandmarkIndices.LEFT_EYE_EAR, minWidth).value!!
        val earFar = LandmarkGeometry.eyeAspectRatio(far, FaceLandmarkIndices.LEFT_EYE_EAR, minWidth).value!!
        assertEquals(earNear, earFar, 0.01)
        val mouthNear = mouth(near)!!
        val mouthFar = mouth(far)!!
        assertEquals(mouthNear, mouthFar, 0.01)
    }

    private fun mouth(points: LandmarkPoints) = LandmarkGeometry.mouthOpenRatio(
        points,
        FaceLandmarkIndices.MOUTH_INNER_VERTICAL_PAIRS,
        FaceLandmarkIndices.MOUTH_RIGHT_CORNER,
        FaceLandmarkIndices.MOUTH_LEFT_CORNER,
        10.0,
    ).value

    @Test
    fun `분모가 0이거나 좌표가 NaN·Infinity 면 무효`() {
        val n = FaceLandmarkIndices.LANDMARK_COUNT
        val zero = LandmarkPoints(640, 480, DoubleArray(n) { 100.0 }, DoubleArray(n) { 100.0 })
        assertNull(LandmarkGeometry.eyeAspectRatio(zero, FaceLandmarkIndices.LEFT_EYE_EAR, minWidth).value)
        assertNull(mouth(zero))
        val nanXs = DoubleArray(n) { it.toDouble() }.also { it[FaceLandmarkIndices.LEFT_EYE_EAR[1]] = Double.NaN }
        val nan = LandmarkPoints(640, 480, nanXs, DoubleArray(n) { it.toDouble() })
        assertNull(LandmarkGeometry.eyeAspectRatio(nan, FaceLandmarkIndices.LEFT_EYE_EAR, minWidth).value)
        val infXs = DoubleArray(n) { it.toDouble() }.also { it[FaceLandmarkIndices.MOUTH_LEFT_CORNER] = Double.POSITIVE_INFINITY }
        assertNull(mouth(LandmarkPoints(640, 480, infXs, DoubleArray(n) { it.toDouble() })))
        // 분모가 기준보다 작으면 무효
        val tiny = projectedFace(distanceCm = 2_000.0)
        assertNull(LandmarkGeometry.eyeAspectRatio(tiny, FaceLandmarkIndices.LEFT_EYE_EAR, minWidth).value)
    }

    @Test
    fun `정규화 좌표는 이미지 너비·높이를 따로 곱해야 종횡비가 달라도 같은 EAR 이 나온다`() {
        val face = projectedFace(width = 640, height = 480)
        val earPx = LandmarkGeometry.eyeAspectRatio(face, FaceLandmarkIndices.LEFT_EYE_EAR, minWidth).value!!
        // 같은 장면을 정규화 좌표로 바꿨다가 다시 픽셀로 복원
        val normalized = FloatArray(FaceLandmarkIndices.LANDMARK_COUNT * 2) { i ->
            val p = face[i / 2]
            if (i % 2 == 0) (p.x / 640).toFloat() else (p.y / 480).toFloat()
        }
        val restored = LandmarkPoints.fromNormalized(normalized, 640, 480)
        assertEquals(earPx, LandmarkGeometry.eyeAspectRatio(restored, FaceLandmarkIndices.LEFT_EYE_EAR, minWidth).value!!, 1e-4)
        // 정규화 좌표를 정사각형으로 착각하면(둘 다 같은 값을 곱하면) EAR 이 틀어진다
        val wrong = LandmarkPoints.fromNormalized(normalized, 640, 640)
        val earWrong = LandmarkGeometry.eyeAspectRatio(wrong, FaceLandmarkIndices.LEFT_EYE_EAR, minWidth).value!!
        assertTrue(abs(earWrong - earPx) / earPx > 0.2)
    }

    @Test
    fun `오일러각 규약은 왕복 변환과 상대 회전에서 일관된다`() {
        for ((y, p, r) in listOf(Triple(20.0, 0.0, 0.0), Triple(-30.0, 15.0, 5.0), Triple(10.0, -25.0, -10.0))) {
            val e = PoseMath.euler(PoseMath.fromEuler(y, p, r))
            assertEquals(y, e.yawDeg, 1e-9)
            assertEquals(p, e.pitchDeg, 1e-9)
            assertEquals(r, e.rollDeg, 1e-9)
        }
        val base = PoseMath.fromEuler(0.0, 10.0, 0.0)
        val current = base * PoseMath.fromEuler(25.0, 0.0, 0.0)
        val rel = PoseMath.euler(PoseMath.relative(base, current))
        assertEquals(25.0, rel.yawDeg, 1e-9)
        assertEquals(0.0, rel.pitchDeg, 1e-9)
        assertTrue(PoseMath.relative(base, base).approxEquals(Mat3.IDENTITY))
    }

    @Test
    fun `머리 방향 부호는 비미러 카메라 이미지에서 기대한 쪽으로 움직인다`() {
        val front = projectedFace()
        val nose = FaceLandmarkIndices.NOSE_TIP
        // 운전자 기준 왼쪽으로 돌리면(yaw+) 비미러 이미지에서 코끝이 오른쪽(+x)으로 간다
        assertTrue(projectedFace(yaw = 20.0)[nose].x > front[nose].x + 5)
        // 숙이면(pitch+) 코끝이 아래(+y)로 간다
        assertTrue(projectedFace(pitch = 20.0)[nose].y > front[nose].y + 5)
        // 운전자 기준 오른쪽 어깨 쪽으로 기울이면(roll+) 운전자 왼쪽 눈(이미지 오른쪽)이 위로 올라간다
        val tilted = projectedFace(roll = 20.0)
        assertTrue(tilted[263].y < tilted[33].y)
        // 정면 투영에서 운전자 오른쪽 눈(33)은 이미지 왼쪽에 있다
        assertTrue(front[33].x < front[263].x)
        assertNotNull(front)
    }
}
