package com.personal.driverguard.domain.analysis.geometry

/**
 * solvePnP 에 쓰는 3D 기준 얼굴 좌표.
 *
 * 출처: MediaPipe face_geometry 모듈의 canonical_face_model.obj (Apache License 2.0)
 *   https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
 * obj 의 n번째 정점(0부터)이 Face Landmarker 의 n번 랜드마크에 대응한다. 값은 파일에서 그대로 옮겼다.
 * 단위: 센티미터. 좌표계: x = 운전자 기준 왼쪽, y = 위, z = 얼굴 앞쪽(카메라 쪽).
 *
 * MediaPipe 가 내보내는 정규화 z 는 실제 깊이가 아니므로 쓰지 않는다. 3D 좌표는 이 평균 얼굴 모델만 쓰며,
 * 개인 얼굴 형태 차이 때문에 절대 각도에는 오차가 생긴다(판정은 캘리브레이션 기준 대비 상대 회전으로 한다).
 */
object CanonicalFaceModel {
    val POINTS: Map<Int, Point3> = mapOf(
        // 자세 추정 점
        10 to Point3(0.000000, 8.261778, 4.481535),
        168 to Point3(0.000000, 3.271027, 5.236015),
        6 to Point3(0.000000, 2.473255, 5.788627),
        1 to Point3(0.000000, -1.126865, 7.475604),
        98 to Point3(-1.405627, -1.714196, 5.241087),
        327 to Point3(1.405627, -1.714196, 5.241087),
        33 to Point3(-4.445859, 2.663991, 3.173422),
        133 to Point3(-1.856432, 2.585245, 3.757904),
        362 to Point3(1.856432, 2.585245, 3.757904),
        263 to Point3(4.445859, 2.663991, 3.173422),
        // 눈꺼풀(EAR 대응 검증용)
        160 to Point3(-3.670075, 2.927714, 3.724325),
        144 to Point3(-3.670075, 2.360153, 3.635230),
        158 to Point3(-2.724032, 2.961810, 3.871767),
        153 to Point3(-2.724032, 2.315802, 3.777151),
        385 to Point3(2.724032, 2.961810, 3.871767),
        380 to Point3(2.724032, 2.315802, 3.777151),
        387 to Point3(3.670075, 2.927714, 3.724325),
        373 to Point3(3.670075, 2.360153, 3.635230),
        // 입술(입 벌림 대응 검증용)
        13 to Point3(0.000000, -3.994436, 5.219482),
        14 to Point3(0.000000, -4.542400, 5.404754),
        82 to Point3(-0.533422, -3.993222, 5.138202),
        87 to Point3(-0.583218, -4.517982, 5.339869),
        312 to Point3(0.533422, -3.993222, 5.138202),
        317 to Point3(0.583218, -4.517982, 5.339869),
        61 to Point3(-2.456206, -4.342621, 4.283884),
        291 to Point3(2.456206, -4.342621, 4.283884),
    )

    fun posePoints(): List<Point3> = FaceLandmarkIndices.POSE_POINTS.map { POINTS.getValue(it) }
}
