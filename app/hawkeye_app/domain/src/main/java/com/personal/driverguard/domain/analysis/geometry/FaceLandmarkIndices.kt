package com.personal.driverguard.domain.analysis.geometry

/**
 * MediaPipe Face Landmarker(face_landmarker.task, 478점 = 얼굴 메시 468 + 홍채 10) 랜드마크 인덱스.
 * 분석에 쓰는 인덱스는 모두 여기에만 둔다.
 *
 * 좌우는 운전자 본인 기준이다(MediaPipe 의 FACE_LANDMARKS_LEFT_EYE 와 같은 규약).
 * 비미러 전면 카메라 이미지에서는 운전자의 오른쪽 눈이 이미지 왼쪽에 보인다.
 *
 * 검증 방법:
 *  - MediaPipe 공식 canonical_face_model.obj 좌표로 대응 관계를 확인했다([CanonicalFaceModel], 단위 테스트).
 *    EAR 의 p2↔p6, p3↔p5 는 x 가 같고 y 만 다른 위·아래 눈꺼풀 점이며, p1·p4 는 눈꼬리다.
 *  - 기기 테스트에서 FaceLandmarker.FACE_LANDMARKS_LEFT_EYE / RIGHT_EYE / LIPS 연결 정의에 포함되는지 확인한다.
 *  - 개발 화면에서 인덱스 번호와 연결선을 영상 위에 그려 눈꺼풀·입술 위치를 눈으로 확인할 수 있다.
 */
object FaceLandmarkIndices {
    /** 모델이 내보내는 점 개수(홍채 포함) */
    const val LANDMARK_COUNT = 478

    /**
     * EAR 6점, Soukupová & Čech(2016)의 순서: p1·p4 눈꼬리, p2·p3 위 눈꺼풀, p6·p5 각각 대응하는 아래 눈꺼풀.
     * 오른쪽 눈: p1=33(바깥 눈꼬리) p2=160 p3=158 p4=133(안쪽 눈꼬리) p5=153 p6=144
     */
    val RIGHT_EYE_EAR = intArrayOf(33, 160, 158, 133, 153, 144)

    /** 왼쪽 눈: p1=362(안쪽 눈꼬리) p2=385 p3=387 p4=263(바깥 눈꼬리) p5=373 p6=380 */
    val LEFT_EYE_EAR = intArrayOf(362, 385, 387, 263, 373, 380)

    /** 입술 안쪽 위·아래 점 쌍(가운데, 왼쪽 운전자 기준 오른편, 왼편). 세로 거리 평균을 쓴다. */
    val MOUTH_INNER_VERTICAL_PAIRS: List<Pair<Int, Int>> = listOf(13 to 14, 82 to 87, 312 to 317)

    /** 입꼬리(바깥). 입 가로 너비 */
    const val MOUTH_RIGHT_CORNER = 61
    const val MOUTH_LEFT_CORNER = 291

    /**
     * solvePnP 에 쓰는 점. 입·턱처럼 표정에 따라 크게 움직이는 점은 빼고 이마·콧등·코·눈꼬리만 쓴다.
     * 10 이마, 168 미간, 6 콧등, 1 코끝, 98·327 콧볼, 33·133 오른쪽 눈꼬리, 362·263 왼쪽 눈꼬리
     */
    val POSE_POINTS = intArrayOf(10, 168, 6, 1, 98, 327, 33, 133, 362, 263)

    const val NOSE_TIP = 1

    /** 개발 화면 오버레이에 강조할 점(EAR·입·자세에 쓰는 점) */
    val HIGHLIGHTED: IntArray = (RIGHT_EYE_EAR + LEFT_EYE_EAR + MOUTH_INNER_VERTICAL_PAIRS.flatMap { listOf(it.first, it.second) } +
        intArrayOf(MOUTH_RIGHT_CORNER, MOUTH_LEFT_CORNER) + POSE_POINTS).distinct().toIntArray()

    /** 개발 화면에 그릴 EAR 윤곽선(p1-p2-p3-p4-p5-p6-p1) */
    fun earOutline(eye: IntArray): List<Pair<Int, Int>> = (eye.indices).map { eye[it] to eye[(it + 1) % eye.size] }
}
