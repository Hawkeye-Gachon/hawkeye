package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.EL_DOWN_L
import hawkeye.drowsy.core.EL_DOWN_R
import hawkeye.drowsy.core.EL_IN_L
import hawkeye.drowsy.core.EL_IN_R
import hawkeye.drowsy.core.EL_OUT_L
import hawkeye.drowsy.core.EL_OUT_R
import hawkeye.drowsy.core.EL_UP_L
import hawkeye.drowsy.core.EL_UP_R
import hawkeye.drowsy.core.Point
import hawkeye.drowsy.core.RawFrame
import hawkeye.drowsy.core.spec.PipelineSpec
import java.util.Collections
import kotlin.math.PI

// 원본: drowsy/signals.py 전체 (SPEC 4, 5장). 계획 5.2절의 선언 그대로다.
// RawFrame 하나를 받아 FrameSignals 하나를 돌려준다. 상태가 없다(R19).
//
// 규약 (signals.py:6-13):
// - 거리와 비율은 모두 픽셀 좌표 (x·w, y·h)에서 구한다.
// - 행렬 16개 값은 열 우선이다: M[i][j] = m[4·j + i]. 3x3 회전은 행 우선 9개: R[i][j] = r[3·i + j].
// - 회전 R = Ry(yaw)·Rx(pitch)·Rz(roll), 각도는 도. 도와 라디안은 상수 곱으로 바꾼다(R20).
// - _l은 MediaPipe Left(L 집합, *Left 블렌드셰이프), _r은 Right다.
//
// 수학 함수는 atan2, hypot, sin, cos만 쓴다(R20). K20 (가): StrictMath를 쓴다. PC와 기기가 같은 비트를 내게 하려는
// 것이다. 가정(A17): Android의 StrictMath가 fdlibm을 따른다(E17에서 PC와 비트 비교).
// 가정(A2): StrictMath와 CPython(glibc libm, 그리고 CPython 3.11 자체의 hypot)의 atan2, hypot, sin, cos는 마지막 몇 자리만
// 다르다. 그래서 이 파일의 결과는 Python과 비트까지 같다고 보지 않는다(벡터 허용 오차, 계획 9.2.4절의 좁은 허용 오차).

internal const val RAD_TO_DEG: Double = 180.0 / PI
internal const val DEG_TO_RAD: Double = PI / 180.0

/** signals.py:39 `_NO_EYE_LOOK`. */
private val NO_EYE_LOOK: List<Double> = Collections.unmodifiableList(listOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

/**
 * 프레임 하나의 원시 신호 (types.py:175-210, SPEC 3.2). 필드 순서는 Python 필드 순서와 같다.
 *
 * - earL, earR: 픽셀 좌표에서 구한 2D EAR. 쓸 수 없는 눈(eyeW < signals.min_eye_width_px)은 0.0이다.
 * - eyeWL, eyeWR: 눈 폭 |p1 − p4| (픽셀). 쓸 수 없는 눈도 측정값 그대로다.
 * - blinkL, blinkR, jawOpen: 블렌드셰이프 값 그대로. eyeLook: eyeLook* 8개, EYE_LOOK_ORDER 순서.
 * - yaw, pitch, roll: 변환 행렬에서 구한 절대 오일러 각(도).
 * - irisHL, irisHR, irisVL, irisVR: 홍채 비율. 계산하지 않았거나 계산할 수 없으면 null이다.
 * 얼굴이 없으면 모든 실수 필드는 0.0, eyeLook은 0.0 여덟 개, 홍채 비율은 null이다.
 */
internal data class FrameSignals(
    val tMs: Long,
    val face: Boolean,
    val earL: Double,
    val earR: Double,
    val eyeWL: Double,
    val eyeWR: Double,
    val blinkL: Double,
    val blinkR: Double,
    val eyeLook: List<Double>,
    val jawOpen: Double,
    val yaw: Double,
    val pitch: Double,
    val roll: Double,
    val irisHL: Double? = null,
    val irisHR: Double? = null,
    val irisVL: Double? = null,
    val irisVR: Double? = null,
)

// ---------------------------------------------------------------------------
// 눈: EAR, 홍채 비율
// ---------------------------------------------------------------------------

/**
 * signals.py:47-58. EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|). 입력은 p1..p6 순서의 픽셀 좌표 6개다.
 * 눈 폭 |p1−p4|가 0 이하이면 0.0이다(signals.py:56-57). 6점이 아니면 IllegalArgumentException(Python은 ValueError).
 */
internal fun eyeAspectRatio(ptsPx: List<Point>): Double {
    if (ptsPx.size != 6) {
        throw IllegalArgumentException("eye_aspect_ratio needs 6 points (p1..p6)")
    }
    val width = distance(ptsPx[0], ptsPx[3])
    if (width <= 0.0) {
        return 0.0
    }
    return (distance(ptsPx[1], ptsPx[5]) + distance(ptsPx[2], ptsPx[4])) / (2.0 * width)
}

/** signals.py:61-84의 돌려주는 값 `(h, v)`. */
internal data class IrisRatio(val h: Double, val v: Double)

/**
 * signals.py:61-84. 홍채 중심의 눈 안 위치 (h, v). 입력은 바깥 눈 꼬리, 안쪽 눈 꼬리, 홍채 중심의 픽셀 좌표다.
 * a = inner − outer, d = iris − outer일 때 h = dot(d, a)/|a|², v = cross(a, d)/|a|², cross(a, d) = a.x·d.y − a.y·d.x.
 * 두 눈 꼬리가 같은 점이면(|a|² <= 0) (0.0, 0.0)이다.
 */
internal fun irisRatio(outer: Point, inner: Point, iris: Point): IrisRatio {
    val ax = inner.x - outer.x
    val ay = inner.y - outer.y
    val dx = iris.x - outer.x
    val dy = iris.y - outer.y
    val normSq = ax * ax + ay * ay
    if (normSq <= 0.0) {
        return IrisRatio(0.0, 0.0)
    }
    val h = (dx * ax + dy * ay) / normSq
    val v = (ax * dy - ay * dx) / normSq
    return IrisRatio(h, v)
}

// ---------------------------------------------------------------------------
// 눈동자 합성값 (시선 규칙 보조)
// ---------------------------------------------------------------------------

/** signals.py:92-101. gaze_h = ((eyeLookInRight + eyeLookOutLeft) − (eyeLookOutRight + eyeLookInLeft)) / 2. */
internal fun gazeH(eyeLook: List<Double>): Double {
    val towardLeft = eyeLook[EL_IN_R] + eyeLook[EL_OUT_L]
    val towardRight = eyeLook[EL_OUT_R] + eyeLook[EL_IN_L]
    return (towardLeft - towardRight) / 2.0
}

/** signals.py:104-108. gaze_v = (eyeLookUp 두 눈 평균) − (eyeLookDown 두 눈 평균). */
internal fun gazeV(eyeLook: List<Double>): Double {
    val up = (eyeLook[EL_UP_L] + eyeLook[EL_UP_R]) / 2.0
    val down = (eyeLook[EL_DOWN_L] + eyeLook[EL_DOWN_R]) / 2.0
    return up - down
}

// ---------------------------------------------------------------------------
// 머리 자세
// ---------------------------------------------------------------------------

/** (yaw, pitch, roll), 도 단위. Python의 세 값 튜플. */
internal data class Euler(val yaw: Double, val pitch: Double, val roll: Double)

/**
 * signals.py:116-128. 열 우선 4x4 변환 행렬(16개)에서 (yaw, pitch, roll)을 도 단위로 구한다.
 * yaw = atan2(m[8], m[10]), pitch = atan2(−m[9], hypot(m[1], m[5])), roll = atan2(m[1], m[5]).
 * 16개가 아니면 IllegalArgumentException.
 */
internal fun eulerFromColMajor(m: List<Double>): Euler {
    if (m.size != 16) {
        throw IllegalArgumentException("transform matrix must hold 16 values (column-major 4x4)")
    }
    val yaw = StrictMath.atan2(m[8], m[10])
    val pitch = StrictMath.atan2(-m[9], StrictMath.hypot(m[1], m[5]))
    val roll = StrictMath.atan2(m[1], m[5])
    return Euler(yaw * RAD_TO_DEG, pitch * RAD_TO_DEG, roll * RAD_TO_DEG)
}

/** signals.py:131-138. 행 우선 3x3 회전(9개)에서 (yaw, pitch, roll). 9개가 아니면 IllegalArgumentException. */
internal fun eulerFromRotation(r: List<Double>): Euler {
    if (r.size != 9) {
        throw IllegalArgumentException("rotation must hold 9 values (row-major 3x3)")
    }
    val yaw = StrictMath.atan2(r[2], r[8])
    val pitch = StrictMath.atan2(-r[5], StrictMath.hypot(r[3], r[4]))
    val roll = StrictMath.atan2(r[3], r[4])
    return Euler(yaw * RAD_TO_DEG, pitch * RAD_TO_DEG, roll * RAD_TO_DEG)
}

/**
 * signals.py:141-164. 도 단위 각도로 R = Ry(yaw)·Rx(pitch)·Rz(roll)을 만든다. 행 우선 9개 값.
 * 원소의 식과 곱의 순서는 Python 그대로다(왼쪽부터).
 */
internal fun rotationFromEuler(yaw: Double, pitch: Double, roll: Double): List<Double> {
    val cy = StrictMath.cos(yaw * DEG_TO_RAD)
    val sy = StrictMath.sin(yaw * DEG_TO_RAD)
    val cp = StrictMath.cos(pitch * DEG_TO_RAD)
    val sp = StrictMath.sin(pitch * DEG_TO_RAD)
    val cr = StrictMath.cos(roll * DEG_TO_RAD)
    val sr = StrictMath.sin(roll * DEG_TO_RAD)
    return Collections.unmodifiableList(
        listOf(
            cy * cr + sy * sp * sr,
            sy * sp * cr - cy * sr,
            sy * cp,
            cp * sr,
            cp * cr,
            -sp,
            cy * sp * sr - sy * cr,
            sy * sr + cy * sp * cr,
            cy * cp,
        ),
    )
}

/**
 * signals.py:167-171. 열 우선 4x4에서 회전 블록을 행 우선 9개 값으로 꺼낸다. R[i][j] = m[4·j + i].
 * 파이프라인 경로 밖이다(SPEC 16.1 끝 목록, docs/SPEC.md:2406-2408). 단위 시험과 C-func 자료가 쓴다.
 */
internal fun rotationFromColMajor(m: List<Double>): List<Double> {
    if (m.size != 16) {
        throw IllegalArgumentException("transform matrix must hold 16 values (column-major 4x4)")
    }
    return Collections.unmodifiableList(listOf(m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]))
}

/**
 * signals.py:174-185. 기준 회전 대비 회전 R0ᵀ·R, 입력과 출력 모두 행 우선 9개.
 * (R0ᵀ·R)[i][j] = r0[i]·r[j] + r0[3+i]·r[3+j] + r0[6+i]·r[6+j], 왼쪽부터 더한다(SPEC 5.3).
 */
internal fun relativeRotation(r: List<Double>, r0: List<Double>): List<Double> {
    if (r.size != 9 || r0.size != 9) {
        throw IllegalArgumentException("rotation must hold 9 values (row-major 3x3)")
    }
    val out = ArrayList<Double>(9)
    for (i in 0 until 3) {
        for (j in 0 until 3) {
            out.add(r0[i] * r[j] + r0[3 + i] * r[3 + j] + r0[6 + i] * r[6 + j])
        }
    }
    return Collections.unmodifiableList(out)
}

/**
 * signals.py:188-193. 기준 자세 대비 (yaw, pitch, roll) = Euler(R0ᵀ·R). m은 열 우선 16개, r0은 행 우선 9개.
 * 파이프라인 경로 밖이다(calibrate.frame_state는 각도에서 R을 다시 만든다). 단위 시험과 C-func 자료가 쓴다.
 */
internal fun relativeEuler(m: List<Double>, r0: List<Double>): Euler =
    eulerFromRotation(relativeRotation(rotationFromColMajor(m), r0))

// ---------------------------------------------------------------------------
// 프레임 하나의 원시 신호
// ---------------------------------------------------------------------------

/** signals.py:201-208. 눈 하나의 기하 신호. 쓸 수 없는 눈은 ear 0.0, 홍채 비율 null, 폭은 그대로다. */
private data class EyeSignals(val ear: Double, val widthPx: Double, val irisH: Double?, val irisV: Double?)

/**
 * signals.py:211-270 (SPEC 5.7). 프레임 하나의 원시 신호.
 *
 * - 얼굴이 없으면 모든 실수 필드 0.0, eyeLook 0.0 여덟 개, 홍채 비율 null.
 * - 눈 폭이 `spec.signals.min_eye_width_px` 미만인 눈은 쓸 수 없다: ear 0.0, 홍채 비율 null, 폭은 측정값 그대로.
 * - 블렌드셰이프와 각도는 눈 폭과 무관하게 채운다.
 * 계산 순서는 Python 그대로다: L 눈, R 눈, eyeLook, 각도. 그래서 입력이 틀렸을 때 먼저 나는 오류도 같다.
 * face = true인데 필요한 랜드마크나 블렌드셰이프가 없으면 IllegalArgumentException(Python은 KeyError, R17),
 * 행렬이 16개가 아니면 IllegalArgumentException(Python은 ValueError).
 */
internal fun compute(frame: RawFrame, spec: PipelineSpec): FrameSignals {
    if (!frame.face) {
        return FrameSignals(
            tMs = frame.tMs,
            face = false,
            earL = 0.0,
            earR = 0.0,
            eyeWL = 0.0,
            eyeWR = 0.0,
            blinkL = 0.0,
            blinkR = 0.0,
            eyeLook = NO_EYE_LOOK,
            jawOpen = 0.0,
            yaw = 0.0,
            pitch = 0.0,
            roll = 0.0,
        )
    }

    val lms = spec.landmarks
    val minWidthPx = spec.signals.minEyeWidthPx
    val eyeL = eyeSignals(frame, lms.earL, lms.cornerOuterL, lms.cornerInnerL, lms.irisL[0], minWidthPx)
    val eyeR = eyeSignals(frame, lms.earR, lms.cornerOuterR, lms.cornerInnerR, lms.irisR[0], minWidthPx)

    val names = spec.blendshapes
    val eyeLook = ArrayList<Double>(names.eyeLook.size)
    for (name in names.eyeLook) {
        eyeLook.add(blendshape(frame, name))
    }

    val euler = eulerFromColMajor(frame.m)

    return FrameSignals(
        tMs = frame.tMs,
        face = true,
        earL = eyeL.ear,
        earR = eyeR.ear,
        eyeWL = eyeL.widthPx,
        eyeWR = eyeR.widthPx,
        blinkL = blendshape(frame, names.eyeBlinkL),
        blinkR = blendshape(frame, names.eyeBlinkR),
        eyeLook = Collections.unmodifiableList(eyeLook),
        jawOpen = blendshape(frame, names.jawOpen),
        yaw = euler.yaw,
        pitch = euler.pitch,
        roll = euler.roll,
        irisHL = eyeL.irisH,
        irisHR = eyeR.irisH,
        irisVL = eyeL.irisV,
        irisVR = eyeR.irisV,
    )
}

/** signals.py:273-288 `_eye_signals`. */
private fun eyeSignals(
    frame: RawFrame,
    earIndices: List<Int>,
    outerIndex: Int,
    innerIndex: Int,
    irisCentreIndex: Int,
    minWidthPx: Double,
): EyeSignals {
    val pts = ArrayList<Point>(earIndices.size)
    for (index in earIndices) {
        pts.add(pixel(frame, index))
    }
    val width = distance(pts[0], pts[3])
    if (width < minWidthPx) {
        return EyeSignals(ear = 0.0, widthPx = width, irisH = null, irisV = null)
    }
    val ratio = irisRatio(pixel(frame, outerIndex), pixel(frame, innerIndex), pixel(frame, irisCentreIndex))
    return EyeSignals(ear = eyeAspectRatio(pts), widthPx = width, irisH = ratio.h, irisV = ratio.v)
}

/** signals.py:291-293 `_pixel`: (x·w, y·h). 없는 인덱스는 명시적 예외다(R17). */
private fun pixel(frame: RawFrame, index: Int): Point {
    val p = frame.lm[index] ?: throw IllegalArgumentException("frame has no landmark $index")
    return Point(p.x * frame.w, p.y * frame.h)
}

/** `frame.bs[name]`. 없는 이름은 명시적 예외다(R17, signals.py:248, 259-262의 KeyError). */
private fun blendshape(frame: RawFrame, name: String): Double =
    frame.bs[name] ?: throw IllegalArgumentException("frame has no blendshape '$name'")

/** signals.py:296-297 `_distance` = hypot(dx, dy). */
private fun distance(a: Point, b: Point): Double = StrictMath.hypot(a.x - b.x, a.y - b.y)
