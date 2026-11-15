package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.BaselineStatus
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.spec.ClosureSpec
import hawkeye.drowsy.core.spec.PipelineSpec
import kotlin.math.abs
import kotlin.math.ceil

// 원본: drowsy/calibrate.py:36-99 (분위수, 중앙값, 전역 기준선), 292-396 (eye_closure, frame_state, _clip01).
// SPEC 8.1, 8.4, 9장. 계획 5.3절의 선언 그대로다. 기준선 추정기(calibrate.py:102-284)는 BaselineEstimator.kt에 있다.
// 상태가 없다(R19).
//
// 가정(A1): 사칙연산, sqrt, floor, ceil, 비교는 JVM·Android와 CPython에서 같은 double을 낸다(곱셈과 덧셈을 합치지 않는다).
// 분위수 순위 ceil(q·n), eyeClosure의 식, closure의 가중 평균 (c_near + w_far·c_far) / (1 + w_far), gaze·jaw의 뺄셈은
// 이 가정 아래에서 Python과 비트가 같아야 하는 엄격 층의 값이다(계획 9.0절, 9.2.1절의 STATE 모드는 수로 같아야 한다).
// 삼각함수를 지나는 yaw_rel, pitch_rel과 그 뒤의 closure는 A2의 값이다(frameState의 주석).

// ---------------------------------------------------------------------------
// 분위수와 중앙값
// ---------------------------------------------------------------------------

/**
 * Python `sorted`와 같은 순서(R7): `<`만 쓰는 비교자로 복사본을 안정 정렬한다. -0.0과 0.0은 같은 것으로 보아 도착
 * 순서를 지킨다. 그래서 중앙값의 0 부호까지 Python과 같다. `DoubleArray.sort()`는 -0.0을 앞에 두므로 쓰지 않는다.
 * 가정(A4): `sortedWith`(객체 목록의 정렬)는 안정하다(KotlinSemanticsTest가 확인한다).
 * 가정: 값에 NaN이 없다. NaN이 있으면 이 비교자는 일관되지 않다. Python의 sorted는 예외 없이 어떤 순서를 내지만, Java의
 * TimSort는 다른 순서를 내거나 "IllegalArgumentException: Comparison method violates its general contract!"를 던진다.
 * 잰 사실(f_s2, Python 3.11 대 JDK 17.0.21): 원소가 32개 미만이면 결과가 비트까지 같았다(Java가 그 길이에서는 이진 삽입
 * 정렬만 한다). 32개 이상이면 NaN 하나로도 갈린다: n = 32에 NaN 3개는 Kotlin이 예외를 던지고 Python은 값을 냈고, n = 40에
 * NaN 1개의 0.9 분위수는 서로 다른 값이었다. BaselineEstimator에서는 600개 얼굴 프레임 가운데 blink_l 하나가 NaN인 흐름이
 * 마감 프레임(60000 ms)에서 이 예외로 멈췄고, Python은 status ok 기준선을 냈다. signals.compute는 NaN 블렌드셰이프 점수와
 * NaN 행렬 원소를 그대로 blink_*와 yaw로 옮기므로, NaN이 기준선 표본에 들어가지 않게 막는 것은 호출자의 일이다
 * (SPEC 2장, K9 (가): 유한하지 않은 MediaPipe 결과는 넣지 않는다). 이 차이를 그대로 둘지, 입구에서 거를지는 소유자와 lead의
 * 결정을 기다린다(f_s2 open issue).
 */
private val PYTHON_ORDER: Comparator<Double> = Comparator { a, b -> if (a < b) -1 else if (b < a) 1 else 0 }

/**
 * calibrate.py:36-53. 최근접 순위 분위수 `sorted[ceil(q·n) − 1]` (SPEC 8.1).
 * q·n은 Double 곱 그대로 올림하고(R5), 순위를 1 이상 n 이하로 자른다(두 자르기를 그대로).
 * 빈 입력은 IllegalArgumentException(Python은 ValueError).
 */
internal fun nearestRankQuantile(values: List<Double>, q: Double): Double {
    val n = values.size
    if (n == 0) {
        throw IllegalArgumentException("quantile of an empty sequence is undefined")
    }
    val ordered = values.sortedWith(PYTHON_ORDER)
    // Python `int(math.ceil(q * n))`. q·n이 Int 범위를 넘으면 toInt()가 포화하지만 바로 아래 자르기가 같은 순위를 낸다.
    var rank = ceil(q * n.toDouble()).toInt()
    if (rank < 1) {
        rank = 1
    }
    if (rank > n) {
        rank = n
    }
    return ordered[rank - 1]
}

/**
 * calibrate.py:56-69. 중앙값. 표본이 짝수 개면 가운데 두 값의 평균이다 (SPEC 8.1).
 * 빈 입력은 IllegalArgumentException(Python은 ValueError).
 */
internal fun median(values: List<Double>): Double {
    val n = values.size
    if (n == 0) {
        throw IllegalArgumentException("median of an empty sequence is undefined")
    }
    val ordered = values.sortedWith(PYTHON_ORDER)
    val mid = n / 2
    if (n % 2 == 1) {
        return ordered[mid]
    }
    return (ordered[mid - 1] + ordered[mid]) / 2.0
}

// ---------------------------------------------------------------------------
// 기준선
// ---------------------------------------------------------------------------

/**
 * calibrate.py:77-83. 전역 상수로 만든 기준선 (SPEC 8.4). 두 눈에 같은 E, B, R0은 spec의 r0,
 * status는 fallback_global, tCompleteMs는 0이다.
 */
internal fun globalBaseline(spec: PipelineSpec): Baseline = baselineFromConstants(spec, 0L)

/** calibrate.py:86-99 `_baseline_from_constants`. BaselineEstimator의 대체(fallback_global)도 이것을 쓴다. */
internal fun baselineFromConstants(spec: PipelineSpec, tCompleteMs: Long): Baseline {
    val constants = spec.globalConstants
    return Baseline(
        eL = constants.e,
        eR = constants.e,
        bL = constants.b,
        bR = constants.b,
        r0 = constants.r0,
        jaw0 = constants.jaw0,
        gazeH0 = constants.gazeH0,
        gazeV0 = constants.gazeV0,
        tCompleteMs = tCompleteMs,
        status = BaselineStatus.FALLBACK_GLOBAL,
    )
}

// ---------------------------------------------------------------------------
// closure와 프레임 상태
// ---------------------------------------------------------------------------

/**
 * calibrate.py:292-309. 눈 하나의 감김 정도 c_e (0~1, SPEC 9장).
 *
 * a_e = clip((E − ear) / (E·(1 − rho)), 0, 1), E·(1 − rho)가 0 이하이면 a_e = 0
 * s_e = clip((blink − B) / max(b_closed − B, b_span_min), 0, 1)
 * c_e = w·a_e + (1 − w)·s_e
 * max는 Python의 `if` 그대로 옮겼다(R6).
 */
internal fun eyeClosure(ear: Double, blink: Double, e: Double, b: Double, cfg: ClosureSpec): Double {
    var a = 0.0
    val earSpan = e * (1.0 - cfg.rho)
    if (earSpan > 0.0) {
        a = clip01((e - ear) / earSpan)
    }
    var blinkSpan = cfg.bClosed - b
    if (blinkSpan < cfg.bSpanMin) {
        blinkSpan = cfg.bSpanMin
    }
    val s = clip01((blink - b) / blinkSpan)
    return cfg.w * a + (1.0 - cfg.w) * s
}

/**
 * calibrate.py:358의 유효 조건을 순수 함수로 뺀 것(계획 5.3절): 얼굴 있음, 쓸 수 있는 눈이 하나 이상, |yaw_rel| <= 40도
 * (`closure.valid_max_abs_yaw_deg`). |yaw_rel|이 정확히 40도인 경우를 삼각함수 없이 시험하려고 뺐다. frameState는 얼굴이
 * 있을 때만 이것을 부른다. 같은 yaw_rel이 들어오면 판정은 Python과 같다(비교의 순서와 `<=`가 같다).
 *
 * 가정(A2): 그 앞의 yaw_rel은 atan2, sin, cos를 지나 마지막 자리가 Python과 다를 수 있고, 값이 정확히 40도 위에 있으면
 * 판정이 갈린다(걸침 T4, 계획 9.2.5절). 잰 반례(f_s2): 기준 회전이 단위 행렬이고 yaw ±40.0, pitch −7.5, roll 3.25인
 * 프레임에서 sin과 cos는 두 쪽이 같은 비트인데, JVM의 atan2(StrictMath와 Math 모두, JDK 17.0.21)·RAD_TO_DEG는
 * ±0x1.4000000000001p5를, CPython 3.11은 ±0x1.4p+5(정확히 40.0)를 낸다. 그래서 Kotlin frameState는 closureValid false와
 * closure 0.0을, Python frame_state는 closure_valid True와 closure 1.0을 낸다. pitch 0, roll 0(tests/test_calibrate.py:715-720의
 * 입력)에서는 두 쪽 모두 정확히 40.0이다. K20 (가)의 StrictMath로도 없앨 수 없는 마지막 자리 차이이고, 계산은 바꾸지 않는다.
 */
internal fun closureValid(face: Boolean, usableEye: Boolean, yawRel: Double, cfg: ClosureSpec): Boolean =
    face && usableEye && abs(yawRel) <= cfg.validMaxAbsYawDeg

/**
 * calibrate.py:312-388. 프레임 하나의 정규화 상태 (SPEC 9장).
 *
 * (yaw_rel, pitch_rel, roll_rel) = Euler(R0ᵀ·R), R은 (yaw, pitch, roll)에서 다시 만든다.
 * w_far = clip((25 − |yaw_rel|) / 10, 0, 1), 먼 쪽 눈은 yaw_rel > 0이면 L, 아니면 R(정확히 0이면 R).
 * closure = (c_near + w_far·c_far) / (1 + w_far). 쓸 수 있는 눈이 하나뿐이면 그 눈의 c_e 그대로.
 * 유효하지 않으면 closure = 0.0, closureValid = false. poseValid = 얼굴 있음이고 기준선이 fallback_global이 아님.
 * provisional은 받은 값 그대로다.
 */
internal fun frameState(s: FrameSignals, b: Baseline, spec: PipelineSpec, provisional: Boolean): FrameState {
    if (!s.face) {
        return FrameState(
            tMs = s.tMs,
            face = false,
            closure = 0.0,
            closureValid = false,
            provisional = provisional,
            yawRel = 0.0,
            pitchRel = 0.0,
            poseValid = false,
            gazeHRel = 0.0,
            gazeVRel = 0.0,
            jawRel = 0.0,
        )
    }

    // 가정(A2): yaw_rel, pitch_rel은 sin, cos, atan2를 지나 Python과 마지막 자리가 다를 수 있다. 그 값이 임계값 위에 정확히
    // 있으면 아래 판정(closureValid의 40도, 걸침 T4)이 갈린다. 잰 반례는 closureValid의 주석.
    val rotation = rotationFromEuler(s.yaw, s.pitch, s.roll)
    val relative = eulerFromRotation(relativeRotation(rotation, b.r0))
    val yawRel = relative.yaw
    val pitchRel = relative.pitch
    val absYawRel = abs(yawRel)

    val cfg = spec.closure
    val usableL = s.eyeWL >= spec.signals.minEyeWidthPx
    val usableR = s.eyeWR >= spec.signals.minEyeWidthPx

    var closure = 0.0
    var closureValid = false
    if (closureValid(true, usableL || usableR, yawRel, cfg)) {
        closureValid = true
        if (usableL && usableR) {
            val cL = eyeClosure(s.earL, s.blinkL, b.eL, b.bL, cfg)
            val cR = eyeClosure(s.earR, s.blinkR, b.eR, b.bR, cfg)
            val wFar = clip01((cfg.farEyeZeroYawDeg - absYawRel) / cfg.farEyeRampDeg)
            val cFar: Double
            val cNear: Double
            if (yawRel > 0.0) {
                cFar = cL
                cNear = cR
            } else {
                cFar = cR
                cNear = cL
            }
            closure = (cNear + wFar * cFar) / (1.0 + wFar)
        } else if (usableL) {
            closure = eyeClosure(s.earL, s.blinkL, b.eL, b.bL, cfg)
        } else {
            closure = eyeClosure(s.earR, s.blinkR, b.eR, b.bR, cfg)
        }
    }

    return FrameState(
        tMs = s.tMs,
        face = true,
        closure = closure,
        closureValid = closureValid,
        provisional = provisional,
        yawRel = yawRel,
        pitchRel = pitchRel,
        poseValid = b.status != BaselineStatus.FALLBACK_GLOBAL,
        gazeHRel = gazeH(s.eyeLook) - b.gazeH0,
        gazeVRel = gazeV(s.eyeLook) - b.gazeV0,
        jawRel = s.jawOpen - b.jaw0,
    )
}

/** calibrate.py:391-396 `_clip01`. Python의 `if` 그대로다(R6: coerceIn을 쓰지 않는다). */
internal fun clip01(x: Double): Double {
    if (x < 0.0) {
        return 0.0
    }
    if (x > 1.0) {
        return 1.0
    }
    return x
}
