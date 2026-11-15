package hawkeye.drowsy.core.spec

// 원본: drowsy/spec.py:69-279 (feature_spec.json의 `pipeline` 블록). 계획 4.1절의 선언 그대로다.
// 필드 순서는 Python dataclass의 필드 순서와 같다. 읽기(SpecLoader)가 이 순서로 필드를 바꾸고, 오류가 나면 Python과
// 같은 자리에서 같은 문장으로 멈춘다.
// 시각과 길이(*_ms)는 Long이다(R1). Int인 칸(랜드마크 인덱스, min_face_frames, velocity_lookback_samples, window_steps,
// num_faces)은 읽을 때 범위를 검사해 줄인다(R22).
// 실수 묶음은 List<Double>이다(R9). 앱이 copy(...)로 검사받지 않은 값을 만들 수는 있지만, 그 값으로 파이프라인을 만드는
// 공개 길이 없다(계획 4.1절).

/** 랜드마크 인덱스 집합 (spec.py:69-88). R은 MediaPipe RIGHT_EYE, L은 LEFT_EYE다. */
data class LandmarkSpec(
    val earR: List<Int>,
    val earL: List<Int>,
    val irisR: List<Int>,
    val irisL: List<Int>,
    val cornerOuterR: Int,
    val cornerInnerR: Int,
    val cornerOuterL: Int,
    val cornerInnerL: Int,
    val contourR: List<Int>,
    val contourL: List<Int>,
    val faceWidth: List<Int>,
) {
    /**
     * RawFrame.lm에 들어 있어야 하는 인덱스(오름차순, 중복 없음). 검사를 통과한 spec에서는 44개다 (spec.py:85-88).
     * Python은 `sorted(set(contour_r) | set(contour_l) | set(iris_r) | set(iris_l) | set(face_width))`다.
     * 정수의 집합과 정렬이라 순회 순서가 결과에 들어가지 않는다.
     */
    fun requiredIndices(): List<Int> {
        val merged = java.util.TreeSet<Int>()
        merged.addAll(contourR)
        merged.addAll(contourL)
        merged.addAll(irisR)
        merged.addAll(irisL)
        merged.addAll(faceWidth)
        return java.util.Collections.unmodifiableList(ArrayList(merged))
    }
}

/** 블렌드셰이프 이름 (spec.py:91-107). RawFrame.bs는 category_name으로 찾는다. */
data class BlendshapeSpec(
    val eyeBlinkL: String,
    val eyeBlinkR: String,
    val eyeLook: List<String>,
    val jawOpen: String,
    val order: List<String>,
) {
    /** RawFrame.bs에 들어 있어야 하는 이름 11개: eye_blink_l, eye_blink_r, eye_look 8개, jaw_open (spec.py:101-103). */
    fun requiredNames(): List<String> {
        val out = ArrayList<String>(2 + eyeLook.size + 1)
        out.add(eyeBlinkL)
        out.add(eyeBlinkR)
        out.addAll(eyeLook)
        out.add(jawOpen)
        return java.util.Collections.unmodifiableList(out)
    }

    /**
     * MediaPipe 블렌드셰이프 인덱스 (spec.py:105-107 `self.order.index(name)`). 처음 나오는 자리를 돌려준다.
     * 없으면 IllegalArgumentException이다(Kotlin `indexOf`의 -1을 쓰지 않는다, 계획 4.1절).
     */
    fun indexOf(name: String): Int {
        for (i in order.indices) {
            if (order[i] == name) return i
        }
        throw IllegalArgumentException("blendshape name '$name' is not in pipeline.blendshapes.order")
    }
}

/**
 * 각도 규약 (spec.py:110-119). 계산식은 고정이고 이 값들은 규약을 해시에 묶어 두는 기록이다.
 * 가정: Python이 이 문자열들을 검사하지 않으므로 Kotlin도 검사하지 않는다. 판정이 Python과 같게 하려는 것이다
 * (리드 판정 2, 소유자 확인 대기).
 */
data class PoseSpec(
    val order: String,
    val yawPos: String,
    val pitchPos: String,
    val rollPos: String,
    val matrixLayout: String,
    val angleUnit: String,
)

/** 시간 규칙 (spec.py:122-128). */
data class ClockSpec(val stepMs: Long, val maxHoldMs: Long, val resetGapMs: Long)

/** 원시 신호 (spec.py:131-135). */
data class SignalSpec(val minEyeWidthPx: Double)

/** 기준선 (spec.py:138-155). */
data class BaselineSpec(
    val durationMs: Long,
    val extendStepMs: Long,
    val maxDurationMs: Long,
    val minFaceFrames: Int,
    val minFaceRatio: Double,
    val eMin: Double,
    val eMax: Double,
    val earQuantile: Double,
    val blinkQuantile: Double,
    val blinkRecordQuantiles: List<Double>,
    val offAxisYawDeg: Double,
)

/** closure 식의 상수 (spec.py:158-174). */
data class ClosureSpec(
    val rho: Double,
    val bClosed: Double,
    val bSpanMin: Double,
    val w: Double,
    val farEyeZeroYawDeg: Double,
    val farEyeRampDeg: Double,
    val validMaxAbsYawDeg: Double,
)

/** `ClosureRunDetector` 설정 (spec.py:177-191). 특징용(pipeline.run_detector)과 규칙용(closure.detector)이 같은 형이다. */
data class RunDetectorSpec(
    val enterThr: Double,
    val exitThr: Double,
    val minDurationMs: Long,
    val maxDtMs: Long,
    val holdMs: Long,
    val velocityLookbackSamples: Int,
)

/** 스텝 특징 상수 (spec.py:194-209). */
data class StepFeatureSpec(
    val names: List<String>,
    val windowSteps: Int,
    val lookbackLongMs: Long,
    val lookbackShortMs: Long,
    val perclosThr: Double,
    val perclosMinValidMs60: Long,
    val perclosMinValidMs10: Long,
    val blinkRateMinValidMs: Long,
    val blinkDurationCapMs: Long,
    val longClosureMinMs: Long,
    val maxClosureCapMs: Long,
    val pitchStdMinFaceMs: Long,
)

/** 하품 이벤트 (spec.py:212-219). */
data class YawnSpec(val enterThr: Double, val exitThr: Double, val countAtMs: Long, val cancelGapMs: Long)

/** 끄덕임 이벤트 (spec.py:222-236). */
data class NodSpec(
    val enterDeg: Double,
    val exitDeg: Double,
    val minDurationMs: Long,
    val maxDurationMs: Long,
    val emaTauMs: Long,
    val emaMaxDtMs: Long,
    val cancelGapMs: Long,
)

/** 전역 상수 (spec.py:239-252). r0은 행 우선 9개다. */
data class GlobalConstants(
    val e: Double,
    val b: Double,
    val jaw0: Double,
    val gazeH0: Double,
    val gazeV0: Double,
    val r0: List<Double>,
)

/** MediaPipe 전제 (spec.py:255-260). */
data class MediaPipeSpec(val modelSha256: String, val numFaces: Int)

/** feature_spec.json의 `pipeline` 블록: 이식 코드가 쓰는 상수 전부 (spec.py:263-279). */
data class PipelineSpec(
    val landmarks: LandmarkSpec,
    val blendshapes: BlendshapeSpec,
    val pose: PoseSpec,
    val clock: ClockSpec,
    val signals: SignalSpec,
    val baseline: BaselineSpec,
    val closure: ClosureSpec,
    val runDetector: RunDetectorSpec,
    val stepFeatures: StepFeatureSpec,
    val yawn: YawnSpec,
    val nod: NodSpec,
    val globalConstants: GlobalConstants,
    val mediapipe: MediaPipeSpec,
)
