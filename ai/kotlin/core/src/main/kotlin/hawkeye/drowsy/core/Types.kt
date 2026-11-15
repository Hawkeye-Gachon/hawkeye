package hawkeye.drowsy.core

// 원본: drowsy/types.py:146-374, drowsy/calibrate.py:102-115. SPEC 3장.
// 모든 형은 불변 값이다(Python의 frozen dataclass). 필드 순서는 Python 필드 순서와 같다(위치 인자와 구조 분해가
// 그 순서에 기댄다). 시각과 길이는 모두 Long이고 원래 t_ms 시계다(R1). 세션 시간으로 적는 것은
// Baseline.tCompleteMs와 StepFeatures.k 둘뿐이다.
// 실수 묶음은 List<Double>이다. 배열을 넣으면 data class의 equals가 참조 비교가 된다(R9).
// data class의 equals는 Double을 비트에 가깝게 비교한다(-0.0 != 0.0, NaN == NaN). 그래서 출력 비교는 객체 전체의
// `==`가 아니라 필드별로 한다(R9, port_map B18.3).
// FrameSignals(types.py:175-210)는 이식 안에서만 쓰는 형이라 internal/Signals.kt에 둔다(계획 5.2절, 단계 2).

/** 정규화 좌표 한 점 (types.py:170의 `tuple[float, float]`). */
data class Point(val x: Double, val y: Double)

/**
 * 앱이 MediaPipe FaceLandmarker에서 프레임 하나에 대해 받는 것 (types.py:146-172, SPEC 3.1).
 *
 * - lm: 랜드마크 인덱스 → 정규화 좌표. 필요한 44개를 담는다. 더 있어도 된다.
 * - bs: 블렌드셰이프 이름 → 값. 필요한 11개는 반드시 있어야 한다.
 * - m: 얼굴 변환 행렬 4x4의 열 우선 16개 값. 얼굴이 없으면 lm, bs, m이 모두 비어 있다.
 */
data class RawFrame(
    val tMs: Long,
    val face: Boolean,
    val w: Int,
    val h: Int,
    val lm: Map<Int, Point>,
    val bs: Map<String, Double>,
    val m: List<Double>,
)

/**
 * 개인 기준선 (types.py:213-240, SPEC 3.3). r0은 행 우선 9개(`R0[i][j] = r0[3*i + j]`).
 * tCompleteMs는 세션 시간이다. 전역 상수로 바로 만든 기준선은 0이다.
 */
data class Baseline(
    val eL: Double,
    val eR: Double,
    val bL: Double,
    val bR: Double,
    val r0: List<Double>,
    val jaw0: Double,
    val gazeH0: Double,
    val gazeV0: Double,
    val tCompleteMs: Long,
    val status: BaselineStatus,
)

/** 기준선으로 정규화한 프레임 상태 (types.py:243-278, SPEC 3.4). */
data class FrameState(
    val tMs: Long,
    val face: Boolean,
    val closure: Double,
    val closureValid: Boolean,
    val provisional: Boolean,
    val yawRel: Double,
    val pitchRel: Double,
    val poseValid: Boolean,
    val gazeHRel: Double,
    val gazeVRel: Double,
    val jawRel: Double,
)

/** 감김 구간 하나 (types.py:281-306, SPEC 3.5). */
data class ClosureRun(
    val startMs: Long,
    val endMs: Long,
    val durationMs: Long,
    val amplitude: Double,
    val peakClosingVelocity: Double,
    val peakOpeningVelocity: Double,
    val velocityValid: Boolean,
    val truncated: Boolean,
)

/** 1초 스텝 하나의 특징 (types.py:309-324, SPEC 3.6). values는 FEATURE_NAMES 순서의 원시 특징 17개다. */
data class StepFeatures(
    val k: Int,
    val tEndMs: Long,
    val fullLookback: Boolean,
    val values: List<Double>,
)

/** 운전자 상태 이벤트 (types.py:327-348, SPEC 3.7). */
data class DriverStateEvent(
    val type: EventType,
    val severity: Int,
    val tStartMs: Long,
    val durationMs: Long,
    val confidence: Double,
    val source: Source,
    val phase: Phase,
    val provisional: Boolean,
)

/** 모델 확률 셋 (types.py:371-372의 `tuple[float, float, float]`). 성분 순서는 CLASS_ORDER와 같다. */
data class ClassProbs(val alert: Double, val lowVigilant: Double, val drowsy: Double)

/**
 * `DriverPipeline.update`가 프레임 하나에 대해 돌려주는 것 (types.py:351-374, SPEC 3.8).
 *
 * - runs: 이 프레임에서 끝난 특징용 감김 구간(0개 또는 1개).
 * - steps: 이 프레임이 발행시킨 스텝, 오래된 것부터. stepProbs는 steps와 길이와 순서가 같다.
 * - probs: 스텝이 있으면 stepProbs의 마지막 원소, 없으면 null.
 * - events: 발행 순서. alerts: AlertPolicy가 경고하기로 한 이벤트, 우선순위 순서.
 */
data class PipelineOutput(
    val state: FrameState,
    val runs: List<ClosureRun>,
    val steps: List<StepFeatures>,
    val stepProbs: List<ClassProbs?>,
    val probs: ClassProbs?,
    val events: List<DriverStateEvent>,
    val alerts: List<DriverStateEvent>,
)

/**
 * 기준선을 확정한 시점의 판정 통계와 분포 기록 (calibrate.py:102-115). Baseline에는 넣지 않는 값이다.
 * DriverState에도 넣지 않는다(K21 (가)).
 *
 * - blinkLQuantiles, blinkRQuantiles: 순서는 spec.baseline.blink_record_quantiles와 같다. 얼굴 프레임이 없으면 0.0이다.
 */
data class BaselineDiagnostics(
    val nFaceFrames: Int,
    val faceRatio: Double,
    val blinkLQuantiles: List<Double>,
    val blinkRQuantiles: List<Double>,
)
