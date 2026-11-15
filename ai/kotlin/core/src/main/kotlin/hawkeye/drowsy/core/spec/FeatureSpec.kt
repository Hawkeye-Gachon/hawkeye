package hawkeye.drowsy.core.spec

// 원본: drowsy/spec.py:287-344 (feature_spec.json 전체). 계획 4.1절의 선언 그대로다. 필드 순서는 Python과 같다.
// index, windowSec, specVersion, stepSec, windowSteps는 Int이고 읽을 때 범위를 검사한다(R22).

/** 스텝 특징 하나 (spec.py:287-299). */
data class FeatureDef(
    val index: Int,
    val name: String,
    val windowSec: Int,
    val unit: String,
    val formula: String,
    val undefinedValue: Double,
    val transform: String,
    val mean: Double,
    val std: Double,
)

/** 그래프 안 전처리 상수 (spec.py:302-309). */
data class Standardization(val location: String, val zClip: Double, val stdFloor: Double, val temperature: Double)

/**
 * feature_spec.json 전체 (spec.py:312-344).
 * pipelineHash는 파일에 적힌 문자열 그대로다. Kotlin은 다시 계산하지 않는다(SPEC 15장, 계획 5.10절).
 */
data class FeatureSpec(
    val specVersion: Int,
    val stepSec: Int,
    val windowSteps: Int,
    val classOrder: List<String>,
    val features: List<FeatureDef>,
    val standardization: Standardization,
    val pipeline: PipelineSpec,
    val pipelineHash: String,
)
