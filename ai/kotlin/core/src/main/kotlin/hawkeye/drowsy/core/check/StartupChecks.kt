package hawkeye.drowsy.core.check

import hawkeye.drowsy.core.CLASS_ORDER
import hawkeye.drowsy.core.N_FEATURES
import hawkeye.drowsy.core.spec.ConstantBits
import hawkeye.drowsy.core.spec.FeatureSpec
import hawkeye.drowsy.core.spec.RulesConfig
import hawkeye.drowsy.core.spec.SpecException
import hawkeye.drowsy.core.spec.SpecLoader
import java.security.MessageDigest

// 시작 검사 (SPEC 15장, 계획 4.1절, 5.11절). 새 코드다: Python에는 앱의 시작 검사가 없고, 기대 코드는 계획 9.4절의 표다.
// 순수 함수다. 파일을 열지 않고, 호출자가 읽은 바이트와 모델에서 읽은 값을 받는다. 바이트를 먼저 해시하고 그다음에 읽는다.
// 문제를 하나에서 멈추지 않고 모두 모은다. 문자열은 있는 그대로 비교한다(대문자 해시, 뒤 공백, "1.0"은 다르다, SPEC 15장).

/** 시작 검사 하나의 종류 (계획 4.1절, 5.11절). */
enum class StartupCheck {
    FILE_MISSING, SPEC_INVALID, RULES_INVALID, MODEL_UNREADABLE, MODEL_IO_CONTRACT, METADATA_KEY_MISSING,
    H1_TASK_FILE_VS_METADATA, H1_SPEC_VS_METADATA, H2_SPEC_FILE_VS_METADATA,
    H3_SPEC_VS_METADATA, H3_SPEC_VS_VERIFIED, H4_RULES_FILE_VS_VERIFIED,
    H5_STEP_SEC, H5_WINDOW_STEPS, H5_CLASS_ORDER, H5_STANDARDIZATION,
    CLOCK_SPEC_VS_RULES, MODEL_VERSION_VS_FILE_NAME, MODEL_FILE_VS_PIN, CONSTANT_BITS_VS_VERIFIED,
}

/** 문제 하나. subject는 파일 이름이나 메타데이터 키, expected·actual은 비교한 두 값(없으면 null)이다. */
data class StartupProblem(val check: StartupCheck, val subject: String, val expected: String?, val actual: String?) {
    /** 영어 한 줄. 기대값과 실제값을 담는다(계획 9.4절). */
    val message: String
        get() = "$check: $subject (expected ${quote(expected)}, actual ${quote(actual)})"

    private fun quote(s: String?): String = if (s == null) "none" else "\"$s\""
}

/**
 * 빌드가 `:core:test`의 출력에서 넣는 검증 값 (계획 3.5절, K11 (가)).
 * pipelineHash, rulesHash: 벡터가 통과한 값. vectorSet: 벡터 묶음 지문, vectorCount: 벡터 수, constantBits: 상수 비트 지문.
 */
data class VerifiedValues(
    val pipelineHash: String,
    val rulesHash: String,
    val vectorSet: String,
    val vectorCount: Int,
    val constantBits: String,
)

/** ONNX 세션에서 읽은 입출력 이름과 모양. */
data class ModelIo(val inputName: String, val inputShape: List<Long>, val outputName: String, val outputShape: List<Long>)

/**
 * 시작 검사의 입력.
 * - modelFileSha256: 모델 파일 바이트의 sha256. null = 파일 없음(FILE_MISSING). 계획 4.1절은 String이지만, 파일이 없을 때와
 *   열지 못할 때를 가르려고 null을 받는다(보고한 계획과의 차이).
 * - modelMetadata, modelIo: null = 모델을 열지 못함(MODEL_UNREADABLE). 파일이 없으면 보지 않는다.
 * - specBytes, rulesBytes: null = 파일 없음. taskSha256: face_landmarker.task 바이트의 sha256, null = 파일 없음.
 */
class StartupInput(
    val modelFileName: String,
    val modelFileSha256: String?,
    val modelSha256Pin: String?,
    val modelMetadata: Map<String, String>?,
    val modelIo: ModelIo?,
    val specFileName: String,
    val specBytes: ByteArray?,
    val rulesBytes: ByteArray?,
    val taskSha256: String?,
    val verified: VerifiedValues,
)

/** StartupChecks.run만 만든다. spec, rules는 읽기와 검사를 통과했으면 null이 아니다(다른 문제가 있어도). */
class StartupResult internal constructor(
    val problems: List<StartupProblem>,
    val spec: FeatureSpec?,
    val rules: RulesConfig?,
) {
    /** problems가 비었다. */
    val passed: Boolean
        get() = problems.isEmpty()
}

object StartupChecks {

    /** 번들 안의 이름 (계획 7.1절). */
    private const val TASK_FILE_NAME: String = "face_landmarker.task"
    private const val RULES_FILE_NAME: String = "rules.json"

    // 메타데이터 키 (SPEC 14.3). 비교하는 여덟 키, 빠진 것을 이 순서로 보고한다(계획 9.4절 SB8).
    private const val K_MEDIAPIPE = "mediapipe_model_sha256"
    private const val K_FEATURE_SPEC_HASH = "feature_spec_hash"
    private const val K_PIPELINE_HASH = "pipeline_hash"
    private const val K_STEP_SEC = "step_sec"
    private const val K_WINDOW_STEPS = "window_steps"
    private const val K_CLASS_ORDER = "class_order"
    private const val K_STANDARDIZATION = "standardization"
    private const val K_MODEL_VERSION = "model_version"
    private val COMPARED_KEYS = listOf(
        K_MEDIAPIPE, K_FEATURE_SPEC_HASH, K_PIPELINE_HASH, K_STEP_SEC, K_WINDOW_STEPS, K_CLASS_ORDER,
        K_STANDARDIZATION, K_MODEL_VERSION,
    )

    // ONNX 계약 (SPEC 14.1): 입력 이름 features, 출력 이름 probs.
    private const val INPUT_NAME = "features"
    private const val OUTPUT_NAME = "probs"

    // 계획 7.2절: drowsiness_v{N}.onnx의 N.
    private val MODEL_FILE = Regex("^drowsiness_v(0|[1-9][0-9]*)\\.onnx$")

    /** 순수 함수. 바이트를 먼저 해시하고 그다음에 읽는다. 문제를 하나에서 멈추지 않고 모두 모은다. */
    @JvmStatic
    fun run(input: StartupInput): StartupResult {
        val problems = ArrayList<StartupProblem>()
        fun add(check: StartupCheck, subject: String, expected: String?, actual: String?) {
            problems.add(StartupProblem(check, subject, expected, actual))
        }

        // 1. 해시가 먼저다. 읽히지 않는 파일도 해시 비교가 보고된다(계획 5.11절).
        val specSha = input.specBytes?.let { sha256Hex(it) }
        val rulesSha = input.rulesBytes?.let { sha256Hex(it) }

        // 2. 없는 파일.
        if (input.taskSha256 == null) add(StartupCheck.FILE_MISSING, TASK_FILE_NAME, null, null)
        if (input.modelFileSha256 == null) add(StartupCheck.FILE_MISSING, input.modelFileName, null, null)
        if (input.specBytes == null) add(StartupCheck.FILE_MISSING, input.specFileName, null, null)
        if (input.rulesBytes == null) add(StartupCheck.FILE_MISSING, RULES_FILE_NAME, null, null)

        // 3. 같은 바이트로 읽기와 검사.
        var spec: FeatureSpec? = null
        var specTree: Map<String, Any?>? = null
        if (input.specBytes != null) {
            try {
                val read = SpecLoader.readFeatureSpec(input.specBytes)
                spec = read.first
                specTree = read.second
            } catch (e: SpecException) {
                add(StartupCheck.SPEC_INVALID, input.specFileName, null, e.message)
            }
        }
        var rules: RulesConfig? = null
        var rulesTree: Map<String, Any?>? = null
        if (input.rulesBytes != null) {
            try {
                val read = SpecLoader.readRules(input.rulesBytes)
                rules = read.first
                rulesTree = read.second
            } catch (e: SpecException) {
                add(StartupCheck.RULES_INVALID, RULES_FILE_NAME, null, e.message)
            }
        }

        // 4. 모델을 열었는지. 파일이 없으면 FILE_MISSING만 보고한다.
        val modelPresent = input.modelFileSha256 != null
        val metadata = if (modelPresent) input.modelMetadata else null
        val io = if (modelPresent) input.modelIo else null
        if (modelPresent && (input.modelMetadata == null || input.modelIo == null)) {
            add(StartupCheck.MODEL_UNREADABLE, input.modelFileName, null, null)
        }

        // 5. 메타데이터와의 비교 (H1, H2, H3, H5, 모델 버전). 빠진 키는 값이 다른 것과 구별해 보고하고 그 비교는 하지 않는다.
        if (metadata != null) {
            for (key in COMPARED_KEYS) {
                if (!metadata.containsKey(key)) add(StartupCheck.METADATA_KEY_MISSING, key, null, null)
            }
            val mediapipe = metadata[K_MEDIAPIPE]
            if (mediapipe != null && input.taskSha256 != null && input.taskSha256 != mediapipe) {
                add(StartupCheck.H1_TASK_FILE_VS_METADATA, TASK_FILE_NAME, mediapipe, input.taskSha256)
            }
            if (mediapipe != null && spec != null && spec.pipeline.mediapipe.modelSha256 != mediapipe) {
                add(StartupCheck.H1_SPEC_VS_METADATA, "pipeline.mediapipe.model_sha256", mediapipe, spec.pipeline.mediapipe.modelSha256)
            }
            val featureSpecHash = metadata[K_FEATURE_SPEC_HASH]
            if (featureSpecHash != null && specSha != null && specSha != featureSpecHash) {
                add(StartupCheck.H2_SPEC_FILE_VS_METADATA, input.specFileName, featureSpecHash, specSha)
            }
            val pipelineHash = metadata[K_PIPELINE_HASH]
            if (pipelineHash != null && spec != null && spec.pipelineHash != pipelineHash) {
                add(StartupCheck.H3_SPEC_VS_METADATA, "pipeline_hash", pipelineHash, spec.pipelineHash)
            }
            // H5: 메타데이터 값 = 번들한 spec의 값(SPEC 14.3절의 값은 읽기 검사 F2, F3, F4, F6과 H3가 묶는다, 계획 5.11절).
            if (spec != null) {
                compareH5(metadata, K_STEP_SEC, spec.stepSec.toString(), StartupCheck.H5_STEP_SEC, problems)
                compareH5(metadata, K_WINDOW_STEPS, spec.windowSteps.toString(), StartupCheck.H5_WINDOW_STEPS, problems)
                compareH5(metadata, K_CLASS_ORDER, spec.classOrder.joinToString(","), StartupCheck.H5_CLASS_ORDER, problems)
                compareH5(
                    metadata, K_STANDARDIZATION, spec.standardization.location, StartupCheck.H5_STANDARDIZATION, problems,
                )
            }
            val modelVersion = metadata[K_MODEL_VERSION]
            if (modelVersion != null) {
                val n = MODEL_FILE.matchEntire(input.modelFileName)?.groupValues?.get(1)
                if (n != modelVersion) {
                    add(StartupCheck.MODEL_VERSION_VS_FILE_NAME, input.modelFileName, modelVersion, n ?: input.modelFileName)
                }
            }
        }

        // 6. 빌드에 넣은 검증 값과의 비교 (H3, H4).
        if (spec != null && spec.pipelineHash != input.verified.pipelineHash) {
            add(StartupCheck.H3_SPEC_VS_VERIFIED, "pipeline_hash", input.verified.pipelineHash, spec.pipelineHash)
        }
        if (rulesSha != null && rulesSha != input.verified.rulesHash) {
            add(StartupCheck.H4_RULES_FILE_VS_VERIFIED, RULES_FILE_NAME, input.verified.rulesHash, rulesSha)
        }

        // 7. SPEC 7.1: rules.clock == pipeline.clock.
        if (spec != null && rules != null && spec.pipeline.clock != rules.clock) {
            add(StartupCheck.CLOCK_SPEC_VS_RULES, "clock", spec.pipeline.clock.toString(), rules.clock.toString())
        }

        // 8. ONNX 입출력 계약: 입력 features [1, window_steps, 17], 출력 probs [1, 3]. spec을 읽지 못했으면 window_steps
        // 칸은 보지 않는다(그 값은 SPEC_INVALID로 이미 보고된다).
        if (io != null) {
            val shapeOk = io.inputShape.size == 3 && io.inputShape[0] == 1L &&
                (spec == null || io.inputShape[1] == spec.windowSteps.toLong()) &&
                io.inputShape[2] == N_FEATURES.toLong()
            if (io.inputName != INPUT_NAME || !shapeOk) {
                val want = "[1, ${spec?.windowSteps ?: "window_steps"}, $N_FEATURES]"
                add(StartupCheck.MODEL_IO_CONTRACT, "input", "$INPUT_NAME $want", "${io.inputName} ${io.inputShape}")
            }
            val outOk = io.outputShape == listOf(1L, CLASS_ORDER.size.toLong())
            if (io.outputName != OUTPUT_NAME || !outOk) {
                add(StartupCheck.MODEL_IO_CONTRACT, "output", "$OUTPUT_NAME [1, ${CLASS_ORDER.size}]", "${io.outputName} ${io.outputShape}")
            }
        }

        // 9. 앱이 준 고정값 (K12 (가)).
        if (input.modelSha256Pin != null && input.modelFileSha256 != null && input.modelSha256Pin != input.modelFileSha256) {
            add(StartupCheck.MODEL_FILE_VS_PIN, input.modelFileName, input.modelSha256Pin, input.modelFileSha256)
        }

        // 10. 상수 비트 지문: 두 파일을 모두 읽었을 때만.
        if (specTree != null && rulesTree != null) {
            val bits = ConstantBits.fingerprint(specTree["pipeline"], rulesTree)
            if (bits != input.verified.constantBits) {
                add(StartupCheck.CONSTANT_BITS_VS_VERIFIED, "constants", input.verified.constantBits, bits)
            }
        }

        return StartupResult(java.util.Collections.unmodifiableList(problems), spec, rules)
    }

    private fun compareH5(
        metadata: Map<String, String>,
        key: String,
        specValue: String,
        check: StartupCheck,
        problems: MutableList<StartupProblem>,
    ) {
        val meta = metadata[key] ?: return
        if (meta != specValue) problems.add(StartupProblem(check, key, specValue, meta))
    }
}

/** 소문자 16진수 64자 (java.util.HexFormat을 쓰지 않는다: Android API 34부터다, R23). */
fun sha256Hex(bytes: ByteArray): String {
    val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
    val hex = "0123456789abcdef"
    val sb = StringBuilder(64)
    for (b in digest) {
        val v = b.toInt() and 0xFF
        sb.append(hex[v ushr 4]).append(hex[v and 0xF])
    }
    return sb.toString()
}
