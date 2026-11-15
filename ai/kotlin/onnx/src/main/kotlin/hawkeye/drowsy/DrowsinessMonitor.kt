package hawkeye.drowsy

import hawkeye.drowsy.core.DrowsyInternalApi
import hawkeye.drowsy.core.ModelFn
import hawkeye.drowsy.core.RawFrame
import hawkeye.drowsy.core.RawFrames
import hawkeye.drowsy.core.check.StartupChecks
import hawkeye.drowsy.core.check.StartupInput
import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.session.DriveSession
import hawkeye.drowsy.core.session.DriverState
import hawkeye.drowsy.core.session.PipelineCounters
import hawkeye.drowsy.core.spec.PipelineSpec
import hawkeye.drowsy.onnx.OnnxDrowsinessModel
import hawkeye.drowsy.onnx.OnnxModelInfo
import java.util.concurrent.locks.ReentrantLock

// 앱이 부르는 하나의 입구 (리드 판정 9의 2번). 계획 4.2절의 :android 입구(DriverMonitoring)를 대신한다: 앱이 카메라와
// MediaPipe를 가지고, 결과 하나마다 이 클래스의 update를 부른다. Android와 MediaPipe를 import하지 않는다.
// 하는 일은 판정 9 그대로다: 바이트의 sha256 → OnnxDrowsinessModel.inspect → StartupChecks.run → 통과했을 때만
// OnnxDrowsinessModel.open과 DriveSession.create. 그 뒤 프레임마다 RawFrames.of(또는 noFace) → DriveSession.feed.
// 모델은 언제나 ONNX 모델이다(판정 9: 모델 없는 모드가 없다). 지금 rules.json의 drowsiness.source는 perclos라 모델 확률은
// 판정에 쓰이지 않고 probs로만 나온다(v0 모델은 무작위 가중치다).

/**
 * 운전 한 번 = 인스턴스 하나. 스레드에 안전하지 않다(update, updateNoFace 호출이 겹치면 IllegalStateException).
 *
 * - close()가 진행 중인 update와 겹치면 그 update가 끝나기를 기다린 뒤 ONNX 세션을 닫는다(계획 5.8절 close: 하던 프레임을
 *   끝내고 닫는다). 그래서 프레임 하나가 반쯤 처리된 채 남지 않는다.
 * - 처리 오류에서 멈춘다(계획 5.8절 K8, 참조 drowsy/live.py:1527-1528, 1624-1631): 파이프라인이 예외를 내면 ONNX 세션을
 *   닫고 그 예외를 그대로 던진다. 그 뒤의 update와 updateNoFace는 IllegalStateException(cause = 그 예외)이다. 다시
 *   시작하려면 새 인스턴스를 연다(새 60초 보정). 인자 검사(RawFrames.of의 IllegalArgumentException)는 파이프라인에 들어가기
 *   전이라 상태를 바꾸지 않으므로 멈추지 않는다.
 *   가정: 참조는 MediaPipe 결과를 RawFrame으로 바꾸다 난 오류에서도 세션을 끝낸다(live.py:1625-1631의 detect). 이 입구에서 그
 *   변환은 앱의 일이고 RawFrames.of의 거부는 아무 상태도 바꾸지 않으므로, 그 거부에서는 멈추지 않는다(README의 차이 목록).
 */
@OptIn(DrowsyInternalApi::class)
class DrowsinessMonitor private constructor(private val parts: Parts) : java.io.Closeable {

    /**
     * 생성자가 받는 것. private 중첩 클래스라 클래스 밖에서는 이름을 쓸 수 없다. companion이 private 생성자를 부르므로 Kotlin이
     * 합성 생성자를 만드는데, 그 생성자의 인자가 이 형이라 공개 API 목록에 DriveSession이나 ONNX 형이 나오지 않는다
     * (OnnxDrowsinessModel.Ort와 같은 방식).
     */
    private class Parts(val session: DriveSession, val model: OnnxDrowsinessModel, val spec: PipelineSpec)

    // update, updateNoFace, close가 함께 쓰는 잠금. update는 tryLock(겹치면 거부), close는 lock(끝나기를 기다림).
    private val lock = ReentrantLock()

    // close()가 불린 순간 선다(계획 5.8절 close: 그 순간 닫힘 표시를 세우고 그 뒤의 새 프레임을 거부한다). 잠금 밖에서도 읽는다.
    @Volatile
    private var closeRequested = false

    // 아래 둘은 lock 안에서만 읽고 쓴다.
    private var modelClosed = false
    private var failure: Throwable? = null

    /**
     * 얼굴이 있는 프레임. 인자는 RawFrames.of와 같다. null = 파이프라인이 버린 프레임(tMs가 직전에 받아들인 값 이하).
     *
     * - tMs: 프레임 시각(ms). 늘 증가해야 한다(SPEC G8). 늘지 않으면 그 프레임은 버려지고 null이다.
     * - imageWidth, imageHeight: MediaPipe에 준, 똑바로 세운 그림의 크기(px).
     * - landmarkX, landmarkY: 얼굴 0의 모든 랜드마크(정규화 좌표, 478점 모델).
     * - blendshapeNames, blendshapeScores: 얼굴 0의 블렌드셰이프 이름과 점수.
     * - matrix: 얼굴 0의 변환 행렬 16개, MediaPipe가 준 순서 그대로.
     * - speedKmh: 그 순간의 속도(km/h). 모르면 null.
     *
     * 닫은 뒤, 또는 처리 오류로 멈춘 뒤에는 IllegalStateException. 필요한 랜드마크·블렌드셰이프가 없거나 행렬이 16개가
     * 아니면 IllegalArgumentException(상태는 그대로이고 멈추지 않는다). 파이프라인이 낸 예외는 그대로 던지고 멈춘다.
     * 가정: 유한하지 않은 값(NaN, 무한대)은 검사하지 않는다(리드 판정 8(b), 8(e)). 호출자가 넣지 않는다(SPEC 2장).
     */
    fun update(
        tMs: Long,
        imageWidth: Int,
        imageHeight: Int,
        landmarkX: FloatArray,
        landmarkY: FloatArray,
        blendshapeNames: List<String>,
        blendshapeScores: FloatArray,
        matrix: FloatArray,
        speedKmh: Double?,
    ): DriverState? = entered {
        val frame = RawFrames.of(
            tMs = tMs,
            w = imageWidth,
            h = imageHeight,
            landmarkX = landmarkX,
            landmarkY = landmarkY,
            blendshapeNames = blendshapeNames,
            blendshapeScores = blendshapeScores,
            matrix = matrix,
            spec = parts.spec,
        )
        feedOrStop(frame, speedKmh)
    }

    /** 얼굴이 없는 프레임(MediaPipe 결과의 얼굴 목록이 비었다). 나머지는 [update]와 같다. */
    fun updateNoFace(tMs: Long, imageWidth: Int, imageHeight: Int, speedKmh: Double?): DriverState? = entered {
        feedOrStop(RawFrames.noFace(tMs = tMs, w = imageWidth, h = imageHeight), speedKmh)
    }

    /** 마지막으로 낸 DriverState. 받아들인 프레임이 아직 없으면 null이다. */
    val lastState: DriverState?
        get() = parts.session.lastState

    /** 파이프라인의 계수 다섯, 지금 값. */
    val counters: PipelineCounters
        get() = parts.session.counters

    /**
     * ONNX 세션을 닫는다. 두 번 불러도 된다. 닫은 뒤 update는 IllegalStateException. 다른 스레드의 update가 진행 중이면 그
     * update가 끝나기를 기다린 뒤 닫는다.
     */
    override fun close() {
        closeRequested = true
        lock.lock()
        try {
            closeModel()
        } finally {
            lock.unlock()
        }
    }

    /** lock 안에서 부른다. */
    private fun closeModel() {
        if (modelClosed) return
        modelClosed = true
        parts.model.close()
    }

    /** update와 updateNoFace의 틀: 겹친 호출 거부, 멈춤·닫힘 확인, 그다음 block(잠금 안). */
    private inline fun entered(block: () -> DriverState?): DriverState? {
        // ReentrantLock은 같은 스레드의 두 번째 tryLock을 받아 주므로, 같은 스레드 안의 겹친 호출도 여기서 거부한다.
        // 잠금을 얻지 못했을 때 close()가 불린 뒤이면(close가 기다리거나 닫는 중) 닫힘으로 거부한다.
        if (lock.isHeldByCurrentThread || !lock.tryLock()) {
            check(!closeRequested) { "the DrowsinessMonitor is closed" }
            throw IllegalStateException("update must be called one at a time")
        }
        try {
            failure?.let { throw IllegalStateException("the DrowsinessMonitor stopped after a processing error: $it", it) }
            check(!closeRequested) { "the DrowsinessMonitor is closed" }
            return block()
        } finally {
            lock.unlock()
        }
    }

    /**
     * lock 안에서 부른다. 파이프라인이 예외를 내면(종류를 가리지 않는다, live.py:1629) 멈춘다: 그 예외를 적고 ONNX 세션을
     * 닫은 뒤 그대로 던진다. 그때 파이프라인은 프레임 하나를 반쯤 처리한 상태라 더 쓰지 않는다(DriveSession.feed).
     */
    private fun feedOrStop(frame: RawFrame, speedKmh: Double?): DriverState? {
        try {
            return parts.session.feed(frame, speedKmh)
        } catch (e: Throwable) {
            failure = e
            try {
                closeModel()
            } catch (c: Throwable) {
                e.addSuppressed(c)
            }
            throw e
        }
    }

    companion object {
        /** JAR 안 번들 파일의 자리(processResources가 export/에서 복사한다, 판정 9). */
        private const val BUNDLE_DIR = "/hawkeye/drowsy/bundle/"
        private const val BUNDLED_MODEL = "drowsiness_v0.onnx"
        private const val BUNDLED_SPEC = "feature_spec_v0.json"
        private const val BUNDLED_RULES = "rules.json"

        // 계획 7.2절: drowsiness_v{N}.onnx와 짝인 feature_spec_v{N}.json. 이름은 문제 메시지의 subject로만 쓴다.
        private val MODEL_FILE = Regex("^drowsiness_v(0|[1-9][0-9]*)\\.onnx$")

        /**
         * JAR에 든 파일(모델, feature_spec, rules)로 연다. 시작 검사를 통과하지 못하면 IllegalStateException(문제 목록을 메시지에).
         * mediapipeModelSha256: 앱이 실제로 읽은 face_landmarker.task 파일의 sha256(소문자 16진수). 있는 그대로 비교한다.
         * 번들 파일은 부를 때마다 클래스 경로에서 읽는다(R19: 상태를 가진 companion을 두지 않는다). 번들 파일이 없으면
         * FILE_MISSING으로 거부한다.
         */
        @JvmStatic
        fun open(mediapipeModelSha256: String): DrowsinessMonitor = openBundled(mediapipeModelSha256, null)

        /** 같은 일을 앱이 준 바이트로 한다(앱이 파일을 assets에 두고 싶을 때). modelFileName은 drowsiness_v{N}.onnx 꼴이다. */
        @JvmStatic
        fun open(
            modelFileName: String,
            modelBytes: ByteArray,
            featureSpecBytes: ByteArray,
            rulesBytes: ByteArray,
            mediapipeModelSha256: String,
        ): DrowsinessMonitor = openChecked(
            modelFileName = modelFileName,
            modelBytes = modelBytes,
            featureSpecBytes = featureSpecBytes,
            rulesBytes = rulesBytes,
            mediapipeModelSha256 = mediapipeModelSha256,
            modelTapForTest = null,
        )

        /** [open]과 같다. modelTapForTest는 시험만 준다: 파이프라인에 넘기는 모델 함수를 감싼다(창을 기록하거나 막는다). */
        internal fun openBundled(mediapipeModelSha256: String, modelTapForTest: ((ModelFn) -> ModelFn)?): DrowsinessMonitor {
            val model = readBundled(BUNDLED_MODEL)
            val spec = readBundled(BUNDLED_SPEC)
            val rules = readBundled(BUNDLED_RULES)
            return openChecked(
                modelFileName = BUNDLED_MODEL,
                modelBytes = model,
                featureSpecBytes = spec,
                rulesBytes = rules,
                mediapipeModelSha256 = mediapipeModelSha256,
                modelTapForTest = modelTapForTest,
            )
        }

        /** 판정 9의 순서: sha256 → inspect → StartupChecks.run → (통과했을 때만) open과 DriveSession.create. */
        internal fun openChecked(
            modelFileName: String,
            modelBytes: ByteArray?,
            featureSpecBytes: ByteArray?,
            rulesBytes: ByteArray?,
            mediapipeModelSha256: String,
            modelTapForTest: ((ModelFn) -> ModelFn)?,
        ): DrowsinessMonitor {
            // 1. 바이트의 sha256을 먼저(StartupChecks가 spec·rules 바이트를 해시하고, 모델은 여기서 한다).
            val modelSha = modelBytes?.let { sha256Hex(it) }
            // 2. 같은 바이트로 메타데이터와 입출력 모양을 읽는다. ONNX Runtime이 읽지 못하면 MODEL_UNREADABLE(StartupChecks).
            val info: OnnxModelInfo? = if (modelBytes == null) {
                null
            } else {
                try {
                    OnnxDrowsinessModel.inspect(modelBytes)
                } catch (e: IllegalArgumentException) {
                    null
                }
            }
            val n = MODEL_FILE.matchEntire(modelFileName)?.groupValues?.get(1)
            val specFileName = if (n != null) "feature_spec_v$n.json" else "feature_spec.json"
            // 3. 시작 검사(문제를 모두 모은다). 모델 파일 고정값(K12)은 이 입구에 없다: 가중치 한 바이트는 보고만 된다(SC2).
            val result = StartupChecks.run(
                StartupInput(
                    modelFileName = modelFileName,
                    modelFileSha256 = modelSha,
                    modelSha256Pin = null,
                    modelMetadata = info?.metadata,
                    modelIo = info?.io,
                    specFileName = specFileName,
                    specBytes = featureSpecBytes,
                    rulesBytes = rulesBytes,
                    taskSha256 = mediapipeModelSha256,
                    verified = VerifiedConstants.VALUES,
                ),
            )
            if (!result.passed) {
                throw IllegalStateException(
                    "startup checks failed (${result.problems.size} problems): " +
                        result.problems.joinToString("; ") { it.message },
                )
            }
            // 4. 통과했을 때만 같은 바이트로 모델 세션과 운전 세션을 만든다.
            val spec = result.spec!!
            val model = OnnxDrowsinessModel.open(modelBytes!!, spec.windowSteps, spec.features.size)
            try {
                val fn: ModelFn = if (modelTapForTest == null) model else modelTapForTest(model)
                val session = DriveSession.create(result, fn)
                return DrowsinessMonitor(Parts(session = session, model = model, spec = spec.pipeline))
            } catch (e: Throwable) {
                model.close()
                throw e
            }
        }

        /** 클래스 경로의 번들 파일 하나. 없으면 null(FILE_MISSING). */
        private fun readBundled(name: String): ByteArray? =
            DrowsinessMonitor::class.java.getResourceAsStream(BUNDLE_DIR + name)?.use { it.readBytes() }
    }
}
