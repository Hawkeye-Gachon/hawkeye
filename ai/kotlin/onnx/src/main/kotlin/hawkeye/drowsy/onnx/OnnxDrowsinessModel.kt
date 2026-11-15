package hawkeye.drowsy.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import hawkeye.drowsy.core.ModelFn
import hawkeye.drowsy.core.check.ModelIo
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections

// ONNX 모델 어댑터 (SPEC 14.1, 계획 4.3절, 5.12절). 원본 동작: drowsy/demo.py:275-295 `make_model_fn`,
// drowsy/vectors.py:504-518 `_session`, `_onnx_probs`.
//
// - 세션: CPU, intra-op 스레드 1, inter-op 스레드 1 (demo.py:282-283, vectors.py:507-510, SPEC 14.1 실행 설정).
// - 입력 `features` float32 [1, windowSteps, nFeatures]: 오래된 스텝부터, 스텝 안에서는 특징 #0부터 `v.toFloat()`를 넣는다.
//   표준화도 log1p도 하지 않는다(SPEC 14.1, 14.2: 그래프 안에서 한다). Python은 `np.asarray(window, dtype=np.float32)`다.
//   가정: JVM의 Double → Float 변환(d2f, 가장 가까운 값, 짝수로)이 numpy `astype(float32)`와 같다(E11의 변환 표가 본다).
// - 출력 `probs` float32 [1, 3]의 3개를 `toDouble()`로 넓힌다(Python `float(value)`, 정확한 넓힘).
// - ONNX Runtime 형은 공개 API에 나오지 않는다(계획 4장). ORT 예외는 IllegalArgumentException이나 IllegalStateException에
//   싸서 던진다.
// 가정(A6): 데스크톱 JAR(compileOnly)와 onnxruntime-android의 ai.onnxruntime 클래스와 메서드가 같아서 이 코드가 기기에서 돈다.
// 가정(A7): OrtEnvironment는 프로세스에 하나이고 라이브러리가 닫지 않는다.
// 가정(A18): ONNX Runtime 1.28.0(이 어댑터)과 1.31.1(Python 참조)이 v0 파일을 열고 절대 1e-4 안에서 같은 확률을 낸다(E11).

/** 모델 파일에서 읽은 메타데이터(SPEC 14.3, 값은 모두 문자열)와 첫 입력·첫 출력의 이름과 모양. */
class OnnxModelInfo(val metadata: Map<String, String>, val io: ModelIo)

/**
 * CPU, intra-op 스레드 1, inter-op 스레드 1 (SPEC 14.1, demo.py:282-283). 입력 이름 features, 출력 이름 probs (ONNX 계약).
 *
 * 생성자는 계획 4.3절대로 private이고, ONNX Runtime 객체 셋은 private 중첩 클래스 [Ort]에 담아 받는다. 그래서 어느 생성자
 * 시그니처에도 ai.onnxruntime 형이 없다. Kotlin의 internal 생성자는 바이트코드에서 public이라 Java 호출자가 [open]의 모양·이름
 * 검사를 건너뛰고 모델을 만들 수 있었다(f_s2에서 재현). companion이 부르는 길은 Kotlin이 만드는 합성(synthetic) 생성자이고,
 * javac는 합성 멤버를 소스에서 부르지 못하게 한다. 앱은 [open]으로만 만든다.
 */
class OnnxDrowsinessModel private constructor(
    private val ort: Ort,
    /** 모델 파일의 메타데이터 (SPEC 14.3). */
    val metadata: Map<String, String>,
    /** 세션에서 읽은 입출력 이름과 모양. */
    val io: ModelIo,
    private val windowSteps: Int,
    private val nFeatures: Int,
) : ModelFn, AutoCloseable {

    /** 이 모델이 가진 ONNX Runtime 객체. 클래스 밖에서는 이름을 쓸 수 없다. */
    private class Ort(val env: OrtEnvironment, val options: OrtSession.SessionOptions, val session: OrtSession)

    private val env: OrtEnvironment get() = ort.env
    private val options: OrtSession.SessionOptions get() = ort.options
    private val session: OrtSession get() = ort.session

    private var closed = false

    /**
     * window: windowSteps x nFeatures Double, window[0]이 가장 오래된 스텝. Float로 바꾸기만 한다. 출력 3개를 Double로 넓힌다.
     * 창의 모양이 다르면 IllegalArgumentException. 출력이 float 3개가 아니거나 유한하지 않으면 IllegalStateException
     * (12장 K9 (가): 어댑터에서 멈춘다. Python 참조는 출력을 검사하지 않는다).
     * 가정: Python `reshape(1, window_steps, n_features)`는 원소 수만 맞으면 다른 모양의 창도 받지만, 파이프라인은 언제나
     * window_steps x 17 창을 넘기므로(pipeline.py:240) 여기서는 모양을 정확히 요구한다.
     */
    override fun predict(window: List<List<Double>>): List<Double> {
        check(!closed) { "the model is closed" }
        require(window.size == windowSteps) { "window must have $windowSteps steps, got ${window.size}" }
        val buffer = ByteBuffer.allocateDirect(windowSteps * nFeatures * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (t in 0 until windowSteps) {
            val row = window[t]
            require(row.size == nFeatures) { "window step $t must have $nFeatures features, got ${row.size}" }
            for (f in 0 until nFeatures) {
                buffer.put(row[f].toFloat())
            }
        }
        buffer.rewind()
        val shape = longArrayOf(1L, windowSteps.toLong(), nFeatures.toLong())
        try {
            OnnxTensor.createTensor(env, buffer, shape).use { tensor ->
                session.run(Collections.singletonMap(INPUT_NAME, tensor), Collections.singleton(OUTPUT_NAME)).use { result ->
                    val value = result.get(OUTPUT_NAME).orElse(null)
                        ?: throw IllegalStateException("the model returned no output named $OUTPUT_NAME")
                    val out = (value as? OnnxTensor)?.floatBuffer
                        ?: throw IllegalStateException("the model output $OUTPUT_NAME is not a float tensor")
                    if (out.remaining() != 3) {
                        throw IllegalStateException("model must return 3 probabilities (alert, low_vigilant, drowsy), got ${out.remaining()}")
                    }
                    val probs = ArrayList<Double>(3)
                    for (i in 0 until 3) {
                        val p = out.get(i).toDouble()
                        // pipeline.py:247-249 `_is_finite`와 같은 식: NaN과 ±무한대에서만 거짓이다.
                        if (!(p - p == 0.0)) {
                            throw IllegalStateException("model output $i is not finite: $p")
                        }
                        probs.add(p)
                    }
                    return Collections.unmodifiableList(probs)
                }
            }
        } catch (e: OrtException) {
            throw IllegalStateException("ONNX Runtime failed to run the model: ${e.message}", e)
        }
    }

    /** 세션과 옵션을 닫는다. 환경(OrtEnvironment)은 닫지 않는다(가정 A7). 여러 번 불러도 된다. */
    override fun close() {
        if (closed) return
        closed = true
        try {
            session.close()
        } catch (e: OrtException) {
            throw IllegalStateException("ONNX Runtime failed to close the session: ${e.message}", e)
        } finally {
            options.close()
        }
    }

    companion object {
        // ONNX 계약 (SPEC 14.1).
        private const val INPUT_NAME = "features"
        private const val OUTPUT_NAME = "probs"

        /**
         * 모델 바이트로 세션을 연다. 첫 입력의 모양이 [1, windowSteps, nFeatures]가 아니면 IllegalArgumentException
         * (demo.py:285-288). 입력 이름이 features가 아니거나 출력 이름이 probs가 아니어도 IllegalArgumentException이다
         * (predict가 그 이름으로 부른다, demo.py:292). ONNX Runtime이 파일을 읽지 못해도 IllegalArgumentException.
         */
        @JvmStatic
        fun open(modelBytes: ByteArray, windowSteps: Int, nFeatures: Int): OnnxDrowsinessModel {
            val env = OrtEnvironment.getEnvironment()
            val options = newOptions()
            val session = try {
                env.createSession(modelBytes, options)
            } catch (e: OrtException) {
                options.close()
                throw IllegalArgumentException("ONNX Runtime cannot read the model: ${e.message}", e)
            }
            try {
                val info = describe(session)
                val expected = listOf(1L, windowSteps.toLong(), nFeatures.toLong())
                if (info.io.inputShape != expected) {
                    throw IllegalArgumentException("model input shape is ${info.io.inputShape}, the pipeline feeds $expected")
                }
                if (info.io.inputName != INPUT_NAME || info.io.outputName != OUTPUT_NAME) {
                    throw IllegalArgumentException(
                        "model input/output names are ${info.io.inputName}/${info.io.outputName}, expected $INPUT_NAME/$OUTPUT_NAME",
                    )
                }
                return OnnxDrowsinessModel(Ort(env, options, session), info.metadata, info.io, windowSteps, nFeatures)
            } catch (e: Throwable) {
                try {
                    session.close()
                } catch (ignored: OrtException) {
                    // 원래 예외를 던진다.
                }
                options.close()
                throw e
            }
        }

        /**
         * 시작 검사용(계획 5.8절 시작 순서, 5.11절): 같은 설정으로 세션을 열어 메타데이터와 첫 입력·첫 출력의 이름과 모양을 읽고
         * 닫는다. 모양과 이름을 검사하지 않는다(계약과 다르면 StartupChecks가 MODEL_IO_CONTRACT로 보고한다).
         * ONNX Runtime이 파일을 읽지 못하면 IllegalArgumentException이다(호출자는 MODEL_UNREADABLE로 보고한다).
         */
        @JvmStatic
        fun inspect(modelBytes: ByteArray): OnnxModelInfo {
            val env = OrtEnvironment.getEnvironment()
            val options = newOptions()
            try {
                val session = try {
                    env.createSession(modelBytes, options)
                } catch (e: OrtException) {
                    throw IllegalArgumentException("ONNX Runtime cannot read the model: ${e.message}", e)
                }
                try {
                    return describe(session)
                } finally {
                    try {
                        session.close()
                    } catch (ignored: OrtException) {
                        // 읽은 값은 이미 있다. 닫기 실패는 결과를 바꾸지 않는다.
                    }
                }
            } finally {
                options.close()
            }
        }

        /**
         * 실행 중인 ONNX Runtime이 알려 주는 버전.
         * 가정(A29): ONNX Runtime Java에 실행 중 버전을 알려 주는 API가 있다. 1.28.0 데스크톱 JAR에는
         * `OrtEnvironment.getVersion()`이 있다(javap로 확인, E11이 "1.28.0"을 읽었다). Android AAR에도 같은 메서드가 있다는 것은 A6에 기댄다.
         */
        @JvmStatic
        fun runtimeVersion(): String? = OrtEnvironment.getEnvironment().version

        private fun newOptions(): OrtSession.SessionOptions {
            val options = OrtSession.SessionOptions()
            try {
                options.setIntraOpNumThreads(1)
                options.setInterOpNumThreads(1)
            } catch (e: OrtException) {
                options.close()
                throw IllegalStateException("ONNX Runtime refused the session options: ${e.message}", e)
            }
            return options
        }

        /** 세션의 메타데이터와 첫 입력·첫 출력. 입력이나 출력이 텐서가 아니거나 없으면 IllegalArgumentException. */
        private fun describe(session: OrtSession): OnnxModelInfo {
            try {
                val metadata = LinkedHashMap<String, String>(session.metadata.customMetadata)
                val inputs = session.inputInfo
                val outputs = session.outputInfo
                require(inputs.isNotEmpty() && outputs.isNotEmpty()) { "the model has no input or no output" }
                val input = inputs.values.first()
                val output = outputs.values.first()
                val inShape = (input.info as? TensorInfo)?.shape
                    ?: throw IllegalArgumentException("model input ${input.name} is not a tensor")
                val outShape = (output.info as? TensorInfo)?.shape
                    ?: throw IllegalArgumentException("model output ${output.name} is not a tensor")
                val io = ModelIo(
                    inputName = input.name,
                    inputShape = Collections.unmodifiableList(inShape.toList()),
                    outputName = output.name,
                    outputShape = Collections.unmodifiableList(outShape.toList()),
                )
                return OnnxModelInfo(Collections.unmodifiableMap(metadata), io)
            } catch (e: OrtException) {
                throw IllegalArgumentException("ONNX Runtime cannot read the model's metadata or io: ${e.message}", e)
            }
        }
    }
}
