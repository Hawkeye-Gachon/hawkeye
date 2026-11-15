package hawkeye.drowsy.onnx

import hawkeye.drowsy.core.check.ModelIo
import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testkit.DiffData
import hawkeye.drowsy.core.testkit.Hkdiff1
import hawkeye.drowsy.core.testkit.VectorComparator
import hawkeye.drowsy.core.testkit.VectorFile
import hawkeye.drowsy.core.testkit.VectorModel
import org.junit.jupiter.api.Assumptions
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * E11 (계획 9.1절): JVM에서 ONNX. 데스크톱 ONNX Runtime 1.28.0 JAR(K6 (가))로 v0 모델을 연다.
 *
 * 기대값은 모두 Python onnxruntime 1.31.1이 낸 값이다: 벡터 `model_v0_inputs`(창 셋)와 `e2e_250s`(첫 창과 그 확률),
 * kotlin/.diffdata/model/model_windows.bin(창 300개, 순서를 뒤집은 창, Double → Float 변환 표), tamper/manifest.json의
 * base_metadata(onnx 1.23.2로 읽은 메타데이터 10개). 통과 기준은 절대 1e-4이고, 같은 창 두 번은 비트가 같아야 하며,
 * 순서를 뒤집은 창은 바른 출력과 1e-4보다 크게 달라야 한다(Python에서 먼저 확인한 경우만, 머리의 order_check.kept).
 */
class OnnxModelTest {

    private val modelFile = File(DiffData.export, "drowsiness_v0.onnx")
    private val spec = SpecLoader.parseFeatureSpec(File(DiffData.export, "feature_spec_v0.json").readBytes())
    private val rulesSha = sha256Hex(File(DiffData.export, "rules.json").readBytes())
    private val model: OnnxDrowsinessModel by lazy { OnnxDrowsinessModel.open(modelFile.readBytes(), spec.windowSteps, N) }
    private var opened = false

    @AfterTest
    fun closeModel() {
        if (opened) model.close()
    }

    private fun model(): OnnxDrowsinessModel {
        opened = true
        return model
    }

    /**
     * 파일 해시, 입출력 이름과 모양, inspect = open, 런타임 버전. export/와 tests/vectors만 읽으므로 .diffdata 없이도 돈다
     * (저장소를 새로 받은 사본에서도 ONNX Runtime이 "1.28.0"을 알려 주는지 본다).
     */
    @Test
    fun fileIoAndRuntimeVersion() {
        val bytes = modelFile.readBytes()
        val sha = sha256Hex(bytes)
        val mv = DiffData.vector("model_v0_inputs")
        val e2e = DiffData.vector("e2e_250s")
        assertEquals((mv.model as VectorModel.Onnx).sha256, sha, "model_v0_inputs model.sha256")
        assertEquals((e2e.model as VectorModel.Onnx).sha256, sha, "e2e_250s model.sha256")
        val m = model()
        assertEquals(ModelIo("features", listOf(1L, 120L, 17L), "probs", listOf(1L, 3L)), m.io)
        val info = OnnxDrowsinessModel.inspect(bytes)
        assertEquals(m.metadata, info.metadata)
        assertEquals(m.io, info.io)
        val version = OnnxDrowsinessModel.runtimeVersion()
        println("OnnxModelTest.fileIoAndRuntimeVersion: runtime ${version}, model sha256 $sha, io ${m.io}, inspect equals open: true")
        assertEquals("1.28.0", version)
    }

    /** 메타데이터 10개가 Python(onnx 1.23.2)이 읽은 tamper/manifest.json의 base_metadata와 같다. .diffdata가 있어야 한다. */
    @Test
    fun metadataEqualsPython() {
        Assumptions.assumeTrue(DiffData.readySmall() != null, "no READY_small.txt: differential data (small) not exported")
        val manifest = DiffData.readyJson("tamper/manifest.json")
        val baseMetadata = DiffData.obj(manifest["base_metadata"]).mapValues { it.value as String }
        val m = model()
        assertEquals(10, baseMetadata.size)
        assertEquals(baseMetadata, m.metadata, "metadata read by ONNX Runtime 1.28.0 vs onnx 1.23.2")
        println("OnnxModelTest.metadataEqualsPython: metadata keys ${m.metadata.size} equal to Python's: true")
    }

    /** 벡터 model_v0_inputs: 창 셋(0 창, e2e_250s 첫 창, 범위 밖 값이 섞인 무작위 창), SPEC 16장 판정과 엄격 층. */
    @Test
    fun modelVector() {
        val file = File(DiffData.vectors, "model_v0_inputs.json")
        val v = VectorFile.read(file.readBytes())
        assertEquals(emptyList(), v.stageZeroProblems(file.nameWithoutExtension, spec.pipelineHash, rulesSha))
        val inputs = v.modelInputs!!
        val actual = inputs.mapIndexed { i, w -> linkedMapOf<String, Any?>("input" to i.toLong(), "probs" to model().predict(w)) }
        val result = VectorComparator.compareVector(v, mapOf("probs" to actual))
        var worst = 0.0
        val expected = VectorFile.list(v.expected["probs"], "probs")
        for (i in expected.indices) {
            val e = VectorFile.list(DiffData.obj(expected[i])["probs"], "probs").map { VectorFile.num(it, "p") }
            @Suppress("UNCHECKED_CAST")
            val a = actual[i]["probs"] as List<Double>
            for (j in 0 until 3) worst = maxOf(worst, Math.abs(a[j] - e[j]))
        }
        println("OnnxModelTest.modelVector: ${inputs.size} windows, passed ${result.passed}, largest |d| $worst, problems ${result.problems}, strict ${result.strict.failures}, notes ${result.strict.notes.size}")
        assertTrue(result.passed, "${result.problems} ${result.strict.failures}")
    }

    /** e2e_250s의 첫 창(expected.model_inputs[0])과 그 스텝의 확률(expected.step_probs, expected.probs의 같은 k). */
    @Test
    fun e2eFirstWindow() {
        val file = File(DiffData.vectors, "e2e_250s.json")
        val v = VectorFile.read(file.readBytes())
        assertEquals(emptyList(), v.stageZeroProblems(file.nameWithoutExtension, spec.pipelineHash, rulesSha))
        val first = DiffData.obj(VectorFile.list(v.expected["model_inputs"], "model_inputs")[0])
        val k = first["k"] as Long
        val window = VectorFile.list(first["features"], "features").map { row -> VectorFile.list(row, "row").map { VectorFile.num(it, "f") } }
        val got = model().predict(window)
        val stepProbs = VectorFile.list(v.expected["step_probs"], "step_probs").map { DiffData.obj(it) }.first { it["k"] == k }
        val probs = VectorFile.list(v.expected["probs"], "probs").map { DiffData.obj(it) }.first { it["k"] == k }
        val problems = ArrayList<String>()
        VectorComparator.compare("step_probs", "probs", stepProbs["probs"], got, v.tolerances.getValue("step_probs"), "expected.step_probs[k=$k].probs", problems)
        VectorComparator.compare("probs", "probs", probs["probs"], got, v.tolerances.getValue("probs"), "expected.probs[k=$k].probs", problems)
        val e = VectorFile.list(stepProbs["probs"], "p").map { VectorFile.num(it, "p") }
        val worst = (0 until 3).maxOf { Math.abs(got[it] - e[it]) }
        println("OnnxModelTest.e2eFirstWindow: k $k, kotlin $got, python $e, largest |d| $worst, problems $problems")
        assertEquals(emptyList(), problems)
    }

    /**
     * model_windows.bin: 창 300개(f64)를 predict에 넣어 Python onnxruntime 1.31.1의 출력(f32)과 절대 1e-4로 비교한다.
     * 같은 창 두 번은 비트가 같아야 한다. 비트가 같은 출력 수를 센다. Double → Float 변환은 창의 값 전부와 변환 표(conv)를
     * numpy astype(float32)의 비트와 비교한다.
     */
    @Test
    fun randomWindowsAgainstPythonOnnxRuntime() {
        Assumptions.assumeTrue(DiffData.readySmall() != null, "no READY_small.txt: differential data (small) not exported")
        val h = DiffData.readyHkdiff1("model/model_windows.bin")
        assertEquals(sha256Hex(modelFile.readBytes()), h.header["model_sha256"], "model_windows.bin was made from this model")
        val f64 = h.f64("windows_f64")
        val f32 = h.f32("windows_f32")
        val probs = h.f32("probs")
        val kind = h.u8("kind")
        val n = kind.size
        assertEquals(300, n)
        val rows = spec.windowSteps
        val per = rows * N
        // Double → Float: 창의 값 전부
        var convMismatch = 0
        for (i in f64.indices) if (f64[i].toFloat().toRawBits() != f32[i].toRawBits()) convMismatch++
        // 변환 표
        val c64 = h.f64("conv_f64")
        val c32 = h.f32("conv_f32")
        var convTableMismatch = 0
        for (i in c64.indices) if (c64[i].toFloat().toRawBits() != c32[i].toRawBits()) convTableMismatch++

        val worstByKind = java.util.TreeMap<Int, Double>()
        var worst = 0.0
        var bitEqual = 0
        var over = 0
        var notRepeatable = 0
        val m = model()
        for (w in 0 until n) {
            val window = List(rows) { t -> List(N) { f -> f64[w * per + t * N + f] } }
            val out = m.predict(window)
            val again = m.predict(window)
            for (j in 0 until 3) {
                if (out[j].toRawBits() != again[j].toRawBits()) notRepeatable++
                val py = probs[w * 3 + j]
                val d = Math.abs(out[j] - py.toDouble())
                if (!(d <= 1e-4)) over++
                if (out[j].toFloat().toRawBits() == py.toRawBits()) bitEqual++
                if (d > worst) worst = d
                if (d > (worstByKind[kind[w]] ?: 0.0)) worstByKind[kind[w]] = d
            }
        }
        println("OnnxModelTest.randomWindows: windows $n (kinds ${kind.groupBy { it }.mapValues { it.value.size }}), outputs ${n * 3}, over 1e-4: $over, largest |d| $worst, per kind $worstByKind")
        println("OnnxModelTest.randomWindows: outputs bit-identical to Python onnxruntime ${h.header["onnxruntime_version"]}: $bitEqual/${n * 3}; same window twice not bit-identical: $notRepeatable")
        println("OnnxModelTest.randomWindows: Double->Float mismatches vs numpy: window values $convMismatch/${f64.size}, conversion table $convTableMismatch/${c64.size}")
        assertEquals(0, convMismatch)
        assertEquals(0, convTableMismatch)
        assertEquals(0, over)
        assertEquals(0, notRepeatable)
    }

    /** 스텝 순서나 특징 순서를 뒤집은 창(Python에서 1e-4보다 크게 다르다고 확인한 경우만)은 바른 출력과 1e-4보다 크게 다르다. */
    @Test
    fun reversedOrderWindowsDiffer() {
        Assumptions.assumeTrue(DiffData.readySmall() != null, "no READY_small.txt: differential data (small) not exported")
        val h = DiffData.readyHkdiff1("model/model_windows.bin")
        val ow = h.f32("order_windows")
        val op = h.f32("order_probs")
        val orderCheck = DiffData.obj(h.header["order_check"])
        val kept = DiffData.list(orderCheck["kept"]).map { it as String }
        val rowNames = DiffData.list(orderCheck["rows"]).map { it as String }
        assertEquals(3, rowNames.size)
        val rows = spec.windowSteps
        val per = rows * N
        val outs = (0 until 3).map { w -> model().predict(List(rows) { t -> List(N) { f -> ow[w * per + t * N + f].toDouble() } }) }
        // 0행은 e2e_250s의 첫 창이다: 벡터의 창을 Float로 바꾼 것과 비트가 같다.
        val e2e = DiffData.vector("e2e_250s")
        val first = VectorFile.list(DiffData.obj(VectorFile.list(e2e.expected["model_inputs"], "mi")[0])["features"], "features")
        for (t in 0 until rows) {
            val row = VectorFile.list(first[t], "row")
            for (f in 0 until N) assertEquals(VectorFile.num(row[f], "v").toFloat().toRawBits(), ow[t * N + f].toRawBits())
        }
        val report = ArrayList<String>()
        for (w in 0 until 3) {
            val vsPython = (0 until 3).maxOf { Math.abs(outs[w][it] - op[w * 3 + it].toDouble()) }
            assertTrue(vsPython <= 1e-4, "${rowNames[w]}: |d| vs Python $vsPython")
            val vsCorrect = (0 until 3).maxOf { Math.abs(outs[w][it] - outs[0][it]) }
            report.add("${rowNames[w]}: vs python $vsPython, vs the correct window $vsCorrect")
            if (w > 0) {
                val name = if (w == 1) "reversed_steps" else "reversed_features"
                if (name in kept) assertTrue(vsCorrect > 1e-4, "$name must differ by more than 1e-4, got $vsCorrect")
            }
        }
        println("OnnxModelTest.reversedOrderWindows: kept $kept; $report")
        assertEquals(listOf("reversed_steps", "reversed_features"), kept)
    }

    /** 계약 밖의 사용: 창 모양이 다르면 IllegalArgumentException, 다른 창 길이로 열면 IllegalArgumentException, 닫은 뒤 IllegalStateException. */
    @Test
    fun contractViolationsAreRefused() {
        val bytes = modelFile.readBytes()
        assertFailsWith<IllegalArgumentException> { OnnxDrowsinessModel.open(bytes, 60, N) }
        assertFailsWith<IllegalArgumentException> { OnnxDrowsinessModel.open(bytes, 120, 16) }
        assertFailsWith<IllegalArgumentException> { OnnxDrowsinessModel.open(byteArrayOf(1, 2, 3), 120, N) }
        val m = OnnxDrowsinessModel.open(bytes, 120, N)
        assertFailsWith<IllegalArgumentException> { m.predict(List(119) { List(N) { 0.0 } }) }
        assertFailsWith<IllegalArgumentException> { m.predict(List(120) { List(N - 1) { 0.0 } }) }
        val zeros = m.predict(List(120) { List(N) { 0.0 } })
        assertEquals(3, zeros.size)
        m.close()
        m.close()
        assertFailsWith<IllegalStateException> { m.predict(List(120) { List(N) { 0.0 } }) }
    }

    /**
     * 계획 4장(공개 API에 ONNX Runtime 형이 없다)과 2.2절(Java 호출자): Java가 부를 수 있는 생성자가 없고, 어느 생성자의
     * 매개변수에도 ai.onnxruntime 형이 없다. Kotlin internal 생성자는 바이트코드에서 public이라, 고치기 전에는 Java 호출자가
     * open()의 모양·이름 검사를 건너뛰고 모델을 만들 수 있었다(f_s2 재현). 남는 비private 생성자는 Kotlin이 companion을 위해
     * 만드는 합성 생성자뿐이고, javac는 합성 멤버를 소스에서 부르지 못하게 한다.
     */
    @Test
    fun noJavaCallableConstructorAndNoOrtTypesInConstructors() {
        val ctors = OnnxDrowsinessModel::class.java.declaredConstructors
        val callable = ctors.filter { !java.lang.reflect.Modifier.isPrivate(it.modifiers) && !it.isSynthetic }
        val ortParams = ctors.flatMap { c -> c.parameterTypes.filter { it.name.startsWith("ai.onnxruntime.") }.map { "${c.toGenericString()}: ${it.name}" } }
        println("OnnxModelTest.constructors: ${ctors.size} declared, Java-callable ${callable.size}, parameters of ai.onnxruntime types ${ortParams.size}")
        assertEquals(emptyList(), callable.map { it.toGenericString() })
        assertEquals(emptyList(), ortParams)
    }

    private companion object {
        const val N = 17
    }
}
