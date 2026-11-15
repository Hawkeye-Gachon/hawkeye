package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.Point
import hawkeye.drowsy.core.RawFrame
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.VectorComparator
import hawkeye.drowsy.core.testkit.VectorFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * E3 (계획 9.1절): `signals` 묶음 9개를 SPEC 16장 판정과 엄격 층으로, 그리고 C-func의 신호 함수 기록, C-errors의 신호 함수
 * 오류 경우, 눈 폭 정확히 6 px의 단위 시험(계획 9.3절). 기대값은 모두 바꾸지 않은 Python 참조에서 왔다(벡터 파일,
 * kotlin/.diffdata/components/, 그리고 six_px_expected.py의 출력).
 */
class SignalsTest {

    private val spec = JsonTrees.shippedSpec.pipeline

    /** signals 묶음 9개: 프레임마다 `signals.compute`의 출력을 비교한다(vectors.py:563-565). */
    @Test
    fun signalsBundle() {
        val vectors = TestData.vectorFiles().map { it to VectorFile.read(it.readBytes()) }.filter { it.second.bundle == "signals" }
        assertEquals(9, vectors.size, "signals vectors")
        val stats = FieldStats { field -> if (field == "yaw" || field == "pitch" || field == "roll") 1e-3 else 1e-6 }
        val failed = ArrayList<String>()
        val notes = ArrayList<String>()
        var frames = 0
        for ((file, v) in vectors) {
            val stage0 = v.stageZeroProblems(file.nameWithoutExtension, JsonTrees.shippedSpec.pipelineHash, JsonTrees.shippedRulesSha256)
            if (stage0.isNotEmpty()) {
                failed.add("${v.name}: constants differ: $stage0")
                continue
            }
            val actual = v.frames.map { JsonTrees.signals(compute(it, spec)) }
            val result = VectorComparator.compareVector(v, mapOf("signals" to actual))
            if (!result.passed) failed.add("${v.name}: ${result.problems} ${result.strict.failures}")
            notes.addAll(result.strict.notes.map { "${v.name}: $it" })
            val expected = VectorFile.list(v.expected["signals"], "expected.signals")
            for (i in expected.indices) stats.row(expected[i] as Map<*, *>, actual[i])
            frames += v.frames.size
        }
        println("SignalsTest.signalsBundle: ${vectors.size - failed.size}/${vectors.size} vectors pass (SPEC 16 and strict layer), frames $frames")
        println("SignalsTest.signalsBundle: frames with every field within 1e-9: ${stats.rowsWithin1e9}/${stats.rows}")
        println("SignalsTest.signalsBundle: max |kotlin - python| per field (fraction of the vector tolerance): ${stats.summary()}")
        println("SignalsTest.signalsBundle: strict-layer notes (above 1e-9, not failures): ${notes.size} $notes")
        assertEquals(emptyList(), failed)
    }

    /** C-func의 신호 함수 기록(FORMATS.md 2.7절): compare가 exact면 수로 같음, abs1e-9면 절대 1e-9 안. */
    @Test
    fun componentFunctionRecords() {
        val data = TestData.readyJson("components/C-func.json")
        val counts = java.util.TreeMap<String, Int>()
        val bad = ArrayList<String>()
        var worstAbs = 0.0
        var exactBits = 0
        var abs9Values = 0
        for ((i, c) in TestData.list(data["cases"]).withIndex()) {
            val case = TestData.obj(c)
            val fn = case["fn"] as String
            if (fn !in SIGNAL_FNS) continue
            counts[fn] = (counts[fn] ?: 0) + 1
            val args = TestData.list(case["args"])
            val where = "case $i $fn"
            val error = case["error"]?.let { TestData.obj(it) }
            val got: Any = try {
                callSignalFn(fn, args, where)
            } catch (e: IllegalArgumentException) {
                if (error == null) {
                    bad.add("$where: unexpected ${e.message}")
                } else if (error["type"] != "ValueError" || error["message"] != e.message) {
                    bad.add("$where: expected ${error["type"]} '${error["message"]}', got IllegalArgumentException '${e.message}'")
                }
                continue
            }
            if (error != null) {
                bad.add("$where: expected ${error["type"]} '${error["message"]}', got $got")
                continue
            }
            val expected = JsonTrees.doubles(if (case["result"] is List<*>) case["result"] else listOf(case["result"]), "$where.result")
            @Suppress("UNCHECKED_CAST")
            val actual = if (got is List<*>) got as List<Double> else listOf(got as Double)
            if (expected.size != actual.size) {
                bad.add("$where: expected ${expected.size} values, got ${actual.size}")
                continue
            }
            val compare = case["compare"] as String
            for (k in expected.indices) {
                val e = expected[k]
                val a = actual[k]
                if (a.toRawBits() == e.toRawBits()) exactBits++
                when (compare) {
                    "exact" -> if (!JsonTrees.numericallyEqual(a, e)) bad.add("$where[$k]: expected ${hex(e)}, got ${hex(a)}")
                    "abs1e-9" -> {
                        abs9Values++
                        val d = Math.abs(a - e)
                        if (!(d <= 1e-9) && !(a.isNaN() && e.isNaN())) bad.add("$where[$k]: expected ${hex(e)}, got ${hex(a)} (|d| $d)")
                        if (d > worstAbs) worstAbs = d
                    }
                    else -> bad.add("$where: unknown compare '$compare'")
                }
            }
        }
        println("SignalsTest.componentFunctionRecords: C-func signal cases per function $counts, failures ${bad.size}")
        println("SignalsTest.componentFunctionRecords: values bit-identical to Python $exactBits; abs1e-9 values $abs9Values, largest |d| $worstAbs")
        assertEquals(SIGNAL_FNS, counts.keys, "every signal function has records")
        assertEquals(emptyList(), bad.take(20))
    }

    /** C-errors의 신호 함수 오류 경우(계획 9.2.6절): 16개가 아닌 행렬, 9개가 아닌 회전, 6점이 아닌 EAR. */
    @Test
    fun componentErrorCases() {
        val data = TestData.readyJson("components/C-errors.json")
        var run = 0
        val bad = ArrayList<String>()
        for (c in TestData.list(data["cases"])) {
            val case = TestData.obj(c)
            if (case["component"] != "function") continue
            val setup = TestData.obj(case["setup"])
            val fn = setup["fn"] as String
            if (fn !in SIGNAL_FNS) continue
            run++
            val error = TestData.obj(TestData.obj(case["out"])["error"])
            try {
                val got = callSignalFn(fn, TestData.list(setup["args"]), case["case"] as String)
                bad.add("${case["case"]}: no exception, got $got")
            } catch (e: IllegalArgumentException) {
                if (error["type"] != "ValueError" || error["message"] != e.message) bad.add("${case["case"]}: got '${e.message}', expected $error")
            }
        }
        println("SignalsTest.componentErrorCases: $run C-errors signal cases, refused with the Python message in ${run - bad.size}")
        assertEquals(7, run)
        assertEquals(emptyList(), bad)
    }

    /**
     * 눈 폭 정확히 6 px (계획 9.3절, SPEC 16.1 끝 목록 `docs/SPEC.md:2453`): w = 512에서 x = 100/512와 106/512, 같은 y라
     * 어느 바른 hypot이든 폭이 정확히 6.0이다. 6.0은 쓸 수 있는 눈이다("미만"만 버린다, signals.py:285). 6.0보다 2^-20 px
     * 좁은 눈은 쓸 수 없다. 기대값은 같은 프레임을 바꾸지 않은 `drowsy.signals.compute`에 넣은 값이다
     * (work/kotlin/impl/s2/six_px_expected.py, .out).
     */
    @Test
    fun eyeExactlySixPixelsWide() {
        val exact = compute(sixPixelFrame(106.0), spec)
        assertEquals(6.0, exact.eyeWL)
        assertEquals(1.706666666666668, exact.earL, 1e-9)
        assertEquals(0.5, exact.irisHL!!, 1e-9)
        assertEquals(0.0, exact.irisVL!!, 1e-9)
        assertEquals(40.0, exact.eyeWR)
        assertEquals(java.lang.Double.parseDouble("0x1.0624dd2f1aa00p-1"), exact.earR, 1e-9)

        val below = compute(sixPixelFrame(106.0 - Math.scalb(1.0, -20)), spec)
        assertEquals(java.lang.Double.parseDouble("0x1.7ffffc0000000p+2"), below.eyeWL)
        assertTrue(below.eyeWL < 6.0)
        assertEquals(0.0, below.earL)
        assertNull(below.irisHL)
        assertNull(below.irisVL)
        assertNotNull(below.irisHR)
    }

    /** six_px_expected.py와 같은 프레임. */
    private fun sixPixelFrame(p4X: Double): RawFrame {
        val lms = spec.landmarks
        val lm = LinkedHashMap<Int, Point>()
        for (i in lms.requiredIndices()) lm[i] = Point(0.5, 0.5)
        val left = listOf(
            Point(100 / 512.0, 0.5), Point(102 / 512.0, 0.49), Point(104 / 512.0, 0.49),
            Point(p4X / 512, 0.5), Point(104 / 512.0, 0.51), Point(102 / 512.0, 0.51),
        )
        for ((k, idx) in lms.earL.withIndex()) lm[idx] = left[k]
        lm[lms.irisL[0]] = Point(103 / 512.0, 0.5)
        val right = listOf(
            Point(300 / 512.0, 0.5), Point(313 / 512.0, 0.48), Point(327 / 512.0, 0.48),
            Point(340 / 512.0, 0.5), Point(327 / 512.0, 0.52), Point(313 / 512.0, 0.52),
        )
        for ((k, idx) in lms.earR.withIndex()) lm[idx] = right[k]
        lm[lms.irisR[0]] = Point(320 / 512.0, 0.5)
        val bs = LinkedHashMap<String, Double>()
        for (n in spec.blendshapes.requiredNames()) bs[n] = 0.0
        bs[spec.blendshapes.eyeBlinkL] = 0.25
        bs[spec.blendshapes.eyeBlinkR] = 0.5
        val m = listOf(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -60.0, 1.0)
        return RawFrame(tMs = 1L, face = true, w = 512, h = 512, lm = lm, bs = bs, m = m)
    }

    /** 얼굴이 있는데 필요한 랜드마크·블렌드셰이프가 없거나 행렬이 16개가 아니면 예외(signals.py:220-221, R17). */
    @Test
    fun missingInputsAreRefused() {
        val ok = sixPixelFrame(106.0)
        val noLandmark = ok.copy(lm = ok.lm.filterKeys { it != spec.landmarks.earL[2] })
        val noBlendshape = ok.copy(bs = ok.bs.filterKeys { it != spec.blendshapes.jawOpen })
        val shortMatrix = ok.copy(m = ok.m.subList(0, 15))
        for (f in listOf(noLandmark, noBlendshape, shortMatrix)) {
            assertTrue(runCatching { compute(f, spec) }.exceptionOrNull() is IllegalArgumentException)
        }
        // 얼굴이 없으면 lm, bs, m을 보지 않는다(signals.py:223-238).
        val none = compute(RawFrame(77L, false, 640, 480, emptyMap(), emptyMap(), emptyList()), spec)
        assertEquals(listOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), none.eyeLook)
        assertNull(none.irisHL)
    }

    private fun callSignalFn(fn: String, args: List<Any?>, where: String): Any = when (fn) {
        "eye_aspect_ratio" -> eyeAspectRatio(VectorFile.list(args[0], "$where.args[0]").mapIndexed { k, p -> JsonTrees.point(p, "$where.p$k") })
        "iris_ratio" -> irisRatio(JsonTrees.point(args[0], where), JsonTrees.point(args[1], where), JsonTrees.point(args[2], where))
            .let { listOf(it.h, it.v) }
        "gaze_h" -> gazeH(JsonTrees.doubles(args[0], where))
        "gaze_v" -> gazeV(JsonTrees.doubles(args[0], where))
        "euler_from_colmajor" -> eulerFromColMajor(JsonTrees.doubles(args[0], where)).let { listOf(it.yaw, it.pitch, it.roll) }
        "euler_from_rotation" -> eulerFromRotation(JsonTrees.doubles(args[0], where)).let { listOf(it.yaw, it.pitch, it.roll) }
        "rotation_from_euler" -> rotationFromEuler(
            VectorFile.num(args[0], where), VectorFile.num(args[1], where), VectorFile.num(args[2], where),
        )
        "rotation_from_colmajor" -> rotationFromColMajor(JsonTrees.doubles(args[0], where))
        "relative_rotation" -> relativeRotation(JsonTrees.doubles(args[0], where), JsonTrees.doubles(args[1], where))
        "relative_euler" -> relativeEuler(JsonTrees.doubles(args[0], where), JsonTrees.doubles(args[1], where))
            .let { listOf(it.yaw, it.pitch, it.roll) }
        else -> throw AssertionError("not a signal function: $fn")
    }

    private fun hex(x: Double): String = java.lang.Double.toHexString(x)

    companion object {
        val SIGNAL_FNS: Set<String> = sortedSetOf(
            "eye_aspect_ratio", "iris_ratio", "gaze_h", "gaze_v", "euler_from_colmajor", "euler_from_rotation",
            "rotation_from_euler", "rotation_from_colmajor", "relative_rotation", "relative_euler",
        )
    }
}
