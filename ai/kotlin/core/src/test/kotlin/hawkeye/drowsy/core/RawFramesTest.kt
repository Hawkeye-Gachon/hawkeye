package hawkeye.drowsy.core

import hawkeye.drowsy.core.spec.PipelineSpec
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * RawFrames (계획 4.1절). 옮긴 단위 테스트: tests/test_extract.py:251-312 `test_result_to_raw_frame_*`.
 * Python 시험의 가짜 결과(랜드마크 478, 블렌드셰이프 52, 4x4 float32 행렬)와 같은 모양을 Android가 주는 평범한 배열로 만든다.
 * Android의 행렬은 이미 열 우선 float[16]이므로(SPEC 1장), Python 시험의 행렬을 열 우선으로 편 배열을 넣는다.
 * 무작위 값은 Python과 같을 필요가 없다(시험은 값을 그대로 옮기는지만 본다).
 */
class RawFramesTest {

    private val spec: PipelineSpec =
        SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes()).pipeline

    private class Fake(val x: FloatArray, val y: FloatArray, val names: List<String>, val scores: FloatArray, val matrix: FloatArray)

    private fun fake(seed: Int): Fake {
        val rng = java.util.Random(seed.toLong())
        val x = FloatArray(478) { rng.nextFloat() }
        val y = FloatArray(478) { rng.nextFloat() }
        val names = spec.blendshapes.order
        val scores = FloatArray(names.size) { rng.nextFloat() }
        // tests/test_extract.py:82-90의 행렬(행 M[i][j])을 열 우선으로 편다: m[4*j + i] = M[i][j]
        val rows = arrayOf(
            floatArrayOf(0.9f, 0.1f, 0.2f, 1.5f + seed),
            floatArrayOf(-0.1f, 0.8f, 0.3f, -2.5f),
            floatArrayOf(-0.2f, -0.3f, 0.7f, -30.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f, 1.0f),
        )
        val m = FloatArray(16)
        for (i in 0 until 4) for (j in 0 until 4) m[4 * j + i] = rows[i][j]
        return Fake(x, y, names, scores, m)
    }

    @Test
    fun copiesSubsetBlendshapesAndSize() {
        val r = fake(3)
        val frame = RawFrames.of(1234L, 640, 480, r.x, r.y, r.names, r.scores, r.matrix, spec)
        assertEquals(1234L, frame.tMs)
        assertTrue(frame.face)
        assertEquals(640, frame.w)
        assertEquals(480, frame.h)
        val required = spec.landmarks.requiredIndices()
        assertEquals(44, required.size)
        assertEquals(required, frame.lm.keys.toList()) // 44개만, 인덱스 오름차순
        for (index in required) {
            assertEquals(Point(r.x[index].toDouble(), r.y[index].toDouble()), frame.lm[index])
        }
        assertEquals(spec.blendshapes.order, frame.bs.keys.toList()) // 52개 전부, 이름으로
        for (i in r.names.indices) assertEquals(r.scores[i].toDouble(), frame.bs[r.names[i]])
        assertTrue(frame.bs.keys.containsAll(spec.blendshapes.requiredNames()))
        assertEquals(r.scores[9].toDouble(), frame.bs["eyeBlinkLeft"])
        assertEquals(r.scores[10].toDouble(), frame.bs["eyeBlinkRight"])
        assertEquals(r.scores[25].toDouble(), frame.bs["jawOpen"])
    }

    @Test
    fun matrixIsColumnMajor() {
        val r = fake(0)
        val frame = RawFrames.of(1L, 640, 480, r.x, r.y, r.names, r.scores, r.matrix, spec)
        assertEquals(16, frame.m.size)
        val expected = listOf(0.9, -0.1, -0.2, 0.0, 0.1, 0.8, -0.3, 0.0, 0.2, 0.3, 0.7, 0.0, 1.5, -2.5, -30.0, 1.0)
        assertEquals(expected.map { it.toFloat().toDouble() }, frame.m) // f32(v)
        assertEquals(listOf(1.5, -2.5, -30.0), frame.m.subList(12, 15))
        assertEquals(0.0, frame.m[3])
        assertEquals(0.0, frame.m[7])
        assertEquals(0.0, frame.m[11])
    }

    @Test
    fun withoutFace() {
        assertEquals(RawFrame(77L, false, 320, 240, emptyMap(), emptyMap(), emptyList()), RawFrames.noFace(77L, 320, 240))
    }

    @Test
    fun rejectsIncompleteResults() {
        val r = fake(0)
        // 468점 결과(홍채 없음)
        val e1 = assertFailsWith<IllegalArgumentException> {
            RawFrames.of(1L, 640, 480, r.x.copyOf(468), r.y.copyOf(468), r.names, r.scores, r.matrix, spec)
        }
        assertTrue(e1.message!!.contains("468 landmarks"), e1.message)
        // 16개가 아닌 행렬
        assertFailsWith<IllegalArgumentException> { RawFrames.of(1L, 640, 480, r.x, r.y, r.names, r.scores, FloatArray(9), spec) }
        // 필요한 블렌드셰이프가 없음
        val names = r.names.map { if (it == "jawOpen") "jawOpenX" else it }
        assertFailsWith<IllegalArgumentException> { RawFrames.of(1L, 640, 480, r.x, r.y, names, r.scores, r.matrix, spec) }
        // 길이가 다른 배열
        assertFailsWith<IllegalArgumentException> { RawFrames.of(1L, 640, 480, r.x, r.y.copyOf(477), r.names, r.scores, r.matrix, spec) }
        assertFailsWith<IllegalArgumentException> { RawFrames.of(1L, 640, 480, r.x, r.y, r.names, r.scores.copyOf(51), r.matrix, spec) }
    }
}
