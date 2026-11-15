package hawkeye.drowsy.core.image

import hawkeye.drowsy.core.DrowsyInternalApi
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * ImageGeometry 전수 시험 (계획 5.12절: 회전 넷 × 반전 둘을 비대칭 그림으로).
 *
 * 기대값: 4 x 3 그림(값 0..11, 행 우선)을 Python `drowsy.video_io.rotate_upright`(반시계 각도)에 시계 각도 r의
 * (360 − r) mod 360을 주어 돌리고, 반전이면 `np.fliplr`을 한 결과다(계획 2.4절의 관계). 만든 스크립트와 출력은 작업 폴더
 * work/kotlin/impl/s1/geometry_expected.py, geometry_expected.out에 있다.
 */
@OptIn(DrowsyInternalApi::class)
class ImageGeometryTest {

    private val w = 4
    private val h = 3
    private val src = IntArray(w * h) { it }

    // rotation, mirrored, out w, out h, pixels (Python 출력 그대로)
    private val expected = listOf(
        Triple(0 to false, 4 to 3, intArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)),
        Triple(0 to true, 4 to 3, intArrayOf(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8)),
        Triple(90 to false, 3 to 4, intArrayOf(8, 4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3)),
        Triple(90 to true, 3 to 4, intArrayOf(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)),
        Triple(180 to false, 4 to 3, intArrayOf(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)),
        Triple(180 to true, 4 to 3, intArrayOf(8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3)),
        Triple(270 to false, 3 to 4, intArrayOf(3, 7, 11, 2, 6, 10, 1, 5, 9, 0, 4, 8)),
        Triple(270 to true, 3 to 4, intArrayOf(11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0)),
    )

    @Test
    fun eightCombinationsMatchPython() {
        for ((rm, size, pixels) in expected) {
            val (rot, mirrored) = rm
            assertEquals(size.first, ImageGeometry.outputWidth(w, h, rot), "width $rot")
            assertEquals(size.second, ImageGeometry.outputHeight(w, h, rot), "height $rot")
            assertContentEquals(pixels, ImageGeometry.transform(src, w, h, rot, mirrored), "rotation $rot mirrored $mirrored")
            var k = 0
            for (y in 0 until size.second) for (x in 0 until size.first) {
                assertEquals(pixels[k++], ImageGeometry.sourceIndex(x, y, w, h, rot, mirrored))
            }
        }
        println("ImageGeometryTest: 8 combinations equal to the Python rotate_upright/fliplr result")
    }

    /** 큰 비대칭 그림에서 성질: 각 결과는 원래 픽셀의 순열, 90도 넷은 항등, 반전 두 번은 항등, 90+90 = 180. */
    @Test
    fun algebraOnALargerPicture() {
        val ww = 7
        val hh = 5
        val pic = IntArray(ww * hh) { it * 31 + 7 }
        for (rot in listOf(0, 90, 180, 270)) for (m in listOf(false, true)) {
            val out = ImageGeometry.transform(pic, ww, hh, rot, m)
            assertEquals(pic.sorted(), out.sorted(), "permutation $rot $m")
        }
        var cur = pic
        var cw = ww
        var ch = hh
        repeat(4) {
            cur = ImageGeometry.transform(cur, cw, ch, 90, false)
            val t = cw; cw = ch; ch = t
        }
        assertContentEquals(pic, cur)
        val once = ImageGeometry.transform(pic, ww, hh, 0, true)
        assertContentEquals(pic, ImageGeometry.transform(once, ww, hh, 0, true))
        val r90 = ImageGeometry.transform(pic, ww, hh, 90, false)
        assertContentEquals(ImageGeometry.transform(pic, ww, hh, 180, false), ImageGeometry.transform(r90, hh, ww, 90, false))
    }

    @Test
    fun rejectsBadInput() {
        for (bad in listOf(-90, 45, 360, 1)) {
            assertFalse(ImageGeometry.isValidRotation(bad))
            assertFailsWith<IllegalArgumentException> { ImageGeometry.transform(src, w, h, bad, false) }
        }
        assertTrue(ImageGeometry.isValidRotation(270))
        assertFailsWith<IllegalArgumentException> { ImageGeometry.transform(IntArray(11), w, h, 0, false) }
        assertFailsWith<IllegalArgumentException> { ImageGeometry.transform(IntArray(0), 0, 0, 0, false) }
        assertFailsWith<IllegalArgumentException> { ImageGeometry.sourceIndex(3, 0, w, h, 90, false) }
    }
}
