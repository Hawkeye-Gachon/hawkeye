package hawkeye.drowsy.core

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** tests/test_pipeline.py:1177-1182 `test_is_finite_helper`를 옮겼다(기대값은 그 Python 시험에 적힌 판정). */
class DriverPipelineHelpersTest {

    @Test
    fun isFiniteHelper() {
        assertTrue(isFinite(0.0))
        assertTrue(isFinite(-1.5e300))
        assertFalse(isFinite(Double.NaN))
        assertFalse(isFinite(Double.POSITIVE_INFINITY))
        assertFalse(isFinite(Double.NEGATIVE_INFINITY))
    }
}
