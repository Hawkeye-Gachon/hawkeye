package hawkeye.drowsy.core.diff

import hawkeye.drowsy.core.DriverPipeline
import hawkeye.drowsy.core.DrowsyInternalApi
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.internal.FrameSignals
import hawkeye.drowsy.core.internal.nearestRankQuantile
import hawkeye.drowsy.core.spec.PipelineSpec
import hawkeye.drowsy.core.spec.RulesConfig
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.DiffData
import org.junit.jupiter.api.Assumptions
import java.io.File
import kotlin.math.floor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * 차등 비교의 알려진 차이 판정과 RAW 비교 범위의 회귀 시험 (검증 f_s6의 발견 1, 2, 4, 5).
 *
 * - 발견 1: [DiffReplay.isNanBaselineThrow]는 판정 8(b)의 예외(기준선 추정기의 정렬이 NaN 표본으로 던지는 것)만 받는다.
 *   예전 규칙은 "세션 어디엔가 유한하지 않은 신호가 있다"뿐이라 다른 예외도 알려진 차이로 셌다.
 *   진짜 예외는 아래 [stream]을 Kotlin 파이프라인에 넣어 얻는다. 바꾸지 않은 참조(Python 3.11.16)는 같은 흐름에서 예외를
 *   내지 않고 600번째 호출에서 기준선(status ok, t_complete_ms 60000)을 확정한다(작업 폴더 f_s6/nan_baseline_python.py, .out).
 * - 발견 2: [DiffReplay.isRoundHalfUpRaise]는 closure_valid 행의 유한하지 않은 closure가 있어야 한다. S-odd/odd/5100027은
 *   참조가 blink.py:339에서 예외를 낸 세션이고(그 파일의 exception), 첫 유한하지 않은 입력은 192행의 gaze_v_rel이다.
 * - 발견 4, 5: RAW는 넣은 모든 프레임의 신호를 비교하고 DriveSession이 돌려준 DriverState의 칸을 비교한다.
 *   components/C-errors_model_inputs.hkd(RAW, 넣은 1700, 받아들인 1687)로 본다. 기대값은 그 파일의 참조 열이다.
 */
@OptIn(DrowsyInternalApi::class)
class DiffReplayCausesTest {

    private val spec: PipelineSpec =
        SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes()).pipeline
    private val rules: RulesConfig = SpecLoader.parseRules(File(TestData.export, "rules.json").readBytes())
    private val minEye = spec.signals.minEyeWidthPx

    private fun frac(x: Double) = x - floor(x)

    private fun sig(t: Long, blinkL: Double, face: Boolean = true, earL: Double = 0.3, eyeWL: Double = 40.0, irisHL: Double? = null) =
        FrameSignals(
            tMs = t, face = face, earL = earL, earR = 0.3, eyeWL = eyeWL, eyeWR = 40.0, blinkL = blinkL, blinkR = 0.2,
            eyeLook = List(8) { 0.0 }, jawOpen = 0.05, yaw = 0.0, pitch = 0.0, roll = 0.0, irisHL = irisHL,
        )

    /** 얼굴 프레임 600개(t = 100·i, blink_l = 0.3·frac(i·0.618...), nanAt에서 NaN)와 t = 60000의 프레임 하나. */
    private fun stream(nanAt: Int): List<FrameSignals> =
        (0 until 600).map { i -> sig(i * 100L, if (i == nanAt) Double.NaN else 0.3 * frac(i * 0.6180339887498949)) } +
            sig(60000L, 0.2)

    /** 흐름을 파이프라인(SIG 입구)에 넣고 던진 예외와 그 프레임 번호. */
    private fun throwOf(frames: List<FrameSignals>): Pair<Int, Throwable>? {
        val pipe = DriverPipeline(spec = spec, rules = rules, baseline = null, featureStartMs = null, model = null)
        for ((i, f) in frames.withIndex()) {
            try {
                pipe.updateFromSignals(f, null)
            } catch (e: Throwable) {
                return i to e
            }
        }
        return null
    }

    @Test
    fun nanBaselineCauseOnlyForTheEstimatorSort() {
        val frames = stream(2)
        val (at, thrown) = assertNotNull(throwOf(frames), "Kotlin must throw on this stream (ruling 8(b))")
        assertEquals(600, at, "the throw is at the deadline frame (Python completes its baseline at call 600)")
        assertEquals(DiffReplay.TIMSORT_MESSAGE, thrown.message)
        val before = frames.subList(0, 600)
        assertTrue(DiffReplay.isNanBaselineThrow(thrown, before, minEye, kotlinHadBaseline = false, pythonHadBaseline = false))

        // 같은 예외라도: NaN이 이 프레임 전의 표본에 없다(예전 규칙은 세션 어디엔가 있으면 셌다).
        val clean = stream(-1).subList(0, 600)
        assertFalse(DiffReplay.isNanBaselineThrow(thrown, clean, minEye, false, false))
        // 표본이 아닌 칸(홍채), 얼굴 없는 프레임, 폭이 min_eye_width_px 아래인 눈의 EAR에만 NaN이 있다.
        val irisOnly = clean.toMutableList().also { it[2] = sig(200L, 0.1, irisHL = Double.NaN) }
        assertFalse(DiffReplay.isNanBaselineThrow(thrown, irisOnly, minEye, false, false))
        val noFace = clean.toMutableList().also { it[2] = sig(200L, Double.NaN, face = false) }
        assertFalse(DiffReplay.isNanBaselineThrow(thrown, noFace, minEye, false, false))
        val narrowEye = clean.toMutableList().also { it[2] = sig(200L, 0.1, earL = Double.NaN, eyeWL = minEye / 2.0) }
        assertFalse(DiffReplay.isNanBaselineThrow(thrown, narrowEye, minEye, false, false))
        // 이미 기준선이 있었다(Kotlin 또는 참조).
        assertFalse(DiffReplay.isNanBaselineThrow(thrown, before, minEye, kotlinHadBaseline = true, pythonHadBaseline = false))
        assertFalse(DiffReplay.isNanBaselineThrow(thrown, before, minEye, kotlinHadBaseline = false, pythonHadBaseline = true))

        // 다른 예외는 NaN 표본이 있어도 세지 않는다.
        val outOfBounds = assertNotNull(runCatching { emptyList<Double>()[1] }.exceptionOrNull())
        assertFalse(DiffReplay.isNanBaselineThrow(outOfBounds, before, minEye, false, false))
        val sameText = IllegalArgumentException(DiffReplay.TIMSORT_MESSAGE)
        assertFalse(DiffReplay.isNanBaselineThrow(sameText, before, minEye, false, false))
        // TimSort의 같은 예외라도 추정기의 마감(tryComplete) 밖에서 났으면 세지 않는다.
        val blinkValues = before.map { it.blinkL }
        val direct = assertNotNull(runCatching { nearestRankQuantile(blinkValues, spec.baseline.blinkRecordQuantiles[0]) }.exceptionOrNull())
        assertEquals(DiffReplay.TIMSORT_MESSAGE, direct.message)
        assertFalse(DiffReplay.isNanBaselineThrow(direct, before, minEye, false, false))
    }

    @Test
    fun roundHalfUpCauseNeedsANonFiniteValidClosure() {
        val ready = DiffCorpus.readyCorpora()
        Assumptions.assumeTrue(ready != null, "no READY_corpora.txt: differential corpora not exported")
        val corpus = DiffCorpus.load("S-odd", ready!!)
        val s = DiffCorpus.read(corpus.entries.first { it.sessionId == "S-odd/odd/5100027" })
        val ex = assertNotNull(s.exception, "the reference raised in this session")
        val fedIndex = (ex["fed_index"] as Long).toInt()
        // DiffReplay의 STATE 입력과 같게: 받아들인 프레임에는 다음 행, 버려진 프레임에는 직전 행을 그 t_ms로.
        val rows = s.states()
        val fed = ArrayList<FrameState>()
        var j = 0
        for (i in 0..fedIndex) fed.add(if (s.accepted[i] != 0) rows[j++] else fed.last().copy(tMs = s.tMs[i]))
        assertTrue(DiffReplay.isRoundHalfUpRaise(ex, fed))

        val firstValidNonFinite = fed.indexOfFirst { it.closureValid && !it.closure.isFinite() }
        assertTrue(firstValidNonFinite > 0)
        val prefix = fed.subList(0, firstValidNonFinite)
        assertTrue(prefix.any { !it.gazeVRel.isFinite() }, "a non-finite gaze_v_rel comes first in this session")
        assertFalse(DiffReplay.isRoundHalfUpRaise(ex, prefix), "other non-finite fields alone are not the cause")
        val overflow = ex + mapOf("type" to "OverflowError", "message" to "cannot convert float infinity to integer")
        assertFalse(DiffReplay.isRoundHalfUpRaise(overflow, fed))
        println("DiffReplayCausesTest: ${s.sid} raise at fed $fedIndex; first closure_valid non-finite closure at fed $firstValidNonFinite")
    }

    @Test
    fun rawComparesSignalsOnEveryFedFrameAndTheDriverState() {
        TestData.requireReadySmall()
        val s = DiffSession(DiffData.readyHkdiff1("components/C-errors_model_inputs.hkd"))
        val accepted = s.accepted.count { it != 0 }
        assertTrue(accepted < s.n, "this file has dropped RAW frames")
        val t = DiffTally()
        DiffReplay.run(s, "RAW", t)
        assertEquals(0L, t.differences, "${t.samples.take(5)}")
        assertEquals(s.n.toLong(), t.fields.getValue("RAW.signals.face").compared, "signals on every fed frame")
        assertEquals(accepted.toLong(), t.fields.getValue("RAW.drive.speed_kmh_null").compared)
        assertEquals(accepted.toLong(), t.fields.getValue("RAW.drive.session_ms").compared)
        assertEquals(5L * accepted, t.fields.getValue("RAW.drive.counters").compared)
        println("DiffReplayCausesTest: C-errors_model_inputs RAW fed ${s.n} accepted $accepted, signals compared on ${s.n} frames, 0 differences")
    }
}
