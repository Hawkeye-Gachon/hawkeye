package com.personal.driverguard.domain.analysis

import com.personal.driverguard.domain.analysis.calibration.CalibrationFailure
import com.personal.driverguard.domain.analysis.calibration.CalibrationManager
import com.personal.driverguard.domain.analysis.calibration.CalibrationState
import com.personal.driverguard.domain.analysis.calibration.ClosedEyeCalibration
import com.personal.driverguard.domain.analysis.quality.FrameMeasurement
import com.personal.driverguard.domain.analysis.quality.ObservationQualityEvaluator
import com.personal.driverguard.domain.demo.DemoScript
import com.personal.driverguard.domain.obs
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.QualityFlag
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationTest {

    private val config = AnalysisConfig()

    private fun run(manager: CalibrationManager, from: Long, to: Long, make: (Long) -> FaceObservation): CalibrationState {
        var state = manager.state
        var t = from
        while (t < to && state is CalibrationState.Collecting) {
            state = manager.add(make(t))
            t += 66
        }
        return state
    }

    @Test
    fun `순간 깜빡임과 이상값은 기준값을 왜곡하지 않는다`() {
        val m = CalibrationManager(config)
        val state = run(m, 0, 12_000) { t ->
            when {
                t % 1_000 < 120 -> obs(t, 0.08, 0.08) // 깜빡임
                t % 2_500 < 70 -> obs(t, 0.9, 0.9) // 튀는 값
                else -> obs(t, 0.31, 0.29)
            }
        }
        val baseline = (state as CalibrationState.Completed).baseline
        assertEquals(0.31, baseline.leftEarOpen, 0.005)
        assertEquals(0.29, baseline.rightEarOpen, 0.005)
        assertTrue(baseline.sampleCount >= config.calibrationMinSamples)
    }

    @Test
    fun `유효 샘플이 부족하면 연장하고 그래도 부족하면 실패 이유를 준다`() {
        val m = CalibrationManager(config)
        val state = run(m, 0, 12_000) { t -> if (t % 1_000 < 100) obs(t) else FaceObservation.notFound(t) }
        val failed = state as CalibrationState.Failed
        assertEquals(CalibrationFailure.NOT_ENOUGH_VALID_SAMPLES, failed.reason)
        assertEquals(QualityFlag.FACE_NOT_FOUND, failed.dominantIssue)
    }

    @Test
    fun `처음에 얼굴이 없었으면 목표 시간을 넘겨 최대 시간 안에 완료한다`() {
        val m = CalibrationManager(config)
        var completedAt = -1L
        var t = 0L
        while (t < 12_000) {
            val s = m.add(if (t < 3_000) FaceObservation.notFound(t) else obs(t))
            if (s is CalibrationState.Completed) {
                completedAt = t
                break
            }
            t += 66
        }
        assertTrue("완료 시각 $completedAt", completedAt in 6_000..10_000)
    }

    @Test
    fun `자세가 흔들리면 실패한다`() {
        val m = CalibrationManager(config)
        val state = run(m, 0, 12_000) { t -> obs(t, yaw = if ((t / 500) % 2 == 0L) -15.0 else 15.0) }
        assertEquals(CalibrationFailure.POSE_UNSTABLE, (state as CalibrationState.Failed).reason)
    }

    @Test
    fun `눈 감김 캘리브레이션은 감은 EAR 을 기록하고 충분히 감지 않으면 실패한다`() {
        val ok = ClosedEyeCalibration(config, DemoScript.BASELINE)
        var t = 0L
        while (ok.result == null) { ok.add(obs(t, 0.10, 0.11)); t += 66 }
        val baseline = (ok.result as CalibrationState.Completed).baseline
        assertEquals(0.10, baseline.leftEarClosed!!, 1e-9)
        assertTrue(baseline.hasClosedEyeCalibration)

        val notClosed = ClosedEyeCalibration(config, DemoScript.BASELINE)
        t = 0
        while (notClosed.result == null) { notClosed.add(obs(t, 0.28, 0.28)); t += 66 }
        assertEquals(CalibrationFailure.EYES_NOT_CLOSED, (notClosed.result as CalibrationState.Failed).reason)
    }

    @Test
    fun `여러 얼굴이 보이면 운전자를 고르지 않고 무효 처리한다`() {
        val evaluator = ObservationQualityEvaluator(config)
        val o = evaluator.evaluate(FrameMeasurement(0, faceCount = 2, landmarks = null, pose = null, faceLuma = null, faceLumaStdDev = null))
        assertEquals(QualityFlag.MULTIPLE_FACES, o.faceQuality)
        assertEquals(QualityFlag.MULTIPLE_FACES, o.eyeQuality)
    }
}
