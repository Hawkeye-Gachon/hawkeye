package com.personal.driverguard.domain.demo

import com.personal.driverguard.domain.analysis.geometry.PoseMath
import com.personal.driverguard.domain.platform.TimeSource
import com.personal.driverguard.entity.CalibrationBaseline
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.Mat3
import com.personal.driverguard.entity.QualityFlag
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow

/**
 * 시연 상황. 카메라·모델 없이 합성 관측을 만들어 실제와 같은 엔진·경고·저장·리포트 흐름을 확인한다.
 * 정해진 시간 동안 재생한 뒤 정상 관측으로 돌아간다. 시연 기록은 항상 DEMO 로 저장된다.
 */
enum class DemoScenario(val durationMs: Long) {
    NORMAL(Long.MAX_VALUE),

    /** 150ms 깜빡임을 1초마다 6번 */
    SHORT_BLINKS(6_000),

    /** 양쪽 눈 감김 2.5초 */
    SUSTAINED_CLOSURE(4_000),

    /** 2초 눈 감김을 3초 간격으로 3번 */
    REPEATED_CLOSURES(15_000),

    /** 입 벌림 3초 */
    MOUTH_OPEN(5_000),

    /** 고개를 운전자 기준 왼쪽으로 40° 돌린 채 4초 */
    HEAD_AWAY(6_000),

    /** 얼굴 미검출 6초 */
    FACE_NOT_FOUND(6_000),

    /** 프레임 공백 4초(관측 자체가 오지 않음) */
    FRAME_GAP(4_000),
}

/** 시연 상황 선택. 주행 화면에서 고른다. */
class DemoController {
    data class Selection(val scenario: DemoScenario, val startedAtMs: Long)

    private val _selection = MutableStateFlow(Selection(DemoScenario.NORMAL, 0L))
    val selection: StateFlow<Selection> = _selection.asStateFlow()

    fun play(scenario: DemoScenario, nowMs: Long) {
        _selection.value = Selection(scenario, nowMs)
    }

    fun reset() {
        _selection.value = Selection(DemoScenario.NORMAL, 0L)
    }
}

object DemoScript {
    /** 시연용 기준값. 실제 사람의 측정값이 아니다. */
    val BASELINE = CalibrationBaseline(
        leftEarOpen = 0.30,
        rightEarOpen = 0.30,
        mouthClosedRatio = 0.05,
        poseRotation = Mat3.IDENTITY,
        yawDeg = 0.0,
        pitchDeg = 0.0,
        rollDeg = 0.0,
        faceWidthRatio = 0.35,
        sampleCount = 0,
        earIqr = 0.0,
        yawIqrDeg = 0.0,
        pitchIqrDeg = 0.0,
    )

    private const val OPEN_EAR = 0.30
    private const val CLOSED_EAR = 0.08

    /**
     * 시나리오 시작 후 [elapsedMs] 시점의 합성 관측. null 이면 그 시각에는 관측이 없다(프레임 공백).
     * 시나리오 시간이 지나면 정상 관측을 돌려준다.
     */
    fun observationAt(scenario: DemoScenario, elapsedMs: Long, timestampMs: Long): FaceObservation? {
        if (elapsedMs >= scenario.durationMs) return normal(timestampMs)
        return when (scenario) {
            DemoScenario.NORMAL -> normal(timestampMs)
            DemoScenario.SHORT_BLINKS -> normal(timestampMs, eyesClosed = elapsedMs % 1_000 < 150)
            DemoScenario.SUSTAINED_CLOSURE -> normal(timestampMs, eyesClosed = elapsedMs in 500 until 3_000)
            DemoScenario.REPEATED_CLOSURES -> normal(timestampMs, eyesClosed = (elapsedMs % 5_000) in 500 until 2_500)
            DemoScenario.MOUTH_OPEN -> normal(timestampMs, mouthOpen = elapsedMs in 500 until 3_500)
            DemoScenario.HEAD_AWAY -> normal(timestampMs, yawDeg = if (elapsedMs in 500 until 4_500) 40.0 else 0.0)
            DemoScenario.FACE_NOT_FOUND -> FaceObservation.notFound(timestampMs)
            DemoScenario.FRAME_GAP -> null
        }
    }

    fun normal(
        timestampMs: Long,
        eyesClosed: Boolean = false,
        mouthOpen: Boolean = false,
        yawDeg: Double = 0.0,
    ): FaceObservation {
        val ear = if (eyesClosed) CLOSED_EAR else OPEN_EAR
        // 고개를 크게 돌리면 실제로는 눈 지표가 무효가 되므로, 품질 판정은 엔진이 기준 대비 각도로 다시 한다
        return FaceObservation(
            timestampMs = timestampMs,
            faceCount = 1,
            faceQuality = QualityFlag.VALID,
            leftEar = ear,
            rightEar = ear,
            eyeQuality = QualityFlag.VALID,
            leftEyeValid = true,
            rightEyeValid = true,
            mouthOpenRatio = if (mouthOpen) 0.6 else 0.05,
            mouthQuality = QualityFlag.VALID,
            pose = PoseMath.poseOf(yawDeg),
            poseQuality = QualityFlag.VALID,
            faceWidthRatio = 0.35,
        )
    }
}

/**
 * 시연 관측 생성기. 사용자가 고른 시나리오를 일정 주기의 합성 관측으로 흘려보낸다.
 * 실제 분석 결과가 아니며, 이 출처의 기록은 항상 DEMO 로 저장된다.
 */
class DemoObservationGenerator(
    private val controller: DemoController,
    private val time: TimeSource,
    private val periodMs: Long = 80,
) {
    fun observations(): Flow<FaceObservation> = flow {
        while (true) {
            val now = time.monotonicMs()
            val selection = controller.selection.value
            val elapsed = now - selection.startedAtMs
            if (selection.scenario != DemoScenario.NORMAL && elapsed >= selection.scenario.durationMs) controller.reset()
            DemoScript.observationAt(selection.scenario, elapsed, now)?.let { emit(it) }
            delay(periodMs)
        }
    }
}
