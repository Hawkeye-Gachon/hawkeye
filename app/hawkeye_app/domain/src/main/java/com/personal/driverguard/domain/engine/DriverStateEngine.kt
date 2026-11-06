package com.personal.driverguard.domain.engine

import com.personal.driverguard.domain.analysis.quality.ObservationQualityEvaluator
import com.personal.driverguard.domain.analysis.quality.RelativeView
import com.personal.driverguard.domain.analysis.temporal.ClosedEyeRatio
import com.personal.driverguard.domain.analysis.temporal.ClosedEyeRatioWindow
import com.personal.driverguard.domain.analysis.temporal.EyeState
import com.personal.driverguard.domain.analysis.temporal.EyeStateTracker
import com.personal.driverguard.domain.analysis.temporal.EyeThresholds
import com.personal.driverguard.domain.analysis.temporal.NodDetector
import com.personal.driverguard.domain.analysis.temporal.RatioStatus
import com.personal.driverguard.domain.analysis.temporal.SustainedTracker
import com.personal.driverguard.entity.AlertKind
import com.personal.driverguard.entity.AlertLevel
import com.personal.driverguard.entity.AnalysisConfig
import com.personal.driverguard.entity.CalibrationBaseline
import com.personal.driverguard.entity.CautionCause
import com.personal.driverguard.entity.DisplayState
import com.personal.driverguard.entity.DrowsinessLevel
import com.personal.driverguard.entity.EventEndReason
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.FaceObservation
import com.personal.driverguard.entity.QualityFlag
import java.util.Locale
import kotlin.math.abs

/**
 * 관측 → 시간 누적 → 상태 판정.
 *
 * 지표별 추적기(눈 감김·입 벌림·고개 방향·끄덕임·분석 불가)가 각각 독립적으로 이벤트를 만들고,
 * 그 위에서 두 판정을 독립적으로 유지한다.
 *  - 졸음 의심: 눈 감김 비율, 반복적 지속 눈 감김, 입 벌림 반복 + 눈 감김. 입 벌림·끄덕임은 보조 신호로 단계만 올린다.
 *  - 시각적 부주의 의심: 기준 자세 대비 고개 방향 이탈의 지속(시선 추적이 아니며 눈만 돌린 경우는 놓칠 수 있다).
 * 표시 상태([DisplayState])는 이 정보를 잃지 않은 채 화면용으로만 집계한다.
 *
 * 시간 규칙
 *  - 모든 지속 시간은 관측 타임스탬프(단조 시계)로 계산한다. 프레임 수를 쓰지 않는다.
 *  - 관측 i 는 [t_i, t_{i+1}) 구간을 대표한다. 간격이 maxContinuityGapMs 를 넘으면 그 구간은 관측 없음이고,
 *    진행 중인 지속 구간(눈 감김 등)은 직전 관측 시각에서 끊는다(공백을 직전 상태로 채우지 않는다).
 *  - 분석 불가는 정상이나 눈 감김으로 바꾸지 않는다.
 *
 * 스레드 안전하지 않다. 한 코루틴에서 순서대로 호출한다. 안드로이드·카메라 객체에 의존하지 않는다.
 */
class DriverStateEngine(
    private val config: AnalysisConfig,
    private val baseline: CalibrationBaseline?,
    startMs: Long,
) {
    private enum class EyeClass { OPEN, CLOSED, INVALID }

    private class Previous(val ts: Long, val eye: EyeClass)

    private val evaluator = ObservationQualityEvaluator(config)
    private val ledger = EventLedger()
    val eyeThresholds: EyeThresholds? = baseline?.let { EyeThresholds.from(it, config) }
    private val eyeTracker = eyeThresholds?.let { EyeStateTracker(it, config.earMedianWindow) }
    private val ratioWindow = ClosedEyeRatioWindow(config.closedRatioWindowMs, config.closedRatioMinCoverage)

    private val closure = SustainedTracker(EventType.EYE_CLOSURE, config.eyeClosureEventMs, AlertLevel.CAUTION, JudgmentRules.EYE_SUSTAINED, ledger)
    private val mouth = SustainedTracker(EventType.MOUTH_OPEN, config.mouthOpenEventMs, AlertLevel.INFO, JudgmentRules.MOUTH_SUSTAINED, ledger)
    private val head = SustainedTracker(EventType.HEAD_AWAY, config.headAwayEventMs, AlertLevel.CAUTION, JudgmentRules.HEAD_AWAY, ledger)
    private val unavailable = SustainedTracker(EventType.ANALYSIS_UNAVAILABLE, config.unavailableNoticeMs, AlertLevel.INFO, JudgmentRules.UNAVAILABLE, ledger)
    private val nod = NodDetector(config.nodPitchDeg, config.nodMinMs, config.nodMaxMs, config.nodWindowMs)

    private var mouthOpen = false
    private var headAway = false

    /** 반복 판정용 이벤트 확인 시각 */
    private val closureTimes = ArrayDeque<Long>()
    private val yawnTimes = ArrayDeque<Long>()
    private val nodEventTimes = ArrayDeque<Long>()

    /** 이미 졸음 판정에 쓴 근거는 다시 쓰지 않는다(이 시각 이후 것만 센다) */
    private var closuresConsumedUntil = Long.MIN_VALUE
    private var yawnsConsumedUntil = Long.MIN_VALUE

    // 졸음 의심 판정
    private var drowsyKey: Long? = null
    private val drowsyRules = linkedSetOf<String>()
    private var drowsyStartMs = 0L
    private var drowsyLastTriggerMs = 0L

    // 시각적 부주의 의심 판정
    private var distractKey: Long? = null
    private var distractStartMs = 0L
    private var distractLastActiveMs = 0L

    /** 확인된 지속 눈 감김이 마지막으로 관측된 시각(주의 표시 유지용) */
    private var lastEyeCautionMs: Long? = null

    // 연속성
    private var referenceMs = startMs
    private var previous: Previous? = null
    private var continuityBroken = false
    private var paused = false
    private var resetWindowOnResume = false
    private var hadValidEye = false
    private var lastUnavailableReason: QualityFlag? = null
    private var lastObservation: FaceObservation? = null
    private var lastRelative: RelativeView? = null

    // 시간 집계
    /** 눈 지표가 유효했던 관측 시간 */
    var validObservedMs = 0L
        private set

    /** 관측은 있었지만 눈 지표가 무효였던 시간 */
    var invalidObservedMs = 0L
        private set

    /** 프레임 공백(관측 없음) 시간. 휴식·앱 중단 시간은 빠진다. */
    var gapMs = 0L
        private set

    init {
        ratioWindow.reset(startMs)
    }

    fun onObservation(obs: FaceObservation): EngineOutput {
        val ts = obs.timestampMs
        val prev = previous
        if (paused || (prev != null && ts <= prev.ts) || ts < referenceMs) {
            // 일시 정지 중이거나 시간이 거꾸로 가는(지연된) 관측은 버린다
            return output(maxOf(ts, prev?.ts ?: ts), mutableSetOf())
        }
        val alerts = mutableSetOf<AlertKind>()
        if (prev != null) {
            val dt = ts - prev.ts
            if (dt <= config.maxContinuityGapMs) attribute(prev.eye, prev.ts, ts) else {
                gapMs += dt
                breakContinuity(prev.ts)
            }
        } else if (ts - referenceMs > config.maxContinuityGapMs) {
            gapMs += ts - referenceMs
        }
        continuityBroken = false

        val rel = evaluator.relativeTo(obs, baseline)
        lastObservation = obs
        lastRelative = rel
        val eyeClass = trackEyes(obs, rel, ts, alerts)
        trackMouth(obs, rel, ts)
        trackHead(rel, ts, alerts)
        previous = Previous(ts, eyeClass)
        return evaluate(ts, alerts)
    }

    /** 프레임이 오지 않아도 주기적으로 불러 공백 처리·판정 유지 시간 만료를 처리한다. */
    fun onTick(nowMs: Long): EngineOutput {
        if (paused) return output(nowMs, mutableSetOf())
        val last = previous?.ts
        val since = last ?: referenceMs
        if (nowMs - since > config.maxContinuityGapMs) {
            if (!continuityBroken) {
                if (last != null) {
                    breakContinuity(last)
                } else if (!unavailable.active) {
                    lastUnavailableReason = QualityFlag.FRAME_GAP
                    unavailable.onActive(referenceMs)
                }
            }
            continuityBroken = true
            unavailable.onActive(nowMs)
        }
        return evaluate(nowMs, mutableSetOf())
    }

    /**
     * 휴식 시작·앱 백그라운드 등으로 분석을 멈춘다. 진행 중인 이벤트와 판정을 [reason] 으로 닫는다.
     * 휴식이면 반복 근거와 눈 감김 비율 창도 비운다(휴식 전 근거로 휴식 후를 판정하지 않는다).
     */
    fun pause(nowMs: Long, reason: EventEndReason): EngineOutput {
        if (paused) return output(nowMs, mutableSetOf())
        closeTail(nowMs)
        closeEverything(nowMs, reason)
        paused = true
        if (reason == EventEndReason.REST_STARTED) {
            closureTimes.clear()
            yawnTimes.clear()
            nodEventTimes.clear()
            nod.clearCount()
            resetWindowOnResume = true
        }
        return output(nowMs, mutableSetOf())
    }

    /** 분석을 다시 시작한다. 멈춘 동안은 관측 시간에도 공백에도 넣지 않는다. */
    fun resume(nowMs: Long) {
        if (!paused) return
        paused = false
        referenceMs = nowMs
        previous = null
        continuityBroken = false
        hadValidEye = false
        if (resetWindowOnResume) ratioWindow.reset(nowMs)
        resetWindowOnResume = false
    }

    /** 주행 종료. 열린 이벤트를 모두 닫는다. */
    fun finish(nowMs: Long): EngineOutput {
        if (!paused) closeTail(nowMs)
        closeEverything(nowMs, EventEndReason.TRIP_ENDED)
        paused = true
        return output(nowMs, mutableSetOf())
    }

    private fun trackEyes(obs: FaceObservation, rel: RelativeView, ts: Long, alerts: MutableSet<AlertKind>): EyeClass {
        val tracker = eyeTracker
        val leftEar = obs.leftEar
        val rightEar = obs.rightEar
        if (tracker == null || !rel.eyeQuality.isValid || leftEar == null || rightEar == null) {
            tracker?.reset()
            closure.breakOff(ts, EventEndReason.OBSERVATION_INVALID)
            lastUnavailableReason = rel.eyeQuality
            unavailable.onActive(ts)
            return EyeClass.INVALID
        }
        hadValidEye = true
        unavailable.onInactive(ts)
        lastUnavailableReason = null
        val state = tracker.update(leftEar, rightEar)
        if (state == EyeState.CLOSED) {
            val left = tracker.filteredLeft
            val right = tracker.filteredRight
            if (closure.onActive(ts) { min("minEarL", left); min("minEarR", right) }) {
                closureTimes.addLast(ts)
                alerts += AlertKind.EYE_CAUTION
            }
            if (closure.confirmed) lastEyeCautionMs = ts
            return EyeClass.CLOSED
        }
        closure.onInactive(ts)
        return EyeClass.OPEN
    }

    private fun trackMouth(obs: FaceObservation, rel: RelativeView, ts: Long) {
        val base = baseline
        val ratio = obs.mouthOpenRatio
        if (base == null || !rel.mouthQuality.isValid || ratio == null) {
            mouthOpen = false
            mouth.breakOff(ts, EventEndReason.OBSERVATION_INVALID)
            return
        }
        mouthOpen = if (mouthOpen) ratio > base.mouthClosedRatio + config.mouthCloseDelta else ratio > base.mouthClosedRatio + config.mouthOpenDelta
        if (!mouthOpen) {
            mouth.onInactive(ts)
            return
        }
        val refractory = !mouth.active && mouth.lastConfirmedEndMs?.let { ts - it < config.mouthRefractoryMs } == true
        if (refractory) return
        if (mouth.onActive(ts) { max("maxRatio", ratio) }) yawnTimes.addLast(ts)
    }

    private fun trackHead(rel: RelativeView, ts: Long, alerts: MutableSet<AlertKind>) {
        val yaw = rel.yawDeg
        val pitch = rel.pitchDeg
        if (!rel.poseQuality.isValid || yaw == null || pitch == null) {
            headAway = false
            head.breakOff(ts, EventEndReason.OBSERVATION_INVALID)
            nod.reset()
            return
        }
        val ay = abs(yaw)
        val ap = abs(pitch)
        headAway = if (headAway) {
            !(ay < config.headYawAwayDeg * config.headReturnFactor && ap < config.headPitchAwayDeg * config.headReturnFactor)
        } else {
            ay > config.headYawAwayDeg || ap > config.headPitchAwayDeg
        }
        if (headAway) {
            if (head.onActive(ts) { max("maxAbsYaw", ay); max("maxAbsPitch", ap) }) {
                alerts += AlertKind.HEAD_CAUTION
                if (distractKey == null) {
                    distractStartMs = head.candidateStartMs ?: ts
                    distractKey = ledger.open(EventType.DISTRACTION_SUSPECTED, distractStartMs, ts, AlertLevel.CAUTION, listOf(JudgmentRules.HEAD_AWAY), head.extremes.format())
                }
            }
            if (head.confirmed) distractLastActiveMs = ts
        } else {
            head.onInactive(ts)
        }

        if (config.nodEnabled && nod.update(pitch, ts) && nod.count(ts) >= config.nodRepeatCount) {
            val first = nod.firstNodMs() ?: ts
            val key = ledger.open(EventType.HEAD_NOD, first, ts, AlertLevel.INFO, listOf(JudgmentRules.NOD_REPEATED), "nods=${config.nodRepeatCount}")
            ledger.close(key, ts, ts - first, EventEndReason.RECOVERED, "nods=${config.nodRepeatCount}")
            nodEventTimes.addLast(ts)
            nod.clearCount()
        }
    }

    /** 관측 i 의 상태로 [from, to) 구간을 채운다(연속 관측일 때만 불린다). */
    private fun attribute(eye: EyeClass, from: Long, to: Long) {
        val d = to - from
        if (eye == EyeClass.INVALID) {
            invalidObservedMs += d
        } else {
            validObservedMs += d
            ratioWindow.add(from, to, eye == EyeClass.CLOSED)
        }
    }

    /** 관측 공백: 진행 중인 지속 구간을 마지막 관측 시각에서 끊고 분석 불가 후보를 시작한다. */
    private fun breakContinuity(lastObservedMs: Long) {
        eyeTracker?.reset()
        closure.breakOff(lastObservedMs, EventEndReason.OBSERVATION_GAP)
        mouth.breakOff(lastObservedMs, EventEndReason.OBSERVATION_GAP)
        head.breakOff(lastObservedMs, EventEndReason.OBSERVATION_GAP)
        mouthOpen = false
        headAway = false
        nod.reset()
        if (!unavailable.active) {
            lastUnavailableReason = QualityFlag.FRAME_GAP
            unavailable.onActive(lastObservedMs)
        }
    }

    /** 마지막 관측부터 [nowMs] 까지를 집계에 넣는다 */
    private fun closeTail(nowMs: Long) {
        val prev = previous
        if (prev == null) {
            if (nowMs - referenceMs > config.maxContinuityGapMs) gapMs += nowMs - referenceMs
            return
        }
        val d = nowMs - prev.ts
        if (d <= config.maxContinuityGapMs) attribute(prev.eye, prev.ts, nowMs) else gapMs += d
    }

    private fun closeEverything(nowMs: Long, reason: EventEndReason) {
        val lastObserved = previous?.ts ?: nowMs
        closure.breakOff(lastObserved, reason)
        mouth.breakOff(lastObserved, reason)
        head.breakOff(lastObserved, reason)
        unavailable.breakOff(nowMs, reason)
        eyeTracker?.reset()
        nod.reset()
        mouthOpen = false
        headAway = false
        drowsyKey?.let { ledger.close(it, nowMs, nowMs - drowsyStartMs, reason, drowsyMetrics(nowMs)) }
        drowsyKey = null
        drowsyRules.clear()
        distractKey?.let {
            val end = maxOf(distractLastActiveMs, distractStartMs)
            ledger.close(it, end, end - distractStartMs, reason, "")
        }
        distractKey = null
        lastEyeCautionMs = null
        lastUnavailableReason = null
    }

    private fun evaluate(nowMs: Long, alerts: MutableSet<AlertKind>): EngineOutput {
        trim(closureTimes, nowMs, config.repeatWindowMs)
        trim(yawnTimes, nowMs, config.repeatWindowMs)
        trim(nodEventTimes, nowMs, config.repeatWindowMs)

        // ── 졸음 의심 ──
        val ratio = ratioWindow.evaluate(nowMs)
        val triggers = linkedSetOf<String>()
        var evidenceStart = nowMs
        val r = ratio.ratio
        if (ratio.status == RatioStatus.AVAILABLE && r != null && r >= config.closedRatioDrowsyThreshold) {
            triggers += JudgmentRules.CLOSED_RATIO
            evidenceStart = minOf(evidenceStart, nowMs - ratio.windowUsedMs)
        }
        val freshClosures = closureTimes.filter { it > closuresConsumedUntil }
        if (freshClosures.size >= config.repeatedClosureCount) {
            triggers += JudgmentRules.REPEATED_CLOSURE
            evidenceStart = minOf(evidenceStart, freshClosures.first())
            closuresConsumedUntil = freshClosures.last()
        }
        val freshYawns = yawnTimes.filter { it > yawnsConsumedUntil }
        if (freshYawns.size >= config.yawnSupportCount && closureTimes.isNotEmpty()) {
            triggers += JudgmentRules.YAWN_WITH_CLOSURE
            evidenceStart = minOf(evidenceStart, freshYawns.first(), closureTimes.first())
            yawnsConsumedUntil = freshYawns.last()
        }
        if (triggers.isNotEmpty()) {
            drowsyLastTriggerMs = nowMs
            if (drowsyKey == null) {
                drowsyStartMs = evidenceStart
                drowsyRules += triggers
                drowsyKey = ledger.open(EventType.DROWSINESS_SUSPECTED, evidenceStart, nowMs, AlertLevel.REST, drowsyRules.toList(), drowsyMetrics(nowMs))
            } else if (!drowsyRules.containsAll(triggers)) {
                drowsyRules += triggers
                ledger.update(drowsyKey!!, AlertLevel.REST, drowsyRules.toList(), drowsyMetrics(nowMs))
            }
        }
        drowsyKey?.let { key ->
            if (nowMs - drowsyLastTriggerMs > config.drowsinessHoldMs) {
                ledger.close(key, nowMs, nowMs - drowsyStartMs, EventEndReason.HOLD_EXPIRED, drowsyMetrics(nowMs))
                drowsyKey = null
                drowsyRules.clear()
            } else {
                val support = buildList {
                    if (yawnTimes.isNotEmpty()) add(JudgmentRules.SUPPORT_YAWN)
                    if (config.nodContributesToDrowsiness && nodEventTimes.isNotEmpty()) add(JudgmentRules.SUPPORT_NOD)
                }
                if (!drowsyRules.containsAll(support)) {
                    drowsyRules += support
                    ledger.update(key, AlertLevel.REST, drowsyRules.toList(), drowsyMetrics(nowMs))
                }
                alerts += AlertKind.REST
            }
        }

        // ── 시각적 부주의 의심 ──
        distractKey?.let { key ->
            if (!head.confirmed && nowMs - distractLastActiveMs > config.cautionHoldMs) {
                val end = maxOf(distractLastActiveMs, distractStartMs)
                ledger.close(key, end, end - distractStartMs, EventEndReason.HOLD_EXPIRED, "")
                distractKey = null
            }
        }

        // 이벤트 지표 요약 갱신(확인된 이벤트만)
        if (closure.confirmed) closure.refreshMetrics()

        return output(nowMs, alerts, ratio)
    }

    private fun output(
        nowMs: Long,
        alerts: MutableSet<AlertKind>,
        ratio: ClosedEyeRatio = ratioWindow.evaluate(nowMs),
    ): EngineOutput {
        val eyeCaution = closure.confirmed || lastEyeCautionMs?.let { nowMs - it <= config.cautionHoldMs } == true
        val display = when {
            paused -> DisplayState.PAUSED
            drowsyKey != null -> DisplayState.REST_RECOMMENDED
            eyeCaution || distractKey != null -> DisplayState.CAUTION
            unavailable.confirmed -> DisplayState.UNAVAILABLE
            baseline == null || !hadValidEye -> DisplayState.PREPARING
            else -> DisplayState.OBSERVING
        }
        if (display == DisplayState.UNAVAILABLE) alerts += AlertKind.UNAVAILABLE
        val cause = when {
            display != DisplayState.CAUTION -> null
            eyeCaution -> CautionCause.EYE_CLOSURE
            else -> CautionCause.HEAD_AWAY
        }
        val drowsiness = when {
            drowsyKey == null -> DrowsinessLevel.NONE
            JudgmentRules.SUPPORT_YAWN in drowsyRules || JudgmentRules.SUPPORT_NOD in drowsyRules -> DrowsinessLevel.SUSPECTED_WITH_SUPPORT
            else -> DrowsinessLevel.SUSPECTED
        }
        return EngineOutput(
            display = display,
            cautionCause = cause,
            drowsiness = drowsiness,
            distraction = distractKey != null,
            unavailableReason = if (unavailable.active) lastUnavailableReason else null,
            changes = ledger.drain(),
            alertRequests = if (paused) emptySet() else alerts,
            snapshot = snapshot(nowMs, ratio),
        )
    }

    private fun snapshot(nowMs: Long, ratio: ClosedEyeRatio): EngineSnapshot {
        val obs = lastObservation
        val rel = lastRelative
        return EngineSnapshot(
            timestampMs = nowMs,
            faceQuality = obs?.faceQuality ?: QualityFlag.FRAME_GAP,
            eyeQuality = rel?.eyeQuality ?: QualityFlag.FRAME_GAP,
            mouthQuality = rel?.mouthQuality ?: QualityFlag.FRAME_GAP,
            poseQuality = rel?.poseQuality ?: QualityFlag.FRAME_GAP,
            faceCount = obs?.faceCount ?: 0,
            leftEar = obs?.leftEar,
            rightEar = obs?.rightEar,
            filteredLeftEar = eyeTracker?.filteredLeft,
            filteredRightEar = eyeTracker?.filteredRight,
            thresholds = eyeThresholds,
            eyeState = eyeTracker?.state ?: EyeState.UNKNOWN,
            eyeClosedMs = closure.durationMs(previous?.ts ?: nowMs),
            closedRatio = ratio,
            mouthRatio = obs?.mouthOpenRatio,
            mouthOpenThreshold = baseline?.let { it.mouthClosedRatio + config.mouthOpenDelta },
            mouthOpenMs = mouth.durationMs(previous?.ts ?: nowMs),
            yawDeg = rel?.yawDeg,
            pitchDeg = rel?.pitchDeg,
            rollDeg = rel?.rollDeg,
            headAwayMs = head.durationMs(previous?.ts ?: nowMs),
            nodCount = nod.count(nowMs),
            closureCount = closureTimes.size,
            yawnCount = yawnTimes.size,
            unavailableMs = unavailable.durationMs(nowMs),
            drowsinessRules = drowsyRules.toSet(),
        )
    }

    private fun drowsyMetrics(nowMs: Long): String {
        val ratio = ratioWindow.evaluate(nowMs)
        return buildString {
            append("ratio=").append(ratio.ratio?.let { String.format(Locale.US, "%.3f", it) } ?: "-")
            append(", coverage=").append(String.format(Locale.US, "%.2f", ratio.coverage))
            append(", ratioStatus=").append(ratio.status.name)
            append(", closures=").append(closureTimes.size)
            append(", yawns=").append(yawnTimes.size)
            if (config.nodContributesToDrowsiness) append(", nods=").append(nodEventTimes.size)
        }
    }

    private fun trim(times: ArrayDeque<Long>, nowMs: Long, windowMs: Long) {
        while (times.isNotEmpty() && nowMs - times.first() > windowMs) times.removeFirst()
    }
}
