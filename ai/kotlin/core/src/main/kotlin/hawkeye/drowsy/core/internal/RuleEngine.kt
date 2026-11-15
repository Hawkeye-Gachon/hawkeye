package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.CLASS_DROWSY
import hawkeye.drowsy.core.ClassProbs
import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.DrowsySource
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.F_FACE_MISSING_RATIO_60
import hawkeye.drowsy.core.F_PERCLOS_60
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.GazeZone
import hawkeye.drowsy.core.Phase
import hawkeye.drowsy.core.Source
import hawkeye.drowsy.core.StepFeatures
import hawkeye.drowsy.core.spec.RulesConfig

// 원본: drowsy/rules.py:415-873 `RuleEngine`과 1015-1019 `_ratio` (SPEC 12장). 계획 5.6절의 선언 그대로다.
// 이벤트 공통 규칙(발행 시각, duration_ms, severity, provisional, confidence의 구간과 신호), 시간 규칙(엄격히 증가하는
// t_ms, 10초 넘는 끊김의 초기화 순서), 종류별 규칙과 원본이 정한 가정(rules.py:25-26, 48-52, 429-430, 438-440, 455-456,
// 471-472)은 원본 모듈·클래스 설명과 SPEC 12장에 있다. 이 파일은 그것을 문장 순서 그대로 옮긴다.
// 시각, 길이, 유효 시간 누적은 모두 Long이다(R1). 정수 둘의 나눗셈은 피연산자마다 toDouble()이다(R3, rules.py:661, 1019).
// 이벤트를 만드는 곳은 이름 붙인 인자로 부른다(R10: Long 둘과 Boolean이 나란히 있다, rules.py:600, 643).
//
// 가정(A1): 사칙연산과 비교는 JVM·Android와 CPython에서 같은 double을 낸다. confidence(정수 ms의 몫), face_ratio
// (정수 ms / window_ms), 1 − face_missing_ratio_60은 이 가정 아래에서 Python과 비트가 같다(엄격 층: source가 model이
// 아닌 confidence는 정확히 같아야 한다, 계획 9.0절).

/** rules.py:94-96 `_LEVEL_*`: 한 감김이 도달한 가장 높은 종류. `<` 비교가 있으므로 Int다(계획 5.6절). */
private const val LEVEL_NONE: Int = 0
private const val LEVEL_MICROSLEEP: Int = 1
private const val LEVEL_SLEEP: Int = 2

/**
 * rules.py:415-873. 규칙 엔진.
 *
 * - [update]: 받아들인 프레임마다 한 번 부른다. 이 프레임에서 생긴 상태 이벤트를 발행 순서대로 돌려준다(calibrating,
 *   unable_to_analyze, gaze_away_long, vats, microsleep, sleep 순서로 판정한다). 10초 넘는 끊김 뒤의 첫 프레임에서는
 *   초기화가 내는 end가 그 앞에 온다.
 * - [onStep]: 스텝이 발행될 때마다 부른다. drowsy 이벤트를 돌려준다. 프레임이 스텝을 발행시켰으면 그 프레임의 [update]보다
 *   먼저, 오래된 스텝부터 부른다.
 * tMs는 [update] 호출마다 엄격히 증가해야 한다(아니면 IllegalArgumentException, Python은 ValueError; 상태를 바꾸기 전에
 * 검사한다). 스레드에 안전하지 않다. 상태는 인스턴스에만 있다(R19).
 */
internal class RuleEngine(private val cfg: RulesConfig) {

    private val maxHoldMs: Long = cfg.clock.maxHoldMs
    private val resetGapMs: Long = cfg.clock.resetGapMs

    // 직전 프레임
    private var hasPrev = false
    private var firstTMs: Long = 0L // 세션의 첫 프레임. 끊김 초기화 뒤에는 그 프레임(분석 불가를 판정하지 않는 구간의 시작)
    private var prevTMs: Long = 0L
    private var prevFace = false
    private var prevPoseValid = false
    private var prevClosureValid = false

    // 유효 시간 누적(유지 규칙 적용). 이벤트 구간의 시작에서 값을 적어 두고 차로 비율을 구한다.
    private var faceMs: Long = 0L
    private var poseMs: Long = 0L
    private var closureValidMs: Long = 0L
    private val faceWindow = IntervalWindow(cfg.unableToAnalyze.windowMs)

    // vats의 t_start_ms는 시작 프레임보다 과거(최근 30초 안)다. 시작할 때 "t_start_ms까지의 pose_valid 누적"을 구하려고
    // 최근 30초의 pose_valid 구간을 들고 있는다(그 용도로만 쓴다).
    private val poseWindow = IntervalWindow(cfg.vats.windowMs)

    // calibrating
    private var calibrating = false
    private var calibStartMs: Long = 0L

    // unable_to_analyze
    private var unable = false
    private var unableStartMs: Long = 0L
    private var unableFaceMs0: Long = 0L

    // 시선
    private val zone = ForwardZone(cfg = cfg.forwardZone, maxHoldMs = cfg.clock.maxHoldMs, awayWindowMs = cfg.vats.windowMs)
    private var gazeActive = false
    private var gazeStartMs: Long = 0L
    private var gazePoseMs0: Long = 0L
    private var gazeSource: Source = Source.HEAD_POSE
    private var vatsActive = false
    private var vatsStartMs: Long = 0L
    private var vatsPoseMs0: Long = 0L // t_start_ms 시점의 pose_valid 누적 시간
    private var vatsSource: Source = Source.HEAD_POSE

    // 감김
    private val detector = ClosureRunDetector(cfg.closure.detector)
    private var runTracked = false // 진행 중인 감김의 시작을 기록했는지
    private var runStartMs: Long = 0L // 그 감김의 start_ms(검출기의 runStartMs()). 이벤트의 t_start_ms다
    private var runFirstMs: Long = 0L // 그 감김을 처음 본 프레임
    private var runValidMs0: Long = 0L // 그 프레임에서의 closure_valid 누적 시간
    private var closureLevel: Int = LEVEL_NONE

    // 졸음
    private val modelRule = StepHysteresis(
        onsetSteps = cfg.drowsiness.model.onsetSteps,
        endSteps = cfg.drowsiness.model.endSteps,
        refractoryMs = cfg.drowsiness.model.refractoryMs,
    )
    private val perclosRule = StepHysteresis(
        onsetSteps = cfg.drowsiness.perclos.onsetSteps,
        endSteps = cfg.drowsiness.perclos.endSteps,
        refractoryMs = cfg.drowsiness.perclos.refractoryMs,
    )
    private var drowsyActive = false
    private var drowsyStartMs: Long = 0L
    private var drowsySource: Source = Source.PERCLOS

    /**
     * rules.py:534-543. 마지막 프레임까지의 전방 주시 상태(표시용, SPEC 12.10). 읽을 때마다 계산한다.
     * 확정된 상태(디바운스 뒤)를 돌려준다. 마지막 프레임의 표지가 UNKNOWN이면(자세 무효, 얼굴 소실 3초 경과, 분석 불가)
     * 상태와 무관하게 UNKNOWN이다.
     */
    val gazeZone: GazeZone
        get() {
            if (zone.label == GazeZone.UNKNOWN) {
                return GazeZone.UNKNOWN
            }
            return zone.zone
        }

    // ------------------------------------------------------------------
    // 프레임
    // ------------------------------------------------------------------

    /** rules.py:549-585. 프레임 하나를 처리하고 이 프레임에서 생긴 이벤트를 발행 순서대로 돌려준다. */
    fun update(state: FrameState): List<DriverStateEvent> {
        val t = state.tMs
        if (hasPrev && t <= prevTMs) {
            throw IllegalArgumentException("t_ms must be strictly greater than the previous frame's t_ms")
        }
        val events = ArrayList<DriverStateEvent>()

        if (!hasPrev) {
            firstTMs = t
        } else {
            val gapMs = t - prevTMs
            var heldMs = gapMs
            if (heldMs > maxHoldMs) {
                heldMs = maxHoldMs
            }
            val heldEndMs = prevTMs + heldMs
            if (prevFace) {
                faceMs += heldMs
                faceWindow.add(startMs = prevTMs, endMs = heldEndMs)
            }
            if (prevPoseValid) {
                poseMs += heldMs
                poseWindow.add(startMs = prevTMs, endMs = heldEndMs)
            }
            if (prevClosureValid) {
                closureValidMs += heldMs
            }
            if (gapMs > resetGapMs) {
                resetAfterGap(state, events)
            }
        }

        ruleCalibrating(state, events)
        val analyzable = ruleUnable(state, events)
        ruleGaze(state, analyzable, events)
        ruleClosure(state, analyzable, events)

        hasPrev = true
        prevTMs = t
        prevFace = state.face
        prevPoseValid = state.poseValid
        prevClosureValid = state.closureValid
        return events
    }

    /**
     * rules.py:587-635 `_reset_after_gap`. 10초 넘는 끊김 뒤의 초기화. 진행 중이던 상태를 끝내고 세션을 시작할 때의 상태로
     * 되돌린다. end를 내는 순서는 drowsy, unable_to_analyze, gaze_away_long, microsleep 또는 sleep이다. vats의 end는
     * 여기서 내지 않는다(합을 비워 두면 이 프레임의 ruleGaze가 낸다). calibrating 상태와 유효 시간 누적값(faceMs, poseMs,
     * closureValidMs)은 건드리지 않는다.
     */
    private fun resetAfterGap(state: FrameState, events: MutableList<DriverStateEvent>) {
        val t = state.tMs

        // 졸음: 값을 실어 줄 스텝이 없으므로 confidence는 0.0이다. provisional은 다른 drowsy 이벤트처럼 false다.
        if (drowsyActive) {
            drowsyActive = false
            events.add(
                event(
                    type = EventType.DROWSY,
                    tStartMs = drowsyStartMs,
                    tEmitMs = t,
                    confidence = 0.0,
                    source = drowsySource,
                    phase = Phase.END,
                    provisional = false,
                ),
            )
        }
        modelRule.reset()
        perclosRule.reset()

        // 분석 불가: 끝내고, 창을 비우고, 판정하지 않는 구간(처음 window_ms)을 이 프레임부터 다시 센다.
        // 직전 프레임이 유지된 시간은 호출 전에 이미 faceMs에 들어갔으므로 신뢰도는 공통 규칙 그대로다.
        if (unable) {
            unable = false
            val confidence = ratio(numeratorMs = faceMs - unableFaceMs0, denominatorMs = t - unableStartMs)
            events.add(
                event(
                    type = EventType.UNABLE_TO_ANALYZE,
                    tStartMs = unableStartMs,
                    tEmitMs = t,
                    confidence = confidence,
                    source = Source.FACE,
                    phase = Phase.END,
                    provisional = state.provisional,
                ),
            )
        }
        faceWindow.clear()
        firstTMs = t

        // 시선: 전방 주시 상태와 VATS 누적
        if (gazeActive) {
            gazeActive = false
            events.add(gazeEvent(state, Phase.END))
        }
        zone.reset()

        // 감김
        if (closureLevel != LEVEL_NONE) {
            events.add(closureEvent(state, closureLevel, Phase.END))
            closureLevel = LEVEL_NONE
        }
        detector.reset()
        runTracked = false
    }

    /** rules.py:637-651 `_rule_calibrating`. */
    private fun ruleCalibrating(state: FrameState, events: MutableList<DriverStateEvent>) {
        val t = state.tMs
        if (!hasPrev) {
            if (state.provisional) {
                calibrating = true
                calibStartMs = t
                events.add(
                    event(
                        type = EventType.CALIBRATING,
                        tStartMs = t,
                        tEmitMs = t,
                        confidence = 0.0,
                        source = Source.FACE,
                        phase = Phase.ONSET,
                        provisional = state.provisional,
                    ),
                )
            }
        } else if (calibrating && !state.provisional) {
            calibrating = false
            val confidence = ratio(numeratorMs = faceMs, denominatorMs = t - calibStartMs)
            events.add(
                event(
                    type = EventType.CALIBRATING,
                    tStartMs = calibStartMs,
                    tEmitMs = t,
                    confidence = confidence,
                    source = Source.FACE,
                    phase = Phase.END,
                    provisional = state.provisional,
                ),
            )
        }
    }

    /** rules.py:653-684 `_rule_unable`. 분석 불가를 판정하고, 이 프레임을 시선과 감김 규칙이 써도 되는지(분석 불가가 아닌지)를 돌려준다. */
    private fun ruleUnable(state: FrameState, events: MutableList<DriverStateEvent>): Boolean {
        val uCfg = cfg.unableToAnalyze
        val t = state.tMs
        faceWindow.trim(t)
        // 세션의 첫 프레임(끊김 초기화가 있었으면 그 뒤의 첫 프레임)부터 window_ms가 지나기 전에는 판정하지 않는다.
        if (t - firstTMs < uCfg.windowMs) {
            return true
        }
        // Python `float(total_ms()) / float(window_ms)` (R3)
        val faceRatio = faceWindow.totalMs().toDouble() / uCfg.windowMs.toDouble()
        if (!unable) {
            if (faceRatio < uCfg.minFaceRatio) {
                unable = true
                unableStartMs = t
                unableFaceMs0 = faceMs
                events.add(
                    event(
                        type = EventType.UNABLE_TO_ANALYZE,
                        tStartMs = t,
                        tEmitMs = t,
                        confidence = 0.0,
                        source = Source.FACE,
                        phase = Phase.ONSET,
                        provisional = state.provisional,
                    ),
                )
            }
        } else if (faceRatio >= uCfg.minFaceRatio) {
            unable = false
            val confidence = ratio(numeratorMs = faceMs - unableFaceMs0, denominatorMs = t - unableStartMs)
            events.add(
                event(
                    type = EventType.UNABLE_TO_ANALYZE,
                    tStartMs = unableStartMs,
                    tEmitMs = t,
                    confidence = confidence,
                    source = Source.FACE,
                    phase = Phase.END,
                    provisional = state.provisional,
                ),
            )
        }
        return !unable
    }

    /** rules.py:686-731 `_rule_gaze`. gaze_away_long과 vats. */
    private fun ruleGaze(state: FrameState, analyzable: Boolean, events: MutableList<DriverStateEvent>) {
        val t = state.tMs
        val zone = this.zone
        zone.update(state, analyzable, poseMs)

        // gaze_away_long
        if (gazeActive) {
            if (zone.confirmedIn) {
                gazeActive = false
                events.add(gazeEvent(state, Phase.END))
            }
        } else if (zone.zone == GazeZone.OUT &&
            (zone.label == GazeZone.OUT || (zone.label == GazeZone.UNKNOWN && analyzable && !zone.candidatePending))
        ) {
            // 표지가 UNKNOWN인 프레임도 본다: 얼굴 소실 뒤 인정 구간 안에서 임계 시간이 찼는데 그 구간에 프레임이 없었으면
            // 여기서 낸다. UNKNOWN 동안에는 unknownMs가 함께 늘어 연속 시간이 멈춰 있으므로 일찍 나지 않는다. 분석 불가 중인
            // 프레임(analyzable = false)과 영역 안 후보가 걸린 프레임(표지 IN)에서는 내지 않는다(rules.py:699-704).
            if (t - zone.outStartMs - zone.unknownMs >= cfg.gazeAwayLong.onsetMs) {
                gazeActive = true
                gazeStartMs = zone.outStartMs
                gazePoseMs0 = zone.outPoseMs
                gazeSource = zone.outSource
                events.add(gazeEvent(state, Phase.ONSET))
            }
        }

        // vats
        val vats = cfg.vats
        if (vats.resetEnabled && zone.zone == GazeZone.IN && zone.label == GazeZone.IN) {
            if (t - zone.inStartMs >= vats.resetForwardMs) {
                zone.away.clear()
            }
        }
        zone.away.trim(t)
        poseWindow.trim(t)
        val awayMs = zone.away.totalMs()
        if (!vatsActive) {
            if (awayMs >= vats.cumulativeMs) {
                vatsActive = true
                vatsStartMs = zone.away.firstStartMs()
                // t_start_ms는 방금 자른 30초 창 안에 있으므로, 그 뒤의 pose_valid 시간이 전부 poseWindow에 남아 있다.
                // 전체 누적에서 빼면 t_start_ms 시점의 누적값이고, 이후의 유효 시간은 gaze_away_long처럼 차로 구한다.
                vatsPoseMs0 = poseMs - poseWindow.sinceMs(vatsStartMs)
                vatsSource = zone.outSource
                events.add(vatsEvent(state, Phase.ONSET))
            }
        } else if (awayMs < vats.cumulativeMs) {
            vatsActive = false
            events.add(vatsEvent(state, Phase.END))
        }
    }

    /** rules.py:733-772 `_rule_closure`. microsleep과 sleep. */
    private fun ruleClosure(state: FrameState, analyzable: Boolean, events: MutableList<DriverStateEvent>) {
        val cCfg = cfg.closure
        val t = state.tMs
        val detector = this.detector
        val valid = state.closureValid && analyzable
        val run = detector.update(t, state.closure, valid)

        // 유효한 감김 프레임(closure ≥ 진입)에서 진행 중인 감김의 길이와 시작을 검출기에서 읽는다.
        // 길이(ongoingMs)는 closure가 진입 임계 이상이었던 시간의 합이라 t − startMs보다 짧을 수 있다.
        // 그래서 감김의 시작은 길이에서 거꾸로 계산하지 않고 runStartMs()로 받는다.
        val closedFrame = valid && state.closure >= cCfg.detector.enterThr && detector.state != RunState.OPEN
        var ongoingMs = 0L
        var startMs = 0L
        if (closedFrame) {
            ongoingMs = detector.ongoingMs()
            startMs = detector.runStartMs()
        }

        // 기록해 둔 감김이 끝났으면(같은 프레임에서 새 감김이 시작됐어도) 도달한 가장 높은 종류의 end를 하나 낸다.
        var ended = run != null || detector.state == RunState.OPEN
        if (closedFrame && runTracked && startMs != runStartMs) {
            ended = true
        }
        if (ended) {
            if (closureLevel != LEVEL_NONE) {
                events.add(closureEvent(state, closureLevel, Phase.END))
                closureLevel = LEVEL_NONE
            }
            runTracked = false
        }

        // 임계 시간은 유효한 감김 프레임에서만 판정한다.
        if (closedFrame) {
            if (!runTracked) {
                runTracked = true
                runStartMs = startMs
                runFirstMs = t
                runValidMs0 = closureValidMs
            }
            if (closureLevel < LEVEL_MICROSLEEP && ongoingMs >= cCfg.microsleepMs) {
                closureLevel = LEVEL_MICROSLEEP
                events.add(closureEvent(state, LEVEL_MICROSLEEP, Phase.ONSET))
            }
            if (closureLevel < LEVEL_SLEEP && ongoingMs >= cCfg.sleepMs) {
                closureLevel = LEVEL_SLEEP
                events.add(closureEvent(state, LEVEL_SLEEP, Phase.ONSET))
            }
        }
    }

    // ------------------------------------------------------------------
    // 스텝
    // ------------------------------------------------------------------

    /**
     * rules.py:778-817 `on_step`. 스텝 하나를 처리하고 drowsy 이벤트(없으면 빈 목록)를 돌려준다.
     * probs가 null인 스텝은 모델 쪽에서 값이 없는 스텝이다. PERCLOS 쪽은 full_lookback인 스텝만 값이 있다(원본의 가정,
     * rules.py:471-472).
     */
    fun onStep(step: StepFeatures, probs: ClassProbs?): List<DriverStateEvent> {
        val dCfg = cfg.drowsiness
        val useModel = dCfg.source == DrowsySource.MODEL || dCfg.source == DrowsySource.EITHER
        val usePerclos = dCfg.source == DrowsySource.PERCLOS || dCfg.source == DrowsySource.EITHER

        val hasP = probs != null
        var pDrowsy = 0.0
        if (probs != null) {
            pDrowsy = probAt(probs, CLASS_DROWSY)
        }
        val perclos = step.values[F_PERCLOS_60]

        if (useModel) {
            modelRule.update(
                k = step.k,
                tEndMs = step.tEndMs,
                hasValue = hasP,
                onsetHit = pDrowsy >= dCfg.model.onsetP,
                endHit = pDrowsy <= dCfg.model.endP,
            )
        }
        if (usePerclos) {
            perclosRule.update(
                k = step.k,
                tEndMs = step.tEndMs,
                hasValue = step.fullLookback,
                onsetHit = perclos >= dCfg.perclos.onsetRatio,
                endHit = perclos < dCfg.perclos.endRatio,
            )
        }
        val modelOn = useModel && modelRule.active
        val perclosOn = usePerclos && perclosRule.active

        if (!drowsyActive && (modelOn || perclosOn)) {
            drowsyActive = true
            if (modelOn) {
                drowsySource = Source.MODEL
                drowsyStartMs = modelRule.startMs
            } else {
                drowsySource = Source.PERCLOS
                drowsyStartMs = perclosRule.startMs
            }
            return listOf(drowsyEvent(step, pDrowsy, Phase.ONSET))
        }
        if (drowsyActive && !(modelOn || perclosOn)) {
            drowsyActive = false
            return listOf(drowsyEvent(step, pDrowsy, Phase.END))
        }
        return emptyList()
    }

    // ------------------------------------------------------------------
    // 이벤트 만들기
    // ------------------------------------------------------------------

    /** rules.py:823-842 `_event`. duration_ms = 발행 시각 − t_start_ms. severity는 설정의 종류별 값이다. */
    private fun event(
        type: EventType,
        tStartMs: Long,
        tEmitMs: Long,
        confidence: Double,
        source: Source,
        phase: Phase,
        provisional: Boolean,
    ): DriverStateEvent = DriverStateEvent(
        type = type,
        severity = cfg.events.get(type).severity,
        tStartMs = tStartMs,
        durationMs = tEmitMs - tStartMs,
        confidence = confidence,
        source = source,
        phase = phase,
        provisional = provisional,
    )

    /** rules.py:844-849 `_gaze_event`. */
    private fun gazeEvent(state: FrameState, phase: Phase): DriverStateEvent {
        val t = state.tMs
        val confidence = ratio(numeratorMs = poseMs - gazePoseMs0, denominatorMs = t - gazeStartMs)
        return event(
            type = EventType.GAZE_AWAY_LONG,
            tStartMs = gazeStartMs,
            tEmitMs = t,
            confidence = confidence,
            source = gazeSource,
            phase = phase,
            provisional = state.provisional,
        )
    }

    /** rules.py:851-855 `_vats_event`. [t_start_ms, 발행 시각] 전체의 pose_valid 시간 비율(30초로 잘리지 않는다). */
    private fun vatsEvent(state: FrameState, phase: Phase): DriverStateEvent {
        val t = state.tMs
        val confidence = ratio(numeratorMs = poseMs - vatsPoseMs0, denominatorMs = t - vatsStartMs)
        return event(
            type = EventType.VATS,
            tStartMs = vatsStartMs,
            tEmitMs = t,
            confidence = confidence,
            source = vatsSource,
            phase = phase,
            provisional = state.provisional,
        )
    }

    /** rules.py:857-863 `_drowsy_event`. 발행 시각은 스텝 끝 시각, provisional은 언제나 false다. */
    private fun drowsyEvent(step: StepFeatures, pDrowsy: Double, phase: Phase): DriverStateEvent {
        var confidence = pDrowsy
        if (drowsySource == Source.PERCLOS) {
            confidence = 1.0 - step.values[F_FACE_MISSING_RATIO_60]
        }
        return event(
            type = EventType.DROWSY,
            tStartMs = drowsyStartMs,
            tEmitMs = step.tEndMs,
            confidence = confidence,
            source = drowsySource,
            phase = phase,
            provisional = false,
        )
    }

    /**
     * rules.py:865-873 `_closure_event`. t_start_ms는 감김의 start_ms, duration_ms는 공통 규칙대로 t − t_start_ms다.
     * 임계 시간을 판정한 길이(진입 임계 이상이었던 시간의 합)가 아니므로, 감김 도중에 반쯤 뜬 시간이 있었으면 그만큼 더 길다.
     */
    private fun closureEvent(state: FrameState, level: Int, phase: Phase): DriverStateEvent {
        val t = state.tMs
        var type = EventType.MICROSLEEP
        if (level == LEVEL_SLEEP) {
            type = EventType.SLEEP
        }
        val confidence = ratio(numeratorMs = closureValidMs - runValidMs0, denominatorMs = t - runFirstMs)
        return event(
            type = type,
            tStartMs = runStartMs,
            tEmitMs = t,
            confidence = confidence,
            source = Source.CLOSURE,
            phase = phase,
            provisional = state.provisional,
        )
    }

    /** `probs[CLASS_DROWSY]` (rules.py:787): ClassProbs의 성분 순서는 CLASS_ORDER와 같다. */
    private fun probAt(p: ClassProbs, index: Int): Double = when (index) {
        0 -> p.alert
        1 -> p.lowVigilant
        2 -> p.drowsy
        else -> throw IllegalArgumentException("class index $index is not 0, 1 or 2")
    }
}

/** rules.py:1015-1019 `_ratio`. 시간 비율. 분모가 0 이하이면 0.0이다. 정수 둘을 실수로 바꿔 나눈다(R3). */
internal fun ratio(numeratorMs: Long, denominatorMs: Long): Double {
    if (denominatorMs <= 0L) {
        return 0.0
    }
    return numeratorMs.toDouble() / denominatorMs.toDouble()
}
