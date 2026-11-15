package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.ClosureRun
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.StepFeatures
import hawkeye.drowsy.core.spec.PipelineSpec
import java.util.Collections
import kotlin.math.sqrt

// 원본: drowsy/step_features.py:215-555 (_Bucket, StepAggregator, _std_from_moments). SPEC 11장. 계획 5.5절의 선언 그대로다.
// 하품·끄덕임 계수기는 ThresholdEventCounter.kt에 있다.
//
// 시간 규칙(step_features.py:7-14, SPEC 6장, 11장). t는 세션 시간 = state.tMs − originMs다.
// - 한 프레임의 상태는 다음 프레임이 올 때까지, 최대 clock.max_hold_ms까지 유지된 것으로 본다. 넘는 부분은 "데이터
//   없음"이다. 그래서 시간 누적은 다음 프레임이 도착했을 때 하고, 유지 구간이 스텝 경계에 걸치면 경계에서 나눈다.
// - 스텝 k는 [1000(k−1), 1000k)를 대표한다. t >= 1000k인 첫 프레임이 스텝 k를 발행시키고 그 프레임은 스텝 k에 들어가지
//   않는다.
// - 프레임 간격이 clock.reset_gap_ms를 넘으면 누적값과 하품·끄덕임 검출기를 초기화한다.
// 원본이 정한 가정(step_features.py:26-53)은 원본 모듈 설명에 있고, 이 파일은 그 동작을 문장 순서 그대로 옮긴다.
//
// 형(계획 5.5절, SPEC 11.1): ms 합은 Long, 개수는 Int, 실수 합은 Double. durSqSum += d·d는 Long이다(R1).
// 합은 적힌 순서로, 구간 합은 오래된 통부터 매번 다시 더한다(R14). 통은 스텝마다 새 객체다(R12).
// 가정(A1): 사칙연산, sqrt, 비교는 JVM·Android와 CPython에서 같은 double을 낸다. 정수 ms의 몫인 특징(#0, #1, #4,
// #15, #16)과 개수·ms 특징(#9~#12, #14)은 이 가정 아래에서 Python과 비트가 같다(엄격 층, 계획 9.0절). Long.toDouble()은
// Python float(int)처럼 가장 가까운 double로 바꾼다(정수 ms는 2^53보다 훨씬 작다).

/** step_features.py:63 `_MS_PER_MINUTE`. */
private const val MS_PER_MINUTE: Double = 60000.0

/** step_features.py:215-253 `_Bucket`. 스텝 하나(1초)의 누적값. 시간은 정수 ms로 더한다. */
private class Bucket {
    var validMs: Long = 0L // closure가 유효한 시간
    var closedMs: Long = 0L // 그중 closure >= perclos_thr인 시간
    var closureSum: Double = 0.0 // Σ closure·ms (유효 시간에 대해)
    var faceMs: Long = 0L // 얼굴이 있는 시간
    var pitchSum: Double = 0.0 // Σ pitch_rel·ms (얼굴 시간에 대해)
    var pitchSqSum: Double = 0.0 // Σ pitch_rel²·ms
    var nEvents: Int = 0 // 잘리지 않은 감김 이벤트 수
    var durSum: Long = 0L // Σ min(duration_ms, 상한)
    var durSqSum: Long = 0L // Σ min(duration_ms, 상한)²
    var nVelocity: Int = 0 // 그중 속도가 유효한 이벤트 수
    var closingVelSum: Double = 0.0
    var openingVelSum: Double = 0.0
    var nLong: Int = 0 // 길이 >= long_closure_min_ms인 감김 수(잘린 것 포함)
    var maxRunMs: Long = 0L // 귀속된 감김 중 가장 긴 것의 길이(잘린 것 포함)
    var nYawn: Int = 0
    var nNod: Int = 0
}

/**
 * step_features.py:256-547 `StepAggregator`. 프레임 상태를 1초 스텝 특징으로 누적한다.
 *
 * - startMs: 누적을 시작하는 세션 시간 T_b. 첫 스텝은 [startMs, startMs + step_ms)다. 음수이거나 clock.step_ms의 배수가
 *   아니면 IllegalArgumentException(Python은 ValueError).
 * - originMs: 처음 받아들인 프레임의 tMs. 세션 시간 = state.tMs − originMs.
 *
 * [update]는 받아들인 프레임마다, 특징용 감김 검출기를 갱신한 뒤에 부른다. run은 그 프레임에서 특징용 검출기가 발행한 구간
 * (없으면 null), ongoingMs는 그 프레임을 처리한 뒤의 `ClosureRunDetector.ongoingMs()`다. 이 프레임이 발행시킨 스텝을
 * 오래된 것부터 돌려준다(읽기 전용 목록). startMs 전의 프레임은 받아도 누적하지 않고 빈 목록을 돌려준다. 세션 시간이
 * 직전 프레임 이하이면 IllegalArgumentException이고 상태를 바꾸지 않는다.
 *
 * 스레드에 안전하지 않다. 상태는 인스턴스에만 있다(R19).
 */
internal class StepAggregator(spec: PipelineSpec, startMs: Long, originMs: Long) {

    private val stepMs: Long = spec.clock.stepMs
    private val maxHoldMs: Long = spec.clock.maxHoldMs
    private val resetGapMs: Long = spec.clock.resetGapMs

    private val lookbackLongMs: Long
    private val lookbackShortMs: Long
    private val nLong: Long
    private val nShort: Long
    private val perclosThr: Double
    private val perclosMinValidMs60: Long
    private val perclosMinValidMs10: Long
    private val blinkRateMinValidMs: Long
    private val blinkDurationCapMs: Long
    private val longClosureMinMs: Long
    private val maxClosureCapMs: Long
    private val pitchStdMinFaceMs: Long

    private val originMs: Long
    private val startMs: Long
    private var accumStartMs: Long // 지금 누적을 시작한 세션 시간(초기화하면 뒤로 옮겨진다)

    // 지금 채우고 있는 스텝 번호. Python int 그대로 Long으로 센다. StepFeatures.k(Int)로 넣을 때만 범위를 검사한다(emit).
    private var curK: Long
    private var cur = Bucket()
    // 끝난 스텝의 통, 오래된 것부터 최대 nLong개 (R13: ArrayDeque의 addLast, removeFirst)
    private var history = ArrayDeque<Bucket>()

    // 직전 프레임(세션 시간). 프레임을 받기 전에는 startMs에 "데이터 없음" 프레임이 있다고 본다.
    private var hasPrev = false
    private var prevT: Long
    private var prevFace = false
    private var prevValid = false
    private var prevClosure: Double = 0.0
    private var prevPitch: Double = 0.0
    private var prevOngoingMs: Long = 0L

    private val yawn: ThresholdEventCounter
    private val nod: NodDetector

    init {
        // step_features.py:270-271. Python의 `or`는 앞이 참이면 뒤를 보지 않는다. 음수가 아니면 %와 floorMod가 같다.
        if (startMs < 0 || startMs % stepMs != 0L) {
            throw IllegalArgumentException("start_ms must be a non-negative multiple of clock.step_ms")
        }
        val sf = spec.stepFeatures
        lookbackLongMs = sf.lookbackLongMs
        lookbackShortMs = sf.lookbackShortMs
        // Python `//`(내림 나눗셈)
        nLong = Math.floorDiv(sf.lookbackLongMs, stepMs)
        nShort = Math.floorDiv(sf.lookbackShortMs, stepMs)
        perclosThr = sf.perclosThr
        perclosMinValidMs60 = sf.perclosMinValidMs60
        perclosMinValidMs10 = sf.perclosMinValidMs10
        blinkRateMinValidMs = sf.blinkRateMinValidMs
        blinkDurationCapMs = sf.blinkDurationCapMs
        longClosureMinMs = sf.longClosureMinMs
        maxClosureCapMs = sf.maxClosureCapMs
        pitchStdMinFaceMs = sf.pitchStdMinFaceMs

        this.originMs = originMs
        this.startMs = startMs
        accumStartMs = startMs

        curK = Math.floorDiv(startMs, stepMs) + 1
        prevT = startMs

        yawn = yawnCounter(spec.yawn)
        nod = NodDetector(spec.nod)
    }

    /** step_features.py:309-338. 프레임 하나를 누적한다. 이 프레임이 발행시킨 스텝을 오래된 것부터 돌려준다. */
    fun update(state: FrameState, run: ClosureRun?, ongoingMs: Long): List<StepFeatures> {
        val t = state.tMs - originMs
        if (t < startMs) {
            return Collections.emptyList()
        }
        if (hasPrev && t <= prevT) {
            throw IllegalArgumentException("t_ms must be strictly greater than the previous frame's t_ms")
        }

        val steps = ArrayList<StepFeatures>()
        if (t - prevT > resetGapMs) {
            restart(t)
        } else {
            advanceTo(t, steps)
        }

        // 이 프레임에서 생긴 이벤트는 이 프레임이 속한 스텝(지금 채우는 통)에 넣는다.
        if (run != null) {
            addRun(run)
        }
        // R11: 하품과 끄덕임의 update는 프레임마다 둘 다 부른다(한 식으로 합치지 않는다).
        if (yawn.update(t, state.jawRel, state.face)) {
            cur.nYawn += 1
        }
        if (nod.update(t, state.pitchRel, state.face)) {
            cur.nNod += 1
        }

        hasPrev = true
        prevT = t
        prevFace = state.face
        prevValid = state.closureValid
        prevClosure = state.closure
        prevPitch = state.pitchRel
        prevOngoingMs = ongoingMs
        return Collections.unmodifiableList(steps)
    }

    /** step_features.py:340-349 `_restart`. 10초 넘는 끊김 뒤의 초기화. t가 속한 스텝부터 다시 누적한다. */
    private fun restart(t: Long) {
        curK = Math.floorDiv(t, stepMs) + 1
        accumStartMs = (curK - 1) * stepMs
        cur = Bucket()
        history = ArrayDeque()
        hasPrev = false
        prevOngoingMs = 0L
        yawn.reset()
        nod.reset()
    }

    /** step_features.py:351-371 `_advance_to`. 직전 프레임의 상태가 유지된 구간을 누적하면서, t까지 끝난 스텝을 발행한다. */
    private fun advanceTo(t: Long, steps: MutableList<StepFeatures>) {
        var pos = prevT
        var holdEnd = prevT
        if (hasPrev) {
            var heldMs = t - prevT
            if (heldMs > maxHoldMs) {
                heldMs = maxHoldMs
            }
            holdEnd = prevT + heldMs
        }

        // R1: curK·stepMs는 Long 곱이다.
        while (curK * stepMs <= t) {
            val boundary = curK * stepMs
            if (pos < holdEnd) {
                var pieceEnd = holdEnd
                if (pieceEnd > boundary) {
                    pieceEnd = boundary
                }
                addTime(pieceEnd - pos)
                pos = pieceEnd
            }
            steps.add(emit())
        }
        if (pos < holdEnd) {
            addTime(holdEnd - pos)
        }
    }

    /** step_features.py:373-384 `_add_time`. 직전 프레임의 상태를 ms만큼 지금 통에 더한다. */
    private fun addTime(ms: Long) {
        val bucket = cur
        // Python의 float·int는 int를 실수로 바꿔 곱한다. pitch²·ms는 왼쪽부터 (p·p)·ms다.
        val msD = ms.toDouble()
        if (prevFace) {
            bucket.faceMs += ms
            bucket.pitchSum += prevPitch * msD
            bucket.pitchSqSum += prevPitch * prevPitch * msD
        }
        if (prevValid) {
            bucket.validMs += ms
            bucket.closureSum += prevClosure * msD
            if (prevClosure >= perclosThr) {
                bucket.closedMs += ms
            }
        }
    }

    /** step_features.py:386-402 `_add_run`. */
    private fun addRun(run: ClosureRun) {
        val bucket = cur
        if (!run.truncated) {
            var durationMs = run.durationMs
            if (durationMs > blinkDurationCapMs) {
                durationMs = blinkDurationCapMs
            }
            bucket.nEvents += 1
            bucket.durSum += durationMs
            bucket.durSqSum += durationMs * durationMs
            if (run.velocityValid) {
                bucket.nVelocity += 1
                bucket.closingVelSum += run.peakClosingVelocity
                bucket.openingVelSum += run.peakOpeningVelocity
            }
        }
        if (run.durationMs >= longClosureMinMs) {
            bucket.nLong += 1
        }
        if (run.durationMs > bucket.maxRunMs) {
            bucket.maxRunMs = run.durationMs
        }
    }

    /** step_features.py:404-419 `_emit`. 지금 통을 닫고 그 스텝의 특징을 만든다. */
    private fun emit(): StepFeatures {
        history.addLast(cur)
        if (history.size.toLong() > nLong) {
            history.removeFirst()
        }
        val k = curK
        val endMs = k * stepMs
        val step = StepFeatures(
            // 가정: 스텝 번호는 Int 범위 안이다(세션 시간 약 2^31초, 68년). 넘으면 조용히 줄이지 않고 ArithmeticException.
            k = Math.toIntExact(k),
            tEndMs = originMs + endMs,
            fullLookback = endMs >= accumStartMs + lookbackLongMs,
            values = features(),
        )
        cur = Bucket()
        curK = k + 1
        return step
    }

    /** step_features.py:421-547 `_features`. 방금 끝난 스텝의 특징 17개. history의 마지막 통이 그 스텝이다. */
    private fun features(): List<Double> {
        val n = history.size
        var firstShort = n.toLong() - nShort
        if (firstShort < 0) {
            firstShort = 0
        }

        var valid60 = 0L
        var closed60 = 0L
        var closureSum60 = 0.0
        var face60 = 0L
        var pitchSum = 0.0
        var pitchSqSum = 0.0
        var nEvents = 0
        var durSum = 0L
        var durSqSum = 0L
        var nVelocity = 0
        var closingVelSum = 0.0
        var openingVelSum = 0.0
        var nLongCount = 0
        var maxRun60 = 0L
        var nYawn = 0
        var nNod = 0
        var valid10 = 0L
        var closed10 = 0L
        var closureSum10 = 0.0
        var face10 = 0L
        var maxRun10 = 0L

        // R14: 오래된 통부터 매번 다시 더한다.
        for (i in 0 until n) {
            val b = history[i]
            valid60 += b.validMs
            closed60 += b.closedMs
            closureSum60 += b.closureSum
            face60 += b.faceMs
            pitchSum += b.pitchSum
            pitchSqSum += b.pitchSqSum
            nEvents += b.nEvents
            durSum += b.durSum
            durSqSum += b.durSqSum
            nVelocity += b.nVelocity
            closingVelSum += b.closingVelSum
            openingVelSum += b.openingVelSum
            nLongCount += b.nLong
            if (b.maxRunMs > maxRun60) {
                maxRun60 = b.maxRunMs
            }
            nYawn += b.nYawn
            nNod += b.nNod
            if (i.toLong() >= firstShort) {
                valid10 += b.validMs
                closed10 += b.closedMs
                closureSum10 += b.closureSum
                face10 += b.faceMs
                if (b.maxRunMs > maxRun10) {
                    maxRun10 = b.maxRunMs
                }
            }
        }

        // 진행 중인 감김: 이 스텝에 속한 마지막 프레임에서 받은 값이다.
        val ongoingMs = prevOngoingMs
        if (ongoingMs >= longClosureMinMs) {
            nLongCount += 1
        }
        if (ongoingMs > maxRun60) {
            maxRun60 = ongoingMs
        }
        if (ongoingMs > maxRun10) {
            maxRun10 = ongoingMs
        }
        if (maxRun60 > maxClosureCapMs) {
            maxRun60 = maxClosureCapMs
        }
        if (maxRun10 > maxClosureCapMs) {
            maxRun10 = maxClosureCapMs
        }

        // R3: 정수 둘을 나누는 곳은 피연산자마다 실수로 바꾼다.
        var perclos60 = 0.0
        if (valid60 >= perclosMinValidMs60) {
            perclos60 = closed60.toDouble() / valid60.toDouble()
        }
        var perclos10 = 0.0
        if (valid10 >= perclosMinValidMs10) {
            perclos10 = closed10.toDouble() / valid10.toDouble()
        }

        var closureMean60 = 0.0
        if (valid60 > 0) {
            closureMean60 = closureSum60 / valid60.toDouble()
        }
        var closureMean10 = 0.0
        if (valid10 > 0) {
            closureMean10 = closureSum10 / valid10.toDouble()
        }

        var blinkRate60 = 0.0
        if (valid60 >= blinkRateMinValidMs) {
            blinkRate60 = nEvents.toDouble() * MS_PER_MINUTE / valid60.toDouble()
        }

        var blinkDurMean60 = 0.0
        var blinkDurStd60 = 0.0
        if (nEvents > 0) {
            blinkDurMean60 = durSum.toDouble() / nEvents.toDouble()
        }
        if (nEvents >= 2) {
            blinkDurStd60 = stdFromMoments(blinkDurMean60, durSqSum.toDouble() / nEvents.toDouble())
        }

        var closingVelMean60 = 0.0
        var openingVelMean60 = 0.0
        if (nVelocity > 0) {
            closingVelMean60 = closingVelSum / nVelocity.toDouble()
            openingVelMean60 = openingVelSum / nVelocity.toDouble()
        }

        var pitchStd60 = 0.0
        if (face60 >= pitchStdMinFaceMs) {
            pitchStd60 = stdFromMoments(pitchSum / face60.toDouble(), pitchSqSum / face60.toDouble())
        }

        // 통이 없는 시간(누적 시작 전)도 얼굴이 없던 시간으로 센다.
        val faceMissingRatio60 = (lookbackLongMs - face60).toDouble() / lookbackLongMs.toDouble()
        val faceMissingRatio10 = (lookbackShortMs - face10).toDouble() / lookbackShortMs.toDouble()

        // FEATURE_NAMES 순서(계획 6.1절). 읽기 전용 목록이다(계획 5.7절: 모델 창에 그대로 넣는다).
        val values = ArrayList<Double>(17)
        values.add(perclos60)
        values.add(perclos10)
        values.add(closureMean60)
        values.add(closureMean10)
        values.add(blinkRate60)
        values.add(blinkDurMean60)
        values.add(blinkDurStd60)
        values.add(closingVelMean60)
        values.add(openingVelMean60)
        values.add(nLongCount.toDouble())
        values.add(maxRun60.toDouble())
        values.add(maxRun10.toDouble())
        values.add(nYawn.toDouble())
        values.add(pitchStd60)
        values.add(nNod.toDouble())
        values.add(faceMissingRatio60)
        values.add(faceMissingRatio10)
        return Collections.unmodifiableList(values)
    }
}

/**
 * step_features.py:550-555 `_std_from_moments`. 평균과 제곱 평균으로 구한 모표준편차. 분산은 제곱근 전에 0 이상으로
 * 자른다(Python의 `if` 그대로, R6).
 */
internal fun stdFromMoments(mean: Double, meanSq: Double): Double {
    var variance = meanSq - mean * mean
    if (variance < 0.0) {
        variance = 0.0
    }
    return sqrt(variance)
}
