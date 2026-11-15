package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.ClosureRun
import hawkeye.drowsy.core.spec.RunDetectorSpec
import kotlin.math.floor

// 원본: drowsy/blink.py 전체 (SPEC 10장). 계획 5.4절의 선언 그대로다.
// closure에 히스테리시스를 거는 상태 머신이다. 같은 클래스를 설정만 바꿔 두 번 쓴다: 스텝 특징용(pipeline.run_detector,
// 진입 0.5 / 이탈 0.3)과 규칙용(rules.closure.detector, 진입 0.7 / 이탈 0.5). 임계값은 모두 설정에서 읽는다.
// 전이표, 시각·길이·속도의 정의, 원본이 정한 가정(blink.py:79-100)은 원본 모듈 설명과 SPEC 10장에 있다. 이 파일은 그것을
// 문장 순서 그대로 옮긴다. 시각과 길이는 모두 Long이다(R1).
//
// 가정(A1): 사칙연산과 비교는 JVM·Android와 CPython에서 같은 double을 낸다. 증가율 (c − prevC)·1000 / (t − prevT),
// 보간 비율 fraction, 그 오프셋 fraction·(t − prevT)는 이 가정 아래에서 Python과 비트가 같다. 그래서 같은 입력에서 보간
// 시각과 구간의 판정이 같다(엄격 층: runs의 시각은 정확히 같아야 한다, 계획 9.0절).

/** blink.py:110-112 `STATE_OPEN`, `STATE_CLOSED`, `STATE_HOLD`. */
internal enum class RunState { OPEN, CLOSED, HOLD }

/**
 * blink.py:115-334 `ClosureRunDetector`. 감김 구간 검출기.
 *
 * 호출 계약(blink.py:69-77, SPEC 10.1):
 * - 프레임 하나에 [update]를 한 번 부른다. 얼굴이 없거나 closure가 유효하지 않은 프레임도 valid = false로 넘긴다
 *   (그때 closure 값은 읽지 않는다).
 * - tMs는 호출마다 엄격히 증가해야 한다. 직전 값 이하이면 IllegalArgumentException(Python은 ValueError)이고 상태를
 *   바꾸지 않는다(순서 검사는 상태를 바꾸기 전에, blink.py:194).
 * - 프레임 간격이 clock.reset_gap_ms를 넘으면 호출자가 [reset]을 부른 뒤 그 프레임을 넘긴다.
 * - [update] 한 번은 구간을 최대 하나 발행한다.
 *
 * 스레드에 안전하지 않다. 상태는 인스턴스에만 있다(R19).
 */
internal class ClosureRunDetector(settings: RunDetectorSpec) {

    private val enterThr: Double = settings.enterThr
    private val exitThr: Double = settings.exitThr
    private val minDurationMs: Long = settings.minDurationMs
    private val maxDtMs: Long = settings.maxDtMs
    private val holdMs: Long = settings.holdMs
    private val lookback: Int = settings.velocityLookbackSamples

    private var stateValue: RunState = RunState.OPEN

    // 마지막으로 받은 프레임(유효 여부와 무관). 시간 순서 검사에만 쓴다.
    private var hasFrame = false
    private var lastTMs: Long = 0L

    // 직전 유효 표본. 무효 프레임이 와도 지우지 않는다.
    private var hasPrev = false
    private var prevTMs: Long = 0L
    private var prevC: Double = 0.0

    // 최근 쌍들의 closure 증가율(0 이상), 오래된 것부터. 길이는 velocity_lookback_samples로 고정이다(blink.py:137-140).
    private val recentUp = DoubleArray(lookback)

    // 진행 중인 구간(CLOSED, HOLD에서만 뜻이 있다)
    private var startMs: Long = 0L
    private var sealedMs: Long = 0L // 끝난 토막들의 길이 합
    private var spanStartMs: Long = 0L // 지금 토막이 시작된 시각(첫 토막이면 startMs)
    private var lastClosedMs: Long = 0L // closure가 enter 이상이었던 마지막 유효 프레임의 시각
    private var lastDownMs: Long = 0L // 마지막으로 enter를 아래로 지난 시각
    private var amplitude: Double = 0.0
    private var closingPeak: Double = 0.0 // 정점까지의 최대 증가율
    private var openingPeak: Double = 0.0 // 정점 뒤의 최대 감소율
    private var pendingUp: Double = 0.0 // 정점 뒤의 최대 증가율. 정점이 뒤로 옮겨지면 감는 속도에 합친다

    /** blink.py:153-156. 지금 상태. */
    val state: RunState
        get() = stateValue

    /**
     * blink.py:158-170. 처음 상태로 되돌린다. 진행 중이던 구간은 발행하지 않고 버린다.
     * 원본 그대로 시각·유효 표본·증가율만 지운다. 구간 필드(startMs, sealedMs, ...)는 OPEN에서 읽히지 않고 다음 구간의
     * 시작(begin)이 모두 다시 쓴다.
     */
    fun reset() {
        stateValue = RunState.OPEN
        hasFrame = false
        lastTMs = 0L
        hasPrev = false
        prevTMs = 0L
        prevC = 0.0
        for (i in 0 until lookback) {
            recentUp[i] = 0.0
        }
    }

    /** blink.py:172-181. 진행 중인 구간의 길이(ms). OPEN이면 0이다. */
    fun ongoingMs(): Long {
        if (stateValue == RunState.OPEN) {
            return 0L
        }
        return sealedMs + (lastClosedMs - spanStartMs)
    }

    /** blink.py:183-190. 진행 중인 구간의 startMs. OPEN이면 0이다. */
    fun runStartMs(): Long {
        if (stateValue == RunState.OPEN) {
            return 0L
        }
        return startMs
    }

    /** blink.py:192-267. 프레임 하나를 처리한다. 이 프레임에서 구간이 끝났으면 ClosureRun을, 아니면 null을 돌려준다. */
    fun update(tMs: Long, closure: Double, valid: Boolean): ClosureRun? {
        if (hasFrame && tMs <= lastTMs) {
            throw IllegalArgumentException("t_ms must be strictly greater than the previous frame's t_ms")
        }
        hasFrame = true
        lastTMs = tMs

        var run: ClosureRun? = null

        // 끊긴 시간이 hold_ms를 넘었으면 먼저 구간을 닫고, 이 프레임은 OPEN에서 처리한다.
        if (stateValue != RunState.OPEN && tMs - prevTMs > holdMs) {
            run = finish(true)
        }

        if (!valid) {
            if (stateValue == RunState.CLOSED) {
                stateValue = RunState.HOLD
            }
            return run
        }

        // 직전 유효 표본과의 쌍
        val pairOk = hasPrev && tMs - prevTMs <= maxDtMs
        var upRate = 0.0
        var downRate = 0.0
        if (pairOk) {
            // Python `(closure - prev_c) * 1000.0 / (t_ms - prev_t_ms)`: 정수 간격은 실수로 바뀌어 나눈다(R3).
            val rate = (closure - prevC) * 1000.0 / (tMs - prevTMs).toDouble()
            if (rate > 0.0) {
                upRate = rate
            } else if (rate < 0.0) {
                downRate = -rate
            }
        } else {
            // 쓸 수 없는 쌍(직전 유효 표본이 없거나 간격이 max_dt_ms 초과): 거슬러 보기를 끊는다.
            for (i in 0 until lookback) {
                recentUp[i] = 0.0
            }
        }
        for (i in 0 until lookback - 1) {
            recentUp[i] = recentUp[i + 1]
        }
        recentUp[lookback - 1] = upRate

        if (stateValue == RunState.OPEN) {
            if (closure >= enterThr) {
                begin(tMs, closure, pairOk)
            }
        } else {
            val resumedFromHold = stateValue == RunState.HOLD
            stateValue = RunState.CLOSED

            if (closure > amplitude) {
                // 정점이 이 표본으로 옮겨졌다. 여기까지의 쌍은 모두 감는 쪽이고 뜨는 쪽은 다시 잰다.
                amplitude = closure
                if (pendingUp > closingPeak) {
                    closingPeak = pendingUp
                }
                if (upRate > closingPeak) {
                    closingPeak = upRate
                }
                pendingUp = 0.0
                openingPeak = 0.0
            } else {
                if (upRate > pendingUp) {
                    pendingUp = upRate
                }
                if (downRate > openingPeak) {
                    openingPeak = downRate
                }
            }

            if (closure >= enterThr) {
                if (prevC < enterThr) {
                    // 다시 감김: 앞 토막을 마지막으로 내려간 시각에서 닫아 더하고, 이번에 올라온 시각에서 새 토막을 연다.
                    sealedMs += lastDownMs - spanStartMs
                    spanStartMs = upCrossingMs(tMs, closure, pairOk)
                }
                lastClosedMs = tMs
            } else {
                if (prevC >= enterThr) {
                    lastDownMs = downCrossingMs(tMs, closure, pairOk)
                }
                if (closure < exitThr) {
                    run = finish(resumedFromHold)
                }
            }
        }

        hasPrev = true
        prevTMs = tMs
        prevC = closure
        return run
    }

    /** blink.py:269-287 `_begin`. OPEN → CLOSED. 이 표본이 enter 이상을 처음 본 표본이다. */
    private fun begin(tMs: Long, closure: Double, pairOk: Boolean) {
        val start = upCrossingMs(tMs, closure, pairOk)

        var peak = 0.0
        for (i in 0 until lookback) {
            if (recentUp[i] > peak) {
                peak = recentUp[i]
            }
        }

        stateValue = RunState.CLOSED
        startMs = start
        sealedMs = 0L
        spanStartMs = start
        lastClosedMs = tMs
        lastDownMs = tMs
        amplitude = closure
        closingPeak = peak
        openingPeak = 0.0
        pendingUp = 0.0
    }

    /**
     * blink.py:289-298 `_up_crossing_ms`. 직전 유효 표본과 이 표본(enter 이상) 사이에서 enter를 위로 지난 시각.
     * 쓸 수 있는 쌍이고 직전 표본이 enter 미만일 때만 보간한다. 아니면 이 프레임의 시각이다.
     * 쌍 안의 오프셋만 반올림하고 직전 표본의 시각(정수)은 그대로 더한다(SPEC 10.2).
     */
    private fun upCrossingMs(tMs: Long, closure: Double, pairOk: Boolean): Long {
        if (!pairOk || prevC >= enterThr) {
            return tMs
        }
        val fraction = (enterThr - prevC) / (closure - prevC)
        return prevTMs + roundHalfUp(fraction * (tMs - prevTMs).toDouble())
    }

    /** blink.py:300-305 `_down_crossing_ms`. 직전 유효 표본(enter 이상)과 이 표본(enter 미만) 사이에서 enter를 아래로 지난 시각. */
    private fun downCrossingMs(tMs: Long, closure: Double, pairOk: Boolean): Long {
        if (!pairOk) {
            return prevTMs
        }
        val fraction = (prevC - enterThr) / (prevC - closure)
        return prevTMs + roundHalfUp(fraction * (tMs - prevTMs).toDouble())
    }

    /** blink.py:307-334 `_finish`. 진행 중인 구간을 닫고 OPEN으로 간다. min_duration_ms보다 짧으면 null을 돌려준다. */
    private fun finish(truncated: Boolean): ClosureRun? {
        stateValue = RunState.OPEN
        val endMs: Long = if (truncated) lastClosedMs else lastDownMs
        // 끝난 토막들에 마지막 토막을 더한다. 토막이 하나뿐이면 endMs − startMs다.
        val durationMs = sealedMs + (endMs - spanStartMs)
        if (durationMs < minDurationMs) {
            return null
        }

        val velocityValid = (!truncated) && closingPeak > 0.0 && openingPeak > 0.0
        var closingVelocity = 0.0
        var openingVelocity = 0.0
        if (velocityValid) {
            closingVelocity = closingPeak
            openingVelocity = openingPeak
        }
        return ClosureRun(
            startMs = startMs,
            endMs = endMs,
            durationMs = durationMs,
            amplitude = amplitude,
            peakClosingVelocity = closingVelocity,
            peakOpeningVelocity = openingVelocity,
            velocityValid = velocityValid,
            truncated = truncated,
        )
    }
}

/**
 * blink.py:337-339 `_round_half_up`: floor(x + 0.5). 계획 R4, 5.4절의 식 그대로다(Python의 round와 Kotlin의 roundToLong,
 * Math.round는 쓰지 않는다).
 *
 * 가정: 참조 구현은 유한하지 않은 인자에서 예외를 낸다(NaN은 ValueError, 무한대는 OverflowError, blink.py:339의
 * `int(math.floor(...))`). Kotlin의 `floor(x + 0.5).toLong()`은 예외 없이 NaN을 0으로, 무한대를 Long의 끝값으로 바꾼다.
 * 소유자가 2026-11-15에 이 식을 그대로 두기로 정했다("지금대로 둠", 리드 판정 4). 그래서 검사를 더하지 않는다. 유한하지 않은
 * 값이 여기 오지 않게 하는 것은 호출자의 일이다(SPEC 2장: NaN과 무한대를 쓰지 않는다). README의 알려진 차이에 적는다.
 * 벡터에는 그런 입력이 없다. 차등 자료 S-odd의 18세션에서는 Python이 이 자리에서 예외를 낸다(단계 5가 알려진 차이로 센다).
 */
internal fun roundHalfUp(x: Double): Long = floor(x + 0.5).toLong()
