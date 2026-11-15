package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.spec.NodSpec
import hawkeye.drowsy.core.spec.YawnSpec

// 원본: drowsy/step_features.py:66-212 (ThresholdEventCounter, yawn_counter, NodDetector). SPEC 11.5절.
// 계획 5.5절의 선언 그대로다. 시각과 길이는 Long이다(R1). 상태는 인스턴스에만 있다(R19).
//
// 가정(A1): 사칙연산과 비교는 JVM·Android와 CPython에서 같은 double을 낸다. 끄덕임의 느린 평균
// m += α·(pitch_rel − m), α = dt / (ema_tau_ms + dt)와 d = pitch_rel − m은 이 가정 아래에서 Python과 비트가 같다.

/**
 * step_features.py:66-139. 히스테리시스 임계 이벤트 계수기 (하품과 끄덕임 공용).
 *
 * 값이 enterThr 이상이면 이벤트가 시작되고 exitThr 미만이면 끝난다. 길이는 프레임 시각의 차다.
 * - countAtEnd = false(하품): 진행 중인 이벤트의 길이가 minDurationMs에 도달한 프레임에서 한 번 센다. maxDurationMs는
 *   쓰지 않는다.
 * - countAtEnd = true(끄덕임): 끝나는 프레임에서 길이가 minDurationMs 이상 maxDurationMs 이하이면 센다.
 * 무효 구간(마지막 유효 프레임 뒤로 흐른 시간)이 cancelGapMs를 넘으면 진행 중이던 판정을 취소한다.
 * [update]는 프레임마다 한 번, 시각이 증가하는 순서로 부른다. 무효 프레임도 valid = false로 넘긴다.
 * 원본이 정한 가정(step_features.py:77-80): 길이는 보간하지 않은 프레임 시각의 차, 하품은 값이 exit 이상으로 유지되는
 * 프레임에서만 센다, 무효 구간은 프레임 자체가 끊긴 시간도 포함한다.
 */
internal class ThresholdEventCounter(
    private val enterThr: Double,
    private val exitThr: Double,
    private val minDurationMs: Long,
    private val maxDurationMs: Long,
    private val countAtEnd: Boolean,
    private val cancelGapMs: Long,
) {
    private var activeValue = false
    private var startMs: Long = 0L
    private var lastValidMs: Long = 0L // 진행 중일 때만 뜻이 있다
    private var counted = false // 진행 중인 이벤트를 이미 셌는지(countAtEnd = false에서만 쓴다)

    /** step_features.py:104-107. 이벤트가 진행 중인지. */
    val active: Boolean
        get() = activeValue

    /** step_features.py:109-114. 처음 상태로 되돌린다. 진행 중이던 판정은 버린다. */
    fun reset() {
        activeValue = false
        startMs = 0L
        lastValidMs = 0L
        counted = false
    }

    /** step_features.py:116-139. 프레임 하나를 처리한다. 이 프레임에서 이벤트를 하나 셌으면 true. */
    fun update(tMs: Long, value: Double, valid: Boolean): Boolean {
        if (activeValue && tMs - lastValidMs > cancelGapMs) {
            activeValue = false
        }
        if (!valid) {
            return false
        }

        var countedNow = false
        if (!activeValue) {
            if (value >= enterThr) {
                activeValue = true
                startMs = tMs
                counted = false
            }
        } else if (value < exitThr) {
            activeValue = false
            if (countAtEnd) {
                val durationMs = tMs - startMs
                countedNow = durationMs >= minDurationMs && durationMs <= maxDurationMs
            }
        } else if (!countAtEnd && !counted) {
            if (tMs - startMs >= minDurationMs) {
                counted = true
                countedNow = true
            }
        }
        lastValidMs = tMs
        return countedNow
    }
}

/**
 * step_features.py:142-154. 하품 계수기. 입력값은 jaw_rel = jaw_open − jaw0이다.
 * 시작 yawn.enter_thr, 끝 yawn.exit_thr, 길이가 yawn.count_at_ms에 도달하면 한 번 센다.
 */
internal fun yawnCounter(cfg: YawnSpec): ThresholdEventCounter = ThresholdEventCounter(
    enterThr = cfg.enterThr,
    exitThr = cfg.exitThr,
    minDurationMs = cfg.countAtMs,
    maxDurationMs = cfg.countAtMs,
    countAtEnd = false,
    cancelGapMs = cfg.cancelGapMs,
)

/**
 * step_features.py:157-212. 끄덕임 검출기.
 *
 * d = pitch_rel − m이 nod.enter_deg 이상으로 시작, nod.exit_deg 미만으로 끝, 길이가 min_duration_ms~max_duration_ms이면
 * 끝날 때 센다. m은 pitch_rel의 느린 평균이다. 유효 프레임마다 d를 먼저 계산한 뒤 m += α·(pitch_rel − m),
 * α = dt / (ema_tau_ms + dt)로 갱신한다. m의 초기값은 첫 유효 표본이고(그 프레임의 d는 0이다) 무효 프레임에서는 갱신하지
 * 않는다. 원본의 가정(step_features.py:165): dt는 직전 유효 표본과의 간격이고 ema_max_dt_ms를 넘으면 그 값으로 자른다.
 */
internal class NodDetector(cfg: NodSpec) {

    private val tauMs: Long = cfg.emaTauMs
    private val maxDtMs: Long = cfg.emaMaxDtMs
    private val counter = ThresholdEventCounter(
        enterThr = cfg.enterDeg,
        exitThr = cfg.exitDeg,
        minDurationMs = cfg.minDurationMs,
        maxDurationMs = cfg.maxDurationMs,
        countAtEnd = true,
        cancelGapMs = cfg.cancelGapMs,
    )
    private var hasMean = false
    private var meanValue: Double = 0.0
    private var prevValidMs: Long = 0L

    /** step_features.py:183-186. 느린 평균 m. 유효 표본을 아직 받지 않았으면 0.0이다. */
    val mean: Double
        get() = meanValue

    /** step_features.py:188-193. 처음 상태로 되돌린다. 느린 평균도 버린다. */
    fun reset() {
        counter.reset()
        hasMean = false
        meanValue = 0.0
        prevValidMs = 0L
    }

    /** step_features.py:195-212. 프레임 하나를 처리한다. 이 프레임에서 끄덕임을 하나 셌으면 true. */
    fun update(tMs: Long, pitchRel: Double, valid: Boolean): Boolean {
        if (!valid) {
            return counter.update(tMs, 0.0, false)
        }

        var d = 0.0
        if (hasMean) {
            d = pitchRel - meanValue
            var dtMs = tMs - prevValidMs
            if (dtMs > maxDtMs) {
                dtMs = maxDtMs
            }
            // Python `float(dt_ms) / float(self._tau_ms + dt_ms)`: 정수 합을 먼저 구하고 둘 다 실수로 바꿔 나눈다(R3).
            val alpha = dtMs.toDouble() / (tauMs + dtMs).toDouble()
            meanValue += alpha * (pitchRel - meanValue)
        } else {
            hasMean = true
            meanValue = pitchRel
        }
        prevValidMs = tMs
        return counter.update(tMs, d, true)
    }
}
