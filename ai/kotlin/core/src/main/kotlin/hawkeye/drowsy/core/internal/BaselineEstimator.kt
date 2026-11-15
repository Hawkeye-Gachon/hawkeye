package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.BaselineDiagnostics
import hawkeye.drowsy.core.BaselineStatus
import hawkeye.drowsy.core.spec.PipelineSpec
import java.util.Collections
import kotlin.math.abs

/**
 * 원본: drowsy/calibrate.py:118-284 `BaselineEstimator` (SPEC 8.2, 8.3). 계획 5.3절의 선언 그대로다.
 *
 * 세션 앞 60초의 얼굴 프레임으로 기준선을 한 번 만든다. 받아들인 프레임마다 [update]를 부른다. 기준선이 확정되는
 * 프레임에서 Baseline을 한 번 돌려주고, 그 전과 그 뒤에는 null을 돌려준다. 확정 뒤에는 [diagnostics]가 채워진다.
 *
 * - 처음 받은 프레임의 tMs가 세션 원점이고, 그 뒤로는 세션 시간(ms)으로 계산한다.
 * - 수집 구간은 세션 시간 [0, 마감)이다. 마감은 baseline.duration_ms(60000)에서 시작한다. 세션 시간이 마감 이상인 첫
 *   프레임이 오면 그 프레임을 넣기 전에 완성 조건(얼굴 프레임 수, 얼굴 시간 비율, 두 눈 E의 범위)을 검사한다.
 *   어기면 마감을 extend_step_ms만큼 늦추고(최대 max_duration_ms), 거기서도 어기면 전역 상수로 대체한다.
 * - 얼굴 프레임은 다음 프레임까지의 간격을 최대 clock.max_hold_ms까지 대표하고, 마감을 넘는 부분은 세지 않는다
 *   (`held_before = min(deadline − prev_t, held)`의 min을 빠뜨리지 않는다, 계획 5.3절).
 * - 프레임이 끊겼다가 와서 마감을 여러 번 지나면 지난 마감을 순서대로 하나씩 검사한다(`while t >= deadline`, R16).
 * - tMs가 직전 프레임 이하이면 IllegalArgumentException(Python은 ValueError)이고 상태를 바꾸지 않는다.
 *
 * 시각과 길이는 모두 Long이다(R1). 스레드에 안전하지 않다.
 */
internal class BaselineEstimator(private val spec: PipelineSpec) {

    private val cfg = spec.baseline
    private val maxHoldMs: Long = spec.clock.maxHoldMs
    private val minEyeWidthPx: Double = spec.signals.minEyeWidthPx

    private var done = false
    private var deadlineMs: Long = cfg.durationMs

    // 직전 프레임(세션 시간). 얼굴 프레임이 대표하는 시간을 다음 프레임이 올 때 확정하려고 둔다.
    private var hasPrev = false
    private var originMs: Long = 0L
    private var prevT: Long = 0L
    private var prevFace = false
    private var faceMs: Long = 0L

    // 표본 목록 열 개 (calibrate.py:164-173). 길이 제한이 없다.
    private val earL = ArrayList<Double>()
    private val earR = ArrayList<Double>()
    private val blinkL = ArrayList<Double>()
    private val blinkR = ArrayList<Double>()
    private val yaw = ArrayList<Double>()
    private val pitch = ArrayList<Double>()
    private val roll = ArrayList<Double>()
    private val jaw = ArrayList<Double>()
    private val gazeHValues = ArrayList<Double>()
    private val gazeVValues = ArrayList<Double>()

    /** 기준선을 확정한 시점의 판정 통계. 확정 전에는 null이다(calibrate.py:175). */
    var diagnostics: BaselineDiagnostics? = null
        private set

    /** calibrate.py:177-214. 프레임 하나를 받는다. 이 프레임에서 기준선이 확정되면 Baseline을, 아니면 null을 돌려준다. */
    fun update(s: FrameSignals): Baseline? {
        if (done) {
            return null
        }

        if (!hasPrev) {
            originMs = s.tMs
        }
        val t = s.tMs - originMs
        if (hasPrev && t <= prevT) {
            throw IllegalArgumentException("t_ms must be strictly greater than the previous frame's t_ms")
        }

        // 직전 프레임이 얼굴 프레임이면 이 프레임까지의 간격을 최대 max_hold_ms까지 대표한다.
        var heldMs = 0L
        if (hasPrev && prevFace) {
            heldMs = t - prevT
            if (heldMs > maxHoldMs) {
                heldMs = maxHoldMs
            }
        }

        while (t >= deadlineMs) {
            var heldBeforeDeadline = deadlineMs - prevT
            if (heldBeforeDeadline > heldMs) {
                heldBeforeDeadline = heldMs
            }
            val baseline = tryComplete(faceMs + heldBeforeDeadline)
            if (baseline != null) {
                done = true
                return baseline
            }
            var nextDeadlineMs = deadlineMs + cfg.extendStepMs
            if (nextDeadlineMs > cfg.maxDurationMs) {
                nextDeadlineMs = cfg.maxDurationMs
            }
            deadlineMs = nextDeadlineMs
        }

        faceMs += heldMs
        if (s.face) {
            collect(s)
        }
        hasPrev = true
        prevT = t
        prevFace = s.face
        return null
    }

    /** calibrate.py:216-228 `_collect`. E의 표본에는 쓸 수 있는 눈(폭 >= min_eye_width_px)의 EAR만 넣는다. */
    private fun collect(s: FrameSignals) {
        if (s.eyeWL >= minEyeWidthPx) {
            earL.add(s.earL)
        }
        if (s.eyeWR >= minEyeWidthPx) {
            earR.add(s.earR)
        }
        blinkL.add(s.blinkL)
        blinkR.add(s.blinkR)
        yaw.add(s.yaw)
        pitch.add(s.pitch)
        roll.add(s.roll)
        jaw.add(s.jawOpen)
        gazeHValues.add(gazeH(s.eyeLook))
        gazeVValues.add(gazeV(s.eyeLook))
    }

    /**
     * calibrate.py:230-275 `_try_complete`. 확정(완성 또는 대체)이면 기준선을, 연장이면 null을 돌려준다.
     *
     * 가정(A1): 사칙연산과 비교는 JVM·Android와 CPython에서 같은 double을 낸다. face_ratio(= face_ms / deadline_ms)를
     * min_face_ratio와, E를 e_min·e_max와, |yaw0|을 off_axis_yaw_deg와 정확히 비교하므로(SPEC 8.3, 걸침 T2·T3는 RAW
     * 입력에서만 생긴다) 같은 입력에서 같은 판정이 나오는 것은 이 가정에 기댄다. yaw0는 중앙값이라 산술만 지난다.
     */
    private fun tryComplete(faceMs: Long): Baseline? {
        val deadline = deadlineMs
        val nFace = blinkL.size
        // R2: Long / Long은 정수 나눗셈이다. calibrate.py:235의 `/`는 실수 나눗셈이다.
        val faceRatio = faceMs.toDouble() / deadline.toDouble()

        var eL = 0.0
        var eR = 0.0
        var eInRange = false
        if (earL.size > 0 && earR.size > 0) {
            eL = nearestRankQuantile(earL, cfg.earQuantile)
            eR = nearestRankQuantile(earR, cfg.earQuantile)
            eInRange = eL >= cfg.eMin && eL <= cfg.eMax && eR >= cfg.eMin && eR <= cfg.eMax
        }

        val complete = nFace >= cfg.minFaceFrames && faceRatio >= cfg.minFaceRatio && eInRange
        if (!complete && deadline < cfg.maxDurationMs) {
            return null
        }

        diagnostics = BaselineDiagnostics(
            nFaceFrames = nFace,
            faceRatio = faceRatio,
            blinkLQuantiles = recordQuantiles(blinkL),
            blinkRQuantiles = recordQuantiles(blinkR),
        )
        if (!complete) {
            return baselineFromConstants(spec, deadline)
        }

        val yaw0 = median(yaw)
        val pitch0 = median(pitch)
        val roll0 = median(roll)
        var status = BaselineStatus.OK
        if (abs(yaw0) > cfg.offAxisYawDeg) {
            status = BaselineStatus.OFF_AXIS
        }
        return Baseline(
            eL = eL,
            eR = eR,
            bL = nearestRankQuantile(blinkL, cfg.blinkQuantile),
            bR = nearestRankQuantile(blinkR, cfg.blinkQuantile),
            r0 = rotationFromEuler(yaw0, pitch0, roll0),
            jaw0 = median(jaw),
            gazeH0 = median(gazeHValues),
            gazeV0 = median(gazeVValues),
            tCompleteMs = deadline,
            status = status,
        )
    }

    /** calibrate.py:277-284 `_record_quantiles`. 표본이 없으면 0.0이다. */
    private fun recordQuantiles(values: List<Double>): List<Double> {
        val out = ArrayList<Double>(cfg.blinkRecordQuantiles.size)
        for (q in cfg.blinkRecordQuantiles) {
            if (values.size > 0) {
                out.add(nearestRankQuantile(values, q))
            } else {
                out.add(0.0)
            }
        }
        return Collections.unmodifiableList(out)
    }
}
