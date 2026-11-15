package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.GazeZone
import hawkeye.drowsy.core.Source
import hawkeye.drowsy.core.spec.ForwardZoneRules

// 원본: drowsy/rules.py:156-335 `ForwardZone` (SPEC 12.4절). 계획 5.6절의 선언 그대로다.
// 관측, 표지, 디바운스, 영역 밖 시간의 정의와 원본이 정한 가정(rules.py:184-190: 초기 상태는 영역 안, 영역 안 관측도
// 이어받는다, 확정은 프레임이 올 때만, 확정 전에 UNKNOWN을 만난 후보는 버린다, out_source는 마지막 영역 밖 관측의
// source)은 원본 클래스 설명과 SPEC 12.4절에 있다. 이 파일은 그것을 문장 순서 그대로 옮긴다.
// 표지 문자열 ZONE_IN, ZONE_OUT, ZONE_UNKNOWN(rules.py:89-91)은 GazeZone 열거형이다. 시각과 길이는 Long이다(R1).
// 실수의 절댓값은 원본처럼 `if (x < 0.0) x = -x`로 구한다(R6: abs, maxOf를 쓰지 않는다).
//
// 가정(A1): 비교는 JVM·Android와 CPython에서 같은 결과를 낸다. yaw_rel, pitch_rel, gaze_*_rel과 설정 임계값의 비교
// (yaw_abs > yaw_abs_max_deg 등)는 같은 double에서 같은 판정이다.

/**
 * rules.py:156-335. 전방 주시 상태 머신.
 *
 * 규칙 엔진이 읽는 값: [zone], [label], [confirmedIn], [outStartMs], [outPoseMs], [outSource], [unknownMs],
 * [inStartMs], [away], [candidatePending]. 스레드에 안전하지 않다. 상태는 인스턴스에만 있다(R19).
 */
internal class ForwardZone(private val cfg: ForwardZoneRules, private val maxHoldMs: Long, awayWindowMs: Long) {

    /** rules.py:196. 확정된 영역 밖 상태에서 표지가 OUT이던 시간 구간. */
    val away: IntervalWindow = IntervalWindow(awayWindowMs)

    /** rules.py:198. 확정된 상태. IN과 OUT만 갖는다. */
    var zone: GazeZone = GazeZone.IN
        private set

    /** rules.py:199. 마지막 프레임의 표지. IN, OUT, UNKNOWN. */
    var label: GazeZone = GazeZone.UNKNOWN
        private set

    /** rules.py:200. 이 프레임에서 영역 안이 확정됐는지. */
    var confirmedIn: Boolean = false
        private set

    /** rules.py:201. 지금(또는 마지막) 영역 밖 상태가 시작된 프레임. */
    var outStartMs: Long = 0L
        private set

    /** rules.py:202. 그 프레임에서의 pose_valid 누적 시간. */
    var outPoseMs: Long = 0L
        private set

    /** rules.py:203. 마지막 유효 영역 밖 관측의 source. */
    var outSource: Source = Source.HEAD_POSE
        private set

    /** rules.py:204. 영역 밖 상태가 시작된 뒤 세지 않은 시간. */
    var unknownMs: Long = 0L
        private set

    /** rules.py:205. 지금(또는 마지막) 영역 안 상태가 시작된 프레임. */
    var inStartMs: Long = 0L
        private set

    private var hasPrev = false
    private var prevTMs: Long = 0L
    private var prevObserved = false // 직전 프레임에 유효 관측이 있었는지
    private var coverLabel: GazeZone = GazeZone.UNKNOWN // 이어받을 표지
    private var lostMs: Long = 0L // 마지막 유효 관측이 유지된 끝
    private var acctMs: Long = 0L // 여기까지 영역 밖 시간을 정산했다

    private var cand = false // 반대 표지의 후보가 걸려 있는지
    private var candMs: Long = 0L
    private var candPoseMs: Long = 0L

    /**
     * rules.py:218-225. 반대 표지의 후보가 확정도 폐기도 되지 않은 채 걸려 있는지.
     * 걸려 있는 동안에는 후보 구간이 unknownMs에 아직 들어가지 않았으므로 t − outStartMs − unknownMs가 후보 구간만큼
     * 크게 읽힌다.
     */
    val candidatePending: Boolean
        get() = cand

    /**
     * rules.py:227-237. 처음 상태로 되돌린다. 영역 밖 누적도 지운다.
     * 원본 그대로 아래 필드만 되돌린다. outStartMs, outPoseMs, outSource, inStartMs, prevTMs, lostMs, acctMs, candMs,
     * candPoseMs는 두며, 다음 프레임(hasPrev = false)이 inStartMs부터 다시 쓴다.
     */
    fun reset() {
        away.clear()
        zone = GazeZone.IN
        label = GazeZone.UNKNOWN
        confirmedIn = false
        unknownMs = 0L
        hasPrev = false
        prevObserved = false
        coverLabel = GazeZone.UNKNOWN
        cand = false
    }

    /** rules.py:239-312. 프레임 하나를 처리한다. poseMs는 이 프레임까지의 pose_valid 누적 시간이다. */
    fun update(state: FrameState, analyzable: Boolean, poseMs: Long) {
        val t = state.tMs
        val creditMs = cfg.faceLossCreditMs
        confirmedIn = false

        val observed = analyzable && state.poseValid
        var obs = GazeZone.UNKNOWN
        if (observed) {
            obs = observe(state)
        }

        if (!hasPrev) {
            inStartMs = t
        } else {
            if (prevObserved) {
                var heldMs = t - prevTMs
                if (heldMs > maxHoldMs) {
                    heldMs = maxHoldMs
                }
                lostMs = prevTMs + heldMs
            }
            val coverEndMs = lostMs + creditMs

            // 지난 구간 [직전 프레임, 이 프레임)의 정산
            if (cand) {
                if (coverLabel == GazeZone.UNKNOWN || t > coverEndMs) {
                    if (zone == GazeZone.OUT) {
                        unknownMs += t - candMs
                    }
                    cand = false
                }
            } else if (zone == GazeZone.OUT) {
                var awayEndMs = acctMs
                if (coverLabel == GazeZone.OUT) {
                    awayEndMs = t
                    if (awayEndMs > coverEndMs) {
                        awayEndMs = coverEndMs
                    }
                    if (awayEndMs < acctMs) {
                        awayEndMs = acctMs
                    }
                }
                away.add(startMs = acctMs, endMs = awayEndMs)
                unknownMs += t - awayEndMs
            }
        }

        // 이 프레임의 표지
        var label = obs
        if (!observed && analyzable && hasPrev) {
            if (coverLabel != GazeZone.UNKNOWN && t < lostMs + creditMs) {
                label = coverLabel
            }
        }
        coverLabel = label

        // 디바운스
        if (label != GazeZone.UNKNOWN) {
            if (label == zone) {
                if (cand) {
                    if (zone == GazeZone.OUT) {
                        away.add(startMs = candMs, endMs = t)
                    }
                    cand = false
                }
            } else {
                if (!cand) {
                    cand = true
                    candMs = t
                    candPoseMs = poseMs
                }
                if (t - candMs >= cfg.debounceMs) {
                    cand = false
                    zone = label
                    if (label == GazeZone.OUT) {
                        outStartMs = candMs
                        outPoseMs = candPoseMs
                        unknownMs = 0L
                        away.add(startMs = candMs, endMs = t)
                    } else {
                        inStartMs = candMs
                        confirmedIn = true
                    }
                }
            }
        }

        this.label = label
        acctMs = t
        hasPrev = true
        prevTMs = t
        prevObserved = observed
    }

    /** rules.py:314-335 `_observe`. 유효 관측 하나의 표지(IN 또는 OUT). OUT이면 outSource를 적는다. */
    private fun observe(state: FrameState): GazeZone {
        var yawAbs = state.yawRel
        if (yawAbs < 0.0) {
            yawAbs = -yawAbs
        }
        var headOut = yawAbs > cfg.yawAbsMaxDeg
        if (state.pitchRel < -cfg.pitchUpMaxDeg || state.pitchRel > cfg.pitchDownMaxDeg) {
            headOut = true
        }
        if (headOut) {
            outSource = Source.HEAD_POSE
            return GazeZone.OUT
        }
        if (cfg.eyeAuxEnabled) {
            var gazeHAbs = state.gazeHRel
            if (gazeHAbs < 0.0) {
                gazeHAbs = -gazeHAbs
            }
            var gazeVAbs = state.gazeVRel
            if (gazeVAbs < 0.0) {
                gazeVAbs = -gazeVAbs
            }
            if (gazeHAbs > cfg.gazeHAbsMax || gazeVAbs > cfg.gazeVAbsMax) {
                outSource = Source.EYE
                return GazeZone.OUT
            }
        }
        return GazeZone.IN
    }
}
