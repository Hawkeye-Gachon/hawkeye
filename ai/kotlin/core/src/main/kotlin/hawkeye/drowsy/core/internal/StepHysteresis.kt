package hawkeye.drowsy.core.internal

// 원본: drowsy/rules.py:338-412 `StepHysteresis` (SPEC 12.6절). 계획 5.6절의 선언 그대로다.
// 원본이 정한 가정(rules.py:348-349): 재시작 금지 시간 동안에도 연속 수는 센다. 금지 시간이 끝난 첫 스텝에서 연속 수가
// 이미 찼으면 그 스텝에서 시작하고, startMs는 그 연속 구간의 첫 스텝 끝(금지 시간 안일 수 있다)이다.
// 시각은 Long이다(R1). if/elif 사슬의 순서를 그대로 둔다(R16).

/**
 * rules.py:338-412. 연속 스텝 수로 시작과 종료를 정하는 판정기. 졸음 판정의 모델 쪽과 PERCLOS 쪽이 같이 쓴다.
 *
 * - 시작 조건을 만족한 스텝이 onsetSteps개 연속이면 시작한다. startMs는 그 연속 구간의 첫 스텝 끝이다.
 * - 종료 조건을 만족한 스텝이 endSteps개 연속이면 끝난다.
 * - 끝난 뒤 refractoryMs 동안은 다시 시작하지 않는다.
 * - 값이 없는 스텝(hasValue = false)과 스텝 번호가 이어지지 않는 경우에는 두 연속 수를 0으로 되돌린다. active는 그대로다.
 * - [reset]은 만들었을 때의 상태로 되돌린다(규칙 엔진이 10초 넘는 끊김 뒤에 부른다).
 * 스레드에 안전하지 않다. 상태는 인스턴스에만 있다(R19).
 */
internal class StepHysteresis(
    private val onsetSteps: Int,
    private val endSteps: Int,
    private val refractoryMs: Long,
) {
    /** rules.py:357. 판정 중인지. */
    var active: Boolean = false
        private set

    /** rules.py:358. 지금(또는 마지막) 판정 구간의 t_start_ms. */
    var startMs: Long = 0L
        private set

    private var hasK = false
    private var lastK: Int = 0
    private var onsetRun: Int = 0
    private var endRun: Int = 0
    private var runStartMs: Long = 0L // 지금 세고 있는 시작 조건 연속 구간의 첫 스텝 끝
    private var hasEnded = false
    private var lastEndMs: Long = 0L

    /** rules.py:368-378. 처음 상태로 되돌린다: 판정 상태, 두 연속 수, 스텝 번호의 이어짐, 재시작 금지 기억을 모두 지운다. */
    fun reset() {
        active = false
        startMs = 0L
        hasK = false
        lastK = 0
        onsetRun = 0
        endRun = 0
        runStartMs = 0L
        hasEnded = false
        lastEndMs = 0L
    }

    /** rules.py:380-412. 스텝 하나를 처리한다. 결과는 [active]와 [startMs]로 읽는다. */
    fun update(k: Int, tEndMs: Long, hasValue: Boolean, onsetHit: Boolean, endHit: Boolean) {
        // Python `k != self._last_k + 1`: Python int는 넘치지 않으므로 Long으로 더한다(lastK가 Int.MAX_VALUE여도 같은 판정).
        if (hasK && k.toLong() != lastK.toLong() + 1L) {
            onsetRun = 0
            endRun = 0
        }
        hasK = true
        lastK = k

        if (!hasValue) {
            onsetRun = 0
            endRun = 0
            return
        }

        if (onsetHit) {
            if (onsetRun == 0) {
                runStartMs = tEndMs
            }
            onsetRun += 1
        } else {
            onsetRun = 0
        }
        if (endHit) {
            endRun += 1
        } else {
            endRun = 0
        }

        if (!active) {
            if (onsetRun >= onsetSteps) {
                if (!hasEnded || tEndMs - lastEndMs >= refractoryMs) {
                    active = true
                    startMs = runStartMs
                }
            }
        } else if (endRun >= endSteps) {
            active = false
            hasEnded = true
            lastEndMs = tEndMs
        }
    }
}
