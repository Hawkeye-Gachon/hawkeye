package hawkeye.drowsy.core.session

import hawkeye.drowsy.core.DriverPipeline
import hawkeye.drowsy.core.DrowsyInternalApi
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.ModelFn
import hawkeye.drowsy.core.RawFrame
import hawkeye.drowsy.core.check.StartupResult
import java.util.concurrent.atomic.AtomicBoolean

// 새 코드 (계획 4.1절, 5.8절의 `DriveSession.feed`, 리드 판정 9의 1번). 녹화 탭(RawFrameTap, LandmarkTable*)은 판정 9가
// 뺐으므로 없다. 파이프라인 출력에 파생 값(sessionMs, active)을 붙일 뿐이고 판정을 바꾸거나 거르지 않는다(계획 1.3절).

/**
 * 운전 한 번 = 인스턴스 하나. 스레드에 안전하지 않다. 호출이 겹치면 IllegalStateException(G11).
 * 앱이 쓰지 않는다. 같은 라이브러리의 입구(`hawkeye.drowsy.DrowsinessMonitor`)와 시험만 쓴다.
 *
 * - pipeline: 시작 검사를 통과한 설정으로 만든 파이프라인(시험은 기준선을 넣은 파이프라인을 줄 수 있다).
 * - alertPriority: rules.json의 alert_priority. `active`의 순서다(계획 6.2절).
 */
@DrowsyInternalApi
class DriveSession internal constructor(
    private val pipeline: DriverPipeline,
    alertPriority: List<EventType>,
) {
    private val fold = ActiveFold(alertPriority)

    // 바쁨 표시(계획 5.8절 1번). 두 스레드가 겹쳐 부르는 것도 잡도록 compareAndSet으로 세운다.
    private val busy = AtomicBoolean(false)

    // 이 세션 첫 DriverState의 tMs(계획 6.2절 sessionMs). 첫 상태 전에는 null이다.
    private var firstTMs: Long? = null

    /** 마지막으로 만든 DriverState. 받아들인 프레임이 아직 없으면 null이다. 버린 프레임은 이것을 바꾸지 않는다. */
    var lastState: DriverState? = null
        private set

    /** 파이프라인의 계수 다섯, 지금 값 (pipeline.py:127-131). */
    val counters: PipelineCounters
        get() = PipelineCounters(
            framesAccepted = pipeline.framesAccepted,
            droppedFrames = pipeline.droppedFrames,
            stepsEmitted = pipeline.stepsEmitted,
            modelCalls = pipeline.modelCalls,
            nonfiniteWindows = pipeline.nonfiniteWindows,
        )

    /**
     * 프레임 하나를 파이프라인에 넣는다(계획 5.8절의 순서). null = 파이프라인이 버린 프레임(tMs가 직전에 받아들인 값 이하)이고,
     * 그때 접기 상태와 lastState는 그대로다.
     *
     * 1. 바쁨 표시를 세운다. 이미 서 있으면 IllegalStateException("update must be called one at a time").
     * 2. out = pipeline.update(frame, speedKmh). null이면 표시를 내리고 null을 돌려준다.
     * 3. 첫 DriverState의 tMs를 원점으로 sessionMs.
     * 4. out.events를 순서대로 접어 active를 고친다.
     * 5. DriverState(out, gazeZone, baseline, counters, speed, sessionMs, active)를 만들고 lastState에 둔다.
     *
     * 파이프라인이 예외를 내면 표시를 내리고 그 예외를 그대로 던진다(그 뒤의 상태는 정하지 않는다; 입구가 멈추고 알린다).
     */
    fun feed(frame: RawFrame, speedKmh: Double?): DriverState? {
        if (!busy.compareAndSet(false, true)) throw IllegalStateException("update must be called one at a time")
        try {
            val out = pipeline.update(frame, speedKmh) ?: return null
            val tMs = out.state.tMs
            val origin = firstTMs ?: tMs.also { firstTMs = it }
            val active = fold.apply(out.events, tMs)
            val state = DriverState(
                output = out,
                gazeZone = pipeline.gazeZone,
                baseline = pipeline.baseline,
                counters = counters,
                speedKmh = speedKmh,
                sessionMs = tMs - origin,
                active = active,
            )
            lastState = state
            return state
        } finally {
            busy.set(false)
        }
    }

    // BCV는 클래스의 표시를 companion에 옮기지 않으므로(단계 0b의 PreviewPlayer와 같다) companion에도 붙인다.
    @DrowsyInternalApi
    companion object {
        /**
         * 시작 검사를 통과한 설정으로만 만든다. checked.passed가 아니면 IllegalArgumentException(계획 9.4절 SD5).
         * 기준선은 넣지 않는다(세션이 스스로 보정한다). model은 그대로 파이프라인에 넘긴다(null이면 확률이 언제나 null).
         */
        @DrowsyInternalApi
        @JvmStatic
        fun create(checked: StartupResult, model: ModelFn?): DriveSession {
            require(checked.passed) {
                "DriveSession needs a StartupResult that passed; problems: " +
                    checked.problems.joinToString("; ") { it.message }
            }
            val spec = checked.spec ?: throw IllegalArgumentException("StartupResult passed without a feature spec")
            val rules = checked.rules ?: throw IllegalArgumentException("StartupResult passed without rules")
            val pipeline = DriverPipeline(
                spec = spec.pipeline,
                rules = rules,
                baseline = null,
                featureStartMs = null,
                model = model,
            )
            return DriveSession(pipeline = pipeline, alertPriority = rules.alertPriority)
        }
    }
}
