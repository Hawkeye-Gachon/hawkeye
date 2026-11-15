package hawkeye.drowsy.core.session

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.BaselineStatus
import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.GazeZone
import hawkeye.drowsy.core.Phase
import hawkeye.drowsy.core.PipelineOutput
import hawkeye.drowsy.core.Source
import java.util.Collections

// 새 코드 (계획 4.1절 `PKG.core.session`, 6.2절). 파이프라인 출력의 순수 함수만 둔다. 판정 로직을 바꾸거나 거르거나
// 평활하지 않는다(계획 1.3절).

/** 보정 상태. `baseline == null`이면 CALIBRATING, 아니면 기준선 status를 같은 이름으로(계획 6.2절). */
enum class CalibrationStatus { CALIBRATING, OK, OFF_AXIS, FALLBACK_GLOBAL }

/** `DriverPipeline`의 계수 다섯 (pipeline.py:127-131, SPEC 7.2). */
data class PipelineCounters(
    val framesAccepted: Long,
    val droppedFrames: Long,
    val stepsEmitted: Long,
    val modelCalls: Long,
    val nonfiniteWindows: Long,
)

/**
 * 지금 진행 중인 상태 하나(계획 6.2절의 접기). onset 이벤트의 필드를 옮긴 것이다.
 *
 * - provisional: 그 onset의 provisional.
 * - onsetFrameTMs: 그 onset을 낸 프레임의 tMs(`output.state.tMs`). tStartMs(상태가 시작된 시각)와 다를 수 있다.
 */
data class ActiveEvent(
    val type: EventType,
    val severity: Int,
    val tStartMs: Long,
    val source: Source,
    val provisional: Boolean,
    val onsetFrameTMs: Long,
)

/**
 * 받아들인 프레임 하나의 결과 (계획 4.1절, 6.2절). 앱은 만들지 않는다.
 *
 * 필드의 출처(계획 6.2절 표):
 * - output: `update`가 돌려준 객체 그대로. 거르지도, 순서를 바꾸지도, 합치지도 않는다.
 * - gazeZone, baseline, counters: 같은 `update` 직후에 읽은 `DriverPipeline`의 속성.
 * - speedKmh: 그 `update`가 받은 속도.
 * - sessionMs: tMs − 이 세션 첫 DriverState의 tMs.
 * - active: 진행 중 이벤트 접기(ActiveFold)의 결과. rules.json `alert_priority` 순서. 읽기 전용 목록이다.
 */
class DriverState internal constructor(
    val output: PipelineOutput,
    val gazeZone: GazeZone,
    val baseline: Baseline?,
    val counters: PipelineCounters,
    val speedKmh: Double?,
    val sessionMs: Long,
    val active: List<ActiveEvent>,
) {
    /** `output.state.tMs`. */
    val tMs: Long get() = output.state.tMs

    /** `output.events`, 발행 순서(SPEC 7.3). */
    val events: List<DriverStateEvent> get() = output.events

    /** `output.alerts`, 우선순위 순서(SPEC 13장). */
    val alerts: List<DriverStateEvent> get() = output.alerts

    /** alerts가 비지 않았으면 alerts[0](가장 급한 경고), 아니면 null (docs/GUIDE.md:939). */
    val primaryAlert: DriverStateEvent?
        get() {
            val a = output.alerts
            return if (a.isEmpty()) null else a[0]
        }

    /** baseline이 null이면 CALIBRATING, 아니면 baseline.status를 같은 이름으로. */
    val calibration: CalibrationStatus
        get() {
            val b = baseline ?: return CalibrationStatus.CALIBRATING
            return when (b.status) {
                BaselineStatus.OK -> CalibrationStatus.OK
                BaselineStatus.OFF_AXIS -> CalibrationStatus.OFF_AXIS
                BaselineStatus.FALLBACK_GLOBAL -> CalibrationStatus.FALLBACK_GLOBAL
            }
        }

    /** active에 그 종류가 있는지. */
    fun isActive(type: EventType): Boolean {
        for (a in active) {
            if (a.type == type) return true
        }
        return false
    }

    override fun toString(): String =
        "DriverState(tMs=$tMs, sessionMs=$sessionMs, calibration=$calibration, gazeZone=$gazeZone, " +
            "speedKmh=$speedKmh, events=${output.events.size}, alerts=${output.alerts.size}, " +
            "active=${active.map { it.type }}, counters=$counters)"
}

/**
 * 진행 중 이벤트 접기 (계획 6.2절, K10 (가), docs/GUIDE.md:940-942).
 *
 * ```
 * 상태: 맵 EventType → ActiveEvent. 새 세션은 비어 있다
 * 받아들인 프레임마다, output.events를 순서대로:
 *     onset: map[type] = ActiveEvent(type, severity, tStartMs, source, onset의 provisional, onsetFrameTMs = output.state.tMs)
 *            type == SLEEP 이면 MICROSLEEP을 지운다
 *     end:   type을 지운다
 *            type == SLEEP 이면 MICROSLEEP을 지운다
 * active = 맵의 값을 rules.json alert_priority 순서로
 * ```
 *
 * 시계, 임계값, 평활이 없다. 버린 프레임에는 부르지 않는다. 이벤트가 없는 프레임에서는 목록이 바뀌지 않는다(같은 객체를
 * 돌려준다). 맵은 순회하지 않는다. 순회는 언제나 alertPriority 목록이다.
 *
 * 스레드에 안전하지 않다. 세션 하나 = 인스턴스 하나.
 */
internal class ActiveFold(alertPriority: List<EventType>) {
    private val priority: List<EventType> = ArrayList(alertPriority)
    private val byType = HashMap<EventType, ActiveEvent>()
    private var view: List<ActiveEvent> = Collections.emptyList()

    init {
        // alert_priority는 종류마다 한 번이어야 한다(rules.json 검사 R19, spec.py:919-922). 빠진 종류가 있으면
        // 그 종류의 진행 중 상태가 목록에서 조용히 사라지므로 여기서도 막는다.
        require(priority.size == EventType.entries.size) {
            "alertPriority must list every event type exactly once, got $alertPriority"
        }
        for (t in EventType.entries) {
            var count = 0
            for (p in priority) {
                if (p == t) count++
            }
            require(count == 1) { "alertPriority must list every event type exactly once, got $alertPriority" }
        }
    }

    /** 지금의 목록(읽기 전용). */
    val active: List<ActiveEvent> get() = view

    /** 새 세션처럼 비운다. */
    fun reset() {
        byType.clear()
        view = Collections.emptyList()
    }

    /** 받아들인 프레임 하나의 events를 순서대로 접고 그 뒤의 목록을 돌려준다. frameTMs = `output.state.tMs`. */
    fun apply(events: List<DriverStateEvent>, frameTMs: Long): List<ActiveEvent> {
        if (events.isEmpty()) return view
        for (e in events) {
            when (e.phase) {
                Phase.ONSET -> {
                    byType[e.type] = ActiveEvent(
                        type = e.type,
                        severity = e.severity,
                        tStartMs = e.tStartMs,
                        source = e.source,
                        provisional = e.provisional,
                        onsetFrameTMs = frameTMs,
                    )
                    if (e.type == EventType.SLEEP) byType.remove(EventType.MICROSLEEP)
                }
                Phase.END -> {
                    byType.remove(e.type)
                    if (e.type == EventType.SLEEP) byType.remove(EventType.MICROSLEEP)
                }
            }
        }
        val next = ArrayList<ActiveEvent>()
        for (t in priority) {
            val a = byType[t]
            if (a != null) next.add(a)
        }
        view = Collections.unmodifiableList(next)
        return view
    }
}
