package hawkeye.drowsy.core.internal

import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.Phase
import hawkeye.drowsy.core.spec.RulesConfig

// 원본: drowsy/rules.py:876-1012 `AlertPolicy`, `_reissued` (SPEC 13장). 계획 5.6절의 선언 그대로다.
// 경고가 되는 것, 재경고 대기, 속도 억제, 미뤄 둔 경고, 되풀이, 우선순위의 규칙과 원본이 정한 가정(rules.py:907-912:
// 같은 호출에서 통과한 경고는 모두 돌려준다, 막힌 onset은 대기를 새로 시작하지 않는다, end 없이 다시 온 onset은 새
// onset으로 바꾼다, sleep의 onset·end는 microsleep의 되풀이 기록도 지운다, unable_to_analyze의 억제 없음은 설정값
// min_speed_kmh 0으로 표현한다)은 원본 클래스 설명과 SPEC 13장에 있다. 이 파일은 그것을 문장 순서 그대로 옮긴다.
//
// 네 맵은 HashMap<EventType, ...>이고 순회하지 않는다(계획 5.6절, R21). 순회는 언제나 alertPriority나 events 목록이다.
// 경고 순서는 우선순위 바깥 고리, 통과한 것 안쪽 고리 그대로다(rules.py:970-974). 시각은 Long이다(R1).

/**
 * rules.py:876-998. 경고 결정. 상태 이벤트와 차량 속도를 받아 실제로 경고할 이벤트를 돌려준다.
 * 프레임마다 [decide]를 한 번 부른다(이벤트가 없어도 부른다. 미뤄 둔 경고와 되풀이 경고의 시각을 재야 한다).
 * 스레드에 안전하지 않다. 상태는 인스턴스에만 있다(R19).
 */
internal class AlertPolicy(private val cfg: RulesConfig) {

    private val lastAlertMs = HashMap<EventType, Long>() // 종류 -> 마지막 경고 시각
    private val pending = HashMap<EventType, DriverStateEvent>() // 되풀이하지 않는 종류 -> 막혀서 미뤄 둔 시작 이벤트
    private val repeatOnset = HashMap<EventType, DriverStateEvent>() // 되풀이하는 종류 -> 이어지는 상태의 시작 이벤트
    private val repeatAlertMs = HashMap<EventType, Long>() // 되풀이하는 종류 -> 그 상태에서 낸 마지막 경고 시각

    /** rules.py:922-975. 이 프레임에서 실제로 경고할 이벤트를 우선순위 순으로 돌려준다. */
    fun decide(events: List<DriverStateEvent>, speedKmh: Double?, tMs: Long): List<DriverStateEvent> {
        val passed = ArrayList<DriverStateEvent>()

        // 1. 이 호출의 이벤트. 상태가 끝났으면 그 종류의 기록을 지우고, 시작이면 경고하거나 미뤄 둔다.
        for (event in events) {
            val policy = cfg.events.get(event.type)
            if (event.phase == Phase.END) {
                forget(event.type)
                if (event.type == EventType.SLEEP) {
                    forget(EventType.MICROSLEEP)
                }
                continue
            }
            if (event.type == EventType.CALIBRATING) {
                continue
            }
            forget(event.type)
            if (event.type == EventType.SLEEP) {
                forget(EventType.MICROSLEEP)
            }
            if (policy.repeatMs > 0L) {
                repeatOnset[event.type] = event
            }
            if (allowed(event.type, speedKmh, tMs)) {
                mark(event.type, tMs)
                passed.add(event)
            } else if (policy.repeatMs == 0L) {
                pending[event.type] = event
            }
        }

        // 2. 미뤄 둔 경고: 막힘이 풀렸으면 한 번 내고 지운다. 이 호출에서 방금 미뤄 둔 것은 같은 조건으로 다시 보는 것이라
        //    여기서 통과하지 못한다(그래서 "이후의 호출"에서만 나간다).
        for (eventType in cfg.alertPriority) {
            val held = pending[eventType] ?: continue
            if (allowed(eventType, speedKmh, tMs)) {
                mark(eventType, tMs)
                passed.add(reissued(held, tMs))
                pending.remove(eventType)
            }
        }

        // 3. 되풀이
        for (eventType in cfg.alertPriority) {
            val onset = repeatOnset[eventType] ?: continue
            val lastRepeat = repeatAlertMs[eventType]
            if (lastRepeat != null) {
                if (tMs - lastRepeat < cfg.events.get(eventType).repeatMs) {
                    continue
                }
            }
            if (allowed(eventType, speedKmh, tMs)) {
                mark(eventType, tMs)
                passed.add(reissued(onset, tMs))
            }
        }

        val alerts = ArrayList<DriverStateEvent>()
        for (eventType in cfg.alertPriority) {
            for (event in passed) {
                if (event.type == eventType) {
                    alerts.add(event)
                }
            }
        }
        return alerts
    }

    /** rules.py:977-984 `_allowed`. 재경고 대기와 속도 억제. 속도가 null이거나 기준이 0 이하이면 억제하지 않는다. */
    private fun allowed(eventType: EventType, speedKmh: Double?, tMs: Long): Boolean {
        val policy = cfg.events.get(eventType)
        val last = lastAlertMs[eventType]
        if (last != null) {
            if (tMs - last < policy.cooldownMs) {
                return false
            }
        }
        if (speedKmh != null && policy.minSpeedKmh > 0.0 && speedKmh < policy.minSpeedKmh) {
            return false
        }
        return true
    }

    /** rules.py:986-989 `_mark`. */
    private fun mark(eventType: EventType, tMs: Long) {
        lastAlertMs[eventType] = tMs
        if (repeatOnset.containsKey(eventType)) {
            repeatAlertMs[eventType] = tMs
        }
    }

    /** rules.py:991-998 `_forget`. 그 종류의 상태가 끝났거나 새로 시작됐다: 미뤄 둔 경고와 되풀이 기록을 지운다(마지막 경고 시각은 남긴다). */
    private fun forget(eventType: EventType) {
        pending.remove(eventType)
        repeatOnset.remove(eventType)
        repeatAlertMs.remove(eventType)
    }
}

/** rules.py:1001-1012 `_reissued`. 시작 이벤트의 사본. durationMs만 지금 시각 기준으로 고친다(미뤄 둔 경고와 되풀이 경고). */
private fun reissued(onset: DriverStateEvent, tMs: Long): DriverStateEvent = DriverStateEvent(
    type = onset.type,
    severity = onset.severity,
    tStartMs = onset.tStartMs,
    durationMs = tMs - onset.tStartMs,
    confidence = onset.confidence,
    source = onset.source,
    phase = onset.phase,
    provisional = onset.provisional,
)
