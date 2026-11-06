package com.personal.driverguard.domain.report

import com.personal.driverguard.entity.DriverEventVo
import com.personal.driverguard.entity.EventType
import com.personal.driverguard.entity.EventTypeSummary

/** 이벤트를 유형별 횟수·지속 시간으로 묶는다. 이벤트가 없는 유형은 빠진다. */
fun summarizeEvents(events: List<DriverEventVo>): List<EventTypeSummary> =
    events.groupBy { it.type }
        .map { (type, list) ->
            EventTypeSummary(
                type = type,
                count = list.size,
                totalDurationMs = list.sumOf { it.durationMs ?: 0L },
                unknownDurationCount = list.count { it.durationMs == null },
            )
        }
        .sortedBy { EventType.entries.indexOf(it.type) }
