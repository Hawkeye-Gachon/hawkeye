package com.personal.driverguard.ui.state

import com.personal.driverguard.data.remote.CalendarResponse
import com.personal.driverguard.data.remote.DailyReport
import com.personal.driverguard.data.remote.DayItem
import com.personal.driverguard.data.remote.DriveListItem
import com.personal.driverguard.data.remote.DriveReport
import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.HourItem
import com.personal.driverguard.data.remote.NavMode
import com.personal.driverguard.data.remote.PlaceType
import com.personal.driverguard.data.remote.Peak
import com.personal.driverguard.data.remote.PeakItem
import com.personal.driverguard.data.remote.ReportSummary
import com.personal.driverguard.data.remote.TriggerType
import com.personal.driverguard.data.remote.WeeklyReport
import java.time.DayOfWeek
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.YearMonth
import java.time.temporal.ChronoUnit
import java.time.temporal.TemporalAdjusters
import kotlin.random.Random

/**
 * 시연 모드의 이전 기록([DemoServer] 가 응답에 쓴다). 이번 주와 지난주의 주행을 날짜별로 정해진 값으로 만든다
 * (같은 날짜는 언제 열어도 같은 기록). 오늘 이후·지금 이후 주행은 만들지 않는다.
 * 응답 형식은 백엔드와 같아서 화면 코드는 서버 데이터와 똑같이 그린다.
 */
object FakeHistory {

    private class Ev(val type: String, val at: LocalDateTime)

    private class Drive(val id: Long, val start: LocalDateTime, val end: LocalDateTime, val distance: Int, val events: List<Ev>) {
        val observed: Int get() = ChronoUnit.SECONDS.between(start, end).toInt()
        val inattention: Int get() = events.count { it.type == EventType.INATTENTION }
        val drowsiness: Int get() = events.count { it.type == EventType.DROWSINESS }
    }

    private fun now() = LocalDateTime.now(SEOUL)
    private fun today() = LocalDate.now(SEOUL)
    private fun iso(t: LocalDateTime) = t.atZone(SEOUL).toOffsetDateTime().truncatedTo(ChronoUnit.SECONDS).toString()

    /** 가짜 기록이 있는 기간: 지난주 월요일부터 오늘까지 */
    private fun firstDay() = today().with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY)).minusWeeks(1)

    /** 시연 모드에서 실제로 끝낸 주행(앱이 켜져 있는 동안만 남는다). 예시 기록과 함께 보인다. */
    private class Recorded(val drive: Drive, val report: DriveReport)
    private val recorded = mutableListOf<Recorded>()

    /** 시연 모드 주행 종료 시 DemoServer 가 리포트를 넣는다 */
    @Synchronized
    fun addRecorded(report: DriveReport) {
        val start = parse(report.startedAt)
        val end = report.endedAt?.let(::parse) ?: start
        val events = report.events.map { Ev(it.type, parse(it.detectedAt)) }
        recorded.removeAll { it.drive.id == report.driveId }
        recorded += Recorded(Drive(report.driveId, start, end, report.distanceMeters ?: 0, events), report)
    }

    private fun parse(iso: String): LocalDateTime = java.time.OffsetDateTime.parse(iso).atZoneSameInstant(SEOUL).toLocalDateTime()

    /** 그날의 주행(시작 시각 순). 시연에서 끝낸 주행을 포함한다. */
    @Synchronized
    private fun drivesOn(d: LocalDate): List<Drive> = (generatedOn(d) + recorded.filter { it.drive.start.toLocalDate() == d }.map { it.drive }).sortedBy { it.start }

    /** 예시로 만든 그날의 주행. 기간 밖이면 없음. */
    private fun generatedOn(d: LocalDate): List<Drive> {
        if (d < firstDay() || d > today()) return emptyList()
        val r = Random(d.toEpochDay() * 7919)
        val weekend = d.dayOfWeek == DayOfWeek.SATURDAY || d.dayOfWeek == DayOfWeek.SUNDAY
        // 하루쯤은 비워 '주행 기록 없음'·'관찰 기록 없음'도 보이게
        if (d.dayOfWeek == DayOfWeek.WEDNESDAY && d < today().minusDays(6)) return emptyList()
        val slots = if (weekend) listOf(14) else listOf(8, 18, 13)
        val count = if (weekend) r.nextInt(0, 2) else r.nextInt(1, 3)
        return slots.take(count).mapIndexedNotNull { i, hour ->
            val start = d.atTime(hour, r.nextInt(0, 45))
            val minutes = r.nextInt(18, 56)
            val end = start.plusMinutes(minutes.toLong()).plusSeconds(r.nextLong(0, 60))
            if (end > now()) return@mapIndexedNotNull null
            val inattention = r.nextInt(0, 5)
            val drowsy = if (r.nextInt(0, 4) == 0) 1 else 0
            val events = (List(inattention) { EventType.INATTENTION } + List(drowsy) { EventType.DROWSINESS })
                .map { Ev(it, start.plusSeconds(r.nextLong(60, minutes * 60L - 30))) }
                .sortedBy { it.at }
            Drive(d.toEpochDay() * 10 + i, start, end, (minutes * r.nextDouble(550.0, 900.0)).toInt(), events)
        }
    }

    private fun DriveListItem(dr: Drive) = DriveListItem(dr.id, iso(dr.start), iso(dr.end), dr.observed, dr.distance, dr.inattention, dr.drowsiness)

    /** 가장 최근 주행일(없으면 오늘) */
    private fun latestDriveDate(): LocalDate {
        var d = today()
        while (d >= firstDay()) {
            if (drivesOn(d).isNotEmpty()) return d
            d = d.minusDays(1)
        }
        return today()
    }

    /** 주행들을 0~23시 구간으로 나눈다(관찰 시간은 구간마다, 감지는 실제 시각의 구간에). */
    private fun hourly(drives: List<Drive>): List<HourItem> = (0 until 24).map { h ->
        var obs = 0; var ina = 0; var dro = 0
        drives.forEach { dr ->
            val from = dr.start.toLocalDate().atTime(h, 0)
            val to = from.plusHours(1)
            val overlap = ChronoUnit.SECONDS.between(maxOf(from, dr.start), minOf(to, dr.end))
            if (overlap > 0) obs += overlap.toInt()
            dr.events.filter { it.at.hour == h }.forEach { if (it.type == EventType.INATTENTION) ina++ else dro++ }
        }
        HourItem(h, obs, ina, dro)
    }

    private fun peakOf(hours: List<HourItem>): Peak {
        fun pick(sel: (HourItem) -> Int): PeakItem {
            val max = hours.maxOfOrNull(sel) ?: 0
            return if (max == 0) PeakItem(emptyList(), 0) else PeakItem(hours.filter { sel(it) == max }.map { it.hour }, max)
        }
        return Peak(pick { it.inattentionCount }, pick { it.drowsinessCount })
    }

    private fun summaryOf(drives: List<Drive>) = ReportSummary(
        drives.size, drives.sumOf { it.observed }, drives.sumOf { it.distance }, drives.sumOf { it.inattention }, drives.sumOf { it.drowsiness },
    )

    fun daily(date: LocalDate?): DailyReport {
        val d = date ?: latestDriveDate()
        val drives = drivesOn(d)
        val hours = hourly(drives)
        return DailyReport(d.toString(), summaryOf(drives), hours, peakOf(hours), drives.sortedByDescending { it.start }.map(::DriveListItem))
    }

    fun weekly(weekStart: LocalDate?): WeeklyReport {
        val start = weekStart ?: latestDriveDate().with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY))
        val days = (0L until 7L).map { start.plusDays(it) }
        val all = days.flatMap(::drivesOn)
        val hours = hourly(all)
        return WeeklyReport(
            weekStart = start.toString(),
            weekEnd = start.plusDays(6).toString(),
            summary = summaryOf(all),
            daily = days.map { d ->
                val ds = drivesOn(d)
                DayItem(d.toString(), d.dayOfWeek.name, ds.size, ds.sumOf { it.observed }, ds.sumOf { it.inattention }, ds.sumOf { it.drowsiness })
            },
            hourly = hours,
            peak = peakOf(hours),
        )
    }

    fun calendar(month: YearMonth): CalendarResponse =
        CalendarResponse(month.toString(), (1..month.lengthOfMonth()).map { month.atDay(it) }.filter { drivesOn(it).isNotEmpty() }.map { it.toString() })

    /** 가장 최근 주행(홈의 '이전 기록') */
    fun latestDrive(): DriveListItem? = drivesOn(latestDriveDate()).maxByOrNull { it.start }?.let(::DriveListItem)

    /** 최근 주행 목록(최신순) */
    fun recent(size: Int): List<DriveListItem> {
        val out = mutableListOf<DriveListItem>()
        var d = today()
        while (d >= firstDay() && out.size < size) {
            out += drivesOn(d).sortedByDescending { it.start }.map(::DriveListItem)
            d = d.minusDays(1)
        }
        return out.take(size)
    }

    /**
     * 주행 리포트. 경로는 [route](경부고속도로 서울→부산)를 처음부터 주행 거리만큼 따라간 것으로 하고,
     * 감지는 시각 비율대로 그 위에 놓는다. 졸음이 있으면 그 뒤 첫 쉼터를 경유지로 추가했고 들러 쉰 것으로 한다.
     */
    fun report(driveId: Long, route: DemoRoute): DriveReport? {
        synchronized(this) { recorded.firstOrNull { it.drive.id == driveId }?.let { return it.report } }
        val dr = generatedOn(LocalDate.ofEpochDay(driveId / 10)).firstOrNull { it.id == driveId } ?: return null
        val length = dr.distance.toDouble().coerceAtMost(route.lengthMeters)
        fun metersAt(t: LocalDateTime) = (ChronoUnit.SECONDS.between(dr.start, t).toDouble() / dr.observed).coerceIn(0.0, 1.0) * length
        val steps = (length / ROUTE_STEP_M).toInt().coerceIn(2, 600)
        val points = (0..steps).map { route.positionAt(length * it / steps) }.map { listOf(round6(it.lat), round6(it.lng)) }
        val waypoints = dr.events.firstOrNull { it.type == EventType.DROWSINESS }?.let { e ->
            val rest = route.nextRest(metersAt(e.at)) ?: return@let null
            val added = e.at.plusSeconds(40)
            val reach = added.plusSeconds(((rest.routeMeters - metersAt(e.at)) / DriveSimulator.DEMO_SPEED_MPS).toLong()).takeIf { it < dr.end }
            listOf(DriveReport.WaypointItem(1, rest.placeId, rest.name, rest.type, NavMode.WAYPOINT, TriggerType.DROWSINESS, iso(added), visited = reach != null, visitedAt = reach?.let(::iso)))
        }.orEmpty()
        return DriveReport(
            driveId = dr.id, status = "ENDED", startedAt = iso(dr.start), endedAt = iso(dr.end),
            observedSeconds = dr.observed, distanceMeters = dr.distance, inattentionCount = dr.inattention, drowsinessCount = dr.drowsiness,
            route = DriveReport.Route(DriveReport.Point(points.first()[0], points.first()[1]), DriveReport.Point(points.last()[0], points.last()[1]), points),
            events = dr.events.map { e -> route.positionAt(metersAt(e.at)).let { p -> DriveReport.EventItem(e.type, iso(e.at), round6(p.lat), round6(p.lng)) } },
            waypoints = waypoints,
        )
    }

    private const val ROUTE_STEP_M = 150.0
    private fun round6(v: Double) = Math.round(v * 1e6) / 1e6
}
