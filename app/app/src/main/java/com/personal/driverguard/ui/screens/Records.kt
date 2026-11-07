package com.personal.driverguard.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.personal.driverguard.R
import com.personal.driverguard.data.remote.DriveListItem
import com.personal.driverguard.data.remote.DriveReport
import com.personal.driverguard.data.remote.EventType
import com.personal.driverguard.data.remote.HourItem
import com.personal.driverguard.data.remote.WeeklyReport
import com.personal.driverguard.ui.AppState
import com.personal.driverguard.ui.LocalHawkeye
import com.personal.driverguard.ui.Navigator
import com.personal.driverguard.ui.Screen
import com.personal.driverguard.ui.design.*
import com.personal.driverguard.ui.state.Fmt
import com.personal.driverguard.ui.state.SEOUL
import com.personal.driverguard.ui.theme.NColor
import java.time.DayOfWeek
import java.time.LocalDate
import java.time.YearMonth

// ------------------------------------------------------------------ report
@Composable
fun ReportScreen(nav: Navigator, state: AppState, fromHistory: Boolean) {
    if (state.preview) PreviewReport(nav, fromHistory) else LiveReport(nav, fromHistory)
}

@Composable
private fun ReportBack(nav: Navigator, fromHistory: Boolean) =
    BackBtn(if (fromHistory) "기록 목록으로" else "홈으로") { if (fromHistory) nav.back() else nav.go(Screen.Home) }

/** 백엔드 리포트(종료 업로드 응답 또는 주행 상세) */
@Composable
private fun LiveReport(nav: Navigator, fromHistory: Boolean) {
    val ui by LocalHawkeye.current.report.collectAsStateWithLifecycle()
    val r = ui.report
    val start = Fmt.parse(r?.startedAt)
    val end = Fmt.parse(r?.endedAt)
    Box(Modifier.fillMaxSize()) {
        SceneCard(Modifier.background(NColor.mapBg)) {
            if (r != null) {
                LiveRouteMap(
                    points = r.route?.points.orEmpty(),
                    pins = r.events.filter { it.latitude != null && it.longitude != null }.map {
                        RoutePin(it.latitude!!, it.longitude!!, eventColor(it.type), Fmt.time(it.detectedAt))
                    },
                    startLabel = "출발 ${Fmt.time(start)}",
                    endLabel = "도착 ${Fmt.time(end)}",
                )
            }
            Box(Modifier.padding(start = 32.dp, top = 28.dp).width(300.dp).clip(RoundedCornerShape(16.dp)).background(Color.White.copy(alpha = .86f)).padding(horizontal = 16.dp, vertical = 12.dp)) {
                T("핀은 도로의 위험도가 아니라 이번 주행에서 운전자 상태 변화가 감지된 위치입니다", S.t(15, 21, color = NColor.ink3))
            }
            Row(
                Modifier.align(Alignment.BottomStart).padding(start = 32.dp, bottom = 28.dp).clip(RoundedCornerShape(16.dp)).background(Color.White.copy(alpha = .86f)).padding(horizontal = 18.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(20.dp),
            ) { listOf(NColor.alert to "부주의", NColor.danger to "졸음").forEach { (c, l) -> LegendDot(c, l) } }
        }
        Sheet {
            SheetBody(top = 28.dp, bottomPad = 36.dp, content = {
                ReportBack(nav, fromHistory)
                when {
                    r == null && ui.loading -> T("리포트를 불러오는 중입니다", S.body, Modifier.padding(top = 12.dp))
                    r == null -> T(ui.error ?: "리포트를 불러오지 못했습니다", S.t(17, 24, color = NColor.dangerInk), Modifier.padding(top = 12.dp))
                    else -> {
                        Column(Modifier.padding(top = 6.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            T(start?.let { Fmt.dateTitle(it.toLocalDate()) } ?: "-", S.h2.copy(fontSize = 32.sp))
                            T("${Fmt.time(start)} – ${Fmt.time(end)}", S.body.copy(color = NColor.ink3, fontFeatureSettings = "tnum"))
                        }
                        Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
                            Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) { Metric(Fmt.duration(r.observedSeconds), "총 관찰 시간", Modifier.weight(1f)); Metric(Fmt.distance(r.distanceMeters), "이동 거리", Modifier.weight(1f)) }
                            Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) { Metric("${r.inattentionCount ?: 0}회", "부주의", Modifier.weight(1f)); Metric("${r.drowsinessCount ?: 0}회", "졸음 감지", Modifier.weight(1f)) }
                        }
                        ReportTimeline(r)
                    }
                }
            })
        }
    }
}

private fun eventColor(type: String) = if (type == EventType.DROWSINESS) NColor.danger else NColor.alert
private fun eventLabel(type: String) = if (type == EventType.DROWSINESS) "졸음" else "부주의"

/** 감지·경유지 추가를 시각순으로. 칸이 부족하면 최근 것부터 줄인다. */
@Composable
private fun ReportTimeline(r: DriveReport) {
    data class Row3(val at: String, val color: Color, val text: String, val sub: String)
    val rows = (r.events.map { Row3(it.detectedAt, eventColor(it.type), eventLabel(it.type), "") } +
        r.waypoints.map { Row3(it.addedAt, NColor.indigo, "${it.placeName ?: "쉼터"} ${if (it.mode == "DESTINATION") "길 안내" else "경유지 추가"}", if (it.visited) "방문" else "") })
        .sortedBy { it.at }
    Column(Modifier.padding(top = 10.dp)) {
        HLine()
        if (rows.isEmpty()) T("이번 주행에서 감지된 기록이 없습니다", S.t(17, 24, color = NColor.ink3), Modifier.padding(vertical = 14.dp))
        val shown = rows.take(MAX_TIMELINE_ROWS)
        shown.forEachIndexed { i, row ->
            Row(
                Modifier.fillMaxWidth().drawBehind { if (i < shown.lastIndex) drawRect(NColor.line, Offset(0f, size.height - 1.dp.toPx()), Size(size.width, 1.dp.toPx())) }.padding(vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                T(Fmt.time(row.at), S.t(16, 22, color = NColor.ink3).copy(fontFeatureSettings = "tnum"), Modifier.width(92.dp))
                Dot(row.color); WGap(14.dp)
                T(buildAnnotatedString { append(row.text); if (row.sub.isNotEmpty()) withStyle(SpanStyle(color = NColor.ink3)) { append("  " + row.sub) } }, S.t(17, 24))
            }
        }
        if (rows.size > shown.size) T("외 ${rows.size - shown.size}건", S.caption, Modifier.padding(top = 8.dp))
    }
}

private const val MAX_TIMELINE_ROWS = 5

/** 시안 확인 모드의 예시 리포트 */
@Composable
private fun PreviewReport(nav: Navigator, fromHistory: Boolean) {
    Box(Modifier.fillMaxSize()) {
        SceneCard(Modifier.background(NColor.mapBg)) {
            RouteMap()
            Box(Modifier.padding(start = 32.dp, top = 28.dp).width(300.dp).clip(RoundedCornerShape(16.dp)).background(Color.White.copy(alpha = .86f)).padding(horizontal = 16.dp, vertical = 12.dp)) {
                T("핀은 도로의 위험도가 아니라 이번 주행에서 운전자 상태 변화가 감지된 위치입니다", S.t(15, 21, color = NColor.ink3))
            }
            Row(
                Modifier.align(Alignment.BottomStart).padding(start = 32.dp, bottom = 28.dp).clip(RoundedCornerShape(16.dp)).background(Color.White.copy(alpha = .86f)).padding(horizontal = 18.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(20.dp),
            ) { listOf(NColor.alert to "부주의", NColor.danger to "졸음", NColor.indigo to "경유지 추가").forEach { (c, l) -> LegendDot(c, l) } }
        }
        Sheet {
            SheetBody(top = 28.dp, bottomPad = 36.dp, content = {
                ReportBack(nav, fromHistory)
                Column(Modifier.padding(top = 6.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    T("11월 6일 금요일", S.h2.copy(fontSize = 32.sp))
                    T("오후 2:30 – 오후 3:12", S.body.copy(color = NColor.ink3, fontFeatureSettings = "tnum"))
                }
                Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) { Metric("42분 18초", "총 관찰 시간", Modifier.weight(1f)); Metric("28.4km", "이동 거리", Modifier.weight(1f)) }
                    Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) { Metric("3회", "부주의", Modifier.weight(1f)); Metric("1회", "졸음 감지", Modifier.weight(1f)) }
                }
                Column(Modifier.padding(top = 10.dp)) {
                    HLine()
                    listOf(
                        Triple("오후 2:37", NColor.alert, "부주의" to "2.1초"), Triple("오후 2:44", NColor.alert, "부주의" to "3.4초"),
                        Triple("오후 2:52", NColor.alert, "부주의" to "1.8초"), Triple("오후 2:55", NColor.danger, "졸음" to "4.2초"),
                        Triple("오후 2:56", NColor.indigo, "천안 졸음쉼터 경유지 추가" to ""),
                    ).forEachIndexed { i, (t, c, ev) ->
                        Row(
                            Modifier.fillMaxWidth().drawBehind { if (i < 4) drawRect(NColor.line, Offset(0f, size.height - 1.dp.toPx()), Size(size.width, 1.dp.toPx())) }.padding(vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            T(t, S.t(16, 22, color = NColor.ink3).copy(fontFeatureSettings = "tnum"), Modifier.width(92.dp))
                            Dot(c); WGap(14.dp)
                            T(buildAnnotatedString { append(ev.first); if (ev.second.isNotEmpty()) withStyle(SpanStyle(color = NColor.ink3)) { append("  " + ev.second) } }, S.t(17, 24))
                        }
                    }
                }
            })
        }
    }
}

@Composable
private fun Metric(v: String, l: String, modifier: Modifier = Modifier) = Column(modifier, verticalArrangement = Arrangement.spacedBy(2.dp)) { T(v, S.metric); T(l, S.label) }

@Composable
private fun LegendDot(c: Color, l: String) = Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) { Dot(c); T(l, S.caption.copy(fontSize = 15.sp)) }

// ------------------------------------------------------------------ history shared parts
@Composable
private fun HistoryHead(nav: Navigator, week: Boolean) = Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
    BackBtn("홈으로") { nav.go(Screen.Home) }
    Fill()
    Segmented(listOf("일간 리포트", "주간 통계"), if (week) 1 else 0) { i ->
        if (i == 0 && week) nav.replace(Screen.HistoryDay)
        if (i == 1 && !week) nav.replace(Screen.HistoryWeek)
    }
}

@Composable
private fun DateNav(label: String, prevEnabled: Boolean = true, nextEnabled: Boolean = false, calendar: Boolean? = null, calOpen: Boolean = false, onPrev: () -> Unit = {}, onNext: () -> Unit = {}, onCal: () -> Unit = {}) =
    Row(Modifier.offset(x = (-12).dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        IconBtn(R.drawable.ic_left, prevEnabled, onClick = onPrev)
        T(label, S.t(if (label.contains("~")) 24 else 26, 32, 600).copy(fontFeatureSettings = "tnum"), Modifier.padding(horizontal = 10.dp))
        IconBtn(R.drawable.ic_right, nextEnabled, onClick = onNext)
        if (calendar != null) { WGap(6.dp); IconBtn(R.drawable.ic_cal, true, dark = calOpen, onClick = onCal) }
    }

@Composable
private fun IconBtn(icon: Int, enabled: Boolean, dark: Boolean = false, onClick: () -> Unit) = Box(
    Modifier.size(52.dp).clip(RoundedCornerShape(16.dp)).background(if (dark) NColor.ink else NColor.fill).clickable(enabled = enabled, onClick = onClick),
    contentAlignment = Alignment.Center,
) { Ic(icon, if (dark) Color.White else NColor.ink.copy(alpha = if (enabled) 1f else .4f)) }

@Composable
private fun SumRow(items: List<Pair<String, String>>) = Row(horizontalArrangement = Arrangement.spacedBy(34.dp)) {
    items.forEach { (v, l) -> Column { T(v, S.t(26, 34, 400, tracking = -.02).copy(fontFeatureSettings = "tnum")); T(l, S.caption) } }
}

@Composable
private fun Legend(items: List<String>) = Row(horizontalArrangement = Arrangement.spacedBy(22.dp), verticalAlignment = Alignment.CenterVertically) {
    items.forEach { k ->
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            when (k) {
                "peak", "peakw" -> Box(Modifier.size(12.dp, 14.dp).clip(RoundedCornerShape(3.dp)).background(NColor.indigo))
                "bar" -> Box(Modifier.size(12.dp, 14.dp).clip(RoundedCornerShape(3.dp)).background(Color(0x33131417)))
                "zero" -> Box(Modifier.size(20.dp, 4.dp).clip(RoundedCornerShape(2.dp)).background(NColor.line2))
                else -> Box(Modifier.size(20.dp, 2.dp).drawBehind { drawLine(NColor.ink4, Offset(0f, 1f), Offset(size.width, 1f), 2.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(2.dp.toPx(), 4.dp.toPx()))) })
            }
            T(when (k) { "peak" -> "가장 많은 시간대"; "peakw" -> "가장 많은 요일·시간대"; "bar" -> "감지 횟수"; "zero" -> "관찰했지만 0회"; else -> "관찰 기록 없음" }, S.caption)
        }
    }
}

private fun peakText(prefix: String, values: List<Int?>, metric: Int) = buildAnnotatedString {
    val pk = peakOf(values).minOrNull()
    val label = if (metric == 0) "부주의" else "졸음"; val josa = if (metric == 0) "가" else "이"
    if (pk == null) append("이 기간에는 $label$josa 감지되지 않았습니다")
    else { append("$prefix${slotName(pk)}에 $label$josa 가장 많이 감지됐습니다 "); withStyle(SpanStyle(color = NColor.indigo)) { append("· ${values[pk]}회") } }
}

/** 시간대 값: 관찰 기록이 없으면 null(점선), 관찰했지만 0회면 0 */
private fun hourValues(hourly: List<HourItem>, metric: Int): List<Int?> =
    (0 until 24).map { h -> hourly.firstOrNull { it.hour == h }?.let { if (it.observedSeconds > 0) (if (metric == 0) it.inattentionCount else it.drowsinessCount) else null } }

private fun hourTip(hourly: List<HourItem>, i: Int, metric: Int, weekly: Boolean): String {
    val item = hourly.firstOrNull { it.hour == i }
    if (item == null || item.observedSeconds == 0) return "${slotName(i)} · 관찰 기록 없음"
    val count = if (metric == 0) item.inattentionCount else item.drowsinessCount
    return "${slotName(i)} · ${if (metric == 0) "부주의" else "졸음"} ${count}회 · ${if (weekly) "주간 누적 관찰" else "관찰"} ${Fmt.minutes(item.observedSeconds)}"
}

@Composable
private fun LoadState(loading: Boolean, error: String?) {
    when {
        loading -> T("불러오는 중입니다", S.t(15, 20, color = NColor.ink3))
        error != null -> T(error, S.t(15, 20, color = NColor.dangerInk))
    }
}

// ------------------------------------------------------------------ daily
@Composable
fun HistoryDayScreen(nav: Navigator, state: AppState, calendar: Boolean) {
    if (state.preview) PreviewHistoryDay(nav, calendar) else LiveHistoryDay(nav, calendar)
}

@Composable
private fun LiveHistoryDay(nav: Navigator, calendar: Boolean) {
    val vm = LocalHawkeye.current
    val ui by vm.daily.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { if (ui.data == null) vm.loadDaily(null) }
    val d = ui.data
    val date = d?.date?.let(LocalDate::parse)
    val today = LocalDate.now(SEOUL)
    var metric by remember { mutableIntStateOf(0) }
    val values = d?.let { hourValues(it.hourly, metric) } ?: List(24) { null }
    var tip by remember(metric, d) { mutableStateOf(peakOf(values).minOrNull()) }
    Box(Modifier.fillMaxSize()) {
        FullCard {
            Column(Modifier.fillMaxSize().padding(start = 40.dp, end = 40.dp, top = 28.dp, bottom = 28.dp)) {
                HistoryHead(nav, week = false)
                Gap(18.dp)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    DateNav(
                        date?.let(Fmt::dateTitle) ?: "-", prevEnabled = date != null, nextEnabled = date != null && date < today, calendar = true, calOpen = calendar,
                        onPrev = { date?.let { vm.loadDaily(it.minusDays(1)) } },
                        onNext = { date?.let { vm.loadDaily(it.plusDays(1)) } },
                        onCal = {
                            if (!calendar) vm.loadCalendar(YearMonth.from(date ?: today))
                            nav.replace(if (calendar) Screen.HistoryDay else Screen.HistoryCal)
                        },
                    )
                    Fill()
                    d?.summary?.let { s -> SumRow(listOf("${s.driveCount}회" to "주행", Fmt.duration(s.observedSeconds) to "총 관찰 시간", Fmt.distance(s.distanceMeters) to "총 이동 거리", "${s.inattentionCount}회" to "부주의", "${s.drowsinessCount}회" to "졸음 감지")) }
                }
                Gap(18.dp); HLine(); Gap(18.dp)
                if (d != null && d.summary.driveCount == 0) {
                    NoDriveDay { vm.loadDaily(null) }
                } else {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        if (d != null) T(peakText("", values, metric), S.peak) else LoadState(ui.loading, ui.error)
                        Fill()
                        if (!calendar) { Legend(listOf("peak", "zero", "none")); WGap(24.dp) }
                        Segmented(listOf("부주의", "졸음"), metric, small = true) { metric = it }
                    }
                    Gap(2.dp)
                    val ti = tip
                    BarChart(values, Sample.hourLabel, 1152.dp, 250.dp, peak = peakOf(values), gap = 10.dp, tipIndex = ti,
                        tip = ti?.let { i -> d?.let { hourTip(it.hourly, i, metric, weekly = false) } }, onBar = { tip = it })
                    if (!calendar) {
                        Gap(14.dp)
                        TripRow(listOf("주행별 기록, 최신순", "관찰 시간", "거리", "부주의", "졸음"), header = true)
                        d?.drives.orEmpty().take(MAX_TRIP_ROWS).forEach { trip ->
                            TripRow(tripCells(trip)) { vm.openReport(trip.driveId); nav.go(Screen.ReportFromHistory) }
                        }
                    }
                }
            }
            if (calendar) LiveCalendarPopover(Modifier.at(28.dp, 150.dp, 400.dp, 470.dp), selected = date, onPick = { picked ->
                vm.loadDaily(picked); nav.replace(Screen.HistoryDay)
            })
        }
    }
}

private const val MAX_TRIP_ROWS = 3

private fun tripCells(t: DriveListItem): List<String> {
    val s = Fmt.parse(t.startedAt); val e = Fmt.parse(t.endedAt)
    return listOf("${Fmt.time(s)} – ${e?.let { Fmt.time(it).substringAfter(' ') } ?: ""}", Fmt.duration(t.observedSeconds), Fmt.distance(t.distanceMeters), "${t.inattentionCount ?: 0}회", "${t.drowsinessCount ?: 0}회")
}

/** 고른 날에 주행 기록이 없을 때 */
@Composable
private fun NoDriveDay(onLatest: () -> Unit) {
    Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically)) {
        Ic(R.drawable.ic_cal, NColor.ink4, 44.dp)
        T("해당 기간에 주행 기록이 없습니다", S.h2)
        T("관찰을 시작하고 주행하면 이날의 기록이 쌓입니다.", S.body)
        Gap(10.dp)
        Btn("가장 최근 주행일 보기", kind = Kind.Ghost, height = 52.dp, radius = 16.dp, style = S.t(18, 24, 600), horizontal = 22.dp, onClick = onLatest)
    }
}

/** 실제 달력: 주행 기록이 있는 날에 점 */
@Composable
private fun LiveCalendarPopover(modifier: Modifier, selected: LocalDate?, onPick: (LocalDate) -> Unit) {
    val vm = LocalHawkeye.current
    val cal by vm.calendar.collectAsStateWithLifecycle()
    val month = cal.month
    val today = LocalDate.now(SEOUL)
    val dates = cal.data?.driveDates.orEmpty().mapNotNull { runCatching { LocalDate.parse(it) }.getOrNull() }.toSet()
    Column(modifier.shadow(30.dp, RoundedCornerShape(28.dp)).clip(RoundedCornerShape(28.dp)).background(Color.White).padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = 20.dp)) {
        Row(Modifier.width(378.dp), verticalAlignment = Alignment.CenterVertically) {
            IconBtn(R.drawable.ic_left, true) { vm.loadCalendar(month.minusMonths(1)) }; Fill()
            T("${month.year}년 ${month.monthValue}월", S.h3); Fill()
            IconBtn(R.drawable.ic_right, month < YearMonth.from(today)) { vm.loadCalendar(month.plusMonths(1)) }
        }
        Gap(14.dp)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) { Sample.days.forEach { T(it, S.caption, Modifier.width(48.dp).height(28.dp), align = TextAlign.Center) } }
        val lead = month.atDay(1).dayOfWeek.value - DayOfWeek.MONDAY.value
        val cells = List(lead) { 0 } + (1..month.lengthOfMonth()).toList()
        cells.chunked(7).forEach { week ->
            Row(Modifier.padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                week.forEach { day ->
                    val date = if (day > 0) month.atDay(day) else null
                    val has = date in dates; val sel = date != null && date == selected; val fut = date != null && date > today
                    Box(
                        Modifier.size(48.dp, 52.dp).clip(CircleShape).background(if (sel) NColor.indigo else Color.Transparent)
                            .then(if (date != null && !fut) Modifier.clickable { onPick(date) } else Modifier),
                        contentAlignment = Alignment.Center,
                    ) {
                        if (date != null) {
                            T("$day", S.t(18, 24, color = when { sel -> Color.White; fut -> NColor.ink4; else -> NColor.ink }).copy(fontFeatureSettings = "tnum"))
                            if (has) Box(Modifier.align(Alignment.BottomCenter).padding(bottom = 6.dp).size(5.dp).clip(CircleShape).background(if (sel) Color.White else NColor.indigo))
                        }
                    }
                }
            }
        }
        Gap(12.dp)
        T(if (cal.loading) "불러오는 중입니다" else "점이 있는 날에 주행 기록이 있습니다", S.t(15, 20, color = NColor.ink3))
    }
}

@Composable
private fun PreviewHistoryDay(nav: Navigator, calendar: Boolean) {
    var metric by remember { mutableIntStateOf(0) }
    val values = if (metric == 0) Sample.dayIn else Sample.dayDr
    var tip by remember(metric) { mutableStateOf(peakOf(values).minOrNull()) }
    Box(Modifier.fillMaxSize()) {
        FullCard {
            Column(Modifier.fillMaxSize().padding(start = 40.dp, end = 40.dp, top = 28.dp, bottom = 28.dp)) {
                HistoryHead(nav, week = false)
                Gap(18.dp)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    DateNav("11월 6일 금요일", calendar = true, calOpen = calendar, onPrev = { nav.replace(Screen.HistoryDayNone) }, onCal = { nav.replace(if (calendar) Screen.HistoryDay else Screen.HistoryCal) })
                    Fill()
                    SumRow(listOf("2회" to "주행", "1시간 17분" to "총 관찰 시간", "47.3km" to "총 이동 거리", "4회" to "부주의", "1회" to "졸음 감지"))
                }
                Gap(18.dp); HLine(); Gap(18.dp)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    T(peakText("", values, metric), S.peak); Fill()
                    if (!calendar) { Legend(listOf("peak", "zero", "none")); WGap(24.dp) }
                    Segmented(listOf("부주의", "졸음"), metric, small = true) { metric = it }
                }
                Gap(2.dp)
                val ti = tip
                BarChart(values, Sample.hourLabel, 1152.dp, 250.dp, peak = peakOf(values), gap = 10.dp, tipIndex = ti,
                    tip = ti?.let { i -> values[i]?.let { "${slotName(i)} · ${if (metric == 0) "부주의" else "졸음"} ${it}회 · 관찰 ${mins(Sample.dayObs[i] ?: 0)}" } ?: "${slotName(i)} · 관찰 기록 없음" },
                    onBar = { tip = it })
                if (!calendar) {
                    Gap(14.dp)
                    TripRow(listOf("주행별 기록, 최신순", "관찰 시간", "거리", "부주의", "졸음"), header = true)
                    TripRow(listOf("오후 2:30 – 3:12", "42분 18초", "28.4km", "3회", "1회")) { nav.go(Screen.ReportFromHistory) }
                    TripRow(listOf("오전 8:12 – 8:47", "35분 0초", "18.9km", "1회", "0회")) { nav.go(Screen.ReportFromHistory) }
                }
            }
            if (calendar) CalendarPopover(Modifier.at(28.dp, 150.dp, 400.dp, 470.dp), onPick = { has -> nav.replace(if (has) Screen.HistoryDay else Screen.HistoryDayNone) })
        }
    }
}

private val tripCols: List<Dp> = listOf(210.dp, 150.dp, 110.dp, 110.dp, 110.dp)

@Composable
private fun TripRow(cells: List<String>, header: Boolean = false, onClick: (() -> Unit)? = null) {
    Row(
        Modifier.fillMaxWidth().height(if (header) 36.dp else 62.dp)
            .drawBehind { drawRect(NColor.line, Offset(0f, size.height - 1.dp.toPx()), Size(size.width, 1.dp.toPx())) }
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        cells.forEachIndexed { i, c ->
            val st = when { header -> S.caption; i == 0 -> S.t(19, 26, 600); else -> S.t(18, 24, color = NColor.ink2) }.copy(fontFeatureSettings = "tnum")
            T(c, st, Modifier.width(tripCols[i]))
        }
        Fill()
        if (!header) Ic(R.drawable.ic_right, NColor.ink4)
    }
}

@Composable
private fun CalendarPopover(modifier: Modifier, onPick: (Boolean) -> Unit) {
    Column(modifier.shadow(30.dp, RoundedCornerShape(28.dp)).clip(RoundedCornerShape(28.dp)).background(Color.White).padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = 20.dp)) {
        Row(Modifier.width(378.dp), verticalAlignment = Alignment.CenterVertically) {
            IconBtn(R.drawable.ic_left, true) {}; Fill(); T("2026년 11월", S.h3); Fill(); IconBtn(R.drawable.ic_right, false) {}
        }
        Gap(14.dp)
        val days = Sample.days
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) { days.forEach { T(it, S.caption, Modifier.width(48.dp).height(28.dp), align = TextAlign.Center) } }
        val cells = List(6) { 0 } + (1..30).toList()
        cells.chunked(7).forEach { week ->
            Row(Modifier.padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                week.forEach { d ->
                    val has = d in 2..6; val sel = d == 6; val fut = d > 6
                    Box(
                        Modifier.size(48.dp, 52.dp).clip(CircleShape).background(if (sel) NColor.indigo else Color.Transparent)
                            .then(if (d in 1..6) Modifier.clickable { onPick(has) } else Modifier),
                        contentAlignment = Alignment.Center,
                    ) {
                        if (d > 0) {
                            T("$d", S.t(18, 24, color = when { sel -> Color.White; fut -> NColor.ink4; else -> NColor.ink }).copy(fontFeatureSettings = "tnum"))
                            if (has) Box(Modifier.align(Alignment.BottomCenter).padding(bottom = 6.dp).size(5.dp).clip(CircleShape).background(if (sel) Color.White else NColor.indigo))
                        }
                    }
                }
            }
        }
        Gap(12.dp)
        T("점이 있는 날에 주행 기록이 있습니다", S.t(15, 20, color = NColor.ink3))
    }
}

@Composable
fun HistoryDayNoneScreen(nav: Navigator, state: AppState) {
    val vm = LocalHawkeye.current
    Box(Modifier.fillMaxSize()) {
        FullCard {
            Column(Modifier.fillMaxSize().padding(start = 40.dp, end = 40.dp, top = 28.dp, bottom = 28.dp)) {
                HistoryHead(nav, week = false)
                Gap(18.dp)
                DateNav("11월 1일 일요일", nextEnabled = true, calendar = true, onNext = { nav.replace(Screen.HistoryDay) }, onCal = { nav.replace(Screen.HistoryCal) })
                Gap(18.dp); HLine()
                Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically)) {
                    Ic(R.drawable.ic_cal, NColor.ink4, 44.dp)
                    T("해당 기간에 주행 기록이 없습니다", S.h2)
                    T("관찰을 시작하고 주행하면 이날의 기록이 쌓입니다.", S.body)
                    Gap(10.dp)
                    Btn("가장 최근 주행일 11월 6일 보기", kind = Kind.Ghost, height = 52.dp, radius = 16.dp, style = S.t(18, 24, 600), horizontal = 22.dp) {
                        if (!state.preview) vm.loadDaily(null)
                        nav.replace(Screen.HistoryDay)
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ weekly
@Composable
fun HistoryWeekScreen(nav: Navigator, state: AppState, empty: Boolean) {
    if (state.preview) PreviewHistoryWeek(nav, empty) else LiveHistoryWeek(nav)
}

@Composable
private fun LiveHistoryWeek(nav: Navigator) {
    val vm = LocalHawkeye.current
    val ui by vm.weekly.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { if (ui.data == null) vm.loadWeekly(null) }
    val w: WeeklyReport? = ui.data
    val start = w?.weekStart?.let(LocalDate::parse)
    val end = w?.weekEnd?.let(LocalDate::parse)
    val today = LocalDate.now(SEOUL)
    var metric by remember { mutableIntStateOf(0) }
    val day: List<Int?> = w?.daily?.map { if (it.observedSeconds > 0 || it.driveCount > 0) (if (metric == 0) it.inattentionCount else it.drowsinessCount) else null } ?: List(7) { null }
    val hour = w?.let { hourValues(it.hourly, metric) } ?: List(24) { null }
    var tip by remember(metric, w) { mutableStateOf(peakOf(hour).minOrNull()) }
    Box(Modifier.fillMaxSize()) {
        FullCard {
            Column(Modifier.fillMaxSize().padding(start = 40.dp, end = 40.dp, top = 28.dp, bottom = 28.dp)) {
                HistoryHead(nav, week = true)
                Gap(18.dp)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    DateNav(
                        if (start != null && end != null) "${Fmt.dateShort(start)} ~ ${Fmt.dateShort(end)}" else "-",
                        prevEnabled = start != null, nextEnabled = end != null && end < today,
                        onPrev = { start?.let { vm.loadWeekly(it.minusWeeks(1)) } },
                        onNext = { start?.let { vm.loadWeekly(it.plusWeeks(1)) } },
                    )
                    Fill()
                    w?.summary?.let { s -> SumRow(listOf("${s.driveCount}회" to "주행", Fmt.duration(s.observedSeconds) to "총 관찰 시간", Fmt.distance(s.distanceMeters) to "총 이동 거리", "${s.inattentionCount}회" to "부주의", "${s.drowsinessCount}회" to "졸음 감지")) }
                }
                Gap(18.dp); HLine(); Gap(18.dp)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    if (w != null) T(peakText("이번 주에는 ", hour, metric), S.peak) else LoadState(ui.loading, ui.error)
                    Fill()
                    Segmented(listOf("부주의", "졸음"), metric, small = true) { metric = it }
                }
                Gap(8.dp)
                Row(horizontalArrangement = Arrangement.spacedBy(48.dp)) {
                    Column {
                        T("요일별, 요일을 누르면 그날의 일간 리포트로 이동합니다", S.t(15, 20, color = NColor.ink3))
                        BarChart(day, { Sample.days[it] }, 360.dp, 410.dp, peak = peakOf(day), onBar = { i ->
                            val date = start?.plusDays(i.toLong())
                            if (date != null && day[i] != null) { vm.loadDaily(date); nav.replace(Screen.HistoryDay) }
                        })
                    }
                    Column {
                        T("시간대별, 한 주 동안 같은 시간대의 기록을 합산합니다", S.t(15, 20, color = NColor.ink3))
                        val ti = tip
                        BarChart(hour, Sample.hourLabel, 744.dp, 410.dp, peak = peakOf(hour), tipIndex = ti,
                            tip = ti?.let { i -> w?.let { hourTip(it.hourly, i, metric, weekly = true) } }, onBar = { tip = it })
                    }
                }
                Gap(14.dp)
                Legend(listOf("peakw", "bar", "zero", "none"))
            }
        }
    }
}

@Composable
private fun PreviewHistoryWeek(nav: Navigator, empty: Boolean) {
    var metric by remember { mutableIntStateOf(if (empty) 1 else 0) }
    val N: Int? = null
    val day = if (empty) listOf(0, 0, 0, 0, 0, N, 0) else if (metric == 0) Sample.weekIn else Sample.weekDr
    val hour = if (empty) listOf(N, N, N, N, N, N, N, N, 0, 0, N, N, N, N, N, N, N, 0, 0, 0, N, N, N, N) else if (metric == 0) Sample.hourIn else Sample.hourDr
    var tip by remember(metric) { mutableStateOf(if (empty) null else peakOf(hour).minOrNull()) }
    Box(Modifier.fillMaxSize()) {
        FullCard {
            Column(Modifier.fillMaxSize().padding(start = 40.dp, end = 40.dp, top = 28.dp, bottom = 28.dp)) {
                HistoryHead(nav, week = true)
                Gap(18.dp)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    if (empty) DateNav("10월 26일(월) ~ 11월 1일(일)", nextEnabled = true, onNext = { nav.replace(Screen.HistoryWeek) })
                    else DateNav("11월 2일(월) ~ 11월 8일(일)", onPrev = { nav.replace(Screen.HistoryWeekEmpty) })
                    Fill()
                    SumRow(if (empty) listOf("10회" to "주행", "6시간 2분" to "총 관찰 시간", "182.0km" to "총 이동 거리", "9회" to "부주의", "0회" to "졸음 감지")
                    else listOf("9회" to "주행", "5시간 34분" to "총 관찰 시간", "213.6km" to "총 이동 거리", "14회" to "부주의", "2회" to "졸음 감지"))
                }
                Gap(18.dp); HLine(); Gap(18.dp)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    T(if (empty) AnnotatedString("이 기간에는 졸음이 감지되지 않았습니다") else peakText("이번 주에는 ", hour, metric), S.peak); Fill()
                    Segmented(listOf("부주의", "졸음"), metric, small = true) { if (!empty) metric = it }
                }
                Gap(8.dp)
                Row(horizontalArrangement = Arrangement.spacedBy(48.dp)) {
                    Column {
                        T(if (empty) "요일별" else "요일별, 요일을 누르면 그날의 일간 리포트로 이동합니다", S.t(15, 20, color = NColor.ink3))
                        BarChart(day, { Sample.days[it] }, 360.dp, 410.dp, peak = peakOf(day), onBar = { i -> if (!empty && day[i] != null) nav.replace(Screen.HistoryDay) })
                    }
                    Column {
                        T(if (empty) "시간대별" else "시간대별, 한 주 동안 같은 시간대의 기록을 합산합니다", S.t(15, 20, color = NColor.ink3))
                        val ti = tip
                        BarChart(hour, Sample.hourLabel, 744.dp, 410.dp, peak = peakOf(hour), tipIndex = ti,
                            tip = ti?.let { i -> hour[i]?.let { "${slotName(i)} · ${if (metric == 0) "부주의" else "졸음"} ${it}회 · 주간 누적 관찰 ${mins(Sample.weekObs[i] ?: 0)}" } ?: "${slotName(i)} · 관찰 기록 없음" },
                            onBar = { if (!empty) tip = it })
                    }
                }
                Gap(14.dp)
                Legend(if (empty) listOf("zero", "none") else listOf("peakw", "bar", "zero", "none"))
            }
        }
    }
}

