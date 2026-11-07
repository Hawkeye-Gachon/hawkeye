package com.personal.driverguard.ui.state

import java.time.DayOfWeek
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.format.TextStyle
import java.util.Locale
import kotlin.math.roundToInt

/** 화면 표시 형식. 시각은 모두 서울 기준으로 보여 준다. */
object Fmt {
    private val KO = Locale.KOREAN

    fun parse(iso: String?): OffsetDateTime? = iso?.let { runCatching { OffsetDateTime.parse(it).atZoneSameInstant(SEOUL).toOffsetDateTime() }.getOrNull() }

    /** 00:18:42 */
    fun clock(seconds: Long): String = "%02d:%02d:%02d".format(seconds / 3600, seconds % 3600 / 60, seconds % 60)

    /** 42분 18초 / 1시간 17분 / 0초 */
    fun duration(seconds: Int?): String {
        val s = (seconds ?: 0).coerceAtLeast(0)
        return when {
            s >= 3600 -> "${s / 3600}시간 ${s % 3600 / 60}분"
            s >= 60 -> "${s / 60}분 ${s % 60}초"
            else -> "${s}초"
        }
    }

    /** 28.4km (1km 미만은 m) */
    fun distance(meters: Int?): String {
        val m = (meters ?: 0).coerceAtLeast(0)
        return if (m < 1000) "${m}m" else "%.1fkm".format(m / 1000.0)
    }

    /** 오후 2:30 */
    fun time(t: OffsetDateTime?): String {
        t ?: return "-"
        val h = t.hour
        val ampm = if (h < 12) "오전" else "오후"
        val h12 = if (h % 12 == 0) 12 else h % 12
        return "$ampm $h12:%02d".format(t.minute)
    }

    fun time(iso: String?) = time(parse(iso))

    /** 11월 6일 금요일 */
    fun dateTitle(d: LocalDate): String = "${d.monthValue}월 ${d.dayOfMonth}일 ${d.dayOfWeek.getDisplayName(TextStyle.FULL, KO)}"

    /** 11월 2일(월) */
    fun dateShort(d: LocalDate): String = "${d.monthValue}월 ${d.dayOfMonth}일(${dayShort(d.dayOfWeek)})"

    fun dayShort(d: DayOfWeek): String = d.getDisplayName(TextStyle.SHORT, KO)

    /** 오늘 / 어제 / 11월 6일 */
    fun relativeDay(d: LocalDate, today: LocalDate = LocalDate.now(SEOUL)): String = when (d) {
        today -> "오늘"
        today.minusDays(1) -> "어제"
        else -> "${d.monthValue}월 ${d.dayOfMonth}일"
    }

    /** 전방 4.2km */
    fun ahead(meters: Int): String = "전방 " + if (meters < 1000) "${meters}m" else "%.1fkm".format(meters / 1000.0)

    fun minutes(seconds: Int): String = (seconds / 60.0).roundToInt().let { if (it >= 60) "${it / 60}시간 ${it % 60}분" else "${it}분" }
}
