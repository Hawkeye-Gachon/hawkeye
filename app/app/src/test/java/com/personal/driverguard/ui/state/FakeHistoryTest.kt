package com.personal.driverguard.ui.state

import java.time.DayOfWeek
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.YearMonth
import java.time.temporal.TemporalAdjusters
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FakeHistoryTest {
    private val today = LocalDate.now(SEOUL)
    private val thisMonday = today.with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY))

    @Test
    fun thisAndLastWeekHaveDrivesAndConsistentTotals() {
        for (start in listOf(thisMonday.minusWeeks(1), thisMonday)) {
            val w = FakeHistory.weekly(start)
            assertEquals(7, w.daily.size)
            assertEquals(24, w.hourly.size)
            assertEquals(w.summary.driveCount, w.daily.sumOf { it.driveCount })
            assertEquals(w.summary.inattentionCount, w.daily.sumOf { it.inattentionCount })
            assertEquals(w.summary.inattentionCount, w.hourly.sumOf { it.inattentionCount })
            assertEquals(w.summary.observedSeconds, w.hourly.sumOf { it.observedSeconds })
        }
        assertTrue("지난주에는 주행이 있어야 한다", FakeHistory.weekly(thisMonday.minusWeeks(1)).summary.driveCount > 0)
    }

    @Test
    fun noFutureDrivesAndOlderWeeksAreEmpty() {
        FakeHistory.daily(today).drives.forEach { assertTrue(OffsetDateTime.parse(it.endedAt).toLocalDateTime() <= java.time.LocalDateTime.now(SEOUL)) }
        assertEquals(0, FakeHistory.daily(today.plusDays(1)).summary.driveCount)
        assertEquals(0, FakeHistory.weekly(thisMonday.minusWeeks(2)).summary.driveCount)
    }

    @Test
    fun latestDriveOpensReportWithRoute() {
        val last = FakeHistory.latestDrive()
        assertNotNull(last)
        val r = FakeHistory.report(last!!.driveId)!!
        assertEquals(last.inattentionCount, r.inattentionCount)
        assertTrue(r.route!!.points.size >= 2)
        r.events.forEach { assertNotNull(it.latitude) }
        assertTrue(FakeHistory.calendar(YearMonth.from(LocalDate.parse(FakeHistory.daily(null).date))).driveDates.isNotEmpty())
    }
}
