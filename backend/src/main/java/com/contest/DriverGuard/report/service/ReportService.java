package com.contest.DriverGuard.report.service;

import com.contest.DriverGuard.exception.ApiException;
import com.contest.DriverGuard.exception.ErrorCode;
import com.contest.DriverGuard.report.dao.ReportDao;
import com.contest.DriverGuard.report.dto.ReportDto;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.YearMonth;
import java.time.ZonedDateTime;
import java.time.format.DateTimeParseException;
import java.time.temporal.TemporalAdjusters;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;
import java.util.function.ToIntFunction;

import static com.contest.DriverGuard.drive.service.HourlyStatsCalculator.SEOUL;

/**
 * 리포트는 drive_hourly_stats (종료된 주행만) 만 읽어서 서울 기준 날짜·시간대로 집계.
 * 기간을 걸친 주행은 그 기간에 해당하는 구간만 합산하고, 주행 횟수에는 포함
 */
@Service
@RequiredArgsConstructor
public class ReportService {

    private final ReportDao reportDao;

    /** 달력: 주행 기록이 있는 날짜 */
    @Transactional(readOnly = true)
    public ReportDto.CalendarResponse calendar(Long userId, String month) {
        YearMonth ym = parse(month, "month", "yyyy-MM", YearMonth::parse);
        List<ReportDto.StatRow> rows = reportDao.findStats(userId,
                startOfDay(ym.atDay(1)), startOfDay(ym.plusMonths(1).atDay(1)));

        TreeSet<LocalDate> dates = new TreeSet<>();
        for (ReportDto.StatRow r : rows) {
            if (r.getObservedSeconds() > 0) {
                dates.add(seoul(r.getHourStart()).toLocalDate());
            }
        }
        return new ReportDto.CalendarResponse(ym.toString(), new ArrayList<>(dates));
    }

    /** 일간 리포트. date 를 생략하면 가장 최근 주행일 (기록이 없으면 오늘) */
    @Transactional(readOnly = true)
    public ReportDto.DailyResponse daily(Long userId, String date) {
        LocalDate day = date == null || date.isBlank()
                ? latestDriveDate(userId)
                : parse(date, "date", "yyyy-MM-dd", LocalDate::parse);
        OffsetDateTime from = startOfDay(day);
        OffsetDateTime to = startOfDay(day.plusDays(1));

        List<ReportDto.StatRow> rows = reportDao.findStats(userId, from, to);
        List<ReportDto.HourItem> hourly = hourly(rows);
        return new ReportDto.DailyResponse(day, summary(rows), hourly, peak(hourly),
                reportDao.findDrives(userId, from, to));
    }

    /** 주간 통계. weekStart(월요일) 를 생략하면 가장 최근 주행일이 속한 주 (기록이 없으면 이번 주) */
    @Transactional(readOnly = true)
    public ReportDto.WeeklyResponse weekly(Long userId, String weekStart) {
        LocalDate monday;
        if (weekStart == null || weekStart.isBlank()) {
            monday = latestDriveDate(userId).with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY));
        } else {
            monday = parse(weekStart, "weekStart", "yyyy-MM-dd", LocalDate::parse);
            if (monday.getDayOfWeek() != DayOfWeek.MONDAY) {
                throw new ApiException(ErrorCode.INVALID_WEEK_START);
            }
        }

        List<ReportDto.StatRow> rows = reportDao.findStats(userId, startOfDay(monday), startOfDay(monday.plusDays(7)));

        List<ReportDto.DayItem> daily = new ArrayList<>(7);
        for (int i = 0; i < 7; i++) {
            LocalDate d = monday.plusDays(i);
            List<ReportDto.StatRow> dayRows = rows.stream()
                    .filter(r -> seoul(r.getHourStart()).toLocalDate().equals(d))
                    .toList();
            ReportDto.Summary s = summary(dayRows);
            daily.add(new ReportDto.DayItem(d, d.getDayOfWeek(), s.getDriveCount(), s.getObservedSeconds(),
                    s.getInattentionCount(), s.getDrowsinessCount()));
        }

        List<ReportDto.HourItem> hourly = hourly(rows);
        return new ReportDto.WeeklyResponse(monday, monday.plusDays(6), summary(rows), daily, hourly, peak(hourly));
    }

    // ---------------------------------------------------------------

    private static ReportDto.Summary summary(List<ReportDto.StatRow> rows) {
        Set<Long> drives = new HashSet<>();
        ReportDto.Summary s = new ReportDto.Summary();
        for (ReportDto.StatRow r : rows) {
            if (r.getObservedSeconds() > 0) {
                drives.add(r.getDriveId());
            }
            s.setObservedSeconds(s.getObservedSeconds() + r.getObservedSeconds());
            s.setDistanceMeters(s.getDistanceMeters() + r.getDistanceMeters());
            s.setInattentionCount(s.getInattentionCount() + r.getInattentionCount());
            s.setDrowsinessCount(s.getDrowsinessCount() + r.getDrowsinessCount());
        }
        s.setDriveCount(drives.size());
        return s;
    }

    /** 0~23시 24개. 행이 없는 시간대는 0 */
    private static List<ReportDto.HourItem> hourly(List<ReportDto.StatRow> rows) {
        List<ReportDto.HourItem> hours = new ArrayList<>(24);
        for (int h = 0; h < 24; h++) {
            hours.add(new ReportDto.HourItem(h, 0, 0, 0));
        }
        for (ReportDto.StatRow r : rows) {
            ReportDto.HourItem item = hours.get(seoul(r.getHourStart()).getHour());
            item.setObservedSeconds(item.getObservedSeconds() + r.getObservedSeconds());
            item.setInattentionCount(item.getInattentionCount() + r.getInattentionCount());
            item.setDrowsinessCount(item.getDrowsinessCount() + r.getDrowsinessCount());
        }
        return hours;
    }

    private static ReportDto.Peak peak(List<ReportDto.HourItem> hourly) {
        return new ReportDto.Peak(peakOf(hourly, ReportDto.HourItem::getInattentionCount),
                peakOf(hourly, ReportDto.HourItem::getDrowsinessCount));
    }

    private static ReportDto.PeakItem peakOf(List<ReportDto.HourItem> hourly, ToIntFunction<ReportDto.HourItem> count) {
        int max = hourly.stream().mapToInt(count).max().orElse(0);
        List<Integer> hours = max == 0
                ? List.of()
                : hourly.stream().filter(h -> count.applyAsInt(h) == max).map(ReportDto.HourItem::getHour).toList();
        return new ReportDto.PeakItem(hours, max);
    }

    private LocalDate latestDriveDate(Long userId) {
        OffsetDateTime latest = reportDao.findLatestStartedAt(userId);
        return latest != null ? seoul(latest).toLocalDate() : LocalDate.now(SEOUL);
    }

    private static OffsetDateTime startOfDay(LocalDate date) {
        return date.atStartOfDay(SEOUL).toOffsetDateTime();
    }

    private static ZonedDateTime seoul(OffsetDateTime t) {
        return t.atZoneSameInstant(SEOUL);
    }

    private static <T> T parse(String value, String field, String format, Function<String, T> parser) {
        try {
            return parser.apply(value);
        } catch (DateTimeParseException e) {
            throw ApiException.invalidField(field, field + " 는 " + format + " 형식이어야 합니다.");
        }
    }
}
