package com.contest.DriverGuard.report.dto;

import com.contest.DriverGuard.drive.dto.DriveDto;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.List;

public class ReportDto {

    @Data
    @AllArgsConstructor
    public static class CalendarResponse {
        private String month;                 // yyyy-MM
        private List<LocalDate> driveDates;   // 주행 기록이 있는 날짜 (점 표시)
    }

    @Data
    @AllArgsConstructor
    public static class DailyResponse {
        private LocalDate date;
        private Summary summary;
        private List<HourItem> hourly;   // 항상 0~23시 24개
        private Peak peak;
        private List<DriveDto.ListItem> drives;   // 그날과 관찰 시간이 겹치는 주행, 최신순 (주행 전체 값)
    }

    @Data
    @AllArgsConstructor
    public static class WeeklyResponse {
        private LocalDate weekStart;     // 월요일
        private LocalDate weekEnd;       // 일요일
        private Summary summary;
        private List<DayItem> daily;     // 항상 월~일 7개
        private List<HourItem> hourly;   // 한 주 동안 같은 시간대 합산 24개
        private Peak peak;
    }

    /** 기간 요약. 기간을 걸친 주행은 그 기간에 해당하는 부분만 합산 */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Summary {
        private int driveCount;
        private int observedSeconds;
        private int distanceMeters;
        private int inattentionCount;
        private int drowsinessCount;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class HourItem {
        private int hour;
        private int observedSeconds;    // 0 이면 "관찰 기록 없음", 0 보다 크고 횟수 0 이면 "0회"
        private int inattentionCount;
        private int drowsinessCount;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DayItem {
        private LocalDate date;
        private DayOfWeek dayOfWeek;
        private int driveCount;
        private int observedSeconds;
        private int inattentionCount;
        private int drowsinessCount;
    }

    @Data
    @AllArgsConstructor
    public static class Peak {
        private PeakItem inattention;
        private PeakItem drowsiness;
    }

    /** 가장 많이 감지된 시간대. 같은 횟수면 여러 개. 감지가 없으면 hours = [], count = 0 */
    @Data
    @AllArgsConstructor
    public static class PeakItem {
        private List<Integer> hours;
        private int count;
    }

    /** drive_hourly_stats 조회 행 */
    @Data
    @NoArgsConstructor
    public static class StatRow {
        private Long driveId;
        private OffsetDateTime hourStart;
        private int observedSeconds;
        private int inattentionCount;
        private int drowsinessCount;
        private int distanceMeters;
    }
}
