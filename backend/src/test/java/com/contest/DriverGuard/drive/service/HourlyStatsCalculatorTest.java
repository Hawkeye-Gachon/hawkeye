package com.contest.DriverGuard.drive.service;

import com.contest.DriverGuard.drive.dto.DriveDto;
import org.junit.jupiter.api.Test;

import java.time.OffsetDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class HourlyStatsCalculatorTest {

    private static OffsetDateTime kst(String s) {
        return OffsetDateTime.parse(s + "+09:00");
    }

    private static DriveDto.Event event(DriveDto.EventType type, String at) {
        return new DriveDto.Event(null, 1L, type, kst(at), null, null);
    }

    @Test
    void 시간대별로_관찰시간과_감지횟수를_나눈다() {
        List<DriveDto.HourlyStat> stats = HourlyStatsCalculator.calculate(1L, 2L,
                kst("2026-11-06T14:40:00"), kst("2026-11-06T15:20:00"), 1000,
                List.of(event(DriveDto.EventType.INATTENTION, "2026-11-06T14:52:31"),
                        event(DriveDto.EventType.DROWSINESS, "2026-11-06T15:11:52"),
                        event(DriveDto.EventType.INATTENTION, "2026-11-06T15:00:00")));

        assertThat(stats).hasSize(2);
        assertThat(stats.get(0).getHourStart()).isEqualTo(kst("2026-11-06T14:00:00"));
        assertThat(stats.get(0).getObservedSeconds()).isEqualTo(1200);
        assertThat(stats.get(0).getInattentionCount()).isEqualTo(1);
        assertThat(stats.get(1).getObservedSeconds()).isEqualTo(1200);
        assertThat(stats.get(1).getInattentionCount()).isEqualTo(1);   // 정각은 다음 구간
        assertThat(stats.get(1).getDrowsinessCount()).isEqualTo(1);
        assertThat(stats.get(0).getDistanceMeters()).isEqualTo(500);
        assertThat(stats.get(1).getDistanceMeters()).isEqualTo(500);
    }

    @Test
    void 자정을_넘는_주행은_날짜별로_나뉘고_거리_합계는_정확히_같다() {
        List<DriveDto.HourlyStat> stats = HourlyStatsCalculator.calculate(1L, 2L,
                kst("2026-11-06T23:30:00"), kst("2026-11-07T00:31:00"), 10001, List.of());

        assertThat(stats).extracting(DriveDto.HourlyStat::getHourStart)
                .containsExactly(kst("2026-11-06T23:00:00"), kst("2026-11-07T00:00:00"));
        assertThat(stats).extracting(DriveDto.HourlyStat::getObservedSeconds).containsExactly(1800, 1860);
        assertThat(stats.stream().mapToInt(DriveDto.HourlyStat::getDistanceMeters).sum()).isEqualTo(10001);
    }

    @Test
    void 종료가_정각이면_빈_구간을_만들지_않고_종료시각_감지는_마지막_구간에_센다() {
        List<DriveDto.HourlyStat> stats = HourlyStatsCalculator.calculate(1L, 2L,
                kst("2026-11-06T14:00:00"), kst("2026-11-06T15:00:00"), 0,
                List.of(event(DriveDto.EventType.DROWSINESS, "2026-11-06T15:00:00")));

        assertThat(stats).hasSize(1);
        assertThat(stats.get(0).getObservedSeconds()).isEqualTo(3600);
        assertThat(stats.get(0).getDrowsinessCount()).isEqualTo(1);
    }

    @Test
    void 시작과_종료가_같으면_구간_1개() {
        List<DriveDto.HourlyStat> stats = HourlyStatsCalculator.calculate(1L, 2L,
                kst("2026-11-06T14:10:00"), kst("2026-11-06T14:10:00"), 0, List.of());

        assertThat(stats).hasSize(1);
        assertThat(stats.get(0).getObservedSeconds()).isZero();
    }

    @Test
    void 다른_시간대_오프셋으로_와도_서울_정시로_나눈다() {
        // 05:40Z = 14:40 KST
        List<DriveDto.HourlyStat> stats = HourlyStatsCalculator.calculate(1L, 2L,
                OffsetDateTime.parse("2026-11-06T05:40:00Z"), OffsetDateTime.parse("2026-11-06T05:50:00Z"), 0, List.of());

        assertThat(stats.get(0).getHourStart()).isEqualTo(kst("2026-11-06T14:00:00"));
    }
}
