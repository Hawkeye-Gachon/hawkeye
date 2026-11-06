package com.contest.DriverGuard.drive.service;

import com.contest.DriverGuard.drive.dto.DriveDto;

import java.time.Duration;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * 주행 1건을 서울 기준 1시간 구간으로 나눠 집계.
 * - 관찰 시간: 구간과 겹치는 시간
 * - 감지 횟수: 감지 시각이 속한 구간
 * - 이동 거리: 앱이 보낸 총 거리를 구간별 관찰 시간 비율로 나눔 (합계는 총 거리와 정확히 같음)
 */
public final class HourlyStatsCalculator {

    public static final ZoneId SEOUL = ZoneId.of("Asia/Seoul");

    private HourlyStatsCalculator() {
    }

    public static List<DriveDto.HourlyStat> calculate(Long driveId, Long userId,
                                                      OffsetDateTime startedAt, OffsetDateTime endedAt,
                                                      int distanceMeters, List<DriveDto.Event> events) {
        Instant start = startedAt.toInstant();
        Instant end = endedAt.toInstant();

        // 1) 구간 나누기 (시작 = 종료면 구간 1개, 관찰 시간 0)
        List<Instant> bucketStarts = new ArrayList<>();
        List<Long> bucketMillis = new ArrayList<>();
        ZonedDateTime bucket = ZonedDateTime.ofInstant(start, SEOUL).truncatedTo(ChronoUnit.HOURS);
        do {
            Instant bStart = bucket.toInstant();
            Instant bEnd = bucket.plusHours(1).toInstant();
            Instant segStart = start.isAfter(bStart) ? start : bStart;
            Instant segEnd = end.isBefore(bEnd) ? end : bEnd;
            bucketStarts.add(bStart);
            bucketMillis.add(Math.max(0, Duration.between(segStart, segEnd).toMillis()));
            bucket = bucket.plusHours(1);
        } while (bucket.toInstant().isBefore(end));

        int n = bucketStarts.size();
        List<DriveDto.HourlyStat> stats = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            stats.add(DriveDto.HourlyStat.builder()
                    .driveId(driveId)
                    .userId(userId)
                    .hourStart(OffsetDateTime.ofInstant(bucketStarts.get(i), SEOUL))
                    .observedSeconds((int) Math.min(3600, Math.round(bucketMillis.get(i) / 1000.0)))
                    .build());
        }

        // 2) 감지 횟수
        for (DriveDto.Event e : events) {
            DriveDto.HourlyStat s = stats.get(indexOf(bucketStarts, e.getDetectedAt().toInstant()));
            if (e.getEventType() == DriveDto.EventType.INATTENTION) {
                s.setInattentionCount(s.getInattentionCount() + 1);
            } else {
                s.setDrowsinessCount(s.getDrowsinessCount() + 1);
            }
        }

        // 3) 거리: 관찰 시간 비율로 나누고, 반올림 오차는 나머지가 큰 구간부터 1m 씩 배분
        long totalMillis = bucketMillis.stream().mapToLong(Long::longValue).sum();
        if (totalMillis == 0) {
            stats.get(0).setDistanceMeters(distanceMeters);
        } else {
            long assigned = 0;
            double[] remainders = new double[n];
            for (int i = 0; i < n; i++) {
                double exact = (double) distanceMeters * bucketMillis.get(i) / totalMillis;
                int floor = (int) Math.floor(exact);
                stats.get(i).setDistanceMeters(floor);
                remainders[i] = exact - floor;
                assigned += floor;
            }
            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                order.add(i);
            }
            order.sort(Comparator.comparingDouble((Integer i) -> remainders[i]).reversed());
            for (int k = 0; k < distanceMeters - assigned; k++) {
                DriveDto.HourlyStat s = stats.get(order.get(k % n));
                s.setDistanceMeters(s.getDistanceMeters() + 1);
            }
        }
        return stats;
    }

    /** 시각이 속한 구간. 범위를 벗어나면(종료 시각이 정시와 같은 경우 등) 가장 가까운 끝 구간 */
    private static int indexOf(List<Instant> bucketStarts, Instant t) {
        for (int i = bucketStarts.size() - 1; i >= 0; i--) {
            if (!t.isBefore(bucketStarts.get(i))) {
                return i;
            }
        }
        return 0;
    }
}
