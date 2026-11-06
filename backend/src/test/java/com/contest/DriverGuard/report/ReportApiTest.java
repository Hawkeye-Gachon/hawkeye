package com.contest.DriverGuard.report;

import com.contest.DriverGuard.support.ApiTestSupport;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.temporal.TemporalAdjusters;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 달력 · 일간 리포트 · 주간 통계.
 * 서울 기준 3일 전 주행 2건(출근길 · 오후)과 2일 전 23:30 ~ 1일 전 00:30 (자정을 넘긴 주행)을 넣고 집계를 확인
 */
class ReportApiTest extends ApiTestSupport {

    private String token;
    private final LocalDate d3 = LocalDate.now(SEOUL).minusDays(3);
    private final LocalDate d2 = d3.plusDays(1);
    private final LocalDate d1 = d3.plusDays(2);

    @BeforeEach
    void drives() throws Exception {
        token = signupAndAgree();
        // 3일 전 08:05 ~ 08:52:42 · 32.8km · 부주의 3
        drive(kst(3, 8, 5, 0), kst(3, 8, 52, 42), 32800,
                event("INATTENTION", kst(3, 8, 10, 0)), event("INATTENTION", kst(3, 8, 20, 0)), event("INATTENTION", kst(3, 8, 30, 0)));
        // 3일 전 14:40 ~ 15:22:18 · 28.4km · 부주의 2 (14시, 15시) · 졸음 1 (15시)
        drive(kst(3, 14, 40, 0), kst(3, 15, 22, 18), 28400,
                event("INATTENTION", kst(3, 14, 50, 0)), event("INATTENTION", kst(3, 15, 5, 0)), event("DROWSINESS", kst(3, 15, 11, 52)));
        // 2일 전 23:30 ~ 1일 전 00:30 · 6km · 부주의 1 (23시) · 졸음 1 (다음날 0시)
        drive(kst(2, 23, 30, 0), kst(1, 0, 30, 0), 6000,
                event("INATTENTION", kst(2, 23, 45, 0)), event("DROWSINESS", kst(1, 0, 10, 0)));
    }

    @Test
    void 일간_리포트는_시간대별로_집계하고_집중_시간대를_알려준다() throws Exception {
        JsonNode daily = body(getJson("/api/reports/daily?date=" + d3, token).andExpect(status().isOk()));

        assertThat(daily.get("summary").toString()).isEqualTo(
                "{\"driveCount\":2,\"observedSeconds\":5400,\"distanceMeters\":61200,\"inattentionCount\":5,\"drowsinessCount\":1}");
        assertThat(daily.get("hourly")).hasSize(24);
        assertThat(hour(daily, 8)).isEqualTo(List.of(2862, 3, 0));     // [관찰 초, 부주의, 졸음]
        assertThat(hour(daily, 14)).isEqualTo(List.of(1200, 1, 0));
        assertThat(hour(daily, 15)).isEqualTo(List.of(1338, 1, 1));
        assertThat(hour(daily, 9)).isEqualTo(List.of(0, 0, 0));          // 관찰 기록 없음
        assertThat(daily.at("/peak/inattention").toString()).isEqualTo("{\"hours\":[8],\"count\":3}");
        assertThat(daily.at("/peak/drowsiness").toString()).isEqualTo("{\"hours\":[15],\"count\":1}");
        assertThat(daily.get("drives")).hasSize(2);
        assertThat(daily.at("/drives/0/startedAt").asText()).startsWith(d3 + "T14:40");   // 최신순
    }

    @Test
    void 자정을_넘긴_주행은_두_날짜에_나눠_집계된다() throws Exception {
        getJson("/api/reports/daily?date=" + d2, token)
                .andExpect(jsonPath("$.summary.driveCount").value(1))
                .andExpect(jsonPath("$.summary.observedSeconds").value(1800))
                .andExpect(jsonPath("$.summary.distanceMeters").value(3000))
                .andExpect(jsonPath("$.summary.inattentionCount").value(1))
                .andExpect(jsonPath("$.hourly[23].observedSeconds").value(1800))
                .andExpect(jsonPath("$.drives[0].observedSeconds").value(3600));   // 주행 목록은 주행 전체 값

        getJson("/api/reports/daily?date=" + d1, token)
                .andExpect(jsonPath("$.summary.observedSeconds").value(1800))
                .andExpect(jsonPath("$.summary.distanceMeters").value(3000))
                .andExpect(jsonPath("$.summary.drowsinessCount").value(1))
                .andExpect(jsonPath("$.hourly[0].observedSeconds").value(1800));
    }

    @Test
    void 날짜를_생략하면_가장_최근_주행일() throws Exception {
        getJson("/api/reports/daily", token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.date").value(d2.toString()));   // 가장 최근 주행의 시작일
    }

    @Test
    void 주행이_없는_날은_0으로_채운다() throws Exception {
        getJson("/api/reports/daily?date=" + d3.minusDays(1), token)
                .andExpect(jsonPath("$.summary.driveCount").value(0))
                .andExpect(jsonPath("$.hourly.length()").value(24))
                .andExpect(jsonPath("$.peak.inattention.count").value(0))
                .andExpect(jsonPath("$.peak.inattention.hours.length()").value(0))
                .andExpect(jsonPath("$.drives.length()").value(0));
    }

    @Test
    void 주간_통계는_월요일부터_7일이고_자정을_넘긴_주행은_한_번만_센다() throws Exception {
        LocalDate monday = d3.with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY));
        JsonNode weekly = body(getJson("/api/reports/weekly?weekStart=" + monday, token).andExpect(status().isOk()));

        assertThat(weekly.get("weekEnd").asText()).isEqualTo(monday.plusDays(6).toString());
        assertThat(weekly.get("daily")).hasSize(7);
        assertThat(weekly.at("/daily/0/dayOfWeek").asText()).isEqualTo("MONDAY");
        assertThat(weekly.get("hourly")).hasSize(24);

        // 날짜별 기대값 [주행 수, 관찰 초, 거리] 중 이번 주에 들어가는 날만 합산
        Map<LocalDate, int[]> perDay = Map.of(d3, new int[]{2, 5400, 61200}, d2, new int[]{1, 1800, 3000}, d1, new int[]{1, 1800, 3000});
        int observed = 0, distance = 0;
        boolean overnightInWeek = false;
        for (var e : perDay.entrySet()) {
            int idx = (int) (e.getKey().toEpochDay() - monday.toEpochDay());
            if (idx < 0 || idx > 6) {
                continue;
            }
            observed += e.getValue()[1];
            distance += e.getValue()[2];
            overnightInWeek |= !e.getKey().equals(d3);
            assertThat(weekly.at("/daily/" + idx + "/driveCount").asInt()).isEqualTo(e.getValue()[0]);
        }
        assertThat(weekly.at("/summary/observedSeconds").asInt()).isEqualTo(observed);
        assertThat(weekly.at("/summary/distanceMeters").asInt()).isEqualTo(distance);
        assertThat(weekly.at("/summary/driveCount").asInt()).isEqualTo(overnightInWeek ? 3 : 2);
    }

    @Test
    void 주간_통계의_weekStart_는_월요일이어야_한다() throws Exception {
        LocalDate tuesday = d3.with(TemporalAdjusters.next(DayOfWeek.TUESDAY));
        getJson("/api/reports/weekly?weekStart=" + tuesday, token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_WEEK_START"));
        getJson("/api/reports/daily?date=2026-13-01", token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("date"));
    }

    @Test
    void 달력에는_주행_기록이_있는_날짜가_표시된다() throws Exception {
        JsonNode calendar = body(getJson("/api/reports/calendar?month=" + d3.toString().substring(0, 7), token)
                .andExpect(status().isOk()));

        List<String> expected = List.of(d3, d2, d1).stream()
                .filter(d -> d.getMonth() == d3.getMonth())
                .map(LocalDate::toString)
                .toList();
        assertThat(calendar.get("driveDates")).extracting(JsonNode::asText).containsExactlyElementsOf(expected);
    }

    // ---------------------------------------------------------------

    private void drive(OffsetDateTime startedAt, OffsetDateTime endedAt, int distance, Map<?, ?>... events) throws Exception {
        long driveId = body(postJson("/api/drives", map("clientDriveId", UUID.randomUUID().toString(),
                "startedAt", startedAt.toString()), token)).get("driveId").asLong();
        postJson("/api/drives/" + driveId + "/end", map("endedAt", endedAt.toString(), "distanceMeters", distance,
                "events", List.of(events), "route", List.of(), "waypoints", List.of()), token)
                .andExpect(status().isOk());
    }

    private static Map<String, Object> event(String type, OffsetDateTime at) {
        return map("type", type, "detectedAt", at.toString());
    }

    private static List<Integer> hour(JsonNode daily, int hour) {
        JsonNode h = daily.get("hourly").get(hour);
        return List.of(h.get("observedSeconds").asInt(), h.get("inattentionCount").asInt(), h.get("drowsinessCount").asInt());
    }
}
