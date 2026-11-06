package com.contest.DriverGuard.drive;

import com.contest.DriverGuard.support.ApiTestSupport;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.nullValue;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 관찰 시작 · 종료 업로드 · 주행 리포트 · 주행 목록 */
class DriveApiTest extends ApiTestSupport {

    @Test
    void 관찰_시작은_같은_clientDriveId_로_다시_보내도_한_번만_만들어진다() throws Exception {
        String token = signupAndAgree();
        Map<String, Object> request = map("clientDriveId", UUID.randomUUID().toString(), "startedAt", minutesAgo(5).toString());

        long driveId = body(postJson("/api/drives", request, token)
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.status").value("IN_PROGRESS"))
                .andExpect(jsonPath("$.abandonedDriveId").value(nullValue()))).get("driveId").asLong();

        postJson("/api/drives", request, token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.driveId").value(driveId));
    }

    @Test
    void 다음_관찰을_시작하면_이전_주행은_ABANDONED_가_되고_나중에_업로드할_수_있다() throws Exception {
        String token = signupAndAgree();
        long first = start(token, minutesAgo(20));

        postJson("/api/drives", map("clientDriveId", UUID.randomUUID().toString(), "startedAt", minutesAgo(5).toString()), token)
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.status").value("IN_PROGRESS"))
                .andExpect(jsonPath("$.abandonedDriveId").value(first));

        postJson("/api/drives/" + first + "/end", endBody(minutesAgo(10), 1000, List.of(), List.of()), token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("ENDED"));
    }

    @Test
    void 진행_중_주행보다_이른_시각으로_등록하면_새_주행이_ABANDONED_로_만들어진다() throws Exception {
        String token = signupAndAgree();
        long current = start(token, minutesAgo(5));

        postJson("/api/drives", map("clientDriveId", UUID.randomUUID().toString(), "startedAt", minutesAgo(60).toString()), token)
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.status").value("ABANDONED"))
                .andExpect(jsonPath("$.abandonedDriveId").value(nullValue()));

        assertThat(jdbcTemplate.queryForObject("SELECT status FROM drives WHERE drive_id = ?", String.class, current))
                .isEqualTo("IN_PROGRESS");
    }

    @Test
    void 관찰_시작_입력값_오류() throws Exception {
        String token = signupAndAgree();

        postJson("/api/drives", map("clientDriveId", "abc", "startedAt", minutesAgo(1).toString()), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("clientDriveId"));
        postJson("/api/drives", map("clientDriveId", UUID.randomUUID().toString(), "startedAt", minutesAgo(-60).toString()), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("startedAt"));
        postJson("/api/drives", map("clientDriveId", UUID.randomUUID().toString(), "startedAt", "2026-11-06T08:00:00"), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("startedAt"));   // 시간대 없는 시각
    }

    @Test
    void 종료_업로드하면_리포트를_돌려주고_기간_밖이나_중복_항목은_건너뛴다() throws Exception {
        String token = signupAndAgree();
        OffsetDateTime startedAt = minutesAgo(40);
        long driveId = start(token, startedAt);
        long placeId = jdbcTemplate.queryForObject(
                "SELECT min(place_id) FROM rest_places WHERE active AND latitude IS NOT NULL", Long.class);

        List<Map<String, Object>> events = List.of(
                event("INATTENTION", startedAt.plusMinutes(10), 37.10, 127.11),
                event("INATTENTION", startedAt.plusMinutes(10), 37.10, 127.11),   // 같은 시각 중복
                event("DROWSINESS", startedAt.plusMinutes(20), 37.14, 127.09),
                event("INATTENTION", startedAt.minusHours(2), null, null));       // 주행 기간 밖
        List<Map<String, Object>> waypoints = List.of(
                map("placeId", placeId, "mode", "WAYPOINT", "trigger", "DROWSINESS",
                        "addedAt", startedAt.plusMinutes(21).toString(), "visitedAt", startedAt.plusMinutes(30).toString()),
                map("placeId", 999_999_999, "mode", "WAYPOINT", "trigger", "DROWSINESS",
                        "addedAt", startedAt.plusMinutes(22).toString()));         // 없는 장소
        Map<String, Object> end = endBody(startedAt.plusMinutes(35), 28400, events, waypoints);

        postJson("/api/drives/" + driveId + "/end", end, token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("ENDED"))
                .andExpect(jsonPath("$.observedSeconds").value(35 * 60))
                .andExpect(jsonPath("$.distanceMeters").value(28400))
                .andExpect(jsonPath("$.inattentionCount").value(1))
                .andExpect(jsonPath("$.drowsinessCount").value(1))
                .andExpect(jsonPath("$.route.start.latitude").value(37.07))
                .andExpect(jsonPath("$.route.end.latitude").value(37.28))
                .andExpect(jsonPath("$.events.length()").value(2))
                .andExpect(jsonPath("$.waypoints.length()").value(1))
                .andExpect(jsonPath("$.waypoints[0].placeName").isNotEmpty())
                .andExpect(jsonPath("$.waypoints[0].visited").value(true))
                .andExpect(jsonPath("$.skipped.events").value(2))
                .andExpect(jsonPath("$.skipped.waypoints").value(1));

        // 응답을 못 받아 다시 보내도 같은 리포트 (두 번 저장되지 않음)
        postJson("/api/drives/" + driveId + "/end", end, token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.events.length()").value(2))
                .andExpect(jsonPath("$.skipped").doesNotExist());

        getJson("/api/drives/" + driveId, token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.driveId").value(driveId))
                .andExpect(jsonPath("$.skipped").doesNotExist());
    }

    @Test
    void 종료_업로드_입력값_오류() throws Exception {
        String token = signupAndAgree();
        OffsetDateTime startedAt = minutesAgo(30);
        long driveId = start(token, startedAt);

        postJson("/api/drives/" + driveId + "/end", endBody(startedAt.minusMinutes(1), 0, List.of(), List.of()), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("endedAt"));
        postJson("/api/drives/" + driveId + "/end", endBody(startedAt.plusMinutes(10), 999_999, List.of(), List.of()), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("distanceMeters"));
        postJson("/api/drives/" + driveId + "/end",
                endBody(startedAt.plusMinutes(10), 0, List.of(map("type", "SLEEP", "detectedAt", startedAt.plusMinutes(1).toString())), List.of()), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("events[0].type"));
    }

    @Test
    void 종료_전_리포트는_409_다른_사용자의_주행은_404() throws Exception {
        String token = signupAndAgree();
        long driveId = start(token, minutesAgo(10));

        getJson("/api/drives/" + driveId, token)
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("DRIVE_NOT_ENDED"));

        String other = signupAndAgree();
        getJson("/api/drives/" + driveId, other)
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value("DRIVE_NOT_FOUND"));
        postJson("/api/drives/" + driveId + "/end", endBody(minutesAgo(1), 0, List.of(), List.of()), other)
                .andExpect(status().isNotFound());
    }

    @Test
    void 주행_목록은_최근_시작순이고_커서로_다음_페이지를_받는다() throws Exception {
        String token = signupAndAgree();
        long[] ids = new long[3];
        for (int i = 0; i < 3; i++) {
            OffsetDateTime startedAt = minutesAgo(90 - i * 30L);
            ids[i] = start(token, startedAt);
            postJson("/api/drives/" + ids[i] + "/end", endBody(startedAt.plusMinutes(20), 1000, List.of(), List.of()), token)
                    .andExpect(status().isOk());
        }

        JsonNode page1 = body(getJson("/api/drives?size=2", token).andExpect(status().isOk()));
        assertThat(page1.get("drives")).extracting(d -> d.get("driveId").asLong()).containsExactly(ids[2], ids[1]);

        JsonNode page2 = body(getJson("/api/drives?size=2&cursor=" + page1.get("nextCursor").asText(), token));
        assertThat(page2.get("drives")).extracting(d -> d.get("driveId").asLong()).containsExactly(ids[0]);
        assertThat(page2.get("nextCursor").isNull()).isTrue();

        getJson("/api/drives?size=0", token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("size"));
    }

    // ---------------------------------------------------------------

    private long start(String token, OffsetDateTime startedAt) throws Exception {
        return body(postJson("/api/drives", map("clientDriveId", UUID.randomUUID().toString(),
                "startedAt", startedAt.toString()), token)).get("driveId").asLong();
    }

    private static Map<String, Object> event(String type, OffsetDateTime at, Double lat, Double lng) {
        return map("type", type, "detectedAt", at.toString(), "latitude", lat, "longitude", lng);
    }

    private static Map<String, Object> endBody(OffsetDateTime endedAt, int distance,
                                               List<Map<String, Object>> events, List<Map<String, Object>> waypoints) {
        return map("endedAt", endedAt.toString(), "distanceMeters", distance, "events", events,
                "route", List.of(List.of(37.07, 127.13), List.of(37.15, 127.09), List.of(37.28, 127.05)),
                "waypoints", waypoints);
    }
}
