package com.contest.DriverGuard.drive.dto;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.OffsetDateTime;
import java.util.List;

public class DriveDto {

    public enum EventType { INATTENTION, DROWSINESS }

    public enum NavMode { WAYPOINT, DESTINATION }

    public enum TriggerType { DROWSINESS, INATTENTION_REPEATED }

    // ================================================================
    // 관찰 시작
    // ================================================================

    @Data
    @NoArgsConstructor
    public static class StartRequest {
        @NotBlank(message = "clientDriveId 를 입력해 주세요.")
        @Pattern(regexp = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
                message = "clientDriveId 는 UUID 형식이어야 합니다.")
        private String clientDriveId;

        @NotNull(message = "startedAt 을 입력해 주세요.")
        private OffsetDateTime startedAt;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class StartResponse {
        private Long driveId;
        private Drive.Status status;
        private OffsetDateTime startedAt;
        private Long abandonedDriveId;   // 이번 요청으로 ABANDONED 가 된 이전 주행. 없으면 null

        @JsonIgnore
        private boolean created;         // false 면 같은 clientDriveId 재전송 (200)
    }

    // ================================================================
    // 관찰 종료 · 기록 업로드
    // ================================================================

    @Data
    @NoArgsConstructor
    public static class EndRequest {
        @NotNull(message = "endedAt 을 입력해 주세요.")
        private OffsetDateTime endedAt;

        /** 앱이 계산한 이동 거리 (m). 위치 기록이 없으면 0 */
        @NotNull(message = "distanceMeters 를 입력해 주세요.")
        @Min(value = 0, message = "distanceMeters 는 0 이상이어야 합니다.")
        private Integer distanceMeters;

        @NotNull(message = "events 를 입력해 주세요. 없으면 빈 배열")
        @Size(max = 2000, message = "events 는 최대 2,000건입니다.")
        private List<@Valid EventRequest> events;

        /** 앱에서 단순화한 경로 [[위도, 경도], ...]. 없으면 빈 배열 */
        @NotNull(message = "route 를 입력해 주세요. 없으면 빈 배열")
        @Size(max = 2000, message = "route 는 최대 2,000점입니다.")
        private List<List<Double>> route;

        @NotNull(message = "waypoints 를 입력해 주세요. 없으면 빈 배열")
        @Size(max = 20, message = "waypoints 는 최대 20건입니다.")
        private List<@Valid WaypointRequest> waypoints;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class EventRequest {
        @NotNull(message = "type 을 입력해 주세요.")
        private EventType type;

        @NotNull(message = "detectedAt 을 입력해 주세요.")
        private OffsetDateTime detectedAt;

        @DecimalMin(value = "-90.0", message = "latitude 범위를 확인해 주세요.")
        @DecimalMax(value = "90.0", message = "latitude 범위를 확인해 주세요.")
        private Double latitude;

        @DecimalMin(value = "-180.0", message = "longitude 범위를 확인해 주세요.")
        @DecimalMax(value = "180.0", message = "longitude 범위를 확인해 주세요.")
        private Double longitude;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class WaypointRequest {
        @NotNull(message = "placeId 를 입력해 주세요.")
        private Long placeId;

        @NotNull(message = "mode 를 입력해 주세요.")
        private NavMode mode;

        @NotNull(message = "trigger 를 입력해 주세요.")
        private TriggerType trigger;

        /** NaviHelper 성공 콜백을 받은 시각 */
        @NotNull(message = "addedAt 을 입력해 주세요.")
        private OffsetDateTime addedAt;

        /** 쉼터 방문을 앱에서 확인한 시각 (방문 버튼). 방문하지 않았으면 null */
        private OffsetDateTime visitedAt;
    }

    // ================================================================
    // 리포트 (종료 응답 · 주행 상세)
    // ================================================================

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class Report {
        private Long driveId;
        private Drive.Status status;
        private OffsetDateTime startedAt;
        private OffsetDateTime endedAt;
        private Integer observedSeconds;
        private Integer distanceMeters;
        private Integer inattentionCount;
        private Integer drowsinessCount;
        private Route route;
        private List<EventItem> events;
        private List<WaypointItem> waypoints;
        private Skipped skipped;   // 종료 업로드 응답에만 포함
    }

    @Data
    @AllArgsConstructor
    public static class Route {
        private Point start;               // 경로 기록이 없으면 null
        private Point end;
        private List<List<Double>> points; // [[위도, 경도], ...]
    }

    @Data
    @AllArgsConstructor
    public static class Point {
        private Double latitude;
        private Double longitude;
    }

    @Data
    @AllArgsConstructor
    public static class EventItem {
        private EventType type;
        private OffsetDateTime detectedAt;
        private Double latitude;
        private Double longitude;
    }

    @Data
    @Builder
    @AllArgsConstructor
    public static class WaypointItem {
        private Long waypointId;
        private Long placeId;
        private String placeName;
        private String placeType;
        private NavMode mode;
        private TriggerType trigger;
        private OffsetDateTime addedAt;
        private boolean visited;
        private OffsetDateTime visitedAt;
    }

    /** 저장하지 않고 건너뛴 항목 수 */
    @Data
    @AllArgsConstructor
    public static class Skipped {
        private int events;
        private int waypoints;
    }

    // ================================================================
    // 주행 목록
    // ================================================================

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ListItem {
        private Long driveId;
        private OffsetDateTime startedAt;
        private OffsetDateTime endedAt;
        private Integer observedSeconds;
        private Integer distanceMeters;
        private Integer inattentionCount;
        private Integer drowsinessCount;

        public static ListItem from(Drive d) {
            return ListItem.builder()
                    .driveId(d.getDriveId())
                    .startedAt(d.getStartedAt())
                    .endedAt(d.getEndedAt())
                    .observedSeconds(d.getObservedSeconds())
                    .distanceMeters(d.getDistanceMeters())
                    .inattentionCount(d.getInattentionCount())
                    .drowsinessCount(d.getDrowsinessCount())
                    .build();
        }
    }

    @Data
    @AllArgsConstructor
    public static class ListResponse {
        private List<ListItem> drives;
        private String nextCursor;   // 마지막 페이지면 null
    }

    // ================================================================
    // 테이블 행
    // ================================================================

    /** drives 테이블 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Drive {
        private Long driveId;
        private Long userId;
        private String clientDriveId;   // UUID 소문자 문자열 (DB 는 UUID 타입, SQL 에서 변환)
        private Status status;
        private OffsetDateTime startedAt;
        private OffsetDateTime endedAt;
        private Integer observedSeconds;
        private Integer distanceMeters;
        private Integer inattentionCount;
        private Integer drowsinessCount;
        private String routeJson;       // route_simplified (JSONB) 를 문자열로
        private OffsetDateTime createdAt;
        private OffsetDateTime updatedAt;

        public enum Status {
            IN_PROGRESS,
            ENDED,
            ABANDONED
        }
    }

    /** drive_events 테이블 */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Event {
        private Long eventId;
        private Long driveId;
        private EventType eventType;
        private OffsetDateTime detectedAt;
        private Double latitude;
        private Double longitude;
    }

    /** drive_waypoints 테이블 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Waypoint {
        private Long waypointId;
        private Long driveId;
        private Long placeId;
        private String placeName;
        private String placeType;
        private Double latitude;
        private Double longitude;
        private NavMode navMode;
        private TriggerType triggerType;
        private OffsetDateTime addedAt;
        private Boolean visited;
        private OffsetDateTime visitedAt;
    }

    /** drive_hourly_stats 테이블 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class HourlyStat {
        private Long driveId;
        private Long userId;
        private OffsetDateTime hourStart;   // 서울 기준 정시
        private int observedSeconds;
        private int inattentionCount;
        private int drowsinessCount;
        private int distanceMeters;
    }
}
