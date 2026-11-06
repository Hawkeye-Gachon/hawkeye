package com.contest.DriverGuard.restplace.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.OffsetDateTime;

public class RestPlaceDto {

    /** rest_places 테이블 - 졸음쉼터 + 고속도로 휴게소 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RestPlace {
        private Long placeId;
        private PlaceType placeType;
        private String sourceKey;        // 원천 데이터의 고유 키 (place_type 과 함께 UNIQUE)
        private String name;
        private String routeName;
        private String direction;
        private Double latitude;
        private Double longitude;
        private String address;
        private Integer parkingSpaces;
        private Boolean active;
        private OffsetDateTime syncedAt;
        private OffsetDateTime createdAt;
        private OffsetDateTime updatedAt;

        public enum PlaceType {
            REST_AREA,
            DROWSY_SHELTER
        }

        /** 국내 좌표 범위 안인지 (테이블 CHECK 제약과 같은 기준) */
        public static boolean isValidCoordinate(Double lat, Double lng) {
            return lat != null && lng != null
                    && lat >= 33 && lat <= 39
                    && lng >= 124 && lng <= 132;
        }
    }

    /** 주변 휴게소/졸음쉼터 조회 요청 */
    @Data
    @NoArgsConstructor
    public static class NearbyRequest {
        @NotNull(message = "lat 을 입력해 주세요.")
        @DecimalMin(value = "-90.0", message = "lat 은 -90 ~ 90 이어야 합니다.")
        @DecimalMax(value = "90.0", message = "lat 은 -90 ~ 90 이어야 합니다.")
        private Double lat;

        @NotNull(message = "lng 를 입력해 주세요.")
        @DecimalMin(value = "-180.0", message = "lng 는 -180 ~ 180 이어야 합니다.")
        @DecimalMax(value = "180.0", message = "lng 는 -180 ~ 180 이어야 합니다.")
        private Double lng;

        @DecimalMin(value = "0.0", message = "heading 은 0 ~ 360 이어야 합니다.")
        @DecimalMax(value = "360.0", message = "heading 은 0 ~ 360 이어야 합니다.")
        private Double heading;                  // 진행 방향 (북쪽 0도, 시계 방향). 없으면 null

        @PositiveOrZero(message = "speed 는 0 이상이어야 합니다.")
        private Double speed;                    // m/s

        private RestPlace.PlaceType type;        // null 이면 전체
    }

    /** 주변 조회 응답: 가장 가까운 쉴 곳 1곳 (앱이 경유지로 추가) */
    @Data
    @AllArgsConstructor
    public static class NearbyResponse {
        private boolean directionApplied;        // 진행 방향을 반영했는지
        // 지금 고속도로 위인지 (길찾기 출발 도로 기준). false 면 place 는 null → 앱은 일반 안내(안전한 곳에 정차 등)
        // null = 확인 못함 (근처에 휴게소가 없어 길찾기를 안 했거나 길찾기 실패)
        private Boolean onHighway;
        private DistanceSource distanceSource;
        private NearestPlace place;              // 없으면 null

        public enum DistanceSource {
            ROAD,         // 길찾기로 확인한 실제 도로 거리 기준 (반대 차선 제외 완료)
            UNAVAILABLE   // 길찾기 실패로 확인 불가. 틀린 안내를 막기 위해 place 는 null
        }
    }

    /** 가장 가까운 쉴 곳 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class NearestPlace {
        private Long placeId;
        private RestPlace.PlaceType type;        // REST_AREA(휴게소) / DROWSY_SHELTER(졸음쉼터)
        private String name;
        private Double latitude;
        private Double longitude;
        private int roadDistanceMeters;          // 앞으로 가야 하는 실제 도로 거리 (카카오 길찾기 기준)
        private int durationSeconds;             // 예상 소요 시간
    }

    /** 내부 검색 결과 1건 (판정·회귀 테스트용. API 로는 NearestPlace 만 나감) */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class NearbyItem {
        private Long placeId;
        private RestPlace.PlaceType type;
        private String name;
        private String routeName;
        private String direction;
        private Double latitude;
        private Double longitude;
        private Integer parkingSpaces;           // 졸음쉼터만 값 있음
        private int straightDistanceMeters;      // 직선 거리
        private int roadDistanceMeters;          // 실제 도로 거리
        private int durationSeconds;             // 예상 소요 시간
    }
}