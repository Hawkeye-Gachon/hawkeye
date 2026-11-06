package com.contest.DriverGuard.restplace.service;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 순환고속도로의 상·하행 판정 (카카오 없이 좌표 계산).
 *
 * 장소의 방향을 회전(시계·반시계)으로 바꿔, 차가 도는 회전과 비교한다.
 * - 휴게소 "구리(일산)": 일산 쪽으로 가는 차선. 같은 "일산"이라도 동쪽(구리)에서는 반시계, 남쪽(의왕)에서는 시계
 *   → 그 지명 쪽으로 더 짧게 도는 회전
 * - 졸음쉼터 "판교기점 + 일산종점": 노선 전체의 진행 방향 → 위치와 상관없이 한 회전 (shelterRotation)
 *
 * 졸음쉼터는 카카오가 길찾기를 못 하고(뒷길로 안내) 순환도로에는 기준이 될 휴게소도 적어서,
 * 일반 판정으로는 거의 나오지 않음 → 순환도로 위에서는 이 판정으로 보완.
 * 원에 가까운 순환도로에만 맞으므로 서울외곽순환만 등록 (2026-11-06).
 */
final class RingRoad {

    enum Rotation { CW, CCW }

    /** 서울외곽순환 (수도권제1순환). 둘레 약 128km */
    static final RingRoad SEOUL = new RingRoad(
            Set.of("서울외곽", "수도권제1순환"),
            37.55, 126.98,
            Map.of("일산", new double[]{37.675, 126.775},
                    "판교", new double[]{37.405, 127.095},
                    "구리", new double[]{37.600, 127.143},
                    "퇴계원", new double[]{37.648, 127.140}),
            Map.of("판교→일산", Rotation.CW, "일산→판교", Rotation.CCW));

    private static final List<RingRoad> ALL = List.of(SEOUL);

    private final Set<String> routes;                    // 정규화된 노선명 (카카오 도로명 / 데이터 노선명)
    private final double centerLat;
    private final double centerLng;
    private final Map<String, double[]> ends;            // 휴게소 방향 지명 좌표
    private final Map<String, Rotation> shelterLabels;   // 졸음쉼터 "기점→종점" 별 회전

    private RingRoad(Set<String> routes, double centerLat, double centerLng, Map<String, double[]> ends,
                     Map<String, Rotation> shelterLabels) {
        this.routes = routes;
        this.centerLat = centerLat;
        this.centerLng = centerLng;
        this.ends = ends;
        this.shelterLabels = shelterLabels;
    }

    /** 정규화된 노선명이 등록된 순환도로면 그 순환도로, 아니면 null */
    static RingRoad of(String normalizedRoute) {
        return ALL.stream().filter(r -> r.contains(normalizedRoute)).findFirst().orElse(null);
    }

    boolean contains(String normalizedRoute) {
        return routes.contains(normalizedRoute);
    }

    /** 차가 도는 방향: 중심에서 본 위치 벡터와 진행 벡터의 외적 부호 */
    Rotation travelRotation(double lat, double lng, double heading) {
        double k = Math.cos(Math.toRadians(centerLat));
        double rx = (lng - centerLng) * k;
        double ry = lat - centerLat;
        double vx = Math.sin(Math.toRadians(heading));   // 동쪽
        double vy = Math.cos(Math.toRadians(heading));   // 북쪽
        return rx * vy - ry * vx > 0 ? Rotation.CCW : Rotation.CW;
    }

    /**
     * 졸음쉼터의 방향. 쉼터 데이터의 "○○기점 + ○○종점"은 그 쉼터가 가리키는 쪽이 아니라 노선 전체의 진행 방향.
     * 서울외곽: 위치와 상관없이 "판교기점 + 일산종점" = 시계, "일산기점 + 판교종점" = 반시계.
     * (2026-11-06 카카오 실제 경로로 쉼터가 어느 방향 차선 오른쪽에 붙어 있는지 확인: 구리남양주·김포·청계·시흥 일치)
     * 그 밖의 표시(예: 성남 "퇴계원기점 + 구리종점", 순환도로에서 약 1km 떨어진 좌표)는 판정하지 않음 → null
     */
    Rotation shelterRotation(String start, String end) {
        if (start == null || end == null) {
            return null;
        }
        return shelterLabels.get(start.trim() + "→" + end.trim());
    }

    /** 휴게소의 방향: 이름의 방향("일산")은 그 쪽으로 가는 차선 → 그 지명 쪽으로 더 짧게 도는 회전. 모르는 지명이면 null */
    Rotation restAreaRotation(double lat, double lng, String destination) {
        double[] end = destination == null ? null : ends.get(destination.trim());
        if (end == null) {
            return null;
        }
        double d = normalize180(angle(end[0], end[1]) - angle(lat, lng));
        return d > 0 ? Rotation.CCW : Rotation.CW;
    }

    /**
     * 진행 방향으로 장소까지 원 둘레를 따라 잰 거리(m). 이미 지나쳤으면(반 바퀴 넘게 돌아야 하면) -1.
     * 반지름은 현재 위치와 장소의 중심 거리 평균 (원에 가까운 순환도로 기준 근사)
     */
    double aheadMeters(double lat, double lng, double placeLat, double placeLng, Rotation rotation) {
        double diff = angle(placeLat, placeLng) - angle(lat, lng);
        double ahead = ((rotation == Rotation.CCW ? diff : -diff) % 360 + 360) % 360;
        if (ahead > 180) {
            return -1;
        }
        double radius = (radiusMeters(lat, lng) + radiusMeters(placeLat, placeLng)) / 2;
        return Math.toRadians(ahead) * radius;
    }

    private double angle(double lat, double lng) {
        return Math.toDegrees(Math.atan2(lat - centerLat, (lng - centerLng) * Math.cos(Math.toRadians(centerLat))));
    }

    private double radiusMeters(double lat, double lng) {
        return GeoUtils.distanceMeters(centerLat, centerLng, lat, lng);
    }

    private static double normalize180(double deg) {
        return ((deg % 360) + 540) % 360 - 180;
    }
}
