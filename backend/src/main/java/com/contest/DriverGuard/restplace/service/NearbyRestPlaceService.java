package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyItem;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyRequest;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyResponse;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyResponse.DistanceSource;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearestPlace;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace.PlaceType;
import com.contest.DriverGuard.restplace.service.KakaoDirectionsClient.RoadPart;
import com.contest.DriverGuard.restplace.service.KakaoDirectionsClient.Route;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 주변 휴게소/졸음쉼터 조회 (졸음 감지 시 앱이 호출).
 * 원칙: 반대 차선 등 "지금 차선에서 바로 들어갈 수 없는 곳"은 내보내지 않는다. 확인이 안 되면 빼거나 UNAVAILABLE.
 *
 * [휴게소] 카카오 길찾기(출발 방향 = 진행 방향)의 경로 모양으로 판정 (2026-11-06 실측 기준):
 *   - DIRECT   : 지나치지 않고 도착 → 내 방향. 도로 거리 = 길찾기 거리
 *   - UTURN    : 휴게소 옆을 지나쳤다가 고속도로로 되돌아와 도착 (다음 IC 에서 회차) → 반대 차선
 *                또는 도착 직전 600m 방향이 진행 방향과 135° 넘게 반대 (종점 JC 연결로로 회차, 예: 춘천(부산)) → 반대 차선
 *   - BACKROAD : 지나쳤다가 일반도로로 도착 (카카오가 휴게소 뒷길로 안내, 예: 목감(서울)) → 이것만으로는 방향을 모름.
 *                같은 휴게소의 다른 쪽이 UTURN(= 반대 차선)으로 확인된 경우에만 내 방향으로 인정하고,
 *                도로 거리 = 본선에서 휴게소 옆을 지나는 지점까지
 *   - 그 밖에 도로 거리가 직선의 2배 이상 + 3km 이상 더 길면(다른 도로로 크게 우회) 제외
 *
 * [졸음쉼터] 카카오 지도에 진입로가 없어 길찾기로 직접 구할 수 없음 (상·하행 모두 일반도로 뒷길로 안내됨).
 *   내 방향으로 확인된 휴게소(반대편이 반대 차선으로 판정된 곳)까지의 경로를 기준으로:
 *   - 그 경로가 쉼터 옆(60m 이내)을 지나가면 = 실제로 가는 길 위. 경로를 따라 잰 거리가 도로 거리
 *   - 상·하행 구분: 같은 노선이고, 기준 휴게소 방향("부산")과 쉼터 종점("서울기점 + 부산종점")이 같은 쪽만
 *   → 기준 휴게소보다 먼 쉼터는 나오지 않지만, 그 경우 휴게소가 더 가까우므로 안내에는 문제 없음
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class NearbyRestPlaceService {

    // 1단계 (메모리)
    private static final double MAX_RADIUS_METERS = 50_000;     // 직선 50km 밖 제외
    private static final double MIN_SPEED_FOR_HEADING = 2.8;    // 약 10km/h 미만이면 GPS 방향값을 믿지 않음
    private static final double AHEAD_HALF_ANGLE = 90;          // 진행 방향 ±90도 = 앞쪽 (굽은 도로 여유)
    private static final int MIN_ROUTE_GROUPS = 4;              // 길찾기할 휴게소 그룹 최소 수 (쉼터 기준점 확보용)

    // 2단계 (카카오)
    private static final long ROUTE_DEADLINE_MS = 2_500;        // 길찾기 전체 대기 한도
    private static final double MAX_DETOUR_RATIO = 2.0;         // 도로/직선 비율 한도
    private static final int MIN_DETOUR_METERS = 3_000;         // 이만큼 이상 더 돌아갈 때만 비율 적용
    private static final double PASS_BY_RADIUS = 400;           // 목적지에 이 거리 안까지 다가갔다가
    private static final double PASS_BY_AWAY = 250;             // 다시 이만큼 멀어지면 = 지나쳤다가 되돌아옴
                                                                // (양재IC 처럼 바로 옆 IC 로 되돌아오면 약 300m 만 멀어짐)
    private static final double APPROACH_METERS = 600;          // 도착 직전 이 구간의 방향을
    private static final double REVERSED_ANGLE = 135;           // 진행 방향과 이 각도 넘게 다르면 = 회차해서 들어감 (반대 차선)
    private static final double REVERSED_CHECK_RADIUS = 15_000; // 도착 방향 반대 판정은 이 직선거리 안에서만
    private static final double TAIL_METERS = 1_000;            // 도착 직전 이 구간에서
    private static final double MAX_REVERSED_NEAR_END = 80;     // 거꾸로 이만큼 이상 가면 = 연결로 회차 (예: 중앙선 종점의 춘천(부산))
    private static final double MAX_PATH_OFFSET = 60;           // 쉼터 좌표가 경로에서 이 거리 안이면 "경로 위"
    private static final double ON_HIGHWAY_MAX_OFFSET = 40;     // 현재 위치가 고속도로 구간에서 이 거리 안이고
    private static final double ON_HIGHWAY_MAX_ANGLE = 45;      // 진행 방향이 이 각도 안이면 = 고속도로 위

    // 순환도로 (RingRoad)
    private static final double MAX_RING_AHEAD_METERS = 60_000; // 진행 방향으로 둘레 60km 안
    private static final double RING_SPEED_MPS = 22;            // 둘레 거리로 계산한 곳의 소요 시간용 (약 80km/h)

    private static final int SEARCH_LIMIT = 3;                // 내부 탐색 폭 (결과는 그중 가장 가까운 1곳)
    private static final Pattern SHELTER_DIRECTION = Pattern.compile("^(.+?)기점\\s*\\+\\s*(.+?)종점$");

    private final RestPlaceCache restPlaceCache;
    private final KakaoDirectionsClient kakaoDirectionsClient;

    // 요청 1건에 최대 (5+1)그룹 x 2 = 12 호출
    private final ExecutorService routeExecutor = Executors.newFixedThreadPool(16);

    @PreDestroy
    void shutdown() {
        routeExecutor.shutdownNow();
    }

    /** 가장 가까운 쉴 곳 1곳 (앱이 경유지로 추가) */
    public NearbyResponse findNearest(NearbyRequest req) {
        Search s = search(req, SEARCH_LIMIT);
        NearestPlace place = s.items().isEmpty() ? null : toNearestPlace(s.items().get(0));
        return new NearbyResponse(s.directionApplied(), s.onHighway(), s.distanceSource(), place);
    }

    /**
     * 내부 검색: 내 방향 쉴 곳을 도로 거리 순으로 최대 limit 곳.
     * (API 는 첫 번째만 쓰지만 탐색 폭은 limit 기준이라 검증된 값(3)으로 고정. 회귀 테스트는 더 많이 받아 반대 차선을 검사)
     */
    record Search(boolean directionApplied, Boolean onHighway, DistanceSource distanceSource, List<NearbyItem> items) {
    }

    Search search(NearbyRequest req, int limit) {
        boolean useHeading = req.getHeading() != null
                && req.getSpeed() != null
                && req.getSpeed() >= MIN_SPEED_FOR_HEADING;
        Double heading = useHeading ? req.getHeading() : null;

        // ===== 1단계: 반경 + 진행 방향 앞쪽 (종류 필터는 마지막에. 졸음쉼터만 원해도 휴게소 경로가 필요) =====
        List<Candidate> candidates = findCandidates(req, heading);
        List<Candidate> restAreas = candidates.stream().filter(c -> c.type() == PlaceType.REST_AREA).toList();
        List<Candidate> shelters = candidates.stream().filter(c -> c.type() == PlaceType.DROWSY_SHELTER).toList();
        if (restAreas.isEmpty()) {
            return new Search(useHeading, null, DistanceSource.ROAD, List.of());
        }

        // 상·하행 쌍을 한 그룹으로 묶어 가까운 그룹부터 (쌍이 자리를 둘씩 차지하지 않게)
        List<List<Candidate>> groups = groupPairs(restAreas).values().stream()
                .sorted(Comparator.comparingDouble(NearbyRestPlaceService::minStraightDistance))
                .toList();
        int batch = Math.max(limit + 1, MIN_ROUTE_GROUPS);

        // ===== 2단계: 길찾기 + 경로 모양으로 내 방향 판정 =====
        List<Routed> routed = new ArrayList<>(routeAll(req, heading, flatten(groups, 0, batch)));
        if (routed.isEmpty()) {
            log.warn("all directions failed: nearby search unavailable");
            return new Search(useHeading, null, DistanceSource.UNAVAILABLE, List.of());
        }

        // 고속도로 위가 아니면 안내하지 않음 (데이터가 고속도로 휴게소/쉼터뿐이고, 일반도로 상·하행 판정은 검증 안 됨)
        String currentRoad = currentRoad(routed, req, heading);
        if (currentRoad == null) {
            log.debug("nearby: not on highway ({}, {})", req.getLat(), req.getLng());
            return new Search(useHeading, false, DistanceSource.ROAD, List.of());
        }
        List<Anchor> ownSide = resolveOwnSide(routed, heading);

        // 2차: 내 방향 휴게소가 모자라거나 지금 달리는 노선의 휴게소를 아직 못 찾았으면 한 번 더.
        // 지금 달리는 노선(예: "경부")의 휴게소를 먼저, 그다음 가까운 순
        boolean foundOnCurrentRoad = ownSide.stream()
                .anyMatch(a -> normalizeRoute(a.candidate().place().getRouteName()).equals(currentRoad));
        List<List<Candidate>> remaining = groups.subList(Math.min(batch, groups.size()), groups.size());
        if ((ownSide.size() < limit || !foundOnCurrentRoad) && !remaining.isEmpty()) {
            List<List<Candidate>> next = remaining.stream()
                    .sorted(Comparator.comparing((List<Candidate> g) ->
                            !normalizeRoute(g.get(0).place().getRouteName()).equals(currentRoad)))
                    .limit(batch)
                    .toList();
            routed.addAll(routeAll(req, heading, flatten(next, 0, next.size())));
            ownSide = resolveOwnSide(routed, heading);
        }

        List<Result> results = new ArrayList<>();
        if (req.getType() == null || req.getType() == PlaceType.REST_AREA) {
            ownSide.forEach(a -> results.add(new Result(a.candidate(), a.roadDistance(), a.duration())));
        }
        if (req.getType() == null || req.getType() == PlaceType.DROWSY_SHELTER) {
            results.addAll(sheltersOnPath(shelters, ownSide));
        }

        // 순환도로 위: 같은 순환도로의 쉴 곳을 원 둘레 방향으로 판정해 우선 안내 (없으면 위 결과 그대로)
        RingRoad ring = heading == null ? null : RingRoad.of(currentRoad);
        List<Result> onRing = ring == null ? List.of() : ringPlaces(ring, req, heading, results);
        List<Result> chosen = onRing.isEmpty() ? results : onRing;

        List<NearbyItem> items = chosen.stream()
                .sorted(Comparator.comparingInt(Result::roadDistance))
                .limit(limit)
                .map(this::toItem)
                .toList();
        return new Search(useHeading, true, DistanceSource.ROAD, items);
    }

    /**
     * 지금 달리는 고속도로 (정규화된 이름, 예: "경부"). 고속도로 위가 아니면 null.
     * 1) 길찾기 출발 도로 중 가장 많이 나온 고속도로
     * 2) 없으면: 경로의 고속도로 구간이 현재 위치 40m 안을 진행 방향과 같은 방향으로 지나가는지.
     *    GPS 가 조금 어긋나면 카카오가 출발지를 옆 일반도로에 붙이는 경우가 있음
     *    (예: 경부고속도로 남사 부근 → "동탄기흥로"에서 출발해 기흥IC 로 다시 올라가는 경로, 2026-11-06 실측)
     */
    private String currentRoad(List<Routed> routed, NearbyRequest req, Double heading) {
        Map<String, Integer> count = new LinkedHashMap<>();
        for (Routed r : routed) {
            String name = r.route().firstRoadName();
            if (name != null && name.contains("고속도로")) {
                count.merge(normalizeRoute(name), 1, Integer::sum);
            }
        }
        if (count.isEmpty()) {
            for (Routed r : routed) {
                for (RoadPart part : r.route().highwayRoads()) {
                    if (passesBy(part.path(), req.getLat(), req.getLng(), heading)) {
                        count.merge(normalizeRoute(part.name()), 1, Integer::sum);
                    }
                }
            }
        }
        return count.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
    }

    /** 도로 구간 중 한 선분이 지점에서 ON_HIGHWAY_MAX_OFFSET 안이고, 진행 방향(있으면)과 같은 방향인지 */
    private static boolean passesBy(List<double[]> road, double lat, double lng, Double heading) {
        for (int i = 0; i + 1 < road.size(); i++) {
            double[] a = road.get(i);
            double[] b = road.get(i + 1);
            if (GeoUtils.closestOnPath(List.of(a, b), lat, lng)[0] > ON_HIGHWAY_MAX_OFFSET) {
                continue;
            }
            if (heading == null) {
                return true;
            }
            double bearing = GeoUtils.bearingDegrees(a[1], a[0], b[1], b[0]);
            if (Math.abs(GeoUtils.relativeAngle(bearing, heading)) <= ON_HIGHWAY_MAX_ANGLE) {
                return true;
            }
        }
        return false;
    }

    private static List<Candidate> flatten(List<List<Candidate>> groups, int from, int to) {
        return groups.subList(Math.min(from, groups.size()), Math.min(to, groups.size())).stream()
                .flatMap(List::stream)
                .toList();
    }

    /** 1단계: 반경·진행 방향 앞쪽 필터 */
    private List<Candidate> findCandidates(NearbyRequest req, Double heading) {
        List<Candidate> result = new ArrayList<>();
        for (RestPlace place : restPlaceCache.getAll()) {
            double distance = GeoUtils.distanceMeters(
                    req.getLat(), req.getLng(), place.getLatitude(), place.getLongitude());
            if (distance > MAX_RADIUS_METERS) {
                continue;
            }
            if (heading != null) {
                double bearing = GeoUtils.bearingDegrees(
                        req.getLat(), req.getLng(), place.getLatitude(), place.getLongitude());
                if (Math.abs(GeoUtils.relativeAngle(bearing, heading)) > AHEAD_HALF_ANGLE) {
                    continue;   // 이미 지나친 곳
                }
            }
            result.add(new Candidate(place, distance));
        }
        return result;
    }

    /** 2단계: 휴게소마다 길찾기를 동시에 호출. 제한 시간 안에 끝난 것만 사용 */
    private List<Routed> routeAll(NearbyRequest req, Double heading, List<Candidate> targets) {
        List<CompletableFuture<Optional<Route>>> futures = targets.stream()
                .map(c -> CompletableFuture.supplyAsync(() -> kakaoDirectionsClient.route(
                        req.getLat(), req.getLng(), heading,
                        c.place().getLatitude(), c.place().getLongitude()), routeExecutor))
                .toList();

        try {
            CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new))
                    .get(ROUTE_DEADLINE_MS, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            log.warn("directions deadline exceeded, use finished results only");
        }

        List<Routed> routed = new ArrayList<>();
        for (int i = 0; i < targets.size(); i++) {
            CompletableFuture<Optional<Route>> f = futures.get(i);
            if (f.isDone() && !f.isCompletedExceptionally()) {
                Optional<Route> route = f.getNow(Optional.empty());
                if (route.isPresent() && !route.get().path().isEmpty()) {
                    routed.add(new Routed(targets.get(i), route.get()));
                }
            } else {
                f.cancel(true);
            }
        }
        return routed;
    }

    private enum Access { DIRECT, UTURN, BACKROAD, UNKNOWN, DETOUR }

    /**
     * 경로 모양으로 접근 방식 판정 (클래스 주석 참고)
     *
     * @param heading 진행 방향 (없으면 경로 처음 구간의 방향을 기준으로 씀)
     */
    private Access access(Routed r, Double heading) {
        RestPlace place = r.candidate().place();
        List<double[]> path = r.route().path();
        double passAlong = GeoUtils.passesThenReturns(path,
                place.getLatitude(), place.getLongitude(), PASS_BY_RADIUS, PASS_BY_AWAY);
        if (passAlong >= 0) {
            return r.route().endsOnHighway() ? Access.UTURN : Access.BACKROAD;
        }

        // 도착 직전 방향이 내 진행 방향과 반대 = 어딘가에서 회차함 (예: 중앙선 종점 춘천JC 연결로를 돌아 춘천(부산)휴게소)
        // (먼 곳은 도로가 크게 굽어 방향이 바뀔 수 있어 직선 15km 이내에만 적용)
        if (r.candidate().straightDistance() <= REVERSED_CHECK_RADIUS && arrivesReversed(r, heading)) {
            return r.route().endsOnHighway() ? Access.UTURN : Access.UNKNOWN;
        }

        // 도착 직전 1km 안에서 진입 방향과 반대로 80m 이상 진행 = 연결로를 돌아 거꾸로 들어감 → 방향 확인 불가
        if (GeoUtils.reversedLengthNearEnd(path, TAIL_METERS, 2_000, REVERSED_ANGLE) >= MAX_REVERSED_NEAR_END) {
            return Access.UNKNOWN;
        }

        double straight = r.candidate().straightDistance();
        int road = r.route().distanceMeters();
        boolean bigDetour = road - straight >= MIN_DETOUR_METERS && road > straight * MAX_DETOUR_RATIO;
        return bigDetour ? Access.DETOUR : Access.DIRECT;
    }

    /** 도착 직전 600m 진행 방향이 내 진행 방향(없으면 경로 처음 방향)과 135° 넘게 반대인지 */
    private boolean arrivesReversed(Routed r, Double heading) {
        List<double[]> path = r.route().path();
        double reference = heading != null ? heading : GeoUtils.pathBearing(path, APPROACH_METERS, false);
        double approach = GeoUtils.pathBearing(path, APPROACH_METERS, true);
        return !Double.isNaN(reference) && !Double.isNaN(approach)
                && Math.abs(GeoUtils.relativeAngle(approach, reference)) > REVERSED_ANGLE;
    }

    /**
     * 휴게소 그룹(상·하행)마다 내 방향 쪽 하나를 고른다. 확인이 안 되는 그룹은 버린다.
     * 결과는 휴게소 결과이자 졸음쉼터 판정의 기준(Anchor)으로 쓰인다.
     */
    private List<Anchor> resolveOwnSide(List<Routed> routed, Double heading) {
        Map<String, List<Routed>> groups = new LinkedHashMap<>();
        for (Routed r : routed) {
            groups.computeIfAbsent(pairKey(r.candidate().place()), k -> new ArrayList<>()).add(r);
        }

        List<Anchor> anchors = new ArrayList<>();
        Map<String, Set<String>> oppositeLabels = new LinkedHashMap<>();   // 노선 → 반대 차선으로 판정된 휴게소 표기
        for (List<Routed> group : groups.values()) {
            Map<Routed, Access> accessOf = new LinkedHashMap<>();
            group.forEach(r -> accessOf.put(r, access(r, heading)));
            if (log.isDebugEnabled()) {
                accessOf.forEach((r, a) -> log.debug("nearby: {} {} straight={}m road={}m start={}", a,
                        r.candidate().place().getName(), Math.round(r.candidate().straightDistance()),
                        r.route().distanceMeters(), r.route().firstRoadName()));
            }
            accessOf.forEach((r, access) -> {
                String label = r.candidate().place().getDirection();
                // 이미 지나친 내 방향 휴게소도 UTURN 이지만(IC 두 번 회차) 같은 방향으로 도착하므로 반대 차선 표기에서 제외
                if (access == Access.UTURN && label != null && arrivesReversed(r, heading)) {
                    oppositeLabels.computeIfAbsent(r.candidate().place().getRouteName(), k -> new HashSet<>()).add(label);
                }
            });

            // 1) 바로 도착하는 쪽이 있으면 그중 가장 가까운 것
            Optional<Routed> direct = group.stream()
                    .filter(r -> accessOf.get(r) == Access.DIRECT)
                    .min(Comparator.comparingInt(r -> r.route().distanceMeters()));
            // 반대 차선 = UTURN 이면서 반대 방향으로 도착 (이미 지나친 내 방향 휴게소는 제외)
            boolean hasUturn = group.stream().anyMatch(r -> accessOf.get(r) == Access.UTURN && arrivesReversed(r, heading));
            if (direct.isPresent()) {
                Routed r = direct.get();
                // 같은 휴게소의 다른 쪽이 반대 차선으로 확인됐으면 방향 표기("부산")를 믿을 수 있음
                anchors.add(new Anchor(r.candidate(), r.route(), r.route().distanceMeters(),
                        r.route().durationSeconds(), Double.MAX_VALUE, hasUturn));
                continue;
            }

            // 2) 뒷길 도착 1개 + 나머지는 모두 반대 차선으로 확인 → 뒷길 쪽이 내 방향
            List<Routed> backroads = group.stream().filter(r -> accessOf.get(r) == Access.BACKROAD).toList();
            boolean othersAreUturn = group.stream()
                    .filter(r -> accessOf.get(r) != Access.BACKROAD)
                    .allMatch(r -> accessOf.get(r) == Access.UTURN && arrivesReversed(r, heading));
            if (backroads.size() == 1 && othersAreUturn && hasUturn) {
                Routed r = backroads.get(0);
                RestPlace place = r.candidate().place();
                double passAlong = GeoUtils.passesThenReturns(r.route().path(),
                        place.getLatitude(), place.getLongitude(), PASS_BY_RADIUS, PASS_BY_AWAY);
                int road = (int) Math.round(passAlong);
                anchors.add(new Anchor(r.candidate(), r.route(), road, scaledDuration(r.route(), road), passAlong, true));
            }
            // 3) 그 밖에는 방향을 확신할 수 없으므로 내보내지 않음
        }
        return keepConsistentDirections(anchors, oppositeLabels);
    }

    /**
     * 노선마다 "내 방향 표기"를 정하고, 그와 어긋나는 휴게소를 제거.
     * 카카오가 종점 JC 연결로 등으로 반대쪽 휴게소에 "바로" 들어가는 경로를 주는 경우가 있음
     * (예: 중앙선 춘천 방향 주행 중 춘천(부산)휴게소, 2026-11-06 실측).
     *
     * 노선별 내 방향 표기:
     * 1) 확인된 휴게소(반대편이 반대 차선으로 판정됨)의 표기. 확인된 표기끼리 충돌하면 그 노선은 판단 불가
     *    → 방향 표기 있는 휴게소 모두 제외
     * 2) 방향 표기가 2가지뿐인 노선(경부선: 부산/서울)에서 한쪽 표기가 반대 차선으로 판정됐으면 나머지 표기
     * 3) 확인된 게 없으면: 바로 도착하는 휴게소들의 표기가 모두 같고 2곳 이상이면 그 표기 (서로 독립인 두 곳이 일치)
     * 4) 표기가 서로 다르면 가장 가까운 휴게소의 표기 (먼 쪽은 어딘가에서 회차해야 들어가는 곳)
     * 1)~3) 으로 정해진 경우에만 confirmed = true (졸음쉼터 판정 기준으로 사용 가능).
     * 내 방향 표기가 반대 차선 표기와 같으면 모순이므로 그 노선의 방향 표기 휴게소는 모두 제외.
     * 양방향 공용 등 표기 없는 휴게소는 항상 유지.
     */
    private List<Anchor> keepConsistentDirections(List<Anchor> anchors, Map<String, Set<String>> oppositeLabels) {
        Map<String, List<Anchor>> byRoute = new LinkedHashMap<>();
        for (Anchor a : anchors) {
            byRoute.computeIfAbsent(a.candidate().place().getRouteName(), k -> new ArrayList<>()).add(a);
        }

        List<Anchor> result = new ArrayList<>();
        for (Map.Entry<String, List<Anchor>> entry : byRoute.entrySet()) {
            List<Anchor> routeAnchors = entry.getValue();
            List<Anchor> labeled = routeAnchors.stream().filter(a -> a.candidate().place().getDirection() != null).toList();
            routeAnchors.stream().filter(a -> a.candidate().place().getDirection() == null).forEach(result::add);
            if (labeled.isEmpty()) {
                continue;
            }

            Set<String> confirmedLabels = new HashSet<>();
            labeled.stream().filter(Anchor::confirmed).forEach(a -> confirmedLabels.add(a.candidate().place().getDirection()));
            Set<String> allLabels = new HashSet<>();
            labeled.forEach(a -> allLabels.add(a.candidate().place().getDirection()));
            Set<String> opposite = oppositeLabels.getOrDefault(entry.getKey(), Set.of());
            Set<String> routeLabels = restPlaceCache.restAreaDirections(entry.getKey());

            String myLabel;
            boolean trusted;
            if (confirmedLabels.isEmpty() && routeLabels.size() == 2 && opposite.size() == 1
                    && routeLabels.containsAll(opposite)) {
                // 2) 방향이 둘뿐인 노선에서 한쪽이 반대 차선 → 나머지가 내 방향
                myLabel = routeLabels.stream().filter(l -> !opposite.contains(l)).findFirst().orElseThrow();
                trusted = true;
            } else if (!confirmedLabels.isEmpty()) {
                if (confirmedLabels.size() > 1) {
                    continue;                                   // 1) 확인된 표기끼리 충돌 → 판단 불가
                }
                myLabel = confirmedLabels.iterator().next();
                trusted = true;
            } else if (allLabels.size() == 1) {
                myLabel = allLabels.iterator().next();
                trusted = labeled.size() >= 2;                  // 2) 두 곳 이상이 일치
            } else {
                myLabel = labeled.stream()                      // 3) 가장 가까운 곳 기준
                        .min(Comparator.comparingInt(Anchor::roadDistance))
                        .orElseThrow().candidate().place().getDirection();
                trusted = false;
            }
            if (opposite.contains(myLabel)) {
                continue;   // 내 방향 표기가 반대 차선 표기와 같음 = 모순 → 판단 불가
            }

            for (Anchor a : labeled) {
                if (a.candidate().place().getDirection().equals(myLabel)) {
                    result.add(new Anchor(a.candidate(), a.route(), a.roadDistance(), a.duration(),
                            a.mainlineUntil(), a.confirmed() || trusted));
                }
            }
        }
        return result;
    }

    /** 졸음쉼터: 내 방향으로 확인된 휴게소까지의 경로 위에 있고, 그 휴게소와 같은 방향인 쉼터만 */
    private List<Result> sheltersOnPath(List<Candidate> shelters, List<Anchor> anchors) {
        Map<Long, Result> best = new LinkedHashMap<>();
        for (Anchor anchor : anchors) {
            RestPlace area = anchor.candidate().place();
            String areaDirection = area.getDirection();          // "부산" (이름 괄호에서 추출). 양방향 공용이면 null
            if (areaDirection == null || !anchor.confirmed()) {   // 방향이 확인된 휴게소만 기준으로 씀
                continue;
            }
            String areaRoute = normalizeRoute(area.getRouteName());

            for (Candidate shelter : shelters) {
                RestPlace place = shelter.place();
                String destination = shelterDestination(place.getDirection());
                if (destination == null
                        || !destination.equals(areaDirection)
                        || !normalizeRoute(place.getRouteName()).equals(areaRoute)) {
                    continue;
                }
                // 뒷길로 빠지기 전(본선) 구간에서만 찾음
                double[] onPath = GeoUtils.closestOnPath(anchor.route().path(),
                        place.getLatitude(), place.getLongitude(), anchor.mainlineUntil());
                if (onPath[0] > MAX_PATH_OFFSET || onPath[1] > anchor.mainlineUntil()) {
                    continue;
                }
                int roadDistance = (int) Math.round(onPath[1]);
                best.merge(place.getPlaceId(), new Result(shelter, roadDistance, scaledDuration(anchor.route(), roadDistance)),
                        (a, b) -> a.roadDistance() <= b.roadDistance() ? a : b);
            }
        }
        return new ArrayList<>(best.values());
    }

    /**
     * 순환도로 위에서 같은 순환도로의 내 방향 쉴 곳 (휴게소 + 졸음쉼터).
     * - 방향: 장소의 회전 == 차가 도는 회전 (RingRoad. 휴게소와 졸음쉼터는 방향 표시의 뜻이 달라 따로 판정)
     * - 거리: 카카오로 확인된 휴게소는 그 도로 거리, 나머지는 원 둘레를 따라 잰 거리
     * - 카카오 판정 결과에 있는데 회전이 반대면 = 두 판정이 엇갈림 → 안전하게 제외하고 기록
     * - 방향을 판정할 수 없는 곳(방향 표시 없음, 성남 쉼터처럼 모르는 표시)은 제외
     */
    private List<Result> ringPlaces(RingRoad ring, NearbyRequest req, double heading, List<Result> verified) {
        double lat = req.getLat();
        double lng = req.getLng();
        RingRoad.Rotation rotation = ring.travelRotation(lat, lng, heading);
        Map<Long, Result> byKakao = new LinkedHashMap<>();
        verified.forEach(r -> byKakao.put(r.candidate().place().getPlaceId(), r));

        List<Result> result = new ArrayList<>();
        for (RestPlace place : restPlaceCache.getAll()) {
            if (!ring.contains(normalizeRoute(place.getRouteName()))
                    || (req.getType() != null && req.getType() != place.getPlaceType())) {
                continue;
            }
            Result kakao = byKakao.get(place.getPlaceId());
            RingRoad.Rotation placeRotation;
            if (place.getPlaceType() == PlaceType.REST_AREA) {
                placeRotation = ring.restAreaRotation(place.getLatitude(), place.getLongitude(), place.getDirection());
            } else {
                Matcher m = place.getDirection() == null ? null : SHELTER_DIRECTION.matcher(place.getDirection().trim());
                placeRotation = m != null && m.matches() ? ring.shelterRotation(m.group(1), m.group(2)) : null;
            }
            if (placeRotation == null) {
                // 방향 표시가 없거나 모르는 표시 → 제외. 카카오가 "갈 수 있다"고 해도 돌아서 들어가는 경로일 수 있음
                // (예: 서하남행복드림쉼터는 시계 차선에만 있는데, 반시계 주행에서 둘레 54km 경로로 확인됨, 2026-11-06)
                continue;
            }
            if (placeRotation != rotation) {
                if (kakao != null) {
                    log.info("ring: direction conflict, dropped placeId={} ({}, {}) at {},{} heading={}",
                            place.getPlaceId(), place.getName(), place.getDirection(), lat, lng, heading);
                }
                continue;
            }
            double ahead = ring.aheadMeters(lat, lng, place.getLatitude(), place.getLongitude(), rotation);
            if (ahead < 0 || ahead > MAX_RING_AHEAD_METERS) {
                continue;
            }
            if (kakao != null) {
                result.add(kakao);
            } else {
                int meters = (int) Math.round(ahead);
                double straight = GeoUtils.distanceMeters(lat, lng, place.getLatitude(), place.getLongitude());
                result.add(new Result(new Candidate(place, straight), meters, (int) Math.round(meters / RING_SPEED_MPS)));
            }
        }
        return result;
    }

    /** 경로의 평균 속도로 일부 구간 소요 시간 계산 */
    private int scaledDuration(Route route, int meters) {
        return (int) Math.round((double) route.durationSeconds() * meters / Math.max(1, route.distanceMeters()));
    }

    /** "서울기점 + 부산종점" → "부산" (형식이 다르면 null → 제외) */
    private String shelterDestination(String direction) {
        if (direction == null) {
            return null;
        }
        Matcher m = SHELTER_DIRECTION.matcher(direction.trim());
        return m.matches() ? m.group(2).trim() : null;
    }

    /** 두 데이터의 노선명 표기 맞추기: "남해선(순천-부산)" / "남해선" → "남해" */
    private String normalizeRoute(String routeName) {
        if (routeName == null) {
            return "";
        }
        String s = routeName.replaceAll("\\([^)]*\\)", "").replace("고속도로", "").replace(" ", "");
        return s.endsWith("선") ? s.substring(0, s.length() - 1) : s;
    }

    private Map<String, List<Candidate>> groupPairs(List<Candidate> candidates) {
        Map<String, List<Candidate>> groups = new LinkedHashMap<>();
        for (Candidate c : candidates) {
            groups.computeIfAbsent(pairKey(c.place()), k -> new ArrayList<>()).add(c);
        }
        return groups;
    }

    /** 휴게소는 "죽전(서울)휴게소"/"죽전(부산)휴게소" 처럼 이름에 방향이 있어 괄호를 빼고 비교 */
    private String pairKey(RestPlace place) {
        return place.getPlaceType() + "|" + place.getName().replaceAll("\\([^)]*\\)", "")
                + "|" + place.getRouteName();
    }

    private static double minStraightDistance(List<Candidate> group) {
        return group.stream().mapToDouble(Candidate::straightDistance).min().orElse(Double.MAX_VALUE);
    }

    private NearbyItem toItem(Result r) {
        RestPlace place = r.candidate().place();
        return NearbyItem.builder()
                .placeId(place.getPlaceId())
                .type(place.getPlaceType())
                .name(place.getName())
                .routeName(place.getRouteName())
                .direction(place.getDirection())
                .latitude(place.getLatitude())
                .longitude(place.getLongitude())
                .parkingSpaces(place.getParkingSpaces())
                .straightDistanceMeters((int) Math.round(r.candidate().straightDistance()))
                .roadDistanceMeters(r.roadDistance())
                .durationSeconds(r.duration())
                .build();
    }

    private static NearestPlace toNearestPlace(NearbyItem item) {
        return NearestPlace.builder()
                .placeId(item.getPlaceId())
                .type(item.getType())
                .name(item.getName())
                .latitude(item.getLatitude())
                .longitude(item.getLongitude())
                .roadDistanceMeters(item.getRoadDistanceMeters())
                .durationSeconds(item.getDurationSeconds())
                .build();
    }

    private record Candidate(RestPlace place, double straightDistance) {
        PlaceType type() {
            return place.getPlaceType();
        }
    }

    private record Routed(Candidate candidate, Route route) {
    }

    /**
     * 내 방향으로 확인된 휴게소.
     * @param mainlineUntil 경로 중 본선으로 믿을 수 있는 구간 끝(m). 뒷길 도착이면 휴게소 옆을 지나는 지점까지
     * @param confirmed     같은 휴게소의 반대편이 반대 차선으로 판정돼 방향 표기를 믿을 수 있음
     */
    private record Anchor(Candidate candidate, Route route, int roadDistance, int duration, double mainlineUntil,
                          boolean confirmed) {
    }

    private record Result(Candidate candidate, int roadDistance, int duration) {
    }
}