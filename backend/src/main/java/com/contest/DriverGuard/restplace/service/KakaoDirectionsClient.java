package com.contest.DriverGuard.restplace.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestClient;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;

/** 카카오모빌리티 자동차 길찾기: 실제 도로 거리·소요 시간·경로 좌표 */
@Slf4j
@Component
public class KakaoDirectionsClient {

    /**
     * @param path          경로 좌표 [경도, 위도] 순서대로 (출발 → 도착)
     * @param endsOnHighway 마지막 도로가 고속도로인지 (false 면 일반도로 = 뒷길로 도착)
     * @param firstRoadName 출발 지점 도로 이름 (예: "경부고속도로", 이름 없는 연결로는 건너뜀) = 지금 달리는 도로
     * @param highwayRoads  경로 중 고속도로 구간들 (이름 + 좌표)
     */
    public record Route(int distanceMeters, int durationSeconds, List<double[]> path, boolean endsOnHighway,
                        String firstRoadName, List<RoadPart> highwayRoads) {
    }

    public record RoadPart(String name, List<double[]> path) {
    }

    private static final long RATE_LIMIT_RETRY_DELAY_MS = 300;

    private static final ParameterizedTypeReference<Map<String, Object>> MAP_TYPE =
            new ParameterizedTypeReference<>() {};

    private final RestClient restClient;
    private final boolean enabled;

    public KakaoDirectionsClient(
            @Value("${driverguard.kakao.rest-api-key}") String apiKey,
            @Value("${driverguard.kakao.directions-timeout-ms:1500}") long timeoutMs) {
        this.enabled = apiKey != null && !apiKey.isBlank();
        if (!enabled) {
            log.warn("driverguard.kakao.rest-api-key is not set: nearby search is unavailable");
        }

        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofMillis(timeoutMs));
        factory.setReadTimeout(Duration.ofMillis(timeoutMs));

        this.restClient = RestClient.builder()
                .baseUrl("https://apis-navi.kakaomobility.com")
                .requestFactory(factory)
                .defaultHeader("Authorization", "KakaoAK " + apiKey)
                .build();
    }

    /**
     * @param heading 진행 방향(0~360). 주면 출발지에서 그 방향으로 출발하는 경로로 탐색 (null 이면 방향 없음)
     */
    public Optional<Route> route(double originLat, double originLng, Double heading,
                                 double destLat, double destLng) {
        // 카카오 좌표 순서: 경도,위도
        String origin = originLng + "," + originLat
                + (heading != null ? ",angle=" + (Math.round(heading) % 360) : "");
        String destination = destLng + "," + destLat;

        try {
            Optional<Route> route = parse(fetch(origin, destination));
            if (route.isEmpty()) {
                log.debug("kakao directions no route: {} -> {}", origin, destination);
            }
            return route;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return Optional.empty();
        } catch (Exception e) {
            log.warn("kakao directions failed: {} -> {}: {}", origin, destination, e.toString());
            return Optional.empty();
        }
    }

    /**
     * 길찾기 API 원본 응답. 키가 없으면 null.
     * 테스트에서 저장해 둔 응답으로 바꿔 끼울 수 있도록 분리 (NearbyRestPlaceServiceTest).
     */
    protected Map<String, Object> fetch(String origin, String destination) throws InterruptedException {
        if (!enabled) {
            return null;
        }
        try {
            return call(origin, destination);
        } catch (HttpStatusCodeException e) {
            if (!isRateLimited(e)) {
                throw e;
            }
            // 순간 호출 한도 초과 ({"code":-10,"msg":"API limit has been exceeded."}): 잠깐 쉬고 한 번만 다시
            Thread.sleep(RATE_LIMIT_RETRY_DELAY_MS + ThreadLocalRandom.current().nextInt(200));
            return call(origin, destination);
        }
    }

    /** 길찾기 원본 응답 → Route (응답이 없거나 경로를 못 찾았으면 empty) */
    @SuppressWarnings("unchecked")
    static Optional<Route> parse(Map<String, Object> body) {
        if (body == null || !(body.get("routes") instanceof List<?> routes) || routes.isEmpty()) {
            return Optional.empty();
        }
        Map<String, Object> route = (Map<String, Object>) routes.get(0);
        if (!(route.get("result_code") instanceof Number code) || code.intValue() != 0) {
            return Optional.empty();
        }

        Map<String, Object> summary = (Map<String, Object>) route.get("summary");
        int distance = ((Number) summary.get("distance")).intValue();
        int duration = ((Number) summary.get("duration")).intValue();

        // sections[].roads[] = {name, vertexes: [x1, y1, x2, y2, ...]}
        List<double[]> path = new ArrayList<>();
        List<RoadPart> highwayRoads = new ArrayList<>();
        String first = "";
        String last = "";
        if (route.get("sections") instanceof List<?> sections) {
            for (Object section : sections) {
                if (!(((Map<String, Object>) section).get("roads") instanceof List<?> roads)) {
                    continue;
                }
                for (Object r : roads) {
                    Map<String, Object> road = (Map<String, Object>) r;
                    String name = road.get("name") == null ? "" : road.get("name").toString();
                    if (first.isEmpty()) {
                        first = name;
                    }
                    last = name;
                    List<double[]> vertexes = vertexes(road);
                    for (double[] p : vertexes) {
                        double[] prev = path.isEmpty() ? null : path.get(path.size() - 1);
                        if (prev == null || prev[0] != p[0] || prev[1] != p[1]) {
                            path.add(p);
                        }
                    }
                    if (name.contains("고속도로") && vertexes.size() >= 2) {
                        highwayRoads.add(new RoadPart(name, vertexes));
                    }
                }
            }
        }
        return Optional.of(new Route(distance, duration, path, last.contains("고속도로"), first, highwayRoads));
    }

    private static List<double[]> vertexes(Map<String, Object> road) {
        List<double[]> result = new ArrayList<>();
        if (road.get("vertexes") instanceof List<?> v) {
            for (int i = 0; i + 1 < v.size(); i += 2) {
                result.add(new double[]{((Number) v.get(i)).doubleValue(), ((Number) v.get(i + 1)).doubleValue()});
            }
        }
        return result;
    }

    private Map<String, Object> call(String origin, String destination) {
        return restClient.get()
                .uri(uri -> uri.path("/v1/directions")
                        .queryParam("origin", origin)
                        .queryParam("destination", destination)
                        .queryParam("priority", "RECOMMEND")
                        .build())
                .retrieve()
                .body(MAP_TYPE);
    }

    private boolean isRateLimited(HttpStatusCodeException e) {
        return e.getStatusCode().value() == 429 || e.getResponseBodyAsString().contains("\"code\":-10");
    }
}
