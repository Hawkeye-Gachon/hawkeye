package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dao.RestPlaceDao;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyItem;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyRequest;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.DeserializationFeature;
import tools.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * 주변 조회 회귀 테스트용 고정 데이터.
 * - rest_places.json      : DB 의 활성 휴게소/졸음쉼터 (테스트는 DB 없이 이 목록으로 동작)
 * - nearby/{이름}.json.gz : 시나리오 1건 = 요청 + 그때 받은 카카오 길찾기 응답들 + 기대 결과
 * 카카오 응답은 NearbyFixtureRecorder 로 한 번만 녹화하고, 테스트는 그것을 재생하므로 카카오를 호출하지 않는다.
 */
final class NearbyFixtures {

    static final JsonMapper JSON = JsonMapper.builder()
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)   // 예전 파일의 request.limit 등
            .build();
    static final Path SCENARIO_DIR = Path.of("src/test/resources/restplace/nearby");

    private static final Pattern SHELTER_DIRECTION = Pattern.compile("^(.+?)기점\\s*\\+\\s*(.+?)종점$");

    private NearbyFixtures() {
    }

    /**
     * @param route   지금 달리는 노선 (정규화된 이름, 예: "경부"). null 이면 방향 검사 안 함
     * @param allowed 그 노선에서 나와도 되는 방향 표기 (예: ["서울"])
     * @param limit   내부 검색 폭 (API 는 3 으로 검색해 첫 번째만 반환. 주행 표본은 5 로 더 많은 결과를 검사)
     * @param kakao   "출발|도착" → 카카오 원본 응답 (경로 없음/실패는 null)
     * @param onHighway 기대 결과의 고속도로 위 여부
     * @param expected 기대 결과 (describe 형식)
     */
    record Scenario(String name, NearbyRequest request, Integer limit, String route, List<String> allowed,
                    Map<String, Map<String, Object>> kakao, Boolean onHighway, List<String> expected) {
    }

    // ===== 쉼터 목록 =====

    static RestPlaceCache loadCache() {
        List<RestPlace> places;
        try (InputStream in = NearbyFixtures.class.getResourceAsStream("/restplace/rest_places.json")) {
            places = JSON.readValue(in, new TypeReference<List<RestPlace>>() {});
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        RestPlaceCache cache = new RestPlaceCache(new FixedDao(places));
        cache.reload();
        return cache;
    }

    private record FixedDao(List<RestPlace> places) implements RestPlaceDao {
        @Override
        public List<RestPlace> findAllActive() {
            return places;
        }

        @Override
        public int upsert(RestPlace row, OffsetDateTime syncedAt) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int deactivateNotSynced(RestPlace.PlaceType placeType, OffsetDateTime syncedAt) {
            throw new UnsupportedOperationException();
        }
    }

    // ===== 카카오 재생 =====

    /** 녹화된 응답을 돌려주는 가짜 길찾기 클라이언트. 녹화에 없는 요청은 기록해 두고 실패(null) 처리 */
    static class ReplayKakaoClient extends KakaoDirectionsClient {
        private final Map<String, Map<String, Object>> responses;
        final Set<String> missing = Collections.synchronizedSet(new LinkedHashSet<>());
        final List<String> requested = Collections.synchronizedList(new ArrayList<>());

        ReplayKakaoClient(Map<String, Map<String, Object>> responses) {
            super("replay", 1);   // 키가 있어야 fetch 까지 옴. 실제 HTTP 는 호출하지 않음
            this.responses = responses;
        }

        @Override
        protected Map<String, Object> fetch(String origin, String destination) {
            String key = key(origin, destination);
            requested.add(key);
            if (!responses.containsKey(key)) {
                missing.add(key);
                return null;
            }
            return responses.get(key);
        }
    }

    static String key(String origin, String destination) {
        return origin + "|" + destination;
    }

    // ===== 결과 검사 =====

    static String describe(NearbyItem item) {
        return item.getType() + " " + item.getName() + " [" + item.getDirection() + "] " + item.getRoadDistanceMeters() + "m";
    }

    /** 지금 노선의 결과 중 허용되지 않은 방향(= 반대 차선) 표기가 있으면 그 설명 */
    static List<String> wrongDirection(Scenario s, List<NearbyItem> items) {
        List<String> wrong = new ArrayList<>();
        if (s.route() == null) {
            return wrong;
        }
        for (NearbyItem item : items) {
            if (!normalizeRoute(item.getRouteName()).equals(s.route()) || item.getDirection() == null) {
                continue;   // 다른 노선 / 양방향 공용
            }
            String label = item.getType() == RestPlace.PlaceType.REST_AREA
                    ? item.getDirection()
                    : shelterDestination(item.getDirection());
            if (!s.allowed().contains(label)) {
                wrong.add(describe(item));
            }
        }
        return wrong;
    }

    static String shelterDestination(String direction) {
        Matcher m = SHELTER_DIRECTION.matcher(direction.trim());
        return m.matches() ? m.group(2).trim() : direction;
    }

    static String normalizeRoute(String routeName) {
        if (routeName == null) {
            return "";
        }
        String s = routeName.replaceAll("\\([^)]*\\)", "").replace("고속도로", "").replace(" ", "");
        return s.endsWith("선") ? s.substring(0, s.length() - 1) : s;
    }

    // ===== 파일 =====

    static List<Path> scenarioFiles() throws IOException {
        if (!Files.isDirectory(SCENARIO_DIR)) {
            return List.of();
        }
        try (var files = Files.list(SCENARIO_DIR)) {
            return files.filter(p -> p.toString().endsWith(".json.gz")).sorted().toList();
        }
    }

    static Path fileOf(String name) {
        return SCENARIO_DIR.resolve(name + ".json.gz");
    }

    static Scenario read(Path file) throws IOException {
        try (InputStream in = new GZIPInputStream(Files.newInputStream(file))) {
            return JSON.readValue(in, Scenario.class);
        }
    }

    static void write(Scenario s) throws IOException {
        Files.createDirectories(SCENARIO_DIR);
        try (OutputStream out = new GZIPOutputStream(Files.newOutputStream(fileOf(s.name())))) {
            JSON.writeValue(out, s);
        }
    }

    /** 카카오 응답에서 판정에 쓰는 값만 남김 (파일 크기 줄이기) */
    @SuppressWarnings("unchecked")
    static Map<String, Object> trim(Map<String, Object> body) {
        if (body == null || !(body.get("routes") instanceof List<?> routes) || routes.isEmpty()) {
            return body;
        }
        Map<String, Object> route = (Map<String, Object>) routes.get(0);
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("result_code", route.get("result_code"));
        r.put("result_msg", route.get("result_msg"));
        if (route.get("summary") instanceof Map<?, ?> summary) {
            r.put("summary", Map.of("distance", summary.get("distance"), "duration", summary.get("duration")));
        }
        if (route.get("sections") instanceof List<?> sections) {
            List<Object> outSections = new ArrayList<>();
            for (Object section : sections) {
                List<Object> outRoads = new ArrayList<>();
                if (((Map<String, Object>) section).get("roads") instanceof List<?> roads) {
                    for (Object road : roads) {
                        Map<String, Object> rd = (Map<String, Object>) road;
                        Map<String, Object> o = new LinkedHashMap<>();
                        o.put("name", rd.get("name"));
                        o.put("vertexes", rd.get("vertexes"));
                        outRoads.add(o);
                    }
                }
                outSections.add(Map.of("roads", outRoads));
            }
            r.put("sections", outSections);
        }
        return Map.of("routes", List.of(r));
    }

    static List<String> labels(String csv) {
        return Arrays.asList(csv.split(","));
    }
}
