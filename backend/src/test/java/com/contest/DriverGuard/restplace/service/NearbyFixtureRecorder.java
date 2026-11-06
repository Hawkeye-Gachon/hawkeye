package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyItem;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyRequest;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace.PlaceType;
import com.contest.DriverGuard.restplace.service.NearbyFixtures.ReplayKakaoClient;
import com.contest.DriverGuard.restplace.service.NearbyFixtures.Scenario;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 회귀 테스트용 카카오 길찾기 응답 녹화. <b>실제 카카오 API 를 호출하므로 평소 빌드에서는 실행되지 않음.</b>
 * <pre>
 *   ./mvnw test -Dtest=NearbyFixtureRecorder -Dnearby.record=true                 # 없는 시나리오만 녹화
 *   ./mvnw test -Dtest=NearbyFixtureRecorder -Dnearby.record=true -Dnearby.only=gb_n_osan   # 특정 시나리오 다시 녹화
 *   ./mvnw test -Dtest=NearbyFixtureRecorder -Dnearby.rebaseline=true             # 카카오 호출 없이 기대 결과만 갱신
 * </pre>
 * 시나리오 1건당 길찾기 약 4~16회. 기대 결과(expected)는 녹화된 응답을 재생한 결과로 저장되므로,
 * 출력되는 결과(특히 WRONG)를 사람이 확인해야 함. rebaseline 은 판정 로직을 의도적으로 바꿨을 때만 사용.
 * 좌표: 2026-11-06 고속도로 12개 방향 주행 시뮬레이션(Corridor) 표본 + 문제가 됐던 지점.
 */
class NearbyFixtureRecorder {

    private record Def(String name, double lat, double lng, Double heading, double speed, int limit, PlaceType type,
                       String route, String allowed) {
    }

    private static Def corridor(String name, double lat, double lng, double heading, String route, String allowed) {
        return new Def(name, lat, lng, heading, 27, 5, null, route, allowed);
    }

    private static final List<Def> SCENARIOS = List.of(
            // 경부 북행 오산 부근: 지나친 내 방향 휴게소가 반대 차선으로 오인돼 결과가 사라졌던 지점
            new Def("gb_n_osan", 37.073253, 127.132555, 340.0, 27, 3, null, "경부", "서울"),
            new Def("gb_n_osan_shelter", 37.073253, 127.132555, 340.0, 27, 3, PlaceType.DROWSY_SHELTER, "경부", "서울"),
            // 정지 상태: 방향값을 쓰지 않음 (방향 검사 없이 결과만 고정)
            new Def("gb_n_osan_stopped", 37.073253, 127.132555, 340.0, 0, 3, null, null, ""),
            // 경부고속도로에서 55m 벗어난 좌표 (동탄): 카카오가 일반도로(동탄기흥로)로 출발해 5~12km 뒤에야 고속도로에 올라감
            // → 고속도로 위가 아님(onHighway=false, 결과 없음)
            new Def("local_dongtan", 37.177956, 127.095514, 200.0, 27, 3, null, "경부", "부산"),
            new Def("local_dongtan_shelter", 37.177956, 127.095514, 200.0, 27, 3, PlaceType.DROWSY_SHELTER, "경부", "부산"),
            // 근처에 아무것도 없음 (바다)
            new Def("none_sea", 34.0, 125.0, 0.0, 27, 3, null, null, ""),
            corridor("gb_n_005km", 35.3138, 129.0673, 314, "경부", "서울"),
            corridor("gb_n_051km", 35.6952, 129.1855, 5, "경부", "서울"),
            corridor("gb_n_081km", 35.8907, 129.0403, 289, "경부", "서울"),
            corridor("gb_n_096km", 37.3761, 127.1034, 360, "경부", "서울"),
            corridor("gb_s_005km", 37.4377, 127.0623, 141, "경부", "부산"),
            corridor("gb_s_051km", 35.7500, 129.1945, 174, "경부", "부산"),
            corridor("gb_s_081km", 35.4958, 129.0939, 193, "경부", "부산"),
            corridor("gb_s_096km", 35.3623, 129.0454, 182, "경부", "부산"),
            corridor("hn_n_005km", 35.2530, 126.8074, 321, "호남", "천안"),
            corridor("hn_n_051km", 35.6209, 126.8829, 47, "호남", "천안"),
            corridor("hn_n_097km", 35.9617, 127.1037, 8, "호남", "천안"),
            corridor("hn_n_142km", 36.2783, 127.3131, 22, "호남", "천안"),
            corridor("hn_n_157km", 36.3990, 127.3569, 50, "호남", "천안"),
            corridor("hn_s_005km", 36.3911, 127.3411, 205, "호남", "순천"),
            corridor("hn_s_050km", 36.0900, 127.1072, 206, "호남", "순천"),
            corridor("hn_s_097km", 35.7080, 126.9689, 225, "호남", "순천"),
            corridor("hn_s_142km", 35.3567, 126.8144, 153, "호남", "순천"),
            corridor("hn_s_157km", 35.2388, 126.8210, 142, "호남", "순천"),
            corridor("ja_n_005km", 35.9720, 128.5513, 357, "중앙", "춘천"),
            corridor("ja_n_050km", 36.3487, 128.5843, 327, "중앙", "춘천"),
            corridor("ja_n_096km", 36.6781, 128.5497, 17, "중앙", "춘천"),
            corridor("ja_n_142km", 36.9617, 128.2996, 321, "중앙", "춘천"),
            corridor("ja_n_187km", 37.2605, 128.0364, 326, "중앙", "춘천"),
            corridor("ja_n_232km", 37.5807, 127.9746, 358, "중앙", "춘천"),
            corridor("ja_n_247km", 37.6752, 127.8670, 283, "중앙", "춘천"),
            corridor("ja_n_262km", 37.7718, 127.7791, 301, "중앙", "춘천"),
            corridor("ja_s_005km", 37.8192, 127.7748, 233, "중앙", "부산"),
            corridor("ja_s_050km", 37.5090, 127.9755, 228, "중앙", "부산"),
            corridor("ja_s_096km", 37.2058, 128.1064, 109, "중앙", "부산"),
            corridor("ja_s_142km", 36.9166, 128.3853, 116, "중앙", "부산"),
            corridor("ja_s_187km", 36.6094, 128.5718, 117, "중앙", "부산"),
            corridor("ja_s_233km", 36.2618, 128.5727, 218, "중앙", "부산"),
            corridor("ja_s_248km", 36.1350, 128.5428, 184, "중앙", "부산"),
            corridor("ja_s_264km", 36.0173, 128.5400, 150, "중앙", "부산"),
            corridor("nh_e_005km", 34.9886, 127.5577, 104, "남해", "부산"),
            corridor("nh_e_050km", 35.0589, 127.9780, 51, "남해", "부산"),
            corridor("nh_e_096km", 35.2869, 128.3322, 46, "남해", "부산"),
            corridor("nh_e_111km", 35.2790, 128.7059, 90, "남해", "부산"),
            corridor("nh_w_005km", 35.2470, 128.7598, 350, "남해", "순천,영암"),
            corridor("nh_w_050km", 35.2795, 128.3050, 249, "남해", "순천,영암"),
            corridor("nh_w_097km", 35.0534, 127.9438, 281, "남해", "순천,영암"),
            corridor("nh_w_127km", 34.9732, 127.6628, 200, "남해", "순천,영암"),
            corridor("nh_w_142km", 34.9873, 127.5204, 265, "남해", "순천,영암"),
            corridor("sh_n_006km", 34.8702, 126.4919, 16, "서해안", "서울"),
            corridor("sh_n_051km", 35.2689, 126.5410, 28, "서해안", "서울"),
            corridor("sh_n_096km", 35.6384, 126.7078, 45, "서해안", "서울"),
            corridor("sh_n_142km", 36.0152, 126.7972, 347, "서해안", "서울"),
            corridor("sh_n_172km", 37.1900, 126.8725, 354, "서해안", "서울"),
            corridor("sh_n_188km", 37.3167, 126.8846, 324, "서해안", "서울"),
            corridor("sh_s_005km", 37.3534, 126.8621, 201, "서해안", "목포"),
            corridor("sh_s_051km", 36.0567, 126.7786, 128, "서해안", "목포"),
            corridor("sh_s_096km", 35.6888, 126.7454, 212, "서해안", "목포"),
            corridor("sh_s_142km", 35.3138, 126.5657, 199, "서해안", "목포"),
            corridor("sh_s_173km", 35.0613, 126.4800, 173, "서해안", "목포"),
            corridor("sh_s_189km", 34.9214, 126.4980, 178, "서해안", "목포"),
            corridor("yd_e_005km", 37.3595, 126.7751, 120, "영동", "강릉"),
            corridor("yd_e_051km", 37.2470, 127.2398, 102, "영동", "강릉"),
            corridor("yd_e_096km", 37.2449, 127.7307, 101, "영동", "강릉"),
            corridor("yd_e_142km", 37.4512, 128.1018, 33, "영동", "강릉"),
            corridor("yd_e_187km", 37.6370, 128.4966, 43, "영동", "강릉"),
            corridor("yd_e_202km", 37.6792, 128.6522, 50, "영동", "강릉"),
            corridor("yd_e_218km", 37.7008, 128.7694, 23, "영동", "강릉"),
            corridor("yd_w_005km", 37.7555, 128.7974, 227, "영동", "인천"),
            corridor("yd_w_050km", 37.6077, 128.4304, 228, "영동", "인천"),
            corridor("yd_w_081km", 37.4673, 128.1480, 242, "영동", "인천"),
            corridor("yd_w_097km", 37.4395, 128.0247, 258, "영동", "인천")
    );

    /** 실제 카카오를 호출해 응답을 녹화 */
    @Test
    @EnabledIfSystemProperty(named = "nearby.record", matches = "true")
    void record() throws Exception {
        String only = System.getProperty("nearby.only");
        RecordingClient recorder = new RecordingClient(kakaoKey());
        RestPlaceCache cache = NearbyFixtures.loadCache();

        int recorded = 0;
        for (Def d : SCENARIOS) {
            if (only != null ? !List.of(only.split(",")).contains(d.name()) : Files.exists(NearbyFixtures.fileOf(d.name()))) {
                continue;
            }
            NearbyRequest req = request(d);
            recorder.responses.clear();
            NearbyRestPlaceService liveService = new NearbyRestPlaceService(cache, recorder);
            liveService.search(req, d.limit());
            Thread.sleep(2_000);   // 제한 시간 뒤 늦게 끝난 호출까지 녹화
            liveService.shutdown();

            saveReplayed(d, req, new LinkedHashMap<>(recorder.responses), cache);
            recorded++;
        }
        System.out.println("recorded scenarios: " + recorded);
    }

    /** 카카오 호출 없이, 녹화된 응답으로 기대 결과만 다시 계산 (판정 로직을 의도적으로 바꾼 뒤) */
    @Test
    @EnabledIfSystemProperty(named = "nearby.rebaseline", matches = "true")
    void rebaseline() throws Exception {
        RestPlaceCache cache = NearbyFixtures.loadCache();
        int count = 0;
        for (Def d : SCENARIOS) {
            if (!Files.exists(NearbyFixtures.fileOf(d.name()))) {
                continue;
            }
            Scenario old = NearbyFixtures.read(NearbyFixtures.fileOf(d.name()));
            saveReplayed(d, old.request(), new LinkedHashMap<>(old.kakao()), cache);
            count++;
        }
        System.out.println("rebaselined scenarios: " + count);
    }

    /** 녹화본을 재생해 기대 결과를 계산하고 저장 (녹화에 없는 요청은 실패로 기록) */
    private static void saveReplayed(Def d, NearbyRequest req, Map<String, Map<String, Object>> kakao,
                                     RestPlaceCache cache) throws IOException {
        NearbyRestPlaceService.Search res;
        while (true) {
            ReplayKakaoClient replay = new ReplayKakaoClient(kakao);
            NearbyRestPlaceService service = new NearbyRestPlaceService(cache, replay);
            res = service.search(req, d.limit());
            service.shutdown();
            if (replay.missing.isEmpty()) {
                break;
            }
            replay.missing.forEach(k -> kakao.put(k, null));
        }

        List<NearbyItem> items = res.items();
        Scenario s = new Scenario(d.name(), req, d.limit(), d.route(),
                d.allowed().isEmpty() ? List.of() : NearbyFixtures.labels(d.allowed()), kakao, res.onHighway(),
                items.stream().map(NearbyFixtures::describe).toList());
        List<String> wrong = NearbyFixtures.wrongDirection(s, items);
        NearbyFixtures.write(s);
        System.out.printf("[%s] kakao=%d onHighway=%s%s%n  %s%n", d.name(), kakao.size(), res.onHighway(),
                wrong.isEmpty() ? "" : " WRONG=" + wrong, String.join("\n  ", s.expected()));
    }

    private static NearbyRequest request(Def d) {
        NearbyRequest req = new NearbyRequest();
        req.setLat(d.lat());
        req.setLng(d.lng());
        req.setHeading(d.heading());
        req.setSpeed(d.speed());
        req.setType(d.type());
        return req;
    }

    /** 실제 카카오를 호출하면서 응답(판정에 쓰는 부분만)을 저장 */
    private static class RecordingClient extends KakaoDirectionsClient {
        final Map<String, Map<String, Object>> responses = Collections.synchronizedMap(new LinkedHashMap<>());

        RecordingClient(String apiKey) {
            super(apiKey, 3_000);
        }

        @Override
        protected Map<String, Object> fetch(String origin, String destination) throws InterruptedException {
            Map<String, Object> body = null;
            try {
                body = NearbyFixtures.trim(super.fetch(origin, destination));
                return body;
            } finally {
                responses.put(NearbyFixtures.key(origin, destination), body);
            }
        }
    }

    /** application.properties 의 driverguard.kakao.rest-api-key (${ENV:기본값} 형식 해석) */
    private static String kakaoKey() throws IOException {
        Properties p = new Properties();
        try (InputStream in = NearbyFixtureRecorder.class.getResourceAsStream("/application.properties")) {
            p.load(in);
        }
        String raw = p.getProperty("driverguard.kakao.rest-api-key", "");
        Matcher m = Pattern.compile("^\\$\\{([^:}]+):?([^}]*)}$").matcher(raw.trim());
        if (!m.matches()) {
            return raw.trim();
        }
        String env = System.getenv(m.group(1));
        return env != null && !env.isBlank() ? env : m.group(2);
    }
}
