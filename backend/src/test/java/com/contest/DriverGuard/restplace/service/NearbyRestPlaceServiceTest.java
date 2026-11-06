package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyItem;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyRequest;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyResponse;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearbyResponse.DistanceSource;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.NearestPlace;
import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace.PlaceType;
import com.contest.DriverGuard.restplace.service.NearbyFixtures.ReplayKakaoClient;
import com.contest.DriverGuard.restplace.service.NearbyFixtures.Scenario;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 주변 조회 회귀 테스트. 녹화해 둔 실제 카카오 길찾기 응답(NearbyFixtureRecorder)을 재생하므로 카카오를 호출하지 않음.
 * 1) 지금 달리는 노선에서 반대 방향(반대 차선) 휴게소/쉼터가 절대 나오지 않을 것
 * 2) 결과가 녹화 당시와 같을 것 (판정 로직을 바꿔 결과가 달라지면 실패 → 의도한 변화인지 확인 후 다시 녹화)
 */
class NearbyRestPlaceServiceTest {

    private static RestPlaceCache cache;
    private final List<NearbyRestPlaceService> services = new ArrayList<>();

    @BeforeAll
    static void loadPlaces() {
        cache = NearbyFixtures.loadCache();
    }

    @AfterEach
    void shutdown() {
        services.forEach(NearbyRestPlaceService::shutdown);
    }

    static Stream<Arguments> scenarios() throws Exception {
        List<Path> files = NearbyFixtures.scenarioFiles();
        assertThat(files).as("녹화된 시나리오가 없음 (NearbyFixtureRecorder 실행 필요)").isNotEmpty();
        return files.stream().map(f -> Arguments.of(f.getFileName().toString().replace(".json.gz", ""), f));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("scenarios")
    void recordedScenario(String name, Path file) throws Exception {
        Scenario s = NearbyFixtures.read(file);
        ReplayKakaoClient kakao = new ReplayKakaoClient(s.kakao());

        NearbyRestPlaceService.Search res = service(kakao).search(s.request(), s.limit());
        List<NearbyItem> items = res.items();

        assertThat(kakao.missing).as("녹화에 없는 길찾기 요청 (판정 로직이 다른 곳을 조회함 → 다시 녹화 필요)").isEmpty();
        assertThat(NearbyFixtures.wrongDirection(s, items)).as("반대 방향 결과").isEmpty();
        assertThat(res.onHighway()).as("고속도로 위 여부").isEqualTo(s.onHighway());
        assertThat(items).extracting(NearbyFixtures::describe).as("결과").containsExactlyElementsOf(s.expected());
    }

    @Test
    void findNearest_returnsOnlyTheClosestPlaceForWaypoint() throws Exception {
        // 경부 북행 오산 부근 (내부 검색 결과: 오산 쉼터 10.3km, 죽전(서울) 30.6km, 화성(서울) 31.7km)
        Scenario s = NearbyFixtures.read(NearbyFixtures.fileOf("gb_n_osan"));

        NearbyResponse res = service(new ReplayKakaoClient(s.kakao())).findNearest(s.request());

        NearestPlace place = res.getPlace();
        assertThat(res.getOnHighway()).isTrue();
        assertThat(res.getDistanceSource()).isEqualTo(DistanceSource.ROAD);
        assertThat(place.getName()).isEqualTo("오산");
        assertThat(place.getType()).isEqualTo(PlaceType.DROWSY_SHELTER);
        assertThat(place.getLatitude()).isEqualTo(37.1542659);
        assertThat(place.getLongitude()).isEqualTo(127.0861027);
        assertThat(place.getRoadDistanceMeters()).isEqualTo(10313);
    }

    @Test
    void findNearest_notOnHighway_returnsNoPlace() throws Exception {
        Scenario s = NearbyFixtures.read(NearbyFixtures.fileOf("local_dongtan"));

        NearbyResponse res = service(new ReplayKakaoClient(s.kakao())).findNearest(s.request());

        assertThat(res.getOnHighway()).isFalse();
        assertThat(res.getPlace()).isNull();
    }

    @Test
    void kakaoDown_returnsUnavailableWithoutStraightLineGuess() {
        // 경부 남행, 카카오가 모두 실패 → 직선거리로 추측해서 내보내지 않음
        ReplayKakaoClient kakao = new ReplayKakaoClient(Map.of());

        NearbyResponse res = service(kakao).findNearest(request(37.177956, 127.095514, 200.0, 27.0));

        assertThat(kakao.requested).isNotEmpty();
        assertThat(res.getDistanceSource()).isEqualTo(DistanceSource.UNAVAILABLE);
        assertThat(res.getPlace()).isNull();
    }

    @Test
    void nothingNearby_returnsEmptyWithoutCallingKakao() {
        ReplayKakaoClient kakao = new ReplayKakaoClient(Map.of());

        NearbyResponse res = service(kakao).findNearest(request(34.0, 125.0, 0.0, 27.0));

        assertThat(kakao.requested).isEmpty();
        assertThat(res.getDistanceSource()).isEqualTo(DistanceSource.ROAD);
        assertThat(res.getPlace()).isNull();
    }

    @Test
    void slowSpeed_ignoresHeading() {
        ReplayKakaoClient kakao = new ReplayKakaoClient(Map.of());

        NearbyResponse res = service(kakao).findNearest(request(37.177956, 127.095514, 200.0, 1.0));

        assertThat(res.isDirectionApplied()).isFalse();
        assertThat(kakao.requested).allSatisfy(k -> assertThat(k).doesNotContain("angle="));
    }

    @Test
    void parse_failedRoute_isEmpty() {
        Map<String, Object> body = Map.of("routes", List.of(Map.of("result_code", 104, "result_msg", "출발지와 도착지가 너무 가까움")));
        assertThat(KakaoDirectionsClient.parse(body)).isEmpty();
        assertThat(KakaoDirectionsClient.parse(null)).isEmpty();
    }

    private NearbyRestPlaceService service(KakaoDirectionsClient kakao) {
        NearbyRestPlaceService service = new NearbyRestPlaceService(cache, kakao);
        services.add(service);
        return service;
    }

    private static NearbyRequest request(double lat, double lng, Double heading, Double speed) {
        NearbyRequest req = new NearbyRequest();
        req.setLat(lat);
        req.setLng(lng);
        req.setHeading(heading);
        req.setSpeed(speed);
        return req;
    }
}
