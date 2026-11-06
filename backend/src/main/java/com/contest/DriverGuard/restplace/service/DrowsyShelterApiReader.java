package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** 공공데이터포털 전국졸음쉼터표준데이터 API (tn_pubr_public_drowsy_shelter_api) 전체 조회 */
@Slf4j
@Component
public class DrowsyShelterApiReader {

    private static final String URL = "https://api.data.go.kr/openapi/tn_pubr_public_drowsy_shelter_api";
    private static final int PAGE_SIZE = 1000;      // 이 API의 한 페이지 최대 건수
    private static final int MAX_PAGES = 100;       // 무한 반복 방지
    private static final ParameterizedTypeReference<Map<String, Object>> MAP_TYPE =
            new ParameterizedTypeReference<>() {};

    private final RestClient restClient;
    private final String serviceKey;

    public DrowsyShelterApiReader(@Value("${driverguard.data-go-kr.service-key}") String serviceKey) {
        this.serviceKey = serviceKey;
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(Duration.ofSeconds(5));
        requestFactory.setReadTimeout(Duration.ofSeconds(30));
        this.restClient = RestClient.builder().requestFactory(requestFactory).build();
    }

    @SuppressWarnings("unchecked")
    public List<RestPlace> read() {
        if (serviceKey == null || serviceKey.isBlank()) {
            throw new IllegalStateException("driverguard.data-go-kr.service-key (DATA_GO_KR_SERVICE_KEY) is not set");
        }

        List<RestPlace> rows = new ArrayList<>();
        int skipped = 0;
        int pageNo = 1;
        int totalPages;

        do {
            Map<String, Object> res;
            try {
                // 키 값의 +, /, = 는 RestClient 가 URI 변수로 인코딩하므로 Decoding 키를 그대로 넘김
                res = restClient.get()
                        .uri(URL + "?serviceKey={key}&type=json&numOfRows={rows}&pageNo={page}",
                                serviceKey, PAGE_SIZE, pageNo)
                        .retrieve()
                        .body(MAP_TYPE);
            } catch (RestClientException e) {
                // 키 오류 등은 type=json 이어도 XML 로 응답해서 JSON 변환에 실패함
                throw new IllegalStateException("drowsy shelter api call failed at page " + pageNo
                        + " (서비스키/트래픽 확인): " + e.getMessage(), e);
            }

            // 실제 응답: {"header":{...},"body":{"items":{"item":[...]},"totalCount":253}}
            Map<String, Object> header = map(res, "header");
            Map<String, Object> body = map(res, "body");

            // 한 페이지라도 실패하면 전체 실패로 처리 (일부만 반영되면 나머지가 비활성화되므로)
            String resultCode = str(header, "resultCode");
            if (!"00".equals(resultCode)) {
                throw new IllegalStateException("drowsy shelter api error at page " + pageNo + ": "
                        + resultCode + " " + str(header, "resultMsg"));
            }

            int totalCount = toInt(str(body, "totalCount"), 0);
            totalPages = Math.max(1, (totalCount + PAGE_SIZE - 1) / PAGE_SIZE);

            List<Map<String, Object>> items = map(body, "items").get("item") instanceof List<?> l
                    ? (List<Map<String, Object>>) l
                    : List.of();

            for (Map<String, Object> item : items) {
                RestPlace row = toRow(item);
                if (row == null) {
                    skipped++;
                } else {
                    rows.add(row);
                }
            }

            log.info("drowsy shelter api page {}/{}: {} rows", pageNo, totalPages, items.size());
            pageNo++;
        } while (pageNo <= totalPages && pageNo <= MAX_PAGES);

        log.info("drowsy shelter api: {} rows read, {} skipped", rows.size(), skipped);
        return rows;
    }

    private RestPlace toRow(Map<String, Object> item) {
        String name = str(item, "shltrNm");
        if (name == null) {
            return null;
        }

        Double lat = toDouble(str(item, "latitude"));
        Double lng = toDouble(str(item, "longitude"));
        if (!RestPlace.isValidCoordinate(lat, lng)) {
            lat = null;
            lng = null;
        }

        String direction = str(item, "roadRouteDrc");
        String roadAddress = str(item, "rdnmadr");

        return RestPlace.builder()
                .placeType(RestPlace.PlaceType.DROWSY_SHELTER)
                // 고유 코드가 없어 이름·노선번호·방향을 합쳐 키로 사용 (노선번호는 키에만 쓰고 저장하지 않음)
                .sourceKey(name + "|" + nullToEmpty(str(item, "roadRouteNo")) + "|" + nullToEmpty(direction))
                .name(name)
                .routeName(str(item, "roadRouteNm"))
                .direction(direction)
                .latitude(lat)
                .longitude(lng)
                .address(roadAddress != null ? roadAddress : str(item, "lnmadr"))
                .parkingSpaces(toInt(str(item, "prkplceCo"), null))
                .build();
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> map(Map<String, Object> m, String key) {
        Object v = m == null ? null : m.get(key);
        return v instanceof Map<?, ?> ? (Map<String, Object>) v : Map.of();
    }

    private String str(Map<String, Object> m, String key) {
        Object v = m.get(key);
        if (v == null) {
            return null;
        }
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? null : s;
    }

    private Double toDouble(String s) {
        try {
            return s == null ? null : Double.valueOf(s);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private Integer toInt(String s, Integer defaultValue) {
        try {
            return s == null ? defaultValue : Integer.valueOf(s.replace(",", ""));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private String nullToEmpty(String v) {
        return v == null ? "" : v;
    }
}