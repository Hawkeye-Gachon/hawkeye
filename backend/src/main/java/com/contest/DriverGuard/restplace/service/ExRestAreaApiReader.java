package com.contest.DriverGuard.restplace.service;

import com.contest.DriverGuard.restplace.dto.RestPlaceDto.RestPlace;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** 한국도로공사 휴게소 위치정보 API (locationinfoRest) 전체 조회 */
@Slf4j
@Component
public class ExRestAreaApiReader {

    private static final String URL = "https://data.ex.co.kr/openapi/locationinfo/locationinfoRest";
    private static final int PAGE_SIZE = 99;        // 이 API의 한 페이지 최대 건수
    private static final int MAX_PAGES = 100;       // 무한 반복 방지
    private static final Pattern DIRECTION = Pattern.compile("\\(([^)]+)\\)");
    private static final ParameterizedTypeReference<Map<String, Object>> MAP_TYPE =
            new ParameterizedTypeReference<>() {};

    private final RestClient restClient;
    private final String apiKey;

    public ExRestAreaApiReader(@Value("${driverguard.ex.api-key}") String apiKey) {
        this.apiKey = apiKey;
        // 타임아웃이 없으면 API가 응답을 안 줄 때 스케줄러 스레드가 계속 묶여 있음
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(Duration.ofSeconds(5));
        requestFactory.setReadTimeout(Duration.ofSeconds(30));
        this.restClient = RestClient.builder().requestFactory(requestFactory).build();
    }

    @SuppressWarnings("unchecked")
    public List<RestPlace> read() {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("driverguard.ex.api-key (EX_API_KEY) is not set");
        }

        List<RestPlace> rows = new ArrayList<>();
        int skipped = 0;
        int pageNo = 1;
        int totalPages;

        do {
            Map<String, Object> res = restClient.get()
                    .uri(URL + "?key={key}&type=json&numOfRows={rows}&pageNo={page}",
                            apiKey, PAGE_SIZE, pageNo)
                    .retrieve()
                    .body(MAP_TYPE);

            // 한 페이지라도 실패하면 전체 실패로 처리 (일부만 반영되면 나머지가 비활성화되므로)
            if (res == null || !"SUCCESS".equals(res.get("code"))) {
                throw new IllegalStateException("rest area api error at page " + pageNo + ": "
                        + (res == null ? "empty response" : res.get("message")));
            }

            totalPages = toInt(res.get("pageSize"), 1);   // pageSize = 전체 페이지 수

            List<Map<String, Object>> list = res.get("list") instanceof List<?> l
                    ? (List<Map<String, Object>>) l
                    : List.of();

            for (Map<String, Object> item : list) {
                RestPlace row = toRow(item);
                if (row == null) {
                    skipped++;
                } else {
                    rows.add(row);
                }
            }

            log.info("rest area api page {}/{}: {} rows", pageNo, totalPages, list.size());
            pageNo++;
        } while (pageNo <= totalPages && pageNo <= MAX_PAGES);

        log.info("rest area api: {} rows read, {} skipped", rows.size(), skipped);
        return rows;
    }

    private RestPlace toRow(Map<String, Object> item) {
        String code = str(item, "serviceAreaCode");
        String name = str(item, "unitName");
        if (code == null || name == null) {
            return null;
        }

        Double lat = toDouble(str(item, "yValue"));   // yValue = 위도
        Double lng = toDouble(str(item, "xValue"));   // xValue = 경도
        if (!RestPlace.isValidCoordinate(lat, lng)) {
            lat = null;
            lng = null;
        }

        return RestPlace.builder()
                .placeType(RestPlace.PlaceType.REST_AREA)
                .sourceKey(code)
                .name(name)
                .routeName(str(item, "routeName"))
                .direction(directionOf(name))
                .latitude(lat)
                .longitude(lng)
                .build();
    }

    /** "죽전(서울)휴게소" → "서울". 괄호가 없으면 null (양방향 공용 등) */
    private String directionOf(String name) {
        Matcher m = DIRECTION.matcher(name);
        return m.find() ? m.group(1) : null;
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

    private int toInt(Object v, int defaultValue) {
        try {
            return v == null ? defaultValue : Integer.parseInt(String.valueOf(v).trim());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
}