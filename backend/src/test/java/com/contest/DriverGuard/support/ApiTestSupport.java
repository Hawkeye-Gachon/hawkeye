package com.contest.DriverGuard.support;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;

/**
 * API 통합 테스트 공통 설정.
 * 실제 서버처럼 HTTP 요청으로 호출하고(MockMvc), 로컬 DB 를 쓰되 테스트마다 끝나면 롤백해서 데이터가 남지 않음.
 * 필요한 것: schema.sql 과 terms-sample.sql 이 적용된 로컬 DB (application.properties 의 datasource)
 *
 * local-cache-scope=statement: 테스트는 메서드 전체가 한 트랜잭션(= 한 MyBatis 세션)이라, 테스트가 JdbcTemplate 으로
 * 직접 바꾼 데이터를 MyBatis 가 앞서 캐시한 조회 결과로 덮어 보지 않도록 함. 실제 요청은 요청마다 세션이 따로라 해당 없음
 */
@SpringBootTest(properties = "mybatis.configuration.local-cache-scope=statement")
@AutoConfigureMockMvc
@Transactional
public abstract class ApiTestSupport {

    protected static final String PASSWORD = "Test1234!";
    protected static final ZoneId SEOUL = ZoneId.of("Asia/Seoul");

    @Autowired
    protected MockMvc mockMvc;

    @Autowired
    protected JsonMapper jsonMapper;

    @Autowired
    protected JdbcTemplate jdbcTemplate;

    // ---------------- 요청 ----------------

    protected ResultActions postJson(String url, Object body, String token) throws Exception {
        return mockMvc.perform(withAuth(post(url), token).contentType(MediaType.APPLICATION_JSON)
                .characterEncoding(StandardCharsets.UTF_8).content(jsonMapper.writeValueAsString(body)));
    }

    protected ResultActions postJson(String url, Object body) throws Exception {
        return postJson(url, body, null);
    }

    protected ResultActions getJson(String url, String token) throws Exception {
        return mockMvc.perform(withAuth(get(url), token));
    }

    protected ResultActions deleteJson(String url, Object body, String token) throws Exception {
        return mockMvc.perform(withAuth(delete(url), token).contentType(MediaType.APPLICATION_JSON)
                .characterEncoding(StandardCharsets.UTF_8).content(jsonMapper.writeValueAsString(body)));
    }

    private MockHttpServletRequestBuilder withAuth(MockHttpServletRequestBuilder builder, String token) {
        return token == null ? builder : builder.header("Authorization", "Bearer " + token);
    }

    protected JsonNode body(ResultActions result) throws Exception {
        return jsonMapper.readTree(result.andReturn().getResponse().getContentAsString(StandardCharsets.UTF_8));
    }

    /** {"a": 1, "b": 2} 를 map("a", 1, "b", 2) 로 (순서 유지) */
    protected static Map<String, Object> map(Object... keyValues) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < keyValues.length; i += 2) {
            m.put((String) keyValues[i], keyValues[i + 1]);
        }
        return m;
    }

    // ---------------- 회원 ----------------

    protected static String uniqueEmail() {
        return "it-" + UUID.randomUUID().toString().substring(0, 8) + "@example.com";
    }

    /** 가입하고 응답 본문 반환 */
    protected JsonNode signup(String email) throws Exception {
        return body(postJson("/api/auth/signup",
                map("email", email, "name", "김하늘", "password", PASSWORD, "ageConfirmed", true)));
    }

    /** 가입 + 현재 필수 약관 모두 동의 → accessToken */
    protected String signupAndAgree() throws Exception {
        String token = signup(uniqueEmail()).get("accessToken").asText();
        agreeAllRequired(token);
        return token;
    }

    protected void agreeAllRequired(String token) throws Exception {
        List<Map<String, Object>> agreements = new ArrayList<>();
        for (JsonNode terms : body(getJson("/api/terms", null)).get("terms")) {
            agreements.add(map("termsId", terms.get("termsId").asLong(), "agreed", terms.get("required").asBoolean()));
        }
        postJson("/api/users/me/terms-agreements", map("agreements", agreements), token);
    }

    // ---------------- 시각 ----------------

    /** 서울 기준 오늘에서 daysAgo 일 전 날짜의 hh:mm:ss (+09:00) */
    protected static OffsetDateTime kst(int daysAgo, int hour, int minute, int second) {
        return LocalDate.now(SEOUL).minusDays(daysAgo).atTime(hour, minute, second).atZone(SEOUL).toOffsetDateTime();
    }

    protected static OffsetDateTime minutesAgo(long minutes) {
        return OffsetDateTime.now(SEOUL).minusMinutes(minutes).withNano(0);
    }
}
