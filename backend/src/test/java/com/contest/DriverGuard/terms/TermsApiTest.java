package com.contest.DriverGuard.terms;

import com.contest.DriverGuard.support.ApiTestSupport;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 약관 목록 · 전문 · 동의 · 동의 전 기능 차단 · 약관 개정 시 재동의 */
class TermsApiTest extends ApiTestSupport {

    @Test
    void 약관_목록과_전문은_로그인_없이_볼_수_있다() throws Exception {
        JsonNode terms = body(getJson("/api/terms", null).andExpect(status().isOk())).get("terms");
        long firstId = terms.get(0).get("termsId").asLong();

        getJson("/api/terms/" + firstId, null)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.termsId").value(firstId))
                .andExpect(jsonPath("$.content").isNotEmpty())
                .andExpect(jsonPath("$.effectiveAt").isNotEmpty());

        getJson("/api/terms/999999999", null)
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value("TERMS_NOT_FOUND"));
    }

    @Test
    void 필수_약관에_동의하기_전에는_기능_API_를_쓸_수_없다() throws Exception {
        String token = signup(uniqueEmail()).get("accessToken").asText();

        getJson("/api/drives", token)
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.code").value("TERMS_AGREEMENT_REQUIRED"));
        postJson("/api/rest-places/nearby", map("lat", 37.0, "lng", 127.0), token)
                .andExpect(status().isForbidden());

        // 내 정보와 약관 API 는 동의 전에도 호출 가능
        getJson("/api/users/me", token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.termsAgreementRequired").value(true));
    }

    @Test
    void 필수_약관을_빠뜨리거나_false_로_보내면_400() throws Exception {
        String token = signup(uniqueEmail()).get("accessToken").asText();
        List<Map<String, Object>> optionalOnly = new ArrayList<>();
        List<Map<String, Object>> requiredFalse = new ArrayList<>();
        for (JsonNode t : currentTerms()) {
            if (!t.get("required").asBoolean()) {
                optionalOnly.add(map("termsId", t.get("termsId").asLong(), "agreed", true));
            }
            requiredFalse.add(map("termsId", t.get("termsId").asLong(), "agreed", false));
        }

        postJson("/api/users/me/terms-agreements", map("agreements", optionalOnly), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("TERMS_REQUIRED_NOT_AGREED"));
        postJson("/api/users/me/terms-agreements", map("agreements", requiredFalse), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("TERMS_REQUIRED_NOT_AGREED"));
        postJson("/api/users/me/terms-agreements", map("agreements", List.of()), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[0].field").value("agreements"));
    }

    @Test
    void 필수_약관에_모두_동의하면_기능_API_를_쓸_수_있다() throws Exception {
        String token = signup(uniqueEmail()).get("accessToken").asText();
        agreeAllRequired(token);

        getJson("/api/users/me/terms-agreements", token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.termsAgreementRequired").value(false));
        getJson("/api/drives", token).andExpect(status().isOk());
    }

    @Test
    void 약관이_개정되면_다시_동의해야_하고_예전_버전으로는_동의할_수_없다() throws Exception {
        String token = signupAndAgree();
        JsonNode first = currentTerms().get(0);
        long oldId = first.get("termsId").asLong();

        // 같은 약관의 새 버전을 지금 시행 (테스트가 끝나면 롤백)
        jdbcTemplate.update("""
                INSERT INTO terms (code, version, title, content, is_required, display_order, effective_at)
                SELECT code, version + 1, title, '# 개정', is_required, display_order, now() - interval '1 minute'
                FROM terms WHERE terms_id = ?""", oldId);
        Long newId = jdbcTemplate.queryForObject(
                "SELECT max(terms_id) FROM terms WHERE code = ?", Long.class, first.get("code").asText());

        getJson("/api/users/me", token).andExpect(jsonPath("$.termsAgreementRequired").value(true));
        getJson("/api/drives", token).andExpect(status().isForbidden());

        postJson("/api/users/me/terms-agreements", map("agreements", List.of(map("termsId", oldId, "agreed", true))), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("TERMS_NOT_CURRENT"));
        postJson("/api/users/me/terms-agreements", map("agreements", List.of(map("termsId", newId, "agreed", true))), token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.termsAgreementRequired").value(false));
    }

    private JsonNode currentTerms() throws Exception {
        return body(getJson("/api/terms", null)).get("terms");
    }
}
