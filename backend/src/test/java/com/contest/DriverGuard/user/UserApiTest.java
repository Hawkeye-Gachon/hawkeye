package com.contest.DriverGuard.user;

import com.contest.DriverGuard.support.ApiTestSupport;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 내 정보 · 회원 탈퇴 */
class UserApiTest extends ApiTestSupport {

    @Test
    void 내_정보() throws Exception {
        String email = uniqueEmail();
        String token = signup(email).get("accessToken").asText();

        getJson("/api/users/me", token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.email").value(email))
                .andExpect(jsonPath("$.name").value("김하늘"))
                .andExpect(jsonPath("$.status").value("ACTIVE"))
                .andExpect(jsonPath("$.createdAt").isNotEmpty());
    }

    @Test
    void 비밀번호가_틀리면_탈퇴되지_않는다() throws Exception {
        String token = signupAndAgree();

        deleteJson("/api/users/me", map("password", "Wrong1234!"), token)
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("PASSWORD_MISMATCH"));
        getJson("/api/users/me", token).andExpect(status().isOk());
    }

    @Test
    void 탈퇴하면_로그인_정보와_기록이_지워지고_같은_이메일로_다시_가입할_수_있다() throws Exception {
        String email = uniqueEmail();
        JsonNode signup = signup(email);
        String token = signup.get("accessToken").asText();
        String refreshToken = signup.get("refreshToken").asText();
        long userId = signup.get("userId").asLong();
        agreeAllRequired(token);
        postJson("/api/drives", map("clientDriveId", UUID.randomUUID().toString(),
                "startedAt", minutesAgo(10).toString()), token).andExpect(status().isCreated());

        deleteJson("/api/users/me", map("password", PASSWORD), token).andExpect(status().isNoContent());

        // 남은 accessToken · refreshToken · 로그인 모두 막힘
        getJson("/api/users/me", token).andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value("USER_NOT_FOUND"));
        getJson("/api/drives", token).andExpect(status().isNotFound());
        postJson("/api/auth/refresh", map("refreshToken", refreshToken)).andExpect(status().isUnauthorized());
        postJson("/api/auth/login", map("email", email, "password", PASSWORD)).andExpect(status().isUnauthorized());

        // 인증 수단 · 토큰 · 주행 · 약관 동의 기록 삭제, users 는 WITHDRAWN 으로 남음
        for (String table : new String[]{"auths", "refresh_tokens", "drives", "user_terms_agreements"}) {
            assertThat(jdbcTemplate.queryForObject("SELECT count(*) FROM " + table + " WHERE user_id = ?", Integer.class, userId))
                    .as(table).isZero();
        }
        assertThat(jdbcTemplate.queryForObject("SELECT status FROM users WHERE user_id = ?", String.class, userId))
                .isEqualTo("WITHDRAWN");

        postJson("/api/auth/signup", map("email", email, "name", "김하늘", "password", PASSWORD, "ageConfirmed", true))
                .andExpect(status().isCreated());
    }
}
