package com.contest.DriverGuard.auth;

import com.contest.DriverGuard.support.ApiTestSupport;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 회원가입 · 로그인 · 토큰 재발급 · 로그아웃 */
class AuthApiTest extends ApiTestSupport {

    @Test
    void 가입하면_토큰이_바로_발급되고_약관_동의가_필요하다() throws Exception {
        String email = uniqueEmail();

        postJson("/api/auth/signup", map("email", email.toUpperCase(), "name", "  김하늘 ",
                "password", "Abcd1234!", "ageConfirmed", true))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.userId").isNumber())
                .andExpect(jsonPath("$.email").value(email))            // 소문자로 저장
                .andExpect(jsonPath("$.name").value("김하늘"))           // 앞뒤 공백 제거
                .andExpect(jsonPath("$.grantType").value("Bearer"))
                .andExpect(jsonPath("$.accessToken").isNotEmpty())
                .andExpect(jsonPath("$.refreshToken").isNotEmpty())
                .andExpect(jsonPath("$.termsAgreementRequired").value(true));
    }

    @Test
    void 입력값이_틀리면_입력란별로_이유를_알려준다() throws Exception {
        postJson("/api/auth/signup", map("email", "haneul.kim@mail", "name", "",
                "password", "12345678", "ageConfirmed", false))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("VALIDATION_FAILED"))
                .andExpect(jsonPath("$.fieldErrors[*].field")
                        .value(containsInAnyOrder("email", "name", "password", "ageConfirmed")));
    }

    @Test
    void 비밀번호는_영문과_숫자를_섞어야_한다() throws Exception {
        for (String password : new String[]{"12345678", "abcdefgh", "abc1234"}) {
            postJson("/api/auth/signup", map("email", uniqueEmail(), "name", "김하늘",
                    "password", password, "ageConfirmed", true))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.fieldErrors[0].field").value("password"));
        }
    }

    @Test
    void 이미_가입한_이메일이면_409() throws Exception {
        String email = uniqueEmail();
        signup(email);

        postJson("/api/auth/signup", map("email", email, "name", "다른사람", "password", PASSWORD, "ageConfirmed", true))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("EMAIL_DUPLICATED"));
    }

    @Test
    void 로그인_성공과_실패() throws Exception {
        String email = uniqueEmail();
        signup(email);

        postJson("/api/auth/login", map("email", email, "password", PASSWORD))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.accessToken").isNotEmpty())
                .andExpect(jsonPath("$.termsAgreementRequired").value(true));

        postJson("/api/auth/login", map("email", email, "password", "Wrong1234!"))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.code").value("LOGIN_FAILED"));
    }

    @Test
    void 재발급하면_refreshToken_이_바뀌고_쓴_토큰은_다시_못_쓴다() throws Exception {
        String refreshToken = signup(uniqueEmail()).get("refreshToken").asText();

        JsonNode refreshed = body(postJson("/api/auth/refresh", map("refreshToken", refreshToken))
                .andExpect(status().isOk()));
        String newRefreshToken = refreshed.get("refreshToken").asText();

        assertThat(newRefreshToken).isNotEqualTo(refreshToken);
        postJson("/api/auth/refresh", map("refreshToken", refreshToken))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.code").value("INVALID_REFRESH_TOKEN"));
    }

    @Test
    void 로그아웃하면_그_refreshToken_으로_재발급할_수_없다() throws Exception {
        String refreshToken = signup(uniqueEmail()).get("refreshToken").asText();

        postJson("/api/auth/logout", map("refreshToken", refreshToken)).andExpect(status().isNoContent());
        postJson("/api/auth/refresh", map("refreshToken", refreshToken)).andExpect(status().isUnauthorized());
    }

    @Test
    void 토큰이_없거나_잘못되면_401() throws Exception {
        getJson("/api/users/me", null)
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.code").value("UNAUTHORIZED"));
        getJson("/api/users/me", "not-a-jwt")
                .andExpect(status().isUnauthorized());
    }
}
