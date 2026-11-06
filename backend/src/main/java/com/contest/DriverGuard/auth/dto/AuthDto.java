package com.contest.DriverGuard.auth.dto;

import jakarta.validation.constraints.AssertTrue;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.OffsetDateTime;

public class AuthDto {

    @Data
    @NoArgsConstructor
    public static class SignupRequest {
        // @Email 은 도메인에 점이 없어도(name@mail) 통과시키므로 점이 있는 도메인을 추가로 요구
        @NotBlank(message = "이메일을 입력해 주세요.")
        @Email(message = "이메일 형식이 아닙니다. 예: name@mail.com")
        @Pattern(regexp = "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", message = "이메일 형식이 아닙니다. 예: name@mail.com")
        @Size(max = 254, message = "이메일은 254자 이하로 입력해 주세요.")
        private String email;

        @NotBlank(message = "이름을 입력해 주세요.")
        @Size(max = 50, message = "이름은 50자 이하로 입력해 주세요.")
        private String name;

        // 영문과 숫자를 각각 1자 이상 포함 (특수문자는 허용). BCrypt 는 72바이트까지만 사용하므로 상한을 둠
        @NotBlank(message = "비밀번호를 입력해 주세요.")
        @Size(min = 8, max = 64, message = "비밀번호는 영문과 숫자를 섞어 8~64자로 입력해 주세요.")
        @Pattern(regexp = "^(?=.*[A-Za-z])(?=.*\\d).*$",
                message = "비밀번호는 영문과 숫자를 섞어 8~64자로 입력해 주세요.")
        private String password;

        @NotNull(message = "만 14세 이상만 가입할 수 있습니다.")
        @AssertTrue(message = "만 14세 이상만 가입할 수 있습니다.")
        private Boolean ageConfirmed;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SignupResponse {
        private Long userId;
        private String email;
        private String name;
        // 가입 즉시 로그인 (약관 동의 화면으로 바로 진행)
        private String grantType;
        private String accessToken;
        private long accessTokenExpiresIn;
        private String refreshToken;
        private boolean termsAgreementRequired;
    }

    @Data
    @NoArgsConstructor
    public static class LoginRequest {
        @NotBlank(message = "이메일을 입력해 주세요.")
        @Email(message = "이메일 형식이 올바르지 않습니다.")
        private String email;

        @NotBlank(message = "비밀번호를 입력해 주세요.")
        private String password;
    }

    /** 회원 탈퇴 요청 (본인 확인용 비밀번호) */
    @Data
    @NoArgsConstructor
    public static class WithdrawRequest {
        @NotBlank(message = "비밀번호를 입력해 주세요.")
        private String password;
    }

    /** 재발급 / 로그아웃 요청 */
    @Data
    @NoArgsConstructor
    public static class RefreshTokenRequest {
        @NotBlank(message = "refreshToken 을 입력해 주세요.")
        private String refreshToken;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TokenResponse {
        private String grantType;            // "Bearer"
        private String accessToken;
        private long accessTokenExpiresIn;   // 초 단위
        private String refreshToken;
        private boolean termsAgreementRequired;   // true 면 앱은 약관 동의 화면으로
    }

    /** auths 테이블 - 사용자 1명이 여러 인증 수단(EMAIL, KAKAO, ...)을 가질 수 있음 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Auth {
        private Long authId;
        private Long userId;
        private Provider provider;
        private String providerUserId;
        private String passwordHash;
        private OffsetDateTime lastLoginAt;
        private OffsetDateTime createdAt;
        private OffsetDateTime updatedAt;

        public enum Provider {
            EMAIL,
            KAKAO,
            GOOGLE
        }
    }

    /** refresh_tokens 테이블 - 토큰 원문이 아닌 SHA-256 해시만 저장 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RefreshToken {
        private Long tokenId;
        private Long userId;
        private String tokenHash;
        private OffsetDateTime expiresAt;
        private OffsetDateTime createdAt;
    }
}
