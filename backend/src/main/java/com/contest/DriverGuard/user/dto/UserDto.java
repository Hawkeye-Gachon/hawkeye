package com.contest.DriverGuard.user.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.OffsetDateTime;

public class UserDto {

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MeResponse {
        private Long userId;
        private String email;
        private String name;
        private User.Status status;
        private OffsetDateTime createdAt;
        private boolean termsAgreementRequired;   // true 면 앱은 약관 동의 화면으로

        public static MeResponse from(User user, boolean termsAgreementRequired) {
            return MeResponse.builder()
                    .userId(user.getUserId())
                    .email(user.getEmail())
                    .name(user.getName())
                    .status(user.getStatus())
                    .createdAt(user.getCreatedAt())
                    .termsAgreementRequired(termsAgreementRequired)
                    .build();
        }
    }

    /** users 테이블 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class User {
        private Long userId;
        private String email;
        private String name;
        private Status status;
        private OffsetDateTime ageConfirmedAt;
        private OffsetDateTime createdAt;
        private OffsetDateTime updatedAt;
        private OffsetDateTime withdrawnAt;

        public enum Status {
            ACTIVE,
            WITHDRAWN
        }
    }
}
