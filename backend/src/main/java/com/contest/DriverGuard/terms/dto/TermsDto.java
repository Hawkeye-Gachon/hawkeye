package com.contest.DriverGuard.terms.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.OffsetDateTime;
import java.util.List;

public class TermsDto {

    /** 약관 목록 항목 (전문 제외) */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Summary {
        private Long termsId;
        private String code;
        private String title;
        private Integer version;
        private Boolean required;
        private OffsetDateTime effectiveAt;

        public static Summary from(Terms terms) {
            return Summary.builder()
                    .termsId(terms.getTermsId())
                    .code(terms.getCode())
                    .title(terms.getTitle())
                    .version(terms.getVersion())
                    .required(terms.getRequired())
                    .effectiveAt(terms.getEffectiveAt())
                    .build();
        }
    }

    @Data
    @AllArgsConstructor
    public static class ListResponse {
        private List<Summary> terms;
    }

    /** 약관 전문 */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DetailResponse {
        private Long termsId;
        private String code;
        private String title;
        private Integer version;
        private Boolean required;
        private OffsetDateTime effectiveAt;
        private String content;   // 마크다운

        public static DetailResponse from(Terms terms) {
            return DetailResponse.builder()
                    .termsId(terms.getTermsId())
                    .code(terms.getCode())
                    .title(terms.getTitle())
                    .version(terms.getVersion())
                    .required(terms.getRequired())
                    .effectiveAt(terms.getEffectiveAt())
                    .content(terms.getContent())
                    .build();
        }
    }

    /** 동의 저장 요청 */
    @Data
    @NoArgsConstructor
    public static class AgreeRequest {
        @NotEmpty(message = "동의할 약관을 1개 이상 보내 주세요.")
        @Valid
        private List<AgreeItem> agreements;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AgreeItem {
        @NotNull(message = "termsId 를 입력해 주세요.")
        private Long termsId;

        @NotNull(message = "agreed 를 입력해 주세요.")
        private Boolean agreed;
    }

    /** 내 동의 현황 (동의 저장 응답도 같은 형식) */
    @Data
    @AllArgsConstructor
    public static class AgreementsResponse {
        private boolean termsAgreementRequired;
        private List<AgreementItem> agreements;
    }

    /** 현재 시행 중인 약관 1개 + 내 동의 여부 */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AgreementItem {
        private Long termsId;
        private String code;
        private String title;
        private Integer version;
        private Boolean required;
        private Boolean agreed;
        private OffsetDateTime agreedAt;
    }

    /** terms 테이블 */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Terms {
        private Long termsId;
        private String code;
        private Integer version;
        private String title;
        private String content;
        private Boolean required;        // is_required
        private Integer displayOrder;
        private OffsetDateTime effectiveAt;
        private OffsetDateTime createdAt;
    }
}
