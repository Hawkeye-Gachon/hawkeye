package com.contest.DriverGuard.terms.controller;

import com.contest.DriverGuard.configuration.SwaggerConfig;
import com.contest.DriverGuard.terms.dto.TermsDto;
import com.contest.DriverGuard.terms.service.TermsService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "Terms", description = "약관")
@SecurityRequirement(name = SwaggerConfig.BEARER)
@RestController
@RequestMapping("/api/users/me/terms-agreements")
@RequiredArgsConstructor
public class TermsAgreementController {

    private final TermsService termsService;

    @Operation(summary = "내 약관 동의 현황",
            description = "현재 시행 중인 약관마다 내 동의 여부. 약관 재동의 화면에서 이전 선택을 미리 채울 때 사용")
    @GetMapping
    public ResponseEntity<TermsDto.AgreementsResponse> myAgreements(
            @Parameter(hidden = true) @AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(termsService.getMyAgreements(userId));
    }

    @Operation(summary = "약관 동의 저장 (동의하고 시작하기)",
            description = "현재 필수 약관이 모두 동의 상태여야 함. 보내지 않은 약관은 기존 상태 유지. "
                    + "400 TERMS_REQUIRED_NOT_AGREED / 400 TERMS_NOT_CURRENT(예전 버전) / 404 TERMS_NOT_FOUND")
    @PostMapping
    public ResponseEntity<TermsDto.AgreementsResponse> agree(
            @Parameter(hidden = true) @AuthenticationPrincipal Long userId,
            @Valid @RequestBody TermsDto.AgreeRequest request) {
        return ResponseEntity.ok(termsService.agree(userId, request));
    }
}
