package com.contest.DriverGuard.terms.controller;

import com.contest.DriverGuard.terms.dto.TermsDto;
import com.contest.DriverGuard.terms.service.TermsService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "Terms", description = "약관")
@RestController
@RequestMapping("/api/terms")
@RequiredArgsConstructor
public class TermsController {

    private final TermsService termsService;

    @Operation(summary = "약관 목록",
            description = "현재 시행 중인 약관을 표시 순서대로 반환 (전문 제외). 인증 불필요")
    @GetMapping
    public ResponseEntity<TermsDto.ListResponse> list() {
        return ResponseEntity.ok(termsService.getCurrentList());
    }

    @Operation(summary = "약관 전문",
            description = "content 는 마크다운. 인증 불필요. 없는 ID 면 404 TERMS_NOT_FOUND")
    @GetMapping("/{termsId}")
    public ResponseEntity<TermsDto.DetailResponse> detail(@PathVariable Long termsId) {
        return ResponseEntity.ok(termsService.getDetail(termsId));
    }
}
