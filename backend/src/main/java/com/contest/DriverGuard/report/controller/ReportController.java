package com.contest.DriverGuard.report.controller;

import com.contest.DriverGuard.configuration.SwaggerConfig;
import com.contest.DriverGuard.report.dto.ReportDto;
import com.contest.DriverGuard.report.service.ReportService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "Report", description = "이전 기록 (달력 · 일간 · 주간)")
@SecurityRequirement(name = SwaggerConfig.BEARER)
@RestController
@RequestMapping("/api/reports")
@RequiredArgsConstructor
public class ReportController {

    private final ReportService reportService;

    @Operation(summary = "달력 주행일", description = "주행 기록이 있는 날짜 (서울 기준). 400 VALIDATION_FAILED(month 형식)")
    @GetMapping("/calendar")
    public ResponseEntity<ReportDto.CalendarResponse> calendar(
            @Parameter(hidden = true) @AuthenticationPrincipal Long userId,
            @Parameter(description = "yyyy-MM", required = true) @RequestParam String month) {
        return ResponseEntity.ok(reportService.calendar(userId, month));
    }

    @Operation(summary = "일간 리포트",
            description = "요약 · 0~23시 시간대별 · 가장 많이 감지된 시간대 · 그날의 주행 목록. "
                    + "date 생략 시 가장 최근 주행일 (기록이 없으면 오늘). 400 VALIDATION_FAILED(date 형식)")
    @GetMapping("/daily")
    public ResponseEntity<ReportDto.DailyResponse> daily(
            @Parameter(hidden = true) @AuthenticationPrincipal Long userId,
            @Parameter(description = "yyyy-MM-dd") @RequestParam(required = false) String date) {
        return ResponseEntity.ok(reportService.daily(userId, date));
    }

    @Operation(summary = "주간 통계",
            description = "요약 · 월~일 요일별 · 시간대별(한 주 합산) · 가장 많이 감지된 시간대. "
                    + "weekStart 생략 시 가장 최근 주행일이 속한 주. 400 INVALID_WEEK_START(월요일 아님) / VALIDATION_FAILED(형식)")
    @GetMapping("/weekly")
    public ResponseEntity<ReportDto.WeeklyResponse> weekly(
            @Parameter(hidden = true) @AuthenticationPrincipal Long userId,
            @Parameter(description = "그 주의 월요일 yyyy-MM-dd") @RequestParam(required = false) String weekStart) {
        return ResponseEntity.ok(reportService.weekly(userId, weekStart));
    }
}
