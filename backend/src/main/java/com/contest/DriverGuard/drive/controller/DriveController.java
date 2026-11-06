package com.contest.DriverGuard.drive.controller;

import com.contest.DriverGuard.configuration.SwaggerConfig;
import com.contest.DriverGuard.drive.dto.DriveDto;
import com.contest.DriverGuard.drive.service.DriveService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "Drive", description = "주행 관찰")
@SecurityRequirement(name = SwaggerConfig.BEARER)
@RestController
@RequestMapping("/api/drives")
@RequiredArgsConstructor
public class DriveController {

    private final DriveService driveService;

    @Operation(summary = "관찰 시작 (주행 등록)",
            description = "새로 만들면 201, 같은 clientDriveId 재전송이면 기존 주행을 200 으로 반환. "
                    + "진행 중 주행이 있으면 ABANDONED 로 바꾸고 그 ID 를 abandonedDriveId 로 반환 "
                    + "(새 startedAt 이 더 이르면 새 주행을 ABANDONED 로 생성). "
                    + "400 VALIDATION_FAILED(UUID 형식, startedAt 은 현재 -7일 ~ +5분) / 403 TERMS_AGREEMENT_REQUIRED")
    @PostMapping
    public ResponseEntity<DriveDto.StartResponse> start(@Parameter(hidden = true) @AuthenticationPrincipal Long userId,
                                                        @Valid @RequestBody DriveDto.StartRequest request) {
        DriveDto.StartResponse response = driveService.start(userId, request);
        return ResponseEntity.status(response.isCreated() ? HttpStatus.CREATED : HttpStatus.OK).body(response);
    }

    @Operation(summary = "관찰 종료 · 기록 업로드",
            description = "종료 시각, 이동 거리(앱 계산), 감지 기록, 단순화한 경로(앱 처리), 경유지(방문 시각 포함)를 한 번에 업로드하고 "
                    + "리포트를 반환. 이미 종료된 주행이면 본문을 무시하고 기존 리포트 반환. "
                    + "주행 기간 밖·중복 항목과 없는 장소는 건너뛰고 skipped 로 개수를 알려 줌. "
                    + "400 VALIDATION_FAILED / 404 DRIVE_NOT_FOUND")
    @PostMapping("/{driveId}/end")
    public ResponseEntity<DriveDto.Report> end(@Parameter(hidden = true) @AuthenticationPrincipal Long userId,
                                               @PathVariable Long driveId,
                                               @Valid @RequestBody DriveDto.EndRequest request) {
        return ResponseEntity.ok(driveService.end(userId, driveId, request));
    }

    @Operation(summary = "관찰 리포트 (주행 상세)",
            description = "404 DRIVE_NOT_FOUND(없거나 다른 사용자 주행) / 409 DRIVE_NOT_ENDED(종료 업로드 전)")
    @GetMapping("/{driveId}")
    public ResponseEntity<DriveDto.Report> report(@Parameter(hidden = true) @AuthenticationPrincipal Long userId,
                                                  @PathVariable Long driveId) {
        return ResponseEntity.ok(driveService.getReport(userId, driveId));
    }

    @Operation(summary = "주행 목록",
            description = "종료된 주행을 최근 시작순으로 반환. 홈 미리보기는 size=1. "
                    + "다음 페이지는 응답의 nextCursor 를 cursor 로 전달 (마지막 페이지면 null)")
    @GetMapping
    public ResponseEntity<DriveDto.ListResponse> list(@Parameter(hidden = true) @AuthenticationPrincipal Long userId,
                                                      @Parameter(description = "1~50, 기본 20")
                                                      @RequestParam(required = false) Integer size,
                                                      @RequestParam(required = false) String cursor) {
        return ResponseEntity.ok(driveService.getList(userId, size, cursor));
    }
}
