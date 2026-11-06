package com.contest.DriverGuard.user.controller;

import com.contest.DriverGuard.auth.dto.AuthDto;
import com.contest.DriverGuard.auth.service.AuthService;
import com.contest.DriverGuard.configuration.SwaggerConfig;
import com.contest.DriverGuard.terms.service.TermsService;
import com.contest.DriverGuard.user.dto.UserDto;
import com.contest.DriverGuard.user.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "User", description = "사용자")
@SecurityRequirement(name = SwaggerConfig.BEARER)
@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;
    private final TermsService termsService;
    private final AuthService authService;

    @Operation(summary = "내 정보 조회", description = "Authorization: Bearer {accessToken} 필요")
    @GetMapping("/me")
    public ResponseEntity<UserDto.MeResponse> me(@Parameter(hidden = true) @AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(UserDto.MeResponse.from(userService.getActiveUser(userId),
                termsService.isAgreementRequired(userId)));
    }

    @Operation(summary = "회원 탈퇴",
            description = "비밀번호로 본인 확인 후 탈퇴. 모든 기기의 refreshToken 폐기, 같은 이메일로 재가입 가능. "
                    + "400 PASSWORD_MISMATCH / 404 USER_NOT_FOUND(이미 탈퇴)")
    @DeleteMapping("/me")
    public ResponseEntity<Void> withdraw(@Parameter(hidden = true) @AuthenticationPrincipal Long userId,
                                         @Valid @RequestBody AuthDto.WithdrawRequest request) {
        authService.withdraw(userId, request);
        return ResponseEntity.noContent().build();
    }
}
