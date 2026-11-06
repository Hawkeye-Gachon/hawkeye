package com.contest.DriverGuard.auth.controller;

import com.contest.DriverGuard.auth.dto.AuthDto;
import com.contest.DriverGuard.auth.service.AuthService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "Auth", description = "회원가입 / 로그인 / 토큰")
@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @Operation(summary = "이메일 회원가입",
            description = "가입 즉시 토큰 발급(termsAgreementRequired=true → 약관 동의 화면). "
                    + "400 VALIDATION_FAILED(fieldErrors) / 409 EMAIL_DUPLICATED")
    @PostMapping("/signup")
    public ResponseEntity<AuthDto.SignupResponse> signup(@Valid @RequestBody AuthDto.SignupRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(authService.signup(request));
    }

    @Operation(summary = "이메일 로그인",
            description = "accessToken(30분) + refreshToken(14일) 발급. termsAgreementRequired=true 면 약관 동의 화면으로. "
                    + "401 LOGIN_FAILED")
    @PostMapping("/login")
    public ResponseEntity<AuthDto.TokenResponse> login(@Valid @RequestBody AuthDto.LoginRequest request) {
        return ResponseEntity.ok(authService.login(request));
    }

    @Operation(summary = "토큰 재발급",
            description = "새 accessToken + refreshToken 발급. 사용한 refreshToken 은 폐기되어 재사용 시 401")
    @PostMapping("/refresh")
    public ResponseEntity<AuthDto.TokenResponse> refresh(@Valid @RequestBody AuthDto.RefreshTokenRequest request) {
        return ResponseEntity.ok(authService.refresh(request));
    }

    @Operation(summary = "로그아웃",
            description = "해당 refreshToken 삭제. accessToken 은 만료될 때까지 유효하므로 앱에서 삭제할 것")
    @PostMapping("/logout")
    public ResponseEntity<Void> logout(@Valid @RequestBody AuthDto.RefreshTokenRequest request) {
        authService.logout(request);
        return ResponseEntity.noContent().build();

    }
}
