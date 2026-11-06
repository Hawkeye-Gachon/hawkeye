package com.contest.DriverGuard.auth.service;

import com.contest.DriverGuard.auth.dao.AuthDao;
import com.contest.DriverGuard.auth.dao.RefreshTokenDao;
import com.contest.DriverGuard.auth.dto.AuthDto;
import com.contest.DriverGuard.drive.service.DriveService;
import com.contest.DriverGuard.exception.ApiException;
import com.contest.DriverGuard.exception.ErrorCode;
import com.contest.DriverGuard.property.JwtProperty;
import com.contest.DriverGuard.security.JwtProvider;
import com.contest.DriverGuard.terms.service.TermsService;
import com.contest.DriverGuard.user.dto.UserDto;
import com.contest.DriverGuard.user.service.UserService;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.util.Base64;
import java.util.HexFormat;
import java.util.Locale;

@Service
public class AuthService {

    private static final SecureRandom SECURE_RANDOM = new SecureRandom();

    private final UserService userService;
    private final TermsService termsService;
    private final DriveService driveService;
    private final AuthDao authDao;
    private final RefreshTokenDao refreshTokenDao;
    private final PasswordEncoder passwordEncoder;
    private final JwtProvider jwtProvider;
    private final Duration refreshTokenValidity;

    /** 없는 이메일로 로그인해도 비밀번호 비교 시간을 비슷하게 맞추기 위한 더미 해시 */
    private final String dummyPasswordHash;

    public AuthService(UserService userService, TermsService termsService, DriveService driveService,
                       AuthDao authDao, RefreshTokenDao refreshTokenDao, PasswordEncoder passwordEncoder,
                       JwtProvider jwtProvider, JwtProperty jwtProperty) {
        this.userService = userService;
        this.termsService = termsService;
        this.driveService = driveService;
        this.authDao = authDao;
        this.refreshTokenDao = refreshTokenDao;
        this.passwordEncoder = passwordEncoder;
        this.jwtProvider = jwtProvider;
        this.refreshTokenValidity = jwtProperty.refreshTokenValidity();
        this.dummyPasswordHash = passwordEncoder.encode("dummy-password");
    }

    /** 이메일 회원가입: users + auths(EMAIL) 를 하나의 트랜잭션으로 저장 */
    @Transactional
    public AuthDto.SignupResponse signup(AuthDto.SignupRequest request) {
        String email = normalizeEmail(request.getEmail());

        if (userService.existsActiveEmail(email)) {
            throw new ApiException(ErrorCode.EMAIL_DUPLICATED);
        }

        UserDto.User user = userService.createActiveUser(email, request.getName().trim(), OffsetDateTime.now());

        authDao.insertAuth(AuthDto.Auth.builder()
                .userId(user.getUserId())
                .provider(AuthDto.Auth.Provider.EMAIL)
                .passwordHash(passwordEncoder.encode(request.getPassword()))
                .build());

        // 가입 즉시 토큰 발급 → 로그인을 다시 거치지 않고 약관 동의 화면으로
        AuthDto.TokenResponse tokens = issueTokens(user.getUserId());

        return AuthDto.SignupResponse.builder()
                .userId(user.getUserId())
                .email(user.getEmail())
                .name(user.getName())
                .grantType(tokens.getGrantType())
                .accessToken(tokens.getAccessToken())
                .accessTokenExpiresIn(tokens.getAccessTokenExpiresIn())
                .refreshToken(tokens.getRefreshToken())
                .termsAgreementRequired(tokens.isTermsAgreementRequired())
                .build();
    }

    /** 이메일 로그인 → Access Token + Refresh Token 발급 */
    @Transactional
    public AuthDto.TokenResponse login(AuthDto.LoginRequest request) {
        AuthDto.Auth auth = authDao.findEmailAuthByEmail(normalizeEmail(request.getEmail()));

        // 이메일이 없는 경우와 비밀번호가 틀린 경우를 구분하지 않음 (계정 존재 여부 노출 방지)
        String hash = auth != null ? auth.getPasswordHash() : dummyPasswordHash;
        boolean matches = passwordEncoder.matches(request.getPassword(), hash);
        if (auth == null || !matches) {
            throw new ApiException(ErrorCode.LOGIN_FAILED);
        }

        authDao.updateLastLoginAt(auth.getAuthId());
        refreshTokenDao.deleteExpiredByUserId(auth.getUserId());   // 만료된 토큰 정리

        return issueTokens(auth.getUserId());
    }

    /**
     * Refresh Token 으로 재발급 (Rotation).
     * 사용한 Refresh Token 은 삭제되고 새 토큰 쌍이 발급됨 → 같은 Refresh Token 은 한 번만 사용 가능.
     */
    @Transactional
    public AuthDto.TokenResponse refresh(AuthDto.RefreshTokenRequest request) {
        Long userId = refreshTokenDao.deleteValidByHash(hash(request.getRefreshToken()));
        if (userId == null) {
            throw new ApiException(ErrorCode.INVALID_REFRESH_TOKEN);
        }

        UserDto.User user = userService.findById(userId);
        if (user == null || user.getStatus() != UserDto.User.Status.ACTIVE) {
            throw new ApiException(ErrorCode.INVALID_REFRESH_TOKEN);
        }

        return issueTokens(userId);
    }

    /** 로그아웃: 해당 기기의 Refresh Token 삭제 (Access Token 은 만료될 때까지 유효) */
    @Transactional
    public void logout(AuthDto.RefreshTokenRequest request) {
        refreshTokenDao.deleteByHash(hash(request.getRefreshToken()));
    }

    /**
     * 회원 탈퇴: 비밀번호 확인 후 users 를 WITHDRAWN 으로 바꾸고 인증 수단·토큰 삭제.
     * users 행은 남기므로 CASCADE 가 걸리지 않아 하위 데이터를 직접 지움.
     * 같은 이메일로 재가입 가능 (이메일 유니크는 ACTIVE 사용자에게만 적용)
     */
    @Transactional
    public void withdraw(Long userId, AuthDto.WithdrawRequest request) {
        userService.getActiveUser(userId);   // 이미 탈퇴했으면 404

        AuthDto.Auth auth = authDao.findEmailAuthByUserId(userId);
        if (auth == null || !passwordEncoder.matches(request.getPassword(), auth.getPasswordHash())) {
            throw new ApiException(ErrorCode.PASSWORD_MISMATCH);
        }

        userService.withdraw(userId);
        authDao.deleteByUserId(userId);          // 탈퇴자의 비밀번호 해시는 남기지 않음
        refreshTokenDao.deleteByUserId(userId);  // 모든 기기 로그아웃
        driveService.deleteAllByUser(userId);    // 주행 기록 (감지·경유지·시간대 통계는 CASCADE)
        termsService.deleteAllByUser(userId);    // 약관 동의 기록
    }

    private AuthDto.TokenResponse issueTokens(Long userId) {
        String refreshToken = generateRefreshToken();

        refreshTokenDao.insertRefreshToken(AuthDto.RefreshToken.builder()
                .userId(userId)
                .tokenHash(hash(refreshToken))
                .expiresAt(OffsetDateTime.now().plus(refreshTokenValidity))
                .build());

        return AuthDto.TokenResponse.builder()
                .grantType("Bearer")
                .accessToken(jwtProvider.createAccessToken(userId))
                .accessTokenExpiresIn(jwtProvider.getAccessTokenValiditySeconds())
                .refreshToken(refreshToken)
                .termsAgreementRequired(termsService.isAgreementRequired(userId))
                .build();
    }

    /** Refresh Token 은 JWT 가 아닌 랜덤 문자열 (DB 에서 검증하므로 서명 불필요) */
    private String generateRefreshToken() {
        byte[] bytes = new byte[32];
        SECURE_RANDOM.nextBytes(bytes);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }

    private String hash(String token) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(token.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private String normalizeEmail(String email) {
        return email.trim().toLowerCase(Locale.ROOT);
    }
}