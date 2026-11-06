package com.contest.DriverGuard.security;

import com.contest.DriverGuard.exception.ApiException;
import com.contest.DriverGuard.exception.ErrorCode;
import com.contest.DriverGuard.terms.service.TermsService;
import com.contest.DriverGuard.user.service.UserService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

/**
 * 기능 API 호출 전 확인: 탈퇴한 사용자면 404 USER_NOT_FOUND,
 * 필수 약관에 동의하지 않았으면 403 TERMS_AGREEMENT_REQUIRED.
 * 적용 경로는 WebConfig 에서 지정. 인증은 이미 Security 필터에서 통과한 상태
 */
@Component
@RequiredArgsConstructor
public class TermsAgreementInterceptor implements HandlerInterceptor {

    private final UserService userService;
    private final TermsService termsService;

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.getPrincipal() instanceof Long userId) {
            userService.getActiveUser(userId);   // 탈퇴 후 남은 accessToken 차단 (404 USER_NOT_FOUND)
            if (termsService.isAgreementRequired(userId)) {
                throw new ApiException(ErrorCode.TERMS_AGREEMENT_REQUIRED);
            }
        }
        return true;
    }
}
