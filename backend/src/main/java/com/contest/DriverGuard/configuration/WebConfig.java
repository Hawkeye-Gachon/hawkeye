package com.contest.DriverGuard.configuration;

import com.contest.DriverGuard.security.TermsAgreementInterceptor;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
@RequiredArgsConstructor
public class WebConfig implements WebMvcConfigurer {

    private final TermsAgreementInterceptor termsAgreementInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        // 필수 약관 동의 전에는 기능 API 사용 불가 (auth, users, terms 는 동의 전에도 호출 가능)
        registry.addInterceptor(termsAgreementInterceptor)
                .addPathPatterns("/api/drives/**", "/api/reports/**", "/api/rest-places/**");
    }
}
