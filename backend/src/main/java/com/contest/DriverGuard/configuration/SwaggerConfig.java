package com.contest.DriverGuard.configuration;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeType;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.security.SecurityScheme;
import org.springframework.context.annotation.Configuration;

/**
 * Swagger UI: http://localhost:8080/swagger-ui.html
 * 로그인 응답의 accessToken 을 우측 상단 Authorize 에 넣으면 인증 API 호출 가능
 */
@Configuration
@OpenAPIDefinition(info = @Info(title = "DriverGuard API", version = "v1"))
@SecurityScheme(name = SwaggerConfig.BEARER, type = SecuritySchemeType.HTTP, scheme = "bearer", bearerFormat = "JWT")
public class SwaggerConfig {
    public static final String BEARER = "bearerAuth";
}
