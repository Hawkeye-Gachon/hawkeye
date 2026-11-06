package com.contest.DriverGuard.property;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

/** application.properties 의 jwt.* 값 */
@ConfigurationProperties(prefix = "jwt")
public record JwtProperty(
        String secret,                  // Base64 인코딩된 HMAC 키 (256bit 이상)
        Duration accessTokenValidity,
        Duration refreshTokenValidity
) {
}