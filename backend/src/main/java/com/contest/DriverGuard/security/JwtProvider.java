package com.contest.DriverGuard.security;

import com.contest.DriverGuard.property.JwtProperty;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.Date;
import javax.crypto.SecretKey;

/** Access Token(JWT) 발급/검증 */
@Component
public class JwtProvider {

    private final SecretKey key;
    private final Duration accessTokenValidity;

    public JwtProvider(JwtProperty jwtProperty) {
        this.key = Keys.hmacShaKeyFor(Decoders.BASE64.decode(jwtProperty.secret()));
        this.accessTokenValidity = jwtProperty.accessTokenValidity();
    }

    public String createAccessToken(Long userId) {
        Date now = new Date();
        return Jwts.builder()
                .subject(userId.toString())
                .issuedAt(now)
                .expiration(new Date(now.getTime() + accessTokenValidity.toMillis()))
                .signWith(key)
                .compact();
    }

    /** 서명/만료 검증 후 userId 반환, 유효하지 않으면 null */
    public Long parseUserId(String token) {
        try {
            String subject = Jwts.parser()
                    .verifyWith(key)
                    .build()
                    .parseSignedClaims(token)
                    .getPayload()
                    .getSubject();
            return Long.parseLong(subject);
        } catch (JwtException | IllegalArgumentException e) {
            return null;
        }
    }

    public long getAccessTokenValiditySeconds() {
        return accessTokenValidity.toSeconds();
    }
}