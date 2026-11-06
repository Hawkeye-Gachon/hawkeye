package com.contest.DriverGuard.auth.dao;

import com.contest.DriverGuard.auth.dto.AuthDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface RefreshTokenDao {
    void insertRefreshToken(AuthDto.RefreshToken refreshToken);

    /**
     * 만료되지 않은 토큰을 삭제하고 그 user_id 반환 (없거나 만료면 null).
     * 조회+삭제를 한 쿼리로 처리해서 같은 토큰으로 동시에 재발급해도 한 번만 성공함.
     */
    Long deleteValidByHash(@Param("tokenHash") String tokenHash);

    void deleteByHash(@Param("tokenHash") String tokenHash);

    void deleteExpiredByUserId(@Param("userId") Long userId);

    /** 사용자의 모든 기기 토큰 삭제 (탈퇴) */
    void deleteByUserId(@Param("userId") Long userId);
}