package com.contest.DriverGuard.auth.dao;

import com.contest.DriverGuard.auth.dto.AuthDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface AuthDao {
    /** insert 후 생성된 auth_id 가 auth.authId 에 채워짐 */
    void insertAuth(AuthDto.Auth auth);

    /** ACTIVE 사용자의 EMAIL 인증 정보 (없으면 null) */
    AuthDto.Auth findEmailAuthByEmail(@Param("email") String email);

    void updateLastLoginAt(@Param("authId") Long authId);

    /** 사용자의 EMAIL 인증 정보 (없으면 null) */
    AuthDto.Auth findEmailAuthByUserId(@Param("userId") Long userId);

    /** 사용자의 모든 인증 수단 삭제 (탈퇴) */
    void deleteByUserId(@Param("userId") Long userId);
}