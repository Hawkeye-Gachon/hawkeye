package com.contest.DriverGuard.user.dao;

import com.contest.DriverGuard.user.dto.UserDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface UserDao {
    /** insert 후 생성된 user_id 가 user.userId 에 채워짐 */
    void insertUser(UserDto.User user);

    UserDto.User findById(@Param("userId") Long userId);

    boolean existsActiveByEmail(@Param("email") String email);

    /** ACTIVE 사용자를 WITHDRAWN 으로 바꾸고 withdrawn_at 기록 */
    void withdraw(@Param("userId") Long userId);
}