package com.contest.DriverGuard.user.service;

import com.contest.DriverGuard.exception.ApiException;
import com.contest.DriverGuard.exception.ErrorCode;
import com.contest.DriverGuard.user.dao.UserDao;
import com.contest.DriverGuard.user.dto.UserDto;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserDao userDao;

    /** 없으면 null */
    @Transactional(readOnly = true)
    public UserDto.User findById(Long userId) {
        return userDao.findById(userId);
    }

    @Transactional(readOnly = true)
    public UserDto.User getActiveUser(Long userId) {
        UserDto.User user = userDao.findById(userId);
        if (user == null || user.getStatus() != UserDto.User.Status.ACTIVE) {
            throw new ApiException(ErrorCode.USER_NOT_FOUND);
        }
        return user;
    }

    @Transactional(readOnly = true)
    public boolean existsActiveEmail(String email) {
        return userDao.existsActiveByEmail(email);
    }

    /** ACTIVE 사용자를 WITHDRAWN 으로 변경 */
    @Transactional
    public void withdraw(Long userId) {
        userDao.withdraw(userId);
    }

    /** email 은 소문자, name 은 앞뒤 공백을 제거한 값이 들어와야 함 */
    @Transactional
    public UserDto.User createActiveUser(String email, String name, OffsetDateTime ageConfirmedAt) {
        UserDto.User user = UserDto.User.builder()
                .email(email)
                .name(name)
                .status(UserDto.User.Status.ACTIVE)
                .ageConfirmedAt(ageConfirmedAt)
                .build();
        try {
            userDao.insertUser(user);
        } catch (DuplicateKeyException e) {
            // 중복 체크와 insert 사이에 동시 가입이 들어온 경우 (uq_users_email_active)
            throw new ApiException(ErrorCode.EMAIL_DUPLICATED);
        }
        return user;
    }
}