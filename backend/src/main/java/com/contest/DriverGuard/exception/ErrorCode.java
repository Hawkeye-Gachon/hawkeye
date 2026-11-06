package com.contest.DriverGuard.exception;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;

/** 에러 응답의 code. 앱은 이 값으로 분기함 */
@Getter
@RequiredArgsConstructor
public enum ErrorCode {

    VALIDATION_FAILED(HttpStatus.BAD_REQUEST, "입력값을 확인해 주세요."),
    PASSWORD_MISMATCH(HttpStatus.BAD_REQUEST, "비밀번호가 일치하지 않습니다."),
    TERMS_REQUIRED_NOT_AGREED(HttpStatus.BAD_REQUEST, "필수 약관에 모두 동의해 주세요."),
    TERMS_NOT_CURRENT(HttpStatus.BAD_REQUEST, "현재 시행 중인 약관이 아닙니다. 약관을 다시 확인해 주세요."),
    INVALID_WEEK_START(HttpStatus.BAD_REQUEST, "weekStart 는 월요일이어야 합니다."),

    UNAUTHORIZED(HttpStatus.UNAUTHORIZED, "인증이 필요합니다."),
    LOGIN_FAILED(HttpStatus.UNAUTHORIZED, "이메일 또는 비밀번호가 올바르지 않습니다."),
    INVALID_REFRESH_TOKEN(HttpStatus.UNAUTHORIZED, "유효하지 않거나 만료된 리프레시 토큰입니다."),

    TERMS_AGREEMENT_REQUIRED(HttpStatus.FORBIDDEN, "필수 약관 동의가 필요합니다."),

    USER_NOT_FOUND(HttpStatus.NOT_FOUND, "사용자를 찾을 수 없습니다."),
    TERMS_NOT_FOUND(HttpStatus.NOT_FOUND, "약관을 찾을 수 없습니다."),
    DRIVE_NOT_FOUND(HttpStatus.NOT_FOUND, "주행 기록을 찾을 수 없습니다."),

    DRIVE_NOT_ENDED(HttpStatus.CONFLICT, "아직 종료되지 않은 주행입니다."),
    EMAIL_DUPLICATED(HttpStatus.CONFLICT, "이미 사용 중인 이메일입니다."),

    INTERNAL_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "서버 오류가 발생했습니다.");

    private final HttpStatus status;
    private final String message;
}
