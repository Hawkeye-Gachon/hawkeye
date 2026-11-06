package com.contest.DriverGuard.exception;

import lombok.Getter;

import java.util.List;

/** 서비스에서 던지는 예외. GlobalExceptionHandler 가 code 를 담아 응답함 */
@Getter
public class ApiException extends RuntimeException {

    private final ErrorCode errorCode;
    private final List<ApiErrorResponse.FieldError> fieldErrors;   // 입력란별 오류가 있을 때만

    public ApiException(ErrorCode errorCode) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.fieldErrors = null;
    }

    private ApiException(ErrorCode errorCode, List<ApiErrorResponse.FieldError> fieldErrors) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.fieldErrors = fieldErrors;
    }

    /** @Valid 로 표현하기 어려운 검증(현재 시각 기준 범위 등) 실패 → 400 VALIDATION_FAILED + fieldErrors */
    public static ApiException invalidField(String field, String message) {
        return new ApiException(ErrorCode.VALIDATION_FAILED, List.of(new ApiErrorResponse.FieldError(field, message)));
    }
}
