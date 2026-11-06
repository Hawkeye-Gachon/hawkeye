package com.contest.DriverGuard.exception;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;

import java.time.Instant;
import java.util.List;

/** 모든 에러 응답의 형식. fieldErrors 는 입력값 검증 실패일 때만 포함 */
@Data
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ApiErrorResponse {

    private Instant timestamp;
    private int status;
    private String error;
    private String code;
    private String message;
    private String path;
    private List<FieldError> fieldErrors;

    @Data
    @AllArgsConstructor
    public static class FieldError {
        private String field;
        private String message;
    }

    public static ApiErrorResponse of(HttpStatusCode status, String code, String message, String path,
                                      List<FieldError> fieldErrors) {
        HttpStatus resolved = HttpStatus.resolve(status.value());
        String error = resolved != null ? resolved.getReasonPhrase() : "";
        return new ApiErrorResponse(Instant.now(), status.value(), error, code, message, path, fieldErrors);
    }

    public static ApiErrorResponse of(ErrorCode errorCode, String path) {
        return of(errorCode.getStatus(), errorCode.name(), errorCode.getMessage(), path, null);
    }
}
