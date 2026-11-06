package com.contest.DriverGuard.exception;

import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.ErrorResponse;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.HandlerMethodValidationException;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import tools.jackson.core.JacksonException;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 컨트롤러에서 발생한 예외를 ApiErrorResponse 형식으로 변환 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(ApiException.class)
    public ResponseEntity<ApiErrorResponse> handleApi(ApiException e, HttpServletRequest request) {
        ErrorCode ec = e.getErrorCode();
        return body(ApiErrorResponse.of(ec.getStatus(), ec.name(), ec.getMessage(), request.getRequestURI(),
                e.getFieldErrors()));
    }

    /** @Valid @RequestBody 검증 실패 → 입력란별 fieldErrors (입력란당 1개) */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiErrorResponse> handleBodyValidation(MethodArgumentNotValidException e,
                                                                 HttpServletRequest request) {
        Map<String, String> byField = new LinkedHashMap<>();
        e.getBindingResult().getFieldErrors()
                .forEach(fe -> byField.putIfAbsent(fe.getField(), fe.getDefaultMessage()));
        List<ApiErrorResponse.FieldError> fieldErrors = byField.entrySet().stream()
                .map(en -> new ApiErrorResponse.FieldError(en.getKey(), en.getValue()))
                .toList();
        return validationFailed(request, fieldErrors);
    }

    /** JSON 값 형식 오류 (enum 값, 시간대 없는 시각 등) → 해당 필드를 fieldErrors 로. JSON 문법 오류는 fieldErrors 없이 */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ApiErrorResponse> handleUnreadable(HttpMessageNotReadableException e,
                                                             HttpServletRequest request) {
        if (e.getCause() instanceof JacksonException je && !je.getPath().isEmpty()) {
            StringBuilder field = new StringBuilder();
            for (JacksonException.Reference ref : je.getPath()) {
                if (ref.getPropertyName() != null) {
                    field.append(field.isEmpty() ? "" : ".").append(ref.getPropertyName());
                } else if (ref.getIndex() >= 0) {
                    field.append('[').append(ref.getIndex()).append(']');
                }
            }
            return validationFailed(request,
                    List.of(new ApiErrorResponse.FieldError(field.toString(), "값의 형식이 올바르지 않습니다.")));
        }
        return validationFailed(request, null);
    }

    /** 쿼리 파라미터 누락·형식 오류 등 */
    @ExceptionHandler({MissingServletRequestParameterException.class,
            MethodArgumentTypeMismatchException.class, HandlerMethodValidationException.class})
    public ResponseEntity<ApiErrorResponse> handleBadInput(Exception e, HttpServletRequest request) {
        return validationFailed(request, null);
    }

    /** 그 외: Spring MVC 예외(404 경로 없음, 405 등)는 해당 상태로, 나머지는 500 */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiErrorResponse> handleOthers(Exception e, HttpServletRequest request) {
        if (e instanceof ErrorResponse er) {
            HttpStatusCode status = er.getStatusCode();
            HttpStatus resolved = HttpStatus.resolve(status.value());
            String code = resolved != null ? resolved.name() : "ERROR";
            String message = er.getBody().getDetail() != null ? er.getBody().getDetail() : code;
            return body(ApiErrorResponse.of(status, code, message, request.getRequestURI(), null));
        }
        log.error("Unhandled exception: {} {}", request.getMethod(), request.getRequestURI(), e);
        return body(ApiErrorResponse.of(ErrorCode.INTERNAL_ERROR, request.getRequestURI()));
    }

    private ResponseEntity<ApiErrorResponse> validationFailed(HttpServletRequest request,
                                                              List<ApiErrorResponse.FieldError> fieldErrors) {
        ErrorCode ec = ErrorCode.VALIDATION_FAILED;
        return body(ApiErrorResponse.of(ec.getStatus(), ec.name(), ec.getMessage(), request.getRequestURI(),
                fieldErrors));
    }

    private ResponseEntity<ApiErrorResponse> body(ApiErrorResponse response) {
        return ResponseEntity.status(response.getStatus()).body(response);
    }
}
