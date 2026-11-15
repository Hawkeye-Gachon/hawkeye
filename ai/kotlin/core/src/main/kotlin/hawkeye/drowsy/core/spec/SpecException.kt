package hawkeye.drowsy.core.spec

/**
 * 설정 파일(feature_spec, rules.json)을 읽거나 검사하다 생긴 오류 (spec.py:60 `SpecError(ValueError)`).
 * 메시지는 Python과 같은 영어 문장이다(계획 4.1절, 5.10절). JSON 문법 오류도 이 예외다(MiniJson).
 */
class SpecException(message: String) : IllegalArgumentException(message)
