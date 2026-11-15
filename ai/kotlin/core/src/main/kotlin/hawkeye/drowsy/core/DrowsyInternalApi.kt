package hawkeye.drowsy.core

/**
 * 라이브러리 안에서만 쓰는 선언의 표시. 다른 Gradle 모듈(:onnx)이 써야 해서 internal로 둘 수 없는 선언에 붙인다
 * (계획 3.2절, 4.1절). 공개 API 목록(binary-compatibility-validator의 nonPublicMarkers)에서 빠지고, Kotlin에서
 * opt-in 없이 쓰면 컴파일 오류가 난다.
 *
 * 가정(A16): @RequiresOptIn은 Java 호출자를 막지 못한다. README에 "앱이 쓰지 않는 것"으로 적는다.
 */
@RequiresOptIn(
    message = "Internal API of the drowsy library. Applications must not use it.",
    level = RequiresOptIn.Level.ERROR,
)
@Retention(AnnotationRetention.BINARY)
@Target(
    AnnotationTarget.CLASS,
    AnnotationTarget.FUNCTION,
    AnnotationTarget.PROPERTY,
    AnnotationTarget.CONSTRUCTOR,
    AnnotationTarget.TYPEALIAS,
)
annotation class DrowsyInternalApi
