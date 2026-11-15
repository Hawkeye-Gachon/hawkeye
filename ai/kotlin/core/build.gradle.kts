// :core — 일곱 모듈의 Kotlin 이식(drowsy-core). Kotlin 표준 라이브러리 말고는 아무것에도 의존하지 않는다(계획 3.1절).
// 공통 설정(언어 2.0 모드, jvmTarget 11, -Xjdk-release=11, stdlib 2.0.21, JUnit 5, 선언된 입력)은 루트 build.gradle.kts에 있다.

plugins {
    `java-library`
    // 가정(A14): java-test-fixtures 플러그인을 Kotlin/JVM 모듈에 쓸 수 있다. :onnx 시험이 같이 쓰는 벡터 모형과 비교기를 둔다.
    `java-test-fixtures`
    kotlin("jvm")
}

base {
    archivesName.set("drowsy-core")
}

// 시험 고정물은 배포하지 않는다. java-test-fixtures는 배포 변형을 저절로 만들므로 둘 다 뺀다(계획 3.2절).
val javaComponent = components["java"] as AdhocComponentWithVariants
javaComponent.withVariantsFromConfiguration(configurations["testFixturesApiElements"]) { skip() }
javaComponent.withVariantsFromConfiguration(configurations["testFixturesRuntimeElements"]) { skip() }
