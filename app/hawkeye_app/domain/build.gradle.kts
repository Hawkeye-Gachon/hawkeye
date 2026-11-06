// 상태 판단·경고·이벤트 병합·여정 기록 규칙과 저장소/플랫폼 인터페이스. 안드로이드 의존성 없음.
plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    api(project(":entity"))
    api(libs.kotlinx.coroutines.core)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}
