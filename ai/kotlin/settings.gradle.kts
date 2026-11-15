// hawkeye 졸음 판정 파이프라인의 Kotlin 이식. 계획: PLAN_KOTLIN_FINAL 3.1절.

pluginManagement {
    // 플러그인을 받는 곳은 google()과 mavenCentral() 둘뿐이다. Gradle 플러그인 포털은 쓰지 않는다(계획 3.1절).
    repositories {
        google()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "hawkeye-drowsy"

include(":core", ":onnx")
