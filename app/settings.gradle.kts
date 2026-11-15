pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // Pleos Connect SDK (차량 내비 연동: NaviHelper)
        maven { url = uri("https://nexus-playground.pleos.ai/repository/maven-releases/") }
    }
}

rootProject.name = "hawkeye_app"
include(":app", ":analysis", ":data", ":domain", ":entity")

// 졸음 판정 라이브러리(AI 팀의 Kotlin 포팅, ai/kotlin). 프로젝트 이름이 core·onnx 라서 모듈 좌표를 직접 이어 준다.
includeBuild("../ai/kotlin") {
    dependencySubstitution {
        substitute(module("hawkeye:drowsy-onnx")).using(project(":onnx"))
    }
}
