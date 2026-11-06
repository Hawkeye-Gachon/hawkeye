// 온디바이스 얼굴 분석: MediaPipe Face Landmarker 추론, OpenCV solvePnP 머리 자세, 프레임 → 관측값 변환.
// 시간 누적·판정 규칙은 여기 없고 순수 Kotlin 인 :domain 에 있다.
plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.personal.driverguard.analysis"
    compileSdk = 36

    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":domain"))
    implementation(libs.mediapipe.tasks.vision)
    implementation(libs.opencv)

    testImplementation(libs.junit)
    androidTestImplementation(libs.junit)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext.junit)
}
