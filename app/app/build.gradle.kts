plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.personal.driverguard"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.personal.driverguard"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.2.0"

        ndk {
            // MediaPipe·OpenCV 네이티브 라이브러리 ABI. 실기기(arm64)와 에뮬레이터(x86_64, arm64)만 포함해 APK 크기를 줄인다.
            // 32비트 기기 지원이 필요하면 armeabi-v7a 를 추가한다(두 라이브러리 모두 제공).
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        // MediaPipe 가 assets 의 .task 모델을 메모리 매핑으로 읽으려면 압축하지 않아야 한다
        noCompress += "task"
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":domain"))
    implementation(project(":data"))
    implementation(project(":analysis"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.core)
    debugImplementation(libs.androidx.compose.ui.tooling)

    implementation(libs.androidx.navigation.compose)
    implementation(libs.kotlinx.serialization.json)

    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)

    // Liquid Glass 근사: 조작층의 배경 블러(API 31+ RenderEffect, 그 아래는 반투명 대체)
    implementation(libs.haze)
    // CameraX 의 ProcessCameraProvider.getInstance 가 돌려주는 ListenableFuture
    implementation(libs.guava)
    // Pleos Connect: 졸음 감지 후 차량 기본 내비에 쉼터 경유지 추가
    implementation(libs.pleos.navihelper)

}
