import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.roborazzi)
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

        // 백엔드 주소. local.properties(또는 -P)에 hawkeye.apiBaseUrl=http://<서버주소>:8080/ 으로 바꾼다.
        // 기본값 10.0.2.2 는 안드로이드 에뮬레이터에서 본 '이 컴퓨터'(localhost)다.
        val localProps = Properties().apply {
            rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
        }
        val apiBaseUrl = (project.findProperty("hawkeye.apiBaseUrl") as String?)
            ?: localProps.getProperty("hawkeye.apiBaseUrl")
            ?: "http://10.0.2.2:8080/"
        buildConfigField("String", "API_BASE_URL", "\"$apiBaseUrl\"")

    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // 제출·시험용 APK 를 바로 설치할 수 있게 debug 키로 서명한다. 스토어에 올릴 때는 전용 키로 바꾼다.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    // MediaPipe·OpenCV·ONNX Runtime 네이티브 라이브러리가 커서 ABI 별로 APK 를 나눈다.
    // arm64-v8a 는 실기기·차량, x86_64 는 에뮬레이터용이다. 32비트 기기 지원이 필요하면 armeabi-v7a 를 추가한다.
    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }
    packaging {
        // 네이티브 라이브러리를 APK 안에서 압축해 파일 크기를 줄인다(설치할 때 풀린다)
        jniLibs { useLegacyPackaging = true }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    testOptions {
        unitTests {
            // 화면 스냅숏(Robolectric + Roborazzi): ./gradlew :app:recordRoborazziDebug → app/build/screens/
            isIncludeAndroidResources = true
            all {
                it.systemProperty("robolectric.graphicsMode", "NATIVE")
                it.maxHeapSize = "3g"
            }
        }
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
    // 졸음·전방 주시 이탈 판정(AI 팀 라이브러리, ai/kotlin) + 모델 실행(ONNX Runtime) + 얼굴 랜드마크(MediaPipe)
    implementation(libs.drowsy.onnx)
    implementation(libs.onnxruntime.android)
    implementation(libs.mediapipe.tasks.vision)
    // Pleos Connect: 졸음 감지 후 차량 기본 내비에 쉼터 경유지 추가
    implementation(libs.pleos.navihelper)
    // Pleos Connect: 졸음 단계별 차량 개입(공조·무드등·창문·시트), 차량 스피커 음성 안내·음성 응답, 차량 위치
    implementation(libs.pleos.vehicle)
    implementation(libs.pleos.tts)
    implementation(libs.pleos.stt)
    implementation(libs.pleos.fusedlocation)

    debugImplementation(libs.androidx.compose.ui.test.manifest)
    testImplementation(libs.junit)
    testImplementation(libs.robolectric)
    testImplementation(platform(libs.androidx.compose.bom))
    testImplementation(libs.androidx.compose.ui.test.junit4)
    testImplementation(libs.roborazzi)
    testImplementation(libs.roborazzi.compose)

}
