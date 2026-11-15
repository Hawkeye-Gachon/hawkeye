// :onnx — ONNX 모델 어댑터(drowsy-onnx). PC와 기기에서 같은 코드가 돈다(계획 3.3절).
// 공통 설정은 루트 build.gradle.kts에 있다.

plugins {
    `java-library`
    kotlin("jvm")
}

base {
    archivesName.set("drowsy-onnx")
}

val onnxRuntimeVersion: String by rootProject.extra

dependencies {
    // 어댑터가 :core의 공개 형(ModelFn)을 구현하므로 api다.
    api(project(":core"))
    // 데스크톱 JAR로 컴파일만 한다. 기기에서는 앱이 넣는 onnxruntime-android의 같은 클래스로 돈다.
    // 가정(A6): 두 아티팩트의 ai.onnxruntime 클래스가 같아서 compileOnly로 컴파일한 코드가 기기에서 돈다.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$onnxRuntimeVersion")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$onnxRuntimeVersion")
    testImplementation(testFixtures(project(":core")))
}

// 번들 파일 셋(리드 판정 9): 저장소의 export/ 폴더(hawkeye.export 속성)에서 바이트 그대로 JAR의 hawkeye/drowsy/bundle/로
// 복사한다. Kotlin 소스 나무에는 사본을 두지 않는다. DrowsinessMonitor.open(mediapipeModelSha256)이 클래스 경로에서 읽는다.
// 받는 쪽 저장소에서 이 프로젝트는 ai/kotlin/이고 ../export가 그대로 맞다(판정 7).
val bundledFiles = listOf("drowsiness_v0.onnx", "feature_spec_v0.json", "rules.json")
val exportDir: File = rootProject.file(providers.gradleProperty("hawkeye.export").get())

tasks.named<ProcessResources>("processResources") {
    // 셋 가운데 하나라도 없으면 조용히 빠진 JAR을 만들지 않고 멈춘다(include는 없는 파일을 건너뛰기 때문이다).
    val sources = bundledFiles.map { File(exportDir, it) }
    inputs.files(sources).withPropertyName("hawkeyeBundledFiles").withPathSensitivity(PathSensitivity.RELATIVE)
    doFirst {
        val missing = sources.filter { !it.isFile }
        if (missing.isNotEmpty()) throw GradleException("bundled files missing in ${exportDir}: ${missing.map { it.name }}")
    }
    from(exportDir) {
        include(bundledFiles)
        into("hawkeye/drowsy/bundle")
    }
}
