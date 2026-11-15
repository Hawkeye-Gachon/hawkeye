// 루트 빌드: 버전 고정, :core와 :onnx의 공통 설정, 선언된 입력. 계획: PLAN_KOTLIN_FINAL 3.1, 3.5, 8.3절.

import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension
import org.jetbrains.kotlin.gradle.dsl.KotlinVersion

plugins {
    // Kotlin 컴파일러는 2.1.21이고 언어·API는 2.0 모드다(계획 8.3절, K5).
    kotlin("jvm") version "2.1.21" apply false
    // 공개 API 목록(.api 파일)을 만들고 apiCheck로 비교한다(계획 3.5절).
    id("org.jetbrains.kotlinx.binary-compatibility-validator") version "0.18.2"
}

// ---------------------------------------------------------------------------
// 고정한 버전. 하위 모듈은 `val x: String by rootProject.extra`로 읽는다.
// ---------------------------------------------------------------------------

// KGP가 저절로 넣는 stdlib(2.1.21)를 끄고(gradle.properties의 kotlin.stdlib.default.dependency=false) 2.0.21을 직접 적는다.
val kotlinStdlibVersion by extra("2.0.21")
// JUnit 5의 마지막 5.x(Maven Central의 junit-bom maven-metadata.xml, 2026-11-15 조회). 6.x는 JUnit 5가 아니라서 쓰지 않는다.
val junitBomVersion by extra("5.14.4")
// K6 (가): 1.28.0. 데스크톱 JAR(:onnx 컴파일과 시험)과 앱이 넣는 onnxruntime-android가 같은 번호다.
val onnxRuntimeVersion by extra("1.28.0")

allprojects {
    // K3: Maven 그룹 hawkeye.
    group = "hawkeye"
    // 가정: 버전 번호 0.1.0은 리드가 고른 것이고 소유자가 정한 것이 아니다.
    version = "0.1.0"
}

// ---------------------------------------------------------------------------
// 시험이 읽는 저장소 파일. gradle.properties의 상대 경로를 루트 기준 절대 경로로 바꾼다.
// Test.workingDir의 기본값은 모듈 폴더라서 상대 경로를 그대로 넘기면 kotlin/tests/vectors를 가리킨다(계획 3.1절).
// -Phawkeye.vectors=<사본>으로 바꿀 수 있다.
// ---------------------------------------------------------------------------

fun repoPath(name: String): File = rootProject.file(providers.gradleProperty(name).get())

val vectorsDir: File = repoPath("hawkeye.vectors")
val exportDir: File = repoPath("hawkeye.export")

// 차등 비교 자료(K23 (가): kotlin/.diffdata/, git 제외)와 그 자료의 참조 지문을 다시 계산할 저장소 파일
// (difftools/FORMATS.md 0.2절: 일곱 모듈, config 둘, export 셋). 단계 0b부터 시험이 읽는다.
// 자료 폴더는 -Phawkeye.diffdata=<사본>으로 바꿀 수 있다.
val repoRootDir: File = rootProject.projectDir.parentFile
val diffDataDir: File = repoPath("hawkeye.diffdata")
val referenceFiles: List<File> = listOf(
    "drowsy/types.py", "drowsy/signals.py", "drowsy/calibrate.py", "drowsy/blink.py",
    "drowsy/step_features.py", "drowsy/rules.py", "drowsy/pipeline.py",
    "config/feature_spec.json", "config/rules.yaml",
).map { File(repoRootDir, it) }
// reference_id 밖이지만 차등 비교 자료의 기대값이 나온 모듈(FORMATS.md 0.2절 depends_on). 단계 1부터 시험이 검사한다.
val dependsOnFiles: List<File> = listOf("drowsy/spec.py", "drowsy/vectors.py").map { File(repoRootDir, it) }
// MediaPipe 모델 파일. 시작 검사 시험 셋(StartupChecksTest, OnnxStartupChecksTest, DrowsinessMonitorOpenTest)이
// tamper/manifest.json의 base.task.path(저장소 루트 기준 models/face_landmarker.task)로 읽고 sha256을 맞춰 본다.
val landmarkerTaskFile: File = File(repoRootDir, "models/face_landmarker.task")

// ---------------------------------------------------------------------------
// binary-compatibility-validator: opt-in 표시가 붙은 선언은 공개 API 목록에서 뺀다.
// 가정(A15): 0.18.2가 KGP 2.1.21에서 돌고 nonPublicMarkers가 표시된 선언을 뺀다(E0에서 확인).
// ---------------------------------------------------------------------------

apiValidation {
    nonPublicMarkers.add("hawkeye.drowsy.core.DrowsyInternalApi")
}

// ---------------------------------------------------------------------------
// :core와 :onnx의 공통 설정 (순수 Kotlin/JVM).
// ---------------------------------------------------------------------------

configure(listOf(project(":core"), project(":onnx"))) {
    pluginManager.withPlugin("org.jetbrains.kotlin.jvm") {
        // Java 쪽 대상도 11로 맞춘다. KGP는 Java와 Kotlin의 JVM 대상이 다르면 빌드를 멈춘다.
        extensions.configure<JavaPluginExtension> {
            sourceCompatibility = JavaVersion.VERSION_11
            targetCompatibility = JavaVersion.VERSION_11
        }
        tasks.withType<JavaCompile>().configureEach {
            options.release.set(11)
        }

        extensions.configure<KotlinJvmProjectExtension> {
            compilerOptions {
                // 가정(A24): 2.0 모드로 만든 클래스를 Kotlin 2.0.0 컴파일러가 읽는다. E0에서 javap -v로 메타데이터 버전을 본다.
                languageVersion.set(KotlinVersion.KOTLIN_2_0)
                apiVersion.set(KotlinVersion.KOTLIN_2_0)
                jvmTarget.set(JvmTarget.JVM_11)
                // 가정(A36): -Xjdk-release=11이 Java 11 밖의 JDK API 사용을 컴파일 오류로 만든다(E0에서 확인).
                freeCompilerArgs.add("-Xjdk-release=11")
            }
        }

        dependencies {
            "api"("org.jetbrains.kotlin:kotlin-stdlib:$kotlinStdlibVersion")
            "testImplementation"("org.jetbrains.kotlin:kotlin-test-junit5:$kotlinStdlibVersion")
            "testImplementation"(platform("org.junit:junit-bom:$junitBomVersion"))
            "testImplementation"("org.junit.jupiter:junit-jupiter")
            // JUnit 5.12부터 Gradle 시험 실행에는 launcher를 시험 실행 경로에 직접 적어야 한다.
            "testRuntimeOnly"("org.junit.platform:junit-platform-launcher")
        }

        tasks.withType<Test>().configureEach {
            useJUnitPlatform()
            // ONNX Runtime 1.29 이상의 리눅스 빌드는 기본으로 사용 통계를 보낸다. 버전과 상관없이 모든 시험에서 끈다(계획 3.3절).
            environment("ORT_DISABLE_TELEMETRY", "1")

            // 선언된 입력(계획 3.5절). 경로 문자열만 넘기면 Gradle은 파일이 바뀐 것을 모르고 시험을 UP-TO-DATE로 건너뛴다.
            // 가정(A32): 선언하지 않은 파일의 변경은 Gradle이 모른다.
            inputs.dir(vectorsDir)
                .withPropertyName("hawkeyeVectors")
                .withPathSensitivity(PathSensitivity.RELATIVE)
            inputs.dir(exportDir)
                .withPropertyName("hawkeyeExport")
                .withPathSensitivity(PathSensitivity.RELATIVE)
            // face_landmarker.task는 git에 없다(저장소 .gitignore의 models/). 없는 PC에서도 시험은 돈다(그 셋은 .diffdata도 있어야 돈다).
            inputs.files(landmarkerTaskFile)
                .withPropertyName("hawkeyeLandmarkerTask")
                .withPathSensitivity(PathSensitivity.RELATIVE)
                .optional()

            // 참조 지문의 원본 파일(export/는 위에서 선언했다).
            inputs.files(referenceFiles)
                .withPropertyName("hawkeyeReferenceFiles")
                .withPathSensitivity(PathSensitivity.RELATIVE)
            // depends_on의 원본(FORMATS.md 0.2절): comparator.json은 drowsy/vectors.py, C-loader.json은 drowsy/spec.py에서 나온다.
            inputs.files(dependsOnFiles)
                .withPropertyName("hawkeyeDependsOnFiles")
                .withPathSensitivity(PathSensitivity.RELATIVE)
            // 차등 비교 자료는 전체가 약 2.6 GB(추정)라 폴더 전체를 입력으로 두지 않는다. 지금 시험이 읽는 작은 자료와
            // READY 파일만 선언한다. 자료가 없는 PC에서도 시험은 돈다(그 시험은 건너뛴다).
            inputs.files(
                fileTree(diffDataDir) {
                    include(
                        "READY_small.txt", "replay0/**", "active_expected/**", "numbers/**", "tamper/**", "model/**",
                        "components/C-loader.json", "components/C-errors.json", "components/C-func.json",
                        "components/C-baseline.json", "components/C-detector.json", "components/C-steps.json",
                        "components/C-hysteresis.json", "components/C-engine.json", "components/C-policy.json",
                        "components/C-errors_model_inputs.hkd",
                        // 리드 판정 10(차등 비교): 묶음 여덟의 READY 줄과 index.json.
                        "READY_corpora.txt",
                        "S-quick/index.json", "S-odd/index.json", "S-edge/index.json", "F-stat/index.json",
                        "F-inj/index.json", "F-cal/index.json", "F-calib/index.json", "G-sig/index.json",
                    )
                },
            )
                .withPropertyName("hawkeyeDiffDataSmall")
                .withPathSensitivity(PathSensitivity.RELATIVE)
                .optional()
            // 차등 비교가 읽는 세션 파일(묶음 여덟, 약 1.5 GB; S-full은 돌리지 않으므로 뺀다). 가정(A32): 선언하지 않으면
            // index.json이 그대로인 채 .hkd가 바뀌어도 Gradle은 시험을 UP-TO-DATE로 건너뛴다(f_s6에서 확인). 선언한 뒤에는
            // .hkd 한 바이트의 변경으로 시험이 다시 돈다(같은 확인).
            inputs.files(
                fileTree(diffDataDir) {
                    include(
                        "S-quick/**/*.hkd", "S-odd/**/*.hkd", "S-edge/**/*.hkd", "F-stat/**/*.hkd",
                        "F-inj/**/*.hkd", "F-cal/**/*.hkd", "F-calib/**/*.hkd", "G-sig/**/*.hkd",
                    )
                },
            )
                .withPropertyName("hawkeyeDiffDataSessions")
                .withPathSensitivity(PathSensitivity.RELATIVE)
                .optional()

            systemProperty("hawkeye.repo", repoRootDir.absolutePath)
            systemProperty("hawkeye.diffdata", diffDataDir.absolutePath)
            systemProperty("hawkeye.vectors", vectorsDir.absolutePath)
            systemProperty("hawkeye.export", exportDir.absolutePath)
        }
    }
}
