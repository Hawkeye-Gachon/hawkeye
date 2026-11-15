package hawkeye.drowsy

import hawkeye.drowsy.core.check.VerifiedValues

// 이 빌드가 검증한 값 (SPEC 15장 H3, H4, 계획 5.11절의 상수 비트 지문, 리드 판정 9의 VerifiedValues).
// 판정 9가 고른 가장 작은 꼴이다: Gradle 생성 작업과 결과 파일(계획 3.5절, 7장) 대신 손으로 적은 상수 한 파일과, 그 값을
// tests/vectors와 번들 파일에서 다시 계산해 다르면 실패하는 시험(:onnx VerifiedConstantsTest)이다. 같은 시험 작업
// (:onnx:test)의 AllVectorsTest가 벡터 102개 전체를 이 번들 파일로 통과시킨다.
// 벡터나 export/의 파일이 바뀌면 VerifiedConstantsTest가 실패하고, 새 값을 이 파일에 적는 것은 사람의 일이다.

internal object VerifiedConstants {

    /** 벡터 파일 모두와 번들 feature_spec_v0.json의 pipeline_hash. */
    const val PIPELINE_HASH: String = "650495e5c8a158db9a071457713f852af846e54e32d80352376592768f9c31a6"

    /** 번들 rules.json 바이트의 sha256 = 벡터 파일 모두의 rules_hash. */
    const val RULES_HASH: String = "9c2d05f29c4c4339d4759425f996c38137677d7216ad73c27091e75fb6cec7b4"

    /**
     * 벡터 묶음 지문(계획 3.5절): tests/vectors의 *.json 파일마다 `<파일 이름> <sha256>` 한 줄, 이름 순, 줄을 `\n`으로 이은
     * UTF-8 문자열(끝에 줄바꿈 없음)의 sha256.
     * 가정: 계획 3.5절의 "위 줄들"은 파일마다의 `이름 sha256` 줄만 뜻하고, 잇는 글자는 `\n`이며 이름은 확장자를 포함한
     * 파일 이름이다(계획이 정하지 않아 ConstantBits와 같은 이음 규칙을 골랐다).
     */
    const val VECTOR_SET: String = "b8eac288f759a8b03a00e20c54f406bafafbcd50537ec8298e2617e92f1a020e"

    /** 벡터 파일 수. */
    const val VECTOR_COUNT: Int = 102

    /** 번들 feature_spec_v0.json의 pipeline 블록과 rules.json의 상수 비트 지문(계획 5.11절, ConstantBits). */
    const val CONSTANT_BITS: String = "005754d304f61b14487156aa7a5912845241643d3d257a424d73c60088ebae5f"

    /** 시작 검사에 넘기는 값. */
    val VALUES: VerifiedValues = VerifiedValues(
        pipelineHash = PIPELINE_HASH,
        rulesHash = RULES_HASH,
        vectorSet = VECTOR_SET,
        vectorCount = VECTOR_COUNT,
        constantBits = CONSTANT_BITS,
    )
}
