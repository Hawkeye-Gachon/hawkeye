package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.check.sha256Hex

/**
 * 설정 상수의 비트 지문 (계획 5.11절). 새 코드다(Python에 같은 것이 없다).
 *
 * `feature_spec`의 `pipeline` 블록과 `rules.json` 전체에서, 문서에 적힌 순서대로(깊이 우선, 파일의 키 순서) 모든 수 잎을
 * `경로 = 값` 한 줄로 쓴다. 정수 잎(정수 글자, MiniJson의 Long)은 10진, 실수 잎(Double)은 IEEE 비트를 소문자 16진 16자로
 * 쓴다. 불리언, 문자열, null은 수가 아니라 줄을 만들지 않는다. 줄을 `\n`으로 이은 UTF-8 문자열의 sha256이 지문이다.
 *
 * 경로 문법(계획 5.11절이 목록 원소의 문법을 정하지 않아 이렇게 정했다): 맨 앞은 `pipeline` 또는 `rules`, 객체의 키는
 * `.키`, 목록의 원소는 `[i]`(spec.py의 오류 위치 `where[i]`와 같은 모양). 예: `pipeline.landmarks.ear_r[0] = 33`,
 * `rules.forward_zone.yaw_abs_max_deg = 4037000000000000`.
 *
 * 쓰임: 빌드 때 `:core:test`가 JVM에서 읽은 값으로 만들고(계획 3.5절), 시작 때 기기에서 같은 바이트를 읽어 다시 만들어
 * 비교한다(CONSTANT_BITS_VS_VERIFIED). H3, H4는 파일이 같은지를 보고, 이 지문은 그 파일을 이 런타임이 같은 double로
 * 읽었는지를 본다. 가정(A35): ART의 `Double.parseDouble`이 가장 가까운 double을 준다(기기마다 이 검사가 본다).
 *
 * 상태가 없다(R19).
 */
internal object ConstantBits {

    /** 지문 (소문자 16진 64자). pipelineBlock은 feature_spec 나무의 "pipeline" 값, rules는 rules.json 나무 전체다. */
    fun fingerprint(pipelineBlock: Any?, rules: Any?): String =
        sha256Hex(text(pipelineBlock, rules).toByteArray(Charsets.UTF_8))

    /** 지문을 만드는 글자열(줄을 `\n`으로 이은 것, 끝에 줄바꿈 없음). 시험과 보고에 쓴다. */
    fun text(pipelineBlock: Any?, rules: Any?): String = lines(pipelineBlock, rules).joinToString("\n")

    fun lines(pipelineBlock: Any?, rules: Any?): List<String> {
        val out = ArrayList<String>()
        walk(pipelineBlock, "pipeline", out)
        walk(rules, "rules", out)
        return out
    }

    private fun walk(node: Any?, path: String, out: MutableList<String>) {
        when (node) {
            is Map<*, *> -> for ((k, v) in node) walk(v, "$path.$k", out)
            is List<*> -> for (i in node.indices) walk(node[i], "$path[$i]", out)
            is Long -> out.add("$path = $node")
            is Double -> out.add("$path = " + java.lang.Long.toHexString(node.toRawBits()).padStart(16, '0'))
            else -> Unit // Boolean, String, null: 수가 아니다
        }
    }
}
