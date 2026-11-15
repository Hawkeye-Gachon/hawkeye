package hawkeye.drowsy.core.spec

import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets

/**
 * 의존성 없는 JSON 읽기 (계획 5.10절, K21 (가)). 재귀 하강이다.
 *
 * 값의 형:
 * - 객체 → `Map<String, Any?>` (파일의 키 순서를 지킨다. 상수 비트 지문이 그 순서를 쓴다, 계획 5.11절)
 * - 배열 → `List<Any?>`
 * - 문자열 → `String`
 * - 정수 글자(소수점도 지수도 없는 수) → `Long`. Long을 넘으면 SpecException(R22: 조용히 줄이지 않는다)
 * - 실수 글자(소수점이나 지수가 있는 수) → `Double` (`String.toDouble()`)
 * - `true`/`false` → `Boolean`, `null` → `null`
 *
 * 정수 글자와 실수 글자를 구별한다. spec.py는 `250.0`을 int로, feature_spec에서 `1`을 float으로 받지 않으므로
 * 읽기 단계가 이 구별을 써야 한다(spec.py:540-554).
 *
 * Python `json.loads`보다 엄격한 곳(계획 5.10절; 지금 있는 두 설정 파일과 벡터에서는 결과가 같다):
 * - 한 객체 안의 중복 키를 거부한다(Python은 뒤의 값을 쓴다).
 * - BOM을 거부한다(Python도 `Unexpected UTF-8 BOM`으로 거부한다).
 * - 값 뒤의 남은 글자를 거부한다(Python도 `Extra data`로 거부한다).
 * - 바이트로 받으면 UTF-8로 엄격하게 푼다(R18). 잘못된 UTF-8은 거부한다.
 * - 배열·객체의 깊이가 [MAX_DEPTH](512)를 넘으면 거부한다. 계획 5.10절의 엄격 목록(중복 키, BOM, 남은 글자)에는 없는
 *   것이라 일부러 둔 차이로 적는다. Python에도 한계가 있지만 고정된 수가 아니다: 이 PC의 Python 3.11.16(재귀 한도 1000)은
 *   맨 위에서 부르면 깊이 993까지 받고 994부터 RecursionError를 내고, 호출 스택이 200 프레임 더 깊으면 793까지만 받는다
 *   (2026-11-15에 잼). 그래서 어느 고정값도 Python과 판정이 언제나 같을 수 없다. Python 쪽 한계가 512보다 큰 동안
 *   (위 측정의 모든 경우) 512는 Python이 받는데 Kotlin이 거부하는 쪽(더 엄격한 쪽)으로만 어긋난다. 실제 파일의 깊이는
 *   `feature_spec_v0.json` 4, `rules.json` 3, 벡터 102개에서 가장 깊은 것 6이다(2026-11-15에 Python으로 잼).
 *
 * Python `json.loads`와 같게 둔 곳: `\u` 뒤의 네 글자는 ASCII 16진 숫자(0-9, a-f, A-F)만 받는다(Python의 C 스캐너와 같다).
 * `NaN`, `Infinity`, `-Infinity` 글자를 Double로 받는다(Python의 기본 동작).
 * 유한한지는 읽기 단계의 검사가 본다(spec.py:553-554 "expected a finite float"). 문자열 안의 제어 문자(U+0000~U+001F)는
 * 거부한다(Python의 strict=True). 공백은 JSON의 넷(공백, \t, \n, \r)만이다.
 *
 * 가정(A3): JVM의 `String.toDouble()`은 가장 가까운 double을 준다(E2에서 숫자 표로 확인한다).
 *
 * 문법 오류는 모두 SpecException이다. 메시지는 영어이고 글자 위치(0부터)를 담는다. Python의 JSONDecodeError 문장과는 다르다.
 *
 * 상태가 없다(R19). 호출마다 새 Parser를 만든다.
 */
internal object MiniJson {

    /** 배열·객체의 최대 깊이. 넘으면 SpecException (스택 넘침을 막는다). */
    const val MAX_DEPTH: Int = 512

    /** UTF-8 바이트를 읽는다. BOM과 잘못된 UTF-8은 거부한다. */
    fun parse(bytes: ByteArray): Any? {
        if (bytes.size >= 3 && bytes[0] == 0xEF.toByte() && bytes[1] == 0xBB.toByte() && bytes[2] == 0xBF.toByte()) {
            throw SpecException("JSON: unexpected UTF-8 BOM at offset 0")
        }
        val decoder = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
        val text = try {
            decoder.decode(ByteBuffer.wrap(bytes)).toString()
        } catch (e: CharacterCodingException) {
            throw SpecException("JSON: input is not valid UTF-8 (${e.javaClass.simpleName})")
        }
        return parse(text)
    }

    /** 이미 푼 글자열을 읽는다. 첫 글자가 BOM(U+FEFF)이면 거부한다. */
    fun parse(text: String): Any? {
        if (text.isNotEmpty() && text[0] == '﻿') {
            throw SpecException("JSON: unexpected UTF-8 BOM at offset 0")
        }
        return Parser(text).parseDocument()
    }

    private class Parser(private val s: String) {
        private var i = 0
        private var depth = 0

        fun parseDocument(): Any? {
            skipWhitespace()
            val value = parseValue()
            skipWhitespace()
            if (i != s.length) fail("extra data after the value")
            return value
        }

        private fun fail(what: String): Nothing = throw SpecException("JSON: $what at offset $i")

        private fun skipWhitespace() {
            while (i < s.length) {
                val c = s[i]
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i++ else return
            }
        }

        private fun parseValue(): Any? {
            if (i >= s.length) fail("expected a value, got end of input")
            return when (val c = s[i]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> parseString()
                't' -> literal("true", true)
                'f' -> literal("false", false)
                'n' -> literal("null", null)
                'N' -> literal("NaN", Double.NaN)
                'I' -> literal("Infinity", Double.POSITIVE_INFINITY)
                else -> if (c == '-' || (c in '0'..'9')) parseNumber() else fail("unexpected character '${printable(c)}'")
            }
        }

        private fun literal(word: String, value: Any?): Any? {
            if (!s.startsWith(word, i)) fail("invalid literal")
            i += word.length
            return value
        }

        private fun enter() {
            depth++
            if (depth > MAX_DEPTH) fail("nesting deeper than $MAX_DEPTH")
        }

        private fun parseObject(): Map<String, Any?> {
            enter()
            i++ // '{'
            val map = LinkedHashMap<String, Any?>()
            skipWhitespace()
            if (i < s.length && s[i] == '}') {
                i++
                depth--
                return map
            }
            while (true) {
                skipWhitespace()
                if (i >= s.length || s[i] != '"') fail("expected a property name in double quotes")
                val keyOffset = i
                val key = parseString()
                if (map.containsKey(key)) {
                    throw SpecException("JSON: duplicate key \"$key\" at offset $keyOffset")
                }
                skipWhitespace()
                if (i >= s.length || s[i] != ':') fail("expected ':'")
                i++
                skipWhitespace()
                map[key] = parseValue()
                skipWhitespace()
                if (i >= s.length) fail("expected ',' or '}', got end of input")
                when (s[i]) {
                    ',' -> i++
                    '}' -> {
                        i++
                        depth--
                        return map
                    }
                    else -> fail("expected ',' or '}'")
                }
            }
        }

        private fun parseArray(): List<Any?> {
            enter()
            i++ // '['
            val list = ArrayList<Any?>()
            skipWhitespace()
            if (i < s.length && s[i] == ']') {
                i++
                depth--
                return list
            }
            while (true) {
                skipWhitespace()
                list.add(parseValue())
                skipWhitespace()
                if (i >= s.length) fail("expected ',' or ']', got end of input")
                when (s[i]) {
                    ',' -> i++
                    ']' -> {
                        i++
                        depth--
                        return list
                    }
                    else -> fail("expected ',' or ']'")
                }
            }
        }

        private fun parseString(): String {
            i++ // 여는 '"'
            val start = i
            // 이스케이프가 없는 흔한 경우는 substring 하나로 끝낸다.
            while (i < s.length) {
                val c = s[i]
                if (c == '"') {
                    val out = s.substring(start, i)
                    i++
                    return out
                }
                if (c == '\\') break
                if (c < ' ') fail("unescaped control character in a string")
                i++
            }
            if (i >= s.length) fail("unterminated string")
            val sb = StringBuilder().append(s, start, i)
            while (true) {
                if (i >= s.length) fail("unterminated string")
                val c = s[i]
                when {
                    c == '"' -> {
                        i++
                        return sb.toString()
                    }
                    c == '\\' -> {
                        i++
                        if (i >= s.length) fail("unterminated escape")
                        when (val e = s[i]) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> {
                                if (i + 4 >= s.length) fail("incomplete \\u escape")
                                var code = 0
                                for (k in 1..4) {
                                    val d = asciiHexDigit(s[i + k])
                                    if (d < 0) fail("invalid \\u escape")
                                    code = code * 16 + d
                                }
                                // Python처럼 짝 없는 대리 문자도 그대로 둔다. 짝이 있으면 두 글자가 이어져 한 글자가 된다.
                                sb.append(code.toChar())
                                i += 4
                            }
                            else -> fail("invalid escape '\\${printable(e)}'")
                        }
                        i++
                    }
                    c < ' ' -> fail("unescaped control character in a string")
                    else -> {
                        sb.append(c)
                        i++
                    }
                }
            }
        }

        /** JSON 수 문법 `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`, 그리고 Python이 받는 `-Infinity`. */
        private fun parseNumber(): Any {
            val start = i
            if (s[i] == '-') {
                i++
                if (s.startsWith("Infinity", i)) {
                    i += "Infinity".length
                    return Double.NEGATIVE_INFINITY
                }
            }
            if (i >= s.length) fail("expected a digit")
            if (s[i] == '0') {
                i++
            } else if (s[i] in '1'..'9') {
                while (i < s.length && s[i] in '0'..'9') i++
            } else {
                fail("expected a digit")
            }
            var isFloat = false
            if (i < s.length && s[i] == '.') {
                isFloat = true
                i++
                if (i >= s.length || s[i] !in '0'..'9') fail("expected a digit after '.'")
                while (i < s.length && s[i] in '0'..'9') i++
            }
            if (i < s.length && (s[i] == 'e' || s[i] == 'E')) {
                isFloat = true
                i++
                if (i < s.length && (s[i] == '+' || s[i] == '-')) i++
                if (i >= s.length || s[i] !in '0'..'9') fail("expected a digit in the exponent")
                while (i < s.length && s[i] in '0'..'9') i++
            }
            val text = s.substring(start, i)
            if (isFloat) return text.toDouble()
            return text.toLongOrNull()
                ?: throw SpecException("JSON: integer $text does not fit in 64 bits at offset $start")
        }

        /**
         * ASCII 16진 숫자 한 글자의 값, 아니면 -1. Python `json`의 C 스캐너처럼 '0'-'9', 'a'-'f', 'A'-'F'만 받는다.
         * `Character.digit(c, 16)`은 쓰지 않는다: 아라비아-인도 숫자(U+0660~), 전각 숫자·글자(U+FF10~, U+FF21~)도 받아서
         * Python이 거부하는 `"\u٠٠٤١"`를 "A"로 읽게 된다.
         */
        private fun asciiHexDigit(c: Char): Int = when (c) {
            in '0'..'9' -> c - '0'
            in 'a'..'f' -> c - 'a' + 10
            in 'A'..'F' -> c - 'A' + 10
            else -> -1
        }

        private fun printable(c: Char): String =
            if (c < ' ' || c > '~') "\\u" + c.code.toString(16).padStart(4, '0') else c.toString()
    }
}
