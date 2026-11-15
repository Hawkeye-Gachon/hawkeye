package hawkeye.drowsy.core.spec

import hawkeye.drowsy.core.DrowsySource
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.testdata.TestData
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * SpecLoader가 실제 설정 파일을 빠짐없이 읽는지와 PyRepr(메시지 안의 Python repr)를 본다.
 *
 * 필드 비교: Python `to_plain`은 dataclass 필드 순서로 JSON을 쓰므로(spec.py:591-601), 배포 파일의 키 순서가 곧 Python 필드
 * 순서다. Kotlin data class의 componentN 순서(선언 순서)가 그 키 순서와 하나씩 맞고 값도 같아야 한다. 그래서 필드 순서,
 * 빠진 필드, 형(Int/Long/Double/String/Boolean/열거형의 wire)을 한 번에 본다.
 */
class SpecLoaderTest {

    /** data class 값의 componentN 값들(선언 순서). 자바 반사만 쓴다(kotlin-reflect가 없다). */
    private fun components(o: Any): List<Any?> {
        val methods = o.javaClass.methods
            .filter { it.name.matches(Regex("component[0-9]+")) && it.parameterCount == 0 }
            .sortedBy { it.name.removePrefix("component").toInt() }
        return methods.map { it.invoke(o) }
    }

    private fun camel(snake: String): String = snake.split('_').mapIndexed { i, p -> if (i == 0) p else p.replaceFirstChar { it.uppercase() } }.joinToString("")

    /** Kotlin 값과 JSON 나무가 같은지. 다르면 그 경로들을 out에 적는다. */
    private fun sameAs(k: Any?, j: Any?, path: String, out: MutableList<String>) {
        when {
            k is EventType -> if (j != k.wire) out.add("$path: ${k.wire} vs $j")
            k is DrowsySource -> if (j != k.wire) out.add("$path: ${k.wire} vs $j")
            k is Int -> if (j !is Long || j != k.toLong()) out.add("$path: Int $k vs $j")
            k is Long -> if (j !is Long || j != k) out.add("$path: Long $k vs $j")
            k is Double -> {
                val jd = when (j) { is Double -> j; is Long -> j.toDouble(); else -> null }
                if (jd == null || jd.toRawBits() != k.toRawBits()) out.add("$path: Double $k vs $j")
            }
            k is String || k is Boolean -> if (k != j) out.add("$path: $k vs $j")
            k is List<*> -> {
                if (j !is List<*> || j.size != k.size) out.add("$path: list $k vs $j")
                else for (i in k.indices) sameAs(k[i], j[i], "$path[$i]", out)
            }
            k != null -> {
                // data class
                if (j !is Map<*, *>) {
                    out.add("$path: object vs $j")
                    return
                }
                val comps = components(k)
                val keys = j.keys.map { it as String }
                if (comps.size != keys.size) out.add("$path: ${comps.size} fields vs ${keys.size} keys $keys")
                val names = k.javaClass.declaredFields.map { it.name }
                for (i in 0 until minOf(comps.size, keys.size)) {
                    // 선언 순서의 i번째 필드 이름 = JSON i번째 키의 camelCase
                    val fieldName = names.getOrNull(i)
                    if (fieldName != camel(keys[i])) out.add("$path: field #$i is $fieldName, key is ${keys[i]}")
                    sameAs(comps[i], j[keys[i]], "$path.${keys[i]}", out)
                }
            }
            else -> if (j != null) out.add("$path: null vs $j")
        }
    }

    @Test
    fun shippedFilesAreReadCompletelyAndInPythonFieldOrder() {
        for (f in listOf(File(TestData.export, "feature_spec_v0.json"), File(TestData.repo, "config/feature_spec.json"))) {
            val spec = SpecLoader.parseFeatureSpec(f.readBytes())
            val out = ArrayList<String>()
            sameAs(spec, TestData.parseFile(f), "feature_spec", out)
            assertEquals(emptyList(), out, f.path)
            assertEquals(44, spec.pipeline.landmarks.requiredIndices().size)
            assertEquals(11, spec.pipeline.blendshapes.requiredNames().size)
            // tests/test_extract.py:268-270: eyeBlinkLeft 9, eyeBlinkRight 10, jawOpen 25
            assertEquals(9, spec.pipeline.blendshapes.indexOf("eyeBlinkLeft"))
            assertEquals(10, spec.pipeline.blendshapes.indexOf("eyeBlinkRight"))
            assertEquals(25, spec.pipeline.blendshapes.indexOf("jawOpen"))
            assertFailsWith<IllegalArgumentException> { spec.pipeline.blendshapes.indexOf("noSuchName") }
        }
        val rulesFile = File(TestData.export, "rules.json")
        val rules = SpecLoader.parseRules(rulesFile.readBytes())
        val out = ArrayList<String>()
        sameAs(rules, TestData.parseFile(rulesFile), "rules", out)
        assertEquals(emptyList(), out)
        assertEquals(DrowsySource.PERCLOS, rules.drowsiness.source)
        assertEquals(EventType.entries.toSet(), rules.alertPriority.toSet())
        for (t in EventType.entries) assertTrue(rules.events.get(t) === components(rules.events)[t.ordinal])
        println("SpecLoaderTest: feature_spec_v0.json, config/feature_spec.json and rules.json read field by field")
    }

    /** pipeline.py:90-91 (R20): rules.clock != spec.clock → IllegalArgumentException, Python 메시지 그대로. */
    @Test
    fun clockMismatchAtPipelineConstruction() {
        val spec = SpecLoader.parseFeatureSpec(File(TestData.export, "feature_spec_v0.json").readBytes()).pipeline
        val rules = SpecLoader.parseRules(File(TestData.export, "rules.json").readBytes())
        SpecValidation.requireClockMatches(spec, rules)
        val other = rules.copy(clock = rules.clock.copy(maxHoldMs = rules.clock.maxHoldMs + 1))
        val e = assertFailsWith<IllegalArgumentException> { SpecValidation.requireClockMatches(spec, other) }
        assertEquals("rules.clock must equal spec.clock", e.message)
    }

    /** 바이트 API: JSON 문법 오류도 SpecException이다. 최상위가 객체가 아니면 Python build_dataclass와 같은 문장. */
    @Test
    fun byteApiErrors() {
        assertFailsWith<SpecException> { SpecLoader.parseRules("{".toByteArray()) }
        assertEquals("rules: expected a mapping, got list", assertFailsWith<SpecException> { SpecLoader.parseRules("[]".toByteArray()) }.message)
        assertEquals("feature_spec: expected a mapping, got NoneType",
            assertFailsWith<SpecException> { SpecLoader.parseFeatureSpec("null".toByteArray()) }.message)
    }

    /**
     * PyRepr의 기대값은 Python 3.11.16의 repr이다(src/test/resources/pyrepr_expected.jsonl; 만든 스크립트는 작업 폴더
     * work/kotlin/impl/s1/pyrepr_expected.py). 값은 JSON 글자로 주고 MiniJson으로 읽는다.
     */
    @Test
    fun pythonReprOfJsonValues() {
        val lines = javaClass.getResource("/pyrepr_expected.jsonl")!!.readText(Charsets.UTF_8).lines().filter { it.isNotBlank() }
        var n = 0
        for (line in lines) {
            val row = TestData.obj(MiniJson.parse(line))
            val value = MiniJson.parse(row["json"] as String)
            assertEquals(row["repr"], PyRepr.value(value), "repr of ${row["json"]}")
            n++
        }
        assertEquals("('none', 'log1p')", PyRepr.tuple(listOf("none", "log1p")))
        assertEquals("(60, 10)", PyRepr.tuple(listOf(60L, 10L)))
        println("SpecLoaderTest: $n Python repr cases equal")
    }

    /**
     * Python `json.dumps`는 실수를 `float.__repr__`으로 쓴다. 그래서 canary.json의 `f64` 배열 글자 10,020개가 그대로 Python repr의
     * 기대값이다. 배열 글자를 쉼표로 잘라 PyRepr.float(같은 글자를 읽은 double)와 비교한다.
     */
    @Test
    fun pythonFloatReprOnTheNumberTable() {
        TestData.readyJson("numbers/canary.json") // READY, reference 검사
        val text = File(TestData.diffData, "numbers/canary.json").readText(Charsets.UTF_8)
        val start = text.indexOf("\"f64\":[")
        assertTrue(start >= 0)
        val end = text.indexOf(']', start)
        val items = text.substring(start + "\"f64\":[".length, end).split(',')
        var mismatches = 0
        val first = ArrayList<String>()
        for (s in items) {
            val v = s.toDouble()
            val r = PyRepr.float(v)
            if (r != s) {
                mismatches++
                if (first.size < 20) first.add("$s -> $r")
            }
        }
        println("SpecLoaderTest: PyRepr.float on ${items.size} canary values, $mismatches mismatches")
        assertEquals(10020, items.size)
        assertEquals(0, mismatches, first.joinToString("\n"))
    }
}
