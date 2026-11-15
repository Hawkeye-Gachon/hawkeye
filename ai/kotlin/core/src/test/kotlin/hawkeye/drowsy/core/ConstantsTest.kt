package hawkeye.drowsy.core

import hawkeye.drowsy.core.testdata.TestData
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * Constants.kt의 문자열 상수와 인덱스가 참조 구현의 코드 상수와 같은지 본다.
 *
 * 기대값은 바꾸지 않은 참조 소스(`drowsy/types.py`, `drowsy/rules.py`)의 대입문과 튜플을 글자로 읽은 것, 그리고
 * Python이 내보낸 `export/feature_spec_v0.json`의 특징 이름과 클래스 순서다. 손으로 적은 기대값은 없다.
 */
class ConstantsTest {

    private val typesPy: List<String> by lazy { File(TestData.repo, "drowsy/types.py").readLines(Charsets.UTF_8) }
    private val rulesPy: List<String> by lazy { File(TestData.repo, "drowsy/rules.py").readLines(Charsets.UTF_8) }

    /** `NAME = "value"` 꼴의 대입문, 파일 순서. */
    private fun stringConstants(lines: List<String>, prefix: String): List<Pair<String, String>> {
        val re = Regex("^($prefix[A-Z0-9_]*) = \"([^\"]*)\"\\s*(#.*)?$")
        return lines.mapNotNull { re.find(it) }.map { it.groupValues[1] to it.groupValues[2] }
    }

    /** `NAME = 123` 꼴의 대입문. */
    private fun intConstants(lines: List<String>, prefix: String): List<Pair<String, Int>> {
        val re = Regex("^($prefix[A-Z0-9_]*) = (\\d+)\\s*(#.*)?$")
        return lines.mapNotNull { re.find(it) }.map { it.groupValues[1] to it.groupValues[2].toInt() }
    }

    /** `NAME: tuple[...] = ( ... )`의 원소 글자(여러 줄일 수 있다). 따옴표 문자열 또는 이름. */
    private fun tupleItems(lines: List<String>, name: String): List<String> {
        val start = lines.indexOfFirst { it.startsWith("$name:") || it.startsWith("$name =") }
        require(start >= 0) { "$name not found" }
        val sb = StringBuilder()
        var i = start
        while (true) {
            sb.append(lines[i]).append('\n')
            if (sb.count { it == '(' } > 0 && sb.count { it == '(' } <= sb.count { it == ')' }) break
            i++
        }
        val body = sb.substring(sb.indexOf("= (") + 3, sb.lastIndexOf(")"))
        return body.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.removeSurrounding("\"") }
    }

    private fun valuesOf(consts: List<Pair<String, String>>, names: List<String>): List<String> =
        names.map { n -> consts.first { it.first == n }.second }

    @Test
    fun enumWireStringsAndOrderEqualTypesPy() {
        val strs = stringConstants(typesPy, "")
        assertEquals(valuesOf(strs, tupleItems(typesPy, "EVENT_TYPES")), EventType.entries.map { it.wire })
        assertEquals(valuesOf(strs, tupleItems(typesPy, "PHASES")), Phase.entries.map { it.wire })
        assertEquals(valuesOf(strs, tupleItems(typesPy, "SOURCES")), Source.entries.map { it.wire })
        assertEquals(valuesOf(strs, tupleItems(typesPy, "DROWSY_SOURCES")), DrowsySource.entries.map { it.wire })
        assertEquals(valuesOf(strs, tupleItems(typesPy, "BASELINE_STATUSES")), BaselineStatus.entries.map { it.wire })
        assertEquals(stringConstants(rulesPy, "ZONE_").map { it.second }, GazeZone.entries.map { it.wire })
        // Kotlin 이름은 Python 상수 이름의 접두어 뒤와 같다(EVENT_GAZE_AWAY_LONG → GAZE_AWAY_LONG).
        assertEquals(tupleItems(typesPy, "EVENT_TYPES").map { it.removePrefix("EVENT_") }, EventType.entries.map { it.name })
        assertEquals(tupleItems(typesPy, "SOURCES").map { it.removePrefix("SOURCE_") }, Source.entries.map { it.name })
    }

    @Test
    fun orderedNameListsAndIndicesEqualTypesPy() {
        assertEquals(tupleItems(typesPy, "CLASS_ORDER"), CLASS_ORDER)
        assertEquals(tupleItems(typesPy, "FEATURE_NAMES"), FEATURE_NAMES)
        assertEquals(tupleItems(typesPy, "EYE_LOOK_ORDER"), EYE_LOOK_ORDER)
        assertEquals(intConstants(typesPy, "N_FEATURES").single().second, N_FEATURES)
        val kotlinIndex = mapOf(
            "CLASS_ALERT" to CLASS_ALERT, "CLASS_LOW_VIGILANT" to CLASS_LOW_VIGILANT, "CLASS_DROWSY" to CLASS_DROWSY,
            "F_PERCLOS_60" to F_PERCLOS_60, "F_PERCLOS_10" to F_PERCLOS_10, "F_CLOSURE_MEAN_60" to F_CLOSURE_MEAN_60,
            "F_CLOSURE_MEAN_10" to F_CLOSURE_MEAN_10, "F_BLINK_RATE_60" to F_BLINK_RATE_60,
            "F_BLINK_DUR_MEAN_60" to F_BLINK_DUR_MEAN_60, "F_BLINK_DUR_STD_60" to F_BLINK_DUR_STD_60,
            "F_CLOSING_VEL_MEAN_60" to F_CLOSING_VEL_MEAN_60, "F_OPENING_VEL_MEAN_60" to F_OPENING_VEL_MEAN_60,
            "F_LONG_CLOSURE_COUNT_60" to F_LONG_CLOSURE_COUNT_60, "F_MAX_CLOSURE_MS_60" to F_MAX_CLOSURE_MS_60,
            "F_MAX_CLOSURE_MS_10" to F_MAX_CLOSURE_MS_10, "F_YAWN_COUNT_60" to F_YAWN_COUNT_60,
            "F_PITCH_STD_60" to F_PITCH_STD_60, "F_NOD_COUNT_60" to F_NOD_COUNT_60,
            "F_FACE_MISSING_RATIO_60" to F_FACE_MISSING_RATIO_60, "F_FACE_MISSING_RATIO_10" to F_FACE_MISSING_RATIO_10,
            "EL_IN_L" to EL_IN_L, "EL_OUT_L" to EL_OUT_L, "EL_UP_L" to EL_UP_L, "EL_DOWN_L" to EL_DOWN_L,
            "EL_IN_R" to EL_IN_R, "EL_OUT_R" to EL_OUT_R, "EL_UP_R" to EL_UP_R, "EL_DOWN_R" to EL_DOWN_R,
        )
        val python = (intConstants(typesPy, "CLASS_") + intConstants(typesPy, "F_") + intConstants(typesPy, "EL_")).toMap()
        assertEquals(python, kotlinIndex)
        assertEquals(tupleItems(typesPy, "IDENTITY_R0").map { it.toDouble() }, IDENTITY_R0)
    }

    @Test
    fun featureNamesAndClassOrderEqualTheExportedSpec() {
        val spec = TestData.obj(TestData.parseFile(File(TestData.export, "feature_spec_v0.json")))
        val features = TestData.list(spec["features"]).map { TestData.obj(it) }
        assertEquals(features.map { (it["index"] as Long).toInt() }, (0 until N_FEATURES).toList())
        assertEquals(features.map { it["name"] }, FEATURE_NAMES)
        assertEquals(TestData.list(spec["class_order"]), CLASS_ORDER)
    }

    @Test
    fun wireLookupRejectsUnknownStrings() {
        for (t in EventType.entries) assertEquals(t, EventType.fromWire(t.wire))
        assertFailsWith<IllegalArgumentException> { EventType.fromWire("SLEEP") }
        assertFailsWith<IllegalArgumentException> { Source.fromWire("") }
        assertFailsWith<IllegalArgumentException> { Phase.fromWire("Onset") }
        assertFailsWith<IllegalArgumentException> { DrowsySource.fromWire("both") }
        assertFailsWith<IllegalArgumentException> { BaselineStatus.fromWire("fallback") }
        assertFailsWith<IllegalArgumentException> { GazeZone.fromWire("IN") }
    }

    @Test
    fun constantListsAreReadOnly() {
        @Suppress("UNCHECKED_CAST")
        val l = FEATURE_NAMES as MutableList<String>
        assertFailsWith<UnsupportedOperationException> { l[0] = "x" }
        @Suppress("UNCHECKED_CAST")
        val c = CLASS_ORDER as MutableList<String>
        assertFailsWith<UnsupportedOperationException> { c.add("x") }
    }
}
