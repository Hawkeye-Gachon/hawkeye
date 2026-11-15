package hawkeye.drowsy.core.testdata

import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.Phase
import hawkeye.drowsy.core.Source
import hawkeye.drowsy.core.spec.MiniJson
import hawkeye.drowsy.core.testkit.DiffData
import org.junit.jupiter.api.Assumptions
import java.io.File
import java.security.MessageDigest

// 시험이 읽는 저장소 파일과 차등 비교 자료(kotlin/.diffdata/)의 자리, 그리고 자료를 쓰기 전에 하는 검사.
// 단계 2부터 자리와 검사는 시험 고정물 DiffData(:onnx 시험과 같이 쓴다)에 있고, 여기는 :core 시험이 쓰던 이름을 그대로
// 둔다. 자료가 다른 참조 구현에서 만들어졌으면(reference_id가 다르면) 건너뛰지 않고 실패한다(FORMATS.md 0.2절).
// 자료가 아예 없으면(READY_small.txt가 없으면) 그 시험만 건너뛴다.
internal object TestData {

    val repo: File get() = DiffData.repo
    val vectors: File get() = DiffData.vectors
    val export: File get() = DiffData.export
    val diffData: File get() = DiffData.diffData

    fun sha256Hex(bytes: ByteArray): String {
        val d = MessageDigest.getInstance("SHA-256").digest(bytes)
        val sb = StringBuilder(64)
        for (b in d) {
            val v = b.toInt() and 0xFF
            sb.append("0123456789abcdef"[v ushr 4]).append("0123456789abcdef"[v and 0xF])
        }
        return sb.toString()
    }

    fun parseFile(f: File): Any? = MiniJson.parse(f.readBytes())

    fun obj(v: Any?): Map<String, Any?> = DiffData.obj(v)

    fun list(v: Any?): List<Any?> = DiffData.list(v)

    /** 벡터 폴더의 *.json (README.md 제외), 이름 순. */
    fun vectorFiles(): List<File> = DiffData.vectorFiles()

    /** export/rules.json의 alert_priority (참조가 쓴 배포 규칙). */
    fun shippedAlertPriority(): List<EventType> {
        val rules = obj(parseFile(File(export, "rules.json")))
        return list(rules["alert_priority"]).map { EventType.fromWire(it as String) }
    }

    /** 벡터·재생 자료의 이벤트 객체(Python 필드 이름) → DriverStateEvent. 그 밖의 키(t_ms)는 무시한다. */
    fun event(m: Map<String, Any?>): DriverStateEvent = DriverStateEvent(
        type = EventType.fromWire(m["type"] as String),
        severity = (m["severity"] as Long).toInt(),
        tStartMs = m["t_start_ms"] as Long,
        durationMs = m["duration_ms"] as Long,
        confidence = num(m["confidence"]),
        source = Source.fromWire(m["source"] as String),
        phase = Phase.fromWire(m["phase"] as String),
        provisional = m["provisional"] as Boolean,
    )

    /** JSON 수 → Double. Python이 실수 칸에 int를 넣었으면 정수 글자로 적힌다. */
    fun num(v: Any?): Double = when (v) {
        is Double -> v
        is Long -> v.toDouble()
        else -> throw AssertionError("not a number: $v")
    }

    // ----- 참조 지문 (FORMATS.md 0.2절) -----

    /** 저장소의 12개 파일로 다시 계산한 reference_id. */
    fun referenceId(): String = DiffData.referenceId()

    /** 자료 파일의 reference 객체가 이 저장소의 참조와 같은지. 다르면 실패(건너뛰지 않는다). */
    fun checkReference(data: Map<String, Any?>, what: String) = DiffData.checkReference(data, what)

    // ----- READY_small.txt (FORMATS.md 0.5절) -----

    /** READY_small.txt가 없으면 시험을 건너뛴다(자료가 아직 없다). 있으면 reference_id를 검사한다. */
    fun requireReadySmall(): DiffData.Ready {
        val ready = DiffData.readySmall()
        Assumptions.assumeTrue(ready != null, "no ${File(diffData, "READY_small.txt").path}: differential data (small) not exported yet")
        return ready!!
    }

    /** 자료 파일 하나의 바이트: READY_small.txt에 있고 sha256과 크기가 맞아야 한다. */
    fun readyBytes(relPath: String): ByteArray {
        requireReadySmall()
        return DiffData.readyBytes(relPath)
    }

    /** 자료 파일 하나: 바이트 검사 뒤 JSON으로 읽고 참조와 depends_on을 검사한다(FORMATS.md 0.2절, 리드 판정 1). */
    fun readyJson(relPath: String): Map<String, Any?> {
        requireReadySmall()
        return DiffData.readyJson(relPath)
    }

    /** FORMATS.md 0.2절 `depends_on`. 다르면 실패한다(건너뛰지 않는다). 리드 판정 1. */
    fun checkDependsOn(data: Map<String, Any?>, what: String) = DiffData.checkDependsOn(data, what)
}
