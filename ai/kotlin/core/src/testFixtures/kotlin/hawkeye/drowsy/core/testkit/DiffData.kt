package hawkeye.drowsy.core.testkit

import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.spec.ConstantBits
import hawkeye.drowsy.core.spec.MiniJson
import java.io.File

/**
 * 시험이 읽는 저장소 파일과 차등 비교 자료(kotlin/.diffdata/)의 자리, 그리고 자료를 쓰기 전에 하는 검사.
 * :core 시험(TestData가 이것을 부른다)과 :onnx 시험이 같이 쓴다. 시험 고정물이다(배포하지 않는다).
 *
 * 경로는 Gradle이 절대 경로로 넘긴다(build.gradle.kts의 systemProperty, 선언된 입력). 자료의 형식: kotlin/difftools/FORMATS.md.
 * 자료가 다른 참조 구현에서 만들어졌으면(reference_id나 depends_on이 다르면) 건너뛰지 않고 실패한다(FORMATS.md 0.2절,
 * 리드 판정 1). READY_small.txt가 없으면 [readySmall]이 null을 돌려주고, 건너뛸지는 부르는 쪽이 정한다.
 */
object DiffData {

    private fun dirProperty(name: String): File {
        val p = System.getProperty(name) ?: error("system property $name is not set (run the tests through Gradle)")
        return File(p)
    }

    val repo: File get() = dirProperty("hawkeye.repo")
    val vectors: File get() = dirProperty("hawkeye.vectors")
    val export: File get() = dirProperty("hawkeye.export")
    val diffData: File get() = dirProperty("hawkeye.diffdata")

    @Suppress("UNCHECKED_CAST")
    fun obj(v: Any?): Map<String, Any?> = v as Map<String, Any?>

    @Suppress("UNCHECKED_CAST")
    fun list(v: Any?): List<Any?> = v as List<Any?>

    // ----- 참조 지문 (FORMATS.md 0.2절) -----

    private val MODULES = listOf(
        "drowsy/types.py", "drowsy/signals.py", "drowsy/calibrate.py", "drowsy/blink.py",
        "drowsy/step_features.py", "drowsy/rules.py", "drowsy/pipeline.py",
    )
    private val CONFIG = listOf("config/feature_spec.json", "config/rules.yaml")
    private val EXPORT = listOf("export/feature_spec_v0.json", "export/rules.json", "export/drowsiness_v0.onnx")

    private var cachedReferenceId: String? = null

    /** export/는 hawkeye.export 폴더에서, 나머지는 저장소에서 읽는다. */
    fun repoFile(p: String): File =
        if (p.startsWith("export/")) File(export, p.removePrefix("export/")) else File(repo, p)

    /** 저장소의 12개 파일로 다시 계산한 reference_id. */
    @Synchronized
    fun referenceId(): String {
        cachedReferenceId?.let { return it }
        val sb = StringBuilder()
        for (group in listOf(MODULES, CONFIG, EXPORT)) {
            for (p in group.sorted()) {
                sb.append(p).append(':').append(sha256Hex(repoFile(p).readBytes())).append('\n')
            }
        }
        val id = sha256Hex(sb.toString().toByteArray(Charsets.US_ASCII))
        cachedReferenceId = id
        return id
    }

    /** 자료 파일의 reference 객체가 이 저장소의 참조와 같은지. 다르면 실패(건너뛰지 않는다). */
    fun checkReference(data: Map<String, Any?>, what: String) {
        val ref = obj(data["reference"] ?: throw AssertionError("$what: no reference object"))
        val id = ref["reference_id"] as String
        if (id != referenceId()) {
            val diffs = ArrayList<String>()
            for ((group, names) in listOf("modules" to MODULES, "config" to CONFIG, "export" to EXPORT)) {
                val g = obj(ref[group])
                for (p in names) {
                    val mine = sha256Hex(repoFile(p).readBytes())
                    if (g[p] != mine) diffs.add("$p data=${g[p]} repo=$mine")
                }
            }
            throw AssertionError("$what: reference_id $id != repository ${referenceId()}; differing files: $diffs")
        }
    }

    // ----- READY_small.txt (FORMATS.md 0.5절) -----

    class Ready(val referenceId: String, val files: Map<String, Pair<String, Long>>)

    private var cachedReady: Ready? = null

    /** READY_small.txt. 없으면 null. 있으면 reference_id를 검사한다(다르면 실패). */
    @Synchronized
    fun readySmall(): Ready? {
        cachedReady?.let { return it }
        val f = File(diffData, "READY_small.txt")
        if (!f.isFile) return null
        var refId: String? = null
        val files = LinkedHashMap<String, Pair<String, Long>>()
        for (line in f.readText(Charsets.UTF_8).lines()) {
            if (line.isBlank()) continue
            if (line.startsWith("#")) {
                val body = line.removePrefix("#").trim()
                if (body.startsWith("reference_id:")) refId = body.removePrefix("reference_id:").trim()
                continue
            }
            val parts = line.trim().split(Regex("\\s+"), limit = 3)
            require(parts.size == 3) { "READY_small.txt: bad line '$line'" }
            files[parts[2]] = parts[0] to parts[1].toLong()
        }
        val id = refId ?: throw AssertionError("READY_small.txt has no reference_id line")
        if (id != referenceId()) throw AssertionError("READY_small.txt reference_id $id != repository ${referenceId()}")
        val r = Ready(id, files)
        cachedReady = r
        return r
    }

    /** READY_small.txt가 있어야 하는 시험: 없으면 실패한다. */
    fun requireReadySmallOrFail(): Ready =
        readySmall() ?: throw AssertionError("no ${File(diffData, "READY_small.txt").path}: differential data (small) missing")

    /** 자료 파일 하나의 바이트: READY_small.txt에 있고 sha256과 크기가 맞아야 한다. */
    fun readyBytes(relPath: String): ByteArray {
        val ready = requireReadySmallOrFail()
        val entry = ready.files[relPath] ?: throw AssertionError("$relPath is not listed in READY_small.txt")
        val f = File(diffData, relPath)
        if (!f.isFile) throw AssertionError("$relPath is listed in READY_small.txt but missing")
        val bytes = f.readBytes()
        if (bytes.size.toLong() != entry.second) throw AssertionError("$relPath: ${bytes.size} bytes, READY says ${entry.second}")
        val sha = sha256Hex(bytes)
        if (sha != entry.first) throw AssertionError("$relPath: sha256 $sha, READY says ${entry.first}")
        return bytes
    }

    /**
     * 자료 파일 하나: [readyBytes]의 검사 뒤 JSON으로 읽고 참조를 검사한다. 파일에 `depends_on`이 있으면 그것도 검사한다
     * (FORMATS.md 0.2절, 리드 판정 1).
     */
    fun readyJson(relPath: String): Map<String, Any?> {
        val data = obj(MiniJson.parse(readyBytes(relPath)))
        checkReference(data, relPath)
        if (data.containsKey("depends_on")) checkDependsOn(data, relPath)
        return data
    }

    /** HKDIFF1 자료 파일 하나: [readyBytes]의 검사, 그릇 검사(Hkdiff1.read), 머리의 참조 검사. */
    fun readyHkdiff1(relPath: String): Hkdiff1 {
        val h = Hkdiff1.read(readyBytes(relPath))
        checkReference(h.header, relPath)
        return h
    }

    /**
     * FORMATS.md 0.2절 `depends_on`: reference_id 밖의 모듈에서 기대값이 나온 파일은 나열된 파일의 sha256이 저장소와 같아야
     * 한다. 다르면 실패한다(건너뛰지 않는다). 리드 판정 1.
     */
    fun checkDependsOn(data: Map<String, Any?>, what: String) {
        val deps = obj(data["depends_on"] ?: throw AssertionError("$what: no depends_on object"))
        if (deps.isEmpty()) throw AssertionError("$what: depends_on is empty")
        val diffs = ArrayList<String>()
        for ((path, sha) in deps) {
            val f = File(repo, path)
            if (!f.isFile) {
                diffs.add("$path missing in the repository")
                continue
            }
            val mine = sha256Hex(f.readBytes())
            if (mine != sha) diffs.add("$path data=$sha repo=$mine")
        }
        if (diffs.isNotEmpty()) throw AssertionError("$what: depends_on differs from the repository: $diffs")
    }

    // ----- 라이브러리 안쪽 함수의 시험용 입구 (:onnx 시험은 :core의 internal을 볼 수 없다) -----

    /** 라이브러리의 JSON 읽기(MiniJson) 그대로. */
    fun parseJson(bytes: ByteArray): Any? = MiniJson.parse(bytes)

    /** 상수 비트 지문(계획 5.11절, ConstantBits) : feature_spec 바이트의 `pipeline` 블록과 rules.json 바이트에서. */
    fun constantBits(specBytes: ByteArray, rulesBytes: ByteArray): String =
        ConstantBits.fingerprint(obj(MiniJson.parse(specBytes))["pipeline"], MiniJson.parse(rulesBytes))

    // ----- 벡터 -----

    /** 벡터 폴더의 *.json (README.md 제외), 이름 순. */
    fun vectorFiles(): List<File> =
        (vectors.listFiles() ?: emptyArray()).filter { it.isFile && it.name.endsWith(".json") }.sortedBy { it.name }

    /** 벡터 하나(이름, 확장자 없이). */
    fun vector(name: String): VectorFile = VectorFile.read(File(vectors, "$name.json").readBytes())
}
