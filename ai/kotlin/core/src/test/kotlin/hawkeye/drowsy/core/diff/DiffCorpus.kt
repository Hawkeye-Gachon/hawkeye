package hawkeye.drowsy.core.diff

import hawkeye.drowsy.core.check.sha256Hex
import hawkeye.drowsy.core.spec.MiniJson
import hawkeye.drowsy.core.testkit.DiffData
import hawkeye.drowsy.core.testkit.Hkdiff1
import hawkeye.drowsy.core.testkit.VectorFile
import java.io.File

/**
 * 차등 비교 묶음(코퍼스)의 자리와 검사 (FORMATS.md 0.5절, 1.4절). READY_corpora.txt의 줄
 * `<index.json의 sha256>  <폴더 바이트>  <폴더>/`로 index.json을 검사하고, index.json이 적은 sha256과 크기로 세션 파일마다
 * 검사한다. 참조 지문(reference_id)이 저장소와 다르면 건너뛰지 않고 실패한다(FORMATS.md 0.2절).
 */
internal object DiffCorpus {

    class Entry(val sessionId: String, val path: String, val sha256: String, val bytes: Long, val nFed: Long, val exception: Boolean)

    class Corpus(val name: String, val entries: List<Entry>, val totals: Map<String, Any?>)

    /** READY_corpora.txt가 없으면 null(자료를 아직 내보내지 않았다; 부르는 쪽이 건너뛴다). */
    fun readyCorpora(): Map<String, String>? {
        val f = File(DiffData.diffData, "READY_corpora.txt")
        if (!f.isFile) return null
        var refId: String? = null
        val dirs = LinkedHashMap<String, String>()
        for (line in f.readText(Charsets.UTF_8).lines()) {
            if (line.isBlank()) continue
            if (line.startsWith("#")) {
                val body = line.removePrefix("#").trim()
                if (body.startsWith("reference_id:")) refId = body.removePrefix("reference_id:").trim()
                continue
            }
            val parts = line.trim().split(Regex("\\s+"), limit = 3)
            require(parts.size == 3) { "READY_corpora.txt: bad line '$line'" }
            if (parts[2].endsWith("/")) dirs[parts[2].removeSuffix("/")] = parts[0]
        }
        val id = refId ?: throw AssertionError("READY_corpora.txt has no reference_id line")
        if (id != DiffData.referenceId()) throw AssertionError("READY_corpora.txt reference_id $id != repository ${DiffData.referenceId()}")
        return dirs
    }

    fun load(name: String, ready: Map<String, String>): Corpus {
        val indexSha = ready[name] ?: throw AssertionError("$name is not listed in READY_corpora.txt")
        val f = File(File(DiffData.diffData, name), "index.json")
        if (!f.isFile) throw AssertionError("$name/index.json is listed in READY_corpora.txt but missing")
        val bytes = f.readBytes()
        val sha = sha256Hex(bytes)
        if (sha != indexSha) throw AssertionError("$name/index.json: sha256 $sha, READY_corpora.txt says $indexSha")
        val idx = VectorFile.obj(MiniJson.parse(bytes), "index.json")
        DiffData.checkReference(idx, "$name/index.json")
        if (idx["corpus"] != name) throw AssertionError("$name/index.json: corpus ${idx["corpus"]}")
        val entries = VectorFile.list(idx["sessions"], "sessions").map {
            val m = VectorFile.obj(it, "session")
            Entry(
                sessionId = m["session_id"] as String,
                path = m["path"] as String,
                sha256 = m["sha256"] as String,
                bytes = m["bytes"] as Long,
                nFed = m["n_fed"] as Long,
                exception = m["exception"] as Boolean,
            )
        }
        if (entries.size.toLong() != idx["count"] as Long) throw AssertionError("$name/index.json: count ${idx["count"]} != ${entries.size}")
        return Corpus(name, entries, VectorFile.obj(idx["totals"], "totals"))
    }

    /** 세션 파일 하나: 크기와 sha256을 index.json과, 그릇(Hkdiff1.read), 머리의 참조와 세션 id를 검사한다. */
    fun read(e: Entry): DiffSession {
        val f = File(DiffData.diffData, e.path)
        if (!f.isFile) throw AssertionError("${e.path} is listed in index.json but missing")
        val bytes = f.readBytes()
        if (bytes.size.toLong() != e.bytes) throw AssertionError("${e.path}: ${bytes.size} bytes, index.json says ${e.bytes}")
        val sha = sha256Hex(bytes)
        if (sha != e.sha256) throw AssertionError("${e.path}: sha256 $sha, index.json says ${e.sha256}")
        val h = Hkdiff1.read(bytes)
        DiffData.checkReference(h.header, e.path)
        val s = DiffSession(h)
        if (s.sid != e.sessionId) throw AssertionError("${e.path}: session_id ${s.sid} != ${e.sessionId}")
        if (s.n.toLong() != e.nFed) throw AssertionError("${e.path}: ${s.n} fed frames, index.json says ${e.nFed}")
        return s
    }
}
