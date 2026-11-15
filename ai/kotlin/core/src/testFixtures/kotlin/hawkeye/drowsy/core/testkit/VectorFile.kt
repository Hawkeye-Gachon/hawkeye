package hawkeye.drowsy.core.testkit

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.BaselineStatus
import hawkeye.drowsy.core.Point
import hawkeye.drowsy.core.RawFrame
import hawkeye.drowsy.core.spec.MiniJson
import java.util.Collections

// 벡터 파일 모형 (계획 3.2절, 9.0절; 형식: tests/vectors/README.md, 원본 읽기: drowsy/vectors.py:386-430, 729-771).
// 시험 고정물이다(배포하지 않는다). :onnx 시험도 같이 쓴다.
// 읽기는 라이브러리의 MiniJson이다(실행기가 쓰는 JSON 읽기 = 숫자 표가 비트를 확인한 읽기, 계획 9.2.2절 끝).

/** 벡터의 model 칸 (README.md: null, stub, onnx). */
sealed class VectorModel {
    /** 대체 모델: 부를 때마다 probs의 다음 행(넘치면 마지막 행, vectors.py:525-555). */
    data class Stub(val probs: List<List<Double>>) : VectorModel()

    /** ONNX 파일: export/의 파일 이름과 sha256. */
    data class Onnx(val file: String, val sha256: String) : VectorModel()
}

/**
 * 벡터 파일 하나. expected는 JSON 나무 그대로다(비교기가 Python `_compare`처럼 나무끼리 비교한다).
 * frames는 RawFrame으로 바꾼 것이다(vectors.py:399-409 `frame_from_json`).
 */
class VectorFile(
    val name: String,
    val bundle: String,
    val description: String,
    val pipelineHash: String,
    val rulesHash: String,
    /** 항목 → 허용 오차 키 → 값. 값은 JSON 그대로(Long 또는 Double): 메시지의 표기가 Python과 같아야 한다. */
    val tolerances: Map<String, Map<String, Any?>>,
    val baseline: Baseline?,
    /** 점 경로 → 값 (JSON 나무 그대로). */
    val configOverrides: Map<String, Any?>,
    val model: VectorModel?,
    val frames: List<RawFrame>,
    /** [t_ms, kmh] 목록, t_ms 순. */
    val speed: List<Pair<Long, Double>>,
    /** model 묶음의 model_inputs ([n][120][17]) 또는 null. */
    val modelInputs: List<List<List<Double>>>?,
    val expected: Map<String, Any?>,
    /** 파일 전체의 JSON 나무. */
    val root: Map<String, Any?>,
) {
    /**
     * 그 프레임의 속도 (vectors.py:609-612): t_ms 이하인 마지막 항목의 값, 없으면 null.
     * 목록은 t_ms 순이어야 한다(vectors.py:599-600, 읽을 때 검사한다).
     */
    fun speedAt(tMs: Long): Double? {
        var lo = 0
        var hi = speed.size
        while (lo < hi) { // bisect_right
            val mid = (lo + hi) ushr 1
            if (speed[mid].first <= tMs) lo = mid + 1 else hi = mid
        }
        return if (lo - 1 >= 0) speed[lo - 1].second else null
    }

    /**
     * 벡터 실행기의 0단계 (계획 9.0절, SPEC 16장, vectors.py:739-742, 758-770). 문제가 없으면 빈 목록이다.
     * 문제가 있으면 그 파일은 아무것도 비교하지 않는다("상수가 다르다").
     * - pipeline_hash = 시험하는 feature_spec의 문자열, rules_hash = 시험하는 rules.json 바이트의 sha256
     * - name = 파일 이름(확장자 제외)
     * - tolerances의 키와 **값**이 [TOLERANCES]와 같다(값이 느슨한 파일은 거부한다)
     */
    fun stageZeroProblems(fileStem: String, specPipelineHash: String, rulesSha256: String): List<String> {
        val problems = ArrayList<String>()
        if (pipelineHash != specPipelineHash) {
            problems.add("pipeline_hash is $pipelineHash, current spec has $specPipelineHash")
        }
        if (rulesHash != rulesSha256) problems.add("rules_hash is $rulesHash, current rules have $rulesSha256")
        if (name != fileStem) problems.add("name is '$name', file is $fileStem.json")
        for ((section, tol) in tolerances) {
            val want = TOLERANCES[section]
            if (want == null) {
                if (tol.isNotEmpty()) problems.add("tolerances.$section: not a section with tolerances")
                continue
            }
            if (tol.keys != want.keys) {
                problems.add("tolerances.$section: expected the keys ${want.keys.sorted()}")
                continue
            }
            for ((k, v) in want) {
                val got = tol[k]
                val gotD = when (got) {
                    is Long -> got.toDouble()
                    is Double -> got
                    else -> Double.NaN
                }
                if (gotD != v) problems.add("tolerances.$section.$k is $got, the table has $v")
            }
        }
        for (section in expected.keys) {
            if (section in TOLERANCES && section !in tolerances) problems.add("tolerances.$section: missing")
        }
        return problems
    }

    companion object {
        /**
         * 항목별 허용 오차. drowsy/vectors.py:224-235 `TOLERANCES`를 옮겨 적었다(참조 구현의 표, SPEC 16장 표와 같은 값).
         * 표에 없는 항목(counters)과 표에 없는 값은 정확히 같아야 한다.
         */
        val TOLERANCES: Map<String, Map<String, Double>> = linkedMapOf(
            "signals" to linkedMapOf("abs" to 1e-6, "angle_abs_deg" to 1e-3),
            "closure" to linkedMapOf("abs" to 1e-6),
            "baseline" to linkedMapOf("abs" to 1e-6),
            "runs" to linkedMapOf("time_abs_ms" to 1.0, "rel" to 1e-4),
            "steps" to linkedMapOf("abs" to 1e-5, "rel" to 1e-5),
            "model_inputs" to linkedMapOf("abs" to 1e-5, "rel" to 1e-5),
            "step_probs" to linkedMapOf("abs" to 1e-4),
            "probs" to linkedMapOf("abs" to 1e-4),
            "events" to linkedMapOf("time_abs_ms" to 1.0, "confidence_abs" to 1e-4),
            "alerts" to linkedMapOf("time_abs_ms" to 1.0, "confidence_abs" to 1e-4),
        )

        /** vectors.py:713-726 `_REQUIRED_KEYS`. */
        val REQUIRED_KEYS: List<String> = listOf(
            "name", "bundle", "description", "pipeline_hash", "rules_hash", "tolerances", "baseline",
            "config_overrides", "model", "frames", "speed", "expected",
        )

        /** 바이트(UTF-8 JSON)에서 읽는다. 형식이 틀리면 IllegalArgumentException(이유와 함께). */
        fun read(bytes: ByteArray): VectorFile = fromTree(MiniJson.parse(bytes))

        @Suppress("UNCHECKED_CAST")
        fun fromTree(tree: Any?): VectorFile {
            require(tree is Map<*, *>) { "a vector file must be a JSON object" }
            val root = tree as Map<String, Any?>
            val missing = REQUIRED_KEYS.filter { !root.containsKey(it) }
            require(missing.isEmpty()) { "missing keys: ${missing.joinToString(", ")}" }
            val tolerances = LinkedHashMap<String, Map<String, Any?>>()
            for ((k, v) in obj(root["tolerances"], "tolerances")) tolerances[k] = obj(v, "tolerances.$k")
            val speed = ArrayList<Pair<Long, Double>>()
            for ((i, item) in list(root["speed"], "speed").withIndex()) {
                val pair = list(item, "speed[$i]")
                require(pair.size == 2) { "speed[$i] must be [t_ms, kmh]" }
                speed.add(long(pair[0], "speed[$i][0]") to num(pair[1], "speed[$i][1]"))
            }
            for (i in 1 until speed.size) {
                require(speed[i - 1].first <= speed[i].first) { "speed entries must be sorted by t_ms" }
            }
            val frames = ArrayList<RawFrame>()
            for ((i, f) in list(root["frames"], "frames").withIndex()) frames.add(frame(f, "frames[$i]"))
            val modelInputs = root["model_inputs"]?.let { mi ->
                list(mi, "model_inputs").mapIndexed { i, w ->
                    list(w, "model_inputs[$i]").mapIndexed { t, row ->
                        list(row, "model_inputs[$i][$t]").mapIndexed { f, v -> num(v, "model_inputs[$i][$t][$f]") }
                    }
                }
            }
            return VectorFile(
                name = str(root["name"], "name"),
                bundle = str(root["bundle"], "bundle"),
                description = str(root["description"], "description"),
                pipelineHash = str(root["pipeline_hash"], "pipeline_hash"),
                rulesHash = str(root["rules_hash"], "rules_hash"),
                tolerances = Collections.unmodifiableMap(tolerances),
                baseline = root["baseline"]?.let { baseline(it, "baseline") },
                configOverrides = obj(root["config_overrides"], "config_overrides"),
                model = root["model"]?.let { model(it) },
                frames = Collections.unmodifiableList(frames),
                speed = Collections.unmodifiableList(speed),
                modelInputs = modelInputs,
                expected = obj(root["expected"], "expected"),
                root = root,
            )
        }

        /** vectors.py:399-409 `frame_from_json`: t_ms, w, h는 정수, 좌표와 값은 실수, lm의 키는 정수 글자. */
        fun frame(v: Any?, where: String): RawFrame {
            val m = obj(v, where)
            val lm = LinkedHashMap<Int, Point>()
            for ((k, xy) in obj(m["lm"], "$where.lm")) {
                val index = k.toIntOrNull() ?: throw IllegalArgumentException("$where.lm key '$k' is not an integer")
                val p = list(xy, "$where.lm.$k")
                require(p.size == 2) { "$where.lm.$k must be [x, y]" }
                lm[index] = Point(num(p[0], "$where.lm.$k[0]"), num(p[1], "$where.lm.$k[1]"))
            }
            val bs = LinkedHashMap<String, Double>()
            for ((k, s) in obj(m["bs"], "$where.bs")) bs[k] = num(s, "$where.bs.$k")
            val mat = list(m["m"], "$where.m").mapIndexed { i, x -> num(x, "$where.m[$i]") }
            return RawFrame(
                tMs = long(m["t_ms"], "$where.t_ms"),
                face = bool(m["face"], "$where.face"),
                w = int(m["w"], "$where.w"),
                h = int(m["h"], "$where.h"),
                lm = Collections.unmodifiableMap(lm),
                bs = Collections.unmodifiableMap(bs),
                m = Collections.unmodifiableList(mat),
            )
        }

        /** vectors.py:418-430 `baseline_from_json`. */
        fun baseline(v: Any?, where: String): Baseline {
            val m = obj(v, where)
            val status = str(m["status"], "$where.status")
            return Baseline(
                eL = num(m["e_l"], "$where.e_l"),
                eR = num(m["e_r"], "$where.e_r"),
                bL = num(m["b_l"], "$where.b_l"),
                bR = num(m["b_r"], "$where.b_r"),
                r0 = list(m["r0"], "$where.r0").mapIndexed { i, x -> num(x, "$where.r0[$i]") },
                jaw0 = num(m["jaw0"], "$where.jaw0"),
                gazeH0 = num(m["gaze_h0"], "$where.gaze_h0"),
                gazeV0 = num(m["gaze_v0"], "$where.gaze_v0"),
                tCompleteMs = long(m["t_complete_ms"], "$where.t_complete_ms"),
                status = BaselineStatus.entries.firstOrNull { it.wire == status }
                    ?: throw IllegalArgumentException("$where.status '$status' is not a baseline status"),
            )
        }

        private fun model(v: Any?): VectorModel {
            val m = obj(v, "model")
            return when (val kind = m["kind"]) {
                "stub" -> VectorModel.Stub(
                    list(m["probs"], "model.probs").mapIndexed { i, row ->
                        list(row, "model.probs[$i]").mapIndexed { j, p -> num(p, "model.probs[$i][$j]") }
                    },
                )
                "onnx" -> VectorModel.Onnx(str(m["file"], "model.file"), str(m["sha256"], "model.sha256"))
                else -> throw IllegalArgumentException("model.kind '$kind' is not stub or onnx")
            }
        }

        // ----- JSON 나무 도우미 -----

        @Suppress("UNCHECKED_CAST")
        fun obj(v: Any?, where: String): Map<String, Any?> {
            require(v is Map<*, *>) { "$where must be an object" }
            return v as Map<String, Any?>
        }

        fun list(v: Any?, where: String): List<Any?> {
            require(v is List<*>) { "$where must be a list" }
            return v
        }

        fun str(v: Any?, where: String): String {
            require(v is String) { "$where must be a string" }
            return v
        }

        fun bool(v: Any?, where: String): Boolean {
            require(v is Boolean) { "$where must be true or false" }
            return v
        }

        fun long(v: Any?, where: String): Long {
            require(v is Long) { "$where must be an integer" }
            return v
        }

        fun int(v: Any?, where: String): Int {
            val l = long(v, where)
            require(l >= Int.MIN_VALUE && l <= Int.MAX_VALUE) { "$where is outside the Int range" }
            return l.toInt()
        }

        /** 실수 칸: Python `float(x)`. 정수 글자는 Double로 바꾼다(|x| <= 2^53이면 값이 그대로다). */
        fun num(v: Any?, where: String): Double = when (v) {
            is Double -> v
            is Long -> v.toDouble()
            else -> throw IllegalArgumentException("$where must be a number")
        }
    }
}
