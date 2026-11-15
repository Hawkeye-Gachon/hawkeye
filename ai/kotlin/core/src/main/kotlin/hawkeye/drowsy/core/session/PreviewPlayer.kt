package hawkeye.drowsy.core.session

import hawkeye.drowsy.core.Baseline
import hawkeye.drowsy.core.BaselineStatus
import hawkeye.drowsy.core.ClassProbs
import hawkeye.drowsy.core.ClosureRun
import hawkeye.drowsy.core.DriverStateEvent
import hawkeye.drowsy.core.DrowsyInternalApi
import hawkeye.drowsy.core.EventType
import hawkeye.drowsy.core.FrameState
import hawkeye.drowsy.core.GazeZone
import hawkeye.drowsy.core.Phase
import hawkeye.drowsy.core.PipelineOutput
import hawkeye.drowsy.core.Source
import hawkeye.drowsy.core.StepFeatures
import hawkeye.drowsy.core.spec.MiniJson
import java.io.InputStream
import java.util.Collections

/**
 * 전달 0(API 미리보기, K19 (가))의 재생기 (계획 5.8절 끝, 9.2.6절 "전달 0 재생 자료").
 *
 * 판정을 하지 않는다. 참조 구현(Python `DriverPipeline`)이 합성 각본 세션에서 낸 프레임별 출력
 * (`kotlin/.diffdata/replay0/<name>.json`, 형식은 `kotlin/difftools/FORMATS.md` 2.6절)을 순서대로 가지고 있다가,
 * 받아들인 프레임마다 [onAcceptedFrame]을 부르면:
 *
 * - 이 재생 차례의 원점(첫 프레임의 모니터 시각)부터 지난 시간이 다음 기록의 세션 시간 이상이면 그 기록 하나를
 *   DriverState로 만들어 돌려준다. 아니면 null이다. 그래서 프레임마다 상태가 오지 않을 수 있고, 한 번에 둘 이상을 내지
 *   않으며, 기록을 건너뛰지 않는다.
 * - active는 DriveSession과 같은 접기(ActiveFold, 계획 6.2절)로 만든다. 파일에 적힌 active는 쓰지 않는다(시험이 둘을 비교한다).
 * - 기록이 끝나면 다음 프레임에서 처음부터 다시 재생한다. 원점, sessionMs, 접기 상태도 새 운전처럼 다시 시작한다.
 *
 * 기록의 세션 시간 = 그 기록의 tMs − 첫 기록의 tMs (DriveSession의 sessionMs와 같은 정의, 계획 6.2절).
 * 참조가 버린 프레임(`output: null`)의 기록은 읽을 때 뺀다. DriveSession에서도 버린 프레임은 상태를 내지 않는다.
 * 그 뒤 기록의 counters에는 버린 수가 그대로 들어 있다.
 *
 * 스레드에 안전하지 않다. 모니터의 파이프라인 스레드 하나에서만 부른다. 상태를 가진 companion을 두지 않는다(R19).
 */
@DrowsyInternalApi
class PreviewPlayer private constructor(
    /** 재생 자료의 `session.name`. */
    val name: String,
    /** 재생 자료의 `session.rules_hash` (참조가 쓴 rules.json 바이트의 sha256). */
    val rulesHash: String,
    private val records: List<Record>,
    alertPriority: List<EventType>,
) {
    private class Record(
        val sessionMs: Long,
        val output: PipelineOutput,
        val gazeZone: GazeZone,
        val baseline: Baseline?,
        val counters: PipelineCounters,
        val speedKmh: Double?,
    )

    private val fold = ActiveFold(alertPriority)
    private var started = false
    private var originMs = 0L
    private var next = 0
    private var loops = 0L

    /** 상태를 내는 기록의 수(버린 프레임의 기록은 빠진다). */
    val recordCount: Int get() = records.size

    /** 시작한 재생 차례의 수. 첫 프레임 뒤에 1, 처음부터 다시 재생할 때마다 1씩 는다. */
    val loopsStarted: Long get() = loops

    /**
     * 받아들인 프레임 하나. monitorTimeMs는 모니터 시계(ms)의 그 프레임 시각이다.
     * 다음 기록을 낼 때가 되었으면 그 기록의 DriverState, 아니면 null.
     */
    fun onAcceptedFrame(monitorTimeMs: Long): DriverState? {
        if (!started || next == records.size) {
            started = true
            originMs = monitorTimeMs
            next = 0
            fold.reset()
            loops++
        }
        val r = records[next]
        if (monitorTimeMs - originMs < r.sessionMs) return null
        next++
        val active = fold.apply(r.output.events, r.output.state.tMs)
        return DriverState(
            output = r.output,
            gazeZone = r.gazeZone,
            baseline = r.baseline,
            counters = r.counters,
            speedKmh = r.speedKmh,
            sessionMs = r.sessionMs,
            active = active,
        )
    }

    // binary-compatibility-validator는 클래스의 표시를 companion에 옮기지 않으므로 companion에도 붙인다(apiDump에서 확인).
    @DrowsyInternalApi
    companion object {
        /** 재생 자료 형식 이름과 판 (FORMATS.md 2.6절). */
        const val FORMAT: String = "hawkeye-replay0"
        const val FORMAT_VERSION: Long = 1L

        /**
         * 재생 자료(UTF-8 JSON)를 읽는다. alertPriority는 그 자료를 만든 rules.json의 `alert_priority`이다
         * (자료에는 적혀 있지 않다). 형식이 다르면 IllegalArgumentException(JSON 문법 오류는 그것을 잇는 SpecException).
         * 스트림은 닫지 않는다.
         */
        @JvmStatic
        fun read(input: InputStream, alertPriority: List<EventType>): PreviewPlayer =
            read(input.readBytes(), alertPriority)

        @JvmStatic
        fun read(bytes: ByteArray, alertPriority: List<EventType>): PreviewPlayer {
            val root = obj(MiniJson.parse(bytes), "$")
            val format = str(req(root, "format", "$"), "$.format")
            require(format == FORMAT) { "replay: $.format is \"$format\", expected \"$FORMAT\"" }
            val version = long(req(root, "version", "$"), "$.version")
            require(version == FORMAT_VERSION) { "replay: $.version is $version, expected $FORMAT_VERSION" }
            val session = obj(req(root, "session", "$"), "$.session")
            val name = str(req(session, "name", "$.session"), "$.session.name")
            val rulesHash = str(req(session, "rules_hash", "$.session"), "$.session.rules_hash")
            val frames = list(req(root, "frames", "$"), "$.frames")
            val records = ArrayList<Record>()
            var firstTMs = 0L
            for ((i, f) in frames.withIndex()) {
                val w = "$.frames[$i]"
                val fr = obj(f, w)
                val tMs = long(req(fr, "t_ms", w), "$w.t_ms")
                val outValue = req(fr, "output", w)
                if (outValue == null) continue
                val output = output(outValue, "$w.output")
                require(output.state.tMs == tMs) { "replay: $w.output.state.t_ms ${output.state.tMs} != $w.t_ms $tMs" }
                if (records.isEmpty()) firstTMs = tMs
                val speedValue = req(fr, "speed_kmh", w)
                val baselineValue = req(fr, "baseline", w)
                records.add(
                    Record(
                        sessionMs = tMs - firstTMs,
                        output = output,
                        gazeZone = gazeZone(str(req(fr, "gaze_zone", w), "$w.gaze_zone"), "$w.gaze_zone"),
                        baseline = if (baselineValue == null) null else baseline(baselineValue, "$w.baseline"),
                        counters = counters(req(fr, "counters", w), "$w.counters"),
                        speedKmh = if (speedValue == null) null else dbl(speedValue, "$w.speed_kmh"),
                    ),
                )
            }
            require(records.isNotEmpty()) { "replay: $.frames has no frame with an output" }
            for (j in 1 until records.size) {
                require(records[j].sessionMs > records[j - 1].sessionMs) {
                    "replay: frame times do not increase at record $j"
                }
            }
            return PreviewPlayer(name, rulesHash, records, alertPriority)
        }

        // ----- 재생 자료의 값 읽기. 이름은 Python 필드 이름이다(FORMATS.md 2.6절). -----

        private val STATE_KEYS = listOf(
            "t_ms", "face", "closure", "closure_valid", "provisional", "yaw_rel", "pitch_rel", "pose_valid",
            "gaze_h_rel", "gaze_v_rel", "jaw_rel",
        )
        private val RUN_KEYS = listOf(
            "start_ms", "end_ms", "duration_ms", "amplitude", "peak_closing_velocity", "peak_opening_velocity",
            "velocity_valid", "truncated",
        )
        private val STEP_KEYS = listOf("k", "t_end_ms", "full_lookback", "values")
        private val EVENT_KEYS = listOf(
            "type", "severity", "t_start_ms", "duration_ms", "confidence", "source", "phase", "provisional",
        )
        private val OUTPUT_KEYS = listOf("state", "runs", "steps", "step_probs", "probs", "events", "alerts")
        private val BASELINE_KEYS = listOf(
            "e_l", "e_r", "b_l", "b_r", "r0", "jaw0", "gaze_h0", "gaze_v0", "t_complete_ms", "status",
        )
        private val COUNTER_KEYS = listOf(
            "frames_accepted", "dropped_frames", "steps_emitted", "model_calls", "nonfinite_windows",
        )

        private fun output(v: Any?, w: String): PipelineOutput {
            val o = exact(v, OUTPUT_KEYS, w)
            val s = exact(o["state"], STATE_KEYS, "$w.state")
            val state = FrameState(
                tMs = long(s["t_ms"], "$w.state.t_ms"),
                face = bool(s["face"], "$w.state.face"),
                closure = dbl(s["closure"], "$w.state.closure"),
                closureValid = bool(s["closure_valid"], "$w.state.closure_valid"),
                provisional = bool(s["provisional"], "$w.state.provisional"),
                yawRel = dbl(s["yaw_rel"], "$w.state.yaw_rel"),
                pitchRel = dbl(s["pitch_rel"], "$w.state.pitch_rel"),
                poseValid = bool(s["pose_valid"], "$w.state.pose_valid"),
                gazeHRel = dbl(s["gaze_h_rel"], "$w.state.gaze_h_rel"),
                gazeVRel = dbl(s["gaze_v_rel"], "$w.state.gaze_v_rel"),
                jawRel = dbl(s["jaw_rel"], "$w.state.jaw_rel"),
            )
            val runs = ArrayList<ClosureRun>()
            for ((j, rv) in list(o["runs"], "$w.runs").withIndex()) {
                val rw = "$w.runs[$j]"
                val r = exact(rv, RUN_KEYS, rw)
                runs.add(
                    ClosureRun(
                        startMs = long(r["start_ms"], "$rw.start_ms"),
                        endMs = long(r["end_ms"], "$rw.end_ms"),
                        durationMs = long(r["duration_ms"], "$rw.duration_ms"),
                        amplitude = dbl(r["amplitude"], "$rw.amplitude"),
                        peakClosingVelocity = dbl(r["peak_closing_velocity"], "$rw.peak_closing_velocity"),
                        peakOpeningVelocity = dbl(r["peak_opening_velocity"], "$rw.peak_opening_velocity"),
                        velocityValid = bool(r["velocity_valid"], "$rw.velocity_valid"),
                        truncated = bool(r["truncated"], "$rw.truncated"),
                    ),
                )
            }
            val steps = ArrayList<StepFeatures>()
            for ((j, sv) in list(o["steps"], "$w.steps").withIndex()) {
                val sw = "$w.steps[$j]"
                val st = exact(sv, STEP_KEYS, sw)
                steps.add(
                    StepFeatures(
                        k = int(st["k"], "$sw.k"),
                        tEndMs = long(st["t_end_ms"], "$sw.t_end_ms"),
                        fullLookback = bool(st["full_lookback"], "$sw.full_lookback"),
                        values = doubles(st["values"], "$sw.values"),
                    ),
                )
            }
            val stepProbs = ArrayList<ClassProbs?>()
            for ((j, pv) in list(o["step_probs"], "$w.step_probs").withIndex()) {
                stepProbs.add(if (pv == null) null else probs(pv, "$w.step_probs[$j]"))
            }
            require(stepProbs.size == steps.size) { "replay: $w.step_probs has ${stepProbs.size} elements, steps ${steps.size}" }
            val pr = o["probs"]
            return PipelineOutput(
                state = state,
                runs = Collections.unmodifiableList(runs),
                steps = Collections.unmodifiableList(steps),
                stepProbs = Collections.unmodifiableList(stepProbs),
                probs = if (pr == null) null else probs(pr, "$w.probs"),
                events = events(o["events"], "$w.events"),
                alerts = events(o["alerts"], "$w.alerts"),
            )
        }

        private fun events(v: Any?, w: String): List<DriverStateEvent> {
            val out = ArrayList<DriverStateEvent>()
            for ((j, ev) in list(v, w).withIndex()) {
                val ew = "$w[$j]"
                val e = exact(ev, EVENT_KEYS, ew)
                out.add(
                    DriverStateEvent(
                        type = wire(ew) { EventType.fromWire(str(e["type"], "$ew.type")) },
                        severity = int(e["severity"], "$ew.severity"),
                        tStartMs = long(e["t_start_ms"], "$ew.t_start_ms"),
                        durationMs = long(e["duration_ms"], "$ew.duration_ms"),
                        confidence = dbl(e["confidence"], "$ew.confidence"),
                        source = wire(ew) { Source.fromWire(str(e["source"], "$ew.source")) },
                        phase = wire(ew) { Phase.fromWire(str(e["phase"], "$ew.phase")) },
                        provisional = bool(e["provisional"], "$ew.provisional"),
                    ),
                )
            }
            return Collections.unmodifiableList(out)
        }

        private fun baseline(v: Any?, w: String): Baseline {
            val b = exact(v, BASELINE_KEYS, w)
            val r0 = doubles(b["r0"], "$w.r0")
            require(r0.size == 9) { "replay: $w.r0 has ${r0.size} values, expected 9" }
            return Baseline(
                eL = dbl(b["e_l"], "$w.e_l"),
                eR = dbl(b["e_r"], "$w.e_r"),
                bL = dbl(b["b_l"], "$w.b_l"),
                bR = dbl(b["b_r"], "$w.b_r"),
                r0 = r0,
                jaw0 = dbl(b["jaw0"], "$w.jaw0"),
                gazeH0 = dbl(b["gaze_h0"], "$w.gaze_h0"),
                gazeV0 = dbl(b["gaze_v0"], "$w.gaze_v0"),
                tCompleteMs = long(b["t_complete_ms"], "$w.t_complete_ms"),
                status = wire(w) { BaselineStatus.fromWire(str(b["status"], "$w.status")) },
            )
        }

        private fun counters(v: Any?, w: String): PipelineCounters {
            val c = exact(v, COUNTER_KEYS, w)
            return PipelineCounters(
                framesAccepted = long(c["frames_accepted"], "$w.frames_accepted"),
                droppedFrames = long(c["dropped_frames"], "$w.dropped_frames"),
                stepsEmitted = long(c["steps_emitted"], "$w.steps_emitted"),
                modelCalls = long(c["model_calls"], "$w.model_calls"),
                nonfiniteWindows = long(c["nonfinite_windows"], "$w.nonfinite_windows"),
            )
        }

        private fun probs(v: Any?, w: String): ClassProbs {
            val p = doubles(v, w)
            require(p.size == 3) { "replay: $w has ${p.size} values, expected 3" }
            return ClassProbs(alert = p[0], lowVigilant = p[1], drowsy = p[2])
        }

        private fun gazeZone(s: String, w: String): GazeZone = wire(w) { GazeZone.fromWire(s) }

        private inline fun <T> wire(w: String, f: () -> T): T =
            try {
                f()
            } catch (e: IllegalArgumentException) {
                throw IllegalArgumentException("replay: $w: ${e.message}", e)
            }

        private fun obj(v: Any?, w: String): Map<String, Any?> {
            require(v is Map<*, *>) { "replay: $w is not an object" }
            @Suppress("UNCHECKED_CAST")
            return v as Map<String, Any?>
        }

        private fun req(m: Map<String, Any?>, key: String, w: String): Any? {
            require(m.containsKey(key)) { "replay: $w has no key \"$key\"" }
            return m[key]
        }

        /** 키 집합이 정확히 keys인 객체. 형식이 바뀌면 조용히 읽지 않고 멈춘다. */
        private fun exact(v: Any?, keys: List<String>, w: String): Map<String, Any?> {
            val m = obj(v, w)
            for (k in keys) require(m.containsKey(k)) { "replay: $w has no key \"$k\"" }
            require(m.size == keys.size) { "replay: $w has keys ${m.keys}, expected $keys" }
            return m
        }

        private fun list(v: Any?, w: String): List<Any?> {
            require(v is List<*>) { "replay: $w is not an array" }
            return v
        }

        private fun str(v: Any?, w: String): String {
            require(v is String) { "replay: $w is not a string" }
            return v
        }

        private fun bool(v: Any?, w: String): Boolean {
            require(v is Boolean) { "replay: $w is not a boolean" }
            return v
        }

        private fun long(v: Any?, w: String): Long {
            require(v is Long) { "replay: $w is not an integer" }
            return v
        }

        /** R22: Int 칸은 범위를 검사해 줄인다. */
        private fun int(v: Any?, w: String): Int {
            val x = long(v, w)
            require(x >= Int.MIN_VALUE && x <= Int.MAX_VALUE) { "replay: $w = $x does not fit in Int" }
            return x.toInt()
        }

        /**
         * 실수 칸. Python이 실수 칸에 int를 넣었으면 JSON에는 정수 글자로 적힌다(예: `0`). 그 값은 Double로 바꾼다.
         * 가정: 재생 자료의 실수 칸에 있는 정수 글자는 |값| <= 2^53이라 바꿔도 값이 같다.
         */
        private fun dbl(v: Any?, w: String): Double = when (v) {
            is Double -> v
            is Long -> v.toDouble()
            else -> throw IllegalArgumentException("replay: $w is not a number")
        }

        private fun doubles(v: Any?, w: String): List<Double> {
            val l = list(v, w)
            val out = ArrayList<Double>(l.size)
            for ((j, x) in l.withIndex()) out.add(dbl(x, "$w[$j]"))
            return Collections.unmodifiableList(out)
        }
    }
}
