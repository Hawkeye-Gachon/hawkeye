package hawkeye.drowsy.core

import hawkeye.drowsy.core.internal.FrameSignals
import hawkeye.drowsy.core.spec.SpecLoader
import hawkeye.drowsy.core.testdata.TestData
import hawkeye.drowsy.core.testkit.DiffData
import hawkeye.drowsy.core.testkit.Hkdiff1
import hawkeye.drowsy.core.testkit.VectorFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * C-errors의 "모델이 2, 4, 0개 값을 돌려준다" 세 경우(E7, 계획 9.2.6절, pipeline.py:241-242; FORMATS.md 2.7절).
 *
 * 입력은 components/C-errors_model_inputs.hkd(HKDIFF1, 모드 RAW, SIG, STATE): tx_fuzz 세션 inj_model 9001의 프레임을 배포
 * 상수, 주입한 기준선, (0.25, 0.5, 0.25)를 돌려주는 대체 모델로 돌려, 첫 모델 호출이 있는 프레임 뒤에서 자른 것이다.
 * 그 파일로 DriverPipeline(pipeline_json, rules_json, baseline_injected, feature_start_ms)을 만들고 대체 모델 대신
 * model_returns를 돌려주는 모델을 넣어 세 모드로 프레임을 넣는다. 마지막으로 넣은 프레임(fed_index)이 Python과 같은 메시지의
 * IllegalArgumentException(Python ValueError)을 내야 하고, 그 앞의 프레임은 예외 없이 지나야 하며, 예외 뒤의 계수
 * (frames_accepted, steps_emitted, model_calls)가 counters_at_error와 같아야 한다.
 *
 * 같은 파일을 파일의 대체 모델로 STATE 모드에서 한 번 더 돌려 파일의 출력 열(상태 뒤의 스텝, 스텝 확률, 이벤트, 경고, 계수,
 * gaze_zone)과 수로 같은지도 본다(FORMATS.md 1.5절의 STATE 재생; 단계 5의 E8이 같은 일을 모든 세션에 한다).
 */
class ModelCallErrorsTest {

    /** FORMATS.md 1.3, 1.5절의 열에서 세 모드의 입력을 다시 만든다(이 파일에 필요한 만큼만). */
    private class Session(val h: Hkdiff1) {
        val header = h.header
        val tMs = h.i64("t_ms")
        val accepted = h.u8("accepted")
        val speed = h.f64("speed")
        val speedNull = h.u8("speed_null")
        val n = tMs.size
        val landmarkIndices = VectorFile.list(header["landmark_indices"], "landmark_indices").map { VectorFile.int(it, "i") }
        val blendshapeNames = VectorFile.list(header["blendshape_names"], "blendshape_names").map { it as String }

        fun speedAt(i: Int): Double? = if (speedNull[i] == 1) null else speed[i]

        fun rawFrames(): List<RawFrame> {
            val face = h.u8("face")
            val w = h.i32("w")
            val hh = h.i32("h")
            val lm = h.f64("lm")
            val bs = h.f64("bs")
            val m = h.f64("m")
            val out = ArrayList<RawFrame>(n)
            var f = 0
            for (i in 0 until n) {
                if (face[i] == 1) {
                    val lmMap = LinkedHashMap<Int, Point>()
                    for ((j, idx) in landmarkIndices.withIndex()) {
                        val o = (f * landmarkIndices.size + j) * 2
                        lmMap[idx] = Point(lm[o], lm[o + 1])
                    }
                    val bsMap = LinkedHashMap<String, Double>()
                    for ((j, name) in blendshapeNames.withIndex()) bsMap[name] = bs[f * blendshapeNames.size + j]
                    val mat = (0 until 16).map { m[f * 16 + it] }
                    out.add(RawFrame(tMs = tMs[i], face = true, w = w[i], h = hh[i], lm = lmMap, bs = bsMap, m = mat))
                    f++
                } else {
                    out.add(RawFrame(tMs = tMs[i], face = false, w = w[i], h = hh[i], lm = emptyMap(), bs = emptyMap(), m = emptyList()))
                }
            }
            return out
        }

        fun signals(): List<FrameSignals> {
            val face = h.u8("sig_face")
            fun col(name: String) = h.f64(name)
            val earL = col("ear_l"); val earR = col("ear_r"); val eyeWL = col("eye_w_l"); val eyeWR = col("eye_w_r")
            val blinkL = col("blink_l"); val blinkR = col("blink_r"); val jaw = col("jaw_open")
            val yaw = col("yaw"); val pitch = col("pitch"); val roll = col("roll")
            val eyeLook = col("eye_look")
            val iris = col("iris")
            val irisNull = h.u8("iris_null")
            fun irisAt(i: Int, k: Int): Double? = if (irisNull[i * 4 + k] == 1) null else iris[i * 4 + k]
            return (0 until n).map { i ->
                FrameSignals(
                    tMs = tMs[i], face = face[i] == 1, earL = earL[i], earR = earR[i], eyeWL = eyeWL[i], eyeWR = eyeWR[i],
                    blinkL = blinkL[i], blinkR = blinkR[i], eyeLook = (0 until 8).map { eyeLook[i * 8 + it] }, jawOpen = jaw[i],
                    yaw = yaw[i], pitch = pitch[i], roll = roll[i],
                    irisHL = irisAt(i, 0), irisHR = irisAt(i, 1), irisVL = irisAt(i, 2), irisVR = irisAt(i, 3),
                )
            }
        }

        /** 받아들인 행마다의 FrameState (t_ms = 받아들인 j번째 넣은 프레임의 t_ms). */
        fun states(): List<FrameState> {
            val closure = h.f64("closure"); val yawRel = h.f64("yaw_rel"); val pitchRel = h.f64("pitch_rel")
            val gh = h.f64("gaze_h_rel"); val gv = h.f64("gaze_v_rel"); val jaw = h.f64("jaw_rel")
            val flags = h.u8("state_flags")
            val acceptedT = (0 until n).filter { accepted[it] != 0 }.map { tMs[it] }
            return closure.indices.map { j ->
                FrameState(
                    tMs = acceptedT[j], face = flags[j] and 1 != 0, closure = closure[j], closureValid = flags[j] and 2 != 0,
                    provisional = flags[j] and 4 != 0, yawRel = yawRel[j], pitchRel = pitchRel[j], poseValid = flags[j] and 8 != 0,
                    gazeHRel = gh[j], gazeVRel = gv[j], jawRel = jaw[j],
                )
            }
        }

        fun pipeline(model: ModelFn?): DriverPipeline = DriverPipeline(
            spec = SpecLoader.parsePipelineSpec((header["pipeline_json"] as String).toByteArray(Charsets.UTF_8)),
            rules = SpecLoader.parseRules((header["rules_json"] as String).toByteArray(Charsets.UTF_8)),
            baseline = header["baseline_injected"]?.let { baselineHex(it) },
            featureStartMs = header["feature_start_ms"] as Long?,
            model = model,
        )

        private fun baselineHex(v: Any?): Baseline {
            val m = VectorFile.obj(v, "baseline_injected")
            fun x(key: String) = java.lang.Double.parseDouble(m[key] as String)
            return Baseline(
                eL = x("e_l"), eR = x("e_r"), bL = x("b_l"), bR = x("b_r"),
                r0 = VectorFile.list(m["r0"], "r0").map { java.lang.Double.parseDouble(it as String) },
                jaw0 = x("jaw0"), gazeH0 = x("gaze_h0"), gazeV0 = x("gaze_v0"),
                tCompleteMs = VectorFile.long(m["t_complete_ms"], "t_complete_ms"),
                status = BaselineStatus.fromWire(m["status"] as String),
            )
        }
    }

    /** 세 모드 가운데 하나로 넣은 프레임 i의 출력. 버린 프레임은 null. */
    private fun interface Feeder {
        fun feed(p: DriverPipeline, i: Int): PipelineOutput?
    }

    private fun feeders(s: Session): Map<String, Feeder> {
        val raw = s.rawFrames()
        val sig = s.signals()
        val states = s.states()
        // FORMATS.md 1.5절: 받아들인 프레임에는 다음 상태 행을, 버려진 프레임에는 그 t_ms를 가진 아무 상태나(직전 행을 그
        // t_ms로) 넣는다. 첫 프레임은 언제나 받아들여진다.
        val stateFor = ArrayList<FrameState>(s.n)
        var j = 0
        for (i in 0 until s.n) {
            stateFor.add(if (s.accepted[i] != 0) states[j++] else states[j - 1].copy(tMs = s.tMs[i]))
        }
        return linkedMapOf(
            "RAW" to Feeder { p, i -> p.update(raw[i], s.speedAt(i)) },
            "SIG" to Feeder { p, i -> p.updateFromSignals(sig[i], s.speedAt(i)) },
            "STATE" to Feeder { p, i -> p.updateFromState(stateFor[i], null, s.speedAt(i)) },
        )
    }

    @Test
    fun modelReturningTwoFourOrZeroValuesIsRefused() {
        val data = TestData.readyJson("components/C-errors.json")
        val cases = TestData.list(data["cases"]).map { TestData.obj(it) }.filter {
            it["component"] == "DriverPipeline" && TestData.obj(it["setup"]).containsKey("model_returns")
        }
        assertEquals(3, cases.size)
        var replays = 0
        for (case in cases) {
            val setup = TestData.obj(case["setup"])
            val s = Session(DiffData.readyHkdiff1(setup["inputs"] as String))
            assertEquals(listOf("RAW", "SIG", "STATE"), VectorFile.list(s.header["modes"], "modes"))
            val returns = VectorFile.list(setup["model_returns"], "model_returns").map { TestData.num(it) }
            val out = TestData.obj(case["out"])
            val fedIndex = VectorFile.int(out["fed_index"], "fed_index")
            val err = TestData.obj(out["error"])
            assertEquals("ValueError", err["type"])
            val cnt = TestData.obj(case["counters_at_error"])
            assertEquals(s.n - 1, fedIndex, "${case["case"]}: the last fed frame raises")
            assertEquals(s.tMs[fedIndex], out["t_ms"])
            for ((mode, feeder) in feeders(s)) {
                var calls = 0
                val p = s.pipeline(ModelFn { w ->
                    calls++
                    assertEquals(120, w.size)
                    returns
                })
                for (i in 0 until fedIndex) {
                    val o = feeder.feed(p, i)
                    if (s.accepted[i] == 0) assertNull(o, "$mode frame $i is dropped") else assertNotNull(o, "$mode frame $i")
                }
                try {
                    feeder.feed(p, fedIndex)
                    fail("${case["case"]} $mode: frame $fedIndex did not raise")
                } catch (e: IllegalArgumentException) {
                    assertEquals(err["message"], e.message, "${case["case"]} $mode")
                }
                assertEquals(1, calls, "${case["case"]} $mode: the model was called once, at the raising frame")
                assertEquals(cnt["frames_accepted"], p.framesAccepted, "${case["case"]} $mode frames_accepted")
                assertEquals(cnt["steps_emitted"], p.stepsEmitted, "${case["case"]} $mode steps_emitted")
                assertEquals(cnt["model_calls"], p.modelCalls, "${case["case"]} $mode model_calls")
                replays++
            }
            println(
                "ModelCallErrorsTest: '${case["case"]}' (model returns ${returns.size} values): RAW, SIG, STATE each raise at fed index " +
                    "$fedIndex (t_ms ${out["t_ms"]}) with '${e(err)}'; counters after the raise frames_accepted ${cnt["frames_accepted"]}, " +
                    "steps_emitted ${cnt["steps_emitted"]}, model_calls ${cnt["model_calls"]}",
            )
        }
        println("ModelCallErrorsTest.modelReturningTwoFourOrZeroValuesIsRefused: ${cases.size} cases x 3 modes = $replays replays, all as Python")
        assertEquals(9, replays)
    }

    private fun e(err: Map<String, Any?>) = "${err["type"]}: ${err["message"]}"

    /** 파일의 대체 모델로 STATE 모드: 파일의 출력 열과 수로 같다. */
    @Test
    fun stubModelStateReplayEqualsTheFile() {
        TestData.requireReadySmall()
        val s = Session(DiffData.readyHkdiff1("components/C-errors_model_inputs.hkd"))
        val h = s.h
        val stub = h.f64("stub_probs")
        val count = VectorFile.int(VectorFile.obj(s.header["model"], "model")["count"], "count")
        var calls = 0
        val p = s.pipeline(ModelFn { _ ->
            val row = minOf(calls, count - 1)
            calls++
            listOf(stub[row * 3], stub[row * 3 + 1], stub[row * 3 + 2])
        })
        val feeder = feeders(s).getValue("STATE")
        val zones = h.u8("gaze_zone")
        val stFrame = h.u32("st_frame"); val stK = h.i64("st_k"); val stT = h.i64("st_t_end_ms"); val stFull = h.u8("st_full")
        val stValues = h.f64("st_values"); val stProbNull = h.u8("st_prob_null"); val stProbs = h.f64("st_probs")
        val enums = VectorFile.obj(s.header["enums"], "enums")
        val types = VectorFile.list(enums["event_types"], "event_types").map { EventType.fromWire(it as String) }
        val sources = VectorFile.list(enums["sources"], "sources").map { Source.fromWire(it as String) }
        val phases = VectorFile.list(enums["phases"], "phases").map { Phase.fromWire(it as String) }
        val zoneNames = VectorFile.list(enums["zones"], "zones").map { GazeZone.fromWire(it as String) }
        fun events(prefix: String): List<Pair<Long, DriverStateEvent>> {
            val fr = h.u32("${prefix}_frame"); val ty = h.u8("${prefix}_type"); val sev = h.u8("${prefix}_severity")
            val times = h.i64("${prefix}_times"); val conf = h.f64("${prefix}_confidence"); val src = h.u8("${prefix}_source")
            val ph = h.u8("${prefix}_phase"); val prov = h.u8("${prefix}_provisional")
            return fr.indices.map { i ->
                fr[i] to DriverStateEvent(
                    type = types[ty[i]], severity = sev[i], tStartMs = times[2 * i], durationMs = times[2 * i + 1],
                    confidence = conf[i], source = sources[src[i]], phase = phases[ph[i]], provisional = prov[i] == 1,
                )
            }
        }
        val expEvents = events("ev")
        val expAlerts = events("al")
        val gotEvents = ArrayList<Pair<Long, DriverStateEvent>>()
        val gotAlerts = ArrayList<Pair<Long, DriverStateEvent>>()
        val gotSteps = ArrayList<Pair<Long, StepFeatures>>()
        val gotProbs = ArrayList<ClassProbs?>()
        val diffs = ArrayList<String>()
        var row = 0L
        for (i in 0 until s.n) {
            val o = feeder.feed(p, i) ?: continue
            for (st in o.steps) gotSteps.add(row to st)
            gotProbs.addAll(o.stepProbs)
            for (e in o.events) gotEvents.add(row to e)
            for (a in o.alerts) gotAlerts.add(row to a)
            if (p.gazeZone != zoneNames[zones[row.toInt()]]) diffs.add("row $row: gaze_zone ${p.gazeZone} != ${zoneNames[zones[row.toInt()]]}")
            row++
        }
        assertEquals(stK.size, gotSteps.size, "steps")
        for (j in stK.indices) {
            val (r, st) = gotSteps[j]
            val same = r == stFrame[j] && st.k.toLong() == stK[j] && st.tEndMs == stT[j] && st.fullLookback == (stFull[j] == 1) &&
                (0 until N_FEATURES).all { f -> st.values[f] == stValues[j * N_FEATURES + f] }
            if (!same) diffs.add("step $j differs: $st at row $r")
            val pr = gotProbs[j]
            val probsSame = if (stProbNull[j] == 1) pr == null else
                pr != null && pr.alert == stProbs[3 * j] && pr.lowVigilant == stProbs[3 * j + 1] && pr.drowsy == stProbs[3 * j + 2]
            if (!probsSame) diffs.add("step $j probs $pr")
        }
        fun cmp(what: String, e: List<Pair<Long, DriverStateEvent>>, a: List<Pair<Long, DriverStateEvent>>) {
            if (e.size != a.size) {
                diffs.add("$what: ${e.size} expected, ${a.size} got")
                return
            }
            for (j in e.indices) {
                val (er, ee) = e[j]
                val (ar, ae) = a[j]
                val same = er == ar && ee.type == ae.type && ee.severity == ae.severity && ee.tStartMs == ae.tStartMs &&
                    ee.durationMs == ae.durationMs && ee.confidence == ae.confidence && ee.source == ae.source &&
                    ee.phase == ae.phase && ee.provisional == ae.provisional
                if (!same) diffs.add("$what $j: expected $ee at row $er, got $ae at row $ar")
            }
        }
        cmp("events", expEvents, gotEvents)
        cmp("alerts", expAlerts, gotAlerts)
        val counters = VectorFile.obj(s.header["counters"], "counters")
        val got = linkedMapOf<String, Any?>(
            "frames_accepted" to p.framesAccepted, "dropped_frames" to p.droppedFrames, "steps_emitted" to p.stepsEmitted,
            "model_calls" to p.modelCalls, "nonfinite_windows" to p.nonfiniteWindows,
        )
        if (counters != got) diffs.add("counters $got != $counters")
        println(
            "ModelCallErrorsTest.stubModelStateReplayEqualsTheFile: STATE replay of ${s.n} fed frames ($row accepted): steps ${gotSteps.size}, " +
                "step_probs non-null ${gotProbs.count { it != null }}, events ${gotEvents.size}, alerts ${gotAlerts.size}, counters $got, " +
                "differences ${diffs.size} ${diffs.take(10)}",
        )
        assertTrue(diffs.isEmpty(), diffs.take(20).toString())
    }
}
