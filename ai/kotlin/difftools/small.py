"""작은 파일: 숫자 표, 비교기 표, 진행 중 이벤트 기대값, 전달 0 재생 자료, 모델 창 (FORMATS.md 2장)."""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import struct

import numpy as np

import hk

RT = hk.RT


def gen_info(*names: str) -> dict:
    return {"files": {n: hk.sha256_file(os.path.join(hk.HERE, n)) for n in ("hk.py", *names)}}


def write_json(path: str, obj) -> None:  # noqa: ANN001
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.write("\n")
    os.replace(path + ".tmp", path)
    with open(path, encoding="utf-8") as fh:
        back = json.load(fh)
    if json.dumps(back, ensure_ascii=False, allow_nan=False, separators=(",", ":")) != text:
        raise AssertionError(f"{path}: JSON read-back differs")


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


# ---------------------------------------------------------------------------------------------- canary
FIXED = [0.1, -0.0, 0.0, 5e-324, 1.7976931348623157e308, 0.30000000000000004, 0.4, 0.6, 16.49993,
         2.2250738585072014e-308, 2.225073858507201e-308, 1e23, 9007199254740993.0, 1e16, 1e-7, 0.5, 23.0,
         1731465123457.0, 1.0000000000000002, 0.9999999999999999]
FIXED_I64 = [0, 1, -1, 2 ** 31 - 1, 2 ** 31, -2 ** 31 - 1, 2 ** 53 + 1, 1731465123457, -1731465123457, 2 ** 63 - 1,
             -2 ** 63]


def canary() -> list[str]:
    rng = random.Random(20261115)
    vals = list(FIXED)
    while len(vals) < len(FIXED) + 4000:
        x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if math.isfinite(x):
            vals.append(x)
    vals += [rng.random() for _ in range(3000)]
    vals += [round(rng.uniform(-1000, 1000), rng.randint(1, 6)) for _ in range(2000)]
    vals += [rng.uniform(-1e6, 1e6) for _ in range(1000)]
    ints = FIXED_I64 + [rng.randint(-2 ** 63, 2 ** 63 - 1) for _ in range(1000)]
    ref = hk.reference()
    p_bin = os.path.join(hk.DATA, "numbers", "canary.bin")
    arrays = hk.write_hk(p_bin, {"reference": ref, "count_f64": len(vals), "count_i64": len(ints),
                                 "fixed_f64": list(range(len(FIXED))), "fixed_i64": list(range(len(FIXED_I64))),
                                 "generator": gen_info("small.py")},
                         [("f64", "f64", np.asarray(vals, dtype="<f8")), ("i64", "i64", np.asarray(ints, dtype="<i8"))])
    hexes = [v.hex() for v in vals]
    p_json = os.path.join(hk.DATA, "numbers", "canary.json")
    write_json(p_json, {"reference": ref, "count_f64": len(vals), "count_i64": len(ints), "f64": vals,
                        "f64_hex": hexes, "i64": ints})
    # 자기 검사: 2진을 다시 읽어 비트로, 10진과 16진 글자를 다시 읽어 비트로
    _h, back = hk.read_hk(p_bin)
    assert not hk.same_arrays(arrays, back)
    with open(p_json, encoding="utf-8") as fh:
        j = json.load(fh)
    bad_dec = sum(1 for a, b in zip(back["f64"], j["f64"]) if bits(float(a)) != bits(float(b)))
    bad_hex = sum(1 for a, b in zip(back["f64"], j["f64_hex"]) if bits(float(a)) != bits(float.fromhex(b)))
    bad_int = sum(1 for a, b in zip(back["i64"], j["i64"]) if int(a) != b)
    print(f"canary: {len(vals)} f64, {len(ints)} i64; decimal mismatches {bad_dec}, hex mismatches {bad_hex}, "
          f"int mismatches {bad_int}")
    assert bad_dec == bad_hex == bad_int == 0
    return [p_bin, p_json]


# ---------------------------------------------------------------------------------------------- comparator
def tv(v):  # noqa: ANN001, ANN201
    if v is None:
        return {"t": "n"}
    if isinstance(v, bool):
        return {"t": "b", "v": v}
    if isinstance(v, int):
        return {"t": "i", "v": v}
    if isinstance(v, float):
        return {"t": "f", "hex": v.hex() if math.isfinite(v) else ("nan" if v != v else ("inf" if v > 0 else "-inf"))}
    if isinstance(v, str):
        return {"t": "s", "v": v}
    if isinstance(v, list):
        return {"t": "l", "v": [tv(x) for x in v]}
    if isinstance(v, dict):
        return {"t": "d", "v": {k: tv(x) for k, x in v.items()}}
    raise TypeError(type(v))


def has_nan(v) -> bool:  # noqa: ANN001
    if isinstance(v, float):
        return v != v
    if isinstance(v, list):
        return any(has_nan(x) for x in v)
    if isinstance(v, dict):
        return any(has_nan(x) for x in v.values())
    return False


def json_or_none(v):  # noqa: ANN001, ANN201
    try:
        return json.dumps(v, allow_nan=False, separators=(",", ":"))
    except ValueError:
        return None


def comparator() -> list[str]:
    from drowsy import vectors as V
    rows = []

    def row(section, key, expected, actual):  # noqa: ANN001
        tol = V.TOLERANCES.get(section, {})
        problems: list[str] = []
        V._compare(section, key, expected, actual, tol, f"expected.{section}", problems)
        rows.append({"id": len(rows), "section": section, "key": key, "tolerance": tol, "expected": tv(expected),
                     "actual": tv(actual), "expected_json": json_or_none(expected), "actual_json": json_or_none(actual),
                     "python_ok": not problems, "python_problems": problems,
                     "nan_row": has_nan(actual) or has_nan(expected)})

    float_keys = [("signals", "ear_l"), ("signals", "yaw"), ("closure", "closure"), ("baseline", "e_l"),
                  ("runs", "amplitude"), ("steps", "values"), ("model_inputs", "features"), ("step_probs", "probs"),
                  ("probs", "probs"), ("events", "confidence"), ("alerts", "confidence")]
    exps = [0.0, -0.0, 1.0, 0.123456789, -2.5, 1e6, 1e-300, 1731465123.457]
    for section, key in float_keys:
        tol = V.TOLERANCES[section]
        if section == "signals" and key == "yaw":
            a_t, r_t = tol["angle_abs_deg"], 0.0
        elif section in ("events", "alerts"):
            a_t, r_t = tol["confidence_abs"], 0.0
        else:
            a_t, r_t = tol.get("abs", 0.0), tol.get("rel", 0.0)
        for e in exps:
            lim = a_t + r_t * abs(e)
            cands = [e, -e, e + lim, e - lim, math.nextafter(e + lim, math.inf), math.nextafter(e + lim, -math.inf),
                     math.nextafter(e - lim, -math.inf), math.nextafter(e - lim, math.inf), e + 2 * lim,
                     float("nan"), float("inf"), float("-inf")]
            for a in cands:
                row(section, key, e, a)
            row(section, key, e, int(e) if math.isfinite(e) and abs(e) < 2 ** 53 else 0)
            row(section, key, e, True)
            row(section, key, e, None)
            row(section, key, e, "1.0")
        row(section, key, 1, 1.0)  # 정수로 적힌 기대값(벡터에 실수 칸이 정수로 적히는 일은 없지만 판정은 정해 둔다)
    time_keys = [("runs", "start_ms"), ("runs", "end_ms"), ("runs", "duration_ms"), ("events", "t_start_ms"),
                 ("events", "duration_ms"), ("alerts", "t_start_ms"), ("alerts", "duration_ms")]
    for section, key in time_keys:
        for e in (0, 1731465123457, -5000):
            for a in (e, e + 1, e - 1, e + 2, e - 2, float(e), float(e) + 0.5, True, None):
                row(section, key, e, a)
    for section, key in (("steps", "k"), ("steps", "t_end_ms"), ("counters", "model_calls"), ("events", "severity"),
                         ("closure", "t_ms")):
        for e in (0, 12, 1731465123457):
            for a in (e, e + 1, e - 1, float(e), True, False, None):
                row(section, key, e, a)
    for section, key in (("closure", "closure_valid"), ("events", "provisional"), ("steps", "full_lookback")):
        for e in (True, False):
            for a in (True, False, 1, 0, None, "true"):
                row(section, key, e, a)
    for section, key in (("events", "type"), ("events", "phase"), ("events", "source"), ("baseline", "status")):
        for e, a in (("sleep", "sleep"), ("sleep", "microsleep"), ("onset", "onset "), ("ok", "OK"), ("ok", None)):
            row(section, key, e, a)
    for section, key in (("signals", "iris_h_l"), ("probs", "probs"), ("baseline", "baseline")):
        for e, a in ((None, None), (None, 0.0), (None, 0), (0.5, None)):
            row(section, key, e, a)
    # 목록과 객체
    row("steps", "values", [0.0, 1.0], [0.0, 1.0])
    row("steps", "values", [0.0, 1.0], [0.0, 1.0, 2.0])
    row("steps", "values", [0.0, 1.0], [0.0])
    row("steps", "values", [0.0, 1.0], [0.0, float("nan")])
    row("steps", "values", [], [])
    row("steps", "values", [0.0], 0.0)
    row("events", "events", {"type": "sleep", "confidence": 1.0}, {"type": "sleep", "confidence": 1.0})
    row("events", "events", {"type": "sleep", "confidence": 1.0}, {"type": "sleep"})
    row("events", "events", {"type": "sleep", "confidence": 1.0}, {"type": "sleep", "confidence": 1.0, "x": 0})
    row("events", "events", {"type": "sleep", "confidence": 1.0}, {"type": "sleep", "conf": 1.0})
    row("events", "events", [{"t_start_ms": 5, "duration_ms": 0}], [{"t_start_ms": 6, "duration_ms": -1}])
    row("events", "events", [{"t_start_ms": 5, "duration_ms": 0}], [{"t_start_ms": 7, "duration_ms": 0}])
    row("counters", "counters", {"frames_accepted": 3}, {"frames_accepted": 3.0})
    row("model_inputs", "model_inputs", [[1.0, 2.0], [3.0]], [[1.0, 2.0], [3.0, 4.0]])
    p = os.path.join(hk.DATA, "numbers", "comparator.json")
    n_ok = sum(1 for r in rows if r["python_ok"])
    n_nan = sum(1 for r in rows if r["nan_row"])
    write_json(p, {"reference": hk.reference(), "generator": gen_info("small.py"),
                   "function": "drowsy.vectors._compare", "depends_on": hk.depends_on("drowsy/vectors.py"),
                   "rows": rows})
    # 자기 검사: 다시 읽은 typed value로 다시 판정하면 같다
    with open(p, encoding="utf-8") as fh:
        back = json.load(fh)

    def untv(x):  # noqa: ANN001, ANN202
        t = x["t"]
        if t == "n":
            return None
        if t in ("b", "i", "s"):
            return x["v"]
        if t == "f":
            return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}.get(x["hex"]) \
                if x["hex"] in ("nan", "inf", "-inf") else float.fromhex(x["hex"])
        if t == "l":
            return [untv(y) for y in x["v"]]
        return {k: untv(y) for k, y in x["v"].items()}
    redo = 0
    for r in back["rows"]:
        probs: list[str] = []
        V._compare(r["section"], r["key"], untv(r["expected"]), untv(r["actual"]), r["tolerance"],
                   f"expected.{r['section']}", probs)
        if (not probs) != r["python_ok"] or probs != r["python_problems"]:
            redo += 1
    print(f"comparator: {len(rows)} rows, python_ok {n_ok}, nan rows {n_nan}, re-judged differently {redo}")
    assert redo == 0
    return [p]


# ---------------------------------------------------------------------------------------------- active fold
def active() -> list[str]:
    from drowsy import demo
    from drowsy.spec import apply_overrides, load_rules
    base_rules = load_rules(os.path.join(hk.REPO, "export", "rules.json"))
    out_files, report = [], []
    total_frames = 0
    for path in sorted(glob.glob(os.path.join(hk.REPO, "tests", "vectors", "*.json"))):
        with open(path, encoding="utf-8") as fh:
            vec = json.load(fh)
        if "events" not in vec["expected"]:
            continue
        name = vec["name"]
        rules = apply_overrides(base_rules, vec.get("config_overrides") or {})
        prio = list(rules.alert_priority)
        evs = vec["expected"]["events"]
        events = [RT.DriverStateEvent(**{k: e[k] for k in ("type", "severity", "t_start_ms", "duration_ms",
                                                            "confidence", "source", "phase", "provisional")})
                  for e in evs]
        frames, order = [], []
        for e, ev in zip(evs, events):
            if not order or order[-1][0] != e["t_ms"]:
                order.append((e["t_ms"], []))
            order[-1][1].append(ev)
        act: dict = {}
        ever = set()
        leave_at: dict = {}  # (type, t_start) -> 빠질 때의 이벤트
        for t_ms, group in order:
            before = dict(act)
            hk.fold_active(act, group, t_ms)
            for typ, ent in act.items():
                ever.add((typ, ent["t_start_ms"]))
            for typ, ent in before.items():
                if typ not in act or act[typ]["t_start_ms"] != ent["t_start_ms"]:
                    # 이 프레임에서 빠졌다: 원인 이벤트 = 같은 종류의 end, 또는 sleep onset/end(microsleep)
                    cause = next((g for g in group if g.type == typ and g.phase == RT.PHASE_END), None)
                    if cause is None and typ == RT.EVENT_MICROSLEEP:
                        cause = next((g for g in group if g.type == RT.EVENT_SLEEP), None)
                    leave_at[(typ, ent["t_start_ms"])] = cause
            frames.append({"t_ms": t_ms, "active": hk.active_list(act, prio)})
        # demo.event_spans와 맞춰 본다
        records = [demo.EventRecord(t_ms=e["t_ms"], event=ev, alert=False) for e, ev in zip(evs, events)]
        last_t = vec["frames"][-1]["t_ms"] if vec["frames"] else 0
        spans = demo.event_spans(records, last_t)
        span_set = {(s.type, s.start_ms) for s in spans}
        problems = []
        if span_set != ever:
            problems.append(f"ever-active {sorted(ever)} != spans {sorted(span_set)}")
        exceptions = 0
        sleep_onsets = {}
        for e, ev in zip(evs, events):
            if ev.type == RT.EVENT_SLEEP and ev.phase == RT.PHASE_ONSET:
                sleep_onsets[e["t_ms"]] = ev
        for s in spans:
            key = (s.type, s.start_ms)
            cause = leave_at.get(key)
            if s.ended:
                if cause is None:
                    problems.append(f"span {key} ended at {s.end_ms} but never left the view")
                    continue
                left_end = cause.t_start_ms + cause.duration_ms
                if s.type == RT.EVENT_MICROSLEEP and cause.type == RT.EVENT_SLEEP and cause.phase == RT.PHASE_ONSET:
                    exceptions += 1  # 선언한 예외: sleep onset에서 빠지고 구간은 sleep end에서 끝난다
                elif left_end != s.end_ms:
                    problems.append(f"span {key} ends {s.end_ms}, left the view at {left_end}")
            else:
                if s.type == RT.EVENT_MICROSLEEP and cause is not None and cause.type == RT.EVENT_SLEEP:
                    exceptions += 1  # 끝나지 않은 sleep의 microsleep: 보기에서는 sleep onset에서 빠졌다
                elif s.type not in act or act[s.type]["t_start_ms"] != s.start_ms:
                    problems.append(f"open span {key} not active at the end")
        open_spans = {(s.type, s.start_ms) for s in spans if not s.ended}
        for typ, ent in act.items():
            if (typ, ent["t_start_ms"]) not in open_spans:
                problems.append(f"active at the end {typ} has no open span")
        if problems:
            raise AssertionError(f"active fold vs demo.event_spans, {name}: {problems}")
        with open(path, "rb") as fh:
            vsha = hashlib.sha256(fh.read()).hexdigest()
        p = os.path.join(hk.DATA, "active_expected", f"{name}.json")
        write_json(p, {"reference": hk.reference(), "generator": gen_info("small.py"), "vector": name,
                       "vector_sha256": vsha, "alert_priority": prio, "frames": frames})
        out_files.append(p)
        total_frames += len(frames)
        report.append({"vector": name, "events": len(evs), "frames": len(frames), "spans": len(spans),
                       "microsleep_exceptions": exceptions, "open_at_end": len(act)})
    p = os.path.join(hk.DATA, "active_expected", "index.json")
    write_json(p, {"reference": hk.reference(), "generator": gen_info("small.py"), "vectors": report,
                   "check": "ever-active (type, t_start_ms) == demo.event_spans spans; leave event end == span end "
                            "except microsleep leaving at a sleep onset; active at end == spans with ended False "
                            "(minus microsleep removed by a sleep onset)"})
    print(f"active: {len(report)} vectors, {total_frames} frames with events, "
          f"{sum(r['events'] for r in report)} events, microsleep exceptions {sum(r['microsleep_exceptions'] for r in report)}, "
          f"vectors with an empty event list {sum(1 for r in report if r['events'] == 0)}, all agree with demo.event_spans")
    return out_files + [p]


# ---------------------------------------------------------------------------------------------- replay0
def _out_json(out: RT.PipelineOutput) -> dict:
    st = out.state
    return {
        "state": {f: getattr(st, f) for f in ("t_ms", "face", "closure", "closure_valid", "provisional", "yaw_rel",
                                              "pitch_rel", "pose_valid", "gaze_h_rel", "gaze_v_rel", "jaw_rel")},
        "runs": [{f: getattr(r, f) for f in ("start_ms", "end_ms", "duration_ms", "amplitude", "peak_closing_velocity",
                                             "peak_opening_velocity", "velocity_valid", "truncated")} for r in out.runs],
        "steps": [{"k": s.k, "t_end_ms": s.t_end_ms, "full_lookback": s.full_lookback, "values": list(s.values)}
                  for s in out.steps],
        "step_probs": [None if p is None else list(p) for p in out.step_probs],
        "probs": None if out.probs is None else list(out.probs),
        "events": [{f: getattr(e, f) for f in EV_FIELDS} for e in out.events],
        "alerts": [{f: getattr(e, f) for f in EV_FIELDS} for e in out.alerts],
    }


EV_FIELDS = ("type", "severity", "t_start_ms", "duration_ms", "confidence", "source", "phase", "provisional")


def _replay_scenarios(spec):  # noqa: ANN001, ANN202
    from drowsy import synth as Y
    segs = []
    t = 0
    # 0-60 s 보정: 가끔 깜빡임
    for k in range(12):
        segs.append(Y.Blink(duration_ms=200, depth=1.0, start_ms=3000 + 5000 * k))
    segs += [
        Y.Closure(hold_ms=1500, depth=1.0, start_ms=70000),            # microsleep
        Y.Closure(hold_ms=4000, depth=1.0, start_ms=80000),            # microsleep -> sleep
        Y.HeadTurn(duration_ms=5000, yaw=40.0, start_ms=95000),        # gaze_away_long
    ]
    for k in range(6):                                                 # vats: 2.5 s 밖, 1 s 안
        segs.append(Y.HeadTurn(duration_ms=2500, yaw=35.0, ramp_ms=100, start_ms=110000 + 3500 * k))
    segs.append(Y.FaceMissing(duration_ms=8000, start_ms=150000))      # unable_to_analyze
    for k in range(14):                                                # PERCLOS: drowsy
        segs.append(Y.Closure(hold_ms=1300, depth=1.0, close_ms=100, open_ms=100, start_ms=175000 + 5000 * k))
    segs.append(Y.Yawn(duration_ms=4000, jaw=0.6, start_ms=262000))
    segs.append(Y.Nod(duration_ms=1000, pitch=20.0, start_ms=270000))
    main = Y.Scenario(segments=tuple(segs), duration_ms=340000, fps=15.0, first_t_ms=1, seed=7)
    speed_main = [(0, 60.0), (100000, 10.0), (135000, 80.0), (300000, None)]
    segs2 = [Y.HeadTurn(duration_ms=9000, yaw=35.0, start_ms=73000), Y.Closure(hold_ms=5000, depth=1.0, start_ms=76000),
             Y.Closure(hold_ms=2000, depth=1.0, start_ms=104000)]
    gap = Y.Scenario(segments=tuple(segs2), duration_ms=130000, fps=10.0, first_t_ms=1731465123457, seed=8,
                     gaps=((80000, 92000),))
    speed_gap = [(0, None), (60000, 30.0), (100000, 5.0)]
    return [("replay0_main", "self-calibration, microsleep, sleep, gaze_away_long, vats, unable_to_analyze, drowsy "
                             "(PERCLOS), yawn, nod; speed below the alert gate between 100 and 135 s", main, speed_main),
            ("replay0_gap", "large first t_ms, a sleep and a head turn cut by a 12 s frame gap (reset ends), "
                            "speed null then 30 then 5 km/h", gap, speed_gap)]


def replay0() -> list[str]:
    from drowsy import synth as Y
    from drowsy.spec import load_feature_spec, load_rules
    fspec = load_feature_spec(os.path.join(hk.REPO, "export", "feature_spec_v0.json"))
    spec = fspec.pipeline
    rules = load_rules(os.path.join(hk.REPO, "export", "rules.json"))
    files = []
    for name, desc, sc, speeds in _replay_scenarios(spec):
        frames = Y.render(sc, spec)

        def run():  # noqa: ANN202
            hk.FEED.mode = "RAW"
            p = hk.DriverPipeline(spec, rules)
            act: dict = {}
            rows = []
            seen: dict = {}
            for fr in frames:
                st = fr.t_ms - sc.first_t_ms
                sp = None
                for t0, v in speeds:
                    if st >= t0:
                        sp = v
                out = p.update(fr, sp)
                row = {"t_ms": fr.t_ms, "speed_kmh": sp, "output": None if out is None else _out_json(out)}
                if out is not None:
                    hk.fold_active(act, out.events, out.state.t_ms)
                    for e in out.events:
                        seen.setdefault(e.type, {"onset": 0, "end": 0})[e.phase] += 1
                row["gaze_zone"] = p.gaze_zone
                row["baseline"] = None if p.baseline is None else hk.baseline_plain(p.baseline)
                row["counters"] = {"frames_accepted": p.frames_accepted, "dropped_frames": p.dropped_frames,
                                   "steps_emitted": p.steps_emitted, "model_calls": p.model_calls,
                                   "nonfinite_windows": p.nonfinite_windows}
                row["active"] = hk.active_list(act, rules.alert_priority)
                rows.append(row)
            return rows, seen
        rows, seen = run()
        rows2, _ = run()
        same = json.dumps(rows, allow_nan=False) == json.dumps(rows2, allow_nan=False)
        assert same
        missing = [t for t in RT.EVENT_TYPES if seen.get(t, {}).get("onset", 0) == 0]
        if name == "replay0_main" and missing:
            raise AssertionError(f"{name}: event types without onset: {missing}")
        p = os.path.join(hk.DATA, "replay0", f"{name}.json")
        write_json(p, {"reference": hk.reference(), "generator": gen_info("small.py"), "format": "hawkeye-replay0",
                       "version": 1,
                       "session": {"name": name, "description": desc, "fps": sc.fps, "first_t_ms": sc.first_t_ms,
                                   "duration_ms": sc.duration_ms, "pipeline_hash": fspec.pipeline_hash,
                                   "rules_hash": hk.SHIPPED_RULES_HASH, "model": None, "baseline_injected": None,
                                   "speed_profile": [[t0, v] for t0, v in speeds],
                                   "speed_rule": "speed of a frame = value of the last entry whose session-time start "
                                                 "<= t_ms - first_t_ms"},
                       "events_seen": seen, "frames": rows, "counters": rows[-1]["counters"]})
        files.append(p)
        print(f"replay0: {name}: {len(rows)} frames, events {json.dumps(seen, sort_keys=True)}, "
              f"missing onsets {missing}, second run identical {same}")
    return files


# ---------------------------------------------------------------------------------------------- model windows
def _harvest_steps(n_sessions: int = 40) -> list[list[list[float]]]:
    """S 생성기의 inj_model 세션과 F-cal식 각본에서 120개 연속 스텝 창을 자른다."""
    import gen_s
    wins = []
    for seed in range(n_sessions):
        S = gen_s.make("harvest", "inj_model", 9100 + seed)
        S.stub = None
        r = hk.run_session(S, "RAW", margins=False)
        vals = r.c.get("st_values", [])
        full = r.c.get("st_full", [])
        rows = [v for v, f in zip(vals, full) if f]
        for start in range(0, max(0, len(rows) - 120), 23):
            wins.append(rows[start:start + 120])
    return wins


def model_windows() -> list[str]:
    import onnxruntime as ort
    from drowsy import vectors as V  # noqa: F401  (세션 설정을 벡터와 같게: 스레드 1)
    mpath = os.path.join(hk.REPO, "export", "drowsiness_v0.onnx")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(mpath, opts, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rng = np.random.default_rng(20261115)
    harvested = _harvest_steps()
    real = [harvested[i] for i in sorted(rng.choice(len(harvested), size=100, replace=len(harvested) < 100))]
    wins, kinds = [], []
    wins += [np.asarray(w, dtype=np.float64) for w in real]
    kinds += [0] * 100
    wins += [rng.uniform(-3.0, 9.0, size=(120, 17)) for _ in range(100)]
    kinds += [1] * 100
    f32max = float(np.finfo(np.float32).max)
    extremes = [np.zeros((120, 17)), np.full((120, 17), 1e6), np.full((120, 17), 1e30), np.full((120, 17), -1e30),
                np.full((120, 17), f32max), np.full((120, 17), -f32max), np.full((120, 17), 1e-45),
                np.full((120, 17), -0.0), np.full((120, 17), 1e-38)]
    while len(extremes) < 50:
        k = len(extremes)
        base = rng.uniform(-1.0, 1.0, size=(120, 17)) * (10.0 ** rng.integers(-8, 31))
        if k % 3 == 0:
            base = -np.abs(base)
        if k % 5 == 0:
            base[rng.integers(0, 120), rng.integers(0, 17)] = f32max
        extremes.append(base)
    wins += extremes
    kinds += [2] * 50
    base = np.asarray(real[0], dtype=np.float64)
    pos = set()
    while len(pos) < 50:
        pos.add((int(rng.integers(0, 120)), int(rng.integers(0, 17))))
    pos_list = sorted(pos)
    for (t, f) in pos_list:
        w = base.copy()
        w[t, f] = w[t, f] + 100.0
        wins.append(w)
    kinds += [3] * 50
    W64 = np.stack(wins).astype("<f8")
    W32 = W64.astype(np.float32)
    probs = np.stack([sess.run(None, {iname: W32[i:i + 1]})[0][0] for i in range(300)]).astype("<f4")
    probs2 = np.stack([sess.run(None, {iname: W32[i:i + 1]})[0][0] for i in range(300)]).astype("<f4")
    assert probs.tobytes() == probs2.tobytes(), "onnxruntime gave different outputs on the same window"
    tri = float(np.max(np.abs(probs.astype(np.float64).sum(axis=1) - 1.0)))
    with open(os.path.join(hk.REPO, "tests", "vectors", "e2e_250s.json"), encoding="utf-8") as fh:
        e2e = json.load(fh)
    w0 = np.asarray(e2e["expected"]["model_inputs"][0]["features"], dtype=np.float64)
    orders = [w0, w0[::-1, :].copy(), w0[:, ::-1].copy()]
    O32 = np.stack(orders).astype(np.float32)
    oprobs = np.stack([sess.run(None, {iname: O32[i:i + 1]})[0][0] for i in range(3)]).astype("<f4")
    d_steps = float(np.max(np.abs(oprobs[1].astype(np.float64) - oprobs[0])))
    d_feats = float(np.max(np.abs(oprobs[2].astype(np.float64) - oprobs[0])))
    kept = [n for n, d in (("reversed_steps", d_steps), ("reversed_features", d_feats)) if d > 1e-4]
    e2e_expected = e2e["expected"]["probs"][0]["probs"]
    d_e2e = float(np.max(np.abs(oprobs[0].astype(np.float64) - np.asarray(e2e_expected))))
    conv = [0.1, 1 / 3, 1.0 + 2.0 ** -24, 1.0 + 3 * 2.0 ** -24, 1.0 - 2.0 ** -25, 1.0 + 2.0 ** -24 + 2.0 ** -60,
            1e-40, 1.4e-45, 7.006492321624085e-46, 7.006492321624087e-46, 2.0 ** -149 * 1.5, 1e-45, 1e30, -1e30,
            3.4028234663852886e38, 3.4028235e38 * 0.999999, 16777217.0, 16777219.0, -0.0, 0.0, 123456.789, 1e-7,
            1592.0000000001, 0.30000000000000004, 2.0 ** -126, 2.0 ** -126 * (1 - 2.0 ** -24)]
    conv += list(rng.uniform(-1e6, 1e6, size=200)) + list(rng.standard_normal(200) * 10.0 ** rng.integers(-40, 31, 200))
    C64 = np.asarray(conv, dtype="<f8")
    C32 = C64.astype(np.float32)
    assert np.all(np.isfinite(C32))
    p = os.path.join(hk.DATA, "model", "model_windows.bin")
    hdr = {"reference": hk.reference(), "generator": gen_info("small.py", "gen_s.py"),
           "model_sha256": hk.sha256_file(mpath), "onnxruntime_version": ort.__version__,
           "session_options": {"intra_op_num_threads": 1, "inter_op_num_threads": 1,
                               "providers": ["CPUExecutionProvider"]},
           "input_name": iname, "seed": 20261115,
           "kinds": {"0": "real-scale windows cut from step values of S inj_model sessions (gen_s inj_model seeds 9100..9139, every 23rd step start)",
                     "1": "uniform [-3, 9]", "2": "extreme finite", "3": "base window (index 0) with one element +100"},
           "changed_positions": [list(x) for x in pos_list],
           "order_check": {"max_abs_diff_reversed_steps": d_steps, "max_abs_diff_reversed_features": d_feats,
                           "kept": kept, "e2e_first_window_vs_vector_max_abs_diff": d_e2e,
                           "rows": ["e2e_250s expected.model_inputs[0]", "steps reversed", "features reversed"]},
           "triple_sum_max_dev": tri, "deterministic_second_run": True}
    arrays = hk.write_hk(p, hdr, [("windows_f64", "f64", W64), ("windows_f32", "f32", W32.astype("<f4")),
                                  ("probs", "f32", probs), ("kind", "u8", np.asarray(kinds, dtype="<u1")),
                                  ("order_windows", "f32", O32.astype("<f4")), ("order_probs", "f32", oprobs),
                                  ("conv_f64", "f64", C64), ("conv_f32", "f32", C32.astype("<f4"))])
    _h, back = hk.read_hk(p)
    assert not hk.same_arrays(arrays, back)
    rerun = np.stack([sess.run(None, {iname: back["windows_f32"][i:i + 1].copy()})[0][0] for i in range(300)])
    assert rerun.astype("<f4").tobytes() == back["probs"].tobytes()
    print(f"model_windows: 300 windows (kinds {np.bincount(kinds).tolist()}), harvested {len(harvested)} real windows, "
          f"triple sum max dev {tri:.3g}, reversed steps diff {d_steps:.3g}, reversed features diff {d_feats:.3g}, "
          f"kept {kept}, e2e first window vs vector {d_e2e:.3g}, re-run from re-read file bit-equal")
    return [p]
