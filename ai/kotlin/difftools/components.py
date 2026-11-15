"""부품 자료 C-* (계획 9.2.6절, FORMATS.md 2.7절). 부품 하나의 호출 기록을 바꾸지 않은 Python 객체에서 만든다.

복사해 둔 생성기의 함수는 `vendor_funcs`로 그 파일의 소스에서 함수 정의만 읽어 쓴다(파일 맨 위의 명령줄 처리와
지워진 임시 폴더 경로를 건너뛰려고. 파일은 고치지 않는다).
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import random
import sys

import hk
from small import gen_info, write_json

RT = hk.RT
from drowsy import calibrate as C  # noqa: E402
from drowsy import rules as R  # noqa: E402
from drowsy import signals as SG  # noqa: E402
from drowsy import spec as SP  # noqa: E402
from drowsy.blink import ClosureRunDetector  # noqa: E402
from drowsy.step_features import NodDetector, StepAggregator, ThresholdEventCounter  # noqa: E402

V = os.path.join(hk.HERE, "vendor")
CD = os.path.join(hk.DATA, "components")


def vendor_funcs(rel: str, names: list[str], ns: dict) -> dict:
    path = os.path.join(V, rel)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in keep} == set(names), (rel, names)
    mod = ast.Module(body=keep, type_ignores=[])
    exec(compile(mod, path, "exec"), ns)  # noqa: S102
    return ns


def review_ns() -> dict:
    """복사한 review/rules/common.py와 gen.py를 import한다(맨 위 코드는 저장소 경로와 설정 읽기뿐이다)."""
    d = os.path.join(V, "review_rules")
    if d not in sys.path:
        sys.path.insert(0, d)
    import common  # noqa: F401
    import gen  # noqa: F401
    ns = {"random": random, "st": common.st, "step_values": gen.step_values, "frame_stream": gen.frame_stream,
          "speed_stream": gen.speed_stream, "step": common.step, "probs": common.probs,
          "DriverStateEvent": RT.DriverStateEvent}
    vendor_funcs("verify_ported/rules/r01_fuzz_engine.py", ["knife_stream", "make_steps"], ns)
    vendor_funcs("verify_ported/rules/r02_alert_modelcheck.py", ["synthetic_calls"], ns)
    return ns


def gfiles() -> dict:
    g = gen_info("small.py", "components.py")
    for rel in ("verify_ported/rules/r01_fuzz_engine.py", "verify_ported/rules/r02_alert_modelcheck.py",
                "review_rules/gen.py", "review_rules/common.py", "verify_ported/frame/d01_detector_fuzz.py",
                "r5bv_spec_tx/tx_fuzz.py"):
        g["files"]["vendor/" + rel] = hk.sha256_file(os.path.join(V, rel))
    return g


def plain(x):  # noqa: ANN001, ANN201
    if isinstance(x, tuple):
        return [plain(v) for v in x]
    if isinstance(x, list):
        return [plain(v) for v in x]
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if x != x else ("Infinity" if x > 0 else "-Infinity")
    return x


def st_json(s: RT.FrameState) -> dict:
    return {f: getattr(s, f) for f in ("t_ms", "face", "closure", "closure_valid", "provisional", "yaw_rel", "pitch_rel",
                                       "pose_valid", "gaze_h_rel", "gaze_v_rel", "jaw_rel")}


def step_json(s: RT.StepFeatures) -> dict:
    return {"k": s.k, "t_end_ms": s.t_end_ms, "full_lookback": s.full_lookback, "values": list(s.values)}


def ev_json(e: RT.DriverStateEvent) -> dict:
    return {f: getattr(e, f) for f in ("type", "severity", "t_start_ms", "duration_ms", "confidence", "source", "phase",
                                       "provisional")}


def run_list(r) -> list | None:  # noqa: ANN001
    if r is None:
        return None
    return [r.start_ms, r.end_ms, r.duration_ms, r.amplitude, r.peak_closing_velocity, r.peak_opening_velocity,
            r.velocity_valid, r.truncated]


def sig_json(s: RT.FrameSignals) -> dict:
    d = {f: getattr(s, f) for f in ("t_ms", "face", "ear_l", "ear_r", "eye_w_l", "eye_w_r", "blink_l", "blink_r",
                                    "jaw_open", "yaw", "pitch", "roll", "iris_h_l", "iris_h_r", "iris_v_l", "iris_v_r")}
    d["eye_look"] = list(s.eye_look)
    return d


def bl_json(b) -> dict | None:  # noqa: ANN001
    return None if b is None else hk.baseline_plain(b)


def err(exc: Exception) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)}


def save(name: str, component: str, cases: list, extra: dict | None = None) -> str:
    p = os.path.join(CD, f"{name}.json")
    obj = {"reference": hk.reference(), "generator": gfiles(), "component": component, "cases": plain(cases)}
    if extra:
        obj.update(plain(extra))
    write_json(p, obj)
    n = len(cases)
    calls = sum(len(c.get("calls", [])) if isinstance(c.get("calls"), list) else 1 for c in cases)
    print(f"  {name}: {n} cases, {calls} calls, {os.path.getsize(p)} bytes", flush=True)
    return p


# ---------------------------------------------------------------------------------------------- C-func
def c_func(rng: random.Random) -> str:
    cases = []

    def call(fn, f, *args, compare="exact"):  # noqa: ANN001
        try:
            res = f(*args)
            cases.append({"fn": fn, "args": list(args), "result": res, "compare": compare})
        except Exception as exc:  # noqa: BLE001
            cases.append({"fn": fn, "args": list(args), "error": err(exc), "compare": compare})
    qs = [0.0, 0.05, 0.1, 0.25, 0.3, 0.5, 0.6, 0.7, 0.75, 0.9, 0.95, 1.0]
    nmax = 2000
    rank = []
    for q in qs:
        row = []
        vals: list[float] = []
        for n in range(1, nmax + 1):
            vals.append(float(n - 1))
            row.append(int(C.nearest_rank_quantile(vals, q)) + 1)
        rank.append(row)
    pool = [0.0, -0.0, 1.0, -1.0, 0.5, 0.25, 0.1, 0.3, 0.27, 0.12, 0.45, 2.0, 1e-300, -1e-300, 15.0, -15.0]
    for _ in range(600):
        n = rng.choice([1, 2, 3, 4, 5, 6, 7, 10, 11, 20, 31, 60])
        vals = [rng.choice(pool) if rng.random() < 0.5 else rng.uniform(-2, 2) for _ in range(n)]
        call("nearest_rank_quantile", C.nearest_rank_quantile, vals, rng.choice(qs + [0.33, 0.66, 0.999, 1.5, -0.5]))
        call("median", C.median, vals)
    for n in range(300, 401, 10):
        vals = [rng.uniform(0.1, 0.5) for _ in range(n)]
        for q in (0.1, 0.5, 0.9):
            call("nearest_rank_quantile", C.nearest_rank_quantile, vals, q)
        call("median", C.median, vals)
    for vals in ([0.0, -0.0], [-0.0, 0.0], [-0.0, -0.0], [0.0, -0.0, 0.0], [-0.0, 0.0, -0.0], [0.0, 0.0, -0.0, -0.0],
                 [-0.0, 0.0, 0.0, -0.0], [1.0, -0.0, 0.0, -1.0], [-0.0], [0.0]):
        call("median", C.median, vals)
        call("nearest_rank_quantile", C.nearest_rank_quantile, vals, 0.5)
    call("median", C.median, [])
    call("nearest_rank_quantile", C.nearest_rank_quantile, [], 0.5)
    cfgs = [dict(rho=0.35, b_closed=0.6, b_span_min=0.2, w=0.5), dict(rho=1.0, b_closed=0.6, b_span_min=0.2, w=0.5),
            dict(rho=0.2, b_closed=0.8, b_span_min=0.4, w=0.0), dict(rho=0.5, b_closed=0.5, b_span_min=0.1, w=1.0),
            dict(rho=1.5, b_closed=0.6, b_span_min=0.2, w=0.3)]
    for _ in range(800):
        cfg = rng.choice(cfgs)
        cs = SP.ClosureSpec(**cfg, far_eye_zero_yaw_deg=25.0, far_eye_ramp_deg=10.0, valid_max_abs_yaw_deg=40.0)
        e = rng.choice([0.27, 0.0, 0.3, 0.12, -0.1, rng.uniform(0, 0.5)])
        b = rng.choice([0.22, 0.6, 0.7, 0.0, rng.uniform(0, 0.8)])
        ear = rng.choice([0.0, e, e * 0.35, rng.uniform(-0.1, 0.6)])
        bl = rng.choice([0.0, b, 1.0, rng.uniform(-0.1, 1.1)])
        try:
            res = C.eye_closure(ear, bl, e, b, cs)
            cases.append({"fn": "eye_closure", "args": [ear, bl, e, b, cfg], "result": res, "compare": "exact"})
        except Exception as exc:  # noqa: BLE001
            cases.append({"fn": "eye_closure", "args": [ear, bl, e, b, cfg], "error": err(exc), "compare": "exact"})
    for _ in range(400):
        cx, cy, w = rng.uniform(50, 600), rng.uniform(50, 400), rng.choice([0.0, 0.0, 3.0, 6.0, rng.uniform(1, 80)])
        hh = rng.uniform(0, 0.5) * w
        pts = [(cx - w / 2, cy), (cx - w / 6, cy - hh / 2), (cx + w / 6, cy - hh / 2), (cx + w / 2, cy),
               (cx + w / 6, cy + hh / 2), (cx - w / 6, cy + hh / 2)]
        if rng.random() < 0.3:
            pts = [(rng.uniform(0, 640), rng.uniform(0, 480)) for _ in range(6)]
        call("eye_aspect_ratio", SG.eye_aspect_ratio, [list(p) for p in pts], compare="abs1e-9")
    call("eye_aspect_ratio", SG.eye_aspect_ratio, [[1.0, 1.0]] * 5, compare="abs1e-9")
    for _ in range(300):
        o = [rng.uniform(0, 640), rng.uniform(0, 480)]
        i = o if rng.random() < 0.1 else [o[0] + rng.uniform(-60, 60), o[1] + rng.uniform(-20, 20)]
        c = [rng.uniform(0, 640), rng.uniform(0, 480)]
        call("iris_ratio", SG.iris_ratio, o, i, c)
    for _ in range(300):
        el = [rng.choice([0.0, rng.random()]) for _ in range(8)]
        call("gaze_h", SG.gaze_h, el)
        call("gaze_v", SG.gaze_v, el)
    for _ in range(400):
        y, p, r = rng.uniform(-170, 170), rng.uniform(-89, 89), rng.uniform(-170, 170)
        if rng.random() < 0.2:
            y, p, r = rng.choice([0.0, 15.0, -15.0, 23.0, 40.0, 90.0]), rng.choice([0.0, 18.0, -25.0]), 0.0
        call("rotation_from_euler", SG.rotation_from_euler, y, p, r, compare="abs1e-9")
        rot = list(SG.rotation_from_euler(y, p, r))
        m = [rot[0], rot[3], rot[6], 0.0, rot[1], rot[4], rot[7], 0.0, rot[2], rot[5], rot[8], 0.0,
             rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-60, -30), 1.0]
        call("euler_from_colmajor", SG.euler_from_colmajor, m, compare="abs1e-9")
        call("euler_from_rotation", SG.euler_from_rotation, rot, compare="abs1e-9")
        call("rotation_from_colmajor", SG.rotation_from_colmajor, m)
        r0 = list(SG.rotation_from_euler(rng.uniform(-30, 30), rng.uniform(-15, 15), rng.uniform(-5, 5)))
        call("relative_rotation", SG.relative_rotation, rot, r0)
        call("relative_euler", SG.relative_euler, m, r0, compare="abs1e-9")
    for num, den in ((0, 0), (5, 0), (5, -1), (0, 10), (3, 10), (10, 10), (1, 3), (2, 3), (59999, 60000),
                     (1731465123457, 1731465123458), (7, 30000), (29900, 60000)):
        call("ratio", R._ratio, num, den)
    for _ in range(100):
        den = rng.randint(-5, 120000)
        call("ratio", R._ratio, rng.randint(0, max(0, den)), den)
    return save("C-func", "functions", cases, {"rank_table": {"qs": qs, "n_max": nmax, "rank": rank}})


# ---------------------------------------------------------------------------------------------- C-detector
DET_BASE = dict(enter_thr=0.5, exit_thr=0.3, min_duration_ms=50, max_dt_ms=200, hold_ms=500, velocity_lookback_samples=2)


def det_cfgs(rng: random.Random) -> list[dict]:
    out = [dict(DET_BASE), dict(DET_BASE, enter_thr=0.7, exit_thr=0.5)]
    clock = SP.ClockSpec(step_ms=1000, max_hold_ms=200, reset_gap_ms=10000)
    while len(out) < 40:
        max_dt = rng.choice([200, 100, 300, 500, 50])
        en = rng.choice([0.5, 0.7, 0.8, 0.6])
        d = dict(enter_thr=en, exit_thr=rng.choice([en - 0.2, en - 0.1, 0.3]), min_duration_ms=rng.choice([0, 50, 300]),
                 max_dt_ms=max_dt, hold_ms=rng.choice([max_dt, 500, 1000, 250]),
                 velocity_lookback_samples=rng.choice([1, 2, 3, 4]))
        try:
            SP._validate_run_detector(SP.RunDetectorSpec(**d), clock, "d")
        except SP.SpecError:
            continue
        out.append(d)
    return out


def det_stream(rng: random.Random, cfg: dict) -> list:
    """d01_detector_fuzz.gen_samples의 방식(임계값 둘레의 수준, 무효 구간, 끊김)에 정확한 x.5 장면과 reset을 더한다."""
    en, ex, hold, max_dt = cfg["enter_thr"], cfg["exit_thr"], cfg["hold_ms"], cfg["max_dt_ms"]
    t = rng.choice([0, 1, rng.randrange(0, 50), -5000, 1731465123457])
    calls = []
    if rng.random() < 0.3:  # 처음이나 둘째 표본부터 감긴 채
        if rng.random() < 0.5:
            calls.append(["u", t, 0.1, True])
            t += rng.choice([33, 1])
        calls.append(["u", t, 1.0, True])
        t += 33
    eps = rng.choice([1e-9, 1e-6, 0.01])
    mid = (en + ex) / 2
    levels = {"open": [0.0, 0.05, ex - eps, ex - 0.1], "mid": [ex, ex + eps, mid, en - eps, en - 0.05],
              "closed": [en, en + eps, en + 0.1, 0.95, 1.0]}
    for _ in range(rng.choice([10, 30, 60])):
        kind = rng.choices(["open", "mid", "closed", "ramp", "noise", "invalid", "gap", "half", "reset"],
                           weights=[5, 6, 7, 3, 3, 3, 3, 4, 1])[0]
        if kind == "reset":
            calls.append(["r"])
            continue
        if kind == "gap":
            t += rng.choice([max_dt, max_dt + 1, hold, hold + 1, hold - 1, 250, 700, 3000, 9999, 10000, 10001, 15000])
            continue
        if kind == "half":  # 0과 1 사이를 홀수 간격으로: 보간 오프셋 x.5
            lo, hi = rng.choice([(0.0, 1.0), (0.2, 0.8), (1.0, 0.0), (0.8, 0.2)])
            for c in (lo, hi, hi, lo):
                calls.append(["u", t, c, True])
                t += rng.choice([33, 67, 101, 1, 3])
            continue
        n = rng.choice([1, 1, 2, 3, 5, 8, 20])
        lv = rng.choice(levels[kind]) if kind in levels else None
        a, b = rng.random(), rng.random()
        for j in range(n):
            if kind == "invalid":
                calls.append(["u", t, 0.0, False])
            else:
                if lv is not None:
                    c = lv if rng.random() < 0.8 else rng.choice(levels[kind])
                elif kind == "ramp":
                    c = a + (b - a) * (j + 1) / n
                else:
                    c = rng.random()
                calls.append(["u", t, c, True])
            dt = rng.choice([33, 33, 40, 66, 100, 150]) + rng.randrange(-6, 7)
            if rng.random() < 0.05:
                dt = rng.choice([max_dt, max_dt + 1, hold, hold + 1])
            t += max(1, dt)
    return calls


def c_detector(rng: random.Random) -> str:
    cases = []
    cfgs = det_cfgs(rng)
    for i in range(400):
        cfg = cfgs[i % len(cfgs)]
        calls = det_stream(rng, cfg)
        d = ClosureRunDetector(SP.RunDetectorSpec(**cfg))
        out = []
        last_t = None
        for c in calls:
            if c[0] == "r":
                d.reset()
                last_t = None
            else:
                if last_t is not None and c[1] <= last_t:
                    c[1] = last_t + 1  # 이 자료는 오류 경우를 넣지 않는다(C-errors에 있다)
                run = d.update(c[1], c[2], c[3])
                last_t = c[1]
                out.append([run_list(run), d.ongoing_ms(), d.run_start_ms(), d.state])
                continue
            out.append([None, d.ongoing_ms(), d.run_start_ms(), d.state])
        cases.append({"cfg": cfg, "calls": calls, "out": out})
    return save("C-detector", "ClosureRunDetector", cases)


# ---------------------------------------------------------------------------------------------- C-steps
def variant_blocks(rng: random.Random, n: int) -> list[dict]:
    import gen_s
    F = gen_s.fuzz()
    out = [copy.deepcopy(hk.PIPE_BLOCK)]
    while len(out) < n:
        pb, _rd = F.variant_configs(rng)
        out.append(pb)
    return out


def c_steps(rng: random.Random, ns: dict) -> str:
    cases = []
    blocks = variant_blocks(rng, 12)
    for i in range(60):
        pb = blocks[i % len(blocks)]
        spec = SP.pipeline_spec_from_dict(pb)
        step_ms = spec.clock.step_ms
        start_ms = rng.choice([0, 0, step_ms * 3, 60000, step_ms * 7])
        frames = ns["frame_stream"](rng, rng.choice([30000, 90000, 200000]), allow_long_gaps=rng.random() < 0.5)
        origin = frames[0].t_ms
        # jaw_rel과 pitch_rel을 하품·끄덕임 임계값 둘레로 바꾼다
        fr2 = []
        for f in frames:
            if f.face and rng.random() < 0.3:
                f = RT.FrameState(t_ms=f.t_ms, face=True, closure=f.closure, closure_valid=f.closure_valid,
                                  provisional=f.provisional, yaw_rel=f.yaw_rel,
                                  pitch_rel=rng.choice([0.0, 10.0, 5.0, 12.0, -3.0, f.pitch_rel]),
                                  pose_valid=f.pose_valid, gaze_h_rel=f.gaze_h_rel, gaze_v_rel=f.gaze_v_rel,
                                  jaw_rel=rng.choice([0.4, 0.25, 0.39, 0.6, 0.0]))
            fr2.append(f)
        det = ClosureRunDetector(spec.run_detector)
        agg = StepAggregator(spec, start_ms, origin)
        calls, out = [], []
        prev = None
        for f in fr2:
            run = None
            if f.t_ms - origin >= start_ms:
                if prev is not None and f.t_ms - prev > spec.clock.reset_gap_ms:
                    det.reset()
                run = det.update(f.t_ms, f.closure, f.closure_valid)
                prev = f.t_ms
            ongoing = det.ongoing_ms()
            steps = agg.update(f, run, ongoing)
            calls.append([st_json(f), run_list(run), ongoing])
            out.append([step_json(s) for s in steps])
        cases.append({"kind": "aggregator", "pipeline": pb, "start_ms": start_ms, "origin_ms": origin, "calls": calls,
                      "out": out})
    for i in range(200):
        args = dict(enter_thr=rng.choice([0.4, 10.0, 0.3]), exit_thr=rng.choice([0.25, 5.0, 0.1]),
                    min_duration_ms=rng.choice([2000, 200, 1, 0]), max_duration_ms=rng.choice([2000, 3000, 500]),
                    count_at_end=rng.random() < 0.5, cancel_gap_ms=rng.choice([500, 200, 1000, 2500]))
        c = ThresholdEventCounter(**args)
        t = rng.choice([0, 1, 1731465123457])
        calls, out = [], []
        for _ in range(rng.choice([50, 200, 600])):
            if rng.random() < 0.02:
                c.reset()
                calls.append(["r"])
                out.append([False, c.active])
                continue
            v = rng.choice([args["enter_thr"], args["exit_thr"], args["enter_thr"] + 0.01, args["exit_thr"] - 0.01,
                            0.0, args["enter_thr"] * 2, rng.uniform(0, args["enter_thr"] * 1.5)])
            valid = rng.random() < 0.9
            calls.append(["u", t, v, valid])
            out.append([c.update(t, v, valid), c.active])
            t += rng.choice([33, 100, 200, 201, 499, 500, 501, 1000, 2500, 2501])
        cases.append({"kind": "counter", "args": args, "calls": calls, "out": out})
    for i in range(200):
        cfg = dict(enter_deg=rng.choice([10.0, 5.0, 8.0]), exit_deg=rng.choice([5.0, 2.0, 4.0]),
                   min_duration_ms=rng.choice([200, 1, 100, 500]), max_duration_ms=rng.choice([3000, 500, 1000]),
                   ema_tau_ms=rng.choice([10000, 1000, 100, 1]), ema_max_dt_ms=rng.choice([200, 50, 1000, 20000]),
                   cancel_gap_ms=rng.choice([500, 200, 1000, 2500]))
        nd = NodDetector(SP.NodSpec(**cfg))
        t = rng.choice([0, 1, 5000])
        calls, out = [], []
        zero_lead = rng.random() < 0.4
        for j in range(rng.choice([60, 300])):
            if rng.random() < 0.02:
                nd.reset()
                calls.append(["r"])
                out.append([False, nd.mean])
                continue
            if zero_lead and j < 20:
                p = 0.0
            else:
                p = rng.choice([0.0, 10.0, 5.0, cfg["enter_deg"], cfg["exit_deg"], 15.0, rng.uniform(-5, 20)])
            valid = rng.random() < 0.92
            calls.append(["u", t, p, valid])
            out.append([nd.update(t, p, valid), nd.mean])
            t += rng.choice([33, 50, 100, 200, 201, 1000])
        cases.append({"kind": "nod", "cfg": cfg, "calls": calls, "out": out})
    return save("C-steps", "StepAggregator, ThresholdEventCounter, NodDetector", cases)


# ---------------------------------------------------------------------------------------------- C-baseline
def calib_signals(rng: random.Random, spec, n_ms: int) -> list:  # noqa: ANN001
    t = rng.choice([0, 1, 1731465123457])
    out = []
    e = rng.choice([0.27, 0.12, 0.45, 0.5, 0.1, 0.3])
    b = rng.uniform(0.05, 0.3)
    yaw0 = rng.choice([0.0, 15.0, -15.0, 16.0, rng.uniform(-20, 20)])
    narrow_until = rng.choice([0, 0, 60000, 90000])
    miss = rng.choice([None, (10000, 45000), (20000, 50000), (0, 31000)])
    end = t + n_ms
    t_rel = 0
    while t < end:
        face = not (miss and miss[0] <= t_rel < miss[1]) and rng.random() > 0.02
        if face:
            w_r = 3.0 if t_rel < narrow_until else rng.choice([6.0, 40.0, 40.0])
            s = RT.FrameSignals(t_ms=t, face=True, ear_l=e + rng.uniform(-0.02, 0.02), ear_r=e + rng.uniform(-0.02, 0.02),
                                eye_w_l=40.0, eye_w_r=w_r, blink_l=b + rng.uniform(0, 0.2), blink_r=b + rng.uniform(0, 0.2),
                                eye_look=tuple(rng.uniform(0, 0.1) for _ in range(8)), jaw_open=rng.uniform(0, 0.2),
                                yaw=yaw0 + rng.choice([0.0, rng.uniform(-2, 2)]), pitch=rng.uniform(-5, 5),
                                roll=rng.uniform(-2, 2), iris_h_l=None, iris_h_r=None, iris_v_l=None, iris_v_r=None)
        else:
            s = RT.FrameSignals(t_ms=t, face=False, ear_l=0.0, ear_r=0.0, eye_w_l=0.0, eye_w_r=0.0, blink_l=0.0,
                                blink_r=0.0, eye_look=(0.0,) * 8, jaw_open=0.0, yaw=0.0, pitch=0.0, roll=0.0)
        out.append(s)
        dt = rng.choice([33, 100, 167, 200, 201, 250])
        if rng.random() < 0.005:
            dt = rng.choice([30000, 70000, 125000])  # 마감 여러 개를 건너뛰는 끊김
        t += dt
        t_rel += dt
    return out


def c_baseline(rng: random.Random) -> str:
    cases = []
    blocks = variant_blocks(rng, 8)
    for i in range(80):
        pb = blocks[i % len(blocks)]
        spec = SP.pipeline_spec_from_dict(pb)
        sigs = calib_signals(rng, spec, rng.choice([70000, 130000, 200000]))
        est = C.BaselineEstimator(spec)
        out = []
        for s in sigs:
            out.append(bl_json(est.update(s)))
        d = est.diagnostics
        cases.append({"pipeline": pb, "calls": [sig_json(s) for s in sigs], "out": out,
                      "diagnostics": None if d is None else {"n_face_frames": d.n_face_frames, "face_ratio": d.face_ratio,
                                                             "blink_l_quantiles": list(d.blink_l_quantiles),
                                                             "blink_r_quantiles": list(d.blink_r_quantiles)}})
    return save("C-baseline", "BaselineEstimator", cases)


# ---------------------------------------------------------------------------------------------- C-hysteresis
def c_hysteresis(rng: random.Random) -> str:
    cases = []
    for i in range(300):
        args = [rng.choice([1, 2, 3, 5, 10]), rng.choice([1, 2, 3, 5, 10]), rng.choice([0, 3000, 10000, 60000])]
        h = R.StepHysteresis(*args)
        k = rng.randint(1, 200)
        origin = rng.choice([0, 1, 1731465123457])
        calls, out = [], []
        for _ in range(rng.choice([50, 300])):
            if rng.random() < 0.01:
                h.reset()
                calls.append(["r"])
            else:
                hv = rng.random() < 0.9
                on = rng.random() < 0.5
                en = (not on) and rng.random() < 0.7
                calls.append(["u", k, origin + 1000 * k, hv, on, en])
                h.update(k, origin + 1000 * k, hv, on, en)
            out.append([h.active, h.start_ms])
            k += rng.choice([1, 1, 1, 1, 2, 0, 13]) if rng.random() < 0.05 else 1
        cases.append({"args": args, "calls": calls, "out": out})
    return save("C-hysteresis", "StepHysteresis", cases)


# ---------------------------------------------------------------------------------------------- C-engine, C-policy
def rules_variants(rng: random.Random, n: int) -> list[dict]:
    import gen_s
    F = gen_s.fuzz()
    out = [copy.deepcopy(hk.RULES_DOC)]
    for src in ("model", "either"):
        d = copy.deepcopy(hk.RULES_DOC)
        d["drowsiness"]["source"] = src
        out.append(d)
    d = copy.deepcopy(hk.RULES_DOC)
    d["events"]["sleep"]["repeat_ms"] = 5000
    d["closure"]["sleep_ms"] = 3000
    out.append(d)
    while len(out) < n:
        _pb, rd = F.variant_configs(rng)
        out.append(rd)
    return out


def c_engine(rng: random.Random, ns: dict) -> str:
    cases = []
    variants = rules_variants(rng, 16)
    for i in range(120):
        rd = variants[i % len(variants)]
        cfg = SP.rules_from_dict(rd)
        dur = rng.choice([40000, 90000, 150000])
        if rng.random() < 0.4:
            frames = ns["knife_stream"](rng, dur)
        else:
            frames = ns["frame_stream"](rng, dur, provisional_ms=rng.choice([0, 0, 20000]))
        steps_by_frame, origin = ns["make_steps"](rng, frames, "pipeline", cfg.clock.reset_gap_ms)
        eng = R.RuleEngine(cfg)
        calls, out = [], []
        for f, sts in zip(frames, steps_by_frame):
            for (k, _t_end, p, perclos, full, missing) in sts:
                s = ns["step"](k, perclos, full=full, missing=missing, origin=origin)
                pr = None if p is None else ns["probs"](p)
                calls.append(["s", step_json(s), None if pr is None else list(pr)])
                out.append([[ev_json(e) for e in eng.on_step(s, pr)], None])
            calls.append(["f", st_json(f)])
            out.append([[ev_json(e) for e in eng.update(f)], eng.gaze_zone])
        cases.append({"rules": rd, "calls": calls, "out": out})
    return save("C-engine", "RuleEngine", cases)


TYPICAL_SOURCE = {"gaze_away_long": "head_pose", "vats": "head_pose", "drowsy": "perclos", "unable_to_analyze": "face",
                  "microsleep": "closure", "sleep": "closure", "calibrating": "face"}


def c_policy(rng: random.Random, ns: dict) -> str:
    cases = []
    variants = rules_variants(rng, 12)
    ov = [{}, {"events.sleep.repeat_ms": 5000}, {"events.sleep.repeat_ms": 0, "events.sleep.cooldown_ms": 4000},
          {"events.microsleep.repeat_ms": 700}, {"events.drowsy.cooldown_ms": 20000, "events.vats.cooldown_ms": 5000}]
    for i in range(160):
        rd = variants[i % len(variants)]
        if i % 3 == 0:
            for k, v in ov[i % len(ov)].items():
                hk_set(rd := copy.deepcopy(rd), k, v)
        cfg = SP.rules_from_dict(rd)
        if i % 2 == 0:
            raw = ns["synthetic_calls"](rng, cfg)
            seq = []
            for (t, events, speed) in raw:
                evs = [RT.DriverStateEvent(type=e.type, severity=e.severity, t_start_ms=e.t_start_ms,
                                           duration_ms=e.duration_ms, confidence=e.confidence,
                                           source=TYPICAL_SOURCE[e.type], phase=e.phase, provisional=e.provisional)
                       for e in events]
                if rng.random() < 0.01:  # end 없이 다시 온 onset, 음수 속도
                    evs.append(RT.DriverStateEvent(type="vats", severity=2, t_start_ms=t - 100, duration_ms=100,
                                                   confidence=1.0, source="head_pose", phase="onset", provisional=False))
                if rng.random() < 0.01:
                    speed = -5.0
                seq.append((t, evs, speed))
            origin = "synthetic (vendored r02 synthetic_calls, sources mapped to the typical source of each type)"
        else:
            frames = ns["knife_stream"](rng, 120000) if rng.random() < 0.5 else ns["frame_stream"](rng, 200000)
            steps_by_frame, org = ns["make_steps"](rng, frames, "pipeline", cfg.clock.reset_gap_ms)
            speeds = ns["speed_stream"](rng, len(frames))
            eng = R.RuleEngine(cfg)
            seq = []
            for j, f in enumerate(frames):
                evs = []
                for (k, _te, p, perclos, full, missing) in steps_by_frame[j]:
                    evs.extend(eng.on_step(ns["step"](k, perclos, full=full, missing=missing, origin=org),
                                           None if p is None else ns["probs"](p)))
                evs.extend(eng.update(f))
                sp = speeds[j]
                if sp is not None and rng.random() < 0.002:
                    sp = -5.0
                seq.append((f.t_ms, evs, sp))
            origin = "engine (RuleEngine on vendored knife_stream / frame_stream)"
        pol = R.AlertPolicy(cfg)
        calls, out = [], []
        for (t, evs, sp) in seq:
            calls.append([[ev_json(e) for e in evs], sp, t])
            out.append([ev_json(a) for a in pol.decide(evs, sp, t)])
        cases.append({"rules": rd, "source": origin, "calls": calls, "out": out})
    return save("C-policy", "AlertPolicy", cases)


def hk_set(doc: dict, dotted: str, value) -> None:  # noqa: ANN001
    node = doc
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value


# ---------------------------------------------------------------------------------------------- C-errors
FNS = {"nearest_rank_quantile": C.nearest_rank_quantile, "median": C.median, "eye_aspect_ratio": SG.eye_aspect_ratio,
       "iris_ratio": SG.iris_ratio, "gaze_h": SG.gaze_h, "gaze_v": SG.gaze_v,
       "euler_from_colmajor": SG.euler_from_colmajor, "euler_from_rotation": SG.euler_from_rotation,
       "rotation_from_euler": SG.rotation_from_euler, "rotation_from_colmajor": SG.rotation_from_colmajor,
       "relative_rotation": SG.relative_rotation, "relative_euler": SG.relative_euler, "ratio": R._ratio}
ERR_FNS = FNS


def c_errors(rng: random.Random) -> list[str]:
    spec = SP.pipeline_spec_from_dict(hk.PIPE_BLOCK)
    rules = SP.rules_from_dict(hk.RULES_DOC)
    cases = []

    def attempt(f):  # noqa: ANN001, ANN202
        try:
            return {"ok": plain(f())}
        except Exception as exc:  # noqa: BLE001
            return {"error": err(exc)}
    # DriverPipeline 생성 오류 셋
    rd2 = copy.deepcopy(hk.RULES_DOC)
    rd2["clock"]["max_hold_ms"] = 250
    pb2 = copy.deepcopy(hk.PIPE_BLOCK)
    pb2["clock"]["max_hold_ms"] = 250
    b = hk.baseline_obj({"e_l": 0.27, "e_r": 0.27, "b_l": 0.22, "b_r": 0.22, "r0": list(RT.IDENTITY_R0), "jaw0": 0.08,
                         "gaze_h0": 0.0, "gaze_v0": 0.0, "t_complete_ms": 0, "status": "ok"})
    for name, kw, rr in (("rules.clock != spec.clock", {}, rd2), ("feature_start_ms without baseline",
                                                                   {"feature_start_ms": 0}, hk.RULES_DOC),
                         ("feature_start_ms not a multiple of step_ms", {"baseline": b, "feature_start_ms": 1500},
                          hk.RULES_DOC),
                         ("feature_start_ms negative", {"baseline": b, "feature_start_ms": -1000}, hk.RULES_DOC),
                         ("baseline t_complete_ms not a multiple", {"baseline": hk.baseline_obj(
                             {**hk.baseline_plain(b), "t_complete_ms": 1500})}, hk.RULES_DOC)):
        r = attempt(lambda: hk.DriverPipeline(spec, SP.rules_from_dict(rr), **kw) and "constructed")
        cases.append({"case": name, "component": "DriverPipeline", "setup": {"pipeline": hk.PIPE_BLOCK, "rules": rr,
                                                                             "kwargs": {
            k: (hk.baseline_plain(v) if isinstance(v, RT.Baseline) else v) for k, v in kw.items()}}, "calls": [],
            "out": r})
    # 3개가 아닌 값을 돌려주는 모델: 창이 차는 첫 스텝에서. 입력은 HKDIFF1 파일 하나에 쓴다(FORMATS 2.7절):
    # 배포 상수, 넣어 준 기준선, 3개를 돌려주는 대체 모델로 첫 모델 호출이 있는 프레임까지 참조를 돌린 세션이다.
    import gen_s
    S = gen_s.make("errors", "inj_model", 9001)
    base = S.baseline if S.baseline else hk.baseline_plain(b)
    full = hk.Session(corpus="C-errors", sid="C-errors/model_inputs/9001", klass="model_inputs", seed=9001,
                      pb=copy.deepcopy(hk.PIPE_BLOCK), rd=copy.deepcopy(hk.RULES_DOC), kind="raw", frames=S.frames,
                      speeds=S.speeds, baseline=base, feature_start_ms=None, stub=[(0.25, 0.5, 0.25)],
                      generator="frames of tx_fuzz.gen_session('inj_model', 9001) + frames_of (vendored r5bv_spec_tx), "
                                "shipped constants, cut after the frame of the first model call",
                      notes={"source_session": S.sid, "baseline_from_source_session": S.baseline is not None})
    outs = {}
    for nret in (2, 4, 0):
        r = hk.run_session(full, "RAW", margins=False, model_override=lambda window, n=nret: [0.1] * n)
        e = r.meta["exception"]
        if e is None or e["type"] != "ValueError":
            raise AssertionError(f"C-errors: model returning {nret} values did not raise ValueError: {e}")
        outs[nret] = (e, r.meta["counters"], r.c["t_ms"][e["fed_index"]])
    fed = {o[0]["fed_index"] for o in outs.values()}
    if len(fed) != 1:
        raise AssertionError(f"C-errors: model cases raise at different frames {fed}")
    n_keep = fed.pop() + 1
    inp = hk.Session(**{**full.__dict__, "frames": S.frames[:n_keep], "speeds": S.speeds[:n_keep],
                        "notes": {**full.notes, "n_kept_frames": n_keep}})
    hkd = os.path.join(CD, "C-errors_model_inputs.hkd")
    summ = hk.export_session(hkd, inp, gfiles()["files"])
    h2, back = hk.read_hk(hkd)
    acc_fed = [i for i in range(len(back["accepted"])) if back["accepted"][i] != 0]
    if (len(back["mw_frame"]) != 1 or acc_fed[int(back["mw_frame"][0])] != n_keep - 1
            or h2["counters"]["model_calls"] != 1):
        raise AssertionError("C-errors: the input file does not end at the frame of the first model call")
    print(f"  C-errors inputs: {os.path.relpath(hkd, hk.DATA)}, {summ['n_fed']} frames, modes {h2['modes']}, "
          f"{summ['selfcheck']}", flush=True)
    for nret in (2, 4, 0):
        e, cnt, t_ms = outs[nret]
        cases.append({"case": f"model returns {nret} values", "component": "DriverPipeline",
                      "setup": {"inputs": os.path.relpath(hkd, hk.DATA), "model_returns": [0.1] * nret,
                                "note": "construct DriverPipeline from the input file (pipeline_json, rules_json, "
                                        "baseline_injected, feature_start_ms) with a model that returns "
                                        "model_returns instead of the file's stub, feed the file's frames in any mode "
                                        "of its `modes`; the first model call raises at the last fed frame"},
                      "calls": [], "out": {"fed_index": e["fed_index"], "t_ms": t_ms,
                                           "error": {"type": e["type"], "message": e["message"]}, "where": e["where"]},
                      "counters_at_error": {k: cnt[k] for k in ("frames_accepted", "steps_emitted", "model_calls")}})
    # 길이 검사: 함수 이름과 인자를 기록한다(다시 돌릴 수 있게)
    fcases = (("euler_from_colmajor 15 values", "euler_from_colmajor", [[0.0] * 15]),
              ("rotation_from_colmajor 17 values", "rotation_from_colmajor", [[0.0] * 17]),
              ("euler_from_rotation 8 values", "euler_from_rotation", [[0.0] * 8]),
              ("relative_rotation 10 values", "relative_rotation", [[0.0] * 10, list(RT.IDENTITY_R0)]),
              ("relative_rotation r0 8 values", "relative_rotation", [list(RT.IDENTITY_R0), [0.0] * 8]),
              ("eye_aspect_ratio 5 points", "eye_aspect_ratio", [[[0.0, 0.0]] * 5]),
              ("eye_aspect_ratio 7 points", "eye_aspect_ratio", [[[0.0, 0.0]] * 7]),
              ("nearest_rank_quantile empty", "nearest_rank_quantile", [[], 0.5]),
              ("median empty", "median", [[]]))
    for name, fn, args in fcases:
        cases.append({"case": name, "component": "function", "setup": {"fn": fn, "args": args}, "calls": [],
                      "out": attempt(lambda: ERR_FNS[fn](*copy.deepcopy(args)))})
    # 증가하지 않는 t: 거부 뒤에도 흐름을 이어 간다
    for comp in ("RuleEngine", "ClosureRunDetector", "BaselineEstimator", "StepAggregator"):
        for rep in range(6):
            calls, out = [], []
            t = rng.choice([1, 1000, 1731465123457])
            if comp == "RuleEngine":
                obj = R.RuleEngine(rules)
            elif comp == "ClosureRunDetector":
                obj = ClosureRunDetector(spec.run_detector)
            elif comp == "BaselineEstimator":
                obj = C.BaselineEstimator(spec)
            else:
                obj = StepAggregator(spec, 0, t)
                det = ClosureRunDetector(spec.run_detector)
            prev = None
            for j in range(rng.choice([400, 900, 2000])):
                bad = prev is not None and rng.random() < 0.02
                tt = rng.choice([prev, prev - 1, prev - 500]) if bad else t
                c_val = rng.choice([0.0, 0.2, 0.6, 0.9, 1.0])
                face = rng.random() < 0.95
                if comp == "BaselineEstimator":
                    arg = RT.FrameSignals(t_ms=tt, face=face, ear_l=0.27, ear_r=0.27 + 0.001 * (j % 7), eye_w_l=40.0,
                                          eye_w_r=40.0, blink_l=0.2, blink_r=0.21, eye_look=(0.0,) * 8, jaw_open=0.05,
                                          yaw=1.0, pitch=0.5, roll=0.0) if face else RT.FrameSignals(
                        t_ms=tt, face=False, ear_l=0.0, ear_r=0.0, eye_w_l=0.0, eye_w_r=0.0, blink_l=0.0, blink_r=0.0,
                        eye_look=(0.0,) * 8, jaw_open=0.0, yaw=0.0, pitch=0.0, roll=0.0)
                    calls.append(sig_json(arg))
                    r = attempt(lambda: bl_json(obj.update(arg)))
                else:
                    stt = RT.FrameState(t_ms=tt, face=face, closure=c_val if face else 0.0, closure_valid=face,
                                        provisional=False, yaw_rel=0.0, pitch_rel=0.0, pose_valid=face,
                                        gaze_h_rel=0.0, gaze_v_rel=0.0, jaw_rel=0.0)
                    if comp == "RuleEngine":
                        calls.append(st_json(stt))
                        r = attempt(lambda: [ev_json(e) for e in obj.update(stt)])
                    elif comp == "ClosureRunDetector":
                        calls.append([tt, stt.closure, stt.closure_valid])
                        r = attempt(lambda: [run_list(obj.update(tt, stt.closure, stt.closure_valid)), obj.ongoing_ms(),
                                             obj.run_start_ms(), obj.state])
                    else:
                        # 누적기에는 검출기의 출력을 함께 넣는다(검출기는 거부하지 않은 프레임만 받는다)
                        run, ong = None, det.ongoing_ms()
                        if not bad:
                            run = det.update(tt, stt.closure, stt.closure_valid)
                            ong = det.ongoing_ms()
                        calls.append([st_json(stt), run_list(run), ong])
                        r = attempt(lambda: [step_json(s) for s in obj.update(stt, run, ong)])
                out.append(r)
                if not bad:
                    prev = t
                    t += rng.choice([33, 100, 200, 1000, 15000]) if rng.random() < 0.01 else rng.choice([33, 34, 100])
            cases.append({"case": f"{comp}: non-increasing t refused, stream continues ({rep})", "component": comp,
                          "setup": {"pipeline": hk.PIPE_BLOCK if comp != "RuleEngine" else None,
                                    "rules": hk.RULES_DOC if comp == "RuleEngine" else None,
                                    "start_ms": 0 if comp == "StepAggregator" else None,
                                    "origin_ms": calls[0][0]["t_ms"] if comp == "StepAggregator" else None},
                          "calls": calls, "out": out})
    return [save("C-errors", "constructors and argument checks", cases), hkd]


# ---------------------------------------------------------------------------------------------- C-loader
def leaves(doc, path=()):  # noqa: ANN001, ANN201
    if isinstance(doc, dict):
        for k, v in doc.items():
            yield from leaves(v, path + (k,))
    elif isinstance(doc, list):
        yield path, doc
        for i, v in enumerate(doc):
            if not isinstance(v, (dict, list)):
                continue
            yield from leaves(v, path + (i,))
    else:
        yield path, doc


def setp(doc, path, value):  # noqa: ANN001
    node = doc
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value


def delp(doc, path):  # noqa: ANN001
    node = doc
    for p in path[:-1]:
        node = node[p]
    del node[path[-1]]


def mutations(v):  # noqa: ANN001, ANN201
    if isinstance(v, bool):
        return [("not", not v), ("int 1", 1), ("string", "true"), ("null", None)]
    if isinstance(v, int):
        return [("0", 0), ("-1", -1), ("+1", v + 1), ("x10", v * 10), ("as float", float(v)), ("2^31", 2 ** 31),
                ("2^63", 2 ** 63), ("-2^31-1", -2 ** 31 - 1), ("string", str(v)), ("bool", True), ("null", None)]
    if isinstance(v, float):
        return [("0.0", 0.0), ("-1.0", -1.0), ("x2", v * 2), ("+0.1", v + 0.1), ("as int", int(v)), ("-0.0", -0.0),
                ("1e300", 1e300), ("string", repr(v)), ("bool", False), ("null", None)]
    if isinstance(v, str):
        return [("empty", ""), ("other", v + "_x"), ("upper", v.upper()), ("int", 1), ("null", None)]
    if isinstance(v, list):
        out = [("empty list", []), ("not a list", 0)]
        if v:
            out += [("one more", v + [v[-1]]), ("one less", v[:-1]), ("reversed", list(reversed(v)))]
        return out
    return []


LOADER_FLAGS = {
    "int_outside_int32": "the changed leaf is an integer outside [-2^31, 2^31 - 1]; Python has no integer bound, a "
                         "Kotlin Int field (plan R22) refuses it: the owner decides before E7",
    "int_outside_int64": "the changed leaf is an integer outside [-2^63, 2^63 - 1]; the Kotlin MiniJson reader refuses "
                         "it (plan 5.10): the owner decides before E7",
    "accepted_changed_string": "Python accepted a changed string (empty, '<v>_x' or upper case); a Kotlin enum with a "
                               "`wire` string (K21) would refuse it: the owner decides before E7",
}


def judge(target: str, doc) -> tuple[bool, dict | None]:  # noqa: ANN001
    fn = {"pipeline": SP.pipeline_spec_from_dict, "rules": SP.rules_from_dict,
          "feature_spec": SP.feature_spec_from_dict}[target]
    try:
        fn(doc)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, err(exc)


def c_loader(rng: random.Random) -> str:
    cases = []
    with open(os.path.join(hk.REPO, "export", "feature_spec_v0.json"), encoding="utf-8") as fh:
        fs_doc = json.load(fh)

    def flags_of(path, change, nv, ok):  # noqa: ANN001, ANN202
        """소유자가 E7 전에 정할 문서의 표지(판정은 Python 그대로다. 표지는 판정을 바꾸지 않는다)."""
        out = []
        if isinstance(nv, int) and not isinstance(nv, bool):
            if not -2 ** 31 <= nv < 2 ** 31:
                out.append("int_outside_int32")
            if not -2 ** 63 <= nv < 2 ** 63:
                out.append("int_outside_int64")
        if ok and isinstance(nv, str) and change in ("empty", "other", "upper"):
            out.append("accepted_changed_string")
        return out

    def add(target, path, change, doc, nv=None):  # noqa: ANN001
        text = json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2)
        ok, e = judge(target, json.loads(text))
        cases.append({"target": target, "path": ".".join(map(str, path)), "change": change, "text": text,
                      "accepted": ok, "error": e, "flags": flags_of(path, change, nv, ok)})
    for target, base in (("pipeline", hk.PIPE_BLOCK), ("rules", hk.RULES_DOC)):
        add(target, (), "unchanged", base)
        for path, v in list(leaves(base)):
            if not path:
                continue
            for change, nv in mutations(v):
                d = copy.deepcopy(base)
                setp(d, path, nv)
                add(target, path, change, d, nv)
            if isinstance(path[-1], str):
                d = copy.deepcopy(base)
                delp(d, path)
                add(target, path, "key removed", d)
        d = copy.deepcopy(base)
        d["extra_key"] = 1
        add(target, ("extra_key",), "unknown top-level key", d)
    add("feature_spec", (), "unchanged", fs_doc)
    for path in (("pipeline_hash",), ("spec_version",), ("step_sec",), ("window_steps",), ("class_order",),
                 ("standardization", "location"), ("standardization", "z_clip"), ("features", 0, "name"),
                 ("features", 3, "transform"), ("features", 5, "std"), ("pipeline", "clock", "step_ms")):
        node = fs_doc
        for p in path:
            node = node[p]
        for change, nv in mutations(node)[:5]:
            d = copy.deepcopy(fs_doc)
            setp(d, path, nv)
            add("feature_spec", path, change, d, nv)
    n_acc = sum(1 for c in cases if c["accepted"])
    print(f"  C-loader: {len(cases)} documents, accepted {n_acc}, refused {len(cases) - n_acc}")
    fl = {}
    for c in cases:
        for f in c["flags"]:
            k = f"{f} ({'accepted' if c['accepted'] else 'refused'})"
            fl[k] = fl.get(k, 0) + 1
    print(f"  C-loader flags: {json.dumps(fl, sort_keys=True)}")
    return save("C-loader", "spec.pipeline_spec_from_dict, rules_from_dict, feature_spec_from_dict", cases,
                {"depends_on": hk.depends_on("drowsy/spec.py"), "flags_meaning": LOADER_FLAGS})


def all_components() -> list[str]:
    os.makedirs(CD, exist_ok=True)
    ns = review_ns()
    out = []
    for name, f in (("C-func", lambda r: c_func(r)), ("C-detector", lambda r: c_detector(r)),
                    ("C-steps", lambda r: c_steps(r, ns)), ("C-baseline", lambda r: c_baseline(r)),
                    ("C-hysteresis", lambda r: c_hysteresis(r)), ("C-engine", lambda r: c_engine(r, ns)),
                    ("C-policy", lambda r: c_policy(r, ns)), ("C-errors", lambda r: c_errors(r)),
                    ("C-loader", lambda r: c_loader(r))):
        rng = random.Random(f"C:{name}:20261115")
        res = f(rng)
        out += res if isinstance(res, list) else [res]
    rerun_check([p for p in out if p.endswith(".json")], ns)
    return out


def rerun_check(paths: list[str], ns: dict) -> None:
    """자기 검사: 기록 파일을 다시 읽어 그 입력으로 Python 객체를 다시 돌리면 기록된 출력과 같다(검출기, 누적기,
    계수기, 끄덕임, 기준선, 판정기, 엔진, 경고, 함수)."""
    def lj(p):  # noqa: ANN001, ANN202
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def fs(d):  # noqa: ANN001, ANN202
        return RT.FrameState(**d)

    def stp(d):  # noqa: ANN001, ANN202
        return RT.StepFeatures(k=d["k"], t_end_ms=d["t_end_ms"], full_lookback=d["full_lookback"], values=tuple(d["values"]))

    def evo(d):  # noqa: ANN001, ANN202
        return RT.DriverStateEvent(**d)

    def runo(r):  # noqa: ANN001, ANN202
        return None if r is None else RT.ClosureRun(*r)
    bad = 0
    n = 0
    D = {os.path.basename(p)[:-5]: lj(p) for p in paths}
    for c in D["C-detector"]["cases"]:
        d = ClosureRunDetector(SP.RunDetectorSpec(**c["cfg"]))
        for call, o in zip(c["calls"], c["out"]):
            if call[0] == "r":
                d.reset()
                got = [None, d.ongoing_ms(), d.run_start_ms(), d.state]
            else:
                got = [run_list(d.update(call[1], call[2], call[3])), d.ongoing_ms(), d.run_start_ms(), d.state]
            n += 1
            bad += plain(got) != o
    for c in D["C-steps"]["cases"]:
        if c["kind"] == "aggregator":
            a = StepAggregator(SP.pipeline_spec_from_dict(c["pipeline"]), c["start_ms"], c["origin_ms"])
            for call, o in zip(c["calls"], c["out"]):
                got = [step_json(s) for s in a.update(fs(call[0]), runo(call[1]), call[2])]
                n += 1
                bad += plain(got) != o
        elif c["kind"] == "counter":
            k = ThresholdEventCounter(**c["args"])
            for call, o in zip(c["calls"], c["out"]):
                if call[0] == "r":
                    k.reset()
                    got = [False, k.active]
                else:
                    got = [k.update(call[1], call[2], call[3]), k.active]
                n += 1
                bad += got != o
        else:
            nd = NodDetector(SP.NodSpec(**c["cfg"]))
            for call, o in zip(c["calls"], c["out"]):
                if call[0] == "r":
                    nd.reset()
                    got = [False, nd.mean]
                else:
                    got = [nd.update(call[1], call[2], call[3]), nd.mean]
                n += 1
                bad += plain(got) != o
    for c in D["C-baseline"]["cases"]:
        est = C.BaselineEstimator(SP.pipeline_spec_from_dict(c["pipeline"]))
        for call, o in zip(c["calls"], c["out"]):
            s = RT.FrameSignals(**{**call, "eye_look": tuple(call["eye_look"])})
            n += 1
            bad += plain(bl_json(est.update(s))) != o
    for c in D["C-hysteresis"]["cases"]:
        h = R.StepHysteresis(*c["args"])
        for call, o in zip(c["calls"], c["out"]):
            if call[0] == "r":
                h.reset()
            else:
                h.update(*call[1:])
            n += 1
            bad += [h.active, h.start_ms] != o
    for c in D["C-engine"]["cases"]:
        eng = R.RuleEngine(SP.rules_from_dict(c["rules"]))
        for call, o in zip(c["calls"], c["out"]):
            if call[0] == "s":
                got = [[ev_json(e) for e in eng.on_step(stp(call[1]), None if call[2] is None else tuple(call[2]))], None]
            else:
                got = [[ev_json(e) for e in eng.update(fs(call[1]))], eng.gaze_zone]
            n += 1
            bad += plain(got) != o
    for c in D["C-policy"]["cases"]:
        pol = R.AlertPolicy(SP.rules_from_dict(c["rules"]))
        for call, o in zip(c["calls"], c["out"]):
            got = [ev_json(a) for a in pol.decide([evo(e) for e in call[0]], call[1], call[2])]
            n += 1
            bad += plain(got) != o
    fns = FNS
    for c in D["C-func"]["cases"]:
        args = c["args"]
        try:
            if c["fn"] == "eye_closure":
                cfg = args[4]
                res = C.eye_closure(args[0], args[1], args[2], args[3], SP.ClosureSpec(
                    **cfg, far_eye_zero_yaw_deg=25.0, far_eye_ramp_deg=10.0, valid_max_abs_yaw_deg=40.0))
            else:
                res = fns[c["fn"]](*args)
            got = {"result": plain(res)}
        except Exception as exc:  # noqa: BLE001
            got = {"error": err(exc)}
        n += 1
        want = {"result": c["result"]} if "result" in c else {"error": c["error"]}
        bad += got != want
    print(f"  components re-run from the re-read files: {n} calls, {bad} differ", flush=True)
    assert bad == 0
    rerun_errors(D["C-errors"])
    rerun_loader(D["C-loader"])


def _attempt(f) -> dict:  # noqa: ANN001
    try:
        return {"ok": plain(f())}
    except Exception as exc:  # noqa: BLE001
        return {"error": err(exc)}


def rerun_errors(doc: dict) -> None:
    """C-errors의 모든 경우를 다시 읽은 setup과 calls만으로 다시 돌린다(생성기 없이).

    모델 경우는 setup.inputs의 HKDIFF1 파일을 그 파일의 모든 모드(RAW, SIG, STATE)로 다시 넣는다.
    """
    n = bad = n_model = 0
    per = {}
    for c in doc["cases"]:
        comp, st = c["component"], c["setup"]
        nb = 0
        if comp == "DriverPipeline" and "inputs" in st:
            h, back = hk.read_hk(os.path.join(hk.DATA, st["inputs"]))
            if h["reference"]["reference_id"] != hk.reference()["reference_id"]:
                raise AssertionError(f"{st['inputs']}: reference_id differs")
            S = hk.Session(corpus=h["corpus"], sid=h["session_id"], klass=h["klass"], seed=h["seed"],
                           pb=json.loads(h["pipeline_json"]), rd=json.loads(h["rules_json"]), kind="state", frames=[],
                           speeds=[None if back["speed_null"][i] else float(back["speed"][i])
                                   for i in range(len(back["t_ms"]))],
                           baseline=hk.unhex_baseline(h["baseline_injected"]), feature_start_ms=h["feature_start_ms"])
            ret = list(st["model_returns"])
            for m in h["modes"]:
                if m == "RAW":
                    S.frames = hk.frames_from_raw(back)
                elif m == "SIG":
                    S.frames = hk.signals_from_cols(back)
                else:
                    S.frames = hk.states_from_cols(back)
                    S.state_baseline = hk.baseline_from_cols(h, back)
                r = hk.run_session(S, m, margins=False, model_override=lambda window: list(ret))
                e = r.meta["exception"]
                got = None if e is None else {"fed_index": e["fed_index"], "t_ms": int(back["t_ms"][e["fed_index"]]),
                                              "error": {"type": e["type"], "message": e["message"]},
                                              "where": e["where"]}
                cnt = {k: r.meta["counters"][k] for k in ("frames_accepted", "steps_emitted", "model_calls")}
                nb += (got != c["out"]) + (cnt != c["counters_at_error"])
                n_model += 1
                n += 1
        elif comp == "DriverPipeline":
            kw = {k: (hk.baseline_obj(v) if k == "baseline" else v) for k, v in st["kwargs"].items()}
            got = _attempt(lambda: hk.DriverPipeline(SP.pipeline_spec_from_dict(st["pipeline"]),
                                                     SP.rules_from_dict(st["rules"]), **kw) and "constructed")
            nb += got != c["out"]
            n += 1
        elif comp == "function":
            got = _attempt(lambda: FNS[st["fn"]](*copy.deepcopy(st["args"])))
            nb += got != c["out"]
            n += 1
        else:
            spec = SP.pipeline_spec_from_dict(st["pipeline"]) if st.get("pipeline") is not None else None
            if comp == "RuleEngine":
                obj = R.RuleEngine(SP.rules_from_dict(st["rules"]))
            elif comp == "ClosureRunDetector":
                obj = ClosureRunDetector(spec.run_detector)
            elif comp == "BaselineEstimator":
                obj = C.BaselineEstimator(spec)
            elif comp == "StepAggregator":
                obj = StepAggregator(spec, st["start_ms"], st["origin_ms"])
            else:
                raise AssertionError(f"C-errors: unknown component {comp}")
            for call, o in zip(c["calls"], c["out"]):
                if comp == "RuleEngine":
                    got = _attempt(lambda: [ev_json(e) for e in obj.update(RT.FrameState(**call))])
                elif comp == "ClosureRunDetector":
                    got = _attempt(lambda: [run_list(obj.update(call[0], call[1], call[2])), obj.ongoing_ms(),
                                            obj.run_start_ms(), obj.state])
                elif comp == "BaselineEstimator":
                    got = _attempt(lambda: bl_json(obj.update(RT.FrameSignals(
                        **{**call, "eye_look": tuple(call["eye_look"])}))))
                else:
                    run = None if call[1] is None else RT.ClosureRun(*call[1])
                    got = _attempt(lambda: [step_json(s) for s in obj.update(RT.FrameState(**call[0]), run, call[2])])
                nb += got != o
                n += 1
            if len(c["calls"]) != len(c["out"]):
                nb += 1
        bad += nb
        per[comp] = per.get(comp, 0) + 1
    print(f"  C-errors re-run from the re-read file: {len(doc['cases'])} cases {json.dumps(per)}, {n} calls "
          f"({n_model} of them model-case replays of the input file), {bad} differ", flush=True)
    assert bad == 0


def rerun_loader(doc: dict) -> None:
    """C-loader: 문서마다 기록된 `text`를 다시 읽어 Python 읽기 함수로 다시 판정한다. depends_on도 본다."""
    if hk.check_depends_on(doc, "C-loader.json"):
        raise AssertionError("C-loader.json: depends_on differs from the repository")
    bad = 0
    per = {}
    for c in doc["cases"]:
        ok, e = judge(c["target"], json.loads(c["text"]))
        bad += (ok != c["accepted"]) or (e != c["error"])
        per[c["target"]] = per.get(c["target"], 0) + 1
    n_acc = sum(1 for c in doc["cases"] if c["accepted"])
    print(f"  C-loader re-judged from the re-read texts: {len(doc['cases'])} documents {json.dumps(per)}, accepted "
          f"{n_acc}, {bad} differ", flush=True)
    assert bad == 0
