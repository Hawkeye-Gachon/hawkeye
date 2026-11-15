"""S-edge (STATE만, FORMATS.md 1.8절): 임계값에 정확히 놓인 `FrameState` 흐름을 참조 `DriverPipeline`에 STATE 입구로 넣는다.

RAW로 만든 세션은 `yaw_rel`, `pitch_rel`이 삼각함수를 거치므로 23.0, -25.0 같은 값을 정확히 낼 수 없다. 그래서 상태를
직접 만든다.
가정: 만든 상태는 참조 `frame_state`가 낼 수 있는 모양이어야 STATE 재생의 기대값으로 뜻이 있다. 그래서 그 모양으로
  맞춘다(얼굴 없으면 0과 거짓, 유효 한계 밖이면 감김 무효, 보정 중이면 provisional, 자세는 기준선이 있을 때만 유효).
  값 자체(정확히 23.0인 yaw_rel 등)는 RAW 입력에서 나오기 어려운 것이다.
"""
from __future__ import annotations

import copy
import os
import random

import hk
from components import review_ns

RT = hk.RT
from drowsy.spec import pipeline_spec_from_dict  # noqa: E402

_NS = None


def ns() -> dict:
    global _NS
    if _NS is None:
        _NS = review_ns()
    return _NS


def gen_files() -> dict:
    V = os.path.join(hk.HERE, "vendor")
    out = {os.path.relpath(p, hk.HERE): hk.sha256_file(p) for p in (
        os.path.join(hk.HERE, "gen_edge.py"), os.path.join(hk.HERE, "components.py"), os.path.join(hk.HERE, "hk.py"))}
    for rel in ("verify_ported/rules/r01_fuzz_engine.py", "review_rules/gen.py", "review_rules/common.py",
                "r5bv_spec_tx/tx_fuzz.py"):
        out["vendor/" + rel] = hk.sha256_file(os.path.join(V, rel))
    return out


def exact_scenes(rng: random.Random, t: int, dt: int) -> tuple[list, int]:
    """정확한 임계값 장면 하나. (t_ms, 바꿀 필드 dict) 목록과 다음 시각."""
    out = []
    kind = rng.choice(["closure_exact", "half", "yaw", "pitch", "gaze", "jaw", "nod", "perclos"])
    if kind == "closure_exact":
        for _ in range(rng.choice([5, 20, 40])):
            out.append((t, {"closure": rng.choice([0.3, 0.5, 0.7, 0.8, 0.0, 1.0])}))
            t += dt
    elif kind == "half":  # 0과 1(또는 0.2와 0.8)을 홀수 간격으로: 보간 오프셋 x.5
        lo, hi = rng.choice([(0.0, 1.0), (0.2, 0.8)])
        for _ in range(rng.choice([2, 4])):
            for c, n in ((lo, 3), (hi, rng.choice([10, 40, 120])), (lo, 3)):
                for _ in range(n):
                    out.append((t, {"closure": c}))
                    t += rng.choice([33, 67, 101, 99])
    elif kind == "yaw":
        v = rng.choice([23.0, -23.0, 40.0, -40.0])
        for _ in range(rng.choice([10, 60, 120])):
            out.append((t, {"yaw_rel": v}))
            t += dt
    elif kind == "pitch":
        v = rng.choice([-25.0, 18.0])
        for _ in range(rng.choice([10, 60, 120])):
            out.append((t, {"pitch_rel": v}))
            t += dt
    elif kind == "gaze":
        f, v = rng.choice([("gaze_h_rel", 0.25), ("gaze_h_rel", -0.25), ("gaze_v_rel", 0.35), ("gaze_v_rel", -0.35)])
        for _ in range(rng.choice([10, 60, 120])):
            out.append((t, {f: v}))
            t += dt
    elif kind == "jaw":
        seq = [0.0] * 3 + [0.4] * rng.choice([10, 30, 70]) + [0.25] * rng.choice([3, 10]) + [0.0] * 3
        for v in seq:
            out.append((t, {"jaw_rel": v}))
            t += dt
    elif kind == "nod":  # 앞의 pitch_rel이 모두 0.0이면 느린 평균이 0.0으로 남는다. 그 뒤 정확히 10.0과 5.0
        for v in [0.0] * 30 + [10.0] * rng.choice([3, 8, 20]) + [5.0] * rng.choice([1, 3]) + [0.0] * 5:
            out.append((t, {"pitch_rel": v}))
            t += dt
    else:  # PERCLOS: 정확히 0.8과 그 밖
        for _ in range(rng.choice([100, 300])):
            out.append((t, {"closure": rng.choice([0.8, 0.8, 0.0, 0.79])}))
            t += dt
    return out, t


def make(seed: int) -> hk.Session:
    rng = random.Random(f"S-edge:{seed}")
    N = ns()
    import gen_s
    F = gen_s.fuzz()
    pb, rd = copy.deepcopy(hk.PIPE_BLOCK), copy.deepcopy(hk.RULES_DOC)
    if rng.random() < 0.4:
        pb, rd = F.variant_configs(rng)
    spec = pipeline_spec_from_dict(pb)
    selfcal = rng.random() < 0.3
    dur = rng.choice([60000, 120000, 200000, 300000])
    if selfcal:
        dur += pb["baseline"]["duration_ms"]
    t0 = rng.choice([1, 0, 5000, 1731465123457])
    base_kind = rng.random()
    if base_kind < 0.4:
        frames = N["knife_stream"](rng, dur)
    else:
        frames = N["frame_stream"](rng, dur, t0=rng.randrange(1, 10_000_000))
    shift = t0 - frames[0].t_ms
    seq = {f.t_ms + shift: f for f in frames}
    # 정확한 장면을 사이사이에 넣는다(그 시각의 기존 프레임을 덮는다)
    t = t0 + rng.randrange(0, 5000)
    hits: dict = {}
    while t < t0 + dur:
        scene, t_end = exact_scenes(rng, t, rng.choice([33, 50, 100]))
        for (tt, ch) in scene:
            if tt >= t0 + dur:
                break
            base = seq.get(tt) or RT.FrameState(t_ms=tt, face=True, closure=0.1, closure_valid=True, provisional=False,
                                                yaw_rel=0.0, pitch_rel=0.0, pose_valid=True, gaze_h_rel=0.0,
                                                gaze_v_rel=0.0, jaw_rel=0.0)
            d = {f: getattr(base, f) for f in ("t_ms", "face", "closure", "closure_valid", "provisional", "yaw_rel",
                                               "pitch_rel", "pose_valid", "gaze_h_rel", "gaze_v_rel", "jaw_rel")}
            d.update(face=True, closure_valid=True, **ch)
            if "closure" not in ch and d["closure"] > 0.95:
                d["closure"] = 0.1
            seq[tt] = RT.FrameState(**d)
            for k, v in ch.items():
                key = f"{k}={v!r}"
                hits[key] = hits.get(key, 0) + 1
        t = t_end + rng.choice([0, 1000, 5000, 20000])
    times = sorted(seq)
    origin = times[0]
    vmax = spec.closure.valid_max_abs_yaw_deg
    states = []
    for tt in times:
        s = seq[tt]
        prov = selfcal and (tt - origin) < pb["baseline"]["duration_ms"]
        if not s.face:
            s = RT.FrameState(t_ms=tt, face=False, closure=0.0, closure_valid=False, provisional=prov, yaw_rel=0.0,
                              pitch_rel=0.0, pose_valid=False, gaze_h_rel=0.0, gaze_v_rel=0.0, jaw_rel=0.0)
        else:
            cv = s.closure_valid and abs(s.yaw_rel) <= vmax
            s = RT.FrameState(t_ms=tt, face=True, closure=s.closure if cv else 0.0, closure_valid=cv, provisional=prov,
                              yaw_rel=s.yaw_rel, pitch_rel=s.pitch_rel, pose_valid=not prov, gaze_h_rel=s.gaze_h_rel,
                              gaze_v_rel=s.gaze_v_rel, jaw_rel=s.jaw_rel)
        states.append(s)
    # 버려질 프레임(같은 시각, 뒤로 간 시각)을 몇 개 넣는다
    fed = []
    for s in states:
        fed.append(s)
        r = rng.random()
        if r < 0.003:
            fed.append((s.t_ms, None))
        elif r < 0.005:
            fed.append((s.t_ms - rng.choice([1, 10, 500]), None))
    speeds = N["speed_stream"](rng, len(fed))
    baseline = None
    state_baseline = None
    if not selfcal:
        baseline = {"e_l": 0.27, "e_r": 0.27, "b_l": 0.22, "b_r": 0.22, "r0": list(RT.IDENTITY_R0), "jaw0": 0.08,
                    "gaze_h0": 0.0, "gaze_v0": 0.0, "t_complete_ms": rng.choice([0, 0, 3000, 60000]),
                    "status": rng.choice(["ok", "ok", "off_axis"])}
        if baseline["t_complete_ms"] % spec.clock.step_ms:
            baseline["t_complete_ms"] = 0
    else:
        a = 0
        idx = None
        prev = None
        for x in fed:
            tt = x.t_ms if isinstance(x, RT.FrameState) else x[0]
            if prev is not None and tt <= prev:
                continue
            prev = tt
            if tt - origin >= pb["baseline"]["duration_ms"]:
                idx = a
                break
            a += 1
        b = hk.RT.Baseline(e_l=0.27, e_r=0.28, b_l=0.2, b_r=0.21, r0=tuple(RT.IDENTITY_R0), jaw0=0.07, gaze_h0=0.01,
                           gaze_v0=-0.02, t_complete_ms=pb["baseline"]["duration_ms"], status="ok")
        diag = hk._calibrate.BaselineDiagnostics(n_face_frames=1000, face_ratio=0.9, blink_l_quantiles=(0.3, 0.5),
                                                 blink_r_quantiles=(0.31, 0.52))
        state_baseline = None if idx is None else (idx, b, diag)
    stub = None
    if rng.random() < 0.5:
        ps = []
        while len(ps) < 400:
            p = rng.choice([0.4, 0.6, 0.4, 0.6, 0.0, 0.9, 0.41, 0.59])
            ps += [p] * rng.choice([1, 2, 5, 10, 11, 25])
        stub = [((1.0 - p) * 0.25, (1.0 - p) * 0.75, p) for p in ps]
        if rng.random() < 0.7:
            rd["drowsiness"]["source"] = rng.choice(["model", "either"])
    S = hk.Session(corpus="S-edge", sid=f"S-edge/{'cal' if selfcal else 'inj'}/{seed}", klass="cal" if selfcal else "inj",
                   seed=seed, pb=pb, rd=rd, kind="state", frames=fed, speeds=speeds, baseline=baseline, stub=stub,
                   generator="gen_edge.make: vendored knife_stream / frame_stream + exact-threshold scenes",
                   notes={"exact_hits": hits, "self_calibrating": selfcal, "duration_ms": dur, "t0": t0})
    S.state_baseline = state_baseline
    return S
