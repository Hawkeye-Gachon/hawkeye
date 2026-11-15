"""F 묶음(RAW, SIG, STATE)과 G-sig(SIG, STATE) 세션 (계획 9.2.3절). 생성기는 저장소의 `drowsy.synth`다."""
from __future__ import annotations

import copy
import math
import os
import random

import hk

RT = hk.RT
from drowsy import signals as SG  # noqa: E402
from drowsy import synth as Y  # noqa: E402
from drowsy.spec import pipeline_spec_from_dict  # noqa: E402

SPEC = pipeline_spec_from_dict(hk.PIPE_BLOCK)
FSEED = 20261115


def gen_files() -> dict:
    return {os.path.relpath(p, hk.HERE): hk.sha256_file(p) for p in (
        os.path.join(hk.HERE, "gen_f.py"), os.path.join(hk.HERE, "hk.py"))}


def speed_profile(rng: random.Random, n: int, times: list[int]) -> list:
    out, v, nxt = [], rng.choice([None, 60.0, 30.0]), 0
    t0 = times[0] if times else 0
    for t in times:
        if t - t0 >= nxt:
            v = rng.choice([None, 0.0, 10.0, 19.999, 20.0, 30.0, 49.99, 50.0, 80.0, -5.0])
            nxt += rng.choice([3000, 10000, 40000])
        out.append(v)
    return out


def gaze_of(el) -> tuple[float, float]:  # noqa: ANN001
    return SG.gaze_h(el), SG.gaze_v(el)


def personal_baseline(profile: Y.FaceProfile, t_complete_ms: int = 0) -> dict:
    gh, gv = gaze_of(profile.eye_look_rest)
    return {"e_l": profile.e_open_l, "e_r": profile.e_open_r, "b_l": profile.b_open_l, "b_r": profile.b_open_r,
            "r0": list(SG.rotation_from_euler(profile.yaw, profile.pitch, profile.roll)), "jaw0": profile.jaw_rest,
            "gaze_h0": gh, "gaze_v0": gv, "t_complete_ms": t_complete_ms,
            "status": "off_axis" if abs(profile.yaw) > 15.0 else "ok"}


def random_profile(rng: random.Random) -> Y.FaceProfile:
    if rng.random() < 0.4:
        return Y.default_profile(SPEC)
    e = rng.uniform(0.22, 0.34)
    b = rng.uniform(0.05, 0.3)
    return Y.FaceProfile(e_open_l=e * rng.uniform(0.97, 1.03), e_open_r=e * rng.uniform(0.97, 1.03),
                         b_open_l=b, b_open_r=b + rng.uniform(-0.02, 0.02), jaw_rest=rng.uniform(0.0, 0.1),
                         yaw=rng.uniform(-18, 18), pitch=rng.uniform(-8, 8), roll=rng.uniform(-4, 4),
                         eye_look_rest=tuple(rng.uniform(0, 0.1) for _ in range(8)),
                         eye_w_px=rng.choice([20.0, 30.0, 44.0, 60.0]),
                         centre=(rng.uniform(0.42, 0.58), rng.uniform(0.38, 0.52)))


def random_segments(rng: random.Random, start: int, end: int, heavy: bool) -> list:
    segs = []
    t = start + rng.choice([0, 500, 2000])
    while t < end - 1500:
        k = rng.choices(["blink", "closure", "linger", "miss", "turn", "look", "yawn", "nod", "drowsy"],
                        weights=[6, 4, 2, 2, 3, 2, 1, 1, 2 if heavy else 0])[0]
        if k == "blink":
            s = Y.Blink(duration_ms=rng.choice([150, 200, 300, 400]), depth=rng.choice([1.0, 0.9, 0.6]), start_ms=t)
        elif k == "closure":
            s = Y.Closure(hold_ms=rng.choice([200, 600, 900, 1100, 2500, 3200, 5000]), depth=rng.choice([1.0, 0.95, 0.8]),
                          close_ms=rng.choice([66, 100, 150]), open_ms=rng.choice([100, 150, 250]), start_ms=t)
        elif k == "linger":
            s = Y.Linger(duration_ms=rng.choice([1000, 3000, 6000]), level=rng.choice([0.4, 0.55, 0.65, 0.75]),
                         start_ms=t)
        elif k == "miss":
            s = Y.FaceMissing(duration_ms=rng.choice([300, 2000, 3500, 6000, 9000]), start_ms=t)
        elif k == "turn":
            s = Y.HeadTurn(duration_ms=rng.choice([1000, 2500, 4000, 8000]), yaw=rng.choice([30.0, -30.0, 45.0, -26.0]),
                           pitch=rng.choice([0.0, 0.0, 20.0, -28.0]), roll=rng.choice([0.0, 5.0]), start_ms=t)
        elif k == "look":
            s = Y.LookAside(duration_ms=rng.choice([1000, 4000]), gaze_h=rng.choice([0.0, 0.35, -0.35]),
                            gaze_v=rng.choice([0.0, 0.45, -0.45]), start_ms=t)
        elif k == "yawn":
            s = Y.Yawn(duration_ms=rng.choice([2500, 4000]), jaw=rng.choice([0.45, 0.6]), start_ms=t)
        elif k == "nod":
            s = Y.Nod(duration_ms=rng.choice([600, 1000, 2000]), pitch=rng.choice([15.0, 20.0]), start_ms=t)
        else:
            for j in range(rng.choice([6, 10])):
                segs.append(Y.Closure(hold_ms=rng.choice([900, 1300, 1800]), depth=1.0, start_ms=t + 4000 * j))
            t += 40000
            continue
        segs.append(s)
        t += getattr(s, "duration_ms", 1000) + rng.choice([200, 1000, 3000, 6000])
    return segs


def _render_session(corpus, klass, seed, sc: Y.Scenario, baseline, stub, rd, rng, notes) -> hk.Session:  # noqa: ANN001
    frames = Y.render(sc, SPEC)
    speeds = speed_profile(rng, len(frames), [f.t_ms for f in frames])
    return hk.Session(corpus=corpus, sid=f"{corpus}/{klass}/{seed}", klass=klass, seed=seed, pb=hk.PIPE_BLOCK, rd=rd,
                      kind="raw", frames=frames, speeds=speeds, baseline=baseline, stub=stub,
                      generator="drowsy.synth.render (scripted Scenario)", notes=notes)


# ---------------------------------------------------------------------------------------------- F-stat
def f_stat(subject: int, class_index: int) -> hk.Session:
    prof = Y.make_subject(subject, FSEED)
    frames = list(Y.generate_session(prof, class_index, 120000, FSEED, SPEC))
    rng = random.Random(f"F-stat:{subject}:{class_index}")
    speeds = speed_profile(rng, len(frames), [f.t_ms for f in frames])
    klass = RT.CLASS_ORDER[class_index]
    return hk.Session(corpus="F-stat", sid=f"F-stat/{klass}/{subject}", klass=klass, seed=subject, pb=hk.PIPE_BLOCK,
                      rd=hk.RULES_DOC, kind="raw", frames=frames, speeds=speeds,
                      generator="drowsy.synth.make_subject + generate_session",
                      notes={"subject": subject, "class_index": class_index, "duration_ms": 120000, "seed": FSEED,
                             "fps": prof.fps, "w": prof.w, "h": prof.h})


# ---------------------------------------------------------------------------------------------- F-inj
def f_inj(fps: int, seed: int) -> hk.Session:
    rng = random.Random(f"F-inj:{fps}:{seed}")
    prof = random_profile(rng)
    dur = rng.choice([20000, 40000, 60000, 90000])
    segs = random_segments(rng, 0, dur, heavy=dur >= 60000)
    gaps = ()
    if rng.random() < 0.3:
        g0 = rng.randrange(2000, max(3000, dur - 15000))
        gaps = ((g0, g0 + rng.choice([250, 600, 3300, 5300, 10001, 12000])),)
    sc = Y.Scenario(segments=tuple(segs), duration_ms=dur, fps=float(fps), w=rng.choice([640, 1280, 480]),
                    h=rng.choice([480, 720, 640]), jitter_ms=rng.choice([0, 0, 3, 10]),
                    drop_prob=rng.choice([0.0, 0.0, 0.02, 0.1]), gaps=gaps,
                    first_t_ms=rng.choice([1, 5000, 1731465123457]), profile=prof, seed=seed)
    tc = rng.choice([0, 0, 0, 3000, 60000]) if dur > 70000 else 0
    base = personal_baseline(prof, tc)
    return _render_session("F-inj", str(fps), seed, sc, base, None, hk.RULES_DOC, rng,
                           {"fps": fps, "duration_ms": dur, "segments": len(segs), "gaps": [list(g) for g in gaps]})


# ---------------------------------------------------------------------------------------------- F-cal
def stub_table(rng: random.Random, n: int = 400) -> list:
    ps = []
    while len(ps) < n:
        p = rng.choice([0.0, 0.3, 0.4, 0.41, 0.59, 0.6, 0.61, 0.9, 0.9, 0.2, 0.7])
        ps += [p] * rng.choice([1, 2, 5, 9, 10, 11, 12, 25])
    return [((1.0 - p) * 0.25, (1.0 - p) * 0.75, p) for p in ps[:n]]


def f_cal(seed: int) -> hk.Session:
    rng = random.Random(f"F-cal:{seed}")
    prof = random_profile(rng)
    if abs(prof.yaw) > 14:
        prof = Y.default_profile(SPEC)
    segs = [Y.Blink(duration_ms=200, start_ms=2000 + 4000 * k) for k in range(14)]
    segs += random_segments(rng, 62000, 300000, heavy=True)
    sc = Y.Scenario(segments=tuple(segs), duration_ms=300000, fps=float(rng.choice([10, 15, 20, 30])),
                    jitter_ms=rng.choice([0, 3]), drop_prob=rng.choice([0.0, 0.01]),
                    first_t_ms=rng.choice([1, 1731465123457]), profile=prof, seed=seed)
    rd = copy.deepcopy(hk.RULES_DOC)
    rd["drowsiness"]["source"] = rng.choice(["model", "model", "either", "perclos"])
    return _render_session("F-cal", "cal", seed, sc, None, stub_table(rng), rd, rng,
                           {"fps": sc.fps, "duration_ms": 300000, "source": rd["drowsiness"]["source"]})


# ---------------------------------------------------------------------------------------------- F-calib
DEADLINES = (60000, 90000, 120000, 150000, 180000)


def _sig(t_ms: int, face: bool, rng: random.Random, e: float, b: float, yaw: float, w_l: float = 40.0,
         w_r: float = 40.0, ear_bias: float = 0.0) -> RT.FrameSignals:
    if not face:
        return RT.FrameSignals(t_ms=t_ms, face=False, ear_l=0.0, ear_r=0.0, eye_w_l=0.0, eye_w_r=0.0, blink_l=0.0,
                               blink_r=0.0, eye_look=(0.0,) * 8, jaw_open=0.0, yaw=0.0, pitch=0.0, roll=0.0)
    return RT.FrameSignals(t_ms=t_ms, face=True, ear_l=e + ear_bias + rng.uniform(-0.01, 0.01),
                           ear_r=e + ear_bias + rng.uniform(-0.01, 0.01), eye_w_l=w_l, eye_w_r=w_r,
                           blink_l=b + rng.uniform(0, 0.05), blink_r=b + rng.uniform(0, 0.05),
                           eye_look=tuple(rng.uniform(0, 0.08) for _ in range(8)), jaw_open=rng.uniform(0.02, 0.1),
                           yaw=yaw + rng.uniform(-1.5, 1.5), pitch=rng.uniform(-3, 3), roll=rng.uniform(-2, 2),
                           iris_h_l=rng.uniform(0.4, 0.6), iris_h_r=rng.uniform(0.4, 0.6), iris_v_l=rng.uniform(-0.05, 0.05),
                           iris_v_r=rng.uniform(-0.05, 0.05))


def f_calib(seed: int) -> hk.Session:
    """보정만 하는 세션. 목표 마감 앞의 마감마다 완성을 막는 까닭 하나와, 모든 마감 바로 앞에 장면 하나."""
    rng = random.Random(f"F-calib:{seed}")
    fps = rng.choice([6, 7, 8, 9, 10])
    target = rng.choice([0, 0, 1, 1, 2, 3, 4, 5])  # 5 = 180초에도 완성하지 못함(대체)
    dur = (DEADLINES[min(target, 4)] + 10000)
    e0, b0 = rng.uniform(0.24, 0.32), rng.uniform(0.08, 0.25)
    yaw0 = rng.choice([0.0, 5.0, -8.0, 14.0, 16.0, -16.0])
    reasons = [rng.choice(["narrow", "miss"]) for _ in range(min(target, 5))]
    exact_half = fps == 10 and target == 0 and rng.random() < 0.5
    if exact_half:
        dur = 100000  # 60초에서 연장된 쪽(29900 ms)이 90초 마감에서 완성되는 것까지 보인다
    times = []
    k = 0
    while True:
        t = math.floor(k * 1000.0 / fps + 0.5)
        if t >= dur:
            break
        times.append(t)
        k += 1
    # 마감 바로 앞 장면: 'faceless' 마지막 프레임이 얼굴 없음, 'gap' 마감을 건너는 200 ms 넘는 끊김, 'exact' 마감에 프레임
    scenes = {}
    for d in DEADLINES:
        if d < dur:
            scenes[d] = rng.choice(["faceless", "gap", "gap", "exact", "none"])
    tset = []
    for t in times:
        drop = False
        for d, kind in scenes.items():
            if kind == "gap" and d - rng.choice([150, 201, 250, 400]) < t < d + 60:
                drop = True
        if not drop:
            tset.append(t)
    for d, kind in scenes.items():
        if kind == "exact" and d < dur:
            tset.append(d)
        if kind == "gap":
            g1 = rng.choice([150, 201, 250, 400])
            tset.append(d - g1)
            tset.append(d + rng.choice([60, 1, 120]))
    tset = sorted(set(x for x in tset if 0 <= x < dur))
    if exact_half:  # 10 fps 격자, 60초 마감에서 얼굴 시간 정확히 30000 ms(통과) 또는 29900 ms(연장)
        tset = list(range(0, dur, 100))
        rng.choice([300, 301])  # 난수열을 예전과 같게 둔다(값은 아래에서 시드로 정한다)
        miss_n = 300 + seed % 2  # 두 쪽이 모두 나오게: 짝수 시드 300(얼굴 30000 ms), 홀수 시드 301(29900 ms)
        miss = set(tset[100:100 + miss_n])
        face_ms_60 = 100 * sum(1 for t in tset if t < 60000 and t not in miss)
        if face_ms_60 != 60000 - 100 * miss_n:
            raise AssertionError(f"F-calib/{seed}: exact-half face time {face_ms_60}")
    first = rng.choice([1, 5000, 1731465123457])
    sigs = []
    prev_deadline = 0
    for t in tset:
        # 이 프레임이 속한 마감 구간
        idx = next((i for i, d in enumerate(DEADLINES) if t < d), 5)
        face = True
        w_r = 40.0
        ear_bias = 0.0
        if exact_half:
            face = t not in miss
        else:
            if idx < len(reasons):
                if reasons[idx] == "narrow" and idx == 0:
                    w_r = 3.0  # 60초 마감에 오른눈 표본이 없다
                elif reasons[idx] == "miss":
                    lo = (DEADLINES[idx - 1] if idx > 0 else 0)
                    span = DEADLINES[idx] - lo
                    face = not (lo + 0.1 * span <= t < lo + 0.1 * span + 0.6 * DEADLINES[idx])
                elif reasons[idx] == "narrow":
                    face = not (DEADLINES[idx] - 0.6 * DEADLINES[idx] <= t < DEADLINES[idx])
            if target == 5:
                ear_bias = 0.25  # E가 e_max를 넘는다: 180초까지 완성하지 못한다
            for d, kind in scenes.items():
                if kind == "faceless" and d - 1000 // fps - 1 <= t < d:
                    face = False
            if rng.random() < 0.01:
                face = False
        sigs.append(_sig(first + t, face, rng, e0, b0, yaw0, w_r=w_r, ear_bias=ear_bias))
    w, h = rng.choice([(640, 480), (1280, 720), (480, 640)])
    frames = [Y.frame_from_signals(s, w, h, SPEC) for s in sigs]
    speeds = [None] * len(frames)
    return hk.Session(corpus="F-calib", sid=f"F-calib/calib/{seed}", klass="calib", seed=seed, pb=hk.PIPE_BLOCK,
                      rd=hk.RULES_DOC, kind="raw", frames=frames, speeds=speeds,
                      generator="drowsy.synth.frame_from_signals on scripted calibration signals",
                      notes={"fps": fps, "target_deadline_index": target, "reasons": reasons, "scenes": {
                          str(k): v for k, v in scenes.items()}, "exact_half_face_ratio": exact_half,
                          "yaw0": yaw0, "duration_ms": dur,
                          "exact_half_face_ms_before_60s": face_ms_60 if exact_half else None})


# ---------------------------------------------------------------------------------------------- G-sig
def g_sig(klass: str, seed: int) -> hk.Session:
    rng = random.Random(f"G-sig:{klass}:{seed}")
    fps = rng.choice([6, 8, 10])
    if klass == "counts" and fps == 6:
        fps = 8  # 6 fps는 60초에 360프레임뿐이라 400개 표본에 닿지 못한다
    dur = rng.choice([70000, 100000, 130000, 190000])
    first = rng.choice([1, 1731465123457])
    times = sorted(set(math.floor(k * 1000.0 / fps + 0.5) for k in range(int(dur * fps / 1000))))
    sigs = []
    base = None
    e_l_val = 0.27
    notes: dict = {"fps": fps, "duration_ms": dur}
    n_l = None
    if klass == "e_edge":  # E가 정확히 e_min이나 e_max(또는 한 ulp 밖)
        which = rng.choice(["e_min", "e_max"])
        v = 0.12 if which == "e_min" else 0.45
        if rng.random() < 0.5:
            v = math.nextafter(v, -1.0 if which == "e_min" else 2.0)
        e_l_val = v
        notes["E_target"] = v.hex()
    elif klass == "yaw15":
        yv = rng.choice([15.0, -15.0, math.nextafter(15.0, 20.0), math.nextafter(-15.0, -20.0)])
        notes["yaw0_target"] = yv.hex()
    elif klass == "width6":
        notes["eye_w"] = "exactly 6.0 and nextafter below on some frames"
    elif klass == "counts":
        rng.choice(list(range(300, 401, 10)))  # 난수열을 예전과 같게 둔다
        n_l = 300 + 10 * (seed % 11)  # 시드 0..39가 300, 310, ..., 400을 모두 덮는다
        notes["left_eye_samples_before_60s"] = n_l
    elif klass == "held":
        notes["scene"] = "faceless frame or a >200 ms gap directly before each deadline"
    elif klass == "e_zero":
        base = {"e_l": 0.0, "e_r": 0.27, "b_l": 0.2, "b_r": 0.2, "r0": list(RT.IDENTITY_R0), "jaw0": 0.05,
                "gaze_h0": 0.0, "gaze_v0": 0.0, "t_complete_ms": 0, "status": "ok"}
        notes["injected_baseline"] = "e_l = 0.0 (E*(1 - rho) <= 0 in eye_closure)"
    left_count = 0
    cut = set()
    if klass == "held":
        for d in DEADLINES:
            if rng.random() < 0.5:
                g1 = rng.choice([201, 250, 400])
                cut |= {t for t in times if d - g1 < t < d + 50}
            else:
                last = max((t for t in times if t < d), default=None)
                if last is not None:
                    cut.add(-last)  # 표지: 얼굴 없음
    for t in times:
        if t in cut:
            continue
        face = rng.random() > 0.03 and (-t) not in cut
        yaw = rng.uniform(-2, 2)
        if klass == "yaw15":
            yaw = yv if rng.random() < 0.7 else yv + rng.choice([-3.0, 3.0])
        w_l, w_r = 40.0, 40.0
        if klass == "width6":
            w_l = rng.choice([6.0, 6.0, math.nextafter(6.0, 0.0), 40.0])
        if klass == "counts":
            usable = t < 60000 and left_count < n_l
            w_l = 40.0 if usable or t >= 60000 else 3.0
            if usable and face:
                left_count += 1
        ear_l = e_l_val if (klass == "e_edge" and rng.random() < 0.95) else (
            0.27 + 1e-7 * len(sigs) if klass == "counts" else rng.uniform(0.2, 0.3))
        if klass == "e_edge" and e_l_val > 0.3 and ear_l != e_l_val:
            ear_l = rng.uniform(0.2, e_l_val)
        if not face:
            sigs.append(_sig(first + t, False, rng, 0.0, 0.0, 0.0))
            continue
        s = RT.FrameSignals(t_ms=first + t, face=True, ear_l=ear_l, ear_r=rng.uniform(0.24, 0.3), eye_w_l=w_l,
                            eye_w_r=w_r, blink_l=rng.uniform(0.05, 0.3), blink_r=rng.uniform(0.05, 0.3),
                            eye_look=tuple(rng.choice([0.0, rng.uniform(0, 0.1)]) for _ in range(8)),
                            jaw_open=rng.choice([0.05, rng.uniform(0, 0.6)]), yaw=yaw, pitch=rng.uniform(-4, 4),
                            roll=rng.uniform(-3, 3),
                            iris_h_l=None if w_l < 6.0 else rng.uniform(0.3, 0.7), iris_h_r=rng.uniform(0.3, 0.7),
                            iris_v_l=None if w_l < 6.0 else rng.uniform(-0.1, 0.1), iris_v_r=rng.uniform(-0.1, 0.1))
        sigs.append(s)
    if klass == "counts" and left_count != n_l:
        raise AssertionError(f"G-sig/counts/{seed}: {left_count} left-eye samples before 60 s, target {n_l}")
    speeds = [None] * len(sigs)
    return hk.Session(corpus="G-sig", sid=f"G-sig/{klass}/{seed}", klass=klass, seed=seed, pb=hk.PIPE_BLOCK,
                      rd=hk.RULES_DOC, kind="sig", frames=sigs, speeds=speeds, baseline=base,
                      generator="hand-made FrameSignals (gen_f.g_sig)", notes=notes)
