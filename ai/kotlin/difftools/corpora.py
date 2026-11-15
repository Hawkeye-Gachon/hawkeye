"""세션 묶음 만들기 (FORMATS.md 1.4절). 묶음마다 작업 목록을 만들고 프로세스 W개(최대 10)로 나눠 돌린다."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time

import hk
from small import gen_info, write_json

S_MIX = (("inj", 35), ("inj_model", 15), ("cal", 20), ("var", 30))
G_KLASSES = (("e_edge", 40), ("yaw15", 40), ("width6", 30), ("counts", 40), ("held", 40), ("e_zero", 10))


def s_tasks(corpus: str, n: int, base: int) -> list:
    out = []
    for klass, share in S_MIX:
        k = round(n * share / 100)
        out += [(corpus, klass, base + i) for i in range(k)]
    return out


def tasks(name: str) -> list:
    if name == "S-quick":
        return s_tasks("S-quick", 600, 5_000_000)
    if name == "S-full":
        return s_tasks("S-full", 12_000, 6_000_000)
    if name == "S-odd":
        return [("S-odd", "odd", 5_100_000 + i) for i in range(500)]
    if name == "S-edge":
        return [("S-edge", "edge", i) for i in range(400)]
    if name == "F-stat":
        return [("F-stat", c, s) for s in range(20) for c in range(3)]
    if name == "F-inj":
        return [("F-inj", fps, s) for fps, n in ((30, 134), (15, 133), (10, 133)) for s in range(n)]
    if name == "F-cal":
        return [("F-cal", "cal", s) for s in range(40)]
    if name == "F-calib":
        return [("F-calib", "calib", s) for s in range(300)]
    if name == "G-sig":
        return [("G-sig", k, s) for k, n in G_KLASSES for s in range(n)]
    raise KeyError(name)


def make_session(task):  # noqa: ANN001, ANN201
    corpus, klass, seed = task
    if corpus.startswith("S-") and corpus != "S-edge":
        import gen_s
        return gen_s.make(corpus, klass, seed), gen_s.gen_files()
    if corpus == "S-edge":
        import gen_edge
        return gen_edge.make(seed), gen_edge.gen_files()
    import gen_f
    if corpus == "F-stat":
        return gen_f.f_stat(seed, klass), gen_f.gen_files()
    if corpus == "F-inj":
        return gen_f.f_inj(klass, seed), gen_f.gen_files()
    if corpus == "F-cal":
        return gen_f.f_cal(seed), gen_f.gen_files()
    if corpus == "F-calib":
        return gen_f.f_calib(seed), gen_f.gen_files()
    return gen_f.g_sig(klass, seed), gen_f.gen_files()


def work(task):  # noqa: ANN001, ANN201
    S, files = make_session(task)
    path = os.path.join(hk.DATA, S.sid + ".hkd")
    s = hk.export_session(path, S, files)
    s["sha256"] = hk.sha256_file(path)
    s["session_id"] = S.sid
    s["mm"] = {f"T{k}": v for k, v in s.pop("min_margin").items()}
    s["notes"] = S.notes
    return s


def coverage_check(name: str, res: list) -> None:
    """계획 9.2.3절의 희귀한 경우가 묶음에 실제로 있는지(FORMATS 1.4절). 없으면 멈춘다."""
    if name == "G-sig":
        got = sorted(r["notes"]["left_eye_samples_before_60s"] for r in res
                     if r["session_id"].startswith("G-sig/counts/"))
        if set(got) != set(range(300, 401, 10)):
            raise AssertionError(f"G-sig: counts sessions cover {sorted(set(got))}")
        print(f"  G-sig: counts sessions {len(got)}, left-eye sample counts before 60 s {sorted(set(got))}", flush=True)
    if name == "F-calib":
        got = sorted(r["notes"]["exact_half_face_ms_before_60s"] for r in res
                     if r["notes"].get("exact_half_face_ms_before_60s") is not None)
        if not {30000, 29900} <= set(got):
            raise AssertionError(f"F-calib: exact-half face times {got}")
        print(f"  F-calib: exact-half sessions {len(got)}, face ms before 60 s {got}", flush=True)


def build(name: str, workers: int) -> str:
    d = os.path.join(hk.DATA, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    ts = tasks(name)
    t0 = time.time()
    res = []
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, 10)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, ts, chunksize=4)):
            res.append(r)
            if (i + 1) % max(1, len(ts) // 10) == 0:
                print(f"  {name}: {i + 1}/{len(ts)} sessions, {time.time() - t0:.0f} s", flush=True)
    res.sort(key=lambda r: r["session_id"])
    coverage_check(name, res)
    minm: dict = {}
    for r in res:
        for k, v in r["mm"].items():
            if v is not None and (k not in minm or v < minm[k]):
                minm[k] = v
    tot = {"sessions": len(res), "frames_fed": sum(r["n_fed"] for r in res),
           "frames_accepted": sum(r["n_accepted"] for r in res),
           "near_RAW": sum(1 for r in res if r["labels"]["RAW"] == "near"),
           "near_SIG": sum(1 for r in res if r["labels"]["SIG"] == "near"),
           "exceptions": sum(1 for r in res if r["exception"]),
           "constructor_errors": sum(1 for r in res if r["constructor_error"]),
           "bytes": sum(r["bytes"] for r in res), "min_margin": minm}
    WALL[name] = round(time.time() - t0, 1)
    sessions = [{"session_id": r["session_id"], "path": r["path"], "sha256": r["sha256"], "bytes": r["bytes"],
                 "n_fed": r["n_fed"], "n_accepted": r["n_accepted"], "labels": r["labels"], "exception": r["exception"],
                 "constructor_error": r["constructor_error"], "min_margin": r["mm"]} for r in res]
    bad = [r["session_id"] for r in res if not (r["selfcheck"] or "").startswith(("read-back", "constructor"))]
    if bad:
        raise AssertionError(f"{name}: self-check missing for {bad[:5]}")
    write_json(os.path.join(d, "index.json"), {"corpus": name, "reference": hk.reference(),
                                               "generator": gen_info("corpora.py"), "count": len(res),
                                               "selfcheck": sorted({r["selfcheck"] for r in res}),
                                               "sessions": sessions, "totals": tot})
    print(f"  {name}: {json.dumps(tot)}", flush=True)
    return d


WALL: dict = {}  # 묶음 이름 -> 이번 실행의 생성 시간(초). index.json에는 넣지 않는다(파일이 실행마다 같도록)


def totals_line(dirs: list[str]) -> dict:
    out = {}
    for d in dirs:
        with open(os.path.join(d, "index.json"), encoding="utf-8") as fh:
            t = json.load(fh)["totals"]
        out[os.path.basename(d)] = (f"sessions {t['sessions']}, frames fed {t['frames_fed']}, accepted "
                                    f"{t['frames_accepted']}, near RAW {t['near_RAW']}, near SIG {t['near_SIG']}, "
                                    f"exceptions {t['exceptions']}, bytes {t['bytes']}, generation {WALL.get(os.path.basename(d))} s")
    return out
