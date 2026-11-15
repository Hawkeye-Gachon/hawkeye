"""차등 비교 자료 내보내기의 입구 (FORMATS.md 0.1절).

    ORT_DISABLE_TELEMETRY=1 nice -n 19 .venv/bin/python kotlin/difftools/export.py small|corpora|full|check
        [--only NAME[,NAME...]] [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import hk  # noqa: E402

SMALL = ("selftest", "canary", "comparator", "tamper", "model_windows", "active", "replay0", "components")
CORPORA = ("S-quick", "S-odd", "S-edge", "F-stat", "F-inj", "F-cal", "F-calib", "G-sig")
FULL = ("S-full",)


def _rel(p: str) -> str:
    return os.path.relpath(p, hk.DATA)


def dir_bytes(d: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def write_ready(name: str, entries: list[str], command: str, wall_s: float, extra: dict) -> str:
    lines = [f"# reference_id: {hk.reference()['reference_id']}", f"# command: {command}",
             f"# wall_time_s: {wall_s:.1f}"]
    lines += [f"# {k}: {v}" for k, v in extra.items()]
    for p in entries:
        if os.path.isdir(p):
            idx = os.path.join(p, "index.json")
            lines.append(f"{hk.sha256_file(idx)}  {dir_bytes(p)}  {_rel(p)}/")
        else:
            lines.append(f"{hk.sha256_file(p)}  {os.path.getsize(p)}  {_rel(p)}")
    path = os.path.join(hk.DATA, name)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(path + ".tmp", path)
    return path


def run_small(only: list[str]) -> list[str]:
    import small
    files: list[str] = []
    for name in only:
        t = time.time()
        if name == "selftest":
            import selftest
            out = selftest.run()
        elif name == "tamper":
            import tamper
            out = tamper.tamper()
        elif name == "components":
            import components
            out = components.all_components()
        else:
            out = getattr(small, name)()
        files += out
        print(f"[{name}] {len(out)} files, {time.time() - t:.1f} s", flush=True)
    return files


def run_corpora(only: list[str], workers: int) -> list[str]:
    import corpora
    dirs = []
    for name in only:
        t = time.time()
        d = corpora.build(name, workers)
        dirs.append(d)
        print(f"[{name}] {time.time() - t:.1f} s, {dir_bytes(d)} bytes", flush=True)
    return dirs


def check() -> int:
    bad = 0
    for name in ("READY_small.txt", "READY_corpora.txt", "READY_full.txt"):
        path = os.path.join(hk.DATA, name)
        if not os.path.exists(path):
            print(f"{name}: absent")
            continue
        n = 0
        n_dep = 0
        for line in open(path, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                if line.startswith("# reference_id:") and line.split(": ", 1)[1].strip() != hk.reference()["reference_id"]:
                    print(f"{name}: reference_id differs from the repository today")
                    bad += 1
                continue
            sha, _size, rel = line.split("  ", 2)
            rel = rel.strip()
            p = os.path.join(hk.DATA, rel)
            if rel.endswith("/"):
                idx = os.path.join(p, "index.json")
                if hk.sha256_file(idx) != sha:
                    print(f"{rel}: index.json sha256 differs")
                    bad += 1
                    continue
                with open(idx, encoding="utf-8") as fh:
                    index = json.load(fh)
                for s in index["sessions"]:
                    if hk.sha256_file(os.path.join(hk.DATA, s["path"])) != s["sha256"]:
                        print(f"{s['path']}: sha256 differs")
                        bad += 1
                    n += 1
            else:
                if hk.sha256_file(p) != sha:
                    print(f"{rel}: sha256 differs")
                    bad += 1
                elif rel.endswith(".json"):  # 보조 모듈의 지문(FORMATS 0.2절)
                    try:  # tamper/의 JSON은 일부러 깨진 것이 있다(BOM, 읽히지 않는 바이트): 지문 검사 대상이 아니다
                        with open(p, encoding="utf-8") as fh:
                            obj = json.load(fh)
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict) and "depends_on" in obj:
                        bad += hk.check_depends_on(obj, rel)
                        n_dep += 1
                n += 1
        print(f"{name}: {n} files checked, depends_on checked in {n_dep}")
    print(f"check: {bad} problems")
    return 1 if bad else 0


def _verify_one(rel: str) -> tuple[str, list, int]:
    r = hk.verify_file(os.path.join(hk.DATA, rel))
    return rel, r["modes"], r["frames"]


def verify(names: list[str], workers: int) -> int:
    """READY에 적힌 묶음의 세션 파일을 모두 새로 읽어, 생성기 없이 파일만으로 참조를 다시 돌려 비교한다(E1)."""
    import multiprocessing as mp
    rels = []
    for name in names:
        idx = os.path.join(hk.DATA, name, "index.json")
        if not os.path.exists(idx):
            print(f"{name}: absent")
            continue
        with open(idx, encoding="utf-8") as fh:
            rels += [s["path"] for s in json.load(fh)["sessions"]]
    t0 = time.time()
    modes: dict = {}
    frames = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for rel, ms, n in pool.imap_unordered(_verify_one, rels, chunksize=4):
            for m in ms:
                modes[m] = modes.get(m, 0) + 1
            frames += n
    print(f"verify: {len(rels)} session files of {names}, {frames} fed frames, re-run per mode {modes}, "
          f"all bit-equal, {time.time() - t0:.1f} s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["small", "corpora", "full", "check", "verify"])
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-ready", action="store_true")
    a = ap.parse_args()
    if os.environ.get("ORT_DISABLE_TELEMETRY") != "1":
        print("set ORT_DISABLE_TELEMETRY=1 first", file=sys.stderr)
        return 2
    workers = min(a.workers, 10)
    t0 = time.time()
    print(f"reference_id {hk.reference()['reference_id']}", flush=True)
    if a.what == "check":
        return check()
    if a.what == "verify":
        return verify(a.only.split(",") if a.only else list(CORPORA + FULL), workers)
    if a.what == "small":
        only = a.only.split(",") if a.only else list(SMALL)
        files = run_small(only)
        if not a.only and not a.no_ready:
            p = write_ready("READY_small.txt", files, "export.py small", time.time() - t0, {})
            print(f"wrote {p}")
    else:
        names = CORPORA if a.what == "corpora" else FULL
        only = a.only.split(",") if a.only else list(names)
        dirs = run_corpora(only, workers)
        if not a.only and not a.no_ready:
            import corpora
            extra = corpora.totals_line(dirs)
            p = write_ready("READY_corpora.txt" if a.what == "corpora" else "READY_full.txt", dirs,
                            f"export.py {a.what} --workers {workers}", time.time() - t0, extra)
            print(f"wrote {p}")
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
