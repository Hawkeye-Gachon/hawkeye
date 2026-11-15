"""S 묶음(STATE): 복사해 둔 `vendor/r5bv_spec_tx/tx_fuzz.py`의 `gen_session`, `frames_of`로 세션을 만든다.

생성기 파일은 고치지 않고 import만 한다. 세션 하나 = (종류, 시드). 프레임은 RAW로 참조에 넣고, 파일에는 STATE 입력과
참조 출력만 쓴다(계획 9.2.2절의 표: SIG와 STATE 세션은 프레임 입력 가운데 `t_ms`, `speed`, `accepted`만).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import hk

VENDOR = os.path.join(hk.HERE, "vendor", "r5bv_spec_tx")
_FUZZ = None


def fuzz():  # noqa: ANN201
    global _FUZZ
    if _FUZZ is None:
        if VENDOR not in sys.path:
            sys.path.insert(0, VENDOR)
        spec = importlib.util.spec_from_file_location("tx_fuzz_vendored", os.path.join(VENDOR, "tx_fuzz.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FUZZ = mod
    return _FUZZ


def gen_files() -> dict:
    return {os.path.relpath(p, hk.HERE): hk.sha256_file(p) for p in (
        os.path.join(VENDOR, "tx_fuzz.py"), os.path.join(VENDOR, "spec_tx.py"),
        os.path.join(hk.HERE, "gen_s.py"), os.path.join(hk.HERE, "hk.py"))}


def make(corpus: str, klass: str, seed: int) -> hk.Session:
    F = fuzz()
    S = F.gen_session(klass, seed)
    frames, speeds = [], []
    for (t_ms, face, w, h, lm, bs, m, speed) in F.frames_of(S):
        frames.append(hk.RT.RawFrame(t_ms=t_ms, face=face, w=w, h=h, lm=lm, bs=bs, m=m))
        speeds.append(speed)
    stub = None
    if S["model"] is not None:
        stub = [((1.0 - p) * 0.25, (1.0 - p) * 0.75, p) for p in S["model"]]
    return hk.Session(
        corpus=corpus, sid=f"{corpus}/{klass}/{seed}", klass=klass, seed=seed, pb=S["pb"], rd=S["rd"], kind="raw",
        frames=frames, speeds=speeds, baseline=S["baseline"], feature_start_ms=S["feature_start_ms"], stub=stub,
        generator="tx_fuzz.gen_session+frames_of (vendored r5bv_spec_tx)",
        notes={"grid": S["grid"], "dur_ms": S["dur"], "t0": S["t0"], "nan_rate": S["nan_rate"],
               "f32_matrix": S["f32"], "size": list(S["size"])})
