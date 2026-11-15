"""HKDIFF1 형식, 참조 구현의 해시, 참조를 돌리는 세 입구(RAW, SIG, STATE)와 세션 기록기.

일곱 이식 모듈은 import만 하고 바꾸지 않는다. STATE와 SIG 입구는 `drowsy.pipeline` 모듈 안의 이름 셋
(`signals`, `calibrate`, `BaselineEstimator`)을 실행 중에만 대리 객체로 바꿔 `DriverPipeline.update`의 코드를 그대로
지나게 한다(파일은 그대로다). 여유(margin) 기록은 `tests/test_vectors.py:964-978`처럼 함수를 감싸 원래 값을 돌려준다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "kotlin", ".diffdata")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.dont_write_bytecode = True

from drowsy import blink as _blink  # noqa: E402
from drowsy import calibrate as _calibrate  # noqa: E402
from drowsy import pipeline as _pipeline  # noqa: E402
from drowsy import signals as _signals  # noqa: E402
from drowsy import step_features as _step_features  # noqa: E402
from drowsy import types as RT  # noqa: E402
from drowsy.pipeline import DriverPipeline  # noqa: E402
from drowsy.spec import pipeline_spec_from_dict, rules_from_dict, rules_to_json, compute_pipeline_hash  # noqa: E402

MAGIC = b"HKDIFF1\n"
FORMAT_VERSION = 1

SEVEN = ("drowsy/types.py", "drowsy/signals.py", "drowsy/calibrate.py", "drowsy/blink.py",
         "drowsy/step_features.py", "drowsy/rules.py", "drowsy/pipeline.py")
HELPERS = ("drowsy/spec.py", "drowsy/synth.py", "drowsy/vectors.py", "drowsy/demo.py")
CONFIG = ("config/feature_spec.json", "config/rules.yaml")
EXPORT = ("export/feature_spec_v0.json", "export/rules.json", "export/drowsiness_v0.onnx")

with open(os.path.join(REPO, "config", "feature_spec.json"), encoding="utf-8") as _fh:
    PIPE_BLOCK: dict = json.load(_fh)["pipeline"]
with open(os.path.join(REPO, "export", "rules.json"), encoding="utf-8") as _fh:
    RULES_DOC: dict = json.load(_fh)
SHIPPED_PIPELINE_HASH = compute_pipeline_hash(PIPE_BLOCK)
with open(os.path.join(REPO, "export", "rules.json"), "rb") as _fh:
    SHIPPED_RULES_HASH = hashlib.sha256(_fh.read()).hexdigest()

_lm = PIPE_BLOCK["landmarks"]
LM44: tuple[int, ...] = tuple(sorted(set(_lm["ear_r"] + _lm["ear_l"] + _lm["iris_r"] + _lm["iris_l"] + _lm["contour_r"]
                                         + _lm["contour_l"] + _lm["face_width"])))
_bsb = PIPE_BLOCK["blendshapes"]
BS11: tuple[str, ...] = (_bsb["eye_blink_l"], _bsb["eye_blink_r"], _bsb["jaw_open"], *_bsb["eye_look"])
assert len(LM44) == 44 and len(BS11) == 11

EVENT_TYPES = RT.EVENT_TYPES
PHASES = RT.PHASES
SOURCES = RT.SOURCES
STATUSES = RT.BASELINE_STATUSES
ZONES = ("in", "out", "unknown")
ENUMS = {"event_types": list(EVENT_TYPES), "phases": list(PHASES), "sources": list(SOURCES),
         "baseline_statuses": list(STATUSES), "zones": list(ZONES)}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_REF: dict | None = None


def reference() -> dict:
    """참조 구현의 지문: 일곱 모듈, 설정 둘, 내보낸 파일 셋, 쓴 보조 모듈, 버전. `reference_id`는 앞 셋의 요약이다."""
    global _REF
    if _REF is None:
        import onnxruntime
        groups = {}
        for name, paths in (("modules", SEVEN), ("config", CONFIG), ("export", EXPORT), ("helpers", HELPERS)):
            groups[name] = {p: sha256_file(os.path.join(REPO, p)) for p in paths}
        lines = "".join(f"{p}:{groups[g][p]}\n" for g in ("modules", "config", "export") for p in sorted(groups[g]))
        try:
            import onnx
            onnx_version = onnx.__version__
        except ImportError:  # pragma: no cover
            onnx_version = None
        _REF = {**groups, "reference_id": hashlib.sha256(lines.encode("ascii")).hexdigest(),
                "reference_id_rule": "sha256 of the ASCII lines '<path>:<sha256>\\n' of modules, config, export, "
                                     "each group sorted by path, groups in that order",
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "onnxruntime": onnxruntime.__version__, "onnx": onnx_version}}
    return _REF


def depends_on(*paths: str) -> dict:
    """기대값이 `reference_id` 밖의 보조 모듈에서 오는 파일의 지문(FORMATS 0.2절): {저장소 경로: sha256}.

    `reference_id`는 바꾸지 않는다(이미 있는 Kotlin 시험이 12개 파일로 다시 계산한다). 이 칸을 가진 파일을 쓰는 시험은
    적힌 파일마다 저장소의 sha256과 같은지 따로 본다.
    """
    return {p: sha256_file(os.path.join(REPO, p)) for p in paths}


def check_depends_on(obj: dict, what: str) -> int:
    """obj의 depends_on이 저장소와 같은지. 다른 파일 수를 돌려주고 다른 것을 출력한다."""
    bad = 0
    for p, sha in (obj.get("depends_on") or {}).items():
        mine = sha256_file(os.path.join(REPO, p))
        if mine != sha:
            print(f"{what}: depends_on {p} {sha} != repository {mine}")
            bad += 1
    return bad


def hexf(x: float | None) -> str | None:
    """float를 Python `float.hex()` 글자로. None은 그대로.

    가정(A31의 한쪽): JVM의 `Double.parseDouble`이 이 글자를 비트 그대로 읽는다. 여기서는 Python 쪽 왕복만 확인한다
    (E1의 16진 검사). JVM 쪽은 E2, E13, E17이 본다.
    """
    return None if x is None else float(x).hex()


# ---------------------------------------------------------------------------------------------- HKDIFF1 파일
DTYPES = {"u8": "<u1", "i32": "<i4", "u32": "<u4", "i64": "<i8", "f64": "<f8", "f32": "<f4"}


def _align8(n: int) -> int:
    return (n + 7) & ~7


def write_hk(path: str, header: dict, columns: list[tuple[str, str, Any]]) -> dict:
    """columns: (이름, dtype, 값). 값은 numpy 배열(모양 그대로 쓴다)이나 목록. 머리에 열 표와 data_sha256을 넣어 쓴다."""
    table, blobs, pos = [], [], 0
    arrays = {}
    for name, dt, value in columns:
        arr = np.ascontiguousarray(np.asarray(value, dtype=DTYPES[dt]))
        raw = arr.tobytes()
        table.append({"name": name, "dtype": dt, "shape": list(arr.shape), "offset": pos, "nbytes": len(raw)})
        blobs.append(raw + b"\0" * (_align8(len(raw)) - len(raw)))
        pos += _align8(len(raw))
        arrays[name] = arr
    data = b"".join(blobs)
    header = dict(header)
    header["format"] = "HKDIFF1"
    header["format_version"] = FORMAT_VERSION
    header["columns"] = table
    header["data_nbytes"] = len(data)
    header["data_sha256"] = hashlib.sha256(data).hexdigest()
    hbytes = json.dumps(header, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    pre = MAGIC + struct.pack("<I", len(hbytes)) + hbytes
    pre += b"\0" * (_align8(len(pre)) - len(pre))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(pre)
        fh.write(data)
    os.replace(tmp, path)
    return arrays


def read_hk(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:8] != MAGIC:
        raise ValueError(f"{path}: not HKDIFF1")
    (hlen,) = struct.unpack("<I", blob[8:12])
    header = json.loads(blob[12:12 + hlen].decode("utf-8"))
    start = _align8(12 + hlen)
    data = blob[start:]
    if len(data) != header["data_nbytes"] or hashlib.sha256(data).hexdigest() != header["data_sha256"]:
        raise ValueError(f"{path}: data_sha256 or length differs")
    cols = {}
    for c in header["columns"]:
        raw = data[c["offset"]:c["offset"] + c["nbytes"]]
        cols[c["name"]] = np.frombuffer(raw, dtype=DTYPES[c["dtype"]]).reshape(c["shape"])
    return header, cols


def same_arrays(a: dict[str, np.ndarray], b: dict[str, np.ndarray], names=None) -> list[str]:
    """이름이 같은 배열의 모양과 바이트가 같은지. 다른 이름의 목록을 돌려준다."""
    bad = []
    for n in (names if names is not None else sorted(set(a) | set(b))):
        x, y = a.get(n), b.get(n)
        if x is None or y is None:
            if (x is None) != (y is None):
                bad.append(n + " (presence)")
            continue
        if x.shape != y.shape or x.dtype.str != y.dtype.str or x.tobytes() != y.tobytes():
            bad.append(n)
    return bad


# ---------------------------------------------------------------------------------------------- 여유 기록
@dataclass
class MarginTap:
    """T1~T8 여유를 프레임마다 모은다. 감싼 함수는 원래 결과를 그대로 돌려준다."""

    on: bool = False
    frame: dict = field(default_factory=dict)  # id -> 그 프레임의 가장 작은 여유
    est_cfg: Any = None

    def put(self, tid: int, value: float) -> None:
        if not self.on:
            return
        v = abs(value)
        if v != v:  # NaN: 여유를 정할 수 없다. 0으로 적어 near로 본다
            v = 0.0
        old = self.frame.get(tid)
        if old is None or v < old:
            self.frame[tid] = v


TAP = MarginTap()
_orig_round = _blink._round_half_up
_orig_try = _calibrate.BaselineEstimator._try_complete
_orig_nod = _step_features.NodDetector.update


def _round_wrap(x: float) -> int:
    if TAP.on and x == x and not math.isinf(x):
        TAP.put(8, x - math.floor(x) - 0.5)
    return _orig_round(x)


def _try_wrap(self, face_ms):  # noqa: ANN001
    """T2, T3: 참조가 이 마감에서 실제로 한 비교만 적는다(`calibrate.py:241-262`).

    T2는 `e_l >= e_min and e_l <= e_max and e_r >= e_min and e_r <= e_max`의 짧은 평가 순서를 따른다. T3(`abs(yaw0)`
    대 `off_axis_yaw_deg`)은 마감이 개인 기준선으로 확정될 때만 있다(완성 조건이 참일 때, `calibrate.py:248-262`).
    완성 여부는 원래 함수의 결과로 안다: 개인 기준선이면 상태가 ok나 off_axis이고, 연장이면 None, 대체면 fallback_global.
    """
    if not TAP.on:
        return _orig_try(self, face_ms)
    cfg = self._cfg
    if len(self._ear_l) > 0 and len(self._ear_r) > 0:
        e_l = _calibrate.nearest_rank_quantile(self._ear_l, cfg.ear_quantile)
        e_r = _calibrate.nearest_rank_quantile(self._ear_r, cfg.ear_quantile)
        chain = ((e_l, cfg.e_min, "ge"), (e_l, cfg.e_max, "le"), (e_r, cfg.e_min, "ge"), (e_r, cfg.e_max, "le"))
        for e, thr, op in chain:
            TAP.put(2, e - thr)
            if not (e >= thr if op == "ge" else e <= thr):
                break
    yaw0 = _calibrate.median(self._yaw) if len(self._yaw) > 0 else None
    res = _orig_try(self, face_ms)
    if res is not None and res.status in (RT.BASELINE_OK, RT.BASELINE_OFF_AXIS) and yaw0 is not None:
        TAP.put(3, abs(yaw0) - cfg.off_axis_yaw_deg)
    return res


def _nod_wrap(self, t_ms, pitch_rel, valid):  # noqa: ANN001
    if TAP.on and valid and self._has_mean:
        d = pitch_rel - self._mean
        TAP.put(7, d - self._counter._enter_thr)
        TAP.put(7, d - self._counter._exit_thr)
    elif TAP.on and valid:
        TAP.put(7, 0.0 - self._counter._enter_thr)
        TAP.put(7, 0.0 - self._counter._exit_thr)
    return _orig_nod(self, t_ms, pitch_rel, valid)


_blink._round_half_up = _round_wrap
_calibrate.BaselineEstimator._try_complete = _try_wrap
_step_features.NodDetector.update = _nod_wrap


def state_margins(spec, rules, st: RT.FrameState) -> None:  # noqa: ANN001
    """T4, T5, T6: 프레임 상태에서 바로 정해지는 비교."""
    if st.face:
        TAP.put(4, abs(st.yaw_rel) - spec.closure.valid_max_abs_yaw_deg)
    if st.closure_valid:
        for thr in (spec.run_detector.enter_thr, spec.run_detector.exit_thr, rules.closure.detector.enter_thr,
                    rules.closure.detector.exit_thr, spec.step_features.perclos_thr):
            TAP.put(5, st.closure - thr)
    if st.pose_valid:
        z = rules.forward_zone
        TAP.put(6, abs(st.yaw_rel) - z.yaw_abs_max_deg)
        TAP.put(6, st.pitch_rel + z.pitch_up_max_deg)
        TAP.put(6, st.pitch_rel - z.pitch_down_max_deg)


def signal_margins(spec, s: RT.FrameSignals) -> None:  # noqa: ANN001
    if s.face:
        TAP.put(1, s.eye_w_l - spec.signals.min_eye_width_px)
        TAP.put(1, s.eye_w_r - spec.signals.min_eye_width_px)


# ---------------------------------------------------------------------------------------------- 세 입구의 대리 객체
class _Feed:
    """지금 넣는 프레임의 입력과, 참조가 계산한 마지막 신호와 상태를 든다."""

    def __init__(self) -> None:
        self.mode = "RAW"
        self.cur_sig = None
        self.cur_state = None
        self.last_sig = None
        self.last_state = None
        self.state_provisional_seen = None
        self.bl_baseline = None
        self.bl_diag = None
        self.bl_token = None
        self.cur_token = None


FEED = _Feed()


class _SignalsProxy:
    def __getattr__(self, name):  # noqa: ANN001
        return getattr(_signals, name)

    def compute(self, frame, spec):  # noqa: ANN001
        if FEED.mode == "RAW":
            s = _signals.compute(frame, spec)
        elif FEED.mode == "SIG":
            s = FEED.cur_sig
            if s.t_ms != frame.t_ms:
                raise AssertionError("SIG feed out of step")
        else:
            s = ("STATE-TOKEN", FEED.cur_token)
        FEED.last_sig = s
        return s


class _CalibrateProxy:
    def __getattr__(self, name):  # noqa: ANN001
        return getattr(_calibrate, name)

    def frame_state(self, s, b, spec, provisional):  # noqa: ANN001
        if FEED.mode == "STATE":
            st = FEED.cur_state
            if st.provisional != provisional:
                raise AssertionError(f"STATE feed: provisional {st.provisional} but pipeline passes {provisional}")
            if FEED.bl_baseline is not None and not provisional and b != FEED.bl_baseline:
                raise AssertionError("STATE feed: pipeline baseline differs from the recorded one")
        else:
            st = _calibrate.frame_state(s, b, spec, provisional)
        FEED.last_state = st
        return st


class _EstimatorProxy:
    """STATE 입구에서만 쓴다: 기록된 프레임에서 기록된 기준선을 돌려준다."""

    def __init__(self, spec) -> None:  # noqa: ANN001
        self.diagnostics = None
        self._done = False

    def update(self, s):  # noqa: ANN001
        if self._done:
            return None
        if s[1] == FEED.bl_token and FEED.bl_baseline is not None:
            self._done = True
            self.diagnostics = FEED.bl_diag
            return FEED.bl_baseline
        return None


def _estimator_factory(spec):  # noqa: ANN001
    if FEED.mode == "STATE":
        return _EstimatorProxy(spec)
    return _calibrate.BaselineEstimator(spec)


# 가정: `DriverPipeline`은 원시 신호, 프레임 상태, 기준선 추정기를 `drowsy.pipeline` 안의 이 세 이름으로만 부른다
#   (`pipeline.py:167, 175, 177, 105`). 확인: RAW로 만든 파일을 RAW, SIG, STATE로 다시 돌린 출력이 모두 비트로 같다
#   (`export.py verify`).
_pipeline.signals = _SignalsProxy()
_pipeline.calibrate = _CalibrateProxy()
_pipeline.BaselineEstimator = _estimator_factory


# ---------------------------------------------------------------------------------------------- 진행 중 이벤트 접기
def fold_active(active: dict, events, frame_t_ms: int) -> None:
    """계획 6.2절의 접기(12장 K10 (가)). active: 종류 -> 항목 dict. 제자리에서 바꾼다."""
    for e in events:
        if e.phase == RT.PHASE_ONSET:
            active[e.type] = {"type": e.type, "severity": e.severity, "t_start_ms": e.t_start_ms, "source": e.source,
                              "provisional": e.provisional, "onset_frame_t_ms": frame_t_ms}
            if e.type == RT.EVENT_SLEEP:
                active.pop(RT.EVENT_MICROSLEEP, None)
        else:
            active.pop(e.type, None)
            if e.type == RT.EVENT_SLEEP:
                active.pop(RT.EVENT_MICROSLEEP, None)


def active_list(active: dict, priority) -> list:  # noqa: ANN001
    return [active[t] for t in priority if t in active]


def active_mask(active: dict) -> int:
    m = 0
    for t in active:
        m |= 1 << EVENT_TYPES.index(t)
    return m


# ---------------------------------------------------------------------------------------------- 세션 정의와 실행
@dataclass
class Session:
    corpus: str
    sid: str
    klass: str
    seed: Any
    pb: dict
    rd: dict
    kind: str  # "raw" | "sig" | "state"
    frames: list  # raw: RawFrame, sig: FrameSignals, state: FrameState 또는 (t_ms, None) (버려질 프레임)
    speeds: list
    baseline: dict | None = None  # 넣어 준 기준선(plain dict, r0 목록)
    feature_start_ms: int | None = None
    stub: list | None = None  # 모델 호출 i에 돌려줄 (alert, low, drowsy), 마지막 행을 되풀이
    generator: str = ""
    notes: dict = field(default_factory=dict)
    # STATE 입구로 다시 돌릴 때만: 기준선이 생긴 받은 프레임의 인덱스와 그 값
    state_baseline: tuple | None = None


def baseline_obj(b: dict) -> RT.Baseline:
    return RT.Baseline(**{**b, "r0": tuple(b["r0"])})


def baseline_plain(b: RT.Baseline) -> dict:
    return {"e_l": b.e_l, "e_r": b.e_r, "b_l": b.b_l, "b_r": b.b_r, "r0": list(b.r0), "jaw0": b.jaw0,
            "gaze_h0": b.gaze_h0, "gaze_v0": b.gaze_v0, "t_complete_ms": b.t_complete_ms, "status": b.status}


def baseline_hex(b: dict | None) -> dict | None:
    if b is None:
        return None
    out = {k: hexf(b[k]) for k in ("e_l", "e_r", "b_l", "b_r", "jaw0", "gaze_h0", "gaze_v0")}
    out["r0"] = [hexf(v) for v in b["r0"]]
    out["t_complete_ms"] = int(b["t_complete_ms"])
    out["status"] = b["status"]
    return out


def bl_vec(b: RT.Baseline) -> list[float]:
    return [b.e_l, b.e_r, b.b_l, b.b_r, *b.r0, b.jaw0, b.gaze_h0, b.gaze_v0]


class Result:
    """한 번의 참조 실행에서 모은 열과 머리의 일부."""

    def __init__(self) -> None:
        self.c: dict[str, list] = {}
        self.meta: dict = {}

    def add(self, name, value):  # noqa: ANN001
        self.c.setdefault(name, []).append(value)


SIG_F = ("ear_l", "ear_r", "eye_w_l", "eye_w_r", "blink_l", "blink_r", "jaw_open", "yaw", "pitch", "roll")
STATE_F = ("closure", "yaw_rel", "pitch_rel", "gaze_h_rel", "gaze_v_rel", "jaw_rel")


def _sig_row(r: Result, s: RT.FrameSignals) -> None:
    r.add("sig_face", 1 if s.face else 0)
    for f in SIG_F:
        r.add(f, getattr(s, f))
    r.add("eye_look", list(s.eye_look))
    iris = (s.iris_h_l, s.iris_h_r, s.iris_v_l, s.iris_v_r)
    r.add("iris", [0.0 if v is None else v for v in iris])
    r.add("iris_null", [1 if v is None else 0 for v in iris])


def _state_row(r: Result, st: RT.FrameState) -> None:
    for f in STATE_F:
        r.add(f, getattr(st, f))
    r.add("state_flags", (1 if st.face else 0) | (2 if st.closure_valid else 0) | (4 if st.provisional else 0)
          | (8 if st.pose_valid else 0))


def _event_rows(r: Result, prefix: str, frame: int, events) -> None:  # noqa: ANN001
    for e in events:
        r.add(prefix + "frame", frame)
        r.add(prefix + "type", EVENT_TYPES.index(e.type))
        r.add(prefix + "severity", e.severity)
        r.add(prefix + "times", [e.t_start_ms, e.duration_ms])
        r.add(prefix + "confidence", e.confidence)
        r.add(prefix + "source", SOURCES.index(e.source))
        r.add(prefix + "phase", PHASES.index(e.phase))
        r.add(prefix + "provisional", 1 if e.provisional else 0)


def run_session(S: Session, mode: str, margins: bool = True, model_override=None) -> Result:  # noqa: ANN001
    """S를 참조에 돌린다. mode: S.frames를 어느 입구로 넣는가(RAW, SIG, STATE).

    model_override: 주면 S.stub 대신 이 모델 함수를 쓴다(C-errors의 "3개가 아닌 값을 돌려주는 모델" 경우만).
    """
    spec = pipeline_spec_from_dict(S.pb)
    rules = rules_from_dict(S.rd)
    r = Result()
    calls: list[tuple[int, list]] = []
    model = model_override
    if model_override is None and S.stub is not None:
        table = S.stub

        def model(window):  # noqa: ANN001
            calls.append((len(calls), [list(map(float, row)) for row in window]))
            row = table[min(len(calls) - 1, len(table) - 1)]
            return (row[0], row[1], row[2])
    kw = {}
    if S.baseline is not None:
        kw["baseline"] = baseline_obj(S.baseline)
        if S.feature_start_ms is not None:
            kw["feature_start_ms"] = S.feature_start_ms
    FEED.mode = mode
    FEED.bl_baseline = FEED.bl_diag = FEED.bl_token = None
    if mode == "STATE" and S.state_baseline is not None:
        FEED.bl_token, FEED.bl_baseline, FEED.bl_diag = S.state_baseline
    r.meta["exception"] = None
    try:
        p = DriverPipeline(spec, rules, model=model, **kw)
    except Exception as exc:  # noqa: BLE001
        r.meta["constructor_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return r
    r.meta["constructor_error"] = None
    priority = tuple(rules.alert_priority)
    active: dict = {}
    had_baseline = S.baseline is not None
    a = 0
    n_calls_before = 0
    for i, fr in enumerate(S.frames):
        speed = S.speeds[i]
        if mode == "RAW":
            frame = fr
            t_ms = fr.t_ms
        elif mode == "SIG":
            FEED.cur_sig = fr
            t_ms = fr.t_ms
            frame = RT.RawFrame(t_ms=t_ms, face=fr.face, w=0, h=0, lm={}, bs={}, m=())
        else:
            st = fr if isinstance(fr, RT.FrameState) else None
            t_ms = st.t_ms if st is not None else fr[0]
            FEED.cur_state = st
            FEED.cur_token = a
            frame = RT.RawFrame(t_ms=t_ms, face=False, w=0, h=0, lm={}, bs={}, m=())
        r.add("t_ms", t_ms)
        r.add("speed", 0.0 if speed is None else speed)
        r.add("speed_null", 1 if speed is None else 0)
        if mode in ("RAW", "SIG"):
            s_rec = _signals.compute(fr, spec) if mode == "RAW" else fr
            _sig_row(r, s_rec)
        TAP.on = margins
        TAP.frame = {}
        FEED.last_state = None
        try:
            out = p.update(frame, speed)
        except Exception as exc:  # noqa: BLE001
            TAP.on = False
            tb = exc.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next
            where = f"{os.path.relpath(tb.tb_frame.f_code.co_filename, REPO)}:{tb.tb_lineno} {tb.tb_frame.f_code.co_name}"
            r.meta["exception"] = {"fed_index": i, "accepted_index": a, "type": type(exc).__name__,
                                   "message": str(exc), "where": where}
            r.add("accepted", 2)
            if FEED.last_state is not None:
                _state_row(r, FEED.last_state)
                r.add("gaze_zone", 255)
                r.add("active_mask", 0)
                r.add("ongoing_feature_ms", 0)
                r.add("margin", math.inf)
                r.add("margin_id", 0)
                r.add("round_margin", math.inf)
            r.meta["exception"]["state_recorded"] = FEED.last_state is not None
            break
        TAP.on = False
        if out is None:
            r.add("accepted", 0)
            continue
        r.add("accepted", 1)
        st = out.state
        _state_row(r, st)
        if margins:
            TAP.on = True
            if mode == "RAW":
                signal_margins(spec, s_rec)
            state_margins(spec, rules, st)
            TAP.on = False
        mg = {k: v for k, v in TAP.frame.items() if k != 8}
        if mg:
            mid = min(mg, key=lambda k: (mg[k], k))
            r.add("margin", mg[mid])
            r.add("margin_id", mid)
        else:
            r.add("margin", math.inf)
            r.add("margin_id", 0)
        r.add("round_margin", TAP.frame.get(8, math.inf))
        for k, v in TAP.frame.items():
            mm = r.meta.setdefault("min_margin", {})
            if k not in mm or v < mm[k]:
                mm[k] = v
        r.add("gaze_zone", ZONES.index(p.gaze_zone))
        fold_active(active, out.events, st.t_ms)
        r.add("active_mask", active_mask(active))
        r.add("ongoing_feature_ms", p._detector.ongoing_ms())
        if not had_baseline and p.baseline is not None:
            had_baseline = True
            b = p.baseline
            r.meta["baseline_completed"] = {"accepted_index": a, "baseline": baseline_plain(b)}
            d = p.baseline_diagnostics
            r.meta["baseline_diag"] = None if d is None else {
                "n_face_frames": d.n_face_frames, "face_ratio": d.face_ratio,
                "blink_l_quantiles": list(d.blink_l_quantiles), "blink_r_quantiles": list(d.blink_r_quantiles)}
        for run in out.runs:
            r.add("run_frame", a)
            r.add("run_times", [run.start_ms, run.end_ms, run.duration_ms])
            r.add("run_values", [run.amplitude, run.peak_closing_velocity, run.peak_opening_velocity])
            r.add("run_flags", (1 if run.velocity_valid else 0) | (2 if run.truncated else 0))
        for step, pr in zip(out.steps, out.step_probs):
            r.add("st_frame", a)
            r.add("st_k", step.k)
            r.add("st_t_end_ms", step.t_end_ms)
            r.add("st_full", 1 if step.full_lookback else 0)
            r.add("st_values", list(step.values))
            r.add("st_prob_null", 1 if pr is None else 0)
            r.add("st_probs", [0.0, 0.0, 0.0] if pr is None else list(pr))
        # 이 프레임의 모델 호출: 확률이 있는 스텝과 차례로 짝이 맞는다
        ks = [s.k for s, pr in zip(out.steps, out.step_probs) if pr is not None]
        new = calls[n_calls_before:]
        if len(new) != len(ks):
            raise AssertionError("model calls and step probabilities do not pair up")
        for (ci, win), k in zip(new, ks):
            r.add("mw_frame", a)
            r.add("mw_k", k)
            arr = np.asarray(win, dtype="<f8")
            r.add("mw_sha", list(hashlib.sha256(arr.tobytes()).digest()))
            prev_k = r.c["mw_k"][-2] if len(r.c["mw_k"]) >= 2 else None
            if prev_k is None or k != prev_k + 1:
                r.add("mwfull_call", ci)
                r.add("mwfull", win)
        n_calls_before = len(calls)
        _event_rows(r, "ev_", a, out.events)
        _event_rows(r, "al_", a, out.alerts)
        a += 1
    TAP.on = False
    r.meta["counters"] = {"frames_accepted": p.frames_accepted, "dropped_frames": p.dropped_frames,
                          "steps_emitted": p.steps_emitted, "model_calls": p.model_calls,
                          "nonfinite_windows": p.nonfinite_windows}
    r.meta["active_at_end"] = active_list(active, priority)
    r.meta.setdefault("baseline_completed", None)
    r.meta.setdefault("baseline_diag", None)
    r.meta["n_accepted_rows"] = a + (1 if (r.meta["exception"] or {}).get("state_recorded") else 0)
    return r


# ---------------------------------------------------------------------------------------------- 열 표
_SHAPES = {  # 이름 -> (dtype, 한 행의 모양)
    "t_ms": ("i64", ()), "speed": ("f64", ()), "speed_null": ("u8", ()), "accepted": ("u8", ()),
    "face": ("u8", ()), "w": ("i32", ()), "h": ("i32", ()), "lm": ("f64", (44, 2)), "bs": ("f64", (11,)),
    "m": ("f64", (16,)),
    "sig_face": ("u8", ()), **{f: ("f64", ()) for f in SIG_F}, "eye_look": ("f64", (8,)), "iris": ("f64", (4,)),
    "iris_null": ("u8", (4,)),
    **{f: ("f64", ()) for f in STATE_F}, "state_flags": ("u8", ()),
    "gaze_zone": ("u8", ()), "active_mask": ("u8", ()), "ongoing_feature_ms": ("i64", ()), "margin": ("f64", ()),
    "margin_id": ("u8", ()), "round_margin": ("f64", ()),
    "run_frame": ("u32", ()), "run_times": ("i64", (3,)), "run_values": ("f64", (3,)), "run_flags": ("u8", ()),
    "st_frame": ("u32", ()), "st_k": ("i64", ()), "st_t_end_ms": ("i64", ()), "st_full": ("u8", ()),
    "st_values": ("f64", (17,)), "st_prob_null": ("u8", ()), "st_probs": ("f64", (3,)),
    "mw_frame": ("u32", ()), "mw_k": ("i64", ()), "mw_sha": ("u8", (32,)), "mwfull_call": ("u32", ()),
    **{p + n: t for p in ("ev_", "al_") for n, t in (("frame", ("u32", ())), ("type", ("u8", ())),
                                                       ("severity", ("u8", ())), ("times", ("i64", (2,))),
                                                       ("confidence", ("f64", ())), ("source", ("u8", ())),
                                                       ("phase", ("u8", ())), ("provisional", ("u8", ())))},
}
FED = ("t_ms", "speed", "speed_null", "accepted")
RAWC = ("face", "w", "h", "lm", "bs", "m")
SIGC = ("sig_face", *SIG_F, "eye_look", "iris", "iris_null")
STATEC = (*STATE_F, "state_flags")
EXTRAC = ("gaze_zone", "active_mask", "ongoing_feature_ms", "margin", "margin_id", "round_margin")
OUTC = ("run_frame", "run_times", "run_values", "run_flags", "st_frame", "st_k", "st_t_end_ms", "st_full",
        "st_values", "st_prob_null", "st_probs", "mw_frame", "mw_k", "mw_sha", "mwfull_call", "mwfull",
        *[p + n for p in ("ev_", "al_") for n in ("frame", "type", "severity", "times", "confidence", "source",
                                                 "phase", "provisional")])


def _arr(name: str, values: list, window_steps: int = 120) -> tuple[str, str, np.ndarray]:
    if name == "mwfull":
        return (name, "f64", np.asarray(values, dtype="<f8").reshape((len(values), window_steps, 17)))
    dt, shape = _SHAPES[name]
    return (name, dt, np.asarray(values, dtype=DTYPES[dt]).reshape((len(values), *shape)))


def raw_columns(frames: list) -> dict[str, list]:
    c = {k: [] for k in RAWC}
    for fr in frames:
        c["face"].append(1 if fr.face else 0)
        c["w"].append(fr.w)
        c["h"].append(fr.h)
        if fr.face:
            c["lm"].append([list(fr.lm[i]) for i in LM44])
            c["bs"].append([fr.bs[n] for n in BS11])
            c["m"].append(list(fr.m))
    return c


def frames_from_raw(cols: dict[str, np.ndarray]) -> list:
    out, f = [], 0
    for i in range(len(cols["t_ms"])):
        t, face, w, h = int(cols["t_ms"][i]), bool(cols["face"][i]), int(cols["w"][i]), int(cols["h"][i])
        if face:
            lm = {idx: (float(cols["lm"][f, j, 0]), float(cols["lm"][f, j, 1])) for j, idx in enumerate(LM44)}
            bs = {n: float(cols["bs"][f, j]) for j, n in enumerate(BS11)}
            m = tuple(float(v) for v in cols["m"][f])
            f += 1
            out.append(RT.RawFrame(t_ms=t, face=True, w=w, h=h, lm=lm, bs=bs, m=m))
        else:
            out.append(RT.RawFrame(t_ms=t, face=False, w=w, h=h, lm={}, bs={}, m=()))
    return out


def signals_from_cols(cols: dict[str, np.ndarray]) -> list:
    out = []
    for i in range(len(cols["t_ms"])):
        iris = [None if cols["iris_null"][i, j] else float(cols["iris"][i, j]) for j in range(4)]
        out.append(RT.FrameSignals(
            t_ms=int(cols["t_ms"][i]), face=bool(cols["sig_face"][i]),
            **{f: float(cols[f][i]) for f in SIG_F}, eye_look=tuple(float(v) for v in cols["eye_look"][i]),
            iris_h_l=iris[0], iris_h_r=iris[1], iris_v_l=iris[2], iris_v_r=iris[3]))
    return out


def states_from_cols(cols: dict[str, np.ndarray]) -> list:
    out, a = [], 0
    for i in range(len(cols["t_ms"])):
        t = int(cols["t_ms"][i])
        if cols["accepted"][i] == 0:
            out.append((t, None))
            continue
        if a >= len(cols["state_flags"]):
            out.append((t, None))  # 예외 프레임에서 상태를 기록하지 못한 경우
            continue
        fl = int(cols["state_flags"][a])
        out.append(RT.FrameState(t_ms=t, face=bool(fl & 1), closure=float(cols["closure"][a]),
                                 closure_valid=bool(fl & 2), provisional=bool(fl & 4),
                                 yaw_rel=float(cols["yaw_rel"][a]), pitch_rel=float(cols["pitch_rel"][a]),
                                 pose_valid=bool(fl & 8), gaze_h_rel=float(cols["gaze_h_rel"][a]),
                                 gaze_v_rel=float(cols["gaze_v_rel"][a]), jaw_rel=float(cols["jaw_rel"][a])))
        a += 1
    return out


def baseline_from_cols(header: dict, cols: dict[str, np.ndarray]):  # noqa: ANN201
    """STATE 입구용 (받은 프레임 인덱스, Baseline, BaselineDiagnostics 또는 None)."""
    if "bl_frame" not in cols:
        return None
    v = [float(x) for x in cols["bl_f64"][0]]
    b = RT.Baseline(e_l=v[0], e_r=v[1], b_l=v[2], b_r=v[3], r0=tuple(v[4:13]), jaw0=v[13], gaze_h0=v[14],
                    gaze_v0=v[15], t_complete_ms=int(cols["bl_t_complete_ms"][0]),
                    status=STATUSES[int(cols["bl_status"][0])])
    d = None
    if "dg_f64" in cols:
        dv = [float(x) for x in cols["dg_f64"][0]]
        nq = (len(dv) - 1) // 2
        d = _calibrate.BaselineDiagnostics(n_face_frames=int(cols["dg_n_face"][0]), face_ratio=dv[0],
                                           blink_l_quantiles=tuple(dv[1:1 + nq]),
                                           blink_r_quantiles=tuple(dv[1 + nq:]))
    return (int(cols["bl_frame"][0]), b, d)


def result_columns(r: Result, S: Session, inputs: str, window_steps: int) -> list:
    """Result를 HKDIFF 열 목록으로. inputs: 'raw' | 'sig' | 'state' (어느 입력 열을 넣는가)."""
    n = len(r.c.get("t_ms", []))
    names = list(FED)
    cols = []
    for nm in names:
        cols.append(_arr(nm, r.c.get(nm, [])))
    if inputs == "raw":
        rc = raw_columns(S.frames[:n])
        for nm in RAWC:
            cols.append(_arr(nm, rc[nm]))
    if inputs in ("raw", "sig"):
        for nm in SIGC:
            cols.append(_arr(nm, r.c.get(nm, [])))
    for nm in (*STATEC, *EXTRAC):
        cols.append(_arr(nm, r.c.get(nm, [])))
    bc = r.meta.get("baseline_completed")
    if bc is not None and S.baseline is None:
        b = baseline_obj(bc["baseline"])
        cols += [("bl_frame", "u32", np.asarray([bc["accepted_index"]], dtype="<u4")),
                 ("bl_f64", "f64", np.asarray([bl_vec(b)], dtype="<f8")),
                 ("bl_t_complete_ms", "i64", np.asarray([b.t_complete_ms], dtype="<i8")),
                 ("bl_status", "u8", np.asarray([STATUSES.index(b.status)], dtype="<u1"))]
        d = r.meta.get("baseline_diag")
        if d is not None:
            cols += [("dg_n_face", "i64", np.asarray([d["n_face_frames"]], dtype="<i8")),
                     ("dg_f64", "f64", np.asarray([[d["face_ratio"], *d["blink_l_quantiles"],
                                                    *d["blink_r_quantiles"]]], dtype="<f8"))]
    for nm in OUTC:
        cols.append(_arr(nm, r.c.get(nm, []), window_steps))
    if S.stub is not None:
        cols.append(("stub_probs", "f64", np.asarray(S.stub, dtype="<f8").reshape((len(S.stub), 3))))
    return cols


OUT_COMPARE = (*EXTRAC[:3], *OUTC)  # 다시 돌린 출력에서 비교하는 열(여유 열은 RAW에서만)


def header_for(S: Session, r: Result, modes: list[str], gen_files: dict) -> dict:
    mm = r.meta.get("min_margin", {})
    mmin = {f"T{k}": hexf(mm.get(k)) for k in range(1, 9)}

    def label(ids):  # noqa: ANN001
        vals = [mm[k] for k in ids if k in mm]
        return "near" if vals and min(vals) < 1e-6 else "clear"
    pj = json.dumps(S.pb, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    rj = rules_to_json(rules_from_dict(S.rd))
    ph = compute_pipeline_hash(S.pb)
    rh = hashlib.sha256(rj.encode("utf-8")).hexdigest()
    spec = pipeline_spec_from_dict(S.pb)
    ref = reference()
    hdr = {
        "corpus": S.corpus, "session_id": S.sid, "klass": S.klass, "seed": S.seed, "modes": modes,
        "generator": {"name": S.generator, "files": gen_files},
        "reference": {"reference_id": ref["reference_id"], "modules": ref["modules"], "config": ref["config"],
                      "export": ref["export"], "helpers": ref["helpers"], "versions": ref["versions"]},
        "pipeline_json": pj, "rules_json": rj, "pipeline_json_sha256": ph, "rules_json_sha256": rh,
        "pipeline_hash": ph if ph == SHIPPED_PIPELINE_HASH else None,
        "rules_hash": rh if rh == SHIPPED_RULES_HASH else None,
        "baseline_injected": baseline_hex(S.baseline), "feature_start_ms": S.feature_start_ms,
        "model": None if S.stub is None else {"kind": "stub", "column": "stub_probs", "count": len(S.stub),
                                              "rule": "call i (0-based) returns row min(i, count - 1)"},
        "window_steps": spec.step_features.window_steps,
        "landmark_indices": list(LM44), "blendshape_names": list(BS11), "enums": ENUMS,
        "alert_priority": list(rules_from_dict(S.rd).alert_priority),
        "counters": r.meta.get("counters"), "constructor_error": r.meta.get("constructor_error"),
        "exception": r.meta.get("exception"),
        "n_fed": len(r.c.get("t_ms", [])), "n_accepted_rows": r.meta.get("n_accepted_rows", 0),
        "baseline_completed": None if r.meta.get("baseline_completed") is None else {
            "accepted_index": r.meta["baseline_completed"]["accepted_index"],
            "baseline": baseline_hex(r.meta["baseline_completed"]["baseline"])},
        "min_margin": mmin,
        "labels": {"RAW": label(range(1, 9)), "SIG": label(range(4, 9))},
        "active_at_end": r.meta.get("active_at_end"),
        "notes": S.notes,
    }
    return hdr


def unhex_baseline(b: dict | None) -> dict | None:
    if b is None:
        return None
    out = {k: float.fromhex(b[k]) for k in ("e_l", "e_r", "b_l", "b_r", "jaw0", "gaze_h0", "gaze_v0")}
    out["r0"] = [float.fromhex(v) for v in b["r0"]]
    out["t_complete_ms"] = b["t_complete_ms"]
    out["status"] = b["status"]
    return out


def verify_file(path: str) -> dict:
    """파일 하나만으로(생성기 없이) 세션을 다시 세워 참조를 파일의 모든 모드로 다시 돌리고 출력을 비트로 비교한다."""
    h, back = read_hk(path)
    if h["reference"]["reference_id"] != reference()["reference_id"]:
        raise AssertionError(f"{path}: reference_id differs")
    stub = None
    if "stub_probs" in back:
        stub = [tuple(float(x) for x in row) for row in back["stub_probs"]]
    S = Session(corpus=h["corpus"], sid=h["session_id"], klass=h["klass"], seed=h["seed"],
                pb=json.loads(h["pipeline_json"]), rd=json.loads(h["rules_json"]), kind="state", frames=[], speeds=[],
                baseline=unhex_baseline(h["baseline_injected"]), feature_start_ms=h["feature_start_ms"], stub=stub)
    if h["constructor_error"] is not None:
        r = run_session(S, "STATE")
        if r.meta["constructor_error"] != h["constructor_error"]:
            raise AssertionError(f"{path}: constructor error differs")
        return {"modes": [], "frames": 0}
    ws = h["window_steps"]
    done = []
    for m in h["modes"]:
        if m == "RAW":
            S.frames = frames_from_raw(back)
        elif m == "SIG":
            S.frames = signals_from_cols(back)
        else:
            S.frames = states_from_cols(back)
            S.state_baseline = baseline_from_cols(h, back)
        S.speeds = [None if back["speed_null"][i] else float(back["speed"][i]) for i in range(len(back["t_ms"]))]
        r2 = run_session(S, m, margins=(m == "RAW"))
        c2 = {nm: arr for nm, _dt, arr in result_columns(r2, S, "sig" if m == "RAW" else "state", ws)}
        names = [nm for nm in OUT_COMPARE if nm in back] + list(FED)
        if m != "STATE":
            names += [nm for nm in STATEC if nm in back]
        if m == "RAW":
            names += ["margin", "margin_id", "round_margin"] + [nm for nm in SIGC if nm in back]
        if m == "STATE" and h["exception"] is not None and not h["exception"].get("state_recorded"):
            names = [nm for nm in names if nm != "accepted"]
        bad = same_arrays({k: back[k] for k in names if k in back}, {k: c2[k] for k in names if k in c2})
        if json.dumps(r2.meta["counters"], sort_keys=True) != json.dumps(h["counters"], sort_keys=True):
            bad.append("counters")
        e1, e2 = h["exception"], r2.meta["exception"]
        if (e1 is None) != (e2 is None) or (e1 is not None and any(e1[k] != e2[k] for k in ("fed_index", "type",
                                                                                            "message", "where"))):
            bad.append("exception")
        b2 = r2.meta.get("baseline_completed")
        b2h = None if b2 is None else {"accepted_index": b2["accepted_index"], "baseline": baseline_hex(b2["baseline"])}
        if S.baseline is None and json.dumps(b2h, sort_keys=True) != json.dumps(h["baseline_completed"], sort_keys=True):
            bad.append("baseline_completed")
        if bad:
            raise AssertionError(f"{path}: verify in {m} differs: {bad[:6]}")
        done.append(m)
    return {"modes": done, "frames": len(back["t_ms"])}


def export_session(path: str, S: Session, gen_files: dict, check: bool = True) -> dict:
    """S를 참조에 돌려 path에 쓰고, 다시 읽어 비트로 비교하고, 다시 읽은 입력으로 참조를 다시 돌린다.

    돌려주는 것: 요약(dict). 자기 검사가 실패하면 AssertionError.
    """
    mode = {"raw": "RAW", "sig": "SIG", "state": "STATE"}[S.kind]
    r = run_session(S, mode, margins=True)
    modes = {"raw": ["RAW", "SIG", "STATE"], "sig": ["SIG", "STATE"], "state": ["STATE"]}[S.kind]
    if S.corpus.startswith("S-") and S.kind == "raw":
        modes = ["STATE"]
    inputs = {"RAW": "raw", "SIG": "sig", "STATE": "state"}[modes[0]]
    ws = pipeline_spec_from_dict(S.pb).step_features.window_steps
    cols = result_columns(r, S, inputs, ws)
    hdr = header_for(S, r, modes, gen_files)
    arrays = write_hk(path, hdr, cols)
    summary = {"sid": S.sid, "path": os.path.relpath(path, DATA), "n_fed": hdr["n_fed"],
               "n_accepted": hdr["counters"]["frames_accepted"] if hdr["counters"] else 0,
               "labels": hdr["labels"], "exception": hdr["exception"] is not None,
               "constructor_error": hdr["constructor_error"] is not None, "bytes": os.path.getsize(path),
               "min_margin": r.meta.get("min_margin", {}), "selfcheck": None}
    if not check:
        return summary
    h2, back = read_hk(path)
    bad = same_arrays(arrays, back)
    if bad or h2 != json.loads(json.dumps({**hdr, **{k: h2[k] for k in ("format", "format_version", "columns",
                                                                         "data_nbytes", "data_sha256")}})):
        raise AssertionError(f"{S.sid}: read-back differs: {bad[:5]}")
    if hdr["constructor_error"] is not None:
        summary["selfcheck"] = "constructor error, nothing to re-run"
        return summary
    reruns = []
    for m in modes:
        S2 = Session(**{**S.__dict__})
        if m == "RAW":
            S2.frames = frames_from_raw(back)
            S2.kind = "raw"
        elif m == "SIG":
            S2.frames = signals_from_cols(back)
            S2.kind = "sig"
        else:
            S2.frames = states_from_cols(back)
            S2.kind = "state"
            S2.state_baseline = baseline_from_cols(h2, back)
        S2.speeds = [None if back["speed_null"][i] else float(back["speed"][i]) for i in range(len(back["t_ms"]))]
        r2 = run_session(S2, m, margins=(m == "RAW"))
        c2 = {nm: arr for nm, _dt, arr in result_columns(r2, S2, "sig" if m == "RAW" else "state", ws)}
        names = [nm for nm in OUT_COMPARE if nm in back] + list(FED)
        if m != "STATE":
            names += [nm for nm in STATEC if nm in back]
        if m == "RAW":
            names += ["margin", "margin_id", "round_margin"] + [nm for nm in SIGC if nm in back]
        if m == "STATE" and r.meta["exception"] is not None and not r.meta["exception"].get("state_recorded"):
            names = [nm for nm in names if nm != "accepted"]
        bad = same_arrays({k: back[k] for k in names if k in back}, {k: c2[k] for k in names if k in c2})
        for key in ("counters", "exception", "baseline_completed"):
            a1, a2 = r.meta.get(key), r2.meta.get(key)
            if key == "exception" and a1 is not None and a2 is not None:
                a1 = {k: a1[k] for k in ("fed_index", "type", "message", "where")}
                a2 = {k: a2[k] for k in ("fed_index", "type", "message", "where")}
            if json.dumps(a1, sort_keys=True) != json.dumps(a2, sort_keys=True):
                bad.append(key)
        if bad:
            raise AssertionError(f"{S.sid}: re-run in {m} differs: {bad[:6]}")
        reruns.append(m)
    summary["selfcheck"] = "read-back bit-equal; re-run identical in " + ",".join(reruns)
    return summary
