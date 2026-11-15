"""시작 검사 변조표 (계획 9.4절, E10). 기대 코드는 계획 9.4절의 것이고, 파일과 사실(해시, 읽힘 여부)은 Python이 만든다."""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np

import hk
from small import gen_info, write_json

BASE = {"task": "models/face_landmarker.task", "onnx": "export/drowsiness_v0.onnx",
        "feature_spec": "export/feature_spec_v0.json", "rules": "export/rules.json"}
NAMES = {"task": "face_landmarker.task", "onnx": "drowsiness_v0.onnx", "feature_spec": "feature_spec_v0.json",
         "rules": "rules.json"}
TD = os.path.join(hk.DATA, "tamper")


def rb(rel: str) -> bytes:
    with open(os.path.join(hk.REPO, rel), "rb") as fh:
        return fh.read()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def base_metadata() -> dict:
    import onnx
    m = onnx.load(os.path.join(hk.REPO, BASE["onnx"]))
    return {p.key: p.value for p in m.metadata_props}


def bump_hex(s: str, i: int = 5) -> str:
    c = s[i]
    n = "0123456789abcdef"[("0123456789abcdef".index(c) + 1) % 16]
    return s[:i] + n + s[i + 1:]


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def spec_facts(b: bytes) -> dict:
    from drowsy.spec import SpecError, compute_pipeline_hash, feature_spec_from_dict
    f = {"sha256": sha(b), "bytes": len(b)}
    try:
        d = json.loads(b.decode("utf-8"))
        f["parses"] = True
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        f["parses"] = False
        f["parse_error"] = f"{type(exc).__name__}: {exc}"
        return f
    f["pipeline_hash_string"] = d.get("pipeline_hash")
    try:
        f["pipeline_block_hashes_to"] = compute_pipeline_hash(d["pipeline"])
        f["mediapipe_model_sha256_string"] = d["pipeline"]["mediapipe"]["model_sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        f["pipeline_block_hashes_to"] = None
        f["pipeline_block_error"] = str(exc)
    try:
        feature_spec_from_dict(d)
        f["drowsy_spec_accepts"] = True
    except (SpecError, ValueError, TypeError, KeyError) as exc:
        f["drowsy_spec_accepts"] = False
        f["drowsy_spec_error"] = f"{type(exc).__name__}: {exc}"
    return f


def rules_facts(b: bytes) -> dict:
    from drowsy.spec import SpecError, rules_from_dict
    f = {"sha256": sha(b), "bytes": len(b)}
    try:
        d = json.loads(b.decode("utf-8"))
        f["parses"] = True
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        f["parses"] = False
        f["parse_error"] = f"{type(exc).__name__}: {exc}"
        return f
    try:
        r = rules_from_dict(d)
        f["drowsy_spec_accepts"] = True
        f["clock"] = {"step_ms": r.clock.step_ms, "max_hold_ms": r.clock.max_hold_ms, "reset_gap_ms": r.clock.reset_gap_ms}
    except (SpecError, ValueError, TypeError, KeyError) as exc:
        f["drowsy_spec_accepts"] = False
        f["drowsy_spec_error"] = f"{type(exc).__name__}: {exc}"
    return f


def onnx_facts(b: bytes, window: np.ndarray) -> dict:
    import onnx
    import onnxruntime as ort
    f = {"sha256": sha(b), "bytes": len(b)}
    try:
        m = onnx.load_from_string(b)
        f["onnx_loads"] = True
        f["metadata"] = {p.key: p.value for p in m.metadata_props}
        f["inputs"] = [[i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]] for i in m.graph.input]
        f["outputs"] = [[i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]] for i in m.graph.output]
    except Exception as exc:  # noqa: BLE001
        f["onnx_loads"] = False
        f["onnx_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        s = ort.InferenceSession(b, o, providers=["CPUExecutionProvider"])
        f["onnxruntime_loads"] = True
        try:
            out = s.run(None, {s.get_inputs()[0].name: window})[0][0]
            f["probs_on_e2e_window"] = [float(x) for x in out]
        except Exception as exc:  # noqa: BLE001
            f["run_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    except Exception as exc:  # noqa: BLE001
        f["onnxruntime_loads"] = False
        f["onnxruntime_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return f


def tamper() -> list[str]:
    import onnx
    shutil.rmtree(TD, ignore_errors=True)
    os.makedirs(TD)
    files_out = []
    base_bytes = {k: rb(v) for k, v in BASE.items()}
    meta = base_metadata()
    assert sha(base_bytes["task"]) == meta["mediapipe_model_sha256"]
    with open(os.path.join(hk.REPO, "tests", "vectors", "e2e_250s.json"), encoding="utf-8") as fh:
        w0 = np.asarray(json.load(fh)["expected"]["model_inputs"][0]["features"], dtype=np.float32).reshape(1, 120, 17)
    spec_text = base_bytes["feature_spec"].decode("utf-8")
    rules_text = base_bytes["rules"].decode("utf-8")
    cases = []

    def put(cid: str, role: str, data: bytes, name: str | None = None) -> str:
        d = os.path.join(TD, cid)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name or NAMES[role])
        with open(p, "wb") as fh:
            fh.write(data)
        files_out.append(p)
        return os.path.relpath(p, hk.DATA)

    def patch_of(old: bytes, new: bytes) -> dict | None:
        if len(old) != len(new):
            return None
        diff = [i for i in range(len(old)) if old[i] != new[i]]
        if len(diff) != 1:
            return None
        i = diff[0]
        return {"offset": i, "old_byte": old[i], "new_byte": new[i]}

    def case(cid, plan_case, change, codes, files=None, missing=None, metadata=None, verified=None, facts=None,  # noqa: ANN001
             note=None):
        files = files or {}
        c = {"id": cid, "plan_case": plan_case, "change": change, "dir": f"tamper/{cid}" if files else None,
             "files": files, "missing": missing or [], "metadata": metadata, "verified_override": verified,
             "expected_codes": codes, "expected_codes_source": "plan 9.4", "facts": facts or {}}
        if note:
            c["note"] = note
        cases.append(c)

    # SA1: task 파일 한 바이트
    t = base_bytes["task"]
    for where, i in (("first", 0), ("middle", len(t) // 2), ("last", len(t) - 1)):
        nb = bytearray(t)
        nb[i] ^= 0x01
        rel = put(f"SA1_{where}", "task", bytes(nb))
        case(f"SA1_{where}", "SA1", f"face_landmarker.task byte {i} xor 0x01", ["H1_TASK_FILE_VS_METADATA"],
             files={"task": rel}, facts={"task_sha256": sha(bytes(nb)), "patch": patch_of(t, bytes(nb))})

    def spec_case(cid, plan_case, change, codes, new_text=None, new_bytes=None, metadata=None, note=None):  # noqa: ANN001
        nb = new_bytes if new_bytes is not None else new_text.encode("utf-8")
        rel = put(cid, "feature_spec", nb)
        facts = spec_facts(nb)
        facts["patch"] = patch_of(base_bytes["feature_spec"], nb)
        case(cid, plan_case, change, codes, files={"feature_spec": rel}, metadata=metadata, facts=facts, note=note)
        return nb

    spec_case("SA2", "SA2", "feature_spec_v0.json: standardization.z_clip 5.0 -> 6.0 (one byte)",
              ["H2_SPEC_FILE_VS_METADATA"], replace_once(spec_text, '"z_clip": 5.0', '"z_clip": 6.0'))
    spec_case("SA3_newline", "SA3", "feature_spec_v0.json: one more trailing LF", ["H2_SPEC_FILE_VS_METADATA"],
              spec_text + "\n")
    spec_case("SA3_crlf", "SA3", "feature_spec_v0.json: every LF -> CRLF", ["H2_SPEC_FILE_VS_METADATA"],
              spec_text.replace("\n", "\r\n"))
    spec_case("SA3b_bom", "SA3b", "feature_spec_v0.json: UTF-8 BOM in front",
              ["H2_SPEC_FILE_VS_METADATA", "SPEC_INVALID"], new_bytes=b"\xef\xbb\xbf" + base_bytes["feature_spec"],
              note="Python json.loads of the decoded text refuses a BOM too (facts.parses)")
    ph = meta["pipeline_hash"]
    spec_case("SA4", "SA4", "feature_spec_v0.json: one hex digit of pipeline_hash",
              ["H2_SPEC_FILE_VS_METADATA", "H3_SPEC_VS_METADATA", "H3_SPEC_VS_VERIFIED"],
              replace_once(spec_text, f'"pipeline_hash": "{ph}"', f'"pipeline_hash": "{bump_hex(ph)}"'))
    ms = meta["mediapipe_model_sha256"]
    spec_case("SA5", "SA5", "feature_spec_v0.json: one hex digit of pipeline.mediapipe.model_sha256",
              ["H2_SPEC_FILE_VS_METADATA", "H1_SPEC_VS_METADATA"],
              replace_once(spec_text, f'"model_sha256": "{ms}"', f'"model_sha256": "{bump_hex(ms)}"'))
    i = spec_text.index('"spec_version"')
    j = spec_text.index(":", i)
    spec_case("SA6", "SA6", "feature_spec_v0.json: the ':' after \"spec_version\" -> ';' (does not parse)",
              ["H2_SPEC_FILE_VS_METADATA", "SPEC_INVALID"], spec_text[:j] + ";" + spec_text[j + 1:])
    sa6b = spec_case("SA6b", "SA6b", "feature_spec_v0.json: pipeline.baseline.e_min 0.12 -> 0.13, pipeline_hash string "
                     "unchanged", ["H2_SPEC_FILE_VS_METADATA", "CONSTANT_BITS_VS_VERIFIED"],
                     replace_once(spec_text, '"e_min": 0.12', '"e_min": 0.13'),
                     note="drowsy.spec refuses this file (pipeline_hash mismatch, facts); the Kotlin check does not "
                          "recompute the hash (SPEC 15) and refuses it through H2 and the constant-bits fingerprint")
    rel = os.path.relpath(os.path.join(TD, "SA6b", NAMES["feature_spec"]), hk.DATA)
    case("SA6c", "SA6c", "SA6b file, metadata feature_spec_hash set to its sha256", ["CONSTANT_BITS_VS_VERIFIED"],
         files={"feature_spec": rel}, metadata={**meta, "feature_spec_hash": sha(sa6b)}, facts=spec_facts(sa6b),
         note="H1-H5 pass here (verify_design 7.4 item 2); only the constant-bits fingerprint refuses it")

    def rules_case(cid, plan_case, change, codes, new_text, verified=None, note=None):  # noqa: ANN001
        nb = new_text.encode("utf-8")
        rel = put(cid, "rules", nb)
        facts = rules_facts(nb)
        facts["patch"] = patch_of(base_bytes["rules"], nb)
        case(cid, plan_case, change, codes, files={"rules": rel}, facts=facts,
             verified=verified if verified is None else {k: (sha(nb) if v == "@sha" else v) for k, v in verified.items()},
             note=note)

    rules_case("SA7", "SA7", "rules.json: forward_zone.yaw_abs_max_deg 23.0 -> 24.0 (one byte)",
               ["H4_RULES_FILE_VS_VERIFIED", "CONSTANT_BITS_VS_VERIFIED"],
               replace_once(rules_text, '"yaw_abs_max_deg": 23.0', '"yaw_abs_max_deg": 24.0'))
    rules_case("SA7b_newline", "SA7b", "rules.json: one more trailing LF", ["H4_RULES_FILE_VS_VERIFIED"],
               rules_text + "\n")
    rules_case("SA7b_crlf", "SA7b", "rules.json: every LF -> CRLF", ["H4_RULES_FILE_VS_VERIFIED"],
               rules_text.replace("\n", "\r\n"))
    sa8 = replace_once(rules_text, '"max_hold_ms": 200', '"max_hold_ms": 250')
    rules_case("SA8", "SA8", "rules.json: clock.max_hold_ms 200 -> 250; the verified rules hash and the constant-bits "
               "fingerprint are taken from this file", ["CLOCK_SPEC_VS_RULES"], sa8,
               verified={"rules_hash": "@sha", "constant_bits": "compute from the files in use"})

    # SB: 메타데이터 맵
    def meta_case(cid, plan_case, change, codes, md, note=None):  # noqa: ANN001
        case(cid, plan_case, change, codes, metadata=md, note=note)

    meta_case("SB1", "SB1", "metadata mediapipe_model_sha256: one hex digit", ["H1_TASK_FILE_VS_METADATA",
              "H1_SPEC_VS_METADATA"], {**meta, "mediapipe_model_sha256": bump_hex(ms)})
    meta_case("SB2", "SB2", "metadata feature_spec_hash: one hex digit", ["H2_SPEC_FILE_VS_METADATA"],
              {**meta, "feature_spec_hash": bump_hex(meta["feature_spec_hash"])})
    meta_case("SB3", "SB3", "metadata pipeline_hash: one hex digit", ["H3_SPEC_VS_METADATA"],
              {**meta, "pipeline_hash": bump_hex(ph)})
    meta_case("SB4", "SB4", "metadata step_sec = 2", ["H5_STEP_SEC"], {**meta, "step_sec": "2"})
    meta_case("SB5", "SB5", "metadata window_steps = 60", ["H5_WINDOW_STEPS"], {**meta, "window_steps": "60"})
    meta_case("SB6", "SB6", "metadata class_order = alert,drowsy,low_vigilant", ["H5_CLASS_ORDER"],
              {**meta, "class_order": "alert,drowsy,low_vigilant"})
    meta_case("SB7", "SB7", "metadata standardization = app", ["H5_STANDARDIZATION"], {**meta, "standardization": "app"})
    for key in ("mediapipe_model_sha256", "feature_spec_hash", "pipeline_hash", "step_sec", "window_steps",
                "class_order", "standardization", "model_version"):
        md = {k: v for k, v in meta.items() if k != key}
        meta_case(f"SB8_{key}", "SB8", f"metadata key {key} removed", ["METADATA_KEY_MISSING"], md,
                  note=f"the problem's subject is {key}")
    meta_case("SB9_upper_mediapipe", "SB9", "metadata mediapipe_model_sha256 in upper case",
              ["H1_TASK_FILE_VS_METADATA", "H1_SPEC_VS_METADATA"], {**meta, "mediapipe_model_sha256": ms.upper()})
    meta_case("SB9_upper_feature_spec", "SB9", "metadata feature_spec_hash in upper case", ["H2_SPEC_FILE_VS_METADATA"],
              {**meta, "feature_spec_hash": meta["feature_spec_hash"].upper()})
    meta_case("SB9_upper_pipeline", "SB9", "metadata pipeline_hash in upper case", ["H3_SPEC_VS_METADATA"],
              {**meta, "pipeline_hash": ph.upper()})
    meta_case("SB9_space", "SB9", "metadata window_steps = '120 '", ["H5_WINDOW_STEPS"], {**meta, "window_steps": "120 "})
    meta_case("SB9_float", "SB9", "metadata step_sec = '1.0'", ["H5_STEP_SEC"], {**meta, "step_sec": "1.0"})
    meta_case("SB9_spaces", "SB9", "metadata class_order = 'alert, low_vigilant, drowsy'", ["H5_CLASS_ORDER"],
              {**meta, "class_order": "alert, low_vigilant, drowsy"})
    # SB10: 다른 pipeline_hash를 가진 spec, 메타데이터를 맞춤
    from drowsy.spec import compute_pipeline_hash
    d = json.loads(spec_text)
    d["pipeline"]["baseline"]["e_min"] = 0.13
    new_ph = compute_pipeline_hash(d["pipeline"])
    t10 = replace_once(replace_once(spec_text, '"e_min": 0.12', '"e_min": 0.13'), f'"pipeline_hash": "{ph}"',
                       f'"pipeline_hash": "{new_ph}"')
    nb10 = t10.encode("utf-8")
    rel = put("SB10", "feature_spec", nb10)
    f10 = spec_facts(nb10)
    assert f10["drowsy_spec_accepts"], f10
    case("SB10", "SB10", "spec with pipeline.baseline.e_min 0.13 and its recomputed pipeline_hash; metadata "
         "pipeline_hash and feature_spec_hash set to match", ["H3_SPEC_VS_VERIFIED", "CONSTANT_BITS_VS_VERIFIED"],
         files={"feature_spec": rel}, metadata={**meta, "pipeline_hash": new_ph, "feature_spec_hash": sha(nb10)},
         facts=f10)
    meta_case("SB11_trained_on", "SB11", "metadata trained_on changed", [], {**meta, "trained_on": "something else"},
              note="display value; no problem expected")
    meta_case("SB11_commit", "SB11", "metadata commit changed", [], {**meta, "commit": "0123abc"},
              note="display value; no problem expected")

    # SC: 실제 ONNX 파일
    ob = base_bytes["onnx"]
    pos = ob.find(ph.encode("ascii"))
    assert pos > 0 and ob.find(ph.encode("ascii"), pos + 1) < 0
    nb = bytearray(ob)
    nb[pos + 5] = ord(bump_hex(ph)[5])
    nb = bytes(nb)
    rel = put("SC1", "onnx", nb)
    f = onnx_facts(nb, w0)
    assert f["metadata"]["pipeline_hash"] == bump_hex(ph)
    f["patch"] = patch_of(ob, nb)
    case("SC1", "SC1", "drowsiness_v0.onnx: one byte inside the metadata value pipeline_hash",
         ["H3_SPEC_VS_METADATA"], files={"onnx": rel}, facts=f)
    m = onnx.load_from_string(ob)
    init = next(x for x in m.graph.initializer if x.name == "net.head.weight")
    raw = init.raw_data
    pos = ob.find(raw)
    assert pos > 0 and ob.find(raw, pos + 1) < 0
    nb = bytearray(ob)
    nb[pos + 1] ^= 0x01  # 첫 float32의 둘째 바이트(가수의 아래쪽)
    nb = bytes(nb)
    rel = put("SC2", "onnx", nb)
    f = onnx_facts(nb, w0)
    f["patch"] = patch_of(ob, nb)
    f["base_probs_on_e2e_window"] = onnx_facts(ob, w0).get("probs_on_e2e_window")
    case("SC2", "SC2", "drowsiness_v0.onnx: one byte inside the weight net.head.weight", [], files={"onnx": rel},
         facts=f, note="K12 (가): accepted when no pin is given (report it by name as an accepted tampering); with "
                       "a pin (sha256 of the base file) the code is MODEL_FILE_VS_PIN")
    nb = bytearray(ob)
    nb[0] = 0xFF
    nb = bytes(nb)
    rel = put("SC3", "onnx", nb)
    f = onnx_facts(nb, w0)
    f["patch"] = patch_of(ob, nb)
    assert not f["onnxruntime_loads"], f
    case("SC3", "SC3", "drowsiness_v0.onnx: first byte (protobuf tag) -> 0xFF", ["MODEL_UNREADABLE"],
         files={"onnx": rel}, facts=f)
    # SD
    for role in ("task", "onnx", "feature_spec", "rules"):
        case(f"SD1_{role}", "SD1", f"{NAMES[role]} missing", ["FILE_MISSING"], missing=[role],
             note=f"the problem's subject is {NAMES[role]}")
    d2 = os.path.join(TD, "SD2")
    os.makedirs(d2)
    for role, name in (("onnx", "drowsiness_v1.onnx"), ("feature_spec", "feature_spec_v1.json"),
                       ("rules", "rules.json")):
        put("SD2", role, base_bytes[role], name)
    case("SD2", "SD2", "v0 model copied to drowsiness_v1.onnx, spec to feature_spec_v1.json (metadata model_version 0); "
         "rules.json as shipped, task from the repository", ["MODEL_VERSION_VS_FILE_NAME"],
         files={"onnx": "tamper/SD2/drowsiness_v1.onnx", "feature_spec": "tamper/SD2/feature_spec_v1.json",
                "rules": "tamper/SD2/rules.json"})
    m = onnx.load_from_string(ob)
    m.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 60
    nb = m.SerializeToString()
    rel = put("SD3", "onnx", nb)
    f = onnx_facts(nb, w0[:, :60, :].copy())
    assert f["metadata"] == meta and f["inputs"][0][1] == [1, 60, 17], f
    case("SD3", "SD3", "drowsiness_v0.onnx re-serialised by onnx 1.23.2 with input shape [1, 60, 17], metadata unchanged",
         ["MODEL_IO_CONTRACT"], files={"onnx": rel}, facts=f, note="H5_WINDOW_STEPS must not appear")
    full = os.path.join(TD, "SD4_SA4_bundle")
    os.makedirs(full)
    for role in ("task", "onnx", "rules"):
        put("SD4_SA4_bundle", role, base_bytes[role])
    shutil.copy(os.path.join(TD, "SA4", NAMES["feature_spec"]), os.path.join(full, NAMES["feature_spec"]))
    files_out.append(os.path.join(full, NAMES["feature_spec"]))
    case("SD4", "SD4", "public path openReplay(BundleSource.directory(dir)) on a complete bundle whose spec is the SA4 "
         "file", ["H2_SPEC_FILE_VS_METADATA", "H3_SPEC_VS_METADATA", "H3_SPEC_VS_VERIFIED"],
         files={r: f"tamper/SD4_SA4_bundle/{NAMES[r]}" for r in NAMES},
         note="expect MonitorStartException(StartupRejected) carrying exactly this code set")
    case("SD5", "SD5", "DriveSession.create with a StartupResult that did not pass (any refused case, e.g. SB3)",
         ["IllegalArgumentException"], note="not a start-up code: the exception type expected from DriveSession.create")
    for c in cases:  # 쓰는 파일의 sha256
        use = {}
        for role in NAMES:
            if role in c["missing"]:
                use[role] = None
            elif role in c["files"]:
                with open(os.path.join(hk.DATA, c["files"][role]), "rb") as fh:
                    use[role] = sha(fh.read())
            else:
                use[role] = sha(base_bytes[role])
        c["facts"]["sha256_in_use"] = use
    p = os.path.join(TD, "manifest.json")
    write_json(p, {"reference": hk.reference(), "generator": gen_info("small.py", "tamper.py"),
                   "base": {r: {"path": BASE[r], "sha256": sha(base_bytes[r]), "bytes": len(base_bytes[r])} for r in BASE},
                   "base_metadata": meta,
                   "verified": {"pipeline_hash": hk.SHIPPED_PIPELINE_HASH, "rules_hash": hk.SHIPPED_RULES_HASH},
                   "cases": cases})
    print(f"tamper: {len(cases)} cases, {len(files_out)} files; SC2 probs base "
          f"{cases[[c['id'] for c in cases].index('SC2')]['facts']['base_probs_on_e2e_window']} -> "
          f"{cases[[c['id'] for c in cases].index('SC2')]['facts'].get('probs_on_e2e_window')}")
    return files_out + [p]
