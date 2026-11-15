"""내보내기 자체의 회귀 검사(파일을 쓰지 않는다). `export.py small`이 맨 먼저 돌리고, 실패하면 멈춘다.

검사 1~3: T2, T3 여유(FORMATS 1.7절)는 참조가 그 마감에서 실제로 한 비교만 적는다. 기대는 참조의 결과에서 온다:
T3은 참조가 그 마감에서 개인 기준선(ok 또는 off_axis)을 만들었을 때만 있다. T2는 참조의 `and` 짧은 평가가 멈춘 곳까지만 있다.
"""
from __future__ import annotations

import math

import hk

RT = hk.RT


def _sigs(n_ms: int, face_until_ms: int | None, ear_l: float, ear_r: float, yaw: float) -> list:
    out = []
    for k in range(n_ms // 100):
        t = 1 + 100 * k
        face = face_until_ms is None or 100 * k < face_until_ms
        if face:
            out.append(RT.FrameSignals(t_ms=t, face=True, ear_l=ear_l, ear_r=ear_r, eye_w_l=40.0, eye_w_r=40.0,
                                       blink_l=0.1, blink_r=0.1, eye_look=(0.0,) * 8, jaw_open=0.05, yaw=yaw,
                                       pitch=0.0, roll=0.0))
        else:
            out.append(RT.FrameSignals(t_ms=t, face=False, ear_l=0.0, ear_r=0.0, eye_w_l=0.0, eye_w_r=0.0,
                                       blink_l=0.0, blink_r=0.0, eye_look=(0.0,) * 8, jaw_open=0.0, yaw=0.0,
                                       pitch=0.0, roll=0.0))
    return out


def _run(name: str, sigs: list) -> hk.Result:
    S = hk.Session(corpus="selftest", sid=f"selftest/{name}", klass=name, seed=0, pb=hk.PIPE_BLOCK, rd=hk.RULES_DOC,
                   kind="sig", frames=sigs, speeds=[None] * len(sigs))
    return hk.run_session(S, "SIG", margins=True)


def run() -> list[str]:
    cfg = hk.pipeline_spec_from_dict(hk.PIPE_BLOCK).baseline
    lines = []
    # 1. 60초에 완성(개인 기준선), 모든 yaw가 정확히 off_axis_yaw_deg: T3 = 0
    r = _run("t3_complete", _sigs(70000, None, 0.27, 0.27, cfg.off_axis_yaw_deg))
    bc = r.meta["baseline_completed"]
    mm = r.meta.get("min_margin", {})
    if bc is None or bc["baseline"]["status"] not in (RT.BASELINE_OK, RT.BASELINE_OFF_AXIS) or mm.get(3) != 0.0:
        raise AssertionError(f"selftest t3_complete: baseline {bc and bc['baseline']['status']}, T3 {mm.get(3)}")
    lines.append(f"t3_complete: reference completed at {bc['baseline']['t_complete_ms']} ms with status "
                 f"{bc['baseline']['status']}; T3 {mm.get(3)}")
    # 2. 얼굴 시간 비율이 모자라 60초에 연장되고 세션이 90초 전에 끝난다: 참조는 yaw를 비교하지 않았다, T3 없음
    r = _run("t3_extended", _sigs(70000, 25000, 0.27, 0.27, cfg.off_axis_yaw_deg))
    mm = r.meta.get("min_margin", {})
    if r.meta["baseline_completed"] is not None or 3 in mm or 2 not in mm:
        raise AssertionError(f"selftest t3_extended: baseline {r.meta['baseline_completed']}, margins {mm}")
    lines.append(f"t3_extended: reference made no baseline; T3 absent, T2 {mm.get(2)}")
    # 3. e_l < e_min: 짧은 평가가 첫 비교에서 멈춘다. e_r를 e_min 한 ulp 위에 두어도 T2는 e_l의 여유뿐이다
    e_l, e_r = 0.05, math.nextafter(cfg.e_min, 1.0)
    r = _run("t2_short_circuit", _sigs(70000, None, e_l, e_r, 0.0))
    mm = r.meta.get("min_margin", {})
    if mm.get(2) != abs(e_l - cfg.e_min) or 3 in mm:
        raise AssertionError(f"selftest t2_short_circuit: margins {mm}")
    lines.append(f"t2_short_circuit: T2 {mm.get(2)} (= |e_l - e_min|), T3 absent")
    # 4. 대조: 넷 모두 범위 안이면 네 비교 모두 한다
    r = _run("t2_all_four", _sigs(70000, None, 0.27, 0.44, 0.0))
    mm = r.meta.get("min_margin", {})
    want = min(abs(0.27 - cfg.e_min), abs(0.27 - cfg.e_max), abs(0.44 - cfg.e_min), abs(0.44 - cfg.e_max))
    if mm.get(2) != want:
        raise AssertionError(f"selftest t2_all_four: T2 {mm.get(2)}, want {want}")
    lines.append(f"t2_all_four: T2 {mm.get(2)} (= |e_r - e_max|)")
    for ln in lines:
        print(f"  selftest {ln}", flush=True)
    return []
