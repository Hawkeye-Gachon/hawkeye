# Differential data: file formats (`kotlin/.diffdata/`)

Producer: the Python exporter in `kotlin/difftools/` (stage 5 of the plan, experiment E1, decision K23 (가)).
Every expected value in these files comes from the **unchanged** Python reference (`drowsy/` seven ported modules)
run by the exporter. Nothing here was typed by hand from SPEC.

Status of this document: current for format version 1. If a file changes, this document changes in the same step.

## 0. Common rules

### 0.1 Commands (run from the repository root)

```
export ORT_DISABLE_TELEMETRY=1
nice -n 19 .venv/bin/python kotlin/difftools/export.py small            # small files, then READY_small.txt
nice -n 19 .venv/bin/python kotlin/difftools/export.py corpora --workers 10   # S-quick, S-odd, S-edge, F-*, G-sig, READY_corpora.txt
nice -n 19 .venv/bin/python kotlin/difftools/export.py full --workers 10      # S-full (12,000 sessions), READY_full.txt
nice -n 19 .venv/bin/python kotlin/difftools/export.py check            # re-hash every file listed in the READY files
nice -n 19 .venv/bin/python kotlin/difftools/export.py verify --workers 10    # re-read every session file and re-run the
                                                                         # reference from the file alone, all modes
```

`small` takes about 25 s, `corpora` about 35 s, `full` about 70 s with 10 workers on this PC (measured 2026-11-15;
section 5). Each command deletes and rewrites what it produces.

Each subcommand also accepts `--only NAME[,NAME...]` (the names in the section headings below, for example
`--only canary,comparator`). Every run writes its log to stdout; the agent keeps the logs of reported runs in its
durable folder. Generation is deterministic: the same reference and the same generator files give byte-identical
files (no time stamps or host names inside any file).

### 0.2 Reference fingerprint (refuse data made from another reference)

Every HKDIFF1 header and every JSON file has a `reference` object:

```
reference.modules    {"drowsy/types.py": sha256, ... 7 ported modules}
reference.config     {"config/feature_spec.json": sha256, "config/rules.yaml": sha256}
reference.export     {"export/feature_spec_v0.json": ..., "export/rules.json": ..., "export/drowsiness_v0.onnx": ...}
reference.helpers    {"drowsy/spec.py", "drowsy/synth.py", "drowsy/vectors.py", "drowsy/demo.py"}  (generators; informative)
reference.reference_id   sha256 of the ASCII text made of one line "<path>:<sha256>\n" per file of modules, config,
                         export (in that group order, each group sorted by path)
reference.versions   {"python", "numpy", "onnxruntime", "onnx"}
```

A Kotlin test computes the sha256 of the 12 files of `modules`, `config`, `export` in the repository, builds
`reference_id` the same way and refuses (fails, does not skip) any data file whose `reference_id` differs. The
same id is in every `READY_*.txt`.

`depends_on`: a JSON file whose expected values come from a module outside `reference_id` has a top-level
`depends_on: {path in repo: sha256}`. Today: `components/C-loader.json` (`drowsy/spec.py`: the accept/refuse
decisions and messages) and `numbers/comparator.json` (`drowsy/vectors.py`: `python_ok`, `python_problems`). A
Kotlin test that uses such a file also refuses (fails) it when any listed file's sha256 differs from the repository;
the `reference_id` check alone does not cover these files. `export.py check` checks `depends_on` too.

### 0.3 Numbers

- HKDIFF1 binary: little-endian, IEEE-754 binary64 (`f64`) / binary32 (`f32`) exactly as computed, integers
  two's complement (`i32`, `i64`) or unsigned (`u8`, `u32`). Arrays are C order (last index fastest).
- JSON files: floats are JSON numbers written by Python `json` (`repr`, the shortest text that round-trips), so
  `-0.0` is written `-0.0`; integers are JSON integers (may exceed 2^31; parse as `Long`). JSON files never contain
  NaN or infinity unless a section says so (then they are the strings `"NaN"`, `"Infinity"`, `"-Infinity"`).
- Wherever a header field is named `*_hex` or documented as "hex", it is Python `float.hex()` text, for example
  `0x1.0000000000000p+0`, `-0x0.0p+0`, `0x0.0000000000001p-1022`. `java.lang.Double.parseDouble` reads this form.
  Python's and Java's hex *writing* differ (`0x1.0p0` from `Double.toHexString`), so compare parsed bits, never text.

### 0.4 Enum tables (also in every HKDIFF1 header under `enums`)

| table | values (index = position) |
|---|---|
| `event_types` | calibrating, unable_to_analyze, gaze_away_long, vats, drowsy, microsleep, sleep |
| `phases` | onset, end |
| `sources` | face, head_pose, eye, model, perclos, closure |
| `baseline_statuses` | ok, off_axis, fallback_global |
| `zones` | in, out, unknown |

`active_mask` bit `i` (value `1 << i`) = event type `event_types[i]` is active.

### 0.5 READY files

`READY_small.txt`, `READY_corpora.txt`, `READY_full.txt` in `.diffdata/`: first lines `# key: value` (reference_id,
command, wall time of the run that wrote them), then one line per file: `<sha256>  <bytes>  <path relative to .diffdata>`.
A corpus directory is listed by its `index.json` (which lists every session file with its sha256) plus one line
`<sha256 of index.json>  <total bytes of the directory>  <dir>/`. A Kotlin test should check `index.json` against
the READY line and each session file against `index.json` before use.

## 1. HKDIFF1 session files (`*.hkd`)

### 1.1 Container

```
bytes 0..7      ASCII "HKDIFF1\n"
bytes 8..11     uint32 LE: header length H
bytes 12..12+H  UTF-8 JSON header (no BOM)
zero bytes      up to the next multiple of 8 = data start D
bytes D..EOF    data region: columns at header.columns[i].offset (relative to D), each 8-aligned, zero padded
```

`header.data_nbytes` = EOF - D; `header.data_sha256` = sha256 of bytes D..EOF. A reader must check both.
`header.columns[i]` = `{name, dtype, shape, offset, nbytes}`; dtype in `u8 i32 u32 i64 f64 f32`; an empty column
has a 0 in its first dimension and `nbytes` 0. Columns not listed are absent (not empty).

### 1.2 Header fields

| field | meaning |
|---|---|
| `format`, `format_version` | `"HKDIFF1"`, `1` |
| `corpus`, `session_id`, `klass`, `seed` | `session_id` = `<corpus>/<klass>/<seed>` |
| `modes` | replay modes this file supports, subset of `RAW`, `SIG`, `STATE` (section 1.5) |
| `generator` | `{name, files: {path relative to kotlin/difftools: sha256}}` |
| `reference` | section 0.2 |
| `pipeline_json` | the session's `pipeline` block as canonical JSON (sorted keys, `,` `:` separators, ASCII) |
| `rules_json` | the session's rules as the exact text `drowsy.spec.rules_to_json` gives (what `export/rules.json` holds for shipped rules) |
| `pipeline_json_sha256`, `rules_json_sha256` | sha256 of those two texts (UTF-8) |
| `pipeline_hash`, `rules_hash` | the same values when they equal the shipped ones, else `null` (class `var`, changed `drowsiness.source`, ...) |
| `baseline_injected` | `null` or `{e_l, e_r, b_l, b_r, jaw0, gaze_h0, gaze_v0 (hex), r0 [9 hex], t_complete_ms, status}` passed to the constructor |
| `feature_start_ms` | constructor argument or `null` |
| `model` | `null` (no model) or `{kind: "stub", column: "stub_probs", count, rule}`: model call number i (0-based, counted over the whole session, only calls that actually happen) returns row `min(i, count-1)` of `stub_probs` |
| `window_steps` | `pipeline.step_features.window_steps` of this session (model windows have this many rows) |
| `landmark_indices` | the 44 indices, ascending; order of the second axis of `lm` |
| `blendshape_names` | the 11 names: eyeBlinkLeft, eyeBlinkRight, jawOpen, then the 8 `eye_look` names in pipeline order; order of `bs` |
| `enums` | section 0.4 |
| `alert_priority` | of this session's rules |
| `counters` | `{frames_accepted, dropped_frames, steps_emitted, model_calls, nonfinite_windows}` after the last frame (or after the frame that raised) |
| `constructor_error` | `null` or `{type, message}` when `DriverPipeline(...)` itself raised; then no frame was fed and all output columns are empty |
| `exception` | `null` or `{fed_index, accepted_index, type, message, where, state_recorded}` when `update` raised (section 1.6) |
| `n_fed`, `n_accepted_rows` | lengths n and a of the column groups |
| `baseline_completed` | `null` or `{accepted_index, baseline}` (hex, as `baseline_injected`) for self-calibrating sessions |
| `min_margin` | `{T1..T8: hex or null}` smallest margin per comparison family over the session (section 1.7) |
| `labels` | `{RAW: clear|near, SIG: clear|near}` (section 1.7) |
| `active_at_end` | the active list after the last frame (section 4.1 entry form) |
| `notes` | generator parameters (informative) |

### 1.3 Columns

n = frames fed, a = accepted rows (frames for which `update` returned an output, plus the raising frame when
`exception.state_recorded`), f = fed frames with `face = 1`, r runs, s steps, c model calls, w full windows,
e events, l alerts. Row order = feeding order.

| group | column | dtype, shape | content |
|---|---|---|---|
| fed | `t_ms` | i64 [n] | `RawFrame.t_ms` of every fed frame (dropped ones too) |
| fed | `speed`, `speed_null` | f64 [n], u8 [n] | speed argument of `update`; `speed_null = 1` means `None` (then `speed` is 0.0) |
| fed | `accepted` | u8 [n] | 1 output returned, 0 dropped (returned `None`), 2 raised (last row only) |
| RAW input | `face`, `w`, `h` | u8 [n], i32 [n], i32 [n] | |
| RAW input | `lm` | f64 [f, 44, 2] | normalised (x, y) per `landmark_indices` |
| RAW input | `bs` | f64 [f, 11] | per `blendshape_names` |
| RAW input | `m` | f64 [f, 16] | column-major matrix |
| signals | `sig_face` | u8 [n] | `FrameSignals.face` |
| signals | `ear_l ear_r eye_w_l eye_w_r blink_l blink_r jaw_open yaw pitch roll` | f64 [n] each | `signals.compute` of every fed frame (SIG input / RAW expected) |
| signals | `eye_look` | f64 [n, 8] | in_l, out_l, up_l, down_l, in_r, out_r, up_r, down_r |
| signals | `iris`, `iris_null` | f64 [n, 4], u8 [n, 4] | `iris_h_l, iris_h_r, iris_v_l, iris_v_r`; null -> value 0.0, flag 1 |
| state | `closure yaw_rel pitch_rel gaze_h_rel gaze_v_rel jaw_rel` | f64 [a] each | `PipelineOutput.state` |
| state | `state_flags` | u8 [a] | bit0 face, bit1 closure_valid, bit2 provisional, bit3 pose_valid |
| extras | `gaze_zone` | u8 [a] | `DriverPipeline.gaze_zone` after the frame, index into `zones` (255 on the raising row) |
| extras | `active_mask` | u8 [a] | active events after the frame (section 4.1), bits per 0.4 |
| extras | `ongoing_feature_ms` | i64 [a] | feature detector (enter 0.5 / exit 0.3) `ongoing_ms()` after the frame |
| extras | `margin`, `margin_id` | f64 [a], u8 [a] | smallest margin over T1..T7 at this frame and its id 1..7 (+inf and 0 when none applies) |
| extras | `round_margin` | f64 [a] | smallest T8 margin at this frame (+inf when no interpolation rounding happened) |
| baseline | `bl_frame` | u32 [1] | accepted row whose `update` completed the baseline (self-calibrating sessions only) |
| baseline | `bl_f64` | f64 [1, 16] | e_l, e_r, b_l, b_r, r0[0..8], jaw0, gaze_h0, gaze_v0 |
| baseline | `bl_t_complete_ms`, `bl_status` | i64 [1], u8 [1] | |
| baseline | `dg_n_face`, `dg_f64` | i64 [1], f64 [1, 1 + 2q] | `BaselineDiagnostics`: n_face_frames; face_ratio, blink_l_quantiles[q], blink_r_quantiles[q] |
| runs | `run_frame` | u32 [r] | accepted row that emitted the run |
| runs | `run_times` | i64 [r, 3] | start_ms, end_ms, duration_ms |
| runs | `run_values` | f64 [r, 3] | amplitude, peak_closing_velocity, peak_opening_velocity |
| runs | `run_flags` | u8 [r] | bit0 velocity_valid, bit1 truncated |
| steps | `st_frame` | u32 [s] | accepted row that emitted the step |
| steps | `st_k`, `st_t_end_ms`, `st_full` | i64 [s], i64 [s], u8 [s] | |
| steps | `st_values` | f64 [s, 17] | feature order of `export/feature_spec_v0.json` |
| steps | `st_prob_null`, `st_probs` | u8 [s], f64 [s, 3] | `PipelineOutput.step_probs[j]`; null -> flag 1 and zeros |
| model | `mw_frame`, `mw_k` | u32 [c], i64 [c] | per model call: accepted row and the step k it was made for |
| model | `mw_sha` | u8 [c, 32] | sha256 of the window passed to the model as f64 LE bytes [window_steps, 17], oldest step first |
| model | `mwfull_call`, `mwfull` | u32 [w], f64 [w, window_steps, 17] | the full window of call 0 and of every call whose k is not the previous call's k + 1 |
| model | `stub_probs` | f64 [count, 3] | the stub table (header `model`) |
| events | `ev_frame` | u32 [e] | accepted row |
| events | `ev_type ev_severity ev_source ev_phase ev_provisional` | u8 [e] each | enum indices, severity, provisional 0/1 |
| events | `ev_times` | i64 [e, 2] | t_start_ms, duration_ms |
| events | `ev_confidence` | f64 [e] | |
| alerts | `al_*` | as `ev_*` | `PipelineOutput.alerts` |

Derived, not stored: `PipelineOutput.probs` of a row = `st_probs` of the last step of that row if that step's
`st_prob_null` is 0, else null (null when the row emitted no step). `FrameState.t_ms` of accepted row j = `t_ms` of
the j-th fed frame with `accepted != 0`. Within a row, runs/steps/events/alerts keep the reference's order.

Which groups are present: `RAW input` only in files whose `modes` has `RAW`; `signals` when `modes` has `RAW` or
`SIG`; `baseline` and `dg_*` only for self-calibrating sessions that completed (and `dg_*` only when diagnostics
exist); `stub_probs` only with a stub; all other groups always (possibly empty).

### 1.4 Corpora

| corpus | dir | sessions | modes | generator | contents |
|---|---|---|---|---|---|
| S-quick | `S-quick/<klass>/<seed>.hkd` | 600: inj 210, inj_model 90, cal 120, var 180 | STATE | `gen_s.py` = vendored `tx_fuzz.gen_session` + `frames_of` (seed base 5000000) | random sessions (plan 9.2.3) |
| S-odd | `S-odd/odd/<seed>.hkd` | 500 | STATE | same, class `odd` (first `t_ms` 0, -1, -5000, -1731465123457, 1; NaN/inf inputs) | outside the app contract |
| S-edge | `S-edge/<inj or cal>/<seed>.hkd` | 400 | STATE | `gen_edge.py`: hand-made `FrameState` streams (exact thresholds) | section 1.8 |
| F-stat | `F-stat/<klass>/<subject>.hkd` | 60 = 20 subjects x alert, low_vigilant, drowsy | RAW, SIG, STATE | `synth.make_subject` + `synth.generate_session`, 120 s, self-calibrating | noisy statistics |
| F-inj | `F-inj/<fps>/<seed>.hkd` | 400 | RAW, SIG, STATE | `synth.render` scripted scenes, injected baseline, 30/15/10 fps, 20-90 s | |
| F-cal | `F-cal/cal/<seed>.hkd` | 40 | RAW, SIG, STATE | `synth.render`, self-calibrating, stub model, 300 s | model output from second 239 |
| F-calib | `F-calib/calib/<seed>.hkd` | 300 | RAW, SIG (STATE also listed) | `synth.frame_from_signals` per frame, calibration only, 6-10 fps, scenes at every deadline (`gen_f.f_calib`) | extensions to 90-180 s, off_axis, fallback; exact-half sessions (`notes.exact_half_face_ratio`): a 10 fps grid of 100 s whose face frames before 60 s give exactly 30000 ms (even seed: ratio 0.5, completes at 60 s) or 29900 ms (odd seed: extends to 90 s); the export stops unless both occur |
| G-sig | `G-sig/<klass>/<seed>.hkd` | 200: `e_edge` 40, `yaw15` 40, `width6` 30, `counts` 40, `held` 40, `e_zero` 10 | SIG, STATE | hand-made `FrameSignals` (no frames), `gen_f.g_sig` | E exactly `e_min`/`e_max` or one ulp outside; median yaw exactly +-15.0 or one ulp outside; eye width exactly 6.0 and one ulp below; `counts`: left-eye sample count before 60 s exactly `notes.left_eye_samples_before_60s` = 300 + 10 (seed mod 11), so the 40 sessions cover 300, 310, ..., 400, with distinct values, 8 or 10 fps (the export stops if a session misses its count or a value is not covered); faceless frame or > 200 ms gap directly before deadlines; injected baseline with `e_l = 0.0` |
| S-full | `S-full/<klass>/<seed>.hkd` | 12,000: inj 4,200, inj_model 1,800, cal 2,400, var 3,600 | STATE | as S-quick (seed base 6000000) | E15 |

Each corpus dir has `index.json`: `{corpus, reference, generator, count, sessions: [{session_id, path, sha256,
bytes, n_fed, n_accepted, labels, exception, constructor_error}], totals: {sessions, frames_fed, frames_accepted,
near_RAW, near_SIG, exceptions, bytes}}`.

### 1.5 Replay modes (what a Kotlin replay feeds)

- RAW: rebuild `RawFrame(t_ms, face, w, h, lm, bs, m)` from the RAW input columns (lm keyed by `landmark_indices`,
  bs by `blendshape_names`, `m` 16 values; faceless frames have empty lm/bs/m) and call `update(frame, speed)`.
- SIG: feed `FrameSignals` rebuilt from the signals columns (t_ms from `t_ms`) to `updateFromSignals`.
- STATE: for each fed frame: if `accepted` is 0, feed any state with that `t_ms` and expect a drop (null output);
  else feed the next state row (t_ms from `t_ms`). For a self-calibrating file (`bl_frame` present) the reference
  had no baseline before accepted row `bl_frame` and had baseline `bl_f64...` from that row on (that row's state was
  already computed with it); pass it at that row (`updateFromState(state, baseline)`), with `BaselineDiagnostics`
  from `dg_*`. Rows before it have `provisional = 1`.
- Everything the reference produced after a frame is in the output columns of that accepted row.
- Self-check done by the exporter for every file (E1): the file is read back and every column compared bit for bit
  with memory; then the reference is run again from the re-read inputs **in every mode the file lists** (RAW from
  the re-read frames, SIG from the re-read signals through a stand-in for `signals.compute`, STATE from the re-read
  states through stand-ins for `signals.compute`, `calibrate.frame_state` and `BaselineEstimator` inside
  `drowsy.pipeline`; the pipeline code itself runs unchanged) and every output column, the counters, the exception
  and the baseline must be bit-equal. A file is written only if this passes.

### 1.6 Exceptions

`exception.fed_index` = index of the fed frame whose `update` raised; `accepted[fed_index] = 2`, no row follows.
If the reference had computed the frame state before raising (`state_recorded: true`), that state is row
`accepted_index` (its extras are 255 / 0 / 0 / +inf / 0 / +inf and it has no runs, steps, events). The Kotlin
replay must raise at that frame too (type name in `type`, Python message in `message`, location in `where`, for
example `drowsy/blink.py:339 _round_half_up` for `int(math.floor(NaN))`). Seen only in S-odd (NaN inputs).

### 1.7 Margins and labels (plan 9.2.5)

Per accepted row the exporter records the smallest `|value - threshold|` of the comparisons that the reference made
at that frame, by family: T1 eye width vs `signals.min_eye_width_px` (RAW: from the computed signals of face
frames); T2 E vs `e_min`/`e_max` at each baseline deadline check, in the reference's short-circuit order
(`e_l >= e_min`, `e_l <= e_max`, `e_r >= e_min`, `e_r <= e_max`, stopping at the first false one); T3 `|yaw0|` vs
`off_axis_yaw_deg` only at a deadline check that completes with a personal baseline (status ok or off_axis), the
only case in which the reference compares yaw (`calibrate.py:248-262`); T4
`|yaw_rel|` vs `closure.valid_max_abs_yaw_deg` (face frames); T5 closure vs the five closure thresholds (rows with
closure_valid); T6 `|yaw_rel|` vs `yaw_abs_max_deg`, `pitch_rel` vs `-pitch_up_max_deg` and `+pitch_down_max_deg`
(pose_valid rows); T7 nod value `pitch_rel - mean` vs `nod.enter_deg`/`exit_deg` (every valid nod update); T8
`|x - floor(x) - 0.5|` of every interpolation rounding (`blink._round_half_up`, both detectors). For T1-T7 a NaN
value counts as margin 0 (an infinite one gives margin +inf). For T8 a NaN or infinite `x` is not recorded: the
rounding itself raises there (section 1.6), so no rounding outcome exists to straddle. `labels.RAW` = `near` if any T1..T8 margin of the session is below 1e-6, else `clear`;
`labels.SIG` the same over T4..T8. STATE replays need no label (no straddling there).

### 1.8 S-edge (addition to plan 9.2.3, STATE only)

Hand-made `FrameState` streams that put values exactly on thresholds, which RAW-made sessions cannot reach because
`yaw_rel`, `pitch_rel` come through trigonometry: closure exactly 0.3, 0.5, 0.7, 0.8 and 0.0 -> 1.0 steps across
odd frame intervals (x.5 ms interpolation offsets), `yaw_rel` exactly +-23.0, `pitch_rel` exactly -25.0 and 18.0,
`gaze_h_rel` +-0.25, `gaze_v_rel` +-0.35, `jaw_rel` exactly 0.4 and 0.25, nod with `pitch_rel` 0.0 then exactly
10.0 and 5.0, stub model returning p_drowsy exactly 0.4 and 0.6, frame times on the credit and debounce instants
(`knife_stream` and `frame_stream` of the vendored rule fuzzers). Reference = `DriverPipeline` with the STATE
stand-ins of 1.5 (injected baseline, or a self-calibration whose baseline appears at the first frame at or after
60 s session time with t_complete_ms 60000). Each header `notes.exact_hits` counts the exact values used.

## 2. Small files

### 2.1 `numbers/canary.bin` and `numbers/canary.json` (E2, E17)  [name: canary]

`canary.bin` is HKDIFF1 with columns `f64 f64[N]` and `i64 i64[M]`; header `count_f64`, `count_i64`, `fixed_f64`
(indices of the fixed awkward values), `reference`. `canary.json`: `{reference, count_f64, count_i64, f64: [N JSON
numbers], f64_hex: [N hex strings], i64: [M JSON integers]}`. Element i of every list is the same value. Kotlin
reads the binary, parses the decimal text with the runner's JSON reader and the hex text with
`Double.parseDouble`, and compares raw bits (`doubleToRawLongBits`). Contents: the fixed values 0.1, -0.0, 0.0,
5e-324, 1.7976931348623157e308, 0.30000000000000004, 0.4, 0.6, 16.49993, 2.2250738585072014e-308 (smallest
normal), 2.225073858507201e-308 (largest subnormal), 1e23, 9007199254740993.0 (rounds to 2^53), 1e16, 1e-7,
0.5, 23.0, 1731465123457.0, 1.0000000000000002, 0.9999999999999999, then 10,000 random doubles: 4,000 random finite
bit patterns, 3,000 uniform in [0, 1), 2,000 values rounded to 1-6 decimal digits, 1,000 in [-1e6, 1e6].
`i64`: 0, 1, -1, 2^31 - 1, 2^31, -2^31 - 1, 2^53 + 1, 1731465123457, -1731465123457, 2^63 - 1, -2^63, and 1,000 random.

### 2.2 `numbers/comparator.json` (E2)  [name: comparator]

`{reference, rows: [...]}`. Each row is one call of the Python vector comparison `drowsy.vectors._compare(section,
key, expected, actual, tolerance, path, problems)` (the SPEC 16 judgement):
`{id, section, key, tolerance: {...}, expected: V, actual: V, expected_json, actual_json, python_ok, python_problems,
nan_row}`. `V` is a typed value: `{"t": "f", "hex": ...}` float, `{"t": "i", "v": n}` int, `{"t": "b", "v": bool}`,
`{"t": "n"}` null, `{"t": "s", "v": str}`, `{"t": "l", "v": [V...]}`, `{"t": "d", "v": {name: V}}`.
`expected_json` / `actual_json` are the same values as JSON text the way vector files write them, or null when the
value holds NaN or infinity. `python_ok` = Python appended no problem. `nan_row` = the actual or the expected value holds NaN
(the one place where the Kotlin comparator must differ: Python passes a NaN actual, Kotlin fails it, plan 9.0).
Infinity does not set `nan_row`: Python already fails an infinite actual, as Kotlin does. The rows cover every
section and tolerance key, values at and one ulp around `abs + rel*|e|`, -0.0, time keys +-1 ms, bool vs int,
int vs float, null, strings, list length and field set differences.

### 2.3 `tamper/` (E10)  [name: tamper]

`tamper/manifest.json`: `{reference, generator, base: {task, onnx, feature_spec, rules: {path in repo, sha256,
bytes}}, base_metadata: {key: value} (the 10 ONNX metadata pairs read with onnx 1.23.2), verified: {pipeline_hash,
rules_hash}, cases: [...]}` (51 cases). A case:
`{id, plan_case, change, dir, files: {role: path relative to .diffdata}, missing: [roles], metadata:
{key: value} or null, verified_override: {...} or null, expected_codes: [...], expected_codes_source, facts: {...},
note?}`. `dir` is null for cases without own files (metadata cases SB*, SD1, SD5). `verified_override` (SA8 only)
= `{rules_hash: sha256 of the tampered rules.json, constant_bits: "compute from the files in use"}`: use these as the
`VerifiedValues` of that case. `facts.sha256_in_use` = sha256 of every role's file in that case (null = missing).
SD4 points at a complete bundle dir (`tamper/SD4_SA4_bundle/`, the SA4 spec plus the base files) for the public
`openReplay(BundleSource.directory(...))` path. SD5 lists `IllegalArgumentException` (an exception type, not a
start-up code). SC2's `facts` hold onnxruntime outputs of the base and the tampered model on the `e2e_250s` window
(they differ in the 7th digit: the start-up check cannot see this change unless a pin is given).
Roles: `task`, `onnx`, `feature_spec`, `rules` (file names inside `dir` are the bundle names, except SD2 which uses
the v1 names). A role absent from `files` and from `missing` means "use the base file from the repository".
`metadata` = the metadata map handed to the start-up check function (null = read it from the onnx file in use).
`expected_codes` are the codes of plan 9.4 (the Kotlin check is new code; `expected_codes_source` says
`plan 9.4`). `facts` are computed by Python for the messages: sha256 of every file in use, the `pipeline_hash`
string inside the spec file, the hash the pipeline block computes to, whether the JSON parses, whether
`drowsy.spec` accepts it. Files: copies of the base files with one change; ONNX cases (SC1-SC3, SD3) are written with
the `onnx` package. `patch: {offset, old_byte, new_byte}` is given for single-byte changes.

### 2.4 `model/model_windows.bin` (E11)  [name: model_windows]

HKDIFF1. Columns `windows_f64 f64[300,120,17]`, `windows_f32 f32[300,120,17]` (numpy `astype(float32)` of the
f64 values, round to nearest even), `probs f32[300,3]` (onnxruntime outputs), `kind u8[300]` (0 real-scale cut from
S/F step values, 1 uniform [-3, 9], 2 extreme finite, 3 one-element changes of a base window), `order_windows
f32[3,120,17]` (the `e2e_250s` first window, the same with the step order reversed, with the feature order
reversed), `order_probs f32[3,3]`, `conv_f64 f64[K]` / `conv_f32 f32[K]` (Double -> Float conversion cases including
ties, subnormals, values up to 1e30). Header: `model_sha256`, `onnxruntime_version`, `session_options:
{intra_op_num_threads: 1, inter_op_num_threads: 1, providers: [CPUExecutionProvider]}`, `seed`, `order_check:
{max_abs_diff_reversed_steps, max_abs_diff_reversed_features, kept: [names]}` (a case is kept only if it differs
from the correct output by more than 1e-4), `triple_sum_max_dev`, `reference`.

### 2.5 `active_expected/<vector>.json` (E13, plan 6.2)  [name: active]

One file per vector of `tests/vectors` that has `expected.events` (53). `{reference, vector, vector_sha256,
alert_priority, frames: [{t_ms, active: [entry...]}]}`: one element per distinct `t_ms` of `expected.events` (file
order), the view after all events of that frame. Entry: `{type, severity, t_start_ms, source, provisional,
onset_frame_t_ms}` (fields of `ActiveEvent`, `onset_frame_t_ms` = t_ms of the frame of the onset), entries in
`alert_priority` order of the vector's rules (after its `config_overrides`). `active_expected/index.json` lists the
files and the cross-check against `demo.event_spans` (section 4.1).

### 2.6 `replay0/<name>.json` (delivery 0, plan 1.2 and K19)  [name: replay0]

`{reference, format: "hawkeye-replay0", version: 1, session: {name, description, fps, first_t_ms, duration_ms,
pipeline_hash, rules_hash, model: null, baseline_injected: null, speed_profile}, events_seen: {type: {onset, end}},
frames: [...], counters}`. Frame: `{t_ms, speed_kmh (number or null), output: null | {state, runs, steps,
step_probs, probs, events, alerts}, gaze_zone, baseline: null | {...}, counters, active: [...]}`. Field names are
the Python ones (`t_ms`, `closure_valid`, `peak_closing_velocity`, ...); `state` has all `FrameState` fields;
`steps[j] = {k, t_end_ms, full_lookback, values[17]}`; events as in the vector files; `baseline` = the pipeline's
`baseline` property after the frame (null before calibration ends); `counters` after the frame; `active` = section
4.1. `output` null = the frame was dropped. The run has no model (K13 (가): with `drowsiness.source: perclos` the
library does not call the model), so `probs` and `step_probs` are null. Every event type occurs at least once.

### 2.7 `components/C-*.json` (E3-E7)  [name: components]

All component records are JSON; every file has `{reference, generator, component, cases: [...]}`; each case is a
call sequence on one fresh Python object with the outputs after every call. Values numerically equal (`==`, plus
"both NaN") unless a case lists `"compare": "abs1e-9"` (functions whose result goes through `atan2`, `hypot`, `sin`,
`cos`: `eye_aspect_ratio`, the Euler and rotation-from-Euler helpers).

| file | object | case form |
|---|---|---|
| `C-func.json` | `calibrate.nearest_rank_quantile`, `median`, `eye_closure`; `signals.eye_aspect_ratio`, `iris_ratio`, `gaze_h`, `gaze_v`, `euler_from_colmajor`, `euler_from_rotation`, `rotation_from_euler`, `rotation_from_colmajor`, `relative_rotation`, `relative_euler`; `rules._ratio` (fn name `ratio`) | `{fn, args: [...], result, compare}` or `{fn, args, error: {type, message}, compare}`; `eye_closure` args = `[ear, blink, e, b, {rho, b_closed, b_span_min, w}]`; `rotation_from_euler` args = `[yaw, pitch, roll]`; tuples are lists. Top level also `rank_table: {qs, n_max, rank}` where `rank[i][n-1]` is the 1-based rank `nearest_rank_quantile` picks for `qs[i]` and n = 1..2000 (read back from the value it picked among 0.0, 1.0, ..., n-1) |
| `C-detector.json` | `ClosureRunDetector(RunDetectorSpec(**cfg))` | `{cfg, calls: [["u", t, c, valid] or ["r"]], out: [[run or null, ongoing_ms, run_start_ms, state]]}` (one `out` per call, also after `"r"`); run = `[start_ms, end_ms, duration_ms, amplitude, peak_closing_velocity, peak_opening_velocity, velocity_valid, truncated]`; state is `open`/`closed`/`hold` |
| `C-steps.json` | `StepAggregator(spec, start_ms, origin_ms)`, `ThresholdEventCounter(**args)`, `NodDetector(NodSpec(**cfg))` | `kind: "aggregator"`: `{pipeline (block), start_ms, origin_ms, calls: [[state, run or null, ongoing_ms]], out: [[step...]]}`; `kind: "counter"`: `{args, calls: [["u", t, value, valid] or ["r"]], out: [[counted, active]]}`; `kind: "nod"`: `{cfg, calls, out: [[counted, mean]]}` |
| `C-baseline.json` | `BaselineEstimator(spec)` | `{pipeline, calls: [FrameSignals object], out: [baseline or null], diagnostics: null or {n_face_frames, face_ratio, blink_l_quantiles, blink_r_quantiles}}` |
| `C-hysteresis.json` | `StepHysteresis(onset_steps, end_steps, refractory_ms)` | `{args: [onset_steps, end_steps, refractory_ms], calls: [["u", k, t_end_ms, has_value, onset_hit, end_hit] or ["r"]], out: [[active, start_ms]]}` |
| `C-engine.json` | `RuleEngine(rules_from_dict(rules))` | `{rules (dict), calls: [["s", step, probs or null] or ["f", state]], out: [[events], gaze_zone]}` (gaze_zone after an `"f"` call, null after `"s"`); steps are fed before the frame they belong to, as `DriverPipeline` does |
| `C-policy.json` | `AlertPolicy(rules_from_dict(rules))` | `{rules, source, calls: [[events, speed or null, t_ms]], out: [[alerts]]}`; `source` says whether the events came from a `RuleEngine` run or the synthetic generator (with repeated onsets without end and speed -5.0) |
| `C-errors.json` | constructors and argument checks | `{case, component, setup, calls, out}`. Function cases: `setup = {fn, args}` (fn names as in `C-func.json`), `out = {ok: value}` or `{error: {type, message}}`. Constructor cases (`DriverPipeline`): `setup = {pipeline, rules, kwargs}` (`kwargs.baseline` as a plain baseline object with `r0` a list), `out` as above. Model returning 2, 4 or 0 values: `setup = {inputs, model_returns, note}`; `inputs` = `components/C-errors_model_inputs.hkd`, an HKDIFF1 file (section 1, modes RAW, SIG, STATE, with its own read-back and re-run check) of the frames of tx_fuzz session `inj_model` seed 9001 with the shipped constants, its injected baseline and a stub model that returns (0.25, 0.5, 0.25), cut after the frame of the first model call (that call is its only `mw_*` row). Build `DriverPipeline` from that file, replace the stub by a model that returns `model_returns`, feed the frames in any of its modes: `out = {fed_index, t_ms, error, where}` (the last fed frame raises) and `counters_at_error = {frames_accepted, steps_emitted, model_calls}` after the raise. Stream cases (`RuleEngine`, `ClosureRunDetector`, `BaselineEstimator`, `StepAggregator`): `calls` = the per-call inputs (state object / `[t, c, valid]` / signals object / `[state, run, ongoing_ms]`), `out[i]` = `{ok: output}` or `{error: {...}}` for a refused (non-increasing t) call; the stream goes on after a refusal and later outputs must still match |
| `C-loader.json` | `spec.pipeline_spec_from_dict`, `rules_from_dict`, `feature_spec_from_dict` | `{target: pipeline|rules|feature_spec, path, change, text, accepted, error: {type, message} or null, flags: [...]}`; `text` is the JSON document text (`indent=2`) that was parsed and judged; one leaf changed per document (value kinds: 0, -1, +1, x10, int<->float, 2^31, 2^63, -2^31-1, string, bool, null, removed key, unknown key, list shorter/longer/reversed/empty). `flags` (top-level `flags_meaning` explains them) mark documents where a Kotlin loader built as planned is expected to judge differently from Python, for the owner to decide before E7: `int_outside_int32` (an integer outside the `Int` range, plan R22), `int_outside_int64` (outside `Long`, plan 5.10), `accepted_changed_string` (Python accepted a changed string, for example in `pipeline.pose.*`). Flags never change `accepted` or `error`, which are Python's. Top-level `depends_on` (section 0.2) |

State, step, signals, event objects inside component records use the Python field names
(`FrameState`: t_ms, face, closure, closure_valid, provisional, yaw_rel, pitch_rel, pose_valid, gaze_h_rel,
gaze_v_rel, jaw_rel; `StepFeatures`: k, t_end_ms, full_lookback, values; events: the 8 fields of the vector files).

## 3. Which experiment uses which file

| file | experiment (plan 9.1) |
|---|---|
| `numbers/canary.*`, `numbers/comparator.json` | E2 (and E17 on the emulator) |
| `components/C-func.json` | E3, E6, E7 |
| `components/C-baseline.json` | E4 |
| `components/C-detector.json` | E5 |
| `components/C-steps.json` | E6 |
| `components/C-hysteresis.json`, `C-engine.json`, `C-policy.json`, `C-errors.json`, `C-loader.json` | E7 |
| `S-quick`, `S-odd`, `S-edge`, F-* (STATE) | E8 |
| `F-stat`, `F-inj`, `F-cal`, `F-calib`, `G-sig` (RAW, SIG) | E9 |
| `tamper/` | E10 |
| `model/model_windows.bin` | E11 |
| `active_expected/` and the `active_mask` column | E13 |
| `S-full` | E15 |
| `replay0/` | delivery 0 (`PreviewPlayer` tests, stage 0b) |

## 4. Definitions used by the exporter

### 4.1 Active fold (plan 6.2, K10 (가))

```
active = map type -> entry; new session: empty
for each accepted frame, for each event of output.events in order:
    onset: active[type] = entry(type, severity, t_start_ms, source, provisional of the onset, onset_frame_t_ms = frame t_ms)
           if type == sleep: remove microsleep
    end:   remove type; if type == sleep: remove microsleep
view = entries ordered by rules.alert_priority
```

Cross-check at export (`active_expected/index.json`): for each of the 53 vectors, the set of `(type, t_start_ms)`
ever active equals the set of `demo.event_spans` spans; each type leaves the view at the event whose
`t_start_ms + duration_ms` is the span's end, except `microsleep`, which leaves at the `sleep` onset while its span
ends at the `sleep` end; the types active after the last event are exactly the spans with `ended = False`.

## 5. What the generation of 2026-11-15 produced (measured)

reference_id `3ee2ed60346dd7417b36613d999dab9046ecbfa980c056caaf90f1411c32a428`. Sizes are `du -sb` bytes
(apparent size); frames = frames fed; near = sessions labelled `near` (section 1.7); times = wall time of the
generation step with 10 worker processes under `nice -n 19` (including each file's read-back and re-runs).

| data set | sessions / files | frames fed | accepted | near RAW / SIG | exceptions | bytes | generation |
|---|---|---|---|---|---|---|---|
| S-quick | 600 | 447,585 | 444,933 | 326 / 321 | 0 | 65,359,728 | 3.5 s |
| S-odd | 500 | 257,437 | 255,955 | 281 / 274 | 18 | 44,194,999 | 2.0 s |
| S-edge | 400 | 1,572,242 | 1,564,338 | 400 / 400 | 0 | 172,731,578 | 6.0 s |
| F-stat | 60 | 146,652 | 146,652 | 0 / 0 | 0 | 176,806,268 | 3.3 s |
| F-inj | 400 | 350,847 | 350,847 | 272 / 272 | 0 | 415,726,966 | 6.2 s |
| F-cal | 40 | 237,256 | 237,256 | 40 / 40 | 0 | 280,347,253 | 5.9 s |
| F-calib | 300 | 302,093 | 302,093 | 0 / 0 | 0 | 260,890,819 | 4.0 s |
| G-sig | 200 | 201,395 | 201,395 | 80 / 0 | 0 | 60,789,285 | 1.6 s |
| S-full | 12,000 | 9,117,007 | 9,062,535 | 6,522 / 6,368 | 0 | 1,320,878,047 | 66.8 s |
| components | 10 files (9 JSON + `C-errors_model_inputs.hkd`) | | | | | 109,702,443 | 16.2 s |
| tamper | 51 cases, 27 files + manifest | | | | | 16,120,963 | 0.4 s |
| model | 1 file | | | | | 7,381,656 | 2.5 s |
| numbers | 3 files | | | | | 1,117,629 | 0.1 s |
| replay0 | 2 files (5,100 and 1,180 frames) | | | | | 4,318,176 | 0.7 s |
| active_expected | 53 files + index | | | | | 147,555 | 3.3 s |

Regenerated the same day after the review of stage "diff" (T2/T3 margins, G-sig `counts`, F-calib exact-half,
C-errors model inputs, `depends_on`, C-loader `flags`): compared with the first generation, the data region changed
in 37 G-sig `counts` sessions and the 2 F-calib exact-half sessions (regenerated on purpose) and in 372 other session
files (F-calib 66, S-quick 22, S-odd 15, S-full 269). In those 372 only the `margin` / `margin_id` columns differ
(measured by regenerating them with the old margin wrapper and comparing every column and the header). 11 S-full
sessions changed their RAW label from `near` to `clear` (T3 margins of yaw comparisons the reference never made).

The S-odd exceptions are all `ValueError: cannot convert float NaN to integer` at `drowsy/blink.py:339
_round_half_up` (NaN inputs). F-inj and F-cal (scripted, noise-free ramps) are mostly `near`: their linear ramps put
closure and interpolation offsets exactly on thresholds in RAW mode too. F-stat and F-calib are all `clear`.

## 6. Files of `kotlin/difftools/`

| file | content |
|---|---|
| `export.py` | command entry (section 0.1), READY files, `check`, `verify` |
| `hk.py` | HKDIFF1 writer/reader, reference fingerprint, the RAW/SIG/STATE stand-ins inside `drowsy.pipeline`, margin taps, active fold, session runner, self-check, `verify_file` |
| `gen_s.py` | S corpora from the vendored `tx_fuzz.py` |
| `gen_edge.py` | S-edge |
| `gen_f.py` | F-stat, F-inj, F-cal, F-calib, G-sig |
| `corpora.py` | corpus task lists, worker pool, `index.json` |
| `small.py` | canary, comparator, active_expected, replay0, model_windows |
| `tamper.py` | tamper cases |
| `components.py` | C-* records and their re-run check (every C-* file, C-errors including the model cases from their input file, C-loader re-judged from `text`) |
| `selftest.py` | regression checks of the exporter itself (T2/T3 margins record only the comparisons the reference made); first step of `export.py small` |
| `vendor/` | copies of the generators from the work folders outside the repository (`vendor/PROVENANCE.tsv`: source path and sha256 at copy time). They are imported (or their function definitions read with `ast`) and never edited |
