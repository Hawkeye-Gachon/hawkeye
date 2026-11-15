"""시험 fold 격리의 회귀 테스트 (PLAN 6장). 명령을 바깥에서만 본다: 입력 파일을 바꾸고 결과 파일을 비교한다.

작은 합성 데이터셋(피험자 4명: 공식 fold 1과 2에 한 명씩, fold 3에 두 명)을 세 벌 만든다.
- `clean`은 그대로다.
- `corrupted`는 바깥 fold 3의 시험 피험자 둘 가운데 25번의 추출 결과만 바꿔치기한 것이다: 다른 시드로 만든 다른 얼굴과 다른
  눈 움직임(특징이 달라진다)을, 다른 상태의 세션을 원래 라벨의 파일 이름으로(라벨이 뒤섞인다), 더 긴 길이로(창 수가
  달라진다) 쓴다. 원본 추출과 15 fps 누락 모사 추출의 두 폴더를 모두 바꾼다. 프레임 누락 변형은 원본 추출에서 만들어지므로
  함께 바뀐다. 같은 시험 fold의 26번은 손대지 않는다.
- `relabelled`는 26번의 라벨 파일만 맞바꾼 것이다: low_vigilant 세션의 내용을 drowsy의 파일 이름으로, drowsy 세션(두 부분)의
  내용을 low_vigilant의 파일 이름으로 쓴다. 프레임은 한 장도 다르지 않고 나머지 피험자의 파일도 그대로다. 달라지는 것은
  26번의 두 영상에 붙는 라벨뿐이다.

세 벌에 같은 명령(축소 설정, CPU)을 돌린 뒤 `clean`과 `corrupted`에서 두 가지를 비교한다.

1. 선택 기록(`selection.json`)
   - 바깥 fold 3의 기록은 두 실행에서 완전히 같아야 한다: 후보별 내부 검증 값과 학습 길이와 온도, 고른 설정, 다시 학습한
     모델의 표준화 상수·온도·가중치 해시(LightGBM은 모델 해시, fold별 전역 상수 실행은 그 상수도). 시험 fold의 데이터가
     선택, 조기 종료, 표준화 상수, 온도, 가중치 어디에도 닿지 않았다는 뜻이다.
   - 25번이 학습이나 검증에 들어가는 다른 fold의 기록은 달라져야 한다. 이 비교가 바뀐 데이터를 알아챈다는 대조군이다.
2. 예측 표(`predictions.parquet`)
   - 손대지 않은 시험 피험자 26번의 행은 모든 묶음에서 두 실행이 같아야 한다. 평가하는 시점에 시험 fold의 다른 창을 쓰는
     누설(시험 fold 창의 평균으로 다시 표준화하기, 시험 fold의 라벨로 온도를 다시 맞추기, 시험 fold에서 임계값 고르기)은
     선택 기록을 바꾸지 않는다. 그런 누설은 25번의 데이터가 바뀔 때 26번의 확률이 따라 바뀌는 것으로만 드러난다.
   - 확률 세 열은 `PROBABILITY_TOLERANCE` 안에서, 나머지 열은 정확히 같아야 한다.
   - 25번의 행은 달라져야 한다(바뀐 데이터가 평가에 실제로 쓰였다는 대조군).

`clean`과 `relabelled`에서는 평가받는 피험자 자신의 라벨이 그 피험자의 예측에 닿는지를 본다(위의 비교로는 보이지 않는다.
26번의 데이터와 라벨이 두 실행에서 같기 때문이다).
3. 26번의 확률 행은 내용이 같은 행끼리 견주면 모든 묶음에서 `clean`과 같아야 한다. 시험 피험자(또는 영상)마다 자기 시험
   라벨로 온도를 다시 맞추거나, 예측에 정답 라벨을 섞거나, 라벨로 임계값을 고르는 누설은 라벨이 바뀔 때 같은 프레임의 확률이
   따라 바뀌는 것으로만 드러난다. 바깥 fold 3의 선택 기록도 그대로여야 하고(그 fold의 학습 피험자는 손대지 않았다), 26번이
   학습이나 검증에 들어가는 fold 1, 2의 기록은 달라져야 한다(바뀐 라벨이 실제로 쓰였다는 대조군). 예측 함수와 예측 표를
   만드는 곳에 그런 누설 넷(피험자별 온도 재보정, 정답 라벨을 섞은 확률: 신경망 두 곳과 LightGBM)을 넣은 사본에서 이 비교가
   실패하는 것을 확인했다.
   가정: alert 영상(라벨 0)은 맞바꾸지 않는다. alert 영상 기준선 조건은 피험자의 alert 영상에서 기준선을 얻으므로(PLAN 4.3절),
   그 파일을 바꾸면 라벨을 정당하게 쓰는 그 경로 때문에 26번의 특징이 달라져 비교할 수 없다.

이 테스트가 잡지 못하는 누설(알려진 한계. 넷째 데이터 묶음은 만들지 않는다. 테스트가 이미 1분쯤 걸린다):
- 평가받는 피험자의 alert 대 나머지 라벨만 쓰는 누설. 26번의 alert 파일은 세 벌에서 모두 같으므로(위의 가정), 예측에
  "alert인지 아닌지"만 섞거나, 시험 피험자마다 자기 시험 라벨을 alert 대 나머지로 줄여 온도를 다시 맞추는 누설은 26번의
  확률 행을 바꾸지 않는다.
- 리포트를 쓰는 시점의 누설. 비교하는 것은 선택 기록(`selection.json`)과 예측 표(`predictions.parquet`)뿐이다.
  `drowsy.evaluate`가 예측 표에서 계산하는 것(창의 투표, 영상 판정, 경보 판정, 지표)에 시험 라벨을 쓰는 누설은 예측 표를
  바꾸지 않으므로 여기서 드러나지 않는다.
- `baseline_lgbm probes`(혼동 변수 점검)와 `train final`(최종 모델)은 돌리지 않는다(아래 `COMMANDS`에 없다).

보는 명령(`COMMANDS`): 신경망(`train cv`: 원본, 15 fps 모사, 프레임 누락 변형을 섞은 학습), LightGBM(`baseline_lgbm cv`),
fold별 전역 상수(`baseline_lgbm cv --arms global --foldwise-global`), 부가 실행 넷(`train cv --ordinal`,
`train cv --no-velocity`, `baseline_lgbm cv --no-velocity`, `baseline_lgbm cv --all-w`), 내보낼 조건이 아닌 정규화 조건으로
학습한 실행(`train cv --arm global`, 다른 실행 이름).

가정: 세션은 명목 18 fps로 만든다(`drowsy.synth`의 기본은 피험자마다 17~30 fps이고 구간마다 그 70%까지 내려간다).
프레임 수를 줄여 재생 시간을 줄이려는 것이고, 보려는 것(어느 피험자의 데이터가 어느 계산에 들어가는가)은 프레임률과
무관하다. 더 낮추지 않는 이유: 프레임 간격이 흐름 제한기의 처리 지연(평균 66.7 ms)보다 길면 15 fps 누락 모사 폴더가
원본과 같은 프레임을 담게 되어 두 출처가 구분되지 않는다.
가정: 확률의 허용 오차는 1e-6이다. 25번의 창 수가 달라지면 시험 fold를 한 번에 넣는 창 수(배치 크기)가 달라지고, GRU는 배치
크기에 따라 float32 마지막 자리가 달라질 수 있다(누설이 아니다). 이 PC에서 무작위 가중치의 GRU(은닉 8)에 같은 창을 배치
크기만 바꿔 넣었을 때 확률이 5.96e-08까지 달랐고, 이 테스트의 두 실행에서는 26번의 확률이 44개 묶음 모두에서 차이 0이었다.
시험 fold의 통계가 평가에 섞이는 누설은 허용 오차보다 몇 자리 큰 차이를 낸다(평가 함수에 그런 누설 둘을 넣은 사본에서
이 파일이 실패하는 것을 확인했다). 라벨만 맞바꾼 비교에서는 26번의 창이 같은 배치 안에서 자리만 바뀐다. 같은 허용 오차를 쓴다.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from drowsy import evaluate, signal_check, synth, train
from drowsy.config import load_config
from drowsy.dataset import STEP_COLUMNS, make_windows, official_fold
from drowsy.extract import write_raw_frames
from drowsy.spec import PipelineSpec
from drowsy.types import FEATURE_NAMES

ARM = "alert_baseline"
OTHER_ARM = "global"  # 내보낼 조건(alert_baseline)이 아닌 정규화 조건
SUBJECTS = (1, 13, 25, 26)  # 공식 fold 1, 2에 한 명씩, fold 3에 두 명
TEST_FOLD = 3  # 이 바깥 fold의 시험 피험자 하나를 바꿔치기한다
CORRUPTED_SUBJECT = 25
UNTOUCHED_SUBJECT = 26  # 25번과 같은 시험 fold. 이 피험자의 예측은 25번의 데이터와 무관해야 한다
OTHER_FOLDS = (1, 2)  # 25번이 학습이나 검증에 들어가는 바깥 fold
MINUTES = 4.1  # 세션 246초: 보폭 1에서 세션마다 창 7개
FPS = 18.0
CORRUPTED_SEED = 1000
CORRUPTED_EXTRA_MS = 20_000  # 바꿔치기한 세션은 이만큼 길다(창 수가 달라진다)
PERMUTED_CLASS = {0: 2, 1: 0, 2: 1}  # 파일의 라벨(클래스) → 그 파일에 실제로 써 넣는 상태
PROBABILITY_TOLERANCE = 1e-6  # 26번의 확률이 두 실행에서 이 안에서 같아야 한다(모듈 설명의 가정)
RELABELLED_SUBJECT = UNTOUCHED_SUBJECT  # 세 번째 데이터셋에서 라벨 파일을 맞바꾸는 피험자
SWAPPED_CLASS = {1: 2, 2: 1}  # 세션의 클래스 → 그 내용을 써 넣는 파일의 클래스(low_vigilant ↔ drowsy). alert는 그대로 둔다
DATASETS = ("clean", "corrupted", "relabelled")

# 데이터셋마다 차례로 돌리는 명령: (모듈, 실행 이름에 붙이는 접미사, 하위 명령과 옵션). 공통 옵션은 `run_commands`가 붙인다.
# 신경망의 주 실행을 먼저 돌린다(두 정규화 조건의 스텝 표를 한 번에 만들어 두면 뒤의 명령은 캐시에서 읽는다).
COMMANDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("train", "", ("cv", "--fast")),
    ("baseline_lgbm", "", ("cv", "--fast")),
    ("baseline_lgbm", "", ("cv", "--fast", "--arms", OTHER_ARM, "--foldwise-global")),
    ("train", "", ("cv", "--fast", "--ordinal")),
    ("train", "", ("cv", "--fast", "--no-velocity")),
    ("baseline_lgbm", "", ("cv", "--fast", "--no-velocity")),
    ("baseline_lgbm", "", ("cv", "--fast", "--all-w")),
    ("train", "_other_arm", ("cv", "--fast", "--arm", OTHER_ARM)),
)


@dataclasses.dataclass(frozen=True)
class Report:
    """명령 하나가 쓴 리포트 폴더.

    - label: 테스트 이름에 붙는 이름. suffix: 실행 이름(`DATASETS`의 이름)에 붙는 접미사. subdir: 실행 폴더 아래의 하위 폴더.
    - n_groups: 그 예측 표의 묶음 수. n_records: 바깥 fold마다의 선택 기록 수.
    """

    label: str
    suffix: str
    subdir: str
    n_groups: int
    n_records: int


# 묶음 수는 명령이 만드는 묶음을 손으로 센 값이다.
#   train cv: 계열 셋(tcn, gru, selected) × 평가 조건 둘(학습 조건, 앱 조건) × 출처 둘(원본, 15 fps 모사) = 12,
#             TCN fold 모델의 ONNX 파일과 INT8 파일 × 평가 조건 둘 = 4
#   부가 실행(ordinal, no_velocity): 계열 셋 × 평가 조건 둘 = 6 (15 fps 평가와 INT8 비교는 주 실행에만 있다)
#   다른 조건(global)의 실행: 계열 셋 × 출처 둘 = 6, ONNX 파일과 INT8 파일 = 2 (앱 조건 평가가 없다)
#   LightGBM: 원본과 15 fps 모사 = 2, fold별 전역 상수: spec 상수와 fold별 상수 = 2, 속도 특징 제외 = 1, W 셋 = 3
REPORTS: tuple[Report, ...] = (
    Report("train_cv", "", "", 16, 1),
    Report("lgbm_cv", "", "lgbm", 2, 1),
    Report("lgbm_fold_global", "", "lgbm_fold_global", 2, 2),
    Report("train_ordinal", "", "ordinal", 6, 1),
    Report("train_no_velocity", "", "no_velocity", 6, 1),
    Report("lgbm_no_velocity", "", "lgbm_no_velocity", 1, 1),
    Report("lgbm_all_w", "", "lgbm_w_grid", 3, 3),
    Report("train_other_arm", "_other_arm", "", 8, 1),
)
REPORT_IDS = [report.label for report in REPORTS]
RUN_KEYS = [*evaluate.GROUP_KEYS, "seed"]
ROW_ORDER = ["video_id", "part", "k"]
PROB_COLUMNS = list(evaluate.PROB_COLUMNS)
OTHER_COLUMNS = [name for name in evaluate.PREDICTION_COLUMNS if name not in PROB_COLUMNS]


def write_session(
    job: synth.SessionJob, root: Path, spec: PipelineSpec, class_index: int, seed: int, extra_ms: int = 0,
    rel_path: str | None = None,
) -> None:  # fmt: skip
    """합성 세션 하나를 job의 경로에 쓴다(`synth.write_session`과 같은 형식, 명목 18 fps).

    class_index와 seed는 내용(상태와 얼굴)을, extra_ms는 길이를 정한다. 경로(피험자, 라벨, 부분)는 job의 것 그대로다.
    rel_path를 주면 같은 내용을 그 경로에 쓴다(라벨 파일을 맞바꿀 때).
    """
    profile = dataclasses.replace(synth.make_subject(job.subject, seed), fps=FPS)
    metadata = {"source": "synthetic (drowsy.synth)", "synthetic": True, "seed": seed, "subject": job.subject}
    frames = synth.generate_session(profile, class_index, job.duration_ms + extra_ms, seed, spec, job.part)
    if job.drop_sim is not None:
        frames = synth.simulate_device_drops(frames, job.drop_sim)
    write_raw_frames(frames, root / (job.rel_path if rel_path is None else rel_path), spec, metadata)


def swapped_rel_path(job: synth.SessionJob) -> str:
    """job의 세션을 맞바꾼 라벨의 파일 이름으로 옮긴 경로. 부분 번호는 그대로다(`26/10_1.parquet` → `26/5_1.parquet`)."""
    label = synth.LABELS[SWAPPED_CLASS[job.class_index]]
    name = f"{label}.parquet" if job.part == 0 else f"{label}_{job.part}.parquet"
    return str(Path(job.rel_path).with_name(name).as_posix())


def write_dataset(root: Path, spec: PipelineSpec, drop_sim: Any, name: str) -> None:
    """데이터셋 name(`DATASETS`)을 root에 쓴다."""
    assert name in DATASETS
    for job in synth.session_jobs(max(SUBJECTS), MINUTES, 0, drop_sim=drop_sim):
        if job.subject not in SUBJECTS:
            continue
        if name == "corrupted" and job.subject == CORRUPTED_SUBJECT:
            write_session(job, root, spec, PERMUTED_CLASS[job.class_index], CORRUPTED_SEED, CORRUPTED_EXTRA_MS)
        elif name == "relabelled" and job.subject == RELABELLED_SUBJECT and job.class_index in SWAPPED_CLASS:
            # 내용은 clean과 한 프레임도 다르지 않다. 파일 이름(라벨)만 다르다.
            write_session(job, root, spec, job.class_index, job.seed, rel_path=swapped_rel_path(job))
        else:
            write_session(job, root, spec, job.class_index, job.seed)


def write_config(config_path: Path, repo_root: Path, tmp: Path) -> Path:
    """축소 설정을 더 줄인 설정 파일(CPU). 데이터 경로는 `--data-root`로 준다."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"].update(
        rldd_root=None,
        synth_dir=str(tmp / "nowhere"),
        feature_spec=str(repo_root / "config" / "feature_spec.json"),
        rules=str(repo_root / "config" / "rules.yaml"),
        reports_dir=str(tmp / "reports"),
        export_dir=str(tmp / "export"),
    )
    data["extraction"]["workers"] = 1
    data["training"].update(device="cpu", batch_size=16)
    data["fast"].update(
        max_units=2, patience_units=1, updates_per_unit=3, seeds=[0], tcn_channels=[8], gru_hidden=[8],
        lgbm_num_leaves=[3], lgbm_min_data_in_leaf=[1], lgbm_max_rounds=5, bootstrap_n=20,
    )  # fmt: skip
    out = tmp / "config.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


# 데이터셋 하나의 명령들을 차례로 돌리는 프로그램. 저장소 루트에서 `python -c`로, 데이터셋마다 프로세스 하나씩 실행한다.
# 인자: [[모듈, 명령 인자 목록], ...]의 JSON, 결과를 쓸 파일. 결과는 명령마다의 종료 코드와 소요 시간(초)이다.
RUNNER = """
import contextlib, io, json, sys, time
from drowsy import baseline_lgbm, train

mains = {"train": train.main, "baseline_lgbm": baseline_lgbm.main}
codes, seconds = [], []
for module, arguments in json.loads(sys.argv[1]):
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        codes.append(mains[module](arguments))
    seconds.append(time.perf_counter() - started)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump({"codes": codes, "seconds": seconds}, handle)
"""


def run_commands(tmp: Path, config_path: Path, repo_root: Path, spec: PipelineSpec) -> SimpleNamespace:
    """세 데이터셋(`DATASETS`)을 tmp에 쓰고 `COMMANDS`를 차례로 돌린다. 결과 폴더와 종료 코드, 소요 시간을 돌려준다.

    데이터셋마다 별도 프로세스(`RUNNER`)가 그 데이터셋의 명령 여덟 개를 차례로 돌리고, 세 프로세스는 동시에 돈다.
    명령은 `python -m drowsy.train ...`으로 실행할 때와 같은 `main`을 같은 인자로 부른 것이다.
    가정: 세 프로세스는 같은 환경 변수(스레드 수 포함)로 같은 기계에서 돈다. 비교하는 값(가중치 해시, 확률)은 실행마다
    같은 값이 나오는 CPU 학습의 결과이므로, 한 프로세스에서 차례로 돌릴 때와 같은 비교가 된다. 세 번째 데이터셋을 더하고도
    이 파일의 소요 시간이 늘지 않게 하려는 것이다(차례로 돌리면 데이터셋마다 약 45초가 든다).
    가정: 환경 변수 `OMP_NUM_THREADS`가 없으면 세 프로세스에 2로 준다(있으면 그 값 그대로). 셋이 동시에 도는데 저마다 모든
    코어를 쓰려 들면 서로 밀려 오히려 느려진다. 이 시험의 모델과 표는 작아서 스레드를 늘려도 빨라지지 않는다.
    """
    config = write_config(config_path, repo_root, tmp)
    drop_sim = load_config(config).extraction.drop_sim
    roots: dict[str, Path] = {}
    seconds: dict[str, list[float]] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "2")
    for name in DATASETS:
        started = time.perf_counter()
        roots[name] = tmp / name / "synth"
        write_dataset(roots[name], spec, None, name)
        write_dataset(tmp / name / "synth_fps15", spec, drop_sim, name)
        seconds[name] = [time.perf_counter() - started]  # 첫 값은 데이터셋을 쓰는 시간, 그 뒤는 명령마다
        common = ["--config", str(config), "--data-root", str(roots[name])]
        commands = [
            [module, [*common, "--run-name", name + suffix, *arguments]] for module, suffix, arguments in COMMANDS
        ]
        # 표준 출력과 표준 오류는 파일로 보낸다(파이프로 받으면 기다리는 동안 다른 프로세스의 버퍼가 찰 수 있다).
        with (tmp / name / "runner.log").open("wb") as log:
            processes[name] = subprocess.Popen(
                [sys.executable, "-c", RUNNER, json.dumps(commands), str(tmp / name / "runner.json")],
                cwd=repo_root, env=environment, stdout=log, stderr=subprocess.STDOUT,
            )  # fmt: skip
    codes: dict[str, list[int]] = {}
    failed: list[str] = []
    for name, process in processes.items():
        if process.wait() != 0:
            log_text = (tmp / name / "runner.log").read_text(encoding="utf-8", errors="replace")
            failed.append(f"{name}: exit code {process.returncode}\n{log_text[-3000:]}")
            continue
        result = json.loads((tmp / name / "runner.json").read_text(encoding="utf-8"))
        codes[name] = [int(code) for code in result["codes"]]
        seconds[name] += [float(value) for value in result["seconds"]]
    assert not failed, "the commands of a data set did not finish:\n" + "\n".join(failed)
    return SimpleNamespace(tmp=tmp, config=config, roots=roots, codes=codes, seconds=seconds, reports=tmp / "reports")


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory, config_path: Path, repo_root: Path, pipeline_spec: PipelineSpec
) -> SimpleNamespace:
    """세 데이터셋(clean, corrupted, relabelled)에 같은 명령들을 돌린 결과 폴더."""
    return run_commands(tmp_path_factory.mktemp("leakage"), config_path, repo_root, pipeline_spec)


def report_dir(runs: SimpleNamespace, name: str, suffix: str = "", subdir: str = "") -> Path:
    return runs.reports / (name + suffix) / subdir


def selection(runs: SimpleNamespace, name: str, *subdir: str, suffix: str = "") -> dict[int, list[dict[str, Any]]]:
    """실행 name의 선택 기록을 바깥 fold 번호별로 묶은 것(fold마다 기록이 여럿일 수 있다)."""
    path = report_dir(runs, name, suffix).joinpath(*subdir, "selection.json")
    records: dict[int, list[dict[str, Any]]] = {}
    for record in json.loads(path.read_text(encoding="utf-8"))["folds"]:
        records.setdefault(int(record["fold"]), []).append(record)
    return records


def predictions(runs: SimpleNamespace, name: str, suffix: str = "", subdir: str = "") -> pd.DataFrame:
    """실행 name의 예측 표(검사를 통과한 형식)."""
    path = report_dir(runs, name, suffix, subdir) / "predictions.parquet"
    return evaluate.validate_predictions(pd.read_parquet(path))


def subject_rows(table: pd.DataFrame, subject: int) -> dict[tuple[Any, ...], pd.DataFrame]:
    """예측 표에서 한 피험자의 행을 묶음(과 시드)별로 나눈 것. 묶음 안에서는 영상, 부분, 스텝 순서다."""
    rows = table[table["subject"] == subject]
    return {
        key: group.sort_values(ROW_ORDER).reset_index(drop=True)
        for key, group in rows.groupby(RUN_KEYS, sort=True)
    }


def untouched_differences(
    clean: pd.DataFrame, corrupted: pd.DataFrame, subject: int = UNTOUCHED_SUBJECT
) -> dict[tuple[Any, ...], float]:
    """두 실행에서 손대지 않은 시험 피험자 subject의 행을 묶음별로 견준다. 묶음 → 확률의 최대 절대 차이.

    확률이 아닌 열(식별자, 스텝 번호, 시각, 라벨, perclos_60, 품질 판정, 온도, 표지)은 여기서 정확히 같은지 확인한다.
    """
    ours, theirs = subject_rows(clean, subject), subject_rows(corrupted, subject)
    assert sorted(ours) == sorted(theirs)
    worst: dict[tuple[Any, ...], float] = {}
    for key, a in ours.items():
        b = theirs[key]
        assert len(a) > 0, key
        pd.testing.assert_frame_equal(a[OTHER_COLUMNS], b[OTHER_COLUMNS], check_exact=True, obj=f"group {key}")
        worst[key] = float(np.max(np.abs(a[PROB_COLUMNS].to_numpy() - b[PROB_COLUMNS].to_numpy())))
    return worst


def runner_log(runs: SimpleNamespace) -> str:
    """세 프로세스가 남긴 출력의 끝부분(명령이 0이 아닌 코드로 끝났을 때 이유를 보여 준다)."""
    return "\n".join(
        f"[{name}] " + (runs.tmp / name / "runner.log").read_text(encoding="utf-8", errors="replace")[-2000:]
        for name in DATASETS
    )


def test_the_corruption_reaches_every_source_of_the_test_fold_and_nothing_else(runs: SimpleNamespace) -> None:
    """바꿔치기가 뜻한 대로 됐는지: 25번의 스텝만, 세 출처(원본, 15 fps 모사, 프레임 누락 변형) 모두에서 달라졌다."""
    assert runs.codes == {name: [0] * len(COMMANDS) for name in DATASETS}, runner_log(runs)
    assert official_fold(CORRUPTED_SUBJECT) == official_fold(UNTOUCHED_SUBJECT) == TEST_FOLD
    assert [official_fold(subject) for subject in SUBJECTS] == [1, 2, 3, 3]
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for name in ("clean", "corrupted"):
        cfg = train.with_data_root(load_config(runs.config), runs.roots[name])
        data = train.load_cv_data(cfg, (ARM,), mix_arms=(ARM,), fps_eval_arms=(ARM,))  # 명령이 만든 캐시에서 읽는다
        tables[name] = {
            "native": data.tables[ARM].steps,
            "fps15": data.extra_tables[ARM].steps,
            "frame_drop_v0": data.frame_drop_tables[ARM][0].steps,
        }
    features = list(FEATURE_NAMES)
    for source in ("native", "fps15", "frame_drop_v0"):
        clean, corrupted = tables["clean"][source], tables["corrupted"][source]
        kept = clean["subject"] != CORRUPTED_SUBJECT
        assert set(clean.loc[kept, "subject"]) == set(SUBJECTS) - {CORRUPTED_SUBJECT}
        pd.testing.assert_frame_equal(
            clean[kept].reset_index(drop=True),
            corrupted[corrupted["subject"] != CORRUPTED_SUBJECT].reset_index(drop=True),
        )
        ours = clean[~kept].reset_index(drop=True)
        theirs = corrupted[corrupted["subject"] == CORRUPTED_SUBJECT].reset_index(drop=True)
        assert len(ours) > 0 and len(theirs) > 0, source
        # 라벨(파일 이름에서 온다)은 그대로인데 내용은 다른 상태의 다른 얼굴이다: 세션마다 특징의 평균이 달라졌다.
        mean_ours = ours.groupby("session_id")[features].mean()
        mean_theirs = theirs.groupby("session_id")[features].mean()
        assert sorted(mean_ours.index) == sorted(mean_theirs.index) and len(mean_ours) == 3, source  # 세션 셋
        differing = (~np.isclose(mean_ours.to_numpy(), mean_theirs.loc[mean_ours.index].to_numpy())).sum(axis=1)
        assert (differing >= 5).all(), source
    # 세 출처는 서로 다른 표다(같은 표를 세 번 본 것이 아니다).
    assert len({train.step_table_fingerprint(table) for table in tables["clean"].values()}) == 3
    # 시험 fold의 창 수도 달라졌다: 25번의 평가 창이 늘었고 26번의 창은 그대로다.
    counts = {
        name: make_windows(tables[name]["native"], 120, 1).meta["subject"].value_counts().to_dict()
        for name in ("clean", "corrupted")
    }
    assert counts["corrupted"][CORRUPTED_SUBJECT] > counts["clean"][CORRUPTED_SUBJECT] > 0
    assert counts["corrupted"][UNTOUCHED_SUBJECT] == counts["clean"][UNTOUCHED_SUBJECT] > 0


def test_neural_network_selection_and_fold_models_do_not_depend_on_the_test_fold(runs: SimpleNamespace) -> None:
    clean, corrupted = selection(runs, "clean"), selection(runs, "corrupted")
    assert sorted(clean) == sorted(corrupted) == [1, 2, 3] and all(len(records) == 1 for records in clean.values())
    ours, theirs = clean[TEST_FOLD][0], corrupted[TEST_FOLD][0]
    assert ours["test_subjects"] == [CORRUPTED_SUBJECT, UNTOUCHED_SUBJECT] and ours["train_subjects"] == [1, 13]

    # 시험 fold의 데이터가 통째로 바뀌어도 그 fold의 기록은 한 글자도 다르지 않다.
    assert set(ours) == {"fold", "test_subjects", "train_subjects", "inner_val_folds", "candidates", "chosen", "refits"}
    assert ours["candidates"] == theirs["candidates"]  # 안쪽 선택: 내부 검증 macro-F1, 학습 길이, 온도
    assert ours["chosen"] == theirs["chosen"]
    assert ours["refits"] == theirs["refits"]  # 다시 학습: 표준화 상수, 온도, 가중치 해시
    assert ours == theirs
    # 그 비교가 실제로 무엇을 담고 있는지: 계열마다 다시 학습한 모델, 세 출처를 정해진 몫으로 섞은 학습.
    assert {item["name"] for item in ours["refits"]} == set(ours["chosen"].values()) and len(ours["refits"]) >= 2
    for item in ours["refits"]:
        assert item["sources"] == [["native", 0.5], ["fps15", 0.25], ["frame_drop_v0", 0.25]]
        assert len(item["mean"]) == len(item["std"]) == len(FEATURE_NAMES) and len(item["weights_sha256"]) == 64
        assert item["temperature"] > 0.0
    assert all(len(candidate["fold_macro_f1"]) == 2 for candidate in ours["candidates"])

    # 대조군: 25번이 학습·검증에 들어가는 fold는 표준화 상수와 가중치가 달라진다.
    for fold in OTHER_FOLDS:
        a, b = clean[fold][0], corrupted[fold][0]
        assert CORRUPTED_SUBJECT in a["train_subjects"] and a != b
        assert len(a["refits"]) == len(b["refits"]) > 0
        for item_a, item_b in zip(a["refits"], b["refits"]):
            if item_a["name"] == item_b["name"]:
                assert item_a["weights_sha256"] != item_b["weights_sha256"]
                assert item_a["mean"] != item_b["mean"] and item_a["std"] != item_b["std"]
        assert [c["temperature"] for c in a["candidates"]] != [c["temperature"] for c in b["candidates"]]


def test_lightgbm_selection_and_fold_models_do_not_depend_on_the_test_fold(runs: SimpleNamespace) -> None:
    clean, corrupted = selection(runs, "clean", "lgbm"), selection(runs, "corrupted", "lgbm")
    assert sorted(clean) == [1, 2, 3] and all(len(records) == 1 for records in clean.values())
    ours, theirs = clean[TEST_FOLD][0], corrupted[TEST_FOLD][0]
    assert ours["test_subjects"] == [CORRUPTED_SUBJECT, UNTOUCHED_SUBJECT]
    assert ours["candidates"] == theirs["candidates"] and ours["refit"] == theirs["refit"] and ours == theirs
    assert len(ours["refit"]["model_sha256"]) == 64 and ours["refit"]["n_columns"] == 102
    for fold in OTHER_FOLDS:
        a, b = clean[fold][0], corrupted[fold][0]
        assert a != b and a["refit"]["model_sha256"] != b["refit"]["model_sha256"]
    # 속도 특징을 뺀 부가 실행은 그 특징의 통계 12개(특징 둘 × 통계 여섯)가 없는 모델이다. 다른 모델이 같은 비교를 통과한다.
    without_velocity = selection(runs, "clean", "lgbm_no_velocity")[TEST_FOLD][0]
    assert without_velocity["refit"]["n_columns"] == 102 - 12
    assert without_velocity["refit"]["model_sha256"] != ours["refit"]["model_sha256"]


def test_foldwise_global_constants_do_not_depend_on_the_test_fold(
    runs: SimpleNamespace, pipeline_spec: PipelineSpec
) -> None:
    clean = selection(runs, "clean", "lgbm_fold_global")
    corrupted = selection(runs, "corrupted", "lgbm_fold_global")
    # fold마다 기록이 둘이다: feature_spec 상수의 묶음(주)과 fold별 상수의 묶음(부가).
    assert all([record.get("feature_source") for record in clean[fold]] == [None, "fold_global"] for fold in (1, 2, 3))
    ours, theirs = clean[TEST_FOLD], corrupted[TEST_FOLD]
    assert ours == theirs
    constants = ours[1]["global_constants"]
    assert constants == theirs[1]["global_constants"] and constants["train_subjects"] == [1, 13]
    assert ours[1]["refit"] == theirs[1]["refit"] and ours[1]["candidates"] == theirs[1]["candidates"]
    info = {
        name: json.loads((runs.reports / name / "lgbm_fold_global" / "run.json").read_text(encoding="utf-8"))
        for name in ("clean", "corrupted")
    }
    fold_key = str(TEST_FOLD)
    assert info["clean"]["fold_global_constants"][fold_key] == info["corrupted"]["fold_global_constants"][fold_key]
    assert info["clean"]["fold_global_constants"][fold_key] == constants
    # 그 fold의 스텝 표는 달라졌다(시험 피험자의 특징이 다르다). 상수와 모델이 같은 것은 그 표의 학습 피험자 쪽만 썼기 때문이다.
    name = f"global [fold {TEST_FOLD} constants]"
    assert info["clean"]["step_fingerprints"][name] != info["corrupted"]["step_fingerprints"][name]

    # 대조군: 25번이 학습 피험자인 fold는 상수와 모델이 달라진다.
    for fold in OTHER_FOLDS:
        a, b = clean[fold][1], corrupted[fold][1]
        assert CORRUPTED_SUBJECT in a["global_constants"]["train_subjects"]
        assert a["global_constants"] != b["global_constants"]
        assert a["refit"]["model_sha256"] != b["refit"]["model_sha256"]
    # 전체 피험자로 계산한 상수(시험 피험자가 들어간 값)는 바꿔치기에 따라 달라진다. fold 3의 상수는 그 값이 아니다.
    everyone: dict[str, tuple[float, float, float]] = {}
    for run_name in ("clean", "corrupted"):
        cfg = train.with_data_root(load_config(runs.config), runs.roots[run_name])
        quality = train.load_cv_data(cfg, ("global",)).tables["global"].quality
        pooled = signal_check.global_constants(quality, pipeline_spec)
        everyone[run_name] = (pooled.e_l, pooled.b_l, pooled.jaw0)
    assert everyone["clean"] != everyone["corrupted"]
    assert (constants["e"], constants["b"], constants["jaw0"]) != everyone["clean"]


@pytest.mark.parametrize("report", REPORTS, ids=REPORT_IDS)
def test_selection_records_of_the_test_fold_are_identical_in_every_run(runs: SimpleNamespace, report: Report) -> None:
    """부가 실행과 다른 조건의 실행을 포함한 모든 명령에서: 바깥 fold 3의 선택 기록은 같고 나머지 fold의 기록은 다르다."""
    parts = (report.subdir,) if report.subdir else ()
    clean = selection(runs, "clean", *parts, suffix=report.suffix)
    corrupted = selection(runs, "corrupted", *parts, suffix=report.suffix)
    assert sorted(clean) == sorted(corrupted) == [1, 2, 3]
    assert [len(clean[fold]) for fold in (1, 2, 3)] == [report.n_records] * 3
    for ours, theirs in zip(clean[TEST_FOLD], corrupted[TEST_FOLD], strict=True):
        assert ours["test_subjects"] == [CORRUPTED_SUBJECT, UNTOUCHED_SUBJECT]
        assert len(ours["candidates"]) >= 1 and ("refits" in ours or "refit" in ours)
        assert ours == theirs
    for fold in OTHER_FOLDS:
        for ours, theirs in zip(clean[fold], corrupted[fold], strict=True):
            assert CORRUPTED_SUBJECT not in ours["test_subjects"] and ours != theirs


@pytest.mark.parametrize("report", REPORTS, ids=REPORT_IDS)
def test_predictions_of_the_untouched_test_subject_are_identical_in_every_group(
    runs: SimpleNamespace, report: Report
) -> None:
    """25번의 데이터가 통째로 바뀌어도 같은 시험 fold의 26번에 대한 예측은 모든 묶음에서 그대로다.

    평가하는 시점의 누설(시험 fold의 창으로 다시 표준화, 시험 fold의 라벨로 온도 보정)은 선택 기록에는 흔적이 없고
    여기서만 드러난다.
    """
    clean = predictions(runs, "clean", report.suffix, report.subdir)
    corrupted = predictions(runs, "corrupted", report.suffix, report.subdir)
    assert len(clean[list(evaluate.GROUP_KEYS)].drop_duplicates()) == report.n_groups
    assert set(clean.loc[clean["fold"] == TEST_FOLD, "subject"]) == {CORRUPTED_SUBJECT, UNTOUCHED_SUBJECT}

    worst = untouched_differences(clean, corrupted)
    assert len(worst) == report.n_groups  # 시드가 하나라 묶음마다 실행 하나다. 26번의 행이 모든 묶음에 있다
    assert max(worst.values()) <= PROBABILITY_TOLERANCE, {key: value for key, value in worst.items() if value > 0.0}

    # 대조군: 바뀐 데이터는 평가에 실제로 쓰였다. 25번의 행은 모든 묶음에서 수나 값(특징에서 온 perclos_60)이 다르다.
    ours, theirs = subject_rows(clean, CORRUPTED_SUBJECT), subject_rows(corrupted, CORRUPTED_SUBJECT)
    assert sorted(ours) == sorted(theirs) == sorted(worst)
    for key, a in ours.items():
        b = theirs[key]
        assert len(a) > 0 and len(b) > 0, key
        assert len(a) != len(b) or not np.allclose(a["perclos_60"].to_numpy(), b["perclos_60"].to_numpy()), key


# ---------------------------------------------------------------------------
# 세 번째 데이터셋(relabelled): 평가받는 피험자 자신의 라벨이 그 피험자의 예측에 닿는가
# ---------------------------------------------------------------------------

LABEL_COLUMNS = ["video_id", "session_id", "label", "class_index"]  # 스텝 표에서 파일 이름(라벨)에서 오는 열


def video_name(subject: int, class_index: int) -> str:
    """논리 영상의 이름(`dataset.build_manifest`의 `video_id`): 피험자 26번의 drowsy 영상은 `S26_10`."""
    return f"S{subject:02d}_{synth.LABELS[class_index]}"


# relabelled 실행의 영상 이름 → 같은 내용이 clean 실행에서 갖는 영상 이름(맞바꾸기라 거꾸로 읽어도 같다).
CLEAN_VIDEO = {
    video_name(RELABELLED_SUBJECT, SWAPPED_CLASS.get(index, index)): video_name(RELABELLED_SUBJECT, index)
    for index in range(len(synth.LABELS))
}


def with_clean_names(rows: pd.DataFrame) -> pd.DataFrame:
    """relabelled 실행의 26번 행에 clean에서의 영상 이름과 클래스를 되돌려 붙인다(내용이 같은 행끼리 맞춘다).

    영상 이름과 클래스만 바꾼다. 나머지 열(부분 번호, 스텝 번호, 시각, 확률, perclos_60 등)은 그대로 둔다.
    """
    out = rows.copy()
    out["video_id"] = out["video_id"].map(CLEAN_VIDEO)
    assert out["video_id"].notna().all()
    out["class_index"] = out["class_index"].map(lambda value: SWAPPED_CLASS.get(int(value), int(value))).astype("int64")
    return out


def test_the_relabelling_exchanges_only_the_labels_of_one_test_subject(runs: SimpleNamespace) -> None:
    """맞바꾸기가 뜻한 대로 됐는지: 26번의 두 영상에 붙는 라벨만 바뀌었고 스텝 특징은 한 값도 다르지 않다."""
    assert CLEAN_VIDEO == {"S26_0": "S26_0", "S26_10": "S26_5", "S26_5": "S26_10"}
    for folder in ("synth", "synth_fps15"):
        names = {
            name: sorted(path.name for path in (runs.tmp / name / folder).rglob(f"{RELABELLED_SUBJECT:02d}/*.parquet"))
            for name in ("clean", "relabelled")
        }
        # 26번은 drowsy 영상이 두 부분으로 나뉜 피험자다. 맞바꾼 쪽에서는 low_vigilant 영상이 두 부분이다.
        assert names["clean"] == ["0.parquet", "10_1.parquet", "10_2.parquet", "5.parquet"], folder
        assert names["relabelled"] == ["0.parquet", "10.parquet", "5_1.parquet", "5_2.parquet"], folder

    features = [name for name in STEP_COLUMNS if name not in LABEL_COLUMNS]
    for arm in (ARM, OTHER_ARM):
        tables: dict[str, dict[str, pd.DataFrame]] = {}
        for name in ("clean", "relabelled"):
            cfg = train.with_data_root(load_config(runs.config), runs.roots[name])
            data = train.load_cv_data(cfg, (arm,), fps_eval_arms=(arm,))  # 명령이 만든 캐시에서 읽는다
            tables[name] = {"native": data.tables[arm].steps, "fps15": data.extra_tables[arm].steps}
        for source in ("native", "fps15"):
            clean, relabelled = tables["clean"][source], tables["relabelled"][source]
            # 다른 피험자의 스텝은 그대로다.
            kept = clean["subject"] != RELABELLED_SUBJECT
            pd.testing.assert_frame_equal(
                clean[kept].reset_index(drop=True),
                relabelled[relabelled["subject"] != RELABELLED_SUBJECT].reset_index(drop=True),
            )
            ours = clean[~kept]
            theirs = relabelled[relabelled["subject"] == RELABELLED_SUBJECT]
            # 26번의 라벨별 스텝 수가 달라졌다: clean에서는 drowsy가 두 세션, relabelled에서는 low_vigilant가 두 세션이다.
            counts = ours.groupby("class_index").size().to_dict()
            assert counts[1] > 0 and counts[2] > counts[1], (arm, source)
            assert theirs.groupby("class_index").size().to_dict() == {0: counts[0], 1: counts[2], 2: counts[1]}
            # 이름을 되돌려 같은 내용끼리 맞추면 특징, 스텝 번호, 시각이 한 값도 다르지 않다(alert 영상 기준선도 같다).
            restored = with_clean_names(theirs)
            order = ["video_id", "part", "k"]
            pd.testing.assert_frame_equal(
                ours.sort_values(order)[features].reset_index(drop=True),
                restored.sort_values(order)[features].reset_index(drop=True),
                check_exact=True,
                obj=f"{arm} {source}",
            )
            assert sorted(restored["class_index"].unique()) == [0, 1, 2]


@pytest.mark.parametrize("report", REPORTS, ids=REPORT_IDS)
def test_selection_records_of_the_test_fold_do_not_depend_on_its_labels(runs: SimpleNamespace, report: Report) -> None:
    """26번의 라벨을 맞바꿔도 바깥 fold 3의 선택 기록은 같다. 26번이 학습·검증에 들어가는 fold 1, 2의 기록은 다르다."""
    parts = (report.subdir,) if report.subdir else ()
    clean = selection(runs, "clean", *parts, suffix=report.suffix)
    relabelled = selection(runs, "relabelled", *parts, suffix=report.suffix)
    assert sorted(clean) == sorted(relabelled) == [1, 2, 3]
    for ours, theirs in zip(clean[TEST_FOLD], relabelled[TEST_FOLD], strict=True):
        assert RELABELLED_SUBJECT in ours["test_subjects"] and ours == theirs
    # 대조군: 바뀐 라벨은 실제로 쓰였다. 26번이 학습 피험자인 fold의 모델과 선택 기록이 달라졌다.
    for fold in OTHER_FOLDS:
        for ours, theirs in zip(clean[fold], relabelled[fold], strict=True):
            assert RELABELLED_SUBJECT not in ours["test_subjects"] and ours != theirs


@pytest.mark.parametrize("report", REPORTS, ids=REPORT_IDS)
def test_predictions_do_not_depend_on_the_labels_of_the_evaluated_subject(
    runs: SimpleNamespace, report: Report
) -> None:
    """26번의 라벨 파일만 맞바꾼 실행: 내용이 같은 행끼리 견주면 26번의 확률은 모든 묶음에서 clean 실행과 같다.

    평가받는 피험자 자신의 시험 라벨을 쓰는 누설(피험자별 온도 재보정, 정답 라벨을 섞은 예측, 라벨로 고른 임계값)은
    그 피험자의 데이터가 같은 두 실행(clean, corrupted)의 비교에는 보이지 않고 여기서만 드러난다.
    """
    clean = predictions(runs, "clean", report.suffix, report.subdir)
    relabelled = predictions(runs, "relabelled", report.suffix, report.subdir)
    assert len(relabelled[list(evaluate.GROUP_KEYS)].drop_duplicates()) == report.n_groups

    ours = subject_rows(clean, RELABELLED_SUBJECT)
    raw = relabelled[relabelled["subject"] == RELABELLED_SUBJECT]
    # 맞바꾸기가 예측 표까지 왔다: 26번은 바깥 fold 3에서만 평가되고, 두 클래스의 행 수가 서로 바뀌었다(두 부분으로 나뉜
    # 영상이 clean에서는 drowsy, relabelled에서는 low_vigilant다).
    assert set(raw["fold"]) == {TEST_FOLD}
    rows_of = {name: table[table["subject"] == RELABELLED_SUBJECT].groupby("class_index").size().to_dict()
               for name, table in (("clean", clean), ("relabelled", relabelled))}  # fmt: skip
    assert rows_of["clean"][2] > rows_of["clean"][1] > 0
    assert rows_of["relabelled"] == {0: rows_of["clean"][0], 1: rows_of["clean"][2], 2: rows_of["clean"][1]}

    theirs = subject_rows(with_clean_names(raw), RELABELLED_SUBJECT)
    assert sorted(ours) == sorted(theirs) and len(ours) == report.n_groups
    worst: dict[tuple[Any, ...], float] = {}
    for key, a in ours.items():
        b = theirs[key]
        assert len(a) == len(b) > 0, key
        # 확률이 아닌 열(스텝 번호, 시각, perclos_60, 품질 판정, 온도, 표지)은 정확히 같다: 같은 프레임의 행을 맞췄다.
        pd.testing.assert_frame_equal(a[OTHER_COLUMNS], b[OTHER_COLUMNS], check_exact=True, obj=f"group {key}")
        worst[key] = float(np.max(np.abs(a[PROB_COLUMNS].to_numpy() - b[PROB_COLUMNS].to_numpy())))
    assert max(worst.values()) <= PROBABILITY_TOLERANCE, {key: value for key, value in worst.items() if value > 0.0}

    # 같은 시험 fold의 25번은 이 데이터셋에서 데이터도 라벨도 그대로다. 그 행도 모든 묶음에서 같다.
    same_fold = untouched_differences(clean, relabelled, CORRUPTED_SUBJECT)
    assert len(same_fold) == report.n_groups and max(same_fold.values()) <= PROBABILITY_TOLERANCE
