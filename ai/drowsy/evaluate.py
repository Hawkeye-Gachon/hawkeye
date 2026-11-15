"""평가 지표와 리포트 (PLAN 6장과 1.2절의 결정 D7, D8, D9, D11. 번호의 뜻은 아래 "결정 번호"). 이식 대상이 아니다.

파일을 쓰는 것은 `write_reports`(교차검증 리포트)와 `run_unlabeled`(라벨 없는 영상의 경고 수 리포트)뿐이고, 나머지는
모두 표와 설정만 받는 순수 함수다. 학습 코드는 아래 형식의 예측 표만 만들면 된다(`prediction_table`이 창 묶음과
확률에서 만들어 준다). 모든 모델 계열이 같은 형식을 쓴다.

결정 번호: 이 파일의 주석과 리포트 문구에 나오는 `D숫자`는 계획서(`docs/PLAN.md`) 1.2절의 표에서 승인된 결정이다.
- D3: 경고의 속도 억제. 시선 이탈·마이크로슬립·수면 경고는 20 km/h 미만, 졸음 경고는 50 km/h 미만에서 내지 않는다.
- D4: 기준선이 완성되기 전에도 감김 규칙은 전역 상수로 계산한 임시 closure로 동작하고 경고를 낸다.
- D7: 영상 단위 정확도의 분모는 논리 영상 전체이고, 창이 하나도 없는 영상은 오답으로 센다. 품질 기준(얼굴 검출 비율
  0.7 이상이고 창 1개 이상)을 통과한 영상만의 값은 보조 행으로 따로 싣는다.
- D8: 2클래스(alert 대 drowsy) 값은 3클래스 모델을 그대로 쓰고 alert·drowsy 영상에서 두 확률 중 큰 쪽으로 판정한다.
- D9: "시간당 졸음 오판정 수"의 주 수치는 판정이 시작된 횟수이고, 창 단위 값(drowsy로 분류된 창 수 / 시간)도 싣는다.
- D10: 보고하는 수치는 같은 절차의 교차검증 추정치이고, 내보낸 모델 파일 자체를 시험한 값이 아니다.
- D11: 교차검증 결과가 정해 둔 기준을 모두 만족할 때만 학습 모델을 규칙 엔진의 기본 졸음 판정원으로 둔다(`model_gate`).
- D13: 얼굴 프레임(화소)은 어디에도 저장하지 않는다.
- D16: INT8 양자화 모델의 정확도 변화를 보고한다(부가 결과).
계획서 14장에 실린 소유자 결정(2026-11-13, 2026-11-14)은 따로 번호를 매기며 이 번호와 다르다. 이 파일에서는 그 결정을
"2026-11-13 소유자 결정 2"처럼 날짜를 붙여 적는다. D 번호를 쓰는 리포트는 머리말에 D 번호가 계획서 1.2절의 결정 번호라는
한 줄(`PLAN_DECISION_NOTE`)을 싣는다: 이 파일의 두 리포트(`write_reports`, `run_unlabeled`)의 summary.md, 그리고 같은 줄을
가져다 쓰는 `train final`의 final/summary.md(`train.fit_final`)와 `baseline_lgbm cv --all-w`의 lgbm_w_grid/w_grid.md
(`baseline_lgbm.run_lgbm_w_grid`).

예측 표: 한 행 = 평가한 창 하나. 열은 `PREDICTION_COLUMNS`, dtype은 `PREDICTION_DTYPES`.
    run_id          str      학습 실행의 이름. 무엇을 어떻게 학습했는지를 가리킨다(같은 계열도 학습 조건이 다르면 다른 run_id)
    family          str      모델 계열: tcn, gru, lgbm, selected(바깥 fold마다 안에서 고른 것) 등
    arm             str      이 창의 특징을 만든 정규화 조건(alert_baseline, global, session_baseline).
                             학습 조건이 아니라 평가에 쓴 특징의 조건이다. alert 영상 기준선으로 학습한 모델을 자기 세션
                             기준선 특징으로 평가한 "앱 조건 평가"(6장)는 run_id가 같고 arm만 session_baseline이다
    fold            int64    바깥(시험) fold. 이 창의 피험자가 시험 대상이었던 fold
    seed            int64    다시 학습할 때의 시드
    subject         int64    피험자 번호
    video_id        str      논리 영상(`S01_0`, `S32_10`). 나뉜 영상의 부분들은 같은 video_id를 쓴다
    part            int64    나뉜 영상의 부분 번호(나뉘지 않았으면 0)
    class_index     int64    정답 클래스(0 alert, 1 low_vigilant, 2 drowsy)
    k               int64    창의 마지막 스텝 번호(`WindowSet.meta`의 k_end)
    t_end_ms        int64    그 스텝의 끝 시각(원래 t_ms 시계)
    p_alert, p_low_vigilant, p_drowsy   float64   모델 출력 확률(온도 적용 뒤, 합 1)
    perclos_60      float64  창의 마지막 스텝의 perclos_60(원시 값)
    gate_pass       bool     이 논리 영상이 D7 품질 기준을 통과했는지
    temperature     float64  이 확률에 적용된 온도. 보정하지 않았으면 1.0 (보정 전후 ECE에 쓴다)
  표지 열(`LABEL_COLUMNS`). 없는 표도 받는다(`validate_predictions`가 채운다. 예전 예측 표가 그대로 통한다)
    feature_source  str      특징을 만든 추출·상수의 출처. 기본 `native`(원본 프레임률 추출, spec의 상수). 그 밖의 예:
                             `fps15`(시험 fold의 15 fps 모사 추출), `fold_global`(fold별로 다시 계산한 전역 상수)
    variant         str      학습 실행의 선택 사항. 기본 `base`. 그 밖의 예: `ordinal`(순서형 소프트 라벨),
                             `no_velocity`(속도 특징 #7, #8 제외)
    supplementary   bool     부가 결과(PLAN 11장 E3의 부가 실험, INT8 양자화 비교)이면 True. 값이 없으면 위 두 표지가 모두
                             기본값일 때 False, 아니면 True다. 표지가 기본값이 아닌데 False인 묶음은 받지 않는다
                             (`validate_predictions`). 부가 묶음은 리포트에서 "부가" 제목 아래에 따로 실리고, 주 결과
                             표·비교표·D11 판정에는 쓰이지 않는다
  family는 자유 문자열이다. INT8 양자화 모델의 예측은 `tcn_int8`처럼 원래 계열에 `_int8`을 붙여 적는다(`INT8_SUFFIX`).
가정: 계획서는 표의 모양을 정하지 않았다. `temperature` 열은 "보정 전후 ECE"를 예측 표만으로 계산하려고 둔 것이다
(softmax(z/T)를 T제곱해 정규화하면 softmax(z)가 된다. `untemper`).
가정: 계획서는 부가 분석을 "부가로 표시"하라고만 했다(11장 E3). 표지 열 셋과 그 기본값, `_int8` 접미사는 여기서 정한 것이다.

용어
- 묶음(group): (run_id, family, arm, feature_source, variant). 시드와 fold를 모은 리포트의 한 단위다(`evaluate_group`).
  묶음 키(`GroupKey`)는 feature_source와 variant가 기본값이면 (run_id, family, arm) 세 값으로 줄여 적는다(`group_key`).
  그래서 표지 열이 생기기 전의 키가 그대로 통한다.
- 실행(run): 묶음 하나의 시드 하나. 지표 함수(`window_metrics`, `video_vote`, `two_class`, `alarm_report`)는
  실행 하나의 표만 받는다. 섞여 있으면 ValueError다(시드가 다른 창을 한 영상의 표로 합치지 않게 한다). `iter_runs`로 나눈다.
- 영상 표(video table): 논리 영상마다 한 행, 열은 `VIDEO_COLUMNS`. 영상 단위 지표의 분모다(D7). 예측 표에는 창이
  하나도 없는 영상의 행이 없으므로, 그런 영상을 오답으로 세려면 이 표를 함께 넘긴다(`video_table`이 품질 표에서 만든다).
- 논리 영상은 (subject, video_id)로 구분한다. 부트스트랩이 같은 피험자를 두 번 뽑으면 `subject`만 바꿔 서로 다른
  영상으로 세기 위해서다.

지표의 정의
- 창의 예측 클래스는 확률이 가장 큰 클래스다.
가정: 확률이 정확히 같으면 클래스 순서가 앞선 쪽이다.
가정: 0으로 나누게 되는 비율(영상이 없는 클래스의 재현율, 평가 시간이 0일 때의 시간당 횟수 등)은 0.0이다.
  PLAN 3.3절이 특징에 정한 규칙을 지표에도 쓴다.
가정: 표준편차는 표본 표준편차(n − 1로 나눔)이고 값이 하나뿐이면 NaN이다(리포트에는 n/a).

리포트 파일의 구성은 `write_reports`의 설명에 있다. 합성 데이터로 만든 파일은 모두 첫 줄에 그 사실을 적는다.

명령 (저장소 루트에서 실행)
    python -m drowsy.evaluate unlabeled --config config/default.yaml [--raw-dir DIR] [--model PATH]
                                        [--speed-kmh X] [--run-name NAME]
  라벨이 없는 영상(팀 수집 차량 영상)의 추출 결과를 `DriverPipeline`에 재생해 시간당 경고 수를 센다(PLAN 11장 M4).
  `reports/<NAME>/unlabeled/`에 `alerts.csv`와 `summary.md`(그리고 설정 사본 `config.yaml`, 규칙 사본 `rules.json`)를
  쓴다. 추출이 남긴 회전 점검 기록도 읽어, 자동 점검이 각도를 가르지 못했는데 수동 회전 보정이 없는 영상의 수를 한 줄로
  출력하고 그 목록을 `summary.md`의 "회전 점검" 절에 싣는다. 자세한 것은 `run_unlabeled`에 있다.
  종료 코드: 0 리포트를 씀, 2 시작할 수 없음(설정, 모델 메타데이터 불일치, 추출 결과 없음, 폴더 이름이 아닌
  `--run-name` 등).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drowsy import extract
from drowsy.config import EvaluationSettings, ExperimentConfig, ModelGate, check_run_name, load_config
from drowsy.dataset import (
    BASELINE_NONE,
    SYNTHETIC_NOTE,
    RotationSummary,
    SubjectOverlapError,
    WindowSet,
    assert_subject_disjoint,
    read_rotation_checks,
    rotation_check_line,
    rotation_check_section,
    scan_unlabeled,
    windows_per_class,
)
from drowsy.rules import StepHysteresis
from drowsy.spec import FeatureSpec, RulesConfig, export_rules_json, load_feature_spec, load_rules, rules_hash
from drowsy.types import (
    BASELINE_FALLBACK_GLOBAL,
    CLASS_ALERT,
    CLASS_DROWSY,
    CLASS_ORDER,
    DROWSY_SOURCE_MODEL,
    DROWSY_SOURCE_PERCLOS,
    DROWSY_SOURCES,
    EVENT_TYPES,
    F_PERCLOS_60,
    FEATURE_NAMES,
    PHASE_END,
    PHASE_ONSET,
)

N_CLASSES = len(CLASS_ORDER)
PROB_COLUMNS: tuple[str, ...] = tuple(f"p_{name}" for name in CLASS_ORDER)  # p_alert, p_low_vigilant, p_drowsy
PERCLOS_COLUMN = FEATURE_NAMES[F_PERCLOS_60]

# 표지 열의 값. feature_source와 variant는 자유 문자열이고(비어 있으면 안 된다) 아래는 이 저장소가 쓰는 이름이다.
FEATURE_SOURCE_NATIVE = "native"  # 기본: 원본 프레임률 추출의 스텝 표, spec의 상수
FEATURE_SOURCE_FPS15 = "fps15"  # 시험 fold의 15 fps 모사 추출(`extraction.drop_sim.subdir`의 기본 이름과 같다)
FEATURE_SOURCE_FOLD_GLOBAL = "fold_global"  # 학습 피험자만으로 fold마다 다시 계산한 전역 상수(PLAN 4.3절 민감도 점검)
VARIANT_BASE = "base"  # 기본: 사전에 고정한 절차 그대로
VARIANT_ORDINAL = "ordinal"  # 순서형 소프트 라벨(PLAN 5.2절 부가 실험)
VARIANT_NO_VELOCITY = "no_velocity"  # 속도 특징 #7, #8 제외(PLAN 4.6절)
INT8_SUFFIX = "_int8"  # INT8 양자화 모델의 계열 이름 = 원래 계열 + 이 접미사(PLAN 1.2절의 결정 D16: INT8 비교)
FAMILY_TCN_INT8 = "tcn" + INT8_SUFFIX

PREDICTION_DTYPES: dict[str, str] = {
    "run_id": "str",
    "family": "str",
    "arm": "str",
    "fold": "int64",
    "seed": "int64",
    "subject": "int64",
    "video_id": "str",
    "part": "int64",
    "class_index": "int64",
    "k": "int64",
    "t_end_ms": "int64",
    PROB_COLUMNS[0]: "float64",
    PROB_COLUMNS[1]: "float64",
    PROB_COLUMNS[2]: "float64",
    PERCLOS_COLUMN: "float64",
    "gate_pass": "bool",
    "temperature": "float64",
    "feature_source": "str",
    "variant": "str",
    "supplementary": "bool",
}
PREDICTION_COLUMNS: tuple[str, ...] = tuple(PREDICTION_DTYPES)  # 검사를 통과한 표의 열(이 순서)
# 표지 열과 그 기본값. 이 열이 없는 표(예전 형식)는 채워서 받는다. supplementary의 기본값 False는 두 표지가 모두
# 기본값인 행에만 쓰인다(표지가 기본값이 아닌 행은 True가 된다. `validate_predictions`)
LABEL_DEFAULTS: dict[str, Any] = {
    "feature_source": FEATURE_SOURCE_NATIVE,
    "variant": VARIANT_BASE,
    "supplementary": False,
}
LABEL_COLUMNS: tuple[str, ...] = tuple(LABEL_DEFAULTS)
REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = tuple(
    name for name in PREDICTION_COLUMNS if name not in LABEL_DEFAULTS
)  # 넘겨받는 표에 반드시 있어야 하는 열
GROUP_KEYS: tuple[str, ...] = ("run_id", "family", "arm", "feature_source", "variant")
RUN_KEYS: tuple[str, ...] = GROUP_KEYS + ("seed",)
VIDEO_KEYS: tuple[str, ...] = ("subject", "video_id")
VIDEO_COLUMNS: tuple[str, ...] = ("video_id", "subject", "fold", "class_index", "gate_pass")
# 영상 표가 품질 표에서 함께 옮겨 오는 기준선 열(PLAN 4.4절의 리포트 항목). 품질 표에 있는 것만 옮긴다
BASELINE_COLUMNS: tuple[str, ...] = ("baseline_status", "t_b_ms", "baseline_used", "alert_fallback")
NOMINAL_BASELINE_MS = 60_000  # 기준선의 기본 수집 시간(feature_spec의 baseline.duration_ms). 이보다 늦게 완성됐으면 "연장"이다
VOTE_COLUMNS: tuple[str, ...] = (
    ("subject", "video_id", "fold", "class_index", "gate_pass", "n_windows")
    + tuple(f"votes_{name}" for name in CLASS_ORDER)
    + tuple(f"mean_p_{name}" for name in CLASS_ORDER)
    + ("pred_class", "correct")
)
ALARM_COLUMNS: tuple[str, ...] = (
    "subject",
    "video_id",
    "fold",
    "class_index",
    "gate_pass",
    "n_steps",
    "evaluated_ms",
    "onsets",
    "alarm_steps",
    "flagged_windows",
)
SWEEP_COLUMNS: tuple[str, ...] = (
    "onset_p",
    "onsets",
    "alert_hours",
    "onsets_per_hour",
    "alert_videos_with_onset",
    "alert_video_onset_ratio",
    "alarm_time_share",
    "flagged_windows_per_hour",
    "drowsy_video_onset_ratio",
)

FAMILY_TCN = "tcn"
FAMILY_GRU = "gru"
FAMILY_LGBM = "lgbm"
FAMILY_SELECTED = "selected"  # 바깥 fold마다 안쪽 검증으로 고른 계열·설정(주 결과 행, PLAN 6장)

NO_PREDICTION = -1  # 표결 표의 pred_class: 창이 하나도 없는 영상
ECE_BINS = 15  # 가정은 `expected_calibration_error`에 적었다
PROB_SUM_TOLERANCE = 1e-3  # float32 소프트맥스 출력의 합 오차를 넉넉히 받는다
CONFIDENCE_LEVEL = 0.95
MS_PER_HOUR = 3_600_000.0
TWO_CLASSES: tuple[int, ...] = (CLASS_ALERT, CLASS_DROWSY)  # 2클래스 보고의 두 클래스(PLAN 1.2절의 결정 D8)

# 실제 데이터 리포트의 첫 줄: 수치는 교차검증 추정치이고 내보낸 파일을 시험한 값이 아니다(PLAN 1.2절의 결정 D10)
REAL_NOTE = "REAL data: cross-validated estimates, not a test of the exported model file"
# 리포트(summary.md)의 머리말에 넣는 한 줄: 리포트 문구에 나오는 D 번호가 무엇의 번호인지 알린다. 번호의 뜻은 이 파일
# 맨 위의 "결정 번호"에 있다
PLAN_DECISION_NOTE = (
    "이 리포트의 D 번호(D 뒤에 숫자가 붙은 표기)는 계획서 `docs/PLAN.md` 1.2절의 표에서 승인된 결정의 번호다. "
    "같은 문서 14장에 실린 소유자 결정(2026-11-13, 2026-11-14)은 따로 번호를 매기며, 그 번호와는 다르다."
)

# 묶음 키: (run_id, family, arm) 또는 (run_id, family, arm, feature_source, variant). `group_key`가 만든다
GroupKey = tuple[str, ...]
RunKey = tuple[Any, ...]  # 묶음 키 뒤에 seed(int)를 붙인 것


def group_key(
    run_id: str,
    family: str,
    arm: str,
    feature_source: str = FEATURE_SOURCE_NATIVE,
    variant: str = VARIANT_BASE,
) -> GroupKey:
    """묶음 키. feature_source와 variant가 기본값이면 `(run_id, family, arm)`, 아니면 다섯 값을 모두 적은 튜플이다.

    `GroupResult.key`, `write_reports`의 gate·paired, `supplementary_pairs`가 모두 이 꼴을 쓴다. 다섯 값을 적은 키도
    두 표지가 기본값이면 세 값짜리와 같은 묶음이다(`write_reports`는 받은 키를 이 함수로 다시 만들어 맞춘다).
    """
    if feature_source == FEATURE_SOURCE_NATIVE and variant == VARIANT_BASE:
        return (str(run_id), str(family), str(arm))
    return (str(run_id), str(family), str(arm), str(feature_source), str(variant))


def _canonical_key(key: Sequence[str]) -> GroupKey:
    if len(key) not in (3, len(GROUP_KEYS)):
        raise ValueError(
            f"a group key is (run_id, family, arm) or (run_id, family, arm, feature_source, variant), got {tuple(key)}"
        )
    return group_key(*key)


def _key_of(row: Mapping[str, Any]) -> GroupKey:
    """표의 한 행(또는 dict)에서 묶음 키를 만든다. 표지 열이 없으면 기본값이다."""
    return group_key(
        str(row["run_id"]),
        str(row["family"]),
        str(row["arm"]),
        str(row.get("feature_source", FEATURE_SOURCE_NATIVE)),
        str(row.get("variant", VARIANT_BASE)),
    )


def _present(names: Sequence[str], table: pd.DataFrame) -> list[str]:
    """names 중 표에 있는 열. 표지 열이 없는 표(검사를 거치지 않은 예전 형식)도 지표 함수가 받도록 쓴다."""
    return [name for name in names if name in table.columns or name not in LABEL_DEFAULTS]


# ---------------------------------------------------------------------------
# 예측 표
# ---------------------------------------------------------------------------


def _require_columns(table: pd.DataFrame, names: Iterable[str], what: str) -> None:
    missing = [name for name in names if name not in table.columns]
    if missing:
        raise ValueError(f"{what} lacks columns {missing}")


def _require_single(table: pd.DataFrame, keys: Sequence[str], what: str) -> None:
    """표가 keys의 한 가지 조합만 담고 있는지 검사한다."""
    if len(table) == 0:
        return
    for name in keys:
        values = table[name].to_numpy()
        if (values != values[0]).any():
            raise ValueError(
                f"{what} mixes several values of {name!r}; pass one {'/'.join(keys)} combination at a time "
                "(see iter_runs)"
            )


def _labelled_rows(table: pd.DataFrame) -> np.ndarray:
    """행마다 표지가 기본값이 아닌지(feature_source가 native가 아니거나 variant가 base가 아니면 True)."""
    source = table["feature_source"].to_numpy(dtype=object)
    variant = table["variant"].to_numpy(dtype=object)
    return np.asarray((source != FEATURE_SOURCE_NATIVE) | (variant != VARIANT_BASE), dtype=bool)


def validate_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """예측 표가 형식에 맞는지 검사하고, 열 순서와 dtype을 맞춘 사본을 돌려준다.

    검사: 열이 모두 있는지, 빈 값이 없는지, dtype(문자열·정수·실수·bool)의 종류, class_index가 0~2, 확률이 유한하고
    [0, 1] 안이며 합이 1(±0.001), perclos_60이 [0, 1], temperature > 0, k ≥ 1, 한 video_id의 subject·class_index·
    gate_pass가 하나씩인지, 같은 실행 안에서 창 (subject, video_id, part, k)이 겹치지 않는지. 어기면 ValueError다.
    형식에 없는 열은 뒤에 그대로 둔다. 행이 없는 표도 받는다.

    표지 열(`LABEL_COLUMNS`: feature_source, variant, supplementary)은 없어도 된다. feature_source와 variant의 없는
    열과 빈 값은 기본값(`LABEL_DEFAULTS`: native, base)으로 채운다. supplementary의 없는 열과 빈 값은 그 행의 두 표지에서
    정한다: 둘 다 기본값이면 False(주 결과), 하나라도 기본값이 아니면 True(부가 결과)다. 그래서 표지 열이 없는 예전
    예측 표와, 그런 표와 부가 묶음의 표를 `pd.concat`한 것(없는 쪽 행이 빈 값이 된다)을 그대로 받는다. 돌려주는 표에는
    항상 `PREDICTION_COLUMNS`가 모두 있다. feature_source와 variant는 빈 문자열일 수 없고, 한 묶음 (run_id, family, arm,
    feature_source, variant)의 supplementary는 한 값이어야 한다.
    feature_source가 native가 아니거나 variant가 base가 아닌 묶음은 주 결과가 될 수 없다. 그런 묶음의 supplementary가
    False이면 ValueError다(15 fps 모사 추출, 순서형 라벨, 속도 특징 제외 같은 묶음이 주 결과 표나 학습 모델을 기본
    졸음 판정원으로 둘지의 판정에 섞이는 것을 여기서 막는다).
    가정: 표지가 둘 다 기본값인 묶음의 부가 여부는 표지로 알 수 없다(ONNX로 다시 돌린 예측, INT8 양자화 모델의 예측,
      내보낼 조건이 아닌 정규화 조건의 실행은 계열 이름이나 실행 이름만 다르다). 그런 묶음은 적혀 있는 supplementary를
      그대로 따르고, 값이 없으면 주 결과(False)로 본다. 부르는 쪽이 True를 직접 적어야 한다.
    가정: gate_pass는 feature_source마다 한 값이면 된다(다른 추출의 품질 표에서 온 값이 다를 수 있다). 리포트에서는
      어느 묶음이든 `write_reports`에 넘긴 영상 표의 gate_pass를 쓴다(`_per_video`).
    """
    _require_columns(preds, REQUIRED_PREDICTION_COLUMNS, "prediction table")
    extra = [name for name in preds.columns if name not in PREDICTION_DTYPES]
    table = preds[[name for name in PREDICTION_COLUMNS if name in preds.columns] + extra].copy()
    for name in ("feature_source", "variant"):
        default = LABEL_DEFAULTS[name]
        if name not in table.columns:
            table[name] = default
            continue
        missing = table[name].isna().to_numpy()
        if missing.any():
            values = table[name].to_numpy(dtype=object, copy=True)
            values[missing] = default
            table[name] = values
    # 부가 표시가 없는 행은 표지에서 정한다(상수 False로 채우지 않는다): 표지가 기본값이 아니면 부가 결과다
    from_labels = _labelled_rows(table)
    if "supplementary" not in table.columns:
        table["supplementary"] = from_labels
    else:
        missing = table["supplementary"].isna().to_numpy()
        if missing.any():
            values = table["supplementary"].to_numpy(dtype=object, copy=True)
            values[missing] = from_labels[missing]
            table["supplementary"] = values
        if table["supplementary"].dtype == object:
            # bool 열과 빈 값을 이어 붙이면 object 열이 된다. 값이 모두 bool일 때만 bool 열로 되돌린다
            if all(isinstance(value, (bool, np.bool_)) for value in table["supplementary"].tolist()):
                table["supplementary"] = table["supplementary"].astype(bool)
    table = table[list(PREDICTION_COLUMNS) + extra]
    if len(table) > 0:
        nulls = [name for name in PREDICTION_COLUMNS if bool(table[name].isna().any())]
        if nulls:
            raise ValueError(f"prediction table has missing values in columns {nulls}")
        for name, dtype in PREDICTION_DTYPES.items():
            column = table[name]
            if dtype == "str":
                matches = pd.api.types.is_string_dtype(column)
            elif dtype == "int64":
                matches = pd.api.types.is_integer_dtype(column)
            elif dtype == "bool":
                matches = pd.api.types.is_bool_dtype(column)
            else:
                matches = pd.api.types.is_float_dtype(column) or pd.api.types.is_integer_dtype(column)
            if not matches:
                raise ValueError(f"prediction table column {name!r} must be {dtype}, got {column.dtype}")
    table = table.astype(PREDICTION_DTYPES).reset_index(drop=True)

    classes = table["class_index"].to_numpy()
    if ((classes < 0) | (classes >= N_CLASSES)).any():
        raise ValueError(f"class_index must be within 0..{N_CLASSES - 1}")
    probs = table[list(PROB_COLUMNS)].to_numpy()
    if not np.isfinite(probs).all() or (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    if (np.abs(probs.sum(axis=1) - 1.0) > PROB_SUM_TOLERANCE).any():
        raise ValueError(f"{', '.join(PROB_COLUMNS)} must sum to 1 in every row")
    perclos = table[PERCLOS_COLUMN].to_numpy()
    if not np.isfinite(perclos).all() or (perclos < 0.0).any() or (perclos > 1.0).any():
        raise ValueError(f"{PERCLOS_COLUMN} must be finite and within [0, 1]")
    temperature = table["temperature"].to_numpy()
    if not np.isfinite(temperature).all() or (temperature <= 0.0).any():
        raise ValueError("temperature must be finite and positive (1.0 when no temperature was applied)")
    if (table["k"].to_numpy() < 1).any():
        raise ValueError("k (last step of the window) must be >= 1")
    for name in ("feature_source", "variant"):
        if (table[name].str.strip() == "").any():
            raise ValueError(f"{name} must not be empty (default: {LABEL_DEFAULTS[name]!r})")

    per_video = table.groupby("video_id", sort=False)[["subject", "class_index"]].nunique()
    inconsistent = per_video.index[(per_video > 1).any(axis=1)].tolist()
    per_source = table.groupby(["feature_source", "video_id"], sort=False)["gate_pass"].nunique()
    inconsistent += [video_id for _, video_id in per_source.index[per_source.to_numpy() > 1].tolist()]
    if inconsistent:
        raise ValueError(
            f"a video_id must have one subject, class_index and gate_pass; inconsistent: {inconsistent[:10]}"
        )
    flags = table.groupby(list(GROUP_KEYS), sort=False)["supplementary"].nunique()
    if (flags.to_numpy() > 1).any():
        mixed = [group_key(*key) for key in flags.index[flags.to_numpy() > 1].tolist()]
        raise ValueError(f"a group must be supplementary in all of its rows or in none; mixed: {mixed[:10]}")
    unflagged = _labelled_rows(table) & ~table["supplementary"].to_numpy(dtype=bool)
    if unflagged.any():
        keys = table.loc[unflagged, list(GROUP_KEYS)].drop_duplicates().itertuples(index=False, name=None)
        named = [group_key(*key) for key in keys]
        raise ValueError(
            f"a group whose feature_source is not {FEATURE_SOURCE_NATIVE!r} or whose variant is not {VARIANT_BASE!r} "
            f"is never a main result and must have supplementary = True; not flagged: {named[:10]}"
        )
    if table.duplicated(list(RUN_KEYS) + ["subject", "video_id", "part", "k"]).any():
        raise ValueError("prediction table holds the same window (run, subject, video_id, part, k) more than once")
    return table


def iter_runs(preds: pd.DataFrame) -> Iterator[tuple[RunKey, pd.DataFrame]]:
    """예측 표를 실행(묶음 하나의 시드 하나)별로 나눠 낸다(키 오름차순).

    키는 묶음 키(`group_key`) 뒤에 seed를 붙인 튜플이다: 표지가 기본값인 실행은 `(run_id, family, arm, seed)`,
    아니면 `(run_id, family, arm, feature_source, variant, seed)`.
    """
    keys = _present(RUN_KEYS, preds)
    _require_columns(preds, keys, "prediction table")
    for key, rows in preds.groupby(keys, sort=True):
        yield (*_key_of(dict(zip(keys, key))), int(key[-1])), rows


def prediction_table(
    windows: WindowSet,
    probs: np.ndarray,
    steps: pd.DataFrame,
    videos: pd.DataFrame,
    *,
    run_id: str,
    family: str,
    arm: str,
    seed: int,
    temperature: float = 1.0,
    feature_source: str = FEATURE_SOURCE_NATIVE,
    variant: str = VARIANT_BASE,
    supplementary: bool | None = None,
) -> pd.DataFrame:
    """창 묶음과 그 확률로 예측 표를 만든다(실행 하나).

    - windows: 평가한 창(`make_windows(steps, 120, stride=1)`). probs는 `[N, 3]`이고 창과 같은 순서다.
    - steps: 창을 만든 스텝 표(`StepTable.steps`). perclos_60을 float32 창 텐서가 아니라 이 표의 float64 값에서 가져온다
      (PERCLOS 판정의 임계값 비교가 앱의 배정밀도 계산과 같아지게 한다). 창의 마지막 스텝이 표에 없거나 값이 창과
      다르면 ValueError다.
    - videos: 영상 표. gate_pass를 여기서 가져온다. 창의 영상이 표에 없으면 ValueError다.
    - fold는 창 메타데이터의 fold(피험자의 공식 fold = 그 피험자가 시험 대상인 바깥 fold)다.
    - feature_source, variant, supplementary: 표지 열(모듈 설명). 주 결과의 묶음은 주지 않는다(기본값).
      부가 분석의 예: 15 fps 모사 추출로 평가한 fold 모델은 `feature_source=FEATURE_SOURCE_FPS15`(창과 steps는 그
      추출의 것), 순서형 라벨 실행은 `variant=VARIANT_ORDINAL`, INT8 모델은 `family=FAMILY_TCN_INT8,
      supplementary=True`.
    가정: supplementary를 주지 않으면(None) feature_source나 variant가 기본값이 아닐 때 True, 둘 다 기본값이면
      False다. 표지를 달아 놓고 부가 표시를 잊은 묶음이 주 결과에 섞이지 않게 한다. 계열 이름만 다른 부가 묶음
      (INT8)은 True를 직접 줘야 한다. 표지가 기본값이 아닌데 False를 직접 주면 ValueError다(`validate_predictions`).
    """
    if supplementary is None:
        supplementary = feature_source != FEATURE_SOURCE_NATIVE or variant != VARIANT_BASE
    meta = windows.meta
    n = len(windows)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (n, N_CLASSES):
        raise ValueError(f"probs must have shape ({n}, {N_CLASSES}), got {probs.shape}")
    _require_columns(steps, ("session_id", "k", PERCLOS_COLUMN), "steps table")
    _require_columns(videos, VIDEO_COLUMNS, "video table")

    step_index = pd.MultiIndex.from_arrays([steps["session_id"].to_numpy(), steps["k"].to_numpy()])
    if not step_index.is_unique:
        raise ValueError("steps table holds the same (session_id, k) more than once")
    positions = step_index.get_indexer(
        pd.MultiIndex.from_arrays([meta["session_id"].to_numpy(), meta["k_end"].to_numpy()])
    )
    if (positions < 0).any():
        raise ValueError(f"steps table lacks the last step of {int((positions < 0).sum())} window(s)")
    perclos = steps[PERCLOS_COLUMN].to_numpy(dtype=np.float64)[positions]
    if not np.array_equal(perclos.astype(np.float32), windows.x[:, -1, F_PERCLOS_60]):
        raise ValueError("steps table is not the one these windows were made from (perclos_60 differs)")

    if videos["video_id"].duplicated().any():
        raise ValueError("video table lists a video_id more than once")
    video_positions = pd.Index(videos["video_id"].to_numpy()).get_indexer(meta["video_id"].to_numpy())
    if (video_positions < 0).any():
        unknown = sorted(set(meta["video_id"].to_numpy()[video_positions < 0].tolist()))
        raise ValueError(f"video table does not list the videos of these windows: {unknown[:10]}")

    table = pd.DataFrame(
        {
            "run_id": run_id,
            "family": family,
            "arm": arm,
            "fold": meta["fold"].to_numpy(),
            "seed": seed,
            "subject": meta["subject"].to_numpy(),
            "video_id": meta["video_id"].to_numpy(),
            "part": meta["part"].to_numpy(),
            "class_index": meta["class_index"].to_numpy(),
            "k": meta["k_end"].to_numpy(),
            "t_end_ms": meta["t_end_ms"].to_numpy(),
            PROB_COLUMNS[0]: probs[:, 0],
            PROB_COLUMNS[1]: probs[:, 1],
            PROB_COLUMNS[2]: probs[:, 2],
            PERCLOS_COLUMN: perclos,
            "gate_pass": videos["gate_pass"].to_numpy(dtype=bool)[video_positions],
            "temperature": float(temperature),
            "feature_source": str(feature_source),
            "variant": str(variant),
            "supplementary": bool(supplementary),
        },
        columns=list(PREDICTION_COLUMNS),
    )
    return validate_predictions(table)


def _joined(values: Iterable[Any]) -> str:
    """부분별 값을 `/`로 잇는다. 모두 같으면 그 값 하나만 적는다."""
    texts = [str(value) for value in values]
    return texts[0] if len(set(texts)) == 1 else "/".join(texts)


def video_table(quality: pd.DataFrame) -> pd.DataFrame:
    """품질 표(`StepTable.quality`, 세션마다 한 행)에서 영상 표를 만든다.

    열: `VIDEO_COLUMNS`와 `n_windows`(보폭 1의 창 수, 부분들의 합), `face_ratio`(얼굴 검출 비율, 시간 기준),
    그리고 품질 표에 있으면 기준선 열(`BASELINE_COLUMNS`, PLAN 4.4절의 리포트 항목):
    - `baseline_status`: 영상 자신의 보정 결과(`ok`, `off_axis`, `fallback_global`, 끝나지 않았으면 `none`).
      나뉜 영상은 부분의 값을 품질 표의 행 순서대로 `/`로 잇고(`ok/none`), 모두 같으면 하나만 적는다.
    - `t_b_ms`: 영상 자신의 기준선이 완성된 세션 시간 T_b. 나뉜 영상은 부분 중 가장 늦은 값, 어느 부분도 보정이
      끝나지 않았으면 −1이다. `NOMINAL_BASELINE_MS`(60초)보다 크면 기준선이 연장된 영상이다.
    - `baseline_used`: 이 품질 표의 정규화 조건에서 특징에 쓴 기준선(`session`, `alert`, `global`, `none`). 잇는 방식은
      `baseline_status`와 같다.
    - `alert_fallback`: alert 영상 기준선을 쓸 수 없어 전역 상수로 대체했는지(부분 중 하나라도 True이면 True).
    논리 영상마다 한 행이고 (subject, class_index) 순이다. 나뉜 영상의 부분들은 한 행으로 합친다.
    가정: 계획서는 "기준선이 연장된 영상을 리포트에 싣는다"고만 했다. 나뉜 영상의 값을 합치는 방식(가장 늦은 T_b,
      상태를 `/`로 이음)은 여기서 정한 것이다.
    """
    _require_columns(quality, VIDEO_COLUMNS + ("video_n_windows", "video_face_ratio"), "quality table")
    columns = list(VIDEO_COLUMNS) + ["video_n_windows", "video_face_ratio"]
    table = quality[columns].drop_duplicates().rename(
        columns={"video_n_windows": "n_windows", "video_face_ratio": "face_ratio"}
    )
    if table["video_id"].duplicated().any():
        raise ValueError("parts of one logical video disagree on subject, fold, class or quality gate values")
    table = table.reset_index(drop=True)
    extra = [name for name in BASELINE_COLUMNS if name in quality.columns]
    if extra:
        grouped = quality.groupby("video_id", sort=False)
        aggregated: dict[str, pd.Series] = {}
        for name in extra:
            if name == "t_b_ms":
                aggregated[name] = grouped[name].max().astype("int64")
            elif name == "alert_fallback":
                aggregated[name] = grouped[name].any().astype(bool)
            else:
                aggregated[name] = grouped[name].agg(_joined)
        for name in extra:
            table[name] = aggregated[name].reindex(table["video_id"].to_numpy()).to_numpy()
    return table.sort_values(["subject", "class_index"], kind="stable").reset_index(drop=True)


def exclusion_list(quality: pd.DataFrame) -> pd.DataFrame:
    """품질 표에서 제외 목록을 만든다: D7 품질 기준을 통과하지 못한 논리 영상마다 한 행.

    열: video_id, subject, fold, class_index, reason, detail.
    - reason `no_window`: 창이 하나도 없다. 주 결과에서는 오답으로 세고(D7) 보조 행(품질 기준 통과 영상)에서는 빠진다.
    - reason `quality_gate`: 창은 있지만 품질 기준(얼굴 검출 비율, 창 수)에 못 미친다. 주 결과에는 들어가고 보조
      행에서만 빠진다.
    detail에는 얼굴 검출 비율, 창 수, 부분별 추출 상태와 보정 상태를 적는다.
    """
    _require_columns(
        quality,
        VIDEO_COLUMNS + ("video_n_windows", "video_face_ratio", "raw_status", "baseline_status"),
        "quality table",
    )
    rows: list[dict[str, Any]] = []
    for video_id, parts in quality[~quality["gate_pass"].to_numpy(dtype=bool)].groupby("video_id", sort=False):
        first = parts.iloc[0]
        n_windows = int(first["video_n_windows"])
        rows.append(
            {
                "video_id": str(video_id),
                "subject": int(first["subject"]),
                "fold": int(first["fold"]),
                "class_index": int(first["class_index"]),
                "reason": "no_window" if n_windows == 0 else "quality_gate",
                "detail": (
                    f"face_ratio={float(first['video_face_ratio']):.3f}, windows={n_windows}, "
                    f"raw={'/'.join(str(v) for v in parts['raw_status'])}, "
                    f"baseline={'/'.join(str(v) for v in parts['baseline_status'])}"
                ),
            }
        )
    table = pd.DataFrame(rows, columns=["video_id", "subject", "fold", "class_index", "reason", "detail"])
    return table.sort_values(["subject", "class_index"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 분류 지표
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """혼동 행렬 하나의 요약. `confusion[i][j]`는 정답 i를 j로 예측한 양(창 수, 또는 영상 가중이면 영상 단위)."""

    confusion: tuple[tuple[float, ...], ...]
    accuracy: float
    macro_f1: float
    recall: tuple[float, ...]
    f1: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    """창 단위 지표 (PLAN 6장). video_weighted가 주 수치, pooled가 보조 수치다."""

    video_weighted: ClassMetrics
    pooled: ClassMetrics
    n_windows: int
    n_videos: int  # 창이 하나 이상 있는 영상 수


@dataclass(frozen=True, slots=True)
class VideoMetrics:
    """영상 단위 지표 (PLAN 1.2절의 결정 D7: 분모는 논리 영상 전체, 창이 없는 영상은 오답).

    `confusion[i][j]`: 정답이 classes[i]인 영상 중 classes[j]로 판정된 수. 마지막 열은 창이 없어 판정하지 못한 영상이다.
    accuracy와 recall의 분모는 표의 모든 영상이다(창이 없는 영상은 오답).
    """

    classes: tuple[int, ...]
    n_videos: int
    n_no_window: int
    accuracy: float
    recall: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1_numbers(confusion: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """정사각 혼동 행렬 → (macro-F1, 클래스별 재현율, 클래스별 F1, 정확도). 분모가 0인 값은 0이다."""
    diagonal = np.diag(confusion).astype(np.float64)
    truth = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(diagonal, truth, out=np.zeros_like(diagonal), where=truth > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    both = precision + recall
    f1 = np.divide(2.0 * precision * recall, both, out=np.zeros_like(diagonal), where=both > 0)
    return float(f1.mean()), recall, f1, _ratio(float(diagonal.sum()), float(confusion.sum()))


def _class_metrics(confusion: np.ndarray) -> ClassMetrics:
    macro_f1, recall, f1, accuracy = _f1_numbers(confusion)
    return ClassMetrics(
        confusion=tuple(tuple(float(value) for value in row) for row in confusion),
        accuracy=accuracy,
        macro_f1=macro_f1,
        recall=tuple(float(value) for value in recall),
        f1=tuple(float(value) for value in f1),
    )


def _per_video(
    first: pd.DataFrame,
    values: Mapping[str, np.ndarray],
    videos: pd.DataFrame | None,
    classes: Sequence[int],
    missing: Mapping[str, float],
) -> pd.DataFrame:
    """영상마다 한 행인 표를 만든다.

    first는 예측에 나온 영상의 첫 행(나온 순서), values는 같은 순서의 영상별 값이다. videos가 있으면 그 표에서 클래스가
    classes인 영상 전체가 행이 되고, 예측에 없는 영상은 missing의 값을 받는다(gate_pass도 videos의 값을 쓴다).
    """
    identity = ["subject", "video_id", "fold", "class_index", "gate_pass"]
    table = first[identity].reset_index(drop=True)
    for name, array in values.items():
        table[name] = array
    if videos is not None:
        _require_columns(videos, VIDEO_COLUMNS, "video table")
        listed = videos.loc[videos["class_index"].isin(list(classes)), identity].reset_index(drop=True)
        if listed.duplicated(list(VIDEO_KEYS)).any():
            raise ValueError("video table lists a logical video more than once")
        listed_index = pd.MultiIndex.from_arrays([listed[name].to_numpy() for name in VIDEO_KEYS])
        positions = listed_index.get_indexer(pd.MultiIndex.from_arrays([table[name].to_numpy() for name in VIDEO_KEYS]))
        if (positions < 0).any():
            unknown = table["video_id"].to_numpy()[positions < 0].tolist()
            raise ValueError(f"predictions hold videos that the video table does not list: {unknown[:10]}")
        for name in ("fold", "class_index"):
            if (listed[name].to_numpy()[positions] != table[name].to_numpy()).any():
                raise ValueError(f"{name} of a video differs between the prediction table and the video table")
        for name, array in values.items():
            full = np.full(len(listed), missing[name], dtype=array.dtype)
            full[positions] = array
            listed[name] = full
        table = listed
    return table.sort_values(["subject", "class_index", "video_id"], kind="stable").reset_index(drop=True)


def _vote(preds: pd.DataFrame, videos: pd.DataFrame | None, classes: tuple[int, ...]) -> pd.DataFrame:
    """classes 안에서만 고르는 영상 표결. `video_vote`(세 클래스)와 `two_class`(alert, drowsy)가 함께 쓴다."""
    run_keys = _present(RUN_KEYS, preds)
    _require_columns(
        preds,
        tuple(run_keys) + ("subject", "video_id", "fold", "class_index", "gate_pass") + PROB_COLUMNS,
        "prediction table",
    )
    _require_single(preds, run_keys, "prediction table")
    rows = preds[preds["class_index"].isin(list(classes))]
    keys = list(VIDEO_KEYS)
    codes = rows.groupby(keys, sort=False).ngroup().to_numpy()
    first = rows.drop_duplicates(keys)
    n_groups = len(first)
    if len(rows[keys + ["class_index"]].drop_duplicates()) != n_groups:
        raise ValueError("a video has more than one class_index in the prediction table")

    probs = rows[list(PROB_COLUMNS)].to_numpy(dtype=np.float64)
    candidates = np.array(classes, dtype=np.int64)
    choice = candidates[np.argmax(probs[:, candidates], axis=1)]  # 같으면 앞선 클래스
    votes = np.zeros((n_groups, N_CLASSES), dtype=np.int64)
    np.add.at(votes, (codes, choice), 1)
    sums = np.zeros((n_groups, N_CLASSES), dtype=np.float64)
    np.add.at(sums, codes, probs)
    n_windows = votes.sum(axis=1)
    mean_p = sums / n_windows[:, None]

    counted = votes[:, candidates]
    tied = counted == counted.max(axis=1, keepdims=True)
    # 표가 같은 클래스끼리는 평균 확률로 가른다. 그것도 같으면 argmax가 앞선 클래스를 고른다.
    pred = candidates[np.argmax(np.where(tied, mean_p[:, candidates], -np.inf), axis=1)]

    values: dict[str, np.ndarray] = {"n_windows": n_windows}
    missing: dict[str, float] = {"n_windows": 0}
    for index, name in enumerate(CLASS_ORDER):
        values[f"votes_{name}"] = votes[:, index]
        missing[f"votes_{name}"] = 0
    for index, name in enumerate(CLASS_ORDER):
        values[f"mean_p_{name}"] = mean_p[:, index]
        missing[f"mean_p_{name}"] = math.nan
    values["pred_class"] = pred
    missing["pred_class"] = NO_PREDICTION
    table = _per_video(first, values, videos, classes, missing)
    table["correct"] = table["pred_class"].to_numpy() == table["class_index"].to_numpy()
    return table[list(VOTE_COLUMNS)]


def video_vote(preds: pd.DataFrame, videos: pd.DataFrame | None = None) -> pd.DataFrame:
    """창 예측의 다수결로 논리 영상을 판정한다 (PLAN 6장, D7). 실행 하나의 예측 표를 받는다.

    - 영상의 판정은 그 영상의 창들이 고른 클래스 중 가장 많은 것이다. 나뉜 영상(`10_1`, `10_2`)의 부분들은 한 영상으로
      합쳐서 센다.
    - 표가 같으면 그 클래스들 중 영상 전체 창의 평균 확률이 큰 쪽이다. 가정: 그것도 정확히 같으면 클래스 순서가 앞선 쪽.
    - videos(영상 표)를 주면 그 표의 영상이 모두 행이 된다. 창이 하나도 없는 영상은 `n_windows = 0`,
      `pred_class = −1`, `correct = False`다(D7: 오답으로 센다). 예측 표에 있는데 영상 표에 없는 영상은 ValueError다.
      videos가 없으면 예측 표에 나온 영상만 행이 된다(분모가 줄어든다는 뜻이므로 리포트에서는 항상 videos를 준다).

    돌려주는 표(`VOTE_COLUMNS`): subject, video_id, fold, class_index, gate_pass, n_windows, votes_{클래스} 3개,
    mean_p_{클래스} 3개(창이 없으면 NaN), pred_class, correct. (subject, class_index) 순이다.
    지표는 `video_metrics`(정확도, 재현율, 혼동 행렬)와 `vote_window_metrics`(창 단위)가 이 표에서 계산한다.
    """
    return _vote(preds, videos, tuple(range(N_CLASSES)))


def two_class(preds: pd.DataFrame, videos: pd.DataFrame | None = None) -> pd.DataFrame:
    """D8의 2클래스(alert 대 drowsy) 영상 표결. 3클래스 모델의 예측 표를 그대로 받는다.

    alert와 drowsy 영상만 대상으로, 창마다 `p_alert`와 `p_drowsy` 중 큰 쪽을 고른 뒤(p_low_vigilant는 보지 않는다)
    영상별 다수결을 한다. RLDD에서는 논리 영상 120개가 분모다.
    가정: 한 창에서 두 확률이 정확히 같으면 alert다. 영상의 표가 같을 때는 `video_vote`처럼 두 확률의 평균으로 가른다.
    가정: D7의 분모 규칙을 여기에도 쓴다. videos를 주면 창이 없는 alert·drowsy 영상도 분모에 들어가고 오답이다.
    돌려주는 표는 `video_vote`와 같은 열이고 low_vigilant 영상의 행이 없다(`votes_low_vigilant`는 항상 0).
    정확도는 `video_metrics(votes, classes=TWO_CLASSES)`로 구한다.
    """
    return _vote(preds, videos, TWO_CLASSES)


def _window_confusions(
    class_index: np.ndarray, votes: np.ndarray, n_windows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """영상별 표 수에서 창 단위 혼동 행렬 둘을 만든다: (창을 그대로 합친 것, 영상마다 합이 1이 되게 나눈 것)."""
    pooled = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    np.add.at(pooled, class_index, votes)
    has_windows = n_windows > 0
    weighted = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    np.add.at(weighted, class_index[has_windows], votes[has_windows] / n_windows[has_windows, None])
    return pooled, weighted


def _vote_arrays(votes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.stack([votes[f"votes_{name}"].to_numpy(dtype=np.float64) for name in CLASS_ORDER], axis=1)
    return votes["class_index"].to_numpy(dtype=np.int64), counts, votes["n_windows"].to_numpy(dtype=np.float64)


def vote_window_metrics(votes: pd.DataFrame) -> WindowMetrics:
    """`video_vote`의 표에서 창 단위 지표를 계산한다(`window_metrics`와 같은 값).

    - pooled(보조): 모든 창을 그대로 합친 혼동 행렬.
    - video_weighted(주): 영상마다 같은 비중. 영상 하나가 자기 창들의 예측 분포(합 1)를 정답 클래스의 행에 더한다.
      창이 없는 영상은 들어가지 않는다(창 단위 지표에는 분모가 없다. 그런 영상은 영상 단위 지표에서 오답으로 센다).
    macro-F1은 각 혼동 행렬에서 세 클래스의 F1을 평균한 값이다(어느 쪽에도 나오지 않은 클래스의 F1은 0).
    """
    _require_columns(votes, ("class_index", "n_windows") + tuple(f"votes_{name}" for name in CLASS_ORDER), "vote table")
    class_index, counts, n_windows = _vote_arrays(votes)
    pooled, weighted = _window_confusions(class_index, counts, n_windows)
    return WindowMetrics(
        video_weighted=_class_metrics(weighted),
        pooled=_class_metrics(pooled),
        n_windows=int(n_windows.sum()),
        n_videos=int((n_windows > 0).sum()),
    )


def window_metrics(preds: pd.DataFrame) -> WindowMetrics:
    """창 단위 macro-F1, 클래스별 재현율, 혼동 행렬 (PLAN 6장). 실행 하나의 예측 표를 받는다.

    영상마다 같은 비중을 주는 값(`video_weighted`)이 주 수치이고 창을 그대로 합친 값(`pooled`)이 보조 수치다.
    정의는 `vote_window_metrics`에 있다.
    """
    return vote_window_metrics(video_vote(preds))


def _video_confusion(class_index: np.ndarray, pred_class: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """영상 단위 혼동 행렬 `[len(classes), len(classes) + 1]`. 마지막 열은 판정 없음(창 없음)."""
    column = {value: index for index, value in enumerate(classes)}
    column[NO_PREDICTION] = len(classes)
    confusion = np.zeros((len(classes), len(classes) + 1), dtype=np.int64)
    for truth, pred in zip(class_index.tolist(), pred_class.tolist()):
        if truth not in column or truth == NO_PREDICTION or pred not in column:
            raise ValueError(f"vote table holds class {truth} / prediction {pred} outside classes {tuple(classes)}")
        confusion[column[truth], column[pred]] += 1
    return confusion


def video_metrics(votes: pd.DataFrame, classes: Sequence[int] = tuple(range(N_CLASSES))) -> VideoMetrics:
    """표결 표(`video_vote` 또는 `two_class`)에서 영상 단위 정확도, 클래스별 재현율, 혼동 행렬을 계산한다 (D7).

    분모는 표의 모든 영상이다. 창이 없는 영상은 혼동 행렬의 마지막 열에 들어가고 오답이다.
    보조 행(품질 기준 통과 영상만)은 `video_metrics(votes[votes["gate_pass"]])`로 구한다.
    `two_class`의 표에는 `classes=TWO_CLASSES`를 준다. 표에 classes 밖의 클래스가 있으면 ValueError다.
    """
    _require_columns(votes, ("class_index", "pred_class"), "vote table")
    confusion = _video_confusion(
        votes["class_index"].to_numpy(dtype=np.int64), votes["pred_class"].to_numpy(dtype=np.int64), classes
    )
    totals = confusion.sum(axis=1)
    hits = np.diag(confusion[:, : len(classes)])
    return VideoMetrics(
        classes=tuple(int(value) for value in classes),
        n_videos=int(confusion.sum()),
        n_no_window=int(confusion[:, -1].sum()),
        accuracy=_ratio(float(hits.sum()), float(confusion.sum())),
        recall=tuple(_ratio(float(hit), float(total)) for hit, total in zip(hits, totals)),
        confusion=tuple(tuple(int(value) for value in row) for row in confusion),
    )


# ---------------------------------------------------------------------------
# 오경보 (PLAN 6장, 1.2절의 결정 D9: 판정 시작 횟수가 주 수치이고 창 단위 값을 함께 싣는다)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class AlarmReport:
    """졸음 판정기를 시험 영상에 돌린 결과 (`alarm_report`). 여러 fold의 표를 넘기면 fold를 합친 값이다.

    alert 영상(오경보)
    - `alert_onsets`: 판정이 시작된 횟수. `alert_hours`: 평가 시간(평가한 스텝 수 × 스텝 길이).
    - `onsets_per_hour` = alert_onsets / alert_hours (주 수치).
    - `alert_videos_with_onset` / `n_alert_videos`, 그 비율 `alert_video_onset_ratio`.
    - `alarm_time_share`: 평가한 스텝 중 판정 상태였던 스텝의 비율.
      가정: 스텝을 처리한 뒤 판정 중이면 그 스텝을 판정 상태로 센다(시작한 스텝은 들어가고 종료한 스텝은 빠진다).
    - `alert_flagged_windows`, `flagged_windows_per_hour`: D9의 창 단위 값. model은 drowsy로 분류된 창(세 확률 중
      p_drowsy가 가장 큼), perclos는 perclos_60 ≥ onset_ratio인 창, either는 둘 중 하나라도 해당하는 창의 수를 평가
      시간으로 나눈 값이다. 가정: D9는 모델의 값만 정의한다. perclos와 either의 정의는 나란히 싣기 위해 맞춘 것이다.
    drowsy 영상
    - `drowsy_videos_with_onset` / `n_drowsy_videos`, 그 비율 `drowsy_video_onset_ratio`.
    `videos`는 영상별 표(`ALARM_COLUMNS`), `sweep`은 임계값을 바꿔 본 표(`SWEEP_COLUMNS`, 요청하지 않았으면 빈 표)다.
    `onset_p`는 쓴 모델 시작 임계값, `rules_hash`는 규칙 설정의 해시(PLAN 4.7절)다.
    """

    source: str
    onset_p: float
    rules_hash: str
    n_alert_videos: int
    alert_onsets: int
    alert_hours: float
    onsets_per_hour: float
    alert_videos_with_onset: int
    alert_video_onset_ratio: float
    alarm_time_share: float
    alert_flagged_windows: int
    flagged_windows_per_hour: float
    n_drowsy_videos: int
    drowsy_videos_with_onset: int
    drowsy_video_onset_ratio: float
    videos: pd.DataFrame
    sweep: pd.DataFrame


def alarm_videos(
    preds: pd.DataFrame,
    cfg: RulesConfig,
    source: str,
    videos: pd.DataFrame | None = None,
    onset_p: float | None = None,
) -> pd.DataFrame:
    """영상마다 졸음 판정기를 돌려 시작 횟수와 판정 상태 스텝 수를 센다. 실행 하나의 예측 표를 받는다.

    판정기는 규칙 엔진이 쓰는 것과 같은 클래스(`drowsy.rules.StepHysteresis`)다. 세션(영상의 부분)마다 새로 만들고
    그 세션의 예측을 k 순서로 넣는다. 시작·종료 조건의 비교와 either의 합성은 `RuleEngine.on_step`과 같다:
    - model: `p_drowsy ≥ onset_p`가 onset_steps 연속이면 시작, `≤ end_p`가 end_steps 연속이면 종료, 종료 뒤
      refractory_ms 동안은 다시 시작하지 않는다.
    - perclos: `perclos_60 ≥ onset_ratio`가 onset_steps 연속이면 시작, `< end_ratio`가 end_steps 연속이면 종료.
    - either: 두 판정기 중 하나라도 판정 중이면 졸음 상태다. 상태가 꺼져 있다가 켜질 때가 시작 한 번이다.
    한번 시작한 판정이 영상 끝까지 이어지면 시작은 한 번이다. 나뉜 영상의 부분 사이에서는 판정기를 새로 만든다
    (앱도 세션마다 새로 시작한다). 한 세션 안에서 스텝 번호가 이어지지 않는 행을 만나면 그 자리에서 판정을 끝내고
    두 판정기를 처음 상태로 되돌린다(아래 "세션 안에서 k가 건너뛸 때").

    - onset_p를 주면 모델 시작 임계값만 그 값으로 바꾼다(임계값 곡선용). end_p보다 커야 한다.
    - videos(영상 표)를 주면 예측이 없는 영상도 0으로 채운 행을 갖는다(판정이 난 영상의 비율의 분모가 된다).
    가정: 예측 표의 행(창이 나온 스텝)만 평가한다. 앱에서 PERCLOS 판정은 조회 구간이 찬 첫 스텝부터 돌아 모델보다
      119스텝 먼저 시작하지만, 여기서는 세 판정원을 같은 평가 시간에서 비교한다.
    가정: 평가 시간은 행 수 × `clock.step_ms`다(보폭 1의 예측을 전제한다). 한 세션에 행이 둘 이상인데 한 스텝 간격인
      행 쌍이 표 전체에 하나도 없으면 보폭이 1이 아닌 표로 보고 ValueError를 낸다(판정이 영원히 나지 않는 표다).

    세션 안에서 k가 건너뛸 때 (2026-11-13 소유자 결정 4 "10초 넘는 프레임 끊김 뒤에는 규칙 엔진이 세션 시작처럼 다시
    시작한다"에 맞춘 것. 앱의 `RuleEngine`과 같게 한다)
    - 언제 건너뛰나: 한 세션의 이어진 두 행에서 k가 직전 행의 k + 1이 아닌 경우는 하나뿐이다. 그 사이에 프레임
      간격이 `clock.reset_gap_ms`(10초)를 넘는 끊김이 있었다. `StepAggregator`는 그 끊김 뒤 첫 프레임이 속한 스텝부터
      다시 누적하고(끊긴 동안의 스텝은 내지 않는다) 조회 구간이 다시 차야 창이 나오므로, 스텝 표의 k는 11 이상,
      예측 표의 k는 189 이상(11 + 조회 구간이 덜 찬 59스텝 + 창이 차기까지 119스텝) 건너뛴다. 조회 구간이 덜 찬
      스텝은 세션의 처음과 그런 끊김 뒤에만 있어서 그 밖의 자리에서 행을 끊지 못한다. 15 fps 모사 추출과 프레임 누락
      변형도 같은 재생이므로 원인이 같다(프레임이 빠져 10초 이하이던 간격이 10초를 넘으면 그 표에만 건너뜀이 생긴다).
      무작위 끊김 세션과 그 프레임 누락 변형을 `DriverPipeline`에 재생해 확인했다(반례 없음).
    - 앱이 하는 일: 누적기가 이미 프레임을 받은 뒤라면 그 끊김은 `RuleEngine`이 다시 시작하는 끊김과 같은 프레임이다.
      앱은 그 프레임에서 졸음 상태를 끝내고 두 판정기(연속 수, 판정 상태, 재시작 금지 기억)를 처음 상태로 되돌린다.
    - 여기서 하는 일: k가 건너뛴 행을 넣기 전에 판정 상태를 끄고 두 판정기에 `StepHysteresis.reset()`을 부른다. 그래서
      끊기는 순간 판정 중이던 영상은 끊김 뒤의 행에서 시작 조건을 onset_steps개 다시 채워야 판정 상태가 되고, 그때
      시작을 하나 더 센다. 끊김 전에 끝난 판정의 재시작 금지 시간도 끊김 뒤로 이어지지 않는다.
    가정: 건너뛴 크기는 보지 않는다. 파이프라인이 낼 수 없는 작은 건너뜀(손으로 만든 표, 행을 솎아 낸 표)도 같은
      다시 시작으로 다룬다.

    앱과의 관계 (2026-11-13 소유자 결정 뒤)
    - 확률을 받는 스텝이 같다. 여기서는 예측 표의 모든 행(창이 나온 모든 스텝)에 확률을 준다(`StepHysteresis.update`의
      has_value가 항상 True). 앱의 `DriverPipeline`도 창이 찬 모든 스텝에서 모델을 부른다: 프레임이 끊겼다가 와서
      프레임 하나가 스텝을 여러 개 발행해도 스텝마다 부르고 그 확률을 `RuleEngine.on_step`에 준다(결정 전에는 마지막
      스텝에서만 불러서, 앞쪽 스텝에서 모델 판정의 연속 수가 0으로 돌아갔다).
    - 다시 시작하는 자리가 같다(위 "세션 안에서 k가 건너뛸 때").
    - 그래서 model 판정원은 끊김의 길이와 무관하게 시작 횟수와 판정 상태 스텝 수가 앱과 같다. 회귀 테스트는
      tests/test_evaluate.py의 `test_alarm_matches_the_app_when_frames_stall_across_step_boundaries`(10초 이하의 끊김)와
      `test_alarm_matches_the_app_across_a_gap_longer_than_the_reset_gap`(10초 넘는 끊김)이고, 무작위 끊김 세션에서
      `DriverPipeline`과 견준 결과도 차이가 없었다.
    - perclos와 either에 남는 차이는 위의 첫 가정(행만 평가한다) 하나다. 앱의 PERCLOS 판정기는 세션의 처음과 10초 넘는
      끊김 뒤마다 첫 행보다 119스텝 먼저 값을 받기 시작한다. 그 119스텝 안에서 perclos_60이 시작 조건을 채우고 있으면
      앱은 더 일찍(첫 행 전일 수도 있다) 시작하므로, 그런 영상에서는 판정 상태 스텝 수가 앱보다 적고 첫 행 전에
      시작해 첫 행 전에 끝난 판정은 여기서 세지 않는다. 그 119스텝에서 perclos_60이 시작 조건(≥ onset_ratio)을 한 번도
      채우지 않은 세션에서는 두 판정원도 앱과 같다. 예측 표에는 그 스텝들의 perclos_60이 없어 이 함수만으로는 맞출 수 없다.
    가정: 앱은 창에 유한하지 않은 값이 있으면 추론을 건너뛰고 그 스텝을 값이 없는 스텝으로 본다. 예측 표의 행에는
      언제나 확률이 있으므로 여기에는 그런 스텝이 없다(스텝 특징은 입력이 유한하면 유한하다. `drowsy.step_features`).

    돌려주는 표(`ALARM_COLUMNS`): subject, video_id, fold, class_index, gate_pass, n_steps, evaluated_ms, onsets,
    alarm_steps(판정 상태였던 스텝 수), flagged_windows(`AlarmReport`의 D9 설명). (subject, class_index) 순이다.
    """
    if source not in DROWSY_SOURCES:
        raise ValueError(f"source must be one of {DROWSY_SOURCES}, got {source!r}")
    run_keys = _present(RUN_KEYS, preds)
    _require_columns(
        preds,
        tuple(run_keys)
        + ("subject", "video_id", "part", "fold", "class_index", "gate_pass", "k", "t_end_ms", PERCLOS_COLUMN)
        + PROB_COLUMNS,
        "prediction table",
    )
    _require_single(preds, run_keys, "prediction table")
    model_cfg = cfg.drowsiness.model
    perclos_cfg = cfg.drowsiness.perclos
    threshold = model_cfg.onset_p if onset_p is None else float(onset_p)
    if not model_cfg.end_p < threshold <= 1.0:
        raise ValueError(f"onset_p must be within ({model_cfg.end_p}, 1], got {threshold}")
    use_model = source != DROWSY_SOURCE_PERCLOS
    use_perclos = source != DROWSY_SOURCE_MODEL

    keys = list(VIDEO_KEYS)
    ordered = preds.sort_values(keys + ["part", "k"], kind="stable")
    video_codes = ordered.groupby(keys, sort=False).ngroup().to_numpy()
    session_codes = ordered.groupby(keys + ["part"], sort=False).ngroup().to_numpy()
    first = ordered.drop_duplicates(keys)
    n_videos = len(first)
    k = ordered["k"].to_numpy(dtype=np.int64)
    same_session = session_codes[1:] == session_codes[:-1]
    apart = np.diff(k)
    if (same_session & (apart <= 0)).any():
        raise ValueError("a session holds the same step k more than once")
    if same_session.any() and not (same_session & (apart == 1)).any():
        raise ValueError(
            "alarm evaluation needs consecutive per-second outputs (evaluation stride 1); "
            "no two rows of a session are one step apart"
        )

    p_drowsy = ordered[PROB_COLUMNS[CLASS_DROWSY]].to_numpy(dtype=np.float64)
    perclos = ordered[PERCLOS_COLUMN].to_numpy(dtype=np.float64)
    flagged = np.zeros(len(ordered), dtype=bool)
    if use_model:
        flagged |= np.argmax(ordered[list(PROB_COLUMNS)].to_numpy(dtype=np.float64), axis=1) == CLASS_DROWSY
    if use_perclos:
        flagged |= perclos >= perclos_cfg.onset_ratio
    flagged_windows = np.zeros(n_videos, dtype=np.int64)
    np.add.at(flagged_windows, video_codes, flagged)

    onsets = [0] * n_videos
    alarm_steps = [0] * n_videos
    k_list = k.tolist()
    t_list = ordered["t_end_ms"].tolist()
    p_list = p_drowsy.tolist()
    perclos_list = perclos.tolist()
    video_list = video_codes.tolist()
    session_list = session_codes.tolist()

    def new_rules() -> tuple[StepHysteresis, StepHysteresis]:
        return (
            StepHysteresis(model_cfg.onset_steps, model_cfg.end_steps, model_cfg.refractory_ms),
            StepHysteresis(perclos_cfg.onset_steps, perclos_cfg.end_steps, perclos_cfg.refractory_ms),
        )

    model_rule, perclos_rule = new_rules()
    active = False
    session = -1
    previous_k = 0
    for index, step_k in enumerate(k_list):
        if session_list[index] != session:  # 세션이 바뀌면 판정기를 새로 만든다
            session = session_list[index]
            model_rule, perclos_rule = new_rules()
            active = False
        elif step_k != previous_k + 1:
            # 세션 안에서 k가 건너뛰었다 = 그 사이에 10초 넘는 끊김이 있었다. 앱의 규칙 엔진은 끊김 뒤 첫 프레임에서
            # 졸음 상태를 끝내고 두 판정기를 처음 상태로 되돌린다. 여기서도 이 행을 넣기 전에 똑같이 한다.
            model_rule.reset()
            perclos_rule.reset()
            active = False
        previous_k = step_k
        t_end_ms = t_list[index]
        if use_model:
            p = p_list[index]
            model_rule.update(step_k, t_end_ms, True, p >= threshold, p <= model_cfg.end_p)
        if use_perclos:
            ratio = perclos_list[index]
            perclos_rule.update(
                step_k, t_end_ms, True, ratio >= perclos_cfg.onset_ratio, ratio < perclos_cfg.end_ratio
            )
        now = (use_model and model_rule.active) or (use_perclos and perclos_rule.active)
        if now:
            video = video_list[index]
            alarm_steps[video] += 1
            if not active:
                onsets[video] += 1
        active = now

    n_steps = np.bincount(video_codes, minlength=n_videos).astype(np.int64)
    values = {
        "n_steps": n_steps,
        "evaluated_ms": n_steps * int(cfg.clock.step_ms),
        "onsets": np.array(onsets, dtype=np.int64),
        "alarm_steps": np.array(alarm_steps, dtype=np.int64),
        "flagged_windows": flagged_windows,
    }
    table = _per_video(first, values, videos, tuple(range(N_CLASSES)), dict.fromkeys(values, 0))
    return table[list(ALARM_COLUMNS)]


def _alarm_numbers(
    class_index: np.ndarray,
    n_steps: np.ndarray,
    evaluated_ms: np.ndarray,
    onsets: np.ndarray,
    alarm_steps: np.ndarray,
    flagged: np.ndarray,
) -> dict[str, float]:
    """영상별 값에서 오경보 수치를 만든다. `alarm_summary`와 리포트의 부트스트랩이 함께 쓴다."""
    alert = class_index == CLASS_ALERT
    drowsy = class_index == CLASS_DROWSY
    hours = float(evaluated_ms[alert].sum()) / MS_PER_HOUR
    alert_onsets = float(onsets[alert].sum())
    with_onset = float((onsets[alert] > 0).sum())
    return {
        "onsets": alert_onsets,
        "alert_hours": hours,
        "onsets_per_hour": _ratio(alert_onsets, hours),
        "alert_videos_with_onset": with_onset,
        "alert_video_onset_ratio": _ratio(with_onset, float(alert.sum())),
        "alarm_time_share": _ratio(float(alarm_steps[alert].sum()), float(n_steps[alert].sum())),
        "flagged_windows_per_hour": _ratio(float(flagged[alert].sum()), hours),
        "drowsy_video_onset_ratio": _ratio(float((onsets[drowsy] > 0).sum()), float(drowsy.sum())),
    }


def alarm_summary(table: pd.DataFrame) -> dict[str, float]:
    """`alarm_videos`의 표(또는 그 일부, 예: fold 하나)에서 오경보 수치를 계산한다.

    키: onsets, alert_hours, onsets_per_hour, alert_videos_with_onset, alert_video_onset_ratio, alarm_time_share,
    flagged_windows_per_hour, drowsy_video_onset_ratio. 뜻은 `AlarmReport`에 있다.
    """
    _require_columns(table, ALARM_COLUMNS, "alarm table")
    return _alarm_numbers(
        table["class_index"].to_numpy(dtype=np.int64),
        table["n_steps"].to_numpy(dtype=np.float64),
        table["evaluated_ms"].to_numpy(dtype=np.float64),
        table["onsets"].to_numpy(dtype=np.float64),
        table["alarm_steps"].to_numpy(dtype=np.float64),
        table["flagged_windows"].to_numpy(dtype=np.float64),
    )


def alarm_sweep(
    preds: pd.DataFrame,
    cfg: RulesConfig,
    thresholds: Sequence[float],
    source: str = DROWSY_SOURCE_MODEL,
    videos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """모델 시작 임계값(`drowsiness.model.onset_p`)을 바꿔 가며 오경보 수치를 다시 계산한다 (PLAN 6장의 "곡선").

    임계값마다 한 행이고 열은 `SWEEP_COLUMNS`(onset_p와 `alarm_summary`의 키)다.
    보고용이며 임계값을 고르는 데 쓰지 않는다(주 수치의 임계값은 사전에 고정돼 있다).
    가정: 계획서는 무엇을 바꾸는 곡선인지 적지 않았다. 시작 임계값만 바꾸고 종료 임계값과 스텝 수, 재시작 금지
      시간은 설정 그대로 둔다.
    source는 model 또는 either다. perclos는 이 임계값과 무관하므로 임계값을 하나라도 주면 ValueError다.
    thresholds가 비어 있으면 빈 표다.
    """
    if len(thresholds) > 0 and source == DROWSY_SOURCE_PERCLOS:
        raise ValueError("the threshold sweep varies drowsiness.model.onset_p and does not apply to source 'perclos'")
    rows = [
        {"onset_p": float(threshold), **alarm_summary(alarm_videos(preds, cfg, source, videos, onset_p=threshold))}
        for threshold in thresholds
    ]
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def alarm_report(
    preds: pd.DataFrame,
    cfg: RulesConfig,
    source: str,
    videos: pd.DataFrame | None = None,
    sweep: Sequence[float] = (),
) -> AlarmReport:
    """졸음 판정기(model, perclos, either)를 시험 영상에 돌린 오경보 리포트 (PLAN 6장, D9).

    실행 하나의 예측 표를 받는다. 판정 방식과 가정은 `alarm_videos`, 수치의 뜻은 `AlarmReport`에 있다.
    여러 fold의 예측을 한 번에 넘기면 fold를 합친 횟수와 시간이 된다(주 수치). fold별 값은
    `alarm_summary(report.videos[report.videos["fold"] == k])`로 구한다.
    sweep에 임계값들을 주면 `alarm_sweep`의 표를 함께 싣는다(source가 perclos이면 ValueError).
    """
    table = alarm_videos(preds, cfg, source, videos)
    numbers = alarm_summary(table)
    class_index = table["class_index"].to_numpy()
    drowsy_onsets = table["onsets"].to_numpy()[class_index == CLASS_DROWSY]
    alert_flagged = table["flagged_windows"].to_numpy()[class_index == CLASS_ALERT]
    return AlarmReport(
        source=source,
        onset_p=float(cfg.drowsiness.model.onset_p),
        rules_hash=rules_hash(cfg),
        n_alert_videos=int((class_index == CLASS_ALERT).sum()),
        alert_onsets=int(numbers["onsets"]),
        alert_hours=numbers["alert_hours"],
        onsets_per_hour=numbers["onsets_per_hour"],
        alert_videos_with_onset=int(numbers["alert_videos_with_onset"]),
        alert_video_onset_ratio=numbers["alert_video_onset_ratio"],
        alarm_time_share=numbers["alarm_time_share"],
        alert_flagged_windows=int(alert_flagged.sum()),
        flagged_windows_per_hour=numbers["flagged_windows_per_hour"],
        n_drowsy_videos=int((class_index == CLASS_DROWSY).sum()),
        drowsy_videos_with_onset=int((drowsy_onsets > 0).sum()),
        drowsy_video_onset_ratio=numbers["drowsy_video_onset_ratio"],
        videos=table,
        sweep=alarm_sweep(preds, cfg, sweep, source, videos),
    )


# ---------------------------------------------------------------------------
# 피험자 단위 부트스트랩
# ---------------------------------------------------------------------------


def _replicates(
    tables: Sequence[pd.DataFrame],
    statistic: Callable[[Sequence[pd.DataFrame]], Sequence[float]],
    n: int,
    seed: int,
) -> np.ndarray:
    """피험자를 복원 추출한 표들에 statistic을 적용한 값 `[n, 통계량 수]`.

    표들은 같은 피험자 집합을 가져야 하고, 한 번의 추출을 모든 표에 똑같이 적용한다(짝지은 비교).
    추출한 표에서는 `subject` 열을 추출 순번(0부터)으로 바꾼다.
    """
    if n < 1:
        raise ValueError("the number of bootstrap replicates must be >= 1")
    prepared: list[tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]] = []
    subjects: np.ndarray | None = None
    for table in tables:
        _require_columns(table, ("subject",), "bootstrap table")
        values = table["subject"].to_numpy()
        order = np.argsort(values, kind="stable")
        unique, starts, counts = np.unique(values[order], return_index=True, return_counts=True)
        if subjects is None:
            subjects = unique
        elif not np.array_equal(subjects, unique):
            raise ValueError("paired bootstrap tables must hold the same subjects")
        prepared.append((table, order, starts, counts))
    if subjects is None or len(subjects) == 0:
        raise ValueError("bootstrap table holds no subjects")
    n_subjects = len(subjects)
    relabel = np.arange(n_subjects)
    draws = np.random.default_rng(seed).integers(0, n_subjects, size=(n, n_subjects))
    out: list[np.ndarray] = []
    for draw in draws:
        resampled: list[pd.DataFrame] = []
        for table, order, starts, counts in prepared:
            positions = np.concatenate([order[starts[i] : starts[i] + counts[i]] for i in draw])
            sample = table.iloc[positions].reset_index(drop=True)
            sample["subject"] = np.repeat(relabel, counts[draw])
            resampled.append(sample)
        values = np.asarray(statistic(resampled), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("the bootstrap statistic returned a non-finite value on a resampled table")
        out.append(values)
    return np.stack(out)


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be within (0, 1)")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def subject_bootstrap(
    table: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    n: int = 10000,
    seed: int = 0,
    level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """피험자 단위(군집) 부트스트랩 구간 (PLAN 6장). `(아래, 위)`를 돌려준다. 기본은 95% 구간이다.

    피험자를 복원 추출해(피험자 수만큼) 뽑힌 피험자의 행을 모두 모은 표에 metric을 적용하기를 n번 되풀이하고, 그 값들의
    백분위수를 구간으로 삼는다. 같은 seed면 결과가 같다(`numpy.random.default_rng(seed)`).
    가정: 계획서는 구간의 방식을 적지 않았다. 백분위수 구간(선형 보간)을 쓰고, 피험자는 fold와 무관하게 추출한다.

    - table은 `subject` 열이 있는 표다. 예측 표도 되고, 영상 단위로 줄인 표(`video_vote`, `two_class`,
      `alarm_videos`의 결과)도 된다. 창 단위 표로는 10,000번이 느리므로 영상 단위 표를 권한다. 창이 없는 영상을
      오답으로 세려면(D7) 영상 표를 넣어 만든 표결 표를 써야 한다.
    - 추출한 표에서는 `subject` 열이 추출 순번으로 바뀐다. 같은 피험자가 두 번 뽑히면 서로 다른 피험자로 센다는
      뜻이다. 이 모듈의 함수는 영상을 (subject, video_id)로 구분하므로 추출한 표에 그대로 쓸 수 있다.
    - metric이 유한하지 않은 값을 내면 ValueError다.
    """
    values = _replicates([table], lambda tables: [metric(tables[0])], n, seed)
    return _interval(values[:, 0], level)


def paired_subject_bootstrap(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    n: int = 10000,
    seed: int = 0,
    level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """두 표의 지표 차 `metric(a) − metric(b)`의 짝지은 피험자 부트스트랩 구간 (PLAN 6장의 "통계적 주의").

    두 표(예: TCN과 GRU의 예측 표, 또는 그것을 영상 단위로 줄인 표)는 같은 피험자 집합을 가져야 한다(아니면
    ValueError). 매번 같은 피험자 추출을 두 표에 똑같이 적용한다. 구간이 0을 벗어날 때만 차이가 있다고 쓴다.
    나머지는 `subject_bootstrap`과 같다.
    """
    values = _replicates([table_a, table_b], lambda tables: [metric(tables[0]) - metric(tables[1])], n, seed)
    return _interval(values[:, 0], level)


# ---------------------------------------------------------------------------
# 확률 보정
# ---------------------------------------------------------------------------


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """기대 보정 오차(ECE). probs는 `[N, 클래스 수]`, labels는 `[N]`의 정답 클래스다.

    창마다 신뢰도(가장 큰 확률)와 맞았는지를 구하고, 신뢰도를 폭이 같은 n_bins개 구간 `[i/n, (i+1)/n)`으로 나눠
    (1.0은 마지막 구간) `Σ (구간의 창 수 / N) · |구간 정확도 − 구간 평균 신뢰도|`를 계산한다. 창이 없으면 0.0이다.
    가정: 계획서에 구간 수가 없다. 기본은 같은 폭 15개(`ECE_BINS`)다.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or labels.shape != (probs.shape[0],):
        raise ValueError(f"probs must be [N, classes] and labels [N], got {probs.shape} and {labels.shape}")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if len(probs) == 0:
        return 0.0
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    bins = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    gap = np.zeros(n_bins, dtype=np.float64)
    np.add.at(gap, bins, correct - confidence)
    return float(np.abs(gap).sum() / len(probs))


def untemper(probs: np.ndarray, temperature: np.ndarray | float) -> np.ndarray:
    """온도를 적용한 확률에서 온도 적용 전의 확률을 되찾는다.

    `p = softmax(z / T)`이면 `p^T`를 정규화한 값이 `softmax(z)`다. temperature는 스칼라이거나 행마다의 값 `[N]`이다.
    확률이 0인 클래스는 0으로 남는다.
    """
    probs = np.asarray(probs, dtype=np.float64)
    scale = np.asarray(temperature, dtype=np.float64)
    if scale.ndim == 1:
        scale = scale[:, None]
    log_p = np.log(np.maximum(probs, np.finfo(np.float64).tiny)) * scale
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    restored = np.where(probs > 0.0, np.exp(log_p), 0.0)
    return restored / restored.sum(axis=1, keepdims=True)


def calibration_error(preds: pd.DataFrame, n_bins: int = ECE_BINS) -> tuple[float, float]:
    """예측 표의 `(온도 적용 전 ECE, 적용 후 ECE)` (PLAN 5.2절). 창을 그대로 합쳐 계산한다.

    적용 후는 표의 확률 그대로, 적용 전은 `temperature` 열로 되돌린 확률(`untemper`)의 ECE다.
    temperature가 1.0이면 두 값이 같다.
    """
    _require_columns(preds, PROB_COLUMNS + ("class_index", "temperature"), "prediction table")
    probs = preds[list(PROB_COLUMNS)].to_numpy(dtype=np.float64)
    labels = preds["class_index"].to_numpy(dtype=np.int64)
    after = expected_calibration_error(probs, labels, n_bins)
    if len(probs) == 0:
        return after, after
    before = expected_calibration_error(
        untemper(probs, preds["temperature"].to_numpy(dtype=np.float64)), labels, n_bins
    )
    return before, after


# ---------------------------------------------------------------------------
# 학습 모델을 기본 졸음 판정원으로 둘지의 판정 (PLAN 1.2절의 결정 D11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateInputs:
    """D11 판정에 넣는 교차검증 수치(모델 쪽은 앱 조건 평가의 값, PLAN 6장). `gate_inputs`가 묶음 결과에서 만든다."""

    macro_f1: float  # 후보 모델의 창 단위 macro-F1(영상 가중)
    lgbm_macro_f1: float  # 기준 LightGBM(alert 영상 기준선 조건, W = 0.5, PLAN 5.3절)의 같은 값
    two_class_video_accuracy: float
    false_alarms_per_hour: float  # alert 영상에서 모델 판정이 시작된 횟수 / 평가 시간
    alert_video_alarm_ratio: float  # 판정이 한 번이라도 난 alert 영상의 비율
    drowsy_video_alarm_ratio: float  # 판정이 한 번이라도 난 drowsy 영상의 비율


@dataclass(frozen=True, slots=True)
class GateCondition:
    """D11의 조건 하나: `value comparison threshold`가 참이면 passed."""

    name: str
    value: float
    comparison: str  # ">=" 또는 "<="
    threshold: float
    passed: bool


@dataclass(frozen=True, slots=True)
class GateResult:
    """D11 판정. source는 조건을 모두 만족하면 `model`, 아니면 `perclos`다(`drowsiness.source`에 넣을 값)."""

    conditions: tuple[GateCondition, ...]
    passed: bool
    source: str


def model_gate(metrics: GateInputs, cfg: ModelGate) -> GateResult:
    """D11을 글자 그대로 판정한다. 조건마다 값과 기준, 통과 여부를 돌려준다.

    다섯 조건을 모두 만족할 때만 규칙 엔진의 기본 졸음 판정원을 모델로 둔다:
      macro_f1 ≥ lgbm_macro_f1 − max_macro_f1_deficit(0.02)
      two_class_video_accuracy ≥ min_two_class_video_accuracy(0.70)
      false_alarms_per_hour ≤ max_false_alarms_per_hour(2)
      alert_video_alarm_ratio ≤ max_alert_video_alarm_ratio(0.10)
      drowsy_video_alarm_ratio ≥ min_drowsy_video_alarm_ratio(0.50)
    점추정 규칙이다. 경계값은 통과이고 부동소수 값을 여유 없이 그대로 비교한다. 유한하지 않은 값은 ValueError다.
    합성 데이터의 수치로는 이 판정을 쓰지 않는다(`write_reports`는 그 경우 판정원을 perclos로 둔다).
    """
    for field in dataclasses.fields(metrics):
        if not math.isfinite(getattr(metrics, field.name)):
            raise ValueError(f"gate input {field.name} is not finite")

    def at_least(name: str, value: float, threshold: float) -> GateCondition:
        return GateCondition(name, value, ">=", threshold, value >= threshold)

    def at_most(name: str, value: float, threshold: float) -> GateCondition:
        return GateCondition(name, value, "<=", threshold, value <= threshold)

    conditions = (
        at_least("macro_f1", metrics.macro_f1, metrics.lgbm_macro_f1 - cfg.max_macro_f1_deficit),
        at_least("two_class_video_accuracy", metrics.two_class_video_accuracy, cfg.min_two_class_video_accuracy),
        at_most("false_alarms_per_hour", metrics.false_alarms_per_hour, cfg.max_false_alarms_per_hour),
        at_most("alert_video_alarm_ratio", metrics.alert_video_alarm_ratio, cfg.max_alert_video_alarm_ratio),
        at_least("drowsy_video_alarm_ratio", metrics.drowsy_video_alarm_ratio, cfg.min_drowsy_video_alarm_ratio),
    )
    passed = all(condition.passed for condition in conditions)
    return GateResult(
        conditions=conditions, passed=passed, source=DROWSY_SOURCE_MODEL if passed else DROWSY_SOURCE_PERCLOS
    )


# ---------------------------------------------------------------------------
# 묶음 평가: fold별 값, 평균 ± 표준편차, 시드 표준편차, 부트스트랩 구간
# ---------------------------------------------------------------------------

_ALARM_STAT_FIELDS: tuple[str, ...] = ("onsets", "alarm_steps", "flagged")
_STAT_COLUMNS: tuple[str, ...] = (
    ("fold", "seed", "class_index", "gate_pass", "n_windows")
    + tuple(f"votes_{name}" for name in CLASS_ORDER)
    + ("correct", "two_correct")
    + tuple(f"{source}_{field}" for source in DROWSY_SOURCES for field in _ALARM_STAT_FIELDS)
)
ALARM_METRICS: tuple[str, ...] = (
    "onsets",
    "alert_hours",
    "onsets_per_hour",
    "alert_videos_with_onset",
    "alert_video_onset_ratio",
    "alarm_time_share",
    "flagged_windows_per_hour",
    "drowsy_video_onset_ratio",
)
CLASSIFICATION_METRICS: tuple[str, ...] = (
    ("video_accuracy", "video_accuracy_gated")
    + tuple(f"video_recall_{name}" for name in CLASS_ORDER)
    + ("window_macro_f1", "window_macro_f1_pooled")
    + tuple(f"window_recall_{name}" for name in CLASS_ORDER)
    + ("two_class_video_accuracy", "ece_before", "ece_after")
)
METRIC_NAMES: tuple[str, ...] = CLASSIFICATION_METRICS + tuple(
    f"{source}_{name}" for source in DROWSY_SOURCES for name in ALARM_METRICS
)
# 부트스트랩 구간을 구하는 지표(fold를 합친 값의 구간)
BOOTSTRAP_METRICS: tuple[str, ...] = (
    "video_accuracy",
    "video_accuracy_gated",
    "window_macro_f1",
    "two_class_video_accuracy",
) + tuple(
    f"{source}_{name}"
    for source in DROWSY_SOURCES
    for name in ("onsets_per_hour", "alert_video_onset_ratio", "drowsy_video_onset_ratio")
)
PAIRED_METRICS: tuple[str, ...] = ("video_accuracy", "window_macro_f1", "two_class_video_accuracy")
SUMMARY_COLUMNS: tuple[str, ...] = ("metric", "fold_mean", "fold_std", "seed_std", "pooled", "ci_low", "ci_high")
CONFUSION_NAMES: tuple[str, ...] = ("window_video_weighted", "window_pooled", "video", "two_class")


@dataclass(frozen=True, slots=True, eq=False)
class GroupResult:
    """묶음 (run_id, family, arm, feature_source, variant) 하나의 평가 결과 (`evaluate_group`).

    - feature_source, variant, supplementary: 예측 표의 표지 열(모듈 설명). `key`는 `group_key`의 꼴이다.
    - per_fold_seed: (fold, seed)마다 한 행, 열은 fold, seed와 `METRIC_NAMES`.
    - per_fold: fold마다 한 행, 값은 시드 평균(PLAN 6장의 "fold별 값").
    - summary: 지표마다 한 행(`SUMMARY_COLUMNS`). fold_mean ± fold_std는 per_fold 값의 평균과 표준편차,
      seed_std는 시드별 fold 평균의 표준편차, pooled는 fold를 합쳐 계산한 값의 시드 평균, ci_low·ci_high는 pooled의
      피험자 부트스트랩 95% 구간이다(`BOOTSTRAP_METRICS`만. 나머지는 NaN).
    - confusion: fold를 합친 혼동 행렬의 시드 평균. 키는 `CONFUSION_NAMES`:
      window_video_weighted(3×3, 영상 단위), window_pooled(3×3, 창 수), video(3×4, 마지막 열은 창 없음),
      two_class(2×3, 행은 alert·drowsy, 마지막 열은 창 없음). fold_confusion은 같은 것의 fold별 값이다.
    - alarm_sweep: 모델 시작 임계값을 바꿔 본 표(`SWEEP_COLUMNS`, 시드 평균).
    - stats: (seed, 영상)마다 한 행인 표. 부트스트랩과 짝지은 비교의 입력이다.
    - n_no_window: 창이 하나도 없어 오답으로 센 영상 수(시드 평균).
    """

    run_id: str
    family: str
    arm: str
    feature_source: str
    variant: str
    supplementary: bool
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    n_videos: int
    n_no_window: float
    per_fold_seed: pd.DataFrame
    per_fold: pd.DataFrame
    summary: pd.DataFrame
    confusion: dict[str, list[list[float]]]
    fold_confusion: dict[int, dict[str, list[list[float]]]]
    alarm_sweep: pd.DataFrame
    stats: pd.DataFrame

    @property
    def key(self) -> GroupKey:
        return group_key(self.run_id, self.family, self.arm, self.feature_source, self.variant)

    def value(self, metric: str, column: str = "pooled") -> float:
        """summary 표의 값 하나. column은 `SUMMARY_COLUMNS` 중 하나다."""
        row = self.summary.loc[self.summary["metric"] == metric, column]
        if len(row) != 1:
            raise KeyError(metric)
        return float(row.iloc[0])


def _aligned(source: pd.DataFrame, column: str, target: pd.DataFrame, fill: Any) -> np.ndarray:
    """source의 영상별 값을 target의 영상 순서로 옮긴다. source에 없는 영상은 fill이다."""
    index = pd.MultiIndex.from_arrays([source[name].to_numpy() for name in VIDEO_KEYS])
    positions = index.get_indexer(pd.MultiIndex.from_arrays([target[name].to_numpy() for name in VIDEO_KEYS]))
    values = source[column].to_numpy()
    out = np.full(len(target), fill, dtype=values.dtype)
    found = positions >= 0
    out[found] = values[positions[found]]
    return out


def _stat_arrays(stats: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: stats[name].to_numpy() for name in _STAT_COLUMNS}


def _stat_metrics(arrays: Mapping[str, np.ndarray], step_ms: int) -> dict[str, float]:
    """통계 표의 행들(시드 하나)에서 지표를 계산한다. ECE는 창 단위 값이라 여기에 없다."""
    class_index = arrays["class_index"]
    correct = arrays["correct"]
    gate = arrays["gate_pass"]
    n_windows = arrays["n_windows"].astype(np.float64)
    out: dict[str, float] = {
        "video_accuracy": _ratio(float(correct.sum()), float(len(correct))),
        "video_accuracy_gated": _ratio(float((correct & gate).sum()), float(gate.sum())),
    }
    for index, name in enumerate(CLASS_ORDER):
        of_class = class_index == index
        out[f"video_recall_{name}"] = _ratio(float((correct & of_class).sum()), float(of_class.sum()))
    votes = np.stack([arrays[f"votes_{name}"] for name in CLASS_ORDER], axis=1).astype(np.float64)
    pooled, weighted = _window_confusions(class_index, votes, n_windows)
    macro_f1, recall, _, _ = _f1_numbers(weighted)
    out["window_macro_f1"] = macro_f1
    out["window_macro_f1_pooled"] = _f1_numbers(pooled)[0]
    for index, name in enumerate(CLASS_ORDER):
        out[f"window_recall_{name}"] = float(recall[index])
    in_two = (class_index == CLASS_ALERT) | (class_index == CLASS_DROWSY)
    out["two_class_video_accuracy"] = _ratio(float((arrays["two_correct"] & in_two).sum()), float(in_two.sum()))
    for source in DROWSY_SOURCES:
        numbers = _alarm_numbers(
            class_index,
            n_windows,
            n_windows * step_ms,
            arrays[f"{source}_onsets"],
            arrays[f"{source}_alarm_steps"],
            arrays[f"{source}_flagged"],
        )
        for name in ALARM_METRICS:
            out[f"{source}_{name}"] = numbers[name]
    return out


def _seed_mean_metrics(arrays: Mapping[str, np.ndarray], step_ms: int) -> dict[str, float]:
    """시드마다 지표를 계산해 평균한다."""
    seeds = arrays["seed"]
    totals: dict[str, float] = {}
    unique = np.unique(seeds)
    for seed in unique:
        mask = seeds == seed
        for name, value in _stat_metrics({key: values[mask] for key, values in arrays.items()}, step_ms).items():
            totals[name] = totals.get(name, 0.0) + value
    return {name: value / len(unique) for name, value in totals.items()}


def _mean_matrices(items: Sequence[Mapping[str, np.ndarray]]) -> dict[str, list[list[float]]]:
    return {name: np.mean([item[name] for item in items], axis=0).tolist() for name in CONFUSION_NAMES}


def _confusions(votes: pd.DataFrame, two: pd.DataFrame) -> dict[str, np.ndarray]:
    window = vote_window_metrics(votes)
    return {
        "window_video_weighted": np.array(window.video_weighted.confusion, dtype=np.float64),
        "window_pooled": np.array(window.pooled.confusion, dtype=np.float64),
        "video": np.array(video_metrics(votes).confusion, dtype=np.float64),
        "two_class": np.array(video_metrics(two, TWO_CLASSES).confusion, dtype=np.float64),
    }


def _check_fold_coverage(
    preds: pd.DataFrame, videos: pd.DataFrame, key: GroupKey
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """묶음 하나의 예측이 영상 표의 fold를 빠짐없이 덮는지 검사하고 (시드들, fold들)을 돌려준다.

    규칙은 `evaluate_group`의 설명에 있다: 시드마다 예측이 있는 fold가 같아야 하고, 영상 표의 fold 가운데 예측이
    하나도 없는 fold가 있으면 ValueError다(영상 표가 그 fold의 영상이 모두 창 0개라고 할 때는 예외).
    """
    seeds = tuple(sorted(int(value) for value in preds["seed"].unique()))
    folds = tuple(sorted(int(value) for value in videos["fold"].unique()))
    covered = {seed: sorted(set(preds.loc[preds["seed"].to_numpy() == seed, "fold"].tolist())) for seed in seeds}
    if any(covered[seed] != covered[seeds[0]] for seed in seeds):
        raise ValueError(f"seeds of one group have predictions for different folds: {covered}")
    unevaluated = sorted(set(folds) - set(covered[seeds[0]]))
    if unevaluated and "n_windows" in videos.columns:
        with_windows = set(videos.loc[videos["n_windows"].to_numpy() > 0, "fold"].tolist())
        unevaluated = [fold for fold in unevaluated if fold in with_windows]
    if unevaluated:
        raise ValueError(
            f"group {key} has no predictions for fold(s) {unevaluated} of the video table; an unevaluated fold "
            "must not be scored as wrong. Evaluate every listed fold or restrict the fold assignment"
        )
    return seeds, folds


def evaluate_group(
    preds: pd.DataFrame,
    videos: pd.DataFrame,
    rules_cfg: RulesConfig,
    cfg: EvaluationSettings,
    bootstrap: bool = True,
) -> GroupResult:
    """묶음 (run_id, family, arm, feature_source, variant) 하나의 예측 표(여러 시드, 여러 fold)를 평가한다 (PLAN 6장).

    - videos(영상 표)의 영상이 분모다(D7). fold는 이 표의 fold 열에서 온다. 예측이 없는 영상은 오답이다.
    - 영상 표에 있는 fold인데 이 묶음의 예측이 하나도 없으면 ValueError다. 평가하지 않은 fold(학습이 중간에
      멈췄거나 일부 fold만 돌린 표)를 0점으로 싣지 않기 위해서다. D7은 "창이 없는 영상은 오답"이라고 했을 뿐
      평가하지 않은 fold를 오답으로 세라고 하지 않았다. 예측이 있는 fold 안에서 창이 없는 영상은 그대로 오답이다.
    - 시드마다 `video_vote`, `two_class`, `alarm_videos`(model, perclos, either)를 계산해 (seed, 영상)마다 한 행인
      통계 표를 만들고, fold별·시드별 값과 fold를 합친 값을 모두 그 표에서 계산한다. ECE는 창에서 따로 계산한다.
    - 부트스트랩(bootstrap=True): 통계 표의 피험자를 `cfg.bootstrap_n`번 복원 추출해(`cfg.bootstrap_seed`)
      fold를 합친 값(시드 평균)의 95% 구간을 구한다.
    결과의 구성은 `GroupResult`에 있다.
    가정: 부트스트랩 구간은 "fold 평균"이 아니라 "fold를 합친 값"의 구간이다(피험자를 fold와 무관하게 추출한다).
      fold 크기가 같으면 정확도는 두 값이 같고 macro-F1은 조금 다를 수 있어 표에 두 값을 함께 싣는다.
    가정: "시드 간 표준편차"는 시드마다 fold별 값을 평균한 뒤 그 평균들의 표준편차다.
    시드마다 예측이 있는 fold가 다르면 ValueError다(시드 평균이 서로 다른 fold를 섞게 된다).
    가정: 영상 표에 `n_windows` 열이 있고 어떤 fold의 영상이 모두 창 0개이면, 그 fold에 예측이 없는 것은 평가하지
      않은 것이 아니라 창이 없는 것이다. 그 fold는 받아들이고 영상을 모두 오답으로 센다(D7. 학습 코드가 창이 없는
      시험 fold를 건너뛰는 경우). `n_windows` 열이 없으면 예측이 없는 fold는 언제나 거부한다.
    표지 열이 없는 표(검사를 거치지 않은 예전 형식)도 받는다. 그 묶음의 표지는 기본값이다.
    """
    _require_columns(preds, REQUIRED_PREDICTION_COLUMNS, "prediction table")
    _require_columns(videos, VIDEO_COLUMNS, "video table")
    if len(preds) == 0:
        raise ValueError("the prediction table of a group is empty")
    _require_single(preds, _present(GROUP_KEYS, preds), "group prediction table")
    first = preds.iloc[0]
    key = _key_of(first)
    is_supplementary = False
    if "supplementary" in preds.columns:
        flags = preds["supplementary"].to_numpy(dtype=bool)
        if (flags != flags[0]).any():
            raise ValueError(f"group {key} is supplementary in some rows only")
        is_supplementary = bool(flags[0])
    seeds, folds = _check_fold_coverage(preds, videos, key)
    step_ms = int(rules_cfg.clock.step_ms)

    frames: list[pd.DataFrame] = []
    confusion_by_seed: list[dict[str, np.ndarray]] = []
    fold_confusion_by_seed: dict[int, list[dict[str, np.ndarray]]] = {fold: [] for fold in folds}
    ece: dict[tuple[int, int], tuple[float, float]] = {}
    pooled_ece: list[tuple[float, float]] = []
    sweeps: list[pd.DataFrame] = []
    no_window: list[int] = []
    for seed in seeds:
        run = preds[preds["seed"].to_numpy() == seed]
        votes = video_vote(run, videos)
        two = two_class(run, videos)
        frame = votes[["subject", "video_id", "fold", "class_index", "gate_pass", "n_windows"]].copy()
        frame["seed"] = seed
        for name in CLASS_ORDER:
            frame[f"votes_{name}"] = votes[f"votes_{name}"].to_numpy()
        frame["correct"] = votes["correct"].to_numpy()
        frame["two_correct"] = _aligned(two, "correct", frame, False)
        for source in DROWSY_SOURCES:
            alarms = alarm_videos(run, rules_cfg, source, videos)
            frame[f"{source}_onsets"] = _aligned(alarms, "onsets", frame, 0)
            frame[f"{source}_alarm_steps"] = _aligned(alarms, "alarm_steps", frame, 0)
            frame[f"{source}_flagged"] = _aligned(alarms, "flagged_windows", frame, 0)
        frames.append(frame)
        no_window.append(int((votes["n_windows"].to_numpy() == 0).sum()))

        confusion_by_seed.append(_confusions(votes, two))
        run_fold = run["fold"].to_numpy()
        for fold in folds:
            fold_confusion_by_seed[fold].append(
                _confusions(votes[votes["fold"].to_numpy() == fold], two[two["fold"].to_numpy() == fold])
            )
            ece[(fold, seed)] = calibration_error(run[run_fold == fold])
        pooled_ece.append(calibration_error(run))
        if len(cfg.alarm_threshold_sweep) > 0:
            sweeps.append(alarm_sweep(run, rules_cfg, cfg.alarm_threshold_sweep, DROWSY_SOURCE_MODEL, videos))

    stats = pd.concat(frames, ignore_index=True)
    arrays = _stat_arrays(stats)
    records: list[dict[str, float]] = []
    for fold in folds:
        for seed in seeds:
            mask = (arrays["fold"] == fold) & (arrays["seed"] == seed)
            metrics = _stat_metrics({name: values[mask] for name, values in arrays.items()}, step_ms)
            metrics["ece_before"], metrics["ece_after"] = ece[(fold, seed)]
            records.append({"fold": fold, "seed": seed, **metrics})
    names = list(METRIC_NAMES)
    per_fold_seed = pd.DataFrame(records, columns=["fold", "seed"] + names)
    per_fold = per_fold_seed.groupby("fold", sort=True)[names].mean().reset_index()
    by_seed = per_fold_seed.groupby("seed", sort=True)[names].mean()

    pooled = _seed_mean_metrics(arrays, step_ms)
    pooled["ece_before"] = float(np.mean([before for before, _ in pooled_ece]))
    pooled["ece_after"] = float(np.mean([after for _, after in pooled_ece]))
    ci_low = dict.fromkeys(names, math.nan)
    ci_high = dict.fromkeys(names, math.nan)
    if bootstrap:

        def statistic(tables: Sequence[pd.DataFrame]) -> list[float]:
            values = _seed_mean_metrics(_stat_arrays(tables[0]), step_ms)
            return [values[name] for name in BOOTSTRAP_METRICS]

        replicates = _replicates([stats], statistic, cfg.bootstrap_n, cfg.bootstrap_seed)
        for index, name in enumerate(BOOTSTRAP_METRICS):
            ci_low[name], ci_high[name] = _interval(replicates[:, index], CONFIDENCE_LEVEL)

    summary = pd.DataFrame(
        {
            "metric": names,
            "fold_mean": [float(per_fold[name].mean()) for name in names],
            "fold_std": [float(per_fold[name].std(ddof=1)) for name in names],
            "seed_std": [float(by_seed[name].std(ddof=1)) for name in names],
            "pooled": [pooled[name] for name in names],
            "ci_low": [ci_low[name] for name in names],
            "ci_high": [ci_high[name] for name in names],
        },
        columns=list(SUMMARY_COLUMNS),
    )
    if sweeps:
        sweep = pd.concat(sweeps, ignore_index=True).groupby("onset_p", sort=True).mean().reset_index()
    else:
        sweep = pd.DataFrame(columns=list(SWEEP_COLUMNS))
    return GroupResult(
        run_id=str(first["run_id"]),
        family=str(first["family"]),
        arm=str(first["arm"]),
        feature_source=str(first.get("feature_source", FEATURE_SOURCE_NATIVE)),
        variant=str(first.get("variant", VARIANT_BASE)),
        supplementary=is_supplementary,
        seeds=seeds,
        folds=folds,
        n_videos=len(frames[0]),
        n_no_window=float(np.mean(no_window)),
        per_fold_seed=per_fold_seed,
        per_fold=per_fold,
        summary=summary,
        confusion=_mean_matrices(confusion_by_seed),
        fold_confusion={fold: _mean_matrices(items) for fold, items in fold_confusion_by_seed.items()},
        alarm_sweep=sweep,
        stats=stats,
    )


def gate_inputs(candidate: GroupResult, reference: GroupResult) -> GateInputs:
    """묶음 결과 둘에서 D11의 입력을 만든다. candidate는 후보 모델의 앱 조건 평가, reference는 기준 LightGBM이다.

    가정: 계획서는 D11의 "교차검증 결과"가 어느 집계인지 적지 않았다. 분류 지표(창 단위 영상 가중 macro-F1, 2클래스
      영상 정확도)는 리포트의 대표값인 fold 평균(시드 평균)을, 오경보 수치(시간당 횟수, 영상 비율)는 6장이 주 수치로
      정한 "5개 fold를 합친 값"(시드 평균)을 쓴다.
    """
    return GateInputs(
        macro_f1=candidate.value("window_macro_f1", "fold_mean"),
        lgbm_macro_f1=reference.value("window_macro_f1", "fold_mean"),
        two_class_video_accuracy=candidate.value("two_class_video_accuracy", "fold_mean"),
        false_alarms_per_hour=candidate.value(f"{DROWSY_SOURCE_MODEL}_onsets_per_hour"),
        alert_video_alarm_ratio=candidate.value(f"{DROWSY_SOURCE_MODEL}_alert_video_onset_ratio"),
        drowsy_video_alarm_ratio=candidate.value(f"{DROWSY_SOURCE_MODEL}_drowsy_video_onset_ratio"),
    )


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiteratureRow:
    """비교표의 문헌 행. 다른 연구자가 보고한 값이며 이 저장소에서 재현한 값이 아니다 (PLAN 2.4절)."""

    method: str
    three_class: str  # 3클래스 영상 단위 정확도
    two_class: str  # 2클래스(alert 대 drowsy) 영상 단위 정확도
    note: str


LITERATURE_LABEL = "문헌 보고값 (다른 연구자가 보고한 값. 이 저장소에서 재현하지 않았다)"
# 값과 주석은 PLAN 2.4절의 것이다. Xu 2025를 3클래스 열에 둔 근거는 `write_reports`의 가정에 적었다.
LITERATURE: tuple[LiteratureRow, ...] = (
    LiteratureRow(
        "HM-LSTM (UTA-RLDD 원 논문)",
        "65.2%",
        "-",
        "30회 깜빡임 시퀀스의 다수결, 5 fold 평균. 클래스별 재현율 0.81 / 0.32 / 0.82. "
        "공개 코드는 174개 영상만 평가하는 것으로 보이고 검증 분할이 없다",
    ),
    LiteratureRow("사람 판정 (UTA-RLDD 원 논문)", "57.8%", "-", "같은 논문이 보고한 사람의 판정"),
    LiteratureRow("Wolter 2025", "52.5%", "80.4%", "피험자 분리 조건"),
    LiteratureRow("Magán 2022", "-", "55~63%", "피험자 분리 조건"),
    LiteratureRow(
        "Xu 2025", "86.5%", "-", "관찰자 판정과 라벨이 다른 구간을 버린 뒤의 값이다. 같은 조건의 값이 아니다"
    ),
)

SUMMARY_FILE = "summary.md"
SUMMARY_JSON_FILE = "summary.json"
PER_FOLD_CSV_FILE = "per_fold.csv"
PER_FOLD_JSON_FILE = "per_fold.json"
ALARM_CSV_FILE = "alarm.csv"
ALARM_SWEEP_CSV_FILE = "alarm_sweep.csv"
VIDEOS_CSV_FILE = "videos.csv"
EXCLUSIONS_CSV_FILE = "exclusions.csv"
CONFIG_COPY_FILE = "config.yaml"
RULES_COPY_FILE = "rules.json"
SUPPLEMENTARY_PREFIX = "supplementary_"  # 부가 묶음의 표 파일: supplementary_per_fold.csv 등
PAIRED_COLUMNS: tuple[str, ...] = (
    "a",
    "b",
    "metric",
    "value_a",
    "value_b",
    "difference",
    "ci_low",
    "ci_high",
    "differs",
    "supplementary",
)


@dataclass(frozen=True, slots=True, eq=False)
class Report:
    """`write_reports`의 결과.

    - results: 주 결과 묶음의 평가 결과(키 오름차순). 부가 묶음은 여기 없다.
    - paired: 주 결과 묶음끼리의 짝지은 비교 표. 열(`PAIRED_COLUMNS`)은 a, b(묶음 이름), metric, value_a, value_b,
      difference, ci_low, ci_high, differs(구간이 0을 벗어났는지), supplementary(이 표에서는 항상 False).
    - gate: D11 판정(요청하지 않았으면 None). 조건별 값은 데이터가 합성이어도 그대로 계산해 싣는다.
    - drowsiness_source: `drowsiness.source`에 넣을 값. 실제 데이터에서 D11을 모두 만족할 때만 `model`이고,
      그 밖에는(판정을 요청하지 않았거나, 합성 데이터이거나, 조건을 만족하지 못하면) `perclos`다.
    - supplementary: 부가 묶음의 평가 결과(키 오름차순).
    - supplementary_paired: 부가 묶음이 한쪽이라도 들어간 짝지은 비교 표(같은 열, supplementary는 항상 True).
    """

    run_dir: Path
    summary_path: Path
    results: tuple[GroupResult, ...]
    paired: pd.DataFrame
    gate: GateResult | None
    drowsiness_source: str
    supplementary: tuple[GroupResult, ...]
    supplementary_paired: pd.DataFrame


def _check_fold_assignment(
    fold_subjects: Mapping[int, Iterable[int]], preds: pd.DataFrame, videos: pd.DataFrame
) -> dict[int, tuple[int, ...]]:
    """fold별 시험 피험자 집합이 서로 겹치지 않는지, 예측 표와 영상 표가 그 배정을 따르는지 검사한다."""
    folds = {int(fold): tuple(sorted({int(s) for s in subjects})) for fold, subjects in fold_subjects.items()}
    if not folds:
        raise ValueError("the fold assignment is empty")
    assert_subject_disjoint(*[folds[fold] for fold in sorted(folds)])

    unknown = sorted({int(fold) for fold in preds["fold"].unique()} - set(folds))
    if unknown:
        raise ValueError(f"prediction table holds folds {unknown} that the fold assignment does not list")
    members = {fold: set(subjects) for fold, subjects in folds.items()}
    for what, table in (("prediction table", preds), ("video table", videos)):
        pairs = table[["subject", "fold"]].drop_duplicates()
        stray = [
            f"subject {subject} in fold {fold}"
            for subject, fold in zip(pairs["subject"].tolist(), pairs["fold"].tolist())
            if fold in members and subject not in members[fold]
        ]
        if stray:
            raise SubjectOverlapError(
                f"{what} evaluates subjects outside the test set of their fold ({len(stray)}): {stray[:10]}"
            )
    listed = set(videos.loc[videos["fold"].isin(list(folds)), "subject"].tolist())
    absent = sorted(set().union(*members.values()) - listed)
    if absent:
        raise ValueError(
            f"the video table has no video of test subjects {absent[:10]}; they would drop out of the denominator"
        )
    return folds


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_count(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"


def _fmt_percent(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return lines


def _group_name(result: GroupResult) -> str:
    """리포트에 적는 묶음 이름. 표지가 기본값이면 `run_id / family / features=arm`이고, 기본값이 아닌 표지만 덧붙인다."""
    name = f"{result.run_id} / {result.family} / features={result.arm}"
    if result.feature_source != FEATURE_SOURCE_NATIVE:
        name += f" [{result.feature_source}]"
    if result.variant != VARIANT_BASE:
        name += f" / variant={result.variant}"
    return name


def _heading(result: GroupResult) -> str:
    return f"### {'부가: ' if result.supplementary else ''}{_group_name(result)}"


def _jsonable(value: Any) -> Any:
    """JSON으로 쓸 수 있는 값으로 바꾼다. NaN과 무한대는 null이다."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(frame: pd.DataFrame, path: Path, note: str) -> None:
    """첫 줄에 데이터 종류를 적은 주석 줄을 두고 CSV를 쓴다. 읽을 때는 `pd.read_csv(path, skiprows=1)`."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {note}\n")
        frame.to_csv(handle, index=False, lineterminator="\n")


_FOLD_TABLE_ROWS: tuple[tuple[str, str], ...] = (
    ("video_accuracy", "영상 정확도, 3클래스 (D7: 분모는 전체 영상)"),
    ("video_accuracy_gated", "영상 정확도, 품질 기준 통과 영상만 (보조)"),
    ("video_recall_alert", "영상 재현율 alert"),
    ("video_recall_low_vigilant", "영상 재현율 low_vigilant"),
    ("video_recall_drowsy", "영상 재현율 drowsy"),
    ("window_macro_f1", "창 macro-F1, 영상 가중 (주)"),
    ("window_macro_f1_pooled", "창 macro-F1, 창 합산 (보조)"),
    ("window_recall_alert", "창 재현율 alert (영상 가중)"),
    ("window_recall_low_vigilant", "창 재현율 low_vigilant (영상 가중)"),
    ("window_recall_drowsy", "창 재현율 drowsy (영상 가중)"),
    ("two_class_video_accuracy", "영상 정확도, 2클래스 (D8)"),
    ("ece_before", "ECE, 온도 적용 전"),
    ("ece_after", "ECE, 온도 적용 후"),
)
_ALARM_TABLE_ROWS: tuple[tuple[str, str, bool], ...] = (
    ("onsets", "alert 영상에서 판정이 시작된 횟수 (fold 합침)", True),
    ("alert_hours", "alert 평가 시간 (h, fold 합침)", False),
    ("onsets_per_hour", "시간당 시작 횟수 (주 수치)", False),
    ("alert_videos_with_onset", "판정이 한 번이라도 난 alert 영상 수", True),
    ("alert_video_onset_ratio", "그 비율", False),
    ("alarm_time_share", "alert 평가 시간 중 판정 상태였던 비율", False),
    ("flagged_windows_per_hour", "창 단위 값 (D9): 해당 창 수 / 시간", False),
    ("drowsy_video_onset_ratio", "drowsy 영상 중 판정이 한 번이라도 난 비율", False),
)


def _with_interval(result: GroupResult, metric: str, value: str) -> str:
    low, high = result.value(metric, "ci_low"), result.value(metric, "ci_high")
    if not (math.isfinite(low) and math.isfinite(high)):
        return value
    return f"{value} [{_fmt(low)}, {_fmt(high)}]"


def _fold_section(result: GroupResult, heading: bool = True) -> list[str]:
    seeds = ", ".join(str(seed) for seed in result.seeds)
    lines = [
        *([_heading(result), ""] if heading else []),
        f"시드 {seeds}. 논리 영상 {result.n_videos}개 중 창이 없어 오답으로 센 영상 {_fmt_count(result.n_no_window)}개.",
        "",
    ]
    header = (
        ["지표"]
        + [f"fold {fold}" for fold in result.folds]
        + ["fold 평균 ± 표준편차", "시드 표준편차", "fold 합침 [95% 구간]"]
    )
    rows = []
    for metric, label in _FOLD_TABLE_ROWS:
        rows.append(
            [f"{label} `{metric}`"]
            + [_fmt(float(value)) for value in result.per_fold[metric]]
            + [
                f"{_fmt(result.value(metric, 'fold_mean'))} ± {_fmt(result.value(metric, 'fold_std'))}",
                _fmt(result.value(metric, "seed_std")),
                _with_interval(result, metric, _fmt(result.value(metric))),
            ]
        )
    return lines + _md_table(header, rows) + [""]


def _matrix_table(
    matrix: Sequence[Sequence[float]], row_labels: Sequence[str], column_labels: Sequence[str]
) -> list[str]:
    integral = all(float(value).is_integer() for row in matrix for value in row)
    rows = [
        [label] + [f"{value:.0f}" if integral else f"{value:.2f}" for value in row]
        for label, row in zip(row_labels, matrix)
    ]
    return _md_table(["정답 \\ 판정"] + list(column_labels), rows)


def _confusion_section(result: GroupResult) -> list[str]:
    three = list(CLASS_ORDER)
    two = [CLASS_ORDER[index] for index in TWO_CLASSES]
    lines = [_heading(result), ""]
    lines += ["창 단위, 영상 가중 (주. 값은 영상 수 단위다. 영상 하나가 자기 창들의 예측 비율을 더한다)", ""]
    lines += _matrix_table(result.confusion["window_video_weighted"], three, three) + [""]
    lines += ["창 단위, 창 합산 (보조. 값은 창 수)", ""]
    lines += _matrix_table(result.confusion["window_pooled"], three, three) + [""]
    lines += ["영상 단위, 3클래스 (D7. 마지막 열은 창이 없어 오답으로 센 영상)", ""]
    lines += _matrix_table(result.confusion["video"], three, three + ["창 없음"]) + [""]
    lines += ["영상 단위, 2클래스 (D8. alert·drowsy 영상만)", ""]
    lines += _matrix_table(result.confusion["two_class"], two, two + ["창 없음"]) + [""]
    return lines


def _alarm_section(result: GroupResult, heading: bool = True) -> list[str]:
    n_alert = int((result.stats["class_index"].to_numpy() == CLASS_ALERT).sum() / len(result.seeds))
    lines = [*([_heading(result), ""] if heading else []), f"alert 논리 영상 {n_alert}개(분모).", ""]
    rows = []
    for metric, label, is_count in _ALARM_TABLE_ROWS:
        row = [label]
        for source in DROWSY_SOURCES:
            name = f"{source}_{metric}"
            value = result.value(name)
            row.append(_with_interval(result, name, _fmt_count(value) if is_count else _fmt(value)))
        rows.append(row)
    row = ["fold별 시작 횟수 / 평가 시간(h)"]
    for source in DROWSY_SOURCES:
        row.append(
            ", ".join(
                f"{int(fold)}: {_fmt_count(float(onsets))}/{_fmt(float(hours), 2)}"
                for fold, onsets, hours in zip(
                    result.per_fold["fold"],
                    result.per_fold[f"{source}_onsets"],
                    result.per_fold[f"{source}_alert_hours"],
                )
            )
        )
    rows.append(row)
    lines += _md_table(["항목"] + list(DROWSY_SOURCES), rows) + [""]
    if len(result.alarm_sweep) > 0:
        lines += ["모델 시작 임계값을 바꿔 본 값 (보고용. 주 수치의 임계값은 사전에 고정한 rules.yaml의 값이다)", ""]
        sweep_rows = [
            [
                _fmt(float(row["onset_p"]), 2),
                _fmt_count(float(row["onsets"])),
                _fmt(float(row["onsets_per_hour"])),
                _fmt_count(float(row["alert_videos_with_onset"])),
                _fmt(float(row["alarm_time_share"])),
                _fmt(float(row["drowsy_video_onset_ratio"])),
            ]
            for _, row in result.alarm_sweep.iterrows()
        ]
        lines += _md_table(
            ["onset_p", "시작 횟수", "시간당", "판정이 난 alert 영상", "판정 상태 시간 비율", "판정이 난 drowsy 영상 비율"],
            sweep_rows,
        ) + [""]
    return lines


def _paired_table(
    pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
    results: Mapping[GroupKey, GroupResult],
    step_ms: int,
    cfg: EvaluationSettings,
) -> pd.DataFrame:
    """짝지은 비교 표(`PAIRED_COLUMNS`). 한쪽이라도 부가 묶음이면 그 행의 supplementary가 True다."""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        key_a, key_b = (_canonical_key(key) for key in pair)
        for key in (key_a, key_b):
            if key not in results:
                raise ValueError(f"paired comparison names a group without predictions: {key}")
        a, b = results[key_a], results[key_b]

        def statistic(tables: Sequence[pd.DataFrame]) -> list[float]:
            first = _seed_mean_metrics(_stat_arrays(tables[0]), step_ms)
            second = _seed_mean_metrics(_stat_arrays(tables[1]), step_ms)
            return [first[name] - second[name] for name in PAIRED_METRICS]

        replicates = _replicates([a.stats, b.stats], statistic, cfg.bootstrap_n, cfg.bootstrap_seed)
        for index, metric in enumerate(PAIRED_METRICS):
            low, high = _interval(replicates[:, index], CONFIDENCE_LEVEL)
            rows.append(
                {
                    "a": _group_name(a),
                    "b": _group_name(b),
                    "metric": metric,
                    "value_a": a.value(metric),
                    "value_b": b.value(metric),
                    "difference": a.value(metric) - b.value(metric),
                    "ci_low": low,
                    "ci_high": high,
                    "differs": bool(low > 0.0 or high < 0.0),
                    "supplementary": bool(a.supplementary or b.supplementary),
                }
            )
    table = pd.DataFrame(rows, columns=list(PAIRED_COLUMNS))
    return table.astype({"differs": bool, "supplementary": bool})


def _paired_lines(paired: pd.DataFrame) -> list[str]:
    return _md_table(
        ["a", "b", "지표", "a", "b", "a − b [95% 구간]", "차이 있음"],
        [
            [
                row["a"],
                row["b"],
                f"`{row['metric']}`",
                _fmt(float(row["value_a"])),
                _fmt(float(row["value_b"])),
                f"{_fmt(float(row['difference']))} [{_fmt(float(row['ci_low']))}, {_fmt(float(row['ci_high']))}]",
                "예" if row["differs"] else "아니오",
            ]
            for _, row in paired.iterrows()
        ],
    )


def _group_labels(result: GroupResult) -> dict[str, Any]:
    """표 파일과 JSON에 적는 묶음의 식별 값(`GROUP_KEYS` 순서)."""
    return {
        "run_id": result.run_id,
        "family": result.family,
        "arm": result.arm,
        "feature_source": result.feature_source,
        "variant": result.variant,
    }


def _per_fold_frame(results: Sequence[GroupResult]) -> pd.DataFrame:
    columns = list(GROUP_KEYS) + ["fold", "n_seeds"] + list(METRIC_NAMES)
    if len(results) == 0:
        return pd.DataFrame(columns=columns)
    frames = [result.per_fold.assign(**_group_labels(result), n_seeds=len(result.seeds)) for result in results]
    return pd.concat(frames, ignore_index=True)[columns]


def _sweep_frame(results: Sequence[GroupResult]) -> pd.DataFrame:
    columns = list(GROUP_KEYS) + list(SWEEP_COLUMNS)
    if len(results) == 0:
        return pd.DataFrame(columns=columns)
    frames = [result.alarm_sweep.assign(**_group_labels(result)) for result in results]
    return pd.concat(frames, ignore_index=True)[columns]


def _alarm_frame(results: Sequence[GroupResult]) -> pd.DataFrame:
    columns = (
        list(GROUP_KEYS) + ["source"] + list(ALARM_METRICS) + ["onsets_per_hour_ci_low", "onsets_per_hour_ci_high"]
    )
    rows: list[dict[str, Any]] = []
    for result in results:
        for source in DROWSY_SOURCES:
            row: dict[str, Any] = _group_labels(result)
            row["source"] = source
            for metric in ALARM_METRICS:
                row[metric] = result.value(f"{source}_{metric}")
            row["onsets_per_hour_ci_low"] = result.value(f"{source}_onsets_per_hour", "ci_low")
            row["onsets_per_hour_ci_high"] = result.value(f"{source}_onsets_per_hour", "ci_high")
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _fold_records(result: GroupResult) -> dict[str, Any]:
    """per_fold.json의 묶음 하나."""
    return {
        **_group_labels(result),
        "supplementary": result.supplementary,
        "seeds": result.seeds,
        "folds": [
            {
                "fold": fold,
                "metrics": result.per_fold.loc[result.per_fold["fold"] == fold, list(METRIC_NAMES)].iloc[0].to_dict(),
                "per_seed": {
                    str(int(row["seed"])): {name: row[name] for name in METRIC_NAMES}
                    for _, row in result.per_fold_seed[result.per_fold_seed["fold"] == fold].iterrows()
                },
                "confusion": result.fold_confusion[fold],
            }
            for fold in result.folds
        ],
    }


def _summary_record(result: GroupResult) -> dict[str, Any]:
    """summary.json의 묶음 하나."""
    return {
        **_group_labels(result),
        "supplementary": result.supplementary,
        "seeds": result.seeds,
        "n_videos": result.n_videos,
        "n_no_window": result.n_no_window,
        "summary": {
            str(row["metric"]): {name: row[name] for name in SUMMARY_COLUMNS[1:]}
            for _, row in result.summary.iterrows()
        },
        "confusion": result.confusion,
        "alarm_sweep": result.alarm_sweep.to_dict(orient="records"),
    }


def window_counts(
    steps_by_label: Mapping[str, pd.DataFrame], window_steps: int, strides: Sequence[int]
) -> dict[str, dict[int, dict[str, int]]]:
    """조건별·보폭별·클래스별 창 수 (PLAN 4.4절의 리포트 항목). `write_reports(window_counts=...)`에 그대로 넘긴다.

    - steps_by_label: 표에 적을 이름 → 스텝 표(`StepTable.steps`). 이름은 보통 정규화 조건 이름이고, 다른 추출의
      스텝 표를 함께 실으려면 `alert_baseline [fps15]`처럼 구분해 적는다.
    - strides: 세어 볼 보폭들(보통 `windows.train_stride`와 `windows.eval_stride`).
    돌려주는 값: 이름 → 보폭 → 클래스 이름 → 창 수. 세는 것은 `dataset.windows_per_class`가 한다(창을 만들지 않는다).
    """
    return {
        str(label): {int(stride): windows_per_class(steps, window_steps, int(stride)) for stride in strides}
        for label, steps in steps_by_label.items()
    }


def supplementary_pairs(preds: pd.DataFrame) -> list[tuple[GroupKey, GroupKey]]:
    """예측 표의 부가 묶음마다 `(부가 묶음, 기준 묶음)` 키 쌍을 만든다. `write_reports(paired=...)`에 더해 넘긴다.

    기준 묶음은 부가 묶음에서 다른 점 하나를 기본으로 되돌린 것이다. 차례로 보고 처음 해당하는 것 하나를 쓴다.
      1. family가 `_int8`로 끝나면 그 접미사를 뗀 계열(float 대 INT8).
      2. feature_source가 native가 아니면 같은 묶음의 native(원본 추출 대 15 fps 모사, spec 상수 대 fold별 상수).
      3. variant가 base가 아니면 같은 묶음의 base(순서형 라벨, 속도 특징 제외).
    기준 묶음이 표에 없으면 그 부가 묶음은 쌍을 만들지 않는다(다른 run_id로 돌린 실행과의 비교는 쌍을 직접 적는다).
    쌍은 부가 묶음의 키 오름차순이다. 표지 열이 없는 표에는 부가 묶음이 없으므로 빈 목록이다.
    가정: 계획서는 부가 분석의 비교 상대를 적지 않았다. "다른 점 하나를 되돌린 묶음"이라는 규칙은 여기서 정한 것이다.
    """
    keys = _present(GROUP_KEYS, preds)
    _require_columns(preds, keys, "prediction table")
    if "supplementary" not in preds.columns:
        return []
    groups = preds[keys + ["supplementary"]].drop_duplicates()
    known: set[GroupKey] = set()
    supplementary: list[tuple[str, str, str, str, str]] = []
    for row in groups.to_dict(orient="records"):
        labels = {name: row.get(name) for name in ("feature_source", "variant")}
        source = FEATURE_SOURCE_NATIVE if pd.isna(labels["feature_source"]) else str(labels["feature_source"])
        variant = VARIANT_BASE if pd.isna(labels["variant"]) else str(labels["variant"])
        full = (str(row["run_id"]), str(row["family"]), str(row["arm"]), source, variant)
        known.add(group_key(*full))
        if not pd.isna(row["supplementary"]) and bool(row["supplementary"]):
            supplementary.append(full)
    pairs: list[tuple[GroupKey, GroupKey]] = []
    for run_id, family, arm, source, variant in sorted(set(supplementary)):
        if family.endswith(INT8_SUFFIX) and len(family) > len(INT8_SUFFIX):
            base = group_key(run_id, family[: -len(INT8_SUFFIX)], arm, source, variant)
        elif source != FEATURE_SOURCE_NATIVE:
            base = group_key(run_id, family, arm, FEATURE_SOURCE_NATIVE, variant)
        elif variant != VARIANT_BASE:
            base = group_key(run_id, family, arm, source, VARIANT_BASE)
        else:
            continue
        if base in known:
            pairs.append((group_key(run_id, family, arm, source, variant), base))
    return pairs


def _baseline_overview(videos: pd.DataFrame, nominal_ms: int) -> dict[str, Any] | None:
    """영상 표의 기준선 열에서 PLAN 4.4절의 리포트 항목을 만든다. 기준선 열이 하나도 없으면 None이다.

    키: columns(영상 표에 있는 기준선 열), nominal_ms, status_counts(상태 → 논리 영상 수), extended(T_b가
    nominal_ms보다 큰 영상), fallback_global(자기 보정이 전역 상수로 대체된 영상), not_completed(보정이 끝나기 전에
    끝난 영상), alert_fallback(alert 영상 기준선 대신 전역 상수를 쓴 영상), alert_fallback_subjects,
    listed(extended, fallback_global, alert_fallback 중 하나라도 해당하는 영상의 표. `flags` 열에 해당 항목).
    열이 없어 알 수 없는 항목은 None이다. 나뉜 영상의 상태는 `/`로 이어져 있으므로 부분 하나라도 해당하면 센다.
    """
    present = [name for name in BASELINE_COLUMNS if name in videos.columns]
    if not present:
        return None
    n = len(videos)
    video_ids = videos["video_id"].astype(str).tolist()
    flags: list[list[str]] = [[] for _ in range(n)]
    overview: dict[str, Any] = {
        "columns": present,
        "nominal_ms": int(nominal_ms),
        "status_counts": None,
        "extended": None,
        "fallback_global": None,
        "not_completed": None,
        "alert_fallback": None,
        "alert_fallback_subjects": None,
    }
    if "t_b_ms" in videos.columns:
        extended = (videos["t_b_ms"].to_numpy(dtype=np.int64) > nominal_ms).tolist()
        overview["extended"] = [video_id for video_id, hit in zip(video_ids, extended) if hit]
        for position, hit in enumerate(extended):
            if hit:
                flags[position].append("extended")
    if "baseline_status" in videos.columns:
        statuses = videos["baseline_status"].astype(str).tolist()
        counts: dict[str, int] = {}
        for status in statuses:
            counts[status] = counts.get(status, 0) + 1
        overview["status_counts"] = dict(sorted(counts.items()))
        parts = [set(status.split("/")) for status in statuses]
        overview["fallback_global"] = [v for v, p in zip(video_ids, parts) if BASELINE_FALLBACK_GLOBAL in p]
        overview["not_completed"] = [v for v, p in zip(video_ids, parts) if BASELINE_NONE in p]
        for position, status_parts in enumerate(parts):
            if BASELINE_FALLBACK_GLOBAL in status_parts:
                flags[position].append(BASELINE_FALLBACK_GLOBAL)
    if "alert_fallback" in videos.columns:
        fallback = videos["alert_fallback"].to_numpy(dtype=bool).tolist()
        overview["alert_fallback"] = [video_id for video_id, hit in zip(video_ids, fallback) if hit]
        overview["alert_fallback_subjects"] = sorted(
            {int(subject) for subject, hit in zip(videos["subject"].tolist(), fallback) if hit}
        )
        for position, hit in enumerate(fallback):
            if hit:
                flags[position].append("alert_fallback")
    keep = [position for position in range(n) if flags[position]]
    listed = videos.iloc[keep][["video_id", "subject", "fold", "class_index"] + present].reset_index(drop=True)
    listed["flags"] = [", ".join(flags[position]) for position in keep]
    overview["listed"] = listed
    return overview


def _baseline_section(
    videos: pd.DataFrame,
    overview: Mapping[str, Any] | None,
    counts: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
) -> list[str]:
    """summary.md의 "기준선과 창 수" 절 (PLAN 4.4절의 마지막 항목)."""
    lines: list[str] = []
    if overview is None:
        lines += [
            "영상 표에 기준선 열(" + ", ".join(BASELINE_COLUMNS) + ")이 없어 기준선이 연장되거나 대체된 영상을 싣지 못했다.",
            "",
        ]
    else:
        seconds = overview["nominal_ms"] / 1000.0
        lines += [
            "영상마다의 값은 `videos.csv`에 있다. `baseline_status`와 `t_b_ms`는 영상 자신의 보정 결과이고, `baseline_used`와 "
            "`alert_fallback`은 영상 표를 만든 정규화 조건의 값이다. 나뉜 영상은 부분의 상태를 `/`로 잇고 T_b는 가장 늦은 값이다.",
            "",
        ]
        if overview["status_counts"] is not None:
            lines.append(
                "- 영상 자신의 기준선 상태별 논리 영상 수: "
                + ", ".join(f"{status} {count}" for status, count in overview["status_counts"].items())
            )
        if overview["extended"] is not None:
            lines.append(f"- 기준선이 연장된 영상(T_b > {seconds:g}초): {len(overview['extended'])}개")
        if overview["fallback_global"] is not None:
            lines.append(
                f"- 자기 보정이 전역 상수로 대체된 영상(`{BASELINE_FALLBACK_GLOBAL}`): {len(overview['fallback_global'])}개"
            )
            lines.append(
                f"- 보정이 끝나기 전에 끝난 영상(`{BASELINE_NONE}`. 창이 없다): {len(overview['not_completed'])}개"
            )
        if overview["alert_fallback"] is not None:
            subjects = overview["alert_fallback_subjects"]
            lines.append(
                "- alert 영상 기준선을 쓸 수 없어 전역 상수를 넣은 영상(`alert_fallback`): "
                f"{len(overview['alert_fallback'])}개, 피험자 {len(subjects)}명"
                + (f" ({', '.join(str(subject) for subject in subjects)})" if subjects else "")
            )
        lines.append("")
        listed = overview["listed"]
        if len(listed) == 0:
            lines += ["기준선이 연장되거나 전역 상수로 대체된 영상이 없다.", ""]
        else:
            lines += ["기준선이 연장되거나 전역 상수로 대체된 영상:", ""]
            columns = list(listed.columns)
            lines += _md_table(columns, [[row[name] for name in columns] for _, row in listed.iterrows()]) + [""]

    lines += ["### 조건별·클래스별 창 수", ""]
    rows: list[list[Any]] = []
    if "n_windows" in videos.columns:
        per_class = [
            int(videos.loc[videos["class_index"].to_numpy() == index, "n_windows"].sum()) for index in range(N_CLASSES)
        ]
        rows.append(["영상 표(평가 대상 영상)", 1] + per_class + [sum(per_class)])
    for label, by_stride in (counts or {}).items():
        for stride, by_class in sorted(by_stride.items()):
            per_class = [int(by_class.get(name, 0)) for name in CLASS_ORDER]
            rows.append([label, stride] + per_class + [sum(per_class)])
    if not rows:
        lines += ["창 수가 전달되지 않았다(`write_reports`의 window_counts).", ""]
        return lines
    if not counts:
        lines += ["조건별 창 수는 전달되지 않았다(`write_reports`의 window_counts). 영상 표의 창 수만 싣는다.", ""]
    else:
        lines += [
            "조건별 행은 그 조건의 스텝 표 전체(모든 fold)에서 센 값이다. 세 정규화 조건은 같은 스텝에서 창을 만들므로 "
            "(PLAN 4.4절) 창 수가 같아야 한다.",
            "",
        ]
    lines += _md_table(["조건", "보폭(스텝)"] + list(CLASS_ORDER) + ["합"], rows) + [""]
    return lines


def write_reports(
    run_dir: Path,
    preds: pd.DataFrame,
    videos: pd.DataFrame,
    fold_subjects: Mapping[int, Iterable[int]],
    rules_cfg: RulesConfig,
    cfg: EvaluationSettings,
    *,
    synthetic: bool,
    description: str,
    command: str,
    exclusions: pd.DataFrame,
    config_path: Path,
    gate: tuple[Sequence[str], Sequence[str]] | None = None,
    paired: Sequence[tuple[Sequence[str], Sequence[str]]] = (),
    window_counts: Mapping[str, Mapping[int, Mapping[str, int]]] | None = None,
    nominal_baseline_ms: int = NOMINAL_BASELINE_MS,
) -> Report:
    """예측 표를 평가해 `run_dir`에 리포트를 쓴다 (PLAN 6장의 "저장", 3.4절, 4.4절).

    입력
    - preds: 예측 표. 여러 묶음, 여러 시드, 여러 fold를 담을 수 있다(`validate_predictions` 통과). 표지 열이 없는
      예전 형식도 받는다.
    - videos: 영상 표(`video_table(quality)`). 영상 단위 지표의 분모다. fold_subjects에 없는 fold의 영상은 뺀다.
      기준선 열(`BASELINE_COLUMNS`)이 있으면 8절에 쓴다.
    - fold_subjects: 바깥 fold 번호 → 그 fold의 시험 피험자. 맨 먼저 `assert_subject_disjoint`로 fold끼리 겹치지 않는지
      검사하고(겹치면 `SubjectOverlapError`), 예측 표와 영상 표의 (subject, fold)가 이 배정과 맞는지도 본다. 자기 fold의
      시험 피험자가 아닌 피험자를 평가한 행이 있으면 `SubjectOverlapError`다(그 피험자는 그 fold의 학습에 쓰였다).
      어떤 묶음이 이 배정의 fold 중 하나에 예측이 없으면 ValueError다(`evaluate_group`).
    - rules_cfg: 졸음 판정기의 설정. cfg: 부트스트랩 횟수와 시드, 임계값 곡선의 값, D11 기준.
    - synthetic, description: 데이터가 합성인지와 한 줄 설명(`DatasetSource`의 값). command: 이 리포트를 만든 명령.
    - exclusions: 제외 목록(`exclusion_list(quality)`. video_id와 reason 열이 있어야 한다).
    - config_path: 이 실행의 실험 설정 파일. `config.yaml`로 복사한다(3.4절: 표마다 명령과 설정 사본을 남긴다).
    - gate: (후보 모델의 묶음 키, 기준 LightGBM의 묶음 키). 주면 D11을 판정해 싣는다(`gate_inputs`, `model_gate`).
      부가 묶음을 넣으면 ValueError다.
    - paired: 짝지은 부트스트랩으로 비교할 묶음 키 쌍들(예: TCN과 GRU). 지표는 `PAIRED_METRICS`다. 한쪽이라도 부가
      묶음인 쌍은 9절에 따로 싣는다(`supplementary_pairs`가 부가 묶음과 그 기준 묶음의 쌍을 만들어 준다).
    - window_counts: 조건 이름 → 보폭 → 클래스 이름 → 창 수(`window_counts()`가 만든다). 8절의 표에 싣는다.
    - nominal_baseline_ms: 기준선의 기본 수집 시간. 영상의 T_b가 이보다 크면 "연장"으로 센다.
    묶음 키는 `(run_id, family, arm)` 또는 `(run_id, family, arm, feature_source, variant)`다(`group_key`).

    부가 묶음(예측 표의 supplementary가 True)
    - 평가는 주 결과와 같은 방식, 같은 영상 표(분모)로 한다. 결과는 summary.md의 9절 "부가 결과"와
      `supplementary_*.csv`, JSON의 `supplementary` 항목에만 싣는다.
    - 1~6절(fold별 표, 혼동 행렬, 오경보, 주 결과끼리의 짝지은 비교, D11, 비교표)과 `per_fold.csv`, `alarm.csv`,
      `alarm_sweep.csv`, JSON의 `groups`·`paired`에는 들어가지 않는다. D11의 후보나 기준으로도 쓸 수 없다.

    쓰는 파일
    - summary.md: 첫 줄이 데이터 종류다(합성이면 `SYNTHETIC_NOTE`, 아니면 `REAL_NOTE`). 머리말에 D 번호가 계획서 1.2절의
      결정 번호라는 한 줄(`PLAN_DECISION_NOTE`)이 있다. 이어서 1 fold별 표와 fold 평균 ±
      표준편차, 시드 표준편차, 부트스트랩 구간, 2 혼동 행렬, 3 오경보 표(model·perclos·either), 4 짝지은 비교, 5 D11,
      6 비교표(문헌 값은 "문헌 보고값"으로 구분. 이 저장소의 행은 실제 데이터일 때만 싣는다), 7 제외 목록,
      8 기준선과 창 수(기준선이 연장되거나 전역 상수로 대체된 영상, alert 영상 기준선을 못 쓴 영상, 조건별·클래스별
      창 수. PLAN 4.4절), 9 부가 결과.
    - per_fold.csv, per_fold.json: (묶음, fold)별 지표(시드 평균)와 시드별 값, fold별 혼동 행렬.
    - summary.json: 묶음별 요약 표와 혼동 행렬, 짝지은 비교, D11, 문헌 행, 기준선 요약(`baselines`), 창 수
      (`window_counts`), 부가 결과(`supplementary`: groups, paired).
    - alarm.csv, alarm_sweep.csv: 오경보 표와 임계값 곡선. videos.csv: 영상 목록(기준선 열 포함). exclusions.csv: 제외 목록.
    - supplementary_per_fold.csv, supplementary_alarm.csv, supplementary_alarm_sweep.csv: 부가 묶음의 같은 표.
      부가 묶음이 없으면 머리글만 있다(같은 폴더에 다시 쓸 때 예전 실행의 부가 표가 남지 않게 항상 쓴다).
    - rules.json(규칙 설정 사본. 그 해시가 rules_hash), config.yaml(실험 설정 사본). 이 두 파일은 바이트 사본이어서
      데이터 종류 표시(합성인지 실제인지)가 없다(config.yaml은 설정 파일의 바이트 그대로이고, rules.json은 그 바이트의
      sha256이 rules_hash라 한 글자도 더할 수 없다). 데이터 종류는 같은 폴더의 summary.md 첫 줄이나 JSON의
      `data_note`에서 확인한다.
    합성 데이터의 점수는 파이프라인 점검용이다. 어느 파일도 그것을 성능으로 적지 않고, D11을 만족해도
    `drowsiness_source`는 perclos다.
    가정: "첫 줄에 합성 데이터임을 적는다"를 summary.md뿐 아니라 모든 표 파일에 적용한다. CSV는 첫 줄이
      `# 데이터 종류` 주석이므로 `pd.read_csv(path, skiprows=1)`로 읽고, JSON은 첫 키가 `data_note`다.
    가정: 영상 표에 있더라도 fold_subjects에 없는 fold의 영상은 평가 대상이 아니다(일부 fold만 평가한 리포트).
    가정: 문헌 행의 값과 주석은 PLAN 2.4절의 것이다. 2.4절은 Xu 2025의 86.5%가 몇 클래스 값인지 적지 않았다.
      조사 기록(영상 단위 3클래스 정확도)에 따라 3클래스 열에 둔다.
    가정: 부가 묶음도 주 결과의 영상 표를 분모와 품질 판정(gate_pass)으로 쓴다. 다른 추출(15 fps 모사)에서 창이
      나오지 않은 영상은 그 묶음에서 오답이다(D7을 그대로 적용).
    가정: "기준선이 연장된 영상"은 T_b가 기본 수집 시간(60초)보다 큰 영상이다. 전역 상수로 대체된 영상과 alert 영상
      기준선을 쓰지 못한 영상(alert_fallback)도 같은 목록에 싣는다(계획서는 연장된 영상만 적었다).
    """
    run_dir = Path(run_dir)
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"config file to copy into the report does not exist: {config_path}")
    preds = validate_predictions(preds)
    _require_columns(videos, VIDEO_COLUMNS, "video table")
    _require_columns(exclusions, ("video_id", "reason"), "exclusion list")
    folds = _check_fold_assignment(fold_subjects, preds, videos)
    videos = videos[videos["fold"].isin(list(folds))].reset_index(drop=True)
    if len(preds) == 0:
        raise ValueError("the prediction table is empty")

    # 요청한 키는 평가(부트스트랩)를 시작하기 전에 검사한다
    groups = preds[list(GROUP_KEYS) + ["supplementary"]].drop_duplicates()
    is_extra = {_key_of(row): bool(row["supplementary"]) for row in groups.to_dict(orient="records")}
    pairs = [(_canonical_key(key_a), _canonical_key(key_b)) for key_a, key_b in paired]
    for pair in pairs:
        for key in pair:
            if key not in is_extra:
                raise ValueError(f"paired comparison names a group without predictions: {key}")
    gate_keys: tuple[GroupKey, GroupKey] | None = None
    if gate is not None:
        gate_keys = (_canonical_key(gate[0]), _canonical_key(gate[1]))
        for key in gate_keys:
            if key not in is_extra:
                raise ValueError(f"the D11 gate names a group without predictions: {key}")
            if is_extra[key]:
                raise ValueError(f"the D11 gate must not use a supplementary group: {key}")
    for _, rows in preds.groupby(list(GROUP_KEYS), sort=True):
        _check_fold_coverage(rows, videos, _key_of(rows.iloc[0]))

    evaluated = tuple(
        evaluate_group(rows, videos, rules_cfg, cfg) for _, rows in preds.groupby(list(GROUP_KEYS), sort=True)
    )
    results = tuple(result for result in evaluated if not result.supplementary)
    extras = tuple(result for result in evaluated if result.supplementary)
    by_key = {result.key: result for result in evaluated}
    step_ms = int(rules_cfg.clock.step_ms)
    all_paired = _paired_table(pairs, by_key, step_ms, cfg)
    is_extra_pair = all_paired["supplementary"].to_numpy(dtype=bool)
    paired_table = all_paired[~is_extra_pair].reset_index(drop=True)
    extra_paired = all_paired[is_extra_pair].reset_index(drop=True)
    gate_result: GateResult | None = None
    if gate_keys is not None:
        gate_result = model_gate(gate_inputs(by_key[gate_keys[0]], by_key[gate_keys[1]]), cfg.model_gate)
    source = DROWSY_SOURCE_PERCLOS
    if gate_result is not None and gate_result.passed and not synthetic:
        source = DROWSY_SOURCE_MODEL
    overview = _baseline_overview(videos, nominal_baseline_ms)

    run_dir.mkdir(parents=True, exist_ok=True)
    note = SYNTHETIC_NOTE if synthetic else REAL_NOTE
    hash_value = export_rules_json(rules_cfg, run_dir / RULES_COPY_FILE)
    shutil.copyfile(config_path, run_dir / CONFIG_COPY_FILE)

    _write_csv(_per_fold_frame(results), run_dir / PER_FOLD_CSV_FILE, note)
    _write_csv(_alarm_frame(results), run_dir / ALARM_CSV_FILE, note)
    _write_csv(_sweep_frame(results), run_dir / ALARM_SWEEP_CSV_FILE, note)
    # 부가 묶음의 표는 묶음이 없어도 쓴다(머리글만). 같은 폴더에 다시 쓸 때 예전 실행의 부가 표가 남지 않는다
    _write_csv(_per_fold_frame(extras), run_dir / (SUPPLEMENTARY_PREFIX + PER_FOLD_CSV_FILE), note)
    _write_csv(_alarm_frame(extras), run_dir / (SUPPLEMENTARY_PREFIX + ALARM_CSV_FILE), note)
    _write_csv(_sweep_frame(extras), run_dir / (SUPPLEMENTARY_PREFIX + ALARM_SWEEP_CSV_FILE), note)
    _write_csv(videos, run_dir / VIDEOS_CSV_FILE, note)
    _write_csv(exclusions, run_dir / EXCLUSIONS_CSV_FILE, note)

    header = {
        "data_note": note,
        "synthetic": synthetic,
        "description": description,
        "command": command,
        "rules_hash": hash_value,
    }
    _write_json(
        {
            **header,
            "groups": [_fold_records(result) for result in results],
            "supplementary_groups": [_fold_records(result) for result in extras],
        },
        run_dir / PER_FOLD_JSON_FILE,
    )
    baselines: dict[str, Any] | None = None
    if overview is not None:
        baselines = {name: value for name, value in overview.items() if name != "listed"}
        baselines["listed"] = overview["listed"].to_dict(orient="records")
    _write_json(
        {
            **header,
            "fold_subjects": folds,
            "bootstrap": {"n": cfg.bootstrap_n, "seed": cfg.bootstrap_seed, "level": CONFIDENCE_LEVEL},
            "groups": [_summary_record(result) for result in results],
            "paired": paired_table.to_dict(orient="records"),
            "gate": None
            if gate_result is None or gate_keys is None
            else {
                "candidate": gate_keys[0],
                "reference": gate_keys[1],
                "conditions": [dataclasses.asdict(condition) for condition in gate_result.conditions],
                "passed": gate_result.passed,
            },
            "drowsiness_source": source,
            "literature_label": LITERATURE_LABEL,
            "literature": [dataclasses.asdict(row) for row in LITERATURE],
            "baselines": baselines,
            "window_counts": None
            if window_counts is None
            else {
                str(label): {str(stride): dict(by_class) for stride, by_class in by_stride.items()}
                for label, by_stride in window_counts.items()
            },
            "supplementary": {
                "groups": [_summary_record(result) for result in extras],
                "paired": extra_paired.to_dict(orient="records"),
            },
        },
        run_dir / SUMMARY_JSON_FILE,
    )

    lines = [f"# {note}", "", description, ""]
    if synthetic:
        lines += ["합성 데이터로 만든 리포트다. 아래 숫자는 파이프라인 점검용이며 성능 수치가 아니다.", ""]
    else:
        lines += ["아래 숫자는 같은 절차의 교차검증 추정치이며, 내보낸 모델 파일 자체를 시험한 값이 아니다 (D10).", ""]
    lines += [PLAN_DECISION_NOTE, ""]
    counts = [int((videos["class_index"] == index).sum()) for index in range(N_CLASSES)]
    lines += [
        f"- 생성 명령: `{command}`",
        f"- 규칙 설정: `{RULES_COPY_FILE}` (rules_hash `{hash_value}`), 실험 설정 사본: `{CONFIG_COPY_FILE}`",
        f"- 평가 대상: fold {', '.join(str(fold) for fold in sorted(folds))}, 피험자 "
        f"{videos['subject'].nunique()}명, 논리 영상 {len(videos)}개 ("
        + ", ".join(f"{name} {count}" for name, count in zip(CLASS_ORDER, counts))
        + f"). D7 품질 기준을 통과한 영상 {int(videos['gate_pass'].sum())}개",
        "- 피험자 분리 검사(`assert_subject_disjoint`) 통과. fold별 시험 피험자 수: "
        + ", ".join(f"fold {fold} {len(folds[fold])}명" for fold in sorted(folds)),
        f"- 부트스트랩: 피험자 단위 복원 추출 {cfg.bootstrap_n}회(시드 {cfg.bootstrap_seed}), 95% 백분위 구간. "
        "구간은 fold를 합쳐 계산한 값의 것이다",
        "- fold별 값은 시드 평균이다. 표준편차는 표본 표준편차이고 값이 하나뿐이면 n/a다",
    ]
    if extras:
        lines.append(
            f"- 부가 묶음 {len(extras)}개는 9절 \"부가 결과\"에만 싣는다. 1~6절과 D11 판정에는 쓰지 않는다"
        )
    no_main = ["주 결과 묶음이 없다. 이 리포트의 묶음은 모두 부가 결과다(9절).", ""]
    lines += ["", "## 1. fold별 결과", ""]
    for result in results:
        lines += _fold_section(result)
    if not results:
        lines += no_main
    lines += ["## 2. 혼동 행렬 (fold 합침, 시드 평균)", ""]
    for result in results:
        lines += _confusion_section(result)
    if not results:
        lines += no_main
    lines += [
        "## 3. 오경보",
        "",
        "판정기는 규칙 엔진과 같은 코드다(`drowsy.rules.StepHysteresis`). model은 묶음의 예측 확률, perclos는 같은 창의 "
        "perclos_60, either는 둘 중 하나라도 판정 중인 상태를 쓴다. 판정은 한번 시작하면 이어지므로 횟수와 함께 영상 수와 "
        "시간 비율을 본다. 대괄호는 피험자 부트스트랩 95% 구간이다.",
        "",
    ]
    for result in results:
        lines += _alarm_section(result)
    if not results:
        lines += no_main
    lines += ["## 4. 짝지은 비교", ""]
    if len(paired_table) == 0:
        lines += ["요청된 비교가 없다.", ""]
    else:
        lines += ["차이의 구간이 0을 벗어날 때만 차이가 있다고 쓴다 (값은 fold를 합친 값의 시드 평균).", ""]
        lines += _paired_lines(paired_table) + [""]
    lines += ["## 5. D11 판정", ""]
    if gate_result is None or gate_keys is None:
        lines += ["요청되지 않았다. `drowsiness.source`는 perclos로 둔다.", ""]
    else:
        lines += [
            f"후보: {_group_name(by_key[gate_keys[0]])}. 기준 LightGBM: {_group_name(by_key[gate_keys[1]])}. "
            "분류 지표는 fold 평균, 오경보 수치는 fold를 합친 값이다.",
            "",
        ]
        lines += _md_table(
            ["조건", "값", "기준", "만족"],
            [
                [
                    f"`{condition.name}`",
                    _fmt(condition.value),
                    f"{condition.comparison} {_fmt(condition.threshold)}",
                    "예" if condition.passed else "아니오",
                ]
                for condition in gate_result.conditions
            ],
        ) + [""]
        if synthetic:
            lines += ["합성 데이터의 수치이므로 이 판정은 쓰지 않는다. `drowsiness.source`는 perclos로 둔다.", ""]
        elif gate_result.passed:
            lines += ["다섯 조건을 모두 만족한다. `drowsiness.source`를 model로 둘 수 있다.", ""]
        else:
            lines += [
                "만족하지 못한 조건이 있다. ONNX는 계약대로 내보내되 \"실험용\"으로 적고 `drowsiness.source`는 perclos로 둔다.",
                "",
            ]
    lines += ["## 6. 비교표", ""]
    comparison: list[list[str]] = []
    if synthetic:
        # 합성 데이터의 점수를 문헌 값 옆에 두면 성능처럼 읽힌다. 이 저장소의 행은 실제 데이터일 때만 싣는다.
        lines += ["합성 데이터 리포트이므로 이 저장소의 결과는 이 표에 싣지 않는다(성능 수치가 아니다).", ""]
    else:
        comparison += [
            [
                _group_name(result),
                "이 저장소, 교차검증 추정치",
                f"{_fmt_percent(result.value('video_accuracy', 'fold_mean'))} ± "
                f"{_fmt_percent(result.value('video_accuracy', 'fold_std'))}",
                f"{_fmt_percent(result.value('two_class_video_accuracy', 'fold_mean'))} ± "
                f"{_fmt_percent(result.value('two_class_video_accuracy', 'fold_std'))}",
                f"fold 평균 ± 표준편차. 분모는 논리 영상 {result.n_videos}개 전체(D7), 2클래스는 D8. 재현율 "
                + " / ".join(_fmt(result.value(f"video_recall_{name}", "fold_mean"), 2) for name in CLASS_ORDER),
            ]
            for result in results
        ]
    comparison += [[row.method, LITERATURE_LABEL, row.three_class, row.two_class, row.note] for row in LITERATURE]
    lines += _md_table(["방법", "출처", "3클래스 영상 정확도", "2클래스 영상 정확도", "조건과 주의"], comparison)
    lines += [
        "",
        "문헌 값은 각 논문이 자기 조건에서 보고한 것이다. 평가한 영상, 분할, 판정 방식이 달라 이 저장소의 결과와 같은 조건의 "
        "비교가 아니다.",
        "",
        "## 7. 제외 목록",
        "",
    ]
    if len(exclusions) == 0:
        lines += ["제외하거나 따로 표시한 영상이 없다.", ""]
    else:
        lines += [
            "`no_window`는 창이 없어 주 결과에서 오답으로 센 영상, `quality_gate`는 보조 행(품질 기준 통과 영상만)에서 뺀 영상이다.",
            "",
        ]
        columns = list(exclusions.columns)
        lines += _md_table(columns, [[row[name] for name in columns] for _, row in exclusions.iterrows()]) + [""]

    lines += ["## 8. 기준선과 창 수 (PLAN 4.4절)", ""]
    lines += _baseline_section(videos, overview, window_counts)
    lines += ["## 9. 부가 결과", ""]
    if not extras:
        lines += ["부가 묶음이 없다.", ""]
    else:
        lines += [
            "부가 분석(PLAN 11장 E3, D16)의 묶음이다. 주 결과(1~6절), 비교표, D11 판정에는 쓰지 않는다. 평가 방식과 분모는 "
            "주 결과와 같다(같은 영상 표. 창이 없는 영상은 오답). 표의 \"(주)\"와 \"(주 수치)\"는 지표의 종류를 가리키는 "
            "말이고 주 결과라는 뜻이 아니다.",
            "",
        ]
        for result in extras:
            lines += _fold_section(result)
            lines += ["오경보 (fold 합침, 시드 평균)", ""]
            lines += _alarm_section(result, heading=False)
        lines += ["### 부가 묶음의 짝지은 비교", ""]
        if len(extra_paired) == 0:
            lines += ["요청된 비교가 없다.", ""]
        else:
            lines += [
                "a는 부가 묶음, b는 그 기준 묶음이다(요청한 쌍 그대로). 차이의 구간이 0을 벗어날 때만 차이가 있다고 쓴다.",
                "",
            ]
            lines += _paired_lines(extra_paired) + [""]

    summary_path = run_dir / SUMMARY_FILE
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return Report(
        run_dir=run_dir,
        summary_path=summary_path,
        results=results,
        paired=paired_table,
        gate=gate_result,
        drowsiness_source=source,
        supplementary=extras,
        supplementary_paired=extra_paired,
    )


# ---------------------------------------------------------------------------
# 라벨이 없는 영상의 시간당 경고 수 (PLAN 6장, 11장 M4)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "config/default.yaml"
DEFAULT_RUN_NAME = "nested_cv"  # `train.DEFAULT_CV_RUN_NAME`과 같은 값(최종 모델의 리포트와 같은 실행 폴더에 쓴다)
UNLABELED_SUBDIR = "unlabeled"
UNLABELED_ALERTS_FILE = "alerts.csv"
UNLABELED_NOTE = "UNLABELED videos (no ground truth): the numbers below are alert counts, not accuracy"
STATUS_OK = "ok"
STATUS_FAILED = "failed"
ALL_EVENTS = "all"  # alerts.csv의 event_type: 모든 종류의 합
TOTAL_ROW = "TOTAL"  # alerts.csv의 video_id: 재생한 영상 전체의 합
UNLABELED_COLUMNS: tuple[str, ...] = (
    "video_id",
    "status",
    "detail",
    "synthetic",
    "n_frames",
    "duration_h",
    "analysable_h",
    "face_ratio",
    "baseline_status",
    "t_b_ms",
    "model_outputs",
    "event_type",
    "state_onsets",
    "state_ends",
    "alerts",
    "immediate_alerts",
    "deferred_alerts",
    "repeat_alerts",
    "alerts_per_hour",
)
# alerts.csv의 횟수 열(이 순서). alerts = immediate_alerts + deferred_alerts + repeat_alerts
UNLABELED_COUNT_COLUMNS: tuple[str, ...] = (
    "state_onsets",
    "state_ends",
    "alerts",
    "immediate_alerts",
    "deferred_alerts",
    "repeat_alerts",
)


class UnlabeledError(ValueError):
    """라벨 없는 영상의 재생을 시작할 수 없을 때(추출 결과 없음, 모델 메타데이터 불일치, 잘못된 속도 등)."""


@dataclass(frozen=True, slots=True)
class UnlabeledVideo:
    """라벨 없는 영상 하나를 재생한 결과 (`replay_unlabeled`).

    - status: `ok` 또는 `failed`(읽을 수 없거나 다른 MediaPipe 모델로 추출한 파일 등. 이유는 detail). 실패한 영상의
      수치는 모두 0이고 합계에 들어가지 않는다.
    - synthetic: 추출 결과에 합성 표시가 있는지.
    - n_frames, duration_ms: 파이프라인이 받아들인 프레임 수와, 그 첫 프레임과 마지막 프레임의 t_ms 차.
    - analysable_ms: 분석 가능한 시간(`analysable_ms()`의 정의).
    - baseline_status, t_b_ms: 영상 자신의 보정 결과와 완성 시각(세션 시간). 끝나지 않았으면 `none`과 −1.
    - model_outputs: 모델 출력 수(다 찬 스텝 120개가 이어진 뒤 매초 하나).
    - state_onsets, state_ends: 이벤트 종류 → 규칙 엔진이 낸 `onset`, `end` 상태 이벤트 수.
    - alerts: 이벤트 종류 → `AlertPolicy`가 실제로 낸 경고 수. 아래 세 종류의 합이다(`demo.AlertKinds`가 가린다).
    - immediate_alerts: 그중 상태가 시작된 프레임에서 바로 나간 경고 수.
    - deferred_alerts: 그중 시작할 때는 재경고 대기나 속도 억제에 막혔다가, 상태가 이어지는 동안 막힘이 풀려 나중에
      나간 첫 경고 수(2026-11-13 소유자 결정 2). 속도가 세션 내내 같으면 재경고 대기가 풀린 경우뿐이다.
    - repeat_alerts: 그중 같은 상태에서 이미 경고가 나간 뒤의 되풀이 경고 수(기본 설정에서는 sleep의 3초 주기).
    """

    video_id: str
    path: Path
    status: str
    detail: str
    synthetic: bool
    n_frames: int
    duration_ms: int
    analysable_ms: int
    baseline_status: str
    t_b_ms: int
    model_outputs: int
    state_onsets: dict[str, int]
    state_ends: dict[str, int]
    alerts: dict[str, int]
    immediate_alerts: dict[str, int]
    deferred_alerts: dict[str, int]
    repeat_alerts: dict[str, int]


@dataclass(frozen=True, slots=True, eq=False)
class UnlabeledReport:
    """`run_unlabeled`의 결과. table은 alerts.csv에 쓴 표(`UNLABELED_COLUMNS`), note는 두 파일의 첫 줄이다.

    rotation은 추출 결과의 회전 점검 기록을 모은 것이다(`dataset.read_rotation_checks`. 확인이 필요한 영상이 들어 있다).
    """

    run_dir: Path
    summary_path: Path
    alerts_path: Path
    videos: tuple[UnlabeledVideo, ...]
    table: pd.DataFrame
    note: str
    rotation: RotationSummary


def analysable_ms(t_ms: Sequence[int], face: Sequence[bool], max_hold_ms: int) -> int:
    """분석 가능한 시간(ms): 얼굴이 검출된 프레임이 대표한 시간의 합.

    프레임 하나는 다음 프레임까지, 최대 max_hold_ms(`clock.max_hold_ms`, 200 ms) 동안을 대표한다(PLAN 3.3절). 마지막
    프레임은 뒤 프레임이 없어 시간을 대표하지 않는다. `dataset.ReplayResult.face_ms`(D7의 얼굴 검출 비율의 분자)와
    같은 정의다. t_ms와 face는 파이프라인이 받아들인 프레임의 값이고 길이가 같아야 한다.
    가정: 계획서는 "시간당 경고 수"의 분모를 정하지 않았다. 얼굴이 없거나 프레임이 끊긴 시간에는 감김·시선·졸음
      규칙이 판정할 것이 없으므로 그 시간을 분모에서 뺀다. 기준선 수집 구간은 넣는다(임시 closure로 감김 규칙이
      동작하고 경고도 나간다. PLAN 1.2절의 결정 D4). 영상 길이 전체를 분모로 한 값이 필요하면 alerts.csv의
      duration_h로 다시 나눈다.
    """
    if len(t_ms) != len(face):
        raise ValueError("t_ms and face must have the same length")
    total = 0
    for index in range(len(t_ms) - 1):
        if face[index]:
            gap = t_ms[index + 1] - t_ms[index]
            total += gap if gap < max_hold_ms else max_hold_ms
    return total


def _zero_counts() -> dict[str, int]:
    return dict.fromkeys(EVENT_TYPES, 0)


def _failed_video(video_id: str, path: Path, detail: str, synthetic: bool = False) -> UnlabeledVideo:
    return UnlabeledVideo(
        video_id=video_id,
        path=path,
        status=STATUS_FAILED,
        detail=detail,
        synthetic=synthetic,
        n_frames=0,
        duration_ms=0,
        analysable_ms=0,
        baseline_status=BASELINE_NONE,
        t_b_ms=-1,
        model_outputs=0,
        state_onsets=_zero_counts(),
        state_ends=_zero_counts(),
        alerts=_zero_counts(),
        immediate_alerts=_zero_counts(),
        deferred_alerts=_zero_counts(),
        repeat_alerts=_zero_counts(),
    )


def replay_unlabeled(
    raw_parquet: Path,
    video_id: str,
    feature_spec: FeatureSpec,
    rules: RulesConfig,
    model_fn: Callable[[Sequence[Sequence[float]]], Sequence[float]] | None,
    speed_kmh: float | Callable[[int], float | None] | None,
) -> UnlabeledVideo:
    """추출 결과 하나를 `DriverPipeline`에 재생해 상태 이벤트와 경고를 센다.

    파이프라인을 여기서 다시 만들지 않는다. `demo.run_pipeline`(기준선을 넣지 않은 `DriverPipeline`: 앱처럼 영상
    자신의 앞부분으로 보정한다)을 그대로 부르고, 돌려받은 이벤트와 경고를 종류별로 센다. 경고는 `demo.run_pipeline`이
    붙인 종류(바로 낸 것, 미뤘다 낸 것, 되풀이)대로 나눠 센다.
    - feature_spec: `load_feature_spec`의 결과. model_fn: `demo.make_model_fn`의 모델 함수(None이면 모델 없이 돈다).
    - speed_kmh: 세션 내내 같은 차량 속도(None이면 속도 억제가 없다). 프레임의 `t_ms`를 받아 속도를 돌려주는 함수도
      받는다(`demo.SpeedFn`. `run_unlabeled`와 명령은 상수만 넘긴다).
    읽을 수 없는 파일, 추출 메타데이터가 없는 parquet, feature_spec과 다른 MediaPipe 모델로 추출한 파일(데모와 같은
    검사), 프레임이 없는 파일은 예외를 내지 않고 `status = failed`인 결과로 돌려준다(이유는 detail).
    가정: 영상 하나의 실패로 전체 실행을 멈추지 않는다. 실패한 파일은 리포트에 이유와 함께 적고 합계에서 뺀다.
    """
    from drowsy import demo  # 지연 import: matplotlib, PyAV, onnxruntime을 이 모듈의 import에 끌어오지 않는다

    raw_parquet = Path(raw_parquet)
    spec = feature_spec.pipeline
    synthetic = False
    try:
        metadata = extract.read_extract_metadata(raw_parquet)
        synthetic = metadata.get("synthetic") is True
        expected_sha = spec.mediapipe.model_sha256
        recorded_sha = metadata.get("model_sha256")
        if recorded_sha is not None and recorded_sha != expected_sha:  # `demo.run_demo_raw`와 같은 검사
            raise UnlabeledError(
                f"extracted with MediaPipe model {recorded_sha} but the feature spec expects {expected_sha}; "
                "extract the video again with the expected face_landmarker.task"
            )
        run = demo.run_pipeline(extract.read_raw_frames(raw_parquet), spec, rules, model_fn, speed_kmh)
    except (demo.DemoError, extract.ExtractError, ValueError, OSError) as exc:
        # pyarrow의 형식 오류는 ValueError, 입출력 오류는 OSError다
        return _failed_video(video_id, raw_parquet, f"{type(exc).__name__}: {exc}", synthetic)

    state_onsets, state_ends = _zero_counts(), _zero_counts()
    for record in run.events:
        if record.event.phase == PHASE_ONSET:
            state_onsets[record.event.type] += 1
        elif record.event.phase == PHASE_END:
            state_ends[record.event.type] += 1
    alerts = _zero_counts()
    by_kind = {kind: _zero_counts() for kind in demo.ALERT_KINDS}
    for alert in run.alerts:
        alerts[alert.event.type] += 1
        by_kind[alert.kind][alert.event.type] += 1
    return UnlabeledVideo(
        video_id=video_id,
        path=raw_parquet,
        status=STATUS_OK,
        detail="",
        synthetic=synthetic,
        n_frames=len(run.t_ms),
        duration_ms=run.duration_ms,
        analysable_ms=analysable_ms(run.t_ms, run.face, spec.clock.max_hold_ms),
        baseline_status=BASELINE_NONE if run.baseline is None else run.baseline.status,
        t_b_ms=-1 if run.baseline is None else run.baseline.t_complete_ms,
        model_outputs=len(run.probs),
        state_onsets=state_onsets,
        state_ends=state_ends,
        alerts=alerts,
        immediate_alerts=by_kind[demo.ALERT_IMMEDIATE],
        deferred_alerts=by_kind[demo.ALERT_DEFERRED],
        repeat_alerts=by_kind[demo.ALERT_REPEAT],
    )


def unlabeled_table(videos: Sequence[UnlabeledVideo]) -> pd.DataFrame:
    """영상별·이벤트 종류별 표(`UNLABELED_COLUMNS`). alerts.csv의 내용이다.

    - 재생한 영상마다 이벤트 종류 7개(`EVENT_TYPES`)의 행과, 모든 종류를 합친 행(event_type = `all`).
    - 실패한 영상은 `all` 행 하나(status = `failed`, 수치 0).
    - 마지막에 재생한 영상 전체의 합(video_id = `TOTAL`): 시간과 횟수를 더한 뒤 나눈 값이다(영상별 값의 평균이 아니다).
    횟수 열은 `UNLABELED_COUNT_COLUMNS`다. alerts는 경고 전체이고 immediate_alerts(바로 낸 것), deferred_alerts(미뤘다 낸
    것), repeat_alerts(되풀이)의 합이다(뜻은 `UnlabeledVideo`).
    alerts_per_hour = alerts / analysable_h. 분석 가능한 시간이 0이면 0.0이다(이 모듈의 공통 규칙).
    """
    rows: list[dict[str, Any]] = []

    def add(
        video_id: str,
        status: str,
        detail: str,
        synthetic: bool,
        n_frames: int,
        duration: int,
        analysable: int,
        baseline_status: str,
        t_b_ms: int,
        model_outputs: int,
        counts: Sequence[tuple[str, Mapping[str, int]]],
    ) -> None:
        hours = analysable / MS_PER_HOUR
        for event_type, numbers in counts:
            rows.append(
                {
                    "video_id": video_id,
                    "status": status,
                    "detail": detail,
                    "synthetic": synthetic,
                    "n_frames": n_frames,
                    "duration_h": duration / MS_PER_HOUR,
                    "analysable_h": hours,
                    "face_ratio": _ratio(analysable, duration),
                    "baseline_status": baseline_status,
                    "t_b_ms": t_b_ms,
                    "model_outputs": model_outputs,
                    "event_type": event_type,
                    **{name: numbers[name] for name in UNLABELED_COUNT_COLUMNS},
                    "alerts_per_hour": _ratio(numbers["alerts"], hours),
                }
            )

    def per_type(by_column: Mapping[str, Mapping[str, int]]) -> list[tuple[str, dict[str, int]]]:
        """횟수 열 이름 → (이벤트 종류 → 횟수)를 이벤트 종류별 행과 모든 종류의 합 행으로 바꾼다."""
        counts = [
            (event_type, {name: by_column[name][event_type] for name in UNLABELED_COUNT_COLUMNS})
            for event_type in EVENT_TYPES
        ]
        total = {name: sum(numbers[name] for _, numbers in counts) for name in UNLABELED_COUNT_COLUMNS}
        return counts + [(ALL_EVENTS, total)]

    done = [video for video in videos if video.status == STATUS_OK]
    for video in videos:
        if video.status != STATUS_OK:  # 실패한 영상: 수치를 모두 0으로 적은 행 하나
            add(video.video_id, video.status, video.detail, video.synthetic, 0, 0, 0, BASELINE_NONE, -1, 0,
                [(ALL_EVENTS, dict.fromkeys(UNLABELED_COUNT_COLUMNS, 0))])  # fmt: skip
            continue
        add(
            video.video_id,
            video.status,
            video.detail,
            video.synthetic,
            video.n_frames,
            video.duration_ms,
            video.analysable_ms,
            video.baseline_status,
            video.t_b_ms,
            video.model_outputs,
            per_type({name: getattr(video, name) for name in UNLABELED_COUNT_COLUMNS}),
        )
    totals = {
        name: {event_type: sum(getattr(video, name)[event_type] for video in done) for event_type in EVENT_TYPES}
        for name in UNLABELED_COUNT_COLUMNS
    }
    add(
        TOTAL_ROW,
        STATUS_OK if done else STATUS_FAILED,
        f"{len(done)} of {len(videos)} videos replayed",
        any(video.synthetic for video in done),
        sum(video.n_frames for video in done),
        sum(video.duration_ms for video in done),
        sum(video.analysable_ms for video in done),
        "",
        -1,
        sum(video.model_outputs for video in done),
        per_type(totals),
    )
    return pd.DataFrame(rows, columns=list(UNLABELED_COLUMNS))


def _unlabeled_summary(
    table: pd.DataFrame,
    videos: Sequence[UnlabeledVideo],
    note: str,
    facts: Sequence[str],
    rotation: RotationSummary,
) -> list[str]:
    """라벨 없는 영상 리포트의 summary.md 줄들. facts는 머리말 아래의 항목(명령, 모델, 해시, 가정)이다.

    머리말의 설명 문단 다음 줄이 `PLAN_DECISION_NOTE`다(항목의 "PLAN 1.2절 결정 D3"이 무엇의 번호인지 알린다).

    rotation은 추출 결과의 회전 점검 기록이다. 4절에 구분별 파일 수와 확인이 필요한 영상의 목록을 싣는다.
    """
    done = [video for video in videos if video.status == STATUS_OK]
    failed = [video for video in videos if video.status != STATUS_OK]
    total = table[table["video_id"] == TOTAL_ROW].set_index("event_type")
    lines = [
        f"# {note}",
        "",
        "라벨이 없는 영상의 추출 결과를 앱과 같은 파이프라인(`DriverPipeline`: 영상마다 자기 앞부분으로 보정한 기준선, ONNX 모델 "
        "콜백, 규칙 엔진, 경고 결정)에 재생해 센 경고 수다. 정답이 없으므로 정확도도 오경보율도 아니다. 경고가 맞았는지는 "
        "이 리포트로 알 수 없다.",
        "",
        PLAN_DECISION_NOTE,
        "",
        *[f"- {fact}" for fact in facts],
        "",
        "## 1. 전체",
        "",
        f"재생한 영상 {len(done)}개. 분석 가능한 시간 {_fmt(float(total.loc[ALL_EVENTS, 'analysable_h']), 4)} h "
        f"(영상 길이의 합 {_fmt(float(total.loc[ALL_EVENTS, 'duration_h']), 4)} h).",
        "",
    ]
    lines += _md_table(
        [
            "이벤트",
            "상태 시작(onset) 수",
            "상태 종료(end) 수",
            "경고 수",
            "바로 낸 경고",
            "미뤘다 낸 경고",
            "되풀이 경고",
            "시간당 경고 수",
        ],
        [
            [
                f"`{event_type}`" if event_type != ALL_EVENTS else "전체",
                *(int(total.loc[event_type, name]) for name in UNLABELED_COUNT_COLUMNS),
                _fmt(float(total.loc[event_type, "alerts_per_hour"]), 2),
            ]
            for event_type in (*EVENT_TYPES, ALL_EVENTS)
        ],
    )
    lines += ["", "## 2. 영상별", ""]
    alert_types = [name for name in EVENT_TYPES if int(total.loc[name, "alerts"]) > 0]
    header = ["video_id", "길이 (h)", "분석 가능 (h)", "얼굴 비율", "기준선", "T_b (초)", "모델 출력 수", "상태 시작 수"]
    header += ["경고 수", "시간당 경고 수"] + [f"`{name}` 경고 (시간당)" for name in alert_types]
    rows: list[list[Any]] = []
    for video in done:
        own = table[table["video_id"] == video.video_id].set_index("event_type")
        rows.append(
            [
                video.video_id,
                _fmt(float(own.loc[ALL_EVENTS, "duration_h"]), 4),
                _fmt(float(own.loc[ALL_EVENTS, "analysable_h"]), 4),
                _fmt(float(own.loc[ALL_EVENTS, "face_ratio"])),
                video.baseline_status,
                "-" if video.t_b_ms < 0 else f"{video.t_b_ms / 1000.0:g}",
                video.model_outputs,
                int(own.loc[ALL_EVENTS, "state_onsets"]),
                int(own.loc[ALL_EVENTS, "alerts"]),
                _fmt(float(own.loc[ALL_EVENTS, "alerts_per_hour"]), 2),
            ]
            + [
                f"{int(own.loc[name, 'alerts'])} ({_fmt(float(own.loc[name, 'alerts_per_hour']), 2)})"
                for name in alert_types
            ]
        )
    lines += _md_table(header, rows)
    lines += [
        "",
        "이벤트 종류별 상태 이벤트 수와 경고 수는 `alerts.csv`에 영상마다 있다(첫 줄은 주석이므로 "
        "`pd.read_csv(path, skiprows=1)`로 읽는다).",
        "",
        "## 3. 재생하지 못한 파일",
        "",
    ]
    if not failed:
        lines += ["없다.", ""]
    else:
        lines += ["아래 파일은 합계에 들어가지 않았다.", ""]
        lines += _md_table(["video_id", "파일", "이유"], [[v.video_id, v.path, v.detail] for v in failed]) + [""]
    lines += ["## 4. 회전 점검", ""]
    lines += rotation_check_section(rotation, extract.UNLABELED_OVERRIDES_KEY) + [""]
    return lines


def run_unlabeled(
    cfg: ExperimentConfig,
    *,
    config_path: Path,
    raw_dir: Path | None = None,
    model_path: Path | None = None,
    speed_kmh: float | None = None,
    run_name: str = DEFAULT_RUN_NAME,
    command: str = "",
) -> UnlabeledReport:
    """라벨 없는 영상(팀 수집 차량 영상)의 추출 결과를 모두 재생해 시간당 경고 수 리포트를 쓴다 (PLAN 6장, 11장 M4).

    - config_path: cfg를 읽은 실험 설정 파일. 리포트 폴더에 `config.yaml`로 복사한다(PLAN 3.4절: reports/의 표는
      생성한 명령과 설정 사본을 함께 남긴다).
    - raw_dir: RawFrame parquet 폴더(`python -m drowsy.extract --unlabeled`의 출력). None이면
      `extract.unlabeled_raw_dir(cfg)`. 그 아래의 parquet을 모두(하위 폴더 포함, 경로순) 재생한다.
    - model_path: 내보낸 ONNX 모델. None이면 데모와 같은 순서로 고른다: `demo.model_path`, 그 파일이 아직 없으면
      `paths.export_dir`의 무작위 가중치 모델 `drowsiness_v0.onnx`.
    - speed_kmh: 경고의 속도 억제(PLAN 1.2절의 결정 D3)에 쓰는 차량 속도. None이면 억제하지 않는다.
    - run_name: 리포트 폴더는 `paths.reports_dir / run_name / unlabeled`다. command: 리포트에 남길 명령.
    모델 불러오기와 메타데이터 대조는 데모의 것을 그대로 쓴다(`demo.verify_model`, `demo.make_model_fn`): 모델의
    feature_spec_hash와 pipeline_hash가 spec 파일들과 어긋나면 `UnlabeledError`다. 영상마다의 재생은
    `replay_unlabeled`가 한다(영상 자신의 기준선, 같은 `DriverPipeline`).

    쓰는 파일(앞의 둘은 첫 줄이 "라벨이 없는 영상이고 숫자는 경고 수이지 정확도가 아니다"라는 문장이다)
    - `alerts.csv`: `unlabeled_table`의 표. 첫 줄은 `# ...` 주석이므로 `pd.read_csv(path, skiprows=1)`로 읽는다.
    - `summary.md`: 명령, 입력, 모델 파일·model_version·sha256·trained_on, pipeline_hash, rules_hash, 적용된 졸음
      판정원(`drowsiness.source`), 속도와 분석 가능한 시간에 대한 가정, 전체와 영상별 표(분석 가능한 시간, 이벤트
      종류별 상태 이벤트 수·경고 수·시간당 경고 수), 재생하지 못한 파일, 회전 점검(추출의 회전 자동 점검이 각도를
      가르지 못했는데 수동 회전 보정이 없는 영상의 목록. `dataset.read_rotation_checks`, `rotation_check_section`).
      머리말에 D 번호가 계획서 1.2절의 결정 번호라는 한 줄(`PLAN_DECISION_NOTE`)이 있다.
    - `config.yaml`(실험 설정 사본), `rules.json`(규칙 설정 사본. 그 바이트의 sha256이 rules_hash). `write_reports`와 같다.
    얼굴 프레임은 읽지도 쓰지도 않는다(입력은 추출 결과의 수치뿐이다. PLAN 1.2절의 결정 D13).
    가정: 속도를 주지 않으면(차량 속도 기록이 없는 영상) 속도 억제를 적용하지 않는다. 모든 경고가 속도와 무관하게
      나간 것으로 센다. 리포트에 그렇게 적는다.
    가정: 속도를 주면 세션 내내 같은 값이다(데모와 같다. 시간에 따라 변하는 속도 기록은 받지 않는다).
    가정: "경고 수"는 운전자에게 실제로 나간 경고의 수다. 상태가 시작된 프레임에서 바로 나간 경고, 재경고 대기나 속도
      억제에 막혔다가 상태가 이어지는 동안 나중에 나간 경고(2026-11-13 소유자 결정 2), 상태가 이어지는 동안의 되풀이
      경고(기본 설정에서 sleep의 3초 주기)를 모두 센다. 세 종류의 수는 `immediate_alerts`, `deferred_alerts`,
      `repeat_alerts`에 따로 적는다(셋의 합이 `alerts`다).
    가정: 입력 가운데 합성 표시가 붙은 파일이 하나라도 있으면 두 파일의 첫 줄에 합성 데이터임을 함께 적는다.
    가정: 리포트 폴더의 기본 이름은 교차검증·최종 학습과 같은 `nested_cv`다(같은 모델의 기록을 한 폴더에 모은다).
    가정: 회전이 확인되지 않은 영상도 재생하고 합계에 넣는다(빼지 않는다). 리포트의 머리말과 4절에서 알리기만 한다.
      틀린 각도로 추출된 영상의 경고 수는 믿을 수 없으므로 그 영상을 직접 보고 수동 회전 보정표에 적어야 한다.
    가정: 회전 점검 기록은 `extraction.unlabeled_rotation_overrides`(`python -m drowsy.extract --unlabeled`의 보정표)와
      대조한다. `--raw-dir`로 다른 추출 폴더를 줘도 같은 표를 쓴다(키가 맞지 않으면 보정이 없는 것으로 본다).
    재생할 수 있는 파일이 하나도 없으면 `UnlabeledError`다(아무것도 쓰지 않는다).
    run_name이 폴더 이름 하나가 아니면 ValueError다(`config.check_run_name`).
    """
    from drowsy import demo, export_onnx  # 지연 import (`replay_unlabeled` 참조)

    run_name = check_run_name(run_name)
    if speed_kmh is not None and not (math.isfinite(speed_kmh) and speed_kmh >= 0.0):
        raise UnlabeledError(f"speed must be a finite value >= 0 km/h, got {speed_kmh}")
    config_path = Path(config_path)
    if not config_path.is_file():
        raise UnlabeledError(f"config file to copy into the report does not exist: {config_path}")
    feature_spec = load_feature_spec(cfg.paths.feature_spec)
    spec = feature_spec.pipeline
    rules = load_rules(cfg.paths.rules)
    raw_dir = Path(raw_dir) if raw_dir is not None else Path(extract.unlabeled_raw_dir(cfg))
    if not raw_dir.is_dir():
        raise UnlabeledError(
            f"raw parquet folder {raw_dir} does not exist. Extract the unlabeled videos first "
            "(python -m drowsy.extract --unlabeled) or pass --raw-dir"
        )
    listing = scan_unlabeled(raw_dir)
    not_raw = [path for path in listing["rel_path"].tolist() if not path.lower().endswith(".parquet")]
    if not_raw:
        raise UnlabeledError(
            f"{raw_dir} holds video files, not raw parquet files (first: {not_raw[0]}). "
            "Extract them first: python -m drowsy.extract --unlabeled"
        )
    if TOTAL_ROW in set(listing["video_id"].tolist()):
        raise UnlabeledError(f"a raw parquet file is named {TOTAL_ROW!r}, which alerts.csv reserves for the totals row")

    notes: list[str] = []
    if model_path is None:
        model_path = cfg.demo.model_path
        if not model_path.is_file():
            fallback = cfg.paths.export_dir / demo.FALLBACK_MODEL_NAME
            notes.append(
                f"`demo.model_path`({model_path})가 아직 없어 무작위 가중치 계약 모델 `{fallback}`을 썼다"
            )
            model_path = fallback
    try:
        model = demo.verify_model(Path(model_path), feature_spec)
        model_fn = demo.make_model_fn(model.path, spec.step_features.window_steps, len(spec.step_features.names))
    except demo.DemoError as exc:
        raise UnlabeledError(str(exc)) from exc

    videos: list[UnlabeledVideo] = []
    for video_id, path in zip(listing["video_id"].tolist(), listing["path"].tolist()):
        video = replay_unlabeled(Path(path), str(video_id), feature_spec, rules, model_fn, speed_kmh)
        if video.status != STATUS_OK:
            print(f"warning: {video.video_id} not replayed: {video.detail}", file=sys.stderr)
        videos.append(video)
    done = [video for video in videos if video.status == STATUS_OK]
    if not done:
        raise UnlabeledError(
            f"none of the {len(videos)} raw parquet files below {raw_dir} could be replayed "
            f"(first: {videos[0].video_id}: {videos[0].detail})"
        )

    synthetic = any(video.synthetic for video in done)
    note = f"{SYNTHETIC_NOTE}. {UNLABELED_NOTE}" if synthetic else UNLABELED_NOTE
    table = unlabeled_table(videos)
    rotation = read_rotation_checks(
        zip(listing["rel_path"].tolist(), (Path(path) for path in listing["path"].tolist())),
        cfg.extraction.unlabeled_rotation_overrides,
    )
    run_dir = cfg.paths.reports_dir / run_name / UNLABELED_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    hash_value = export_rules_json(rules, run_dir / RULES_COPY_FILE)
    shutil.copyfile(config_path, run_dir / CONFIG_COPY_FILE)
    drowsy_source = rules.drowsiness.source
    version = model.metadata["model_version"]
    trained_on = model.metadata.get("trained_on", "unknown")
    max_hold_ms = spec.clock.max_hold_ms
    features = spec.step_features
    facts = [
        f"생성 명령: `{command}`",
        f"입력: `{raw_dir}`의 추출 결과 {len(videos)}개 (재생 {len(done)}개, 실패 {len(videos) - len(done)}개)",
        f"모델: `{model.path}`, model_version {version}, sha256 `{model.sha256}`, trained_on: {trained_on}",
        *notes,
        f"feature_spec: `{cfg.paths.feature_spec}`, pipeline_hash `{feature_spec.pipeline_hash}` "
        "(모델 메타데이터의 값과 같음을 확인했다)",
        f"규칙: `{cfg.paths.rules}`, rules_hash `{hash_value}` (사본 `{RULES_COPY_FILE}`). 실험 설정 사본: "
        f"`{CONFIG_COPY_FILE}`",
        f"적용된 졸음 판정원(`drowsiness.source`): {drowsy_source}"
        + (
            ". 모델 출력은 계산하지만 `drowsy` 판정에는 쓰지 않았다"
            if drowsy_source == DROWSY_SOURCE_PERCLOS
            else ". `drowsy` 판정에 모델 출력을 썼다"
        ),
    ]
    if model.metadata.get("trained_on") == export_onnx.TRAINED_ON_NONE:
        facts.append("이 모델은 무작위 가중치다. 모델 출력에는 의미가 없다")
    if rotation.flagged:
        facts.append(
            f"**주의: 회전이 확인되지 않은 영상이 {len(rotation.flagged)}개 있다**(추출의 회전 자동 점검이 각도를 가르지 "
            "못했는데 수동 회전 보정이 없거나, 보정표의 각도가 추출에 적용되지 않았다). 그 영상의 경고 수는 틀린 "
            "각도에서 나왔을 수 있고 아래 합계에 그대로 들어 있다. 목록은 4절에 있다"
        )
    if speed_kmh is None:
        facts.append(
            "차량 속도: 주지 않았다. 가정: 속도 입력이 없으므로 경고의 속도 억제(PLAN 1.2절 결정 D3)를 적용하지 않았다. "
            "아래 경고 수는 속도와 무관하게 낸 경고를 모두 센 값이다"
        )
    else:
        facts.append(
            f"차량 속도: {speed_kmh:g} km/h. 가정: 세션 내내 같은 값으로 두고 경고의 속도 억제(PLAN 1.2절 결정 D3)를 "
            "적용했다"
        )
    facts += [
        f"분석 가능한 시간(가정): 얼굴이 검출된 프레임이 대표한 시간의 합(다음 프레임까지, 최대 {max_hold_ms} ms). "
        "얼굴이 없거나 프레임이 끊긴 시간은 뺀다. 기준선 수집 구간은 들어간다",
        "경고 수는 `AlertPolicy`가 실제로 낸 경고(재경고 대기와 되풀이 경고를 적용한 뒤)이고, 상태 이벤트 수는 규칙 엔진이 "
        "낸 이벤트 수다. 시간당 경고 수 = 경고 수 / 분석 가능한 시간",
        "경고 수 = 바로 낸 경고(상태가 시작된 프레임에서 나간 것) + 미뤘다 낸 경고(시작할 때는 재경고 대기나 속도 억제에 "
        "막혔다가, 상태가 이어지는 동안 막힘이 풀린 첫 프레임에서 나간 그 상태의 첫 경고) + 되풀이 경고(같은 상태에서 이미 "
        "경고가 나간 뒤의 경고)",
        f"`drowsy`는 기준선이 완성되고 조회 구간 {features.lookback_long_ms / 1000.0:g}초가 찬 스텝부터 판정한다(모델 출력은 "
        f"그런 스텝이 {features.window_steps}개 이어진 뒤부터 나온다). 그보다 짧은 영상에서는 `drowsy` 경고가 나올 수 없지만 "
        "그 영상의 시간도 분석 가능한 시간에 들어 있다",
    ]

    alerts_path = run_dir / UNLABELED_ALERTS_FILE
    summary_path = run_dir / SUMMARY_FILE
    _write_csv(table, alerts_path, note)
    summary_path.write_text("\n".join(_unlabeled_summary(table, videos, note, facts, rotation)), encoding="utf-8")
    return UnlabeledReport(
        run_dir=run_dir,
        summary_path=summary_path,
        alerts_path=alerts_path,
        videos=tuple(videos),
        table=table,
        note=note,
        rotation=rotation,
    )


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m drowsy.evaluate`. 종료 코드: 0 리포트를 씀, 2 시작할 수 없음.

    하위 명령 `unlabeled`는 다른 일을 하기 전에 `--run-name`을 검사한다(`config.check_run_name`): 경로 구분자나 `..`이
    든 이름은 리포트를 `paths.reports_dir` 밖에 쓰게 되므로 한 줄 오류와 함께 2로 끝난다.
    """
    parser = argparse.ArgumentParser(
        prog="python -m drowsy.evaluate",
        description="Evaluation commands that do not need labels (PLAN section 11, M4).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="experiment config YAML")
    commands = parser.add_subparsers(dest="command", required=True)
    unlabeled = commands.add_parser(
        "unlabeled",
        help="alerts per hour on unlabeled videos (replays their raw parquet files through DriverPipeline)",
    )
    unlabeled.add_argument("--config", default=argparse.SUPPRESS, help="experiment config YAML")
    unlabeled.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="folder of raw parquet files (default: the unlabeled extraction folder of the config)",
    )
    unlabeled.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "exported ONNX model (default: demo.model_path; while that file does not exist, "
            "export/drowsiness_v0.onnx)"
        ),
    )
    unlabeled.add_argument(
        "--speed-kmh",
        type=float,
        default=None,
        help="constant vehicle speed for the alert speed gates (default: not given, alerts are not suppressed)",
    )
    unlabeled.add_argument(
        "--run-name", default=DEFAULT_RUN_NAME, help="report folder name below paths.reports_dir"
    )
    args = parser.parse_args(argv)
    arguments = sys.argv[1:] if argv is None else list(argv)
    command = " ".join(["python -m drowsy.evaluate", *arguments])

    try:
        check_run_name(args.run_name)
        cfg = load_config(args.config)
        report = run_unlabeled(
            cfg,
            config_path=Path(args.config),
            raw_dir=args.raw_dir,
            model_path=args.model,
            speed_kmh=args.speed_kmh,
            run_name=args.run_name,
            command=command,
        )
    except (ValueError, OSError, extract.ExtractError) as exc:  # SpecError, ManifestError, UnlabeledError도 ValueError다
        print(f"error: {exc}", file=sys.stderr)
        return 2

    total = report.table[report.table["video_id"] == TOTAL_ROW].set_index("event_type").loc[ALL_EVENTS]
    done = sum(video.status == STATUS_OK for video in report.videos)
    print(report.note)
    print(f"wrote {report.alerts_path}")
    print(f"wrote {report.summary_path}")
    print(
        f"videos replayed {done} of {len(report.videos)}, analysable {float(total['analysable_h']):.4f} h, "
        f"alerts {int(total['alerts'])}, alerts per hour {float(total['alerts_per_hour']):.2f}"
    )
    print(rotation_check_line(report.rotation, f"section 4 of {report.summary_path}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
