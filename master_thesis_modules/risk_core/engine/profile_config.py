"""Create RiskConfig objects from AHP/Fuzzy profile names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from master_thesis_modules.risk_core.aggregators.legacy_ahp_adapter import (
    load_ahp_weights_from_matrix_csv,
)
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.schema import node_ids as ids


DEFAULT_COMMON_DIR = Path("master_thesis_modules/database/common")
DEFAULT_FUZZY_QUESTIONNAIRE_PATH = Path(
    "master_thesis_modules/scripts_202511/2_中村さんと百武さんのAHPとFuzzy/questionaire_1b.csv"
)


def make_profile_risk_config(
    ahp_profile_name: str | None = None,
    fuzzy_profile_name: str | None = None,
    common_dir: str | Path = DEFAULT_COMMON_DIR,
    model_type: str = "spatial_context",
) -> RiskConfig:
    common_dir = Path(common_dir)
    base = RiskConfig(
        model_type=model_type,
        ahp_profile_name=ahp_profile_name,
        fuzzy_profile_name=fuzzy_profile_name,
    )
    action_weights = base.action_weights
    object_weights = base.object_weights

    if ahp_profile_name:
        action_csv = common_dir / f"comparison_mtx_30000001_{ahp_profile_name}.csv"
        object_csv = common_dir / f"comparison_mtx_30000010_{ahp_profile_name}.csv"
        if action_csv.exists():
            action_weights = dict(
                zip(ids.ACTION_RISK_NODES, load_ahp_weights_from_matrix_csv(action_csv))
            )
        if object_csv.exists():
            object_weights = dict(
                zip(ids.OBJECT_RISK_NODES, load_ahp_weights_from_matrix_csv(object_csv))
            )

    fuzzy_rule_results = {}
    if fuzzy_profile_name:
        tfn_csv = common_dir / f"TFN_{fuzzy_profile_name}.csv"
        if tfn_csv.exists():
            fuzzy_rule_results = _fuzzy_rule_results_from_tfn_csv(tfn_csv)
        else:
            fuzzy_rule_results = _fuzzy_rule_results_from_questionnaire(
                fuzzy_profile_name,
                DEFAULT_FUZZY_QUESTIONNAIRE_PATH,
            )
            if not fuzzy_rule_results:
                fuzzy_rule_results = _fallback_fuzzy_profile(fuzzy_profile_name)

    return RiskConfig(
        model_type=model_type,
        ahp_profile_name=ahp_profile_name,
        fuzzy_profile_name=fuzzy_profile_name,
        fuzzy_rule_results=fuzzy_rule_results,
        action_weights=action_weights,
        object_weights=object_weights,
    )


def _fuzzy_rule_results_from_tfn_csv(path: Path) -> dict[int, tuple[float, ...]]:
    data = pd.read_csv(path, names=["l", "c", "r"])
    return {
        ids.EXTERNAL_RISK: (
            float(data.loc[5, "c"]),
            float(data.loc[4, "c"]),
            float(data.loc[7, "c"]),
            float(data.loc[6, "c"]),
        ),
        ids.TOTAL_RISK: (
            float(data.loc[8, "c"]),
            float(data.loc[9, "c"]),
            float(data.loc[10, "c"]),
            float(data.loc[11, "c"]),
        ),
    }


def _fallback_fuzzy_profile(profile_name: str) -> dict[int, tuple[float, ...]]:
    """Last-resort fallback when neither TFN nor questionnaire CSVs are available."""

    if profile_name == "中村":
        return {
            ids.EXTERNAL_RISK: (0.75, 0.0, 0.25, 0.0),
            ids.TOTAL_RISK: (1.0, 0.0, 0.75, 0.25),
        }
    if profile_name == "百武":
        return {
            ids.EXTERNAL_RISK: (1.0, 0.75, 0.75, 0.5),
            ids.TOTAL_RISK: (1.0, 0.5, 0.25, 0.0),
        }
    return {}


def _fuzzy_rule_results_from_questionnaire(
    profile_name: str,
    path: Path,
) -> dict[int, tuple[float, ...]]:
    if not path.exists():
        return {}
    data = pd.read_csv(path, index_col="1b")
    if profile_name not in data.columns:
        return {}
    values = [_convert_questionnaire_score(value) for value in data[profile_name].tolist()]
    if len(values) < 12:
        return {}
    return {
        ids.EXTERNAL_RISK: (
            values[5],
            values[4],
            values[7],
            values[6],
        ),
        ids.TOTAL_RISK: (
            values[8],
            values[9],
            values[10],
            values[11],
        ),
    }


def _convert_questionnaire_score(value: int | float) -> float:
    convert_dict = {
        5: 1.0,
        4: 0.75,
        3: 0.5,
        2: 0.25,
        1: 0.0,
    }
    return convert_dict[int(value)]
