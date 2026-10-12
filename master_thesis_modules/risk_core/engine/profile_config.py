"""Create RiskConfig objects from AHP/Fuzzy profile names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from master_thesis_modules.risk_core.aggregators.fuzzy import validate_internal_rule_results
from master_thesis_modules.risk_core.aggregators.legacy_ahp_adapter import (
    load_ahp_weights_from_matrix_csv,
)
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.schema import node_ids as ids


DEFAULT_COMMON_DIR = Path("master_thesis_modules/database/common")
LEGACY_COMMON_DIR = Path("/media/hayashide/MasterThesis/common")
DEFAULT_FUZZY_QUESTIONNAIRE_PATH = Path(
    "master_thesis_modules/scripts_202511/2_中村さんと百武さんのAHPとFuzzy/questionaire_1b.csv"
)


def make_profile_risk_config(
    ahp_profile_name: str | None = None,
    fuzzy_profile_name: str | None = None,
    common_dir: str | Path = DEFAULT_COMMON_DIR,
    model_type: str = "spatial_context",
    action_aggregation: str = "weighted_sum",
) -> RiskConfig:
    common_dir = Path(common_dir)
    base = RiskConfig(
        model_type=model_type,
        action_aggregation=action_aggregation,
        ahp_profile_name=ahp_profile_name,
        fuzzy_profile_name=fuzzy_profile_name,
    )
    action_weights = base.action_weights
    object_weights = base.object_weights

    if ahp_profile_name:
        action_csv = _resolve_common_file(
            common_dir,
            f"comparison_mtx_30000001_{ahp_profile_name}.csv",
        )
        object_csv = _resolve_common_file(
            common_dir,
            f"comparison_mtx_30000010_{ahp_profile_name}.csv",
        )
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
        tfn_csv = _resolve_common_file(common_dir, f"TFN_{fuzzy_profile_name}.csv")
        if tfn_csv.exists():
            fuzzy_rule_results = _fuzzy_rule_results_from_tfn_csv(tfn_csv)
        else:
            fuzzy_rule_results = _fuzzy_rule_results_from_questionnaire(
                fuzzy_profile_name,
                DEFAULT_FUZZY_QUESTIONNAIRE_PATH,
            )
            if ids.INTERNAL_RISK not in fuzzy_rule_results:
                raise ValueError(
                    f"Internal Fuzzy consequents unavailable for profile: {fuzzy_profile_name}"
                )

    return RiskConfig(
        model_type=model_type,
        action_aggregation=action_aggregation,
        ahp_profile_name=ahp_profile_name,
        fuzzy_profile_name=fuzzy_profile_name,
        fuzzy_rule_results=fuzzy_rule_results,
        action_weights=action_weights,
        object_weights=object_weights,
    )


def _resolve_common_file(common_dir: Path, filename: str) -> Path:
    path = common_dir / filename
    if path.exists():
        return path
    legacy_path = LEGACY_COMMON_DIR / filename
    if legacy_path.exists():
        return legacy_path
    return path


def _fuzzy_rule_results_from_tfn_csv(path: Path) -> dict[int, tuple[float, ...]]:
    try:
        data = pd.read_csv(path, header=None)
        if data.shape[1] != 3 or len(data) < 12:
            raise ValueError("TFN CSV requires three columns and at least 12 rows")
        data.columns = ["l", "c", "r"]
        internal_results = validate_internal_rule_results(tuple(data.loc[:3, "c"]))
        # Parse every TFN component; the range restriction applies only to internal c.
        data = data.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Invalid TFN CSV: {path}: {exc}") from exc
    return {
        ids.INTERNAL_RISK: internal_results,
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
    answers = data[profile_name].tolist()
    if len(answers) < 12:
        return {}
    try:
        if any(float(value) not in {1.0, 2.0, 3.0, 4.0, 5.0} for value in answers[:4]):
            raise ValueError("Internal questionnaire answers must be scores from 1 to 5")
        values = [_convert_questionnaire_score(value) for value in answers]
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ValueError(f"Invalid Fuzzy questionnaire profile: {profile_name}: {path}") from exc
    return {
        ids.INTERNAL_RISK: validate_internal_rule_results(tuple(values[:4])),
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
