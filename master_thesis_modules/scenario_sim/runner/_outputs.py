"""Shared scenario output helpers."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.features.dataframe_adapter import (
    evaluated_dataframes_to_long_dataframe,
)
from master_thesis_modules.risk_core.notification.notification_history import (
    NotificationHistory,
)
from master_thesis_modules.risk_core.notification.notification_policy import (
    NotificationPolicy,
)
from master_thesis_modules.risk_core.schema import node_ids as ids


def build_source_dataframes(sequences) -> dict[str, pd.DataFrame]:
    return {
        person_id: sequence.to_feature_dataframe()
        for person_id, sequence in sequences.items()
    }


def save_evaluation_outputs(
    output_dir: str | Path,
    evaluated_dataframes: dict[str, pd.DataFrame],
    results_by_person: dict[str, list[RiskResult]],
    staff_count: int = 1,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for person_id, dataframe in evaluated_dataframes.items():
        dataframe.to_csv(output_dir / f"data_{person_id}_eval.csv", index=False)

    long_df = evaluated_dataframes_to_long_dataframe(evaluated_dataframes)
    long_df.to_csv(output_dir / "risk_timeseries.csv", index=False)

    rankable_lookup = build_rankable_patient_lookup(evaluated_dataframes)
    ranking_df = build_ranking_dataframe(results_by_person, rankable_lookup)
    ranking_df.to_csv(output_dir / "ranking.csv", index=False)

    notification_history = build_notification_history(
        results_by_person,
        staff_count,
        rankable_lookup,
    )
    notification_history.save_csv(output_dir / "notification_log.csv")

    explanations = build_explanations(results_by_person)
    with (output_dir / "explanations.json").open("w", encoding="utf-8") as handle:
        json.dump(explanations, handle, ensure_ascii=False, indent=2)

    return {
        "risk_timeseries": output_dir / "risk_timeseries.csv",
        "ranking": output_dir / "ranking.csv",
        "notification_log": output_dir / "notification_log.csv",
        "explanations": output_dir / "explanations.json",
    }


def build_ranking_dataframe(
    results_by_person: dict[str, list[RiskResult]],
    rankable_lookup: dict[tuple[str, float], bool] | None = None,
) -> pd.DataFrame:
    by_time: dict[float, list[RiskResult]] = {}
    for results in results_by_person.values():
        for result in results:
            by_time.setdefault(result.time_s, []).append(result)

    rows = []
    for timestamp, results in sorted(by_time.items()):
        eligible_results = [
            result for result in results if _is_rankable_result(result, rankable_lookup)
        ]
        ranked = sorted(eligible_results, key=lambda result: result.total_risk, reverse=True)
        for rank, result in enumerate(ranked, start=1):
            rows.append(
                {
                    "timestamp": timestamp,
                    "patient_id": result.person_id,
                    "rank": rank,
                    str(ids.TOTAL_RISK): result.total_risk,
                }
            )
    return pd.DataFrame(rows)


def build_notification_history(
    results_by_person: dict[str, list[RiskResult]],
    staff_count: int = 1,
    rankable_lookup: dict[tuple[str, float], bool] | None = None,
) -> NotificationHistory:
    policy = NotificationPolicy()
    history = NotificationHistory()
    by_time: dict[float, list[RiskResult]] = {}
    for results in results_by_person.values():
        for result in results:
            by_time.setdefault(result.time_s, []).append(result)
    for timestamp, results in sorted(by_time.items()):
        eligible_results = [
            result for result in results if _is_rankable_result(result, rankable_lookup)
        ]
        history.append_many(
            policy.evaluate_timestep(timestamp, eligible_results, staff_count=staff_count)
        )
    return history


def build_explanations(results_by_person: dict[str, list[RiskResult]]) -> list[dict[str, object]]:
    rows = []
    for person_id, results in results_by_person.items():
        for result in results:
            rows.append(
                {
                    "timestamp": result.time_s,
                    "patient_id": person_id,
                    "total_risk": result.total_risk,
                    "message": result.explanation,
                }
            )
    return rows


def build_rankable_patient_lookup(
    evaluated_dataframes: dict[str, pd.DataFrame],
) -> dict[tuple[str, float], bool]:
    lookup = {}
    for person_id, dataframe in evaluated_dataframes.items():
        for _, row in dataframe.iterrows():
            timestamp = float(row.get("timestamp", 0.0))
            label = _patient_label_from_row(row)
            lookup[(str(person_id), timestamp)] = _is_rankable_patient_label(label)
    return lookup


def _is_rankable_result(
    result: RiskResult,
    rankable_lookup: dict[tuple[str, float], bool] | None,
) -> bool:
    if rankable_lookup is None:
        return True
    return rankable_lookup.get((str(result.person_id), float(result.time_s)), True)


def _patient_label_from_row(row: pd.Series) -> object:
    if ids.IS_PATIENT in row:
        return row.get(ids.IS_PATIENT)
    if str(ids.IS_PATIENT) in row:
        return row.get(str(ids.IS_PATIENT))
    if "is_patient_label" in row:
        return row.get("is_patient_label")
    return "yes"


def _is_rankable_patient_label(label: object) -> bool:
    if label is True:
        return True
    if label is False:
        return False
    text = str(label).strip().lower()
    return text not in {"no", "false", "0", "staff"}
