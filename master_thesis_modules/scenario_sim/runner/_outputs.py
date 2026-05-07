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

    ranking_df = build_ranking_dataframe(results_by_person)
    ranking_df.to_csv(output_dir / "ranking.csv", index=False)

    notification_history = build_notification_history(results_by_person, staff_count)
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


def build_ranking_dataframe(results_by_person: dict[str, list[RiskResult]]) -> pd.DataFrame:
    by_time: dict[float, list[RiskResult]] = {}
    for results in results_by_person.values():
        for result in results:
            by_time.setdefault(result.time_s, []).append(result)

    rows = []
    for timestamp, results in sorted(by_time.items()):
        ranked = sorted(results, key=lambda result: result.total_risk, reverse=True)
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
) -> NotificationHistory:
    policy = NotificationPolicy()
    history = NotificationHistory()
    by_time: dict[float, list[RiskResult]] = {}
    for results in results_by_person.values():
        for result in results:
            by_time.setdefault(result.time_s, []).append(result)
    for timestamp, results in sorted(by_time.items()):
        history.append_many(policy.evaluate_timestep(timestamp, results, staff_count=staff_count))
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

