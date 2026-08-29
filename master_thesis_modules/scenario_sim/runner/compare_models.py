"""Run multiple risk model configurations for one scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from master_thesis_modules.risk_core.engine.batch_risk_engine import BatchRiskEngine
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig, VALID_MODEL_TYPES
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.scenario_sim.encoder.dataframe_builder import (
    ScenarioDataFrameBuilder,
)
from master_thesis_modules.scenario_sim.encoder.scenario_loader import ScenarioLoader
from master_thesis_modules.scenario_sim.runner._outputs import build_ranking_dataframe


def compare_models(
    scenario: str | Path,
    models: list[str],
    output: str | Path,
) -> dict[str, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    world_state = ScenarioLoader().load(scenario)
    sequences = ScenarioDataFrameBuilder().build_sequences(world_state)

    rows = []
    rankings = {}
    for model in models:
        results = BatchRiskEngine(RiskEngine(RiskConfig(model_type=model))).evaluate(sequences)
        ranking_df = build_ranking_dataframe(results)
        rankings[model] = ranking_df
        for _, row in ranking_df.iterrows():
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "patient_id": row["patient_id"],
                    "model_type": model,
                    "total_risk": row["10000000"],
                    "rank": row["rank"],
                }
            )

    comparison_df = pd.DataFrame(rows)
    comparison_csv = output / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    summary = _ranking_agreement(rankings)
    summary_json = output / "ranking_agreement_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return {"model_comparison": comparison_csv, "ranking_agreement_summary": summary_json}


def _ranking_agreement(rankings: dict[str, pd.DataFrame]) -> dict[str, object]:
    if len(rankings) < 2:
        return {}
    models = list(rankings)
    base = rankings[models[0]]
    summary = {"baseline": models[0], "pairs": []}
    for model in models[1:]:
        merged = base.merge(
            rankings[model],
            on=["timestamp", "patient_id"],
            suffixes=("_base", "_other"),
        )
        summary["pairs"].append(
            {
                "model": model,
                "rank_match_rate": float((merged["rank_base"] == merged["rank_other"]).mean()),
                "top1_match_rate": _top1_match_rate(base, rankings[model]),
            }
        )
    return summary


def _top1_match_rate(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_top = left[left["rank"] == 1][["timestamp", "patient_id"]]
    right_top = right[right["rank"] == 1][["timestamp", "patient_id"]]
    merged = left_top.merge(right_top, on="timestamp", suffixes=("_left", "_right"))
    if merged.empty:
        return 0.0
    return float((merged["patient_id_left"] == merged["patient_id_right"]).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(VALID_MODEL_TYPES),
        default=["action_only", "patient_context", "spatial_context"],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = compare_models(args.scenario, args.models, args.output)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
