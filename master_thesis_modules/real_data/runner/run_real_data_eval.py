"""Evaluate legacy real-data inputs with the new risk core."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_thesis_modules.risk_core.engine.batch_risk_engine import BatchRiskEngine
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.features.dataframe_adapter import (
    data_dicts_to_feature_sequences,
    results_to_dataframe,
)
from master_thesis_modules.real_data.export.save_eval_csv import save_eval_csvs
from master_thesis_modules.real_data.loader.trial_loader import load_trial_input
from master_thesis_modules.scenario_sim.runner._outputs import save_evaluation_outputs


def run_real_data_eval(
    input_path: str | Path,
    output: str | Path,
    model: str = "spatial_context",
) -> dict[str, Path]:
    data_dicts = load_trial_input(input_path)
    sequences = data_dicts_to_feature_sequences(data_dicts)
    engine = BatchRiskEngine(RiskEngine(RiskConfig(model_type=model)))
    results = engine.evaluate(sequences)
    evaluated = {
        person_id: results_to_dataframe(data_dicts[person_id], results[person_id])
        for person_id in data_dicts
    }
    save_eval_csvs(evaluated, output)
    return save_evaluation_outputs(output, evaluated, results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="spatial_context")
    args = parser.parse_args()
    paths = run_real_data_eval(args.input, args.output, args.model)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

