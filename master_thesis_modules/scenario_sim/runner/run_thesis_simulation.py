"""Run thesis-style multi-timestamp scenario simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_thesis_modules.risk_core.engine.batch_risk_engine import BatchRiskEngine
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig, VALID_MODEL_TYPES
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.features.dataframe_adapter import results_to_dataframe
from master_thesis_modules.scenario_sim.encoder.dataframe_builder import (
    ScenarioDataFrameBuilder,
)
from master_thesis_modules.scenario_sim.encoder.scenario_loader import ScenarioLoader
from master_thesis_modules.scenario_sim.runner._outputs import (
    build_source_dataframes,
    save_evaluation_outputs,
)
from master_thesis_modules.scenario_sim.visualization.plot_notification_log import (
    plot_notification_log,
)
from master_thesis_modules.scenario_sim.visualization.plot_ranking import plot_ranking
from master_thesis_modules.scenario_sim.visualization.plot_risk_timeseries import (
    plot_risk_timeseries,
)


def run_thesis_simulation(
    scenario: str | Path,
    output: str | Path,
    model: str = "spatial_context",
    notification_message_style: str = "current",
) -> dict[str, Path]:
    world_state = ScenarioLoader().load(scenario)
    sequences = ScenarioDataFrameBuilder().build_sequences(world_state)
    source_dataframes = build_source_dataframes(sequences)
    batch_engine = BatchRiskEngine(RiskEngine(RiskConfig(model_type=model)))
    results = batch_engine.evaluate(sequences)
    evaluated_dataframes = {
        person_id: results_to_dataframe(source_dataframes[person_id], results[person_id])
        for person_id in sequences
    }
    paths = save_evaluation_outputs(
        output,
        evaluated_dataframes,
        results,
        staff_count=len(world_state.staff),
        notification_message_style=notification_message_style,
    )
    plot_risk_timeseries(paths["risk_timeseries"], Path(output) / "risk_timeseries.png")
    plot_ranking(paths["ranking"], Path(output) / "ranking.png")
    plot_notification_log(paths["risk_timeseries"], paths["notification_log"], Path(output) / "notification_log.png")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--model",
        choices=sorted(VALID_MODEL_TYPES),
        default="spatial_context",
        help="risk model. spatial_context uses patient and spatial context; patient_context ignores object/staff context",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--notification-message-style",
        choices=["current", "legacy"],
        default="current",
        help="notification_log.csvの通知文面。legacyはnotification_generator_v5.py互換の文面にする",
    )
    args = parser.parse_args()
    paths = run_thesis_simulation(
        args.scenario,
        args.output,
        args.model,
        notification_message_style=args.notification_message_style,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
