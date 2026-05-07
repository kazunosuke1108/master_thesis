"""Run a semantic scenario file through the MVP risk engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.schema.node_labels import NODE_LABELS
from master_thesis_modules.scenario_sim.encoder.feature_encoder import FeatureEncoder
from master_thesis_modules.scenario_sim.encoder.scenario_loader import ScenarioLoader


def run_scenario(path: str | Path) -> str:
    world_state = ScenarioLoader().load(path)
    frames = FeatureEncoder().encode(world_state)
    engine = RiskEngine()
    results = [engine.evaluate(frame) for frame in frames]
    ranked = sorted(results, key=lambda result: result.total_risk, reverse=True)

    lines = [f"Scenario: {world_state.scenario_name}", "Ranking:"]
    for rank, result in enumerate(ranked, start=1):
        lines.append(
            f"{rank}. {result.person_id}: total_risk={result.total_risk:.3f}"
        )
        lines.append("   factors:")
        for node_id, value in sorted(
            result.factor_risks.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            lines.append(f"   - {NODE_LABELS[node_id]}: {value:.3f}")
        lines.append(f"   explanation: {result.explanation}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML/JSON")
    args = parser.parse_args()
    print(run_scenario(args.scenario))


if __name__ == "__main__":
    main()

