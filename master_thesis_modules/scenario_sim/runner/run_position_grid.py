"""Generate a patient/staff position-grid comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D


def run_position_grid(output: str | Path, model: str = "spatial_context") -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    engine = RiskEngine(RiskConfig(model_type=model))
    patient_positions = [Position2D(1, 1), Position2D(3, 3), Position2D(5, 5)]
    staff_positions = [Position2D(1, 2), Position2D(5, 5), Position2D(0, 0)]
    rows = []
    for p_idx, patient_position in enumerate(patient_positions):
        for s_idx, staff_position in enumerate(staff_positions):
            result = engine.evaluate(
                FeatureFrame(
                    person_id=f"P{p_idx}_S{s_idx}",
                    time_s=0.0,
                    is_patient_label="yes",
                    is_patient_confidence=1.0,
                    age_group_label="old",
                    age_confidence=1.0,
                    pose_features=PoseFeatures(0.2, 0.2, 0.8, 0.3),
                    patient_position=patient_position,
                    nearest_handrail_position=Position2D(0.0, patient_position.y),
                    nearest_staff_position=staff_position,
                    nearest_staff_velocity=Velocity2D(-0.2, -0.2),
                )
            )
            rows.append(
                {
                    "patient_position": p_idx,
                    "staff_position": s_idx,
                    "model_type": model,
                    "total_risk": result.total_risk,
                }
            )
    path = output / "position_grid.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="spatial_context")
    args = parser.parse_args()
    print(run_position_grid(args.output, args.model))


if __name__ == "__main__":
    main()

