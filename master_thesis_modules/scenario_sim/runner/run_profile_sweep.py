"""Run scenario simulations while sweeping AHP and Fuzzy profile names."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_thesis_modules.risk_core.engine.batch_risk_engine import BatchRiskEngine
from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config
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
from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    visualize_profile_sweep,
)


def run_profile_sweep(
    scenario: str | Path,
    output: str | Path,
    staff_names: list[str],
    common_dir: str | Path = "master_thesis_modules/database/common",
    model: str = "spatial_context",
) -> list[Path]:
    world_state = ScenarioLoader().load(scenario)
    sequences = ScenarioDataFrameBuilder().build_sequences(world_state)
    source_dataframes = build_source_dataframes(sequences)
    output = Path(output)
    written_dirs = []

    for staff_name_ahp in staff_names:
        for staff_name_fuzzy in staff_names:
            config = make_profile_risk_config(
                ahp_profile_name=staff_name_ahp,
                fuzzy_profile_name=staff_name_fuzzy,
                common_dir=common_dir,
                model_type=model,
            )
            batch_engine = BatchRiskEngine(RiskEngine(config))
            results = batch_engine.evaluate(sequences)
            evaluated_dataframes = {
                person_id: results_to_dataframe(
                    source_dataframes[person_id],
                    results[person_id],
                )
                for person_id in sequences
            }
            run_dir = output / f"ahp_{staff_name_ahp}__fuzzy_{staff_name_fuzzy}"
            save_evaluation_outputs(
                run_dir,
                evaluated_dataframes,
                results,
                staff_count=len(world_state.staff),
            )
            written_dirs.append(run_dir)
    return written_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--staff-names", nargs="+", default=["中村", "百武"])
    parser.add_argument("--common-dir", default="master_thesis_modules/database/common")
    parser.add_argument("--model", default="spatial_context")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="profile sweepの集計図と要約CSVも作成する",
    )
    args = parser.parse_args()
    written_dirs = run_profile_sweep(
        scenario=args.scenario,
        output=args.output,
        staff_names=args.staff_names,
        common_dir=args.common_dir,
        model=args.model,
    )
    for path in written_dirs:
        print(path)
    if args.visualize:
        for name, path in visualize_profile_sweep(args.output).items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
