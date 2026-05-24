"""Evaluate legacy real-data inputs with the new risk core."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_thesis_modules.risk_core.engine.batch_risk_engine import BatchRiskEngine
from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.features.dataframe_adapter import (
    data_dicts_to_feature_sequences,
    results_to_dataframe,
)
from master_thesis_modules.real_data.loader.trial_loader import load_trial_input
from master_thesis_modules.scenario_sim.runner._outputs import save_evaluation_outputs
from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    visualize_profile_sweep,
)


def run_real_data_eval(
    input_path: str | Path,
    output: str | Path,
    staff_names: list[str],
    common_dir: str | Path = "master_thesis_modules/database/common",
    model: str = "spatial_context",
    staff_count: int = 1,
    action_aggregation: str = "weighted_sum",
    notification_message_style: str = "current",
) -> list[Path]:
    data_dicts = load_trial_input(input_path)
    sequences = data_dicts_to_feature_sequences(data_dicts)
    output = Path(output)
    written_dirs = []

    for staff_name_ahp in staff_names:
        for staff_name_fuzzy in staff_names:
            config = make_profile_risk_config(
                ahp_profile_name=staff_name_ahp,
                fuzzy_profile_name=staff_name_fuzzy,
                common_dir=common_dir,
                model_type=model,
                action_aggregation=action_aggregation,
            )
            engine = BatchRiskEngine(RiskEngine(config))
            results = engine.evaluate(sequences)
            evaluated = {
                person_id: results_to_dataframe(
                    data_dicts[person_id],
                    results[person_id],
                )
                for person_id in data_dicts
            }
            run_dir = output / f"ahp_{staff_name_ahp}__fuzzy_{staff_name_fuzzy}"
            save_evaluation_outputs(
                run_dir,
                evaluated,
                results,
                staff_count=staff_count,
                notification_message_style=notification_message_style,
            )
            written_dirs.append(run_dir)
    return written_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--staff-names", nargs="+", default=["中村", "百武"])
    parser.add_argument("--common-dir", default="master_thesis_modules/database/common")
    parser.add_argument("--model", default="spatial_context")
    parser.add_argument("--staff-count", type=int, default=1)
    parser.add_argument(
        "--action-aggregation",
        choices=["weighted_sum", "weighted_max"],
        default="weighted_sum",
        help="30000001の動作リスク集約方法。weighted_sumは従来のAHP重み和、weighted_maxはmax_j(w_j * r_j)",
    )
    parser.add_argument(
        "--notification-message-style",
        choices=["current", "legacy"],
        default="current",
        help="notification_log.csvの通知文面。legacyはnotification_generator_v5.py互換の文面にする",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="profile sweepの集計図と要約CSVも作成する",
    )
    args = parser.parse_args()
    written_dirs = run_real_data_eval(
        input_path=args.input,
        output=args.output,
        staff_names=args.staff_names,
        common_dir=args.common_dir,
        model=args.model,
        staff_count=args.staff_count,
        action_aggregation=args.action_aggregation,
        notification_message_style=args.notification_message_style,
    )
    for path in written_dirs:
        print(path)
    if args.visualize:
        visualization_paths = visualize_profile_sweep(args.output)
        for name, path in visualization_paths.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
