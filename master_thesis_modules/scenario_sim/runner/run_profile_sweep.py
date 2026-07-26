"""Run scenario simulations while sweeping AHP and Fuzzy profile names."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_thesis_modules.risk_core.engine.batch_risk_engine import BatchRiskEngine
from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config
from master_thesis_modules.risk_core.engine.risk_config import VALID_MODEL_TYPES
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.features.dataframe_adapter import (
    data_dicts_to_feature_sequences,
    results_to_dataframe,
)
from master_thesis_modules.scenario_sim.encoder.dataframe_builder import (
    ScenarioDataFrameBuilder,
)
from master_thesis_modules.scenario_sim.encoder.master_v5_compat import (
    build_master_v5_default_source_dataframes,
)
from master_thesis_modules.scenario_sim.encoder.scenario_loader import ScenarioLoader
from master_thesis_modules.scenario_sim.runner._outputs import (
    build_source_dataframes,
    save_evaluation_outputs,
)
from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    visualize_profile_sweep,
)
from master_thesis_modules.scenario_sim.visualization.plot_scenario_storyboard import (
    visualize_scenario_storyboard,
)


def _expand_all_profile_names(
    profile_names: list[str], common_dir: str | Path
) -> list[str]:
    """Replace the ``all`` selector with complete profiles in ``common_dir``."""
    if "all" not in profile_names:
        return profile_names
    if profile_names != ["all"]:
        raise ValueError("'all' cannot be combined with individual profile names")

    common_dir = Path(common_dir)
    action_profiles = {
        path.stem.removeprefix("comparison_mtx_30000001_")
        for path in common_dir.glob("comparison_mtx_30000001_*.csv")
    }
    object_profiles = {
        path.stem.removeprefix("comparison_mtx_30000010_")
        for path in common_dir.glob("comparison_mtx_30000010_*.csv")
    }
    fuzzy_profiles = {
        path.stem.removeprefix("TFN_") for path in common_dir.glob("TFN_*.csv")
    }
    complete_profiles = sorted(action_profiles & object_profiles & fuzzy_profiles)
    if not complete_profiles:
        raise ValueError(
            f"No complete AHP/Fuzzy profiles found in common directory: {common_dir}"
        )
    return complete_profiles


def run_profile_sweep(
    scenario: str | Path,
    output: str | Path,
    staff_names: list[str] | None = None,
    common_dir: str | Path = "master_thesis_modules/database/common",
    model: str = "spatial_context",
    action_aggregation: str = "weighted_sum",
    notification_message_style: str = "current",
    ahp_staff_names: list[str] | None = None,
    fuzzy_staff_names: list[str] | None = None,
) -> list[Path]:
    """Evaluate every requested AHP/Fuzzy profile combination.

    ``staff_names`` is the shared candidate list retained for backwards
    compatibility.  Supplying either dedicated list restricts only that side
    of the Cartesian product.
    """
    staff_names = _expand_all_profile_names(
        staff_names or ["中村", "百武"], common_dir
    )
    ahp_staff_names = (
        _expand_all_profile_names(ahp_staff_names, common_dir)
        if ahp_staff_names
        else staff_names
    )
    fuzzy_staff_names = (
        _expand_all_profile_names(fuzzy_staff_names, common_dir)
        if fuzzy_staff_names
        else staff_names
    )
    world_state = ScenarioLoader().load(scenario)
    use_master_v5_source = world_state.scenario_name == "thesis_4_5_multi_patient_action_demo"
    if use_master_v5_source:
        source_dataframes = build_master_v5_default_source_dataframes()
        sequences = data_dicts_to_feature_sequences(source_dataframes)
    else:
        sequences = ScenarioDataFrameBuilder().build_sequences(world_state)
        source_dataframes = build_source_dataframes(sequences)
    output = Path(output)
    written_dirs = []

    for staff_name_ahp in ahp_staff_names:
        for staff_name_fuzzy in fuzzy_staff_names:
            config = make_profile_risk_config(
                ahp_profile_name=staff_name_ahp,
                fuzzy_profile_name=staff_name_fuzzy,
                common_dir=common_dir,
                model_type=model,
                action_aggregation=action_aggregation,
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
            if use_master_v5_source:
                evaluated_dataframes = {
                    person_id: dataframe.drop(columns=["explanation"], errors="ignore")
                    for person_id, dataframe in evaluated_dataframes.items()
                }
            run_dir = output / f"ahp_{staff_name_ahp}__fuzzy_{staff_name_fuzzy}"
            save_evaluation_outputs(
                run_dir,
                evaluated_dataframes,
                results,
                staff_count=len(world_state.staff),
                notification_message_style=notification_message_style,
            )
            written_dirs.append(run_dir)
    return written_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--staff-names",
        nargs="+",
        default=None,
        help="AHP/Fuzzy両方に使うプロファイル候補。allならcommon-dir内でAHP/Fuzzyが揃う全員を使う",
    )
    parser.add_argument(
        "--ahp-staff-names",
        nargs="+",
        default=None,
        help="AHPプロファイル候補。allならcommon-dir内でAHP/Fuzzyが揃う全員を使う",
    )
    parser.add_argument(
        "--fuzzy-staff-names",
        nargs="+",
        default=None,
        help="Fuzzyプロファイル候補。allならcommon-dir内でAHP/Fuzzyが揃う全員を使う",
    )
    parser.add_argument("--common-dir", default="master_thesis_modules/database/common")
    parser.add_argument(
        "--model",
        choices=sorted(VALID_MODEL_TYPES),
        default="spatial_context",
        help="risk model. spatial_context uses patient and spatial context; patient_context ignores object/staff context",
    )
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
    written_dirs = run_profile_sweep(
        scenario=args.scenario,
        output=args.output,
        staff_names=args.staff_names,
        common_dir=args.common_dir,
        model=args.model,
        action_aggregation=args.action_aggregation,
        notification_message_style=args.notification_message_style,
        ahp_staff_names=args.ahp_staff_names,
        fuzzy_staff_names=args.fuzzy_staff_names,
    )
    for path in written_dirs:
        print(path)
    if args.visualize:
        visualization_paths = visualize_profile_sweep(args.output)
        visualization_dir = Path(args.output) / "visualization"
        visualization_paths.update(
            visualize_scenario_storyboard(
                ScenarioLoader().load(args.scenario),
                visualization_dir / "scenario_storyboard.png",
                visualization_dir / "scenario_storyboard_snapshots.csv",
            )
        )
        for name, path in visualization_paths.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
