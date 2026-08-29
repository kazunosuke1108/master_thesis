from pathlib import Path

import pandas as pd

from master_thesis_modules.scenario_sim.runner.run_profile_sweep import (
    _expand_all_profile_names,
    run_profile_sweep,
)
from master_thesis_modules.scenario_sim.visualization.plot_fuzzy_profile_rankings import (
    _profile_x_positions,
    build_ci_patient_means,
    build_fuzzy_profile_rankings,
    load_fuzzy_di,
    load_fuzzy_ci,
)
from master_thesis_modules.scenario_sim.encoder.scenario_loader import ScenarioLoader
from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    ProfileRun,
    visualize_profile_sweep,
)
from master_thesis_modules.scenario_sim.visualization.plot_scenario_storyboard import (
    _plot_bounds,
    build_scenario_snapshots,
    visualize_scenario_storyboard,
)


def test_profile_sweep_visualization_outputs_summary_and_figures(tmp_path):
    output = tmp_path / "profile_sweep"
    run_profile_sweep(
        scenario=Path("master_thesis_modules/scenario_sim/scenarios/thesis_4_5_multi_patient_action_demo.yaml"),
        output=output,
        staff_names=["中村", "百武"],
    )

    paths = visualize_profile_sweep(output)

    assert paths["profile_summary"].exists()
    assert paths["profile_ranking_summary"].exists()
    assert paths["profile_total_risk_grid"].exists()
    assert paths["profile_top_risk_comparison"].exists()
    assert paths["profile_notification_counts"].exists()


def test_profile_sweep_can_fix_ahp_while_sweeping_fuzzy_profiles(tmp_path):
    output = tmp_path / "fixed_ahp_profile_sweep"

    written_dirs = run_profile_sweep(
        scenario=Path("master_thesis_modules/scenario_sim/scenarios/thesis_4_5_multi_patient_action_demo.yaml"),
        output=output,
        staff_names=["中村", "百武"],
        ahp_staff_names=["中村"],
    )

    assert {path.name for path in written_dirs} == {
        "ahp_中村__fuzzy_中村",
        "ahp_中村__fuzzy_百武",
    }


def test_all_profile_selector_uses_only_complete_profiles(tmp_path):
    for filename in (
        "comparison_mtx_30000001_山口.csv",
        "comparison_mtx_30000010_山口.csv",
        "TFN_山口.csv",
        "comparison_mtx_30000001_百武.csv",
        "comparison_mtx_30000010_百武.csv",
    ):
        (tmp_path / filename).touch()

    assert _expand_all_profile_names(["all"], tmp_path) == ["山口"]


def test_fuzzy_profile_ranking_uses_c_column_and_mean_total_risk(tmp_path):
    tfn = pd.DataFrame({"l": [0.0] * 12, "c": range(12), "r": [0.0] * 12})
    tfn.to_csv(tmp_path / "TFN_テスト.csv", index=False, header=False)
    run = ProfileRun(
        profile_name="ahp_AHP__fuzzy_テスト",
        ahp_profile="AHP",
        fuzzy_profile="テスト",
        path=tmp_path,
        risk_timeseries=pd.DataFrame(
            {
                "patient_id": ["A", "A", "B", "B", "staff"],
                "is_rankable_patient": [True, True, True, True, False],
                "10000000": [0.4, 0.8, 0.7, 0.7, 1.0],
            }
        ),
        ranking=pd.DataFrame(),
        notification_log=pd.DataFrame(),
    )

    rankings = build_fuzzy_profile_rankings([run], tmp_path)

    assert load_fuzzy_ci(tmp_path / "TFN_テスト.csv") == 1.0
    assert load_fuzzy_di(tmp_path / "TFN_テスト.csv") == 3.0
    assert rankings[["rank", "patient_id"]].values.tolist() == [[1, "B"], [2, "A"]]
    assert rankings[["patient_id", "normalized_mean_total_risk"]].values.tolist() == [
        ["B", 1.0],
        ["A", 0.0],
    ]


def test_equal_ci_profiles_are_offset_around_their_numeric_value():
    positions = _profile_x_positions(
        pd.DataFrame(
            {
                "fuzzy_profile": ["A", "B", "C"],
                "C_i": [0.0, 0.0, 0.5],
            }
        )
    )

    assert positions["A"] < 0 < positions["B"]
    assert positions["C"] == 0.5


def test_equal_ci_profiles_are_ordered_by_descending_normalized_c_minus_b():
    positions = _profile_x_positions(
        pd.DataFrame(
            {
                "fuzzy_profile": ["small", "small", "large", "large"],
                "patient_id": ["B", "C", "B", "C"],
                "C_i": [0.0, 0.0, 0.0, 0.0],
                "normalized_mean_total_risk": [0.3, 0.8, 0.9, 0.2],
            }
        )
    )

    assert positions["small"] < positions["large"]


def test_ci_patient_means_average_profiles_with_the_same_ci():
    means = build_ci_patient_means(
        pd.DataFrame(
            {
                "C_i": [0.5, 0.5, 0.5],
                "patient_id": ["A", "A", "B"],
                "fuzzy_profile": ["first", "second", "first"],
                "normalized_mean_total_risk": [0.2, 0.8, 0.6],
            }
        )
    )

    assert means.to_dict("records") == [
        {
            "C_i": 0.5,
            "patient_id": "A",
            "mean_normalized_total_risk": 0.5,
            "profile_count": 2,
        },
        {
            "C_i": 0.5,
            "patient_id": "B",
            "mean_normalized_total_risk": 0.6,
            "profile_count": 1,
        },
    ]


def test_scenario_storyboard_visualization_outputs_figure_and_table(tmp_path):
    world_state = ScenarioLoader().load(
        Path("master_thesis_modules/scenario_sim/scenarios/20260507_standup.yaml")
    )

    paths = visualize_scenario_storyboard(
        world_state,
        tmp_path / "scenario_storyboard.png",
        tmp_path / "scenario_storyboard_snapshots.csv",
    )

    assert paths["scenario_storyboard"].exists()
    assert paths["scenario_storyboard_snapshots"].exists()


def test_scenario_storyboard_bounds_include_room_walls():
    world_state = ScenarioLoader().load(
        Path("master_thesis_modules/scenario_sim/scenarios/20260507_standup.yaml")
    )

    min_x, max_x, min_y, _ = _plot_bounds(build_scenario_snapshots(world_state))

    assert min_x < 0.0
    assert max_x >= 8.0
    assert min_y < 0.0
