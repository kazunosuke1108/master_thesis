from pathlib import Path

from master_thesis_modules.scenario_sim.runner.run_profile_sweep import run_profile_sweep
from master_thesis_modules.scenario_sim.encoder.scenario_loader import ScenarioLoader
from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    visualize_profile_sweep,
)
from master_thesis_modules.scenario_sim.visualization.plot_scenario_storyboard import (
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
