from pathlib import Path

from master_thesis_modules.scenario_sim.runner.run_thesis_simulation import (
    run_thesis_simulation,
)


def test_thesis_4_5_demo_runs_and_writes_logs(tmp_path):
    output = tmp_path / "thesis_4_5"

    run_thesis_simulation(
        Path("master_thesis_modules/scenario_sim/scenarios/thesis_4_5_multi_patient_action_demo.yaml"),
        output,
    )

    assert (output / "risk_timeseries.csv").exists()
    assert (output / "ranking.csv").exists()
    assert (output / "notification_log.csv").exists()
    assert (output / "explanations.json").exists()

