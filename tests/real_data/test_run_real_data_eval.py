import pickle

import pandas as pd

from master_thesis_modules.real_data.runner.run_real_data_eval import run_real_data_eval
from master_thesis_modules.risk_core.schema import node_ids as ids
from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    visualize_profile_sweep,
)


def test_run_real_data_eval_writes_profile_sweep_outputs(tmp_path):
    input_path = tmp_path / "data_dicts.pickle"
    output = tmp_path / "real_data_eval"
    data_dicts = {
        "A": pd.DataFrame(
            {
                "timestamp": [0.0, 1.0],
                str(ids.IS_PATIENT): ["yes", "yes"],
                str(ids.AGE_CATEGORY): ["old", "old"],
                str(ids.POSE_STANDING_DEGREE): [0.0, 0.8],
                str(ids.POSE_TRUNK_TILT): [0.0, 0.2],
                str(ids.POSE_WRIST_DISTANCE_FROM_HIP): [0.1, 0.4],
                str(ids.POSE_ANKLE_SPREAD): [0.5, 0.7],
                str(ids.PERSON_X): [0.0, 0.2],
                str(ids.PERSON_Y): [0.0, 0.2],
            }
        )
    }
    with input_path.open("wb") as handle:
        pickle.dump(data_dicts, handle)

    written_dirs = run_real_data_eval(
        input_path=input_path,
        output=output,
        staff_names=["中村", "百武"],
    )

    assert len(written_dirs) == 4
    for run_dir in written_dirs:
        assert run_dir.exists()
        assert (run_dir / "risk_timeseries.csv").exists()
        assert (run_dir / "ranking.csv").exists()
        assert (run_dir / "notification_log.csv").exists()
        assert (run_dir / "explanations.json").exists()
        assert (run_dir / "data_A_eval.csv").exists()

    paths = visualize_profile_sweep(output)
    assert paths["profile_summary"].exists()
    assert paths["profile_ranking_summary"].exists()
    assert paths["profile_total_risk_grid"].exists()
