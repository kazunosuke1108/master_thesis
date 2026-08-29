import pickle

import pandas as pd

from master_thesis_modules.real_data.runner.run_real_data_eval import run_real_data_eval
from master_thesis_modules.risk_core.schema import node_ids as ids
from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    visualize_profile_sweep,
)
from master_thesis_modules.scenario_sim.visualization.plot_fuzzy_profile_rankings import (
    analyze_fuzzy_profile_rankings,
)


def test_run_real_data_eval_writes_profile_sweep_outputs(tmp_path):
    input_path = tmp_path / "data_dicts.pickle"
    output = tmp_path / "real_data_eval"
    data_dicts = {
        "00001": pd.DataFrame(
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
        assert (run_dir / "data_00001_eval.csv").exists()

    paths = visualize_profile_sweep(output)
    assert paths["profile_summary"].exists()
    assert paths["profile_ranking_summary"].exists()
    assert paths["profile_total_risk_grid"].exists()

    analysis_paths = analyze_fuzzy_profile_rankings(output, ahp_profile="中村")
    assert analysis_paths["fuzzy_profile_ranking_plot"].exists()
    assert analysis_paths["fuzzy_ci_patient_mean"].exists()
    assert analysis_paths["fuzzy_profile_di_plot"].exists()
    assert analysis_paths["fuzzy_di_patient_mean"].exists()


def test_run_real_data_eval_can_fix_ahp_while_sweeping_fuzzy_profiles(tmp_path):
    input_path = tmp_path / "data_dicts.pickle"
    data_dicts = {
        "00001": pd.DataFrame(
            {
                "timestamp": [0.0],
                str(ids.IS_PATIENT): ["yes"],
                str(ids.AGE_CATEGORY): ["old"],
                str(ids.PERSON_X): [0.0],
                str(ids.PERSON_Y): [0.0],
            }
        )
    }
    with input_path.open("wb") as handle:
        pickle.dump(data_dicts, handle)

    written_dirs = run_real_data_eval(
        input_path=input_path,
        output=tmp_path / "fixed_ahp",
        staff_names=["中村", "百武"],
        ahp_staff_names=["中村"],
    )

    assert {path.name for path in written_dirs} == {
        "ahp_中村__fuzzy_中村",
        "ahp_中村__fuzzy_百武",
    }
