import inspect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.fuzzy.fuzzy_reasoning_v5 import FuzzyReasoning
from scripts.master_v5 import Master
from scripts.visualize.visualizer_v5 import Visualizer


def make_master_without_init(data_dicts):
    master = object.__new__(Master)
    master.data_dicts = data_dicts
    master.patients = list(data_dicts)
    master.spatial_normalization_param = np.sqrt(2) * 6
    master.AHP_dict = {
        30000001: {"weights": np.array([0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])},
        30000010: {"weights": np.array([0.3, 0.3, 0.4])},
    }
    master.graph_dicts = {
        patient: {"weight_dict": defaultdict(dict)} for patient in data_dicts
    }
    master.risky_motion_dict = {
        40000010: {"label": "standUp", "features": np.array([1, 0, 0, 1])},
        40000011: {"label": "releaseBrake", "features": np.array([0, 0.5, 0.5, 0.5])},
        40000012: {"label": "moveWheelchair", "features": np.array([0, 0.5, 0.5, 0.5])},
        40000013: {"label": "loseBalance", "features": np.array([0, 1, np.nan, np.nan])},
        40000014: {"label": "moveHand", "features": np.array([np.nan, 0, 1, np.nan])},
        40000015: {"label": "coughUp", "features": np.array([np.nan, 0.5, 0.5, np.nan])},
        40000016: {"label": "touchFace", "features": np.array([np.nan, 0, 0.5, np.nan])},
    }
    FuzzyReasoning.define_rules(master)
    return master


def test_fuzzy_total_risk_is_higher_for_higher_inputs():
    reasoning = FuzzyReasoning()

    safe = reasoning.calculate_fuzzy(
        input_nodes={20000000: 0.1, 20000001: 0.1}, output_node=10000000
    )
    dangerous = reasoning.calculate_fuzzy(
        input_nodes={20000000: 0.9, 20000001: 0.9}, output_node=10000000
    )

    assert dangerous > safe


def test_standing_feature_increases_internal_and_total_risk():
    reasoning = FuzzyReasoning()
    safe_action_risk = 0.1
    standing_action_risk = 0.9

    safe_internal = 0.1 * 0.5 + 0.9 * safe_action_risk
    standing_internal = 0.1 * 0.5 + 0.9 * standing_action_risk

    safe_total = reasoning.calculate_fuzzy(
        input_nodes={20000000: safe_internal, 20000001: 0.2}, output_node=10000000
    )
    standing_total = reasoning.calculate_fuzzy(
        input_nodes={20000000: standing_internal, 20000001: 0.2},
        output_node=10000000,
    )

    assert standing_total > safe_total


def test_staff_distance_and_watch_loss_raise_external_risk():
    reasoning = FuzzyReasoning()

    near_watched = reasoning.calculate_fuzzy(
        input_nodes={40000110: 0.1, 40000111: 0.1}, output_node=30000011
    )
    far_unwatched = reasoning.calculate_fuzzy(
        input_nodes={40000110: 0.9, 40000111: 0.9}, output_node=30000011
    )
    watch_lost = reasoning.calculate_fuzzy(
        input_nodes={40000110: 0.1, 40000111: 0.9}, output_node=30000011
    )

    assert far_unwatched > near_watched
    assert watch_lost > near_watched


def test_profile_rules_can_prioritize_personal_or_spatial_context():
    reasoning = FuzzyReasoning()

    personal_profile = pd.DataFrame({"c": [0.5] * 12})
    personal_profile.loc[8, "c"] = 1.0
    personal_profile.loc[9, "c"] = 0.8
    personal_profile.loc[10, "c"] = 0.2
    personal_profile.loc[11, "c"] = 0.0
    reasoning.define_custom_rules(personal_profile)
    personal_motion = reasoning.calculate_fuzzy(
        input_nodes={20000000: 1.0, 20000001: 0.0}, output_node=10000000
    )
    personal_context = reasoning.calculate_fuzzy(
        input_nodes={20000000: 0.0, 20000001: 1.0}, output_node=10000000
    )

    spatial_profile = pd.DataFrame({"c": [0.5] * 12})
    spatial_profile.loc[8, "c"] = 1.0
    spatial_profile.loc[9, "c"] = 0.2
    spatial_profile.loc[10, "c"] = 0.8
    spatial_profile.loc[11, "c"] = 0.0
    reasoning.define_custom_rules(spatial_profile)
    spatial_motion = reasoning.calculate_fuzzy(
        input_nodes={20000000: 1.0, 20000001: 0.0}, output_node=10000000
    )
    spatial_context = reasoning.calculate_fuzzy(
        input_nodes={20000000: 0.0, 20000001: 1.0}, output_node=10000000
    )

    assert personal_motion > personal_context
    assert spatial_context > spatial_motion


def test_high_risk_sort_order_is_descending():
    patients = np.array(["A", "B", "C"])
    risks = np.array([0.2, 0.9, 0.5])

    ranked_patients = patients[np.argsort(-risks)]

    assert ranked_patients.tolist() == ["B", "C", "A"]


def test_plot_matplotlib_uses_internal_risk_value_without_inversion():
    source = inspect.getsource(Visualizer.plot_matplotlib)

    assert "plt.ylabel(\"Risk value\")" in source
    assert "data_dict[id_name][export_label].rolling(w).mean()" in source
    assert "1-data_dict" not in source.replace(" ", "")
    assert "1 - data_dict" not in source


@pytest.mark.xfail(
    strict=True,
    reason="master_v5.staff_risk overwrites the full 40000111 column with the last row",
)
def test_master_staff_watch_risk_changes_by_timestamp():
    df = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0],
            60010000: [0.0, 0.0],
            60010001: [0.0, 0.0],
            50001100: [-1.0, -1.0],
            50001101: [0.0, 0.0],
            50001110: [1.0, -1.0],
            50001111: [0.0, 0.0],
        }
    )
    master = make_master_without_init({"A": df})

    master.staff_risk()

    assert master.data_dicts["A"].loc[0, 40000111] < 0.1
    assert master.data_dicts["A"].loc[1, 40000111] > 0.9


@pytest.mark.xfail(
    strict=True,
    reason="Current rank assignment zips patients with argsort indices instead of inverse ranks",
)
def test_notification_rank_assignment_maps_each_patient_to_own_rank():
    patients = ["A", "B", "C"]
    risks = np.array([0.2, 0.9, 0.5])

    argsort_indices = (-risks).argsort()
    current_assignment = {
        patient: rank for patient, rank in zip(patients, argsort_indices)
    }

    assert current_assignment == {"A": 2, "B": 0, "C": 1}


@pytest.mark.xfail(
    strict=True,
    reason="Real-data object preprocessing stores existence probability in fields consumed as xy coordinates",
)
def test_object_features_are_coordinates_before_master_distance_risk():
    data_dict = {
        "60010000": -8.0,
        "60010001": 12.0,
        "50001000": 1.0,
        "50001001": 1.0,
        "50001010": 1.0,
        "50001011": 1.0,
        "50001020": -8.0,
        "50001021": 9.0,
    }
    iv_pole_xy = np.array([data_dict["50001000"], data_dict["50001001"]])
    patient_xy = np.array([data_dict["60010000"], data_dict["60010001"]])

    assert np.linalg.norm(iv_pole_xy - patient_xy) < 3.0
