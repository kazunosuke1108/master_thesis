from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.fuzzy.fuzzy_reasoning_v5 import FuzzyReasoning
from scripts.notification.rank_utils import get_risk_rank_by_patient


def make_profile(personal_weight, spatial_weight):
    profile = pd.DataFrame({"c": [0.5] * 12})
    profile.loc[5, "c"] = 1.0
    profile.loc[6, "c"] = 0.0
    profile.loc[7, "c"] = 1.0
    profile.loc[8, "c"] = 1.0
    profile.loc[9, "c"] = personal_weight
    profile.loc[10, "c"] = spatial_weight
    profile.loc[11, "c"] = 0.0
    return profile


PROFILES = {
    "nakamura": make_profile(personal_weight=0.8, spatial_weight=0.8),
    "nakatake": make_profile(personal_weight=0.8, spatial_weight=0.2),
    "hyakumura": make_profile(personal_weight=0.2, spatial_weight=0.8),
    "hyakutake": make_profile(personal_weight=0.8, spatial_weight=0.2),
}


def score_patients(profile, scenario):
    reasoning = FuzzyReasoning()
    reasoning.define_custom_rules(profile)
    scores = {}
    factors = {}
    for patient, values in scenario.items():
        internal = 0.1 * 0.5 + 0.9 * values["standing"]
        external_dynamic = reasoning.calculate_fuzzy(
            input_nodes={40000110: values["staff_distance"], 40000111: values["watch_loss"]},
            output_node=30000011,
        )
        external = reasoning.calculate_fuzzy(
            input_nodes={30000010: 0.1, 30000011: external_dynamic},
            output_node=20000001,
        )
        scores[patient] = reasoning.calculate_fuzzy(
            input_nodes={20000000: internal, 20000001: external},
            output_node=10000000,
        )
        factors[patient] = "standing" if internal >= external else "watch_loss"
    return scores, factors


def top_patients(scores, threshold=0.45):
    ranked = get_risk_rank_by_patient(list(scores), list(scores.values()))
    return [patient for patient, rank in sorted(ranked.items(), key=lambda item: item[1]) if scores[patient] >= threshold]


def test_simulation_ideal_profiles_at_standing_event():
    scenario = {
        "A": {"standing": 0.1, "staff_distance": 0.1, "watch_loss": 0.1},
        "B": {"standing": 0.9, "staff_distance": 0.1, "watch_loss": 0.1},
        "C": {"standing": 0.1, "staff_distance": 0.1, "watch_loss": 0.1},
    }

    expected_top = {
        "nakamura": ["B"],
        "nakatake": ["B"],
        "hyakumura": [],
        "hyakutake": ["B"],
    }
    for profile_name, profile in PROFILES.items():
        scores, factors = score_patients(profile, scenario)
        assert top_patients(scores) == expected_top[profile_name]
        for patient in expected_top[profile_name]:
            assert factors[patient] == "standing"


def test_simulation_ideal_profiles_at_watch_loss_event():
    scenario = {
        "A": {"standing": 0.1, "staff_distance": 0.9, "watch_loss": 0.9},
        "B": {"standing": 0.1, "staff_distance": 0.1, "watch_loss": 0.1},
        "C": {"standing": 0.1, "staff_distance": 0.9, "watch_loss": 0.9},
    }

    expected_top = {
        "nakamura": ["A", "C"],
        "nakatake": [],
        "hyakumura": ["A", "C"],
        "hyakutake": [],
    }
    for profile_name, profile in PROFILES.items():
        scores, factors = score_patients(profile, scenario)
        assert top_patients(scores) == expected_top[profile_name]
        for patient in expected_top[profile_name]:
            assert factors[patient] == "watch_loss"
