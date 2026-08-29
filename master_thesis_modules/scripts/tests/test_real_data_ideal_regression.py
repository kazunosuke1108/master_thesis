from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.notification.rank_utils import get_risk_rank_by_patient


REAL_DATA_COLUMNS = [
    "50000100",
    "40000010",
    "40000110",
    "40000111",
    "30000001",
    "30000011",
    "10000000",
]


def make_real_data_ideal_fixture():
    return {
        "00021": pd.DataFrame(
            {
                "timestamp": [74.0, 76.0],
                "50000100": [0.2, 0.95],
                "40000010": [0.2, 0.95],
                "40000110": [0.2, 0.9],
                "40000111": [0.2, 0.9],
                "30000001": [0.2, 0.9],
                "30000011": [0.2, 0.9],
                "10000000": [0.2, 0.92],
            }
        ),
        "00006": pd.DataFrame(
            {
                "timestamp": [74.0, 76.0],
                "50000100": [0.1, 0.1],
                "40000010": [0.1, 0.1],
                "40000110": [0.2, 0.2],
                "40000111": [0.2, 0.2],
                "30000001": [0.1, 0.1],
                "30000011": [0.2, 0.2],
                "10000000": [0.15, 0.2],
            }
        ),
    }


def test_real_data_ideal_fixture_has_required_feature_table_after_75s():
    data_dicts = make_real_data_ideal_fixture()
    row = data_dicts["00021"][data_dicts["00021"]["timestamp"] >= 75].iloc[0]

    feature_table = row[REAL_DATA_COLUMNS]

    assert list(feature_table.index) == REAL_DATA_COLUMNS
    assert feature_table["50000100"] > 0.9
    assert feature_table["40000010"] > 0.9
    assert feature_table["40000110"] > 0.8
    assert feature_table["40000111"] > 0.8
    assert feature_table["10000000"] > 0.9


def test_real_data_ideal_fixture_ranks_00021_highest_after_75s():
    data_dicts = make_real_data_ideal_fixture()
    patients = list(data_dicts)
    risks = [
        data_dicts[patient][data_dicts[patient]["timestamp"] >= 75]["10000000"].iloc[0]
        for patient in patients
    ]

    ranks = get_risk_rank_by_patient(patients, risks)

    assert ranks["00021"] == 0
    assert np.argmax(risks) == patients.index("00021")
