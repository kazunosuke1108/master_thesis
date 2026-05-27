import pickle

import pandas as pd

from master_thesis_modules.real_data.loader.data_dicts_loader import load_data_dicts
from master_thesis_modules.risk_core.features.dataframe_adapter import (
    data_dicts_to_feature_sequences,
)
from master_thesis_modules.risk_core.schema import node_ids as ids


def test_data_dicts_pickle_loads_and_converts_to_sequences(tmp_path):
    path = tmp_path / "data_dicts.pickle"
    data_dicts = {
        "A": pd.DataFrame(
            {
                "timestamp": [0.0],
                str(ids.IS_PATIENT): ["yes"],
                str(ids.AGE_CATEGORY): ["old"],
                str(ids.POSE_STANDING_DEGREE): [0.0],
                str(ids.POSE_TRUNK_TILT): [0.0],
                str(ids.POSE_WRIST_DISTANCE_FROM_HIP): [0.0],
                str(ids.POSE_ANKLE_SPREAD): [0.5],
                str(ids.PERSON_X): [0.0],
                str(ids.PERSON_Y): [0.0],
            }
        )
    }
    with path.open("wb") as handle:
        pickle.dump(data_dicts, handle)

    loaded = load_data_dicts(path)
    sequences = data_dicts_to_feature_sequences(loaded)

    assert sequences["A"].person_id == "A"
    assert len(sequences["A"].frames) == 1

