import pandas as pd

from master_thesis_modules.risk_core.engine.batch_risk_engine import BatchRiskEngine
from master_thesis_modules.risk_core.features.dataframe_adapter import (
    data_dicts_to_feature_sequences,
    results_to_dataframe,
)
from master_thesis_modules.risk_core.schema import node_ids as ids


def test_batch_engine_handles_multiple_people_and_timestamps():
    data_dicts = {
        "A": pd.DataFrame(_rows([0.1, 0.9])),
        "B": pd.DataFrame(_rows([0.1, 0.1])),
    }
    sequences = data_dicts_to_feature_sequences(data_dicts)
    results = BatchRiskEngine().evaluate(sequences)
    evaluated_a = results_to_dataframe(data_dicts["A"], results["A"])

    assert set(results) == {"A", "B"}
    assert evaluated_a["timestamp"].tolist() == [0.0, 1.0]
    assert ids.TOTAL_RISK in evaluated_a.columns


def test_all_nan_height_column_does_not_overwrite_standing_risk():
    data_dicts = {
        "B": pd.DataFrame(
            {
                **_rows([0.1, 0.9]),
                ids.PERSON_HEIGHT_MAX: [float("nan"), float("nan")],
            }
        ),
    }
    sequences = data_dicts_to_feature_sequences(data_dicts)
    results = BatchRiskEngine().evaluate(sequences)
    evaluated_b = results_to_dataframe(data_dicts["B"], results["B"])

    assert evaluated_b.loc[1, ids.STANDING_RISK] > evaluated_b.loc[0, ids.STANDING_RISK]
    assert evaluated_b[ids.STANDING_RISK].notna().all()


def _rows(standing_values):
    return {
        "timestamp": [0.0, 1.0],
        ids.IS_PATIENT: ["yes", "yes"],
        ids.AGE_CATEGORY: ["old", "old"],
        ids.POSE_STANDING_DEGREE: standing_values,
        ids.POSE_TRUNK_TILT: [0.0, 0.0],
        ids.POSE_WRIST_DISTANCE_FROM_HIP: [0.0, 0.0],
        ids.POSE_ANKLE_SPREAD: [0.5, 1.0],
        ids.PERSON_X: [0.0, 0.0],
        ids.PERSON_Y: [0.0, 0.0],
        ids.STAFF_X: [1.0, 1.0],
        ids.STAFF_Y: [0.0, 0.0],
        ids.STAFF_VX: [1.0, -1.0],
        ids.STAFF_VY: [0.0, 0.0],
    }
