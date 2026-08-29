"""Adapters between legacy node-number DataFrames and FeatureFrameSequence."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.features.feature_sequence import FeatureFrameSequence
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D
from master_thesis_modules.risk_core.schema import node_ids as ids


def normalize_legacy_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    columns = []
    for column in dataframe.columns:
        try:
            columns.append(int(column))
        except (TypeError, ValueError):
            columns.append(column)
    dataframe.columns = columns
    if "timestamp" not in dataframe.columns:
        dataframe.insert(0, "timestamp", range(len(dataframe)))
    return dataframe.reset_index(drop=True)


def dataframe_to_feature_sequence(
    person_id: str,
    dataframe: pd.DataFrame,
) -> FeatureFrameSequence:
    data = normalize_legacy_columns(dataframe)
    frames = [_row_to_feature_frame(person_id, row) for _, row in data.iterrows()]
    return FeatureFrameSequence(person_id=person_id, frames=frames)


def data_dicts_to_feature_sequences(
    data_dicts: dict[str, pd.DataFrame],
) -> dict[str, FeatureFrameSequence]:
    return {
        str(person_id): dataframe_to_feature_sequence(str(person_id), dataframe)
        for person_id, dataframe in data_dicts.items()
    }


def results_to_dataframe(
    source_dataframe: pd.DataFrame,
    results: list[RiskResult],
) -> pd.DataFrame:
    output = normalize_legacy_columns(source_dataframe)
    for result in results:
        for node_id, value in {**result.factor_risks, **result.upper_risks}.items():
            if node_id not in output.columns:
                output[node_id] = None if isinstance(value, tuple) else pd.NA
            elif isinstance(value, tuple) and output[node_id].dtype != "object":
                output[node_id] = output[node_id].astype("object")
    for result in results:
        row_index = _find_row_index(output, result.time_s)
        for node_id, value in result.factor_risks.items():
            output.at[row_index, node_id] = value
        for node_id, value in result.upper_risks.items():
            output.at[row_index, node_id] = value
        output.at[row_index, "explanation"] = result.explanation
    return output


def evaluated_dataframes_to_long_dataframe(
    evaluated: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for person_id, dataframe in evaluated.items():
        data = normalize_legacy_columns(dataframe)
        for _, row in data.iterrows():
            record = {"patient_id": person_id, "timestamp": row["timestamp"]}
            patient_label = _patient_label_from_row(row)
            if patient_label is not None:
                record["is_patient_label"] = patient_label
                record["is_rankable_patient"] = _is_rankable_patient_label(patient_label)
            for node_id in ids.EVALUATED_OUTPUT_NODES:
                if node_id in row:
                    record[str(node_id)] = row[node_id]
            if "explanation" in row:
                record["explanation"] = row["explanation"]
            rows.append(record)
    return pd.DataFrame(rows)


def _row_to_feature_frame(person_id: str, row: pd.Series) -> FeatureFrame:
    return FeatureFrame(
        person_id=person_id,
        time_s=float(row.get("timestamp", 0.0)),
        is_patient_label=_label(row.get(ids.IS_PATIENT, "yes")),
        is_patient_confidence=float(_value_or_default(row.get(50000001), 1.0)),
        age_group_label=_label(row.get(ids.AGE_CATEGORY, "middle")),
        age_confidence=float(_value_or_default(row.get(50000011), 1.0)),
        pose_features=PoseFeatures(
            float(_value_or_default(row.get(ids.POSE_STANDING_DEGREE), 0.0)),
            float(_value_or_default(row.get(ids.POSE_TRUNK_TILT), 0.0)),
            float(_value_or_default(row.get(ids.POSE_WRIST_DISTANCE_FROM_HIP), 0.0)),
            float(_value_or_default(row.get(ids.POSE_ANKLE_SPREAD), 0.0)),
        ),
        patient_position=Position2D(
            float(_value_or_default(row.get(ids.PERSON_X), 0.0)),
            float(_value_or_default(row.get(ids.PERSON_Y), 0.0)),
        ),
        nearest_iv_position=_optional_position(row, ids.IV_POLE_X, ids.IV_POLE_Y),
        nearest_wheelchair_position=_optional_position(
            row,
            ids.WHEELCHAIR_X,
            ids.WHEELCHAIR_Y,
        ),
        nearest_handrail_position=_optional_position(row, ids.HANDRAIL_X, ids.HANDRAIL_Y),
        nearest_staff_position=_optional_position(row, ids.STAFF_X, ids.STAFF_Y),
        nearest_staff_velocity=_optional_velocity(row, ids.STAFF_VX, ids.STAFF_VY),
        height_max=_optional_float(row.get(ids.PERSON_HEIGHT_MAX)),
    )


def _find_row_index(dataframe: pd.DataFrame, timestamp: float) -> int:
    matches = dataframe.index[dataframe["timestamp"] == timestamp].tolist()
    if matches:
        return int(matches[0])
    return int((dataframe["timestamp"] - timestamp).abs().idxmin())


def _optional_position(row: pd.Series, x_node: int, y_node: int) -> Position2D | None:
    x = _optional_float(row.get(x_node))
    y = _optional_float(row.get(y_node))
    if x is None or y is None:
        return None
    return Position2D(x, y)


def _optional_velocity(row: pd.Series, vx_node: int, vy_node: int) -> Velocity2D | None:
    vx = _optional_float(row.get(vx_node))
    vy = _optional_float(row.get(vy_node))
    if vx is None or vy is None:
        return None
    return Velocity2D(vx, vy)


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _value_or_default(value: Any, default: float) -> float:
    numeric = _optional_float(value)
    return default if numeric is None else numeric


def _label(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    try:
        if math.isnan(value):
            return "unknown"
    except TypeError:
        pass
    return str(value)


def _patient_label_from_row(row: pd.Series) -> str | None:
    if ids.IS_PATIENT in row:
        return _label(row.get(ids.IS_PATIENT))
    if str(ids.IS_PATIENT) in row:
        return _label(row.get(str(ids.IS_PATIENT)))
    if "is_patient_label" in row:
        return _label(row.get("is_patient_label"))
    return None


def _is_rankable_patient_label(label: str) -> bool:
    return label.strip().lower() not in {"no", "false", "0", "staff"}
