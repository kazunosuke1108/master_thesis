"""Build the default ``master_v5.py`` pseudo-data scenario exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd


LEGACY_EVAL_COLUMNS = [
    "timestamp",
    10000000,
    20000000,
    20000001,
    30000000,
    30000001,
    30000010,
    30000011,
    40000000,
    40000001,
    40000010,
    40000011,
    40000012,
    40000013,
    40000014,
    40000015,
    40000016,
    40000100,
    40000101,
    40000102,
    40000110,
    40000111,
    50000000,
    50000001,
    50000010,
    50000011,
    50000100,
    50000101,
    50000102,
    50000103,
    50001000,
    50001001,
    50001002,
    50001003,
    50001010,
    50001011,
    50001012,
    50001013,
    50001020,
    50001021,
    50001022,
    50001023,
    50001100,
    50001101,
    50001110,
    50001111,
    60010000,
    60010001,
    60010002,
    70000000,
]


def build_master_v5_default_source_dataframes() -> dict[str, pd.DataFrame]:
    """Return the source DataFrames used by ``master_v5.py``'s default Sim run."""

    start_timestamp = 0.0
    end_timestamp = 10.0
    fps = 20
    timestamps = np.arange(start_timestamp, end_timestamp + 1e-4, step=1 / fps)
    patients = ["A", "B", "C"]
    positions = {
        "A": (2.0, 5.0),
        "B": (2.0, 2.0),
        "C": (5.0, 2.0),
        "NS": (5.0, 5.0),
    }
    actions = {
        "A": (("sit", 0.0, end_timestamp),),
        "B": (
            ("sit", 0.0, 2.0),
            ("standup", 2.0, 4.0),
            ("stand", 4.0, 6.0),
            ("sitdown", 6.0, 8.0),
            ("sit", 8.0, end_timestamp),
        ),
        "C": (("sit", 0.0, end_timestamp),),
    }
    surrounding_objects = {
        "A": ("wheelchair", "ivPole"),
        "B": ("wheelchair",),
        "C": (),
    }

    dataframes = {
        patient: pd.DataFrame(np.nan, index=range(len(timestamps)), columns=LEGACY_EVAL_COLUMNS)
        for patient in patients
    }
    for patient, dataframe in dataframes.items():
        dataframe["timestamp"] = timestamps
        dataframe[50000000] = "yes"
        dataframe[50000001] = 1
        dataframe[50000010] = "old"
        dataframe[50000011] = 1
        dataframe[60010000] = positions[patient][0]
        dataframe[60010001] = positions[patient][1]
        _apply_patient_actions(dataframe, actions[patient])

    object_positions: dict[str, np.ndarray] = {}
    for patient in patients:
        for object_type in surrounding_objects[patient]:
            object_positions[f"{object_type}_{patient}"] = np.array(positions[patient])
    for patient, dataframe in dataframes.items():
        _apply_nearest_objects(dataframe, positions[patient], object_positions)
        _apply_nearest_wall(dataframe, positions[patient], xrange=(0.0, 6.0), yrange=(0.0, 6.0))

    _apply_staff_trajectory(dataframes, positions)
    _apply_background_difference(dataframes, actions)
    for dataframe in dataframes.values():
        _restore_legacy_dtypes(dataframe)
    return dataframes


def _apply_patient_actions(
    dataframe: pd.DataFrame,
    action_segments: tuple[tuple[str, float, float], ...],
) -> None:
    for label, start, end in action_segments:
        mask = (dataframe["timestamp"] >= start) & (dataframe["timestamp"] <= end)
        t = dataframe.loc[mask, "timestamp"]
        if label == "sit":
            dataframe.loc[mask, [50000100, 50000101, 50000102]] = 0.0
            dataframe.loc[mask, 50000103] = 0.5
        elif label == "stand":
            dataframe.loc[mask, 50000100] = 1.0
            dataframe.loc[mask, [50000101, 50000102]] = 0.0
            dataframe.loc[mask, 50000103] = 1.0
        elif label == "standup":
            dataframe.loc[mask, 50000100] = np.interp(t, [start, end], [0.0, 1.0])
            dataframe.loc[mask, 50000101] = np.interp(
                t,
                [start, (start + end) / 2, end],
                [0.0, 0.3, 0.0],
            )
            dataframe.loc[mask, 50000102] = 0.0
            dataframe.loc[mask, 50000103] = np.interp(t, [start, end], [0.5, 1.0])
        elif label == "sitdown":
            dataframe.loc[mask, 50000100] = np.interp(t, [start, end], [1.0, 0.0])
            dataframe.loc[mask, 50000101] = np.interp(
                t,
                [start, (start + end) / 2, end],
                [0.0, 0.3, 0.0],
            )
            dataframe.loc[mask, 50000102] = 0.0
            dataframe.loc[mask, 50000103] = np.interp(t, [start, end], [1.0, 0.5])
        else:
            raise ValueError(f"Unsupported master_v5 action label: {label}")


def _apply_nearest_objects(
    dataframe: pd.DataFrame,
    patient_position: tuple[float, float],
    object_positions: dict[str, np.ndarray],
) -> None:
    position = np.array(patient_position)
    iv_position = _nearest_position(position, object_positions, "ivPole")
    wheelchair_position = _nearest_position(position, object_positions, "wheelchair")
    dataframe[50001000] = iv_position[0]
    dataframe[50001001] = iv_position[1]
    dataframe[50001010] = wheelchair_position[0]
    dataframe[50001011] = wheelchair_position[1]


def _nearest_position(
    position: np.ndarray,
    object_positions: dict[str, np.ndarray],
    object_type: str,
) -> np.ndarray:
    candidates = [
        object_position
        for object_id, object_position in object_positions.items()
        if object_type in object_id
    ]
    return min(candidates, key=lambda object_position: np.linalg.norm(object_position - position))


def _apply_nearest_wall(
    dataframe: pd.DataFrame,
    patient_position: tuple[float, float],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
) -> None:
    x, y = patient_position
    closest_wall = int(np.argmin([abs(xrange[0] - x), abs(yrange[0] - y), abs(xrange[1] - x), abs(yrange[1] - y)]))
    if closest_wall == 0:
        dataframe[50001020] = xrange[0]
        dataframe[50001021] = y
    elif closest_wall == 1:
        dataframe[50001020] = x
        dataframe[50001021] = yrange[0]
    elif closest_wall == 2:
        dataframe[50001020] = xrange[1]
        dataframe[50001021] = y
    else:
        dataframe[50001020] = x
        dataframe[50001021] = yrange[1]


def _apply_staff_trajectory(
    dataframes: dict[str, pd.DataFrame],
    positions: dict[str, tuple[float, float]],
) -> None:
    segments = (
        ("work", None, 0.0, 5.0),
        ("approach", "B", 5.0, 7.0),
        ("work", "B", 7.0, 9.0),
        ("leave", "B", 9.0, 10.0),
    )
    timestamp = next(iter(dataframes.values()))["timestamp"]
    for label, target, start, end in segments:
        mask = (timestamp >= start) & (timestamp <= end)
        if label == "approach" and target is not None:
            staff_x = np.interp(timestamp[mask], [start, end], [positions["NS"][0], positions[target][0]])
            staff_y = np.interp(timestamp[mask], [start, end], [positions["NS"][1], positions[target][1]])
        elif label == "work" and target is not None:
            staff_x = np.interp(timestamp[mask], [start, end], [positions[target][0], positions[target][0]])
            staff_y = np.interp(timestamp[mask], [start, end], [positions[target][1], positions[target][1]])
        elif label == "leave" and target is not None:
            staff_x = np.interp(timestamp[mask], [start, end], [positions[target][0], positions["NS"][0]])
            staff_y = np.interp(timestamp[mask], [start, end], [positions[target][1], positions["NS"][1]])
        else:
            staff_x = np.full(mask.sum(), positions["NS"][0])
            staff_y = np.full(mask.sum(), positions["NS"][1])
        for dataframe in dataframes.values():
            dataframe.loc[mask, 50001100] = staff_x
            dataframe.loc[mask, 50001101] = staff_y
    for dataframe in dataframes.values():
        dataframe[50001110] = dataframe[50001100].diff().values
        dataframe[50001111] = dataframe[50001101].diff().values


def _apply_background_difference(
    dataframes: dict[str, pd.DataFrame],
    actions: dict[str, tuple[tuple[str, float, float], ...]],
) -> None:
    values = {
        "sit": 0.1,
        "stand": 0.4,
        "standup": 0.7,
        "sitdown": 0.7,
    }
    for patient, action_segments in actions.items():
        dataframe = dataframes[patient]
        for label, start, end in action_segments:
            mask = (dataframe["timestamp"] >= start) & (dataframe["timestamp"] <= end)
            dataframe.loc[mask, 70000000] = values[label]


def _restore_legacy_dtypes(dataframe: pd.DataFrame) -> None:
    integer_columns = (
        50000001,
        50000011,
        50001000,
        50001001,
        50001010,
        50001011,
        50001020,
        50001021,
        60010000,
        60010001,
    )
    for column in integer_columns:
        values = dataframe[column]
        if values.notna().all() and np.allclose(values, values.astype(int)):
            dataframe[column] = values.astype(int)
