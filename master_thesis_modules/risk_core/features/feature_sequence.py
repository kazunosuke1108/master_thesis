"""Time series of FeatureFrame objects for one person."""

from dataclasses import dataclass

import pandas as pd

from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame


@dataclass(frozen=True)
class FeatureFrameSequence:
    person_id: str
    frames: list[FeatureFrame]

    def timestamps(self) -> list[float]:
        return [frame.time_s for frame in self.frames]

    def to_feature_dataframe(self) -> pd.DataFrame:
        rows = []
        for frame in self.frames:
            rows.append(
                {
                    "timestamp": frame.time_s,
                    "person_id": frame.person_id,
                    "is_patient_label": frame.is_patient_label,
                    "age_group_label": frame.age_group_label,
                    "standing_degree": frame.pose_features.standing_degree,
                    "trunk_tilt": frame.pose_features.trunk_tilt,
                    "wrist_distance_from_hip": frame.pose_features.wrist_distance_from_hip,
                    "ankle_spread": frame.pose_features.ankle_spread,
                    "x": frame.patient_position.x,
                    "y": frame.patient_position.y,
                    "height_max": frame.height_max,
                }
            )
        return pd.DataFrame(rows)

