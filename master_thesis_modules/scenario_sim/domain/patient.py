"""Patient state loaded from scenario YAML."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.features.position import Position2D


@dataclass(frozen=True)
class Patient:
    person_id: str
    is_patient_label: str
    age_group_label: str
    position: Position2D
    action_label: str | None = None
    height_max: float | None = None
    pose_features: PoseFeatures | None = None
