"""A one-person, one-timestamp feature record."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D


@dataclass(frozen=True)
class FeatureFrame:
    person_id: str
    time_s: float
    is_patient_label: str
    is_patient_confidence: float
    age_group_label: str
    age_confidence: float
    pose_features: PoseFeatures
    patient_position: Position2D
    nearest_iv_position: Position2D | None = None
    nearest_wheelchair_position: Position2D | None = None
    nearest_handrail_position: Position2D | None = None
    nearest_staff_position: Position2D | None = None
    nearest_staff_velocity: Velocity2D | None = None
    height_max: float | None = None
    action_label: str | None = None

