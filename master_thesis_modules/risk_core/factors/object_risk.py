"""Object-distance risk calculators compatible with ``master_v5.py``."""

from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.features.position import Position2D
from master_thesis_modules.risk_core.factors.common import (
    SPATIAL_NORMALIZATION_PARAM,
    clip01,
)
from master_thesis_modules.risk_core.schema import node_ids as ids


def near_object_risk(
    patient_position: Position2D,
    object_position: Position2D | None,
    normalization: float = SPATIAL_NORMALIZATION_PARAM,
) -> float:
    if object_position is None:
        return 0.0
    return clip01(1.0 - patient_position.distance_to(object_position) / normalization)


def far_from_object_risk(
    patient_position: Position2D,
    object_position: Position2D | None,
    normalization: float = SPATIAL_NORMALIZATION_PARAM,
) -> float:
    if object_position is None:
        return 0.0
    return clip01(patient_position.distance_to(object_position) / normalization)


class ObjectRiskCalculator:
    def __init__(self, normalization: float = SPATIAL_NORMALIZATION_PARAM) -> None:
        self.normalization = normalization

    def calculate(self, frame: FeatureFrame) -> dict[int, float]:
        patient = frame.patient_position
        return {
            ids.IV_POLE_RISK: near_object_risk(
                patient,
                frame.nearest_iv_position,
                self.normalization,
            ),
            ids.WHEELCHAIR_RISK: near_object_risk(
                patient,
                frame.nearest_wheelchair_position,
                self.normalization,
            ),
            ids.HANDRAIL_DISTANCE_RISK: far_from_object_risk(
                patient,
                frame.nearest_handrail_position,
                self.normalization,
            ),
        }
