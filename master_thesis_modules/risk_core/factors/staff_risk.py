"""Staff-distance and watching-direction risk calculators."""

from __future__ import annotations

import math

from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D
from master_thesis_modules.risk_core.factors.common import (
    SPATIAL_NORMALIZATION_PARAM,
    clip01,
)
from master_thesis_modules.risk_core.schema import node_ids as ids


def staff_distance_risk(
    patient_position: Position2D,
    staff_position: Position2D | None,
    normalization: float = SPATIAL_NORMALIZATION_PARAM,
) -> float:
    if staff_position is None:
        return 1.0
    return clip01(patient_position.distance_to(staff_position) / normalization)


def staff_not_watching_risk(
    patient_position: Position2D,
    staff_position: Position2D | None,
    staff_velocity: Velocity2D | None,
) -> float:

    # スタッフの方向が取れなかった場合、リスクを中立にする
    if staff_position is None or staff_velocity is None:
        return 1.0
    relative_x = patient_position.x - staff_position.x
    relative_y = patient_position.y - staff_position.y
    relative_norm = math.hypot(relative_x, relative_y)
    velocity_norm = staff_velocity.norm
    norm_product = relative_norm * velocity_norm
    if norm_product == 0.0 or not math.isfinite(norm_product):
        # master_v5 returned NaN here. The new core uses neutral risk to keep
        # batch evaluation finite while making this behavior explicit.
        return 0.5

    # スタッフの方向が取れなかった場合、リスクをnanで埋め、のちに0で埋める（ゼロにするのはまずいと思い、不採用）
    # if staff_position is None:
    #     return 1.0
    # if staff_velocity is None:
    #     return math.nan

    # relative_x = patient_position.x - staff_position.x
    # relative_y = patient_position.y - staff_position.y
    # relative_norm = math.hypot(relative_x, relative_y)
    # velocity_norm = staff_velocity.norm
    # norm_product = relative_norm * velocity_norm
    # if norm_product == 0.0 or not math.isfinite(norm_product):
    #     return math.nan

    cos_theta = (
        relative_x * staff_velocity.vx + relative_y * staff_velocity.vy
    ) / norm_product
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return clip01(1.0 - (cos_theta / 2.0 + 0.5))


class StaffRiskCalculator:
    def __init__(self, normalization: float = SPATIAL_NORMALIZATION_PARAM) -> None:
        self.normalization = normalization

    def calculate(self, frame: FeatureFrame) -> dict[int, float]:
        return {
            ids.STAFF_DISTANCE_RISK: staff_distance_risk(
                frame.patient_position,
                frame.nearest_staff_position,
                self.normalization,
            ),
            ids.STAFF_NOT_WATCHING_RISK: staff_not_watching_risk(
                frame.patient_position,
                frame.nearest_staff_position,
                frame.nearest_staff_velocity,
            ),
        }
