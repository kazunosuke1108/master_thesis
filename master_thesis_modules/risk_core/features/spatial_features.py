"""Spatial feature bundle around a patient at one timestamp."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D


@dataclass(frozen=True)
class SpatialFeatures:
    patient_position: Position2D
    nearest_iv_position: Position2D | None = None
    nearest_wheelchair_position: Position2D | None = None
    nearest_handrail_position: Position2D | None = None
    nearest_staff_position: Position2D | None = None
    nearest_staff_velocity: Velocity2D | None = None

