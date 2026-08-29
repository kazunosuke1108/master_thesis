"""Staff state loaded from scenario YAML."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D


@dataclass(frozen=True)
class Staff:
    staff_id: str
    position: Position2D
    velocity: Velocity2D

