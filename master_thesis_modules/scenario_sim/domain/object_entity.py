"""Object state loaded from scenario YAML."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.features.position import Position2D


@dataclass(frozen=True)
class ObjectEntity:
    object_id: str
    object_type: str
    position: Position2D

