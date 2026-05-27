"""Position and velocity primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Position2D:
    x: float
    y: float

    def distance_to(self, other: "Position2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Velocity2D:
    vx: float
    vy: float

    @property
    def norm(self) -> float:
        return math.hypot(self.vx, self.vy)

