"""Four-dimensional pose features used by the thesis action-risk formula."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PoseFeatures:
    standing_degree: float
    trunk_tilt: float
    wrist_distance_from_hip: float
    ankle_spread: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.standing_degree,
            self.trunk_tilt,
            self.wrist_distance_from_hip,
            self.ankle_spread,
        )

    def finite_values(self) -> tuple[float, ...]:
        return tuple(v for v in self.as_tuple() if math.isfinite(v))

