"""Action risk based on four-dimensional pose-template similarity."""

from __future__ import annotations

from dataclasses import dataclass
import math

from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.factors.action_templates import (
    DEFAULT_ACTION_TEMPLATES,
    RiskyMotionTemplate,
)
from master_thesis_modules.risk_core.factors.common import clip01, safe_mean
from master_thesis_modules.risk_core.schema import node_ids as ids


@dataclass(frozen=True)
class HeightStandingRiskConfig:
    sigmoid_gain: float = 5.0
    sigmoid_center: float = 1.0


class ActionRiskCalculator:
    def __init__(
        self,
        templates: tuple[RiskyMotionTemplate, ...] = DEFAULT_ACTION_TEMPLATES,
        height_config: HeightStandingRiskConfig | None = None,
    ) -> None:
        self.templates = templates
        self.height_config = height_config or HeightStandingRiskConfig()

    def calculate(
        self,
        pose_features: PoseFeatures,
        height_max: float | None = None,
    ) -> dict[int, float]:
        risks = {
            template.node_id: pose_similarity(pose_features, template.reference_pose)
            for template in self.templates
        }
        if height_max is not None and math.isfinite(height_max):
            risks[ids.STANDING_RISK] = standing_risk_from_height(
                height_max,
                self.height_config,
            )
        return {node_id: clip01(value) for node_id, value in risks.items()}


def pose_distance(observed: PoseFeatures, reference: PoseFeatures) -> float:
    """Mean absolute difference, ignoring NaNs in the reference template."""

    diffs = []
    for observed_value, reference_value in zip(observed.as_tuple(), reference.as_tuple()):
        if not math.isfinite(reference_value) or not math.isfinite(observed_value):
            continue
        diffs.append(abs(reference_value - observed_value))
    return safe_mean(diffs)


def pose_similarity(observed: PoseFeatures, reference: PoseFeatures) -> float:
    similarity = 1.0 - clip01(pose_distance(observed, reference))
    return clip01(similarity**4)


def standing_risk_from_height(
    height_max: float,
    config: HeightStandingRiskConfig | None = None,
) -> float:
    config = config or HeightStandingRiskConfig()
    exponent = -config.sigmoid_gain * (height_max - config.sigmoid_center)
    return clip01(1.0 / (1.0 + math.exp(exponent)))
