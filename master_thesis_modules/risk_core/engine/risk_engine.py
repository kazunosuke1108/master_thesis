"""MVP risk engine that evaluates one FeatureFrame at a time."""

import math

from master_thesis_modules.risk_core.aggregators.fuzzy import LegacyLikeFuzzyAggregator
from master_thesis_modules.risk_core.aggregators.weighted_sum import (
    WeightedMaxAggregator,
    WeightedSumAggregator,
)
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.explanation.explanation_generator import (
    ExplanationGenerator,
)
from master_thesis_modules.risk_core.factors.action_risk import (
    ActionRiskCalculator,
    HeightStandingRiskConfig,
)
from master_thesis_modules.risk_core.factors.common import clip01
from master_thesis_modules.risk_core.factors.object_risk import ObjectRiskCalculator
from master_thesis_modules.risk_core.factors.staff_risk import StaffRiskCalculator
from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.schema import node_ids as ids


class RiskEngine:
    def __init__(
        self,
        config: RiskConfig | None = None,
        explanation_generator: ExplanationGenerator | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.action_calculator = ActionRiskCalculator(
            height_config=HeightStandingRiskConfig(
                sigmoid_gain=self.config.height_sigmoid_gain,
                sigmoid_center=self.config.height_sigmoid_center,
            )
        )
        self.object_calculator = ObjectRiskCalculator(self.config.room_diagonal_m)
        self.staff_calculator = StaffRiskCalculator(self.config.room_diagonal_m)
        self.explanation_generator = explanation_generator or ExplanationGenerator()

    def evaluate(self, frame: FeatureFrame) -> RiskResult:
        factor_risks = {}
        factor_risks.update(
            self.action_calculator.calculate(frame.pose_features, frame.height_max)
        )
        factor_risks.update(self.object_calculator.calculate(frame))
        factor_risks.update(self.staff_calculator.calculate(frame))
        if self.config.model_type == "action_only":
            factor_risks = {
                node_id: (factor_risks[node_id] if node_id in ids.ACTION_RISK_NODES else 0.0)
                for node_id in factor_risks
            }
        elif self.config.model_type == "action_attribute":
            factor_risks = {
                node_id: (
                    factor_risks[node_id]
                    if node_id in ids.ACTION_RISK_NODES
                    else 0.0
                )
                for node_id in factor_risks
            }

        patient_risk = patient_attribute_tfn(frame.is_patient_label)
        age_risk = age_attribute_tfn(frame.age_group_label)
        if self.config.model_type == "action_only":
            patient_risk = (0.0, 0.0, 0.0)
            age_risk = (0.0, 0.0, 0.0)
        internal_static = legacy_fuzzy_multiply(patient_risk, age_risk)
        action_inputs = {node_id: factor_risks[node_id] for node_id in ids.ACTION_RISK_NODES}
        internal_dynamic = self._aggregate_action_risks(action_inputs)
        external_static = WeightedSumAggregator(
            self.config.object_weights
        ).aggregate({node_id: factor_risks[node_id] for node_id in ids.OBJECT_RISK_NODES})
        staff_inputs = {node_id: factor_risks[node_id] for node_id in ids.STAFF_RISK_NODES}
        if self.config.use_legacy_like_fuzzy:
            external_dynamic = LegacyLikeFuzzyAggregator(
                ids.EXTERNAL_DYNAMIC_RISK,
                self.config.fuzzy_rule_results.get(ids.EXTERNAL_DYNAMIC_RISK),
            ).aggregate(staff_inputs)
        else:
            external_dynamic = WeightedSumAggregator(self.config.staff_weights).aggregate(
                staff_inputs
            )

        internal = clip01(
            self.config.internal_static_weight * internal_static
            + self.config.internal_dynamic_weight * internal_dynamic
        )
        external_inputs = {
            ids.EXTERNAL_STATIC_RISK: external_static,
            ids.EXTERNAL_DYNAMIC_RISK: external_dynamic,
        }
        if self.config.use_legacy_like_fuzzy:
            external = LegacyLikeFuzzyAggregator(
                ids.EXTERNAL_RISK,
                self.config.fuzzy_rule_results.get(ids.EXTERNAL_RISK),
            ).aggregate(external_inputs)
        else:
            external = clip01((external_static + external_dynamic) / 2.0)

        weighted_factor_total = WeightedSumAggregator(
            self.config.total_factor_weights
        ).aggregate(factor_risks)
        hierarchical_total = clip01(
            self.config.internal_weight * internal
            + self.config.external_weight * external
        )
        if self.config.model_type == "action_only":
            total = internal_dynamic
        elif self.config.model_type == "action_attribute":
            total = internal
        elif self.config.use_legacy_like_fuzzy:
            total = LegacyLikeFuzzyAggregator(
                ids.TOTAL_RISK,
                self.config.fuzzy_rule_results.get(ids.TOTAL_RISK),
            ).aggregate(
                {ids.INTERNAL_RISK: internal, ids.EXTERNAL_RISK: external},
            )
        else:
            total = clip01((weighted_factor_total + hierarchical_total) / 2.0)

        upper_risks = {
            ids.PATIENT_ATTRIBUTE_RISK: patient_risk,
            ids.AGE_ATTRIBUTE_RISK: age_risk,
            ids.INTERNAL_STATIC_RISK: internal_static,
            ids.INTERNAL_DYNAMIC_RISK: internal_dynamic,
            ids.EXTERNAL_STATIC_RISK: external_static,
            ids.EXTERNAL_DYNAMIC_RISK: external_dynamic,
            ids.INTERNAL_RISK: internal,
            ids.EXTERNAL_RISK: external,
            ids.TOTAL_RISK: total,
        }
        explanation = self.explanation_generator.generate(frame.person_id, factor_risks)
        return RiskResult(
            person_id=frame.person_id,
            time_s=frame.time_s,
            factor_risks=factor_risks,
            upper_risks=upper_risks,
            total_risk=total,
            explanation=explanation,
        )

    def _aggregate_action_risks(self, action_inputs: dict[int, float]) -> float:
        if self.config.action_aggregation == "weighted_sum":
            return WeightedSumAggregator(self.config.action_weights).aggregate(action_inputs)
        if self.config.action_aggregation == "weighted_max":
            return WeightedMaxAggregator(self.config.action_weights).aggregate(action_inputs)
        raise ValueError(f"Unknown action_aggregation: {self.config.action_aggregation}")


def patient_attribute_tfn(label: str) -> tuple[float, float, float]:
    if label == "yes":
        return (0.4, 0.7, 1.0)
    if label == "no":
        return (0.0, 0.3, 0.6)
    return (math.nan, math.nan, math.nan)


def age_attribute_tfn(label: str) -> tuple[float, float, float]:
    if label == "young":
        return (0.0, 0.25, 0.5)
    if label == "middle":
        return (0.25, 0.5, 0.75)
    if label == "old":
        return (0.5, 0.75, 1.0)
    return (math.nan, math.nan, math.nan)


def legacy_fuzzy_multiply(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    if not all(math.isfinite(value) for value in (*first, *second)):
        return 0.0
    left, right = (first, second) if first[1] < second[1] else (second, first)
    denominator = (right[1] - right[0]) + (left[2] - left[1])
    if denominator == 0.0:
        return 0.0
    x_cross = (
        left[2] * (right[1] - right[0])
        + right[0] * (left[2] - left[1])
    ) / denominator
    return clip01((left[2] + right[0] + x_cross) / 3.0)
