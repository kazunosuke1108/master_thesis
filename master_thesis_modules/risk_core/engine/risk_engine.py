"""MVP risk engine that evaluates one FeatureFrame at a time."""

from master_thesis_modules.risk_core.aggregators.fuzzy import LegacyLikeFuzzyAggregator
from master_thesis_modules.risk_core.aggregators.weighted_sum import WeightedSumAggregator
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.engine.risk_result import RiskResult
from master_thesis_modules.risk_core.explanation.explanation_generator import (
    ExplanationGenerator,
)
from master_thesis_modules.risk_core.factors.action_risk import (
    ActionRiskCalculator,
    HeightStandingRiskConfig,
)
from master_thesis_modules.risk_core.factors.attribution_risk import (
    age_attribute_risk,
    patient_attribute_risk,
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

        patient_risk = patient_attribute_risk(
            frame.is_patient_label,
            frame.is_patient_confidence,
        )
        age_risk = age_attribute_risk(frame.age_group_label, frame.age_confidence)
        if self.config.model_type == "action_only":
            patient_risk = 0.0
            age_risk = 0.0
        internal_static = clip01((patient_risk + age_risk) / 2.0)
        internal_dynamic = WeightedSumAggregator(
            self.config.action_weights
        ).aggregate({node_id: factor_risks[node_id] for node_id in ids.ACTION_RISK_NODES})
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
