"""Small helpers exposing the legacy schema in the new package."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.schema import node_ids as ids
from master_thesis_modules.risk_core.schema.evaluation_order import EvaluationStep


@dataclass(frozen=True)
class LegacyNodeSpec:
    code: int
    semantic_name: str
    layer: int


LEGACY_NODE_SPECS = {
    ids.TOTAL_RISK: LegacyNodeSpec(ids.TOTAL_RISK, "total_risk", 1),
    ids.INTERNAL_RISK: LegacyNodeSpec(ids.INTERNAL_RISK, "internal_risk", 2),
    ids.EXTERNAL_RISK: LegacyNodeSpec(ids.EXTERNAL_RISK, "external_risk", 2),
    ids.INTERNAL_STATIC_RISK: LegacyNodeSpec(ids.INTERNAL_STATIC_RISK, "internal_static_risk", 3),
    ids.INTERNAL_DYNAMIC_RISK: LegacyNodeSpec(ids.INTERNAL_DYNAMIC_RISK, "internal_dynamic_risk", 3),
    ids.EXTERNAL_STATIC_RISK: LegacyNodeSpec(ids.EXTERNAL_STATIC_RISK, "external_static_risk", 3),
    ids.EXTERNAL_DYNAMIC_RISK: LegacyNodeSpec(ids.EXTERNAL_DYNAMIC_RISK, "external_dynamic_risk", 3),
}


def legacy_evaluation_steps() -> tuple[EvaluationStep, ...]:
    from master_thesis_modules.risk_core.schema.evaluation_order import LEGACY_EVALUATION_ORDER

    return LEGACY_EVALUATION_ORDER

