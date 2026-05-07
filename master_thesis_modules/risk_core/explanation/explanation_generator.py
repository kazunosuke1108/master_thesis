"""Generate short deterministic explanations from factor risks."""

from master_thesis_modules.risk_core.explanation.factor_extractor import (
    top_factor_contributions,
    top_factors,
)
from master_thesis_modules.risk_core.explanation.templates import REASON_TEMPLATES


class ExplanationGenerator:
    def generate(self, person_id: str, factor_risks: dict[int, float]) -> str:
        factors = top_factors(factor_risks)
        if not factors:
            return f"{person_id}さんは、突出した危険要因が少ないため危険度は低めです。"

        reasons = [REASON_TEMPLATES.get(node_id, str(node_id)) for node_id, _ in factors]
        if len(reasons) == 1:
            reason_text = reasons[0]
        else:
            reason_text = "、".join(reasons[:-1]) + "、および" + reasons[-1]
        return f"{person_id}さんは、{reason_text}ため危険度が高く評価されました。"

    def structured(self, person_id: str, factor_risks: dict[int, float]) -> dict[str, object]:
        factors = top_factor_contributions(factor_risks)
        return {
            "person_id": person_id,
            "top_factors": [
                {"node_id": item.node_id, "label": item.label, "value": item.value}
                for item in factors
            ],
            "message": self.generate(person_id, factor_risks),
        }
