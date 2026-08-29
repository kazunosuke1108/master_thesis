"""Extract top contributing factor risks."""

from dataclasses import dataclass

from master_thesis_modules.risk_core.schema.node_labels import NODE_LABELS


@dataclass(frozen=True)
class FactorContribution:
    node_id: int
    label: str
    value: float


def top_factors(
    factor_risks: dict[int, float],
    top_n: int = 3,
    threshold: float = 0.25,
) -> list[tuple[int, float]]:
    return [
        (node_id, value)
        for node_id, value in sorted(
            factor_risks.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if value >= threshold
    ][:top_n]


def top_factor_contributions(
    factor_risks: dict[int, float],
    top_n: int = 3,
    threshold: float = 0.25,
) -> list[FactorContribution]:
    return [
        FactorContribution(node_id, NODE_LABELS.get(node_id, str(node_id)), value)
        for node_id, value in top_factors(factor_risks, top_n=top_n, threshold=threshold)
    ]
