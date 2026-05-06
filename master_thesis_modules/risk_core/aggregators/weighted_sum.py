"""Weighted-sum aggregation with normalized weights."""

from master_thesis_modules.risk_core.factors.common import clip01


class WeightedSumAggregator:
    def __init__(self, weights: dict[int, float] | None = None) -> None:
        self.weights = weights or {}

    def aggregate(self, values: dict[int, float], default_weight: float = 1.0) -> float:
        if not values:
            return 0.0
        weighted_values = []
        total_weight = 0.0
        for node_id, value in values.items():
            weight = self.weights.get(node_id, default_weight)
            if weight <= 0.0:
                continue
            weighted_values.append(value * weight)
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return clip01(sum(weighted_values) / total_weight)
