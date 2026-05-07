"""Fuzzy-compatible aggregators."""

from master_thesis_modules.risk_core.factors.common import clip01
from master_thesis_modules.risk_core.schema import node_ids as ids


class MaxMeanFuzzyAggregator:
    """Small deterministic approximation: blend mean and max risk."""

    def __init__(self, max_weight: float = 0.35) -> None:
        self.max_weight = max_weight

    def aggregate(self, values: dict[int, float]) -> float:
        if not values:
            return 0.0
        mean_value = sum(values.values()) / len(values)
        max_value = max(values.values())
        return clip01((1.0 - self.max_weight) * mean_value + self.max_weight * max_value)


class LegacyLikeFuzzyAggregator:
    """Standalone implementation of scripts/fuzzy/fuzzy_reasoning_v5.py rules."""

    DEFAULT_RULES = {
        ids.EXTERNAL_DYNAMIC_RISK: (
            ({ids.STAFF_DISTANCE_RISK: "high", ids.STAFF_NOT_WATCHING_RISK: "high"}, "high"),
            ({ids.STAFF_DISTANCE_RISK: "low", ids.STAFF_NOT_WATCHING_RISK: "high"}, "high"),
            ({ids.STAFF_DISTANCE_RISK: "high", ids.STAFF_NOT_WATCHING_RISK: "low"}, "middle"),
            ({ids.STAFF_DISTANCE_RISK: "low", ids.STAFF_NOT_WATCHING_RISK: "low"}, "low"),
        ),
        ids.EXTERNAL_RISK: (
            ({ids.EXTERNAL_STATIC_RISK: "high", ids.EXTERNAL_DYNAMIC_RISK: "high"}, "high"),
            ({ids.EXTERNAL_STATIC_RISK: "high", ids.EXTERNAL_DYNAMIC_RISK: "low"}, "middle"),
            ({ids.EXTERNAL_STATIC_RISK: "low", ids.EXTERNAL_DYNAMIC_RISK: "high"}, "middle"),
            ({ids.EXTERNAL_STATIC_RISK: "low", ids.EXTERNAL_DYNAMIC_RISK: "low"}, "low"),
        ),
        ids.TOTAL_RISK: (
            ({ids.INTERNAL_RISK: "high", ids.EXTERNAL_RISK: "high"}, "high"),
            ({ids.INTERNAL_RISK: "high", ids.EXTERNAL_RISK: "low"}, "middle"),
            ({ids.INTERNAL_RISK: "low", ids.EXTERNAL_RISK: "high"}, "middle"),
            ({ids.INTERNAL_RISK: "low", ids.EXTERNAL_RISK: "low"}, "low"),
        ),
    }

    def __init__(
        self,
        output_node: int,
        custom_results: tuple[str | float, ...] | None = None,
    ) -> None:
        self.output_node = output_node
        self.custom_results = custom_results

    def aggregate(self, inputs: dict[int, float]) -> float:
        rules = self.DEFAULT_RULES.get(self.output_node)
        if rules is None:
            return MaxMeanFuzzyAggregator().aggregate(inputs)
        numerator = 0.0
        denominator = 0.0
        for index, (conditions, result) in enumerate(rules):
            if self.custom_results is not None and index < len(self.custom_results):
                result = self.custom_results[index]
            height = 1.0
            for node_id, membership_name in conditions.items():
                height *= _membership(clip01(inputs.get(node_id, 0.0)), membership_name)
            numerator += _peak(result) * height
            denominator += height
        if denominator == 0.0:
            return 0.0
        return clip01(numerator / denominator)


def _membership(value: float, membership_name: str) -> float:
    if membership_name == "high":
        return value
    if membership_name == "middle":
        return 2.0 * value if value <= 0.5 else -2.0 * value + 2.0
    if membership_name == "low":
        return 1.0 - value
    raise ValueError(f"Unknown membership: {membership_name}")


def _peak(result: str | float) -> float:
    if result == "low":
        return 0.0
    if result == "middle":
        return 0.5
    if result == "high":
        return 1.0
    return float(result)
