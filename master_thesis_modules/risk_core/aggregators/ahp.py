"""AHP-compatible weighted aggregator."""

from master_thesis_modules.risk_core.aggregators.weighted_sum import WeightedSumAggregator


class AHPAggregator(WeightedSumAggregator):
    """Use explicit weights now; later this can load pairwise matrices."""


AhpAggregator = AHPAggregator

