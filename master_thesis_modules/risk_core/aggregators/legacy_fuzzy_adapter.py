"""Compatibility wrapper around the legacy fuzzy rule shape."""

from master_thesis_modules.risk_core.aggregators.fuzzy import LegacyLikeFuzzyAggregator


class LegacyFuzzyAdapter(LegacyLikeFuzzyAggregator):
    """The default implementation mirrors `fuzzy_reasoning_v5` without imports."""

