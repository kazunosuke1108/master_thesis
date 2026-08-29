from master_thesis_modules.risk_core.aggregators.weighted_sum import (
    WeightedMaxAggregator,
    WeightedSumAggregator,
)
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.schema import node_ids as ids


def test_weighted_max_uses_largest_weighted_action_risk():
    values = {
        ids.STANDING_RISK: 0.5,
        ids.WHEELCHAIR_BRAKE_RELEASE_RISK: 1.0,
    }
    weights = {
        ids.STANDING_RISK: 0.2,
        ids.WHEELCHAIR_BRAKE_RELEASE_RISK: 0.8,
    }

    assert WeightedMaxAggregator(weights).aggregate(values) == 0.8


def test_action_aggregation_option_changes_internal_dynamic_risk():
    values = {
        ids.STANDING_RISK: 0.5,
        ids.WHEELCHAIR_BRAKE_RELEASE_RISK: 1.0,
    }
    weights = {
        ids.STANDING_RISK: 0.2,
        ids.WHEELCHAIR_BRAKE_RELEASE_RISK: 0.8,
    }

    weighted_sum = RiskEngine(
        RiskConfig(action_weights=weights, action_aggregation="weighted_sum")
    )._aggregate_action_risks(values)
    weighted_max = RiskEngine(
        RiskConfig(action_weights=weights, action_aggregation="weighted_max")
    )._aggregate_action_risks(values)

    assert weighted_sum == WeightedSumAggregator(weights).aggregate(values)
    assert weighted_max == WeightedMaxAggregator(weights).aggregate(values)
    assert weighted_max < weighted_sum
