"""REQ-001: independently calculated internal fuzzy values and engine wiring."""

import pytest

from master_thesis_modules.risk_core.aggregators.fuzzy import LegacyLikeFuzzyAggregator
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.features.position import Position2D
from master_thesis_modules.risk_core.schema import node_ids as ids


@pytest.mark.parametrize(
    "static,dynamic,expected",
    [(1.0, 1.0, 0.9), (1.0, 0.0, 0.7), (0.0, 1.0, 0.3), (0.0, 0.0, 0.1),
     (0.25, 0.6, 0.37)],
)
def test_internal_four_corners_and_independent_intermediate(static, dynamic, expected):
    # At (.25,.6), HH/HL/LH/LL = .15/.10/.45/.30; weighted sum = .37.
    result = LegacyLikeFuzzyAggregator(ids.INTERNAL_RISK, (0.9, 0.7, 0.3, 0.1)).aggregate(
        {ids.INTERNAL_STATIC_RISK: static, ids.INTERNAL_DYNAMIC_RISK: dynamic}
    )
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("static,dynamic,expected", [(1, 1, 1), (1, 0, 1), (0, 1, .5), (0, 0, 0), (.25, .6, .475)])
def test_unspecified_internal_uses_legacy_default(static, dynamic, expected):
    assert LegacyLikeFuzzyAggregator(ids.INTERNAL_RISK).aggregate(
        {ids.INTERNAL_STATIC_RISK: static, ids.INTERNAL_DYNAMIC_RISK: dynamic}
    ) == pytest.approx(expected)


@pytest.mark.parametrize("rules", [(), (0.1,), (0., .2, .3), (0., .2, .3, .4, .5),
    (float("nan"), .2, .3, .4), (float("inf"), .2, .3, .4), (-.1, .2, .3, .4),
    (1.1, .2, .3, .4), ("invalid", .2, .3, .4)])
def test_direct_aggregator_rejects_invalid_internal_consequents(rules):
    with pytest.raises(ValueError):
        LegacyLikeFuzzyAggregator(ids.INTERNAL_RISK, rules).aggregate(
            {ids.INTERNAL_STATIC_RISK: .25, ids.INTERNAL_DYNAMIC_RISK: .6}
        )


@pytest.mark.parametrize("model", ["action_only", "patient_context", "action_attribute", "spatial_context"])
@pytest.mark.parametrize("fuzzy", [True, False])
def test_engine_rejects_invalid_direct_internal_config_for_every_model(model, fuzzy):
    with pytest.raises(ValueError):
        RiskEngine(RiskConfig(model_type=model, use_legacy_like_fuzzy=fuzzy,
            fuzzy_rule_results={ids.INTERNAL_RISK: (float("nan"), 0., 0., 0.)}))


def _frame():
    return FeatureFrame(person_id="test", time_s=0., is_patient_label="yes",
        is_patient_confidence=1., age_group_label="old", age_confidence=1.,
        pose_features=PoseFeatures(standing_degree=.8, trunk_tilt=.2,
            wrist_distance_from_hip=.4, ankle_spread=.7), patient_position=Position2D(0., 0.))


@pytest.mark.parametrize("model", ["action_only", "patient_context", "action_attribute", "spatial_context"])
def test_staff_internal_results_reach_each_model_and_action_only_total_is_preserved(model):
    frame = _frame()
    low = RiskEngine(RiskConfig(model_type=model, fuzzy_rule_results={ids.INTERNAL_RISK: (0.,) * 4})).evaluate(frame)
    high = RiskEngine(RiskConfig(model_type=model, fuzzy_rule_results={ids.INTERNAL_RISK: (1.,) * 4})).evaluate(frame)
    assert low.upper_risks[ids.INTERNAL_RISK] == pytest.approx(0.)
    assert high.upper_risks[ids.INTERNAL_RISK] == pytest.approx(1.)
    if model == "action_only":
        assert low.total_risk == high.total_risk == low.upper_risks[ids.INTERNAL_DYNAMIC_RISK]
    elif model in {"patient_context", "action_attribute"}:
        assert low.total_risk == pytest.approx(0.)
        assert high.total_risk == pytest.approx(1.)
    else:
        # Unchanged total defaults HH=1, HL=LH=.5, LL=0 give (internal+external)/2.
        assert low.total_risk == pytest.approx(low.upper_risks[ids.EXTERNAL_RISK] / 2.)
        assert high.total_risk == pytest.approx((1. + high.upper_risks[ids.EXTERNAL_RISK]) / 2.)


@pytest.mark.parametrize("model", ["action_only", "patient_context", "action_attribute", "spatial_context"])
def test_fuzzy_disabled_keeps_original_weighted_internal(model):
    result = RiskEngine(RiskConfig(model_type=model, use_legacy_like_fuzzy=False,
        fuzzy_rule_results={ids.INTERNAL_RISK: (1.,) * 4})).evaluate(_frame())
    assert result.upper_risks[ids.INTERNAL_RISK] == pytest.approx(
        .1 * result.upper_risks[ids.INTERNAL_STATIC_RISK]
        + .9 * result.upper_risks[ids.INTERNAL_DYNAMIC_RISK]
    )
