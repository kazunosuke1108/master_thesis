"""REQ-004 boundary validation and personal-rule wiring, without private data."""

from copy import deepcopy

import numpy as np
import pytest

from master_thesis_modules.risk_core.aggregators.fuzzy import LegacyLikeFuzzyAggregator
from master_thesis_modules.risk_core.aggregators.legacy_ahp_adapter import load_ahp_weights_from_matrix_csv
from master_thesis_modules.risk_core.engine.personalized_config import (
    make_personalized_risk_config,
    validate_personalized_knowledge,
)
from master_thesis_modules.risk_core.engine.risk_config import RiskConfig
from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
from master_thesis_modules.risk_core.features.feature_frame import FeatureFrame
from master_thesis_modules.risk_core.features.pose_features import PoseFeatures
from master_thesis_modules.risk_core.features.position import Position2D, Velocity2D
from master_thesis_modules.risk_core.schema import node_ids as ids


@pytest.fixture
def knowledge():
    # S01 synthetic pilot matrices; no real-staff identifying data.
    upper = [
        [1, 3, 4, .5, 5, 4, 7],
        [0, 1, 2, .2, 3, 2, 5],
        [0, 0, 1, 1 / 6, 2, 2, 4],
        [0, 0, 0, 1, 8, 7, 9],
        [0, 0, 0, 0, 1, .5, 2],
        [0, 0, 0, 0, 0, 1, 3],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    for i in range(7):
        for j in range(i):
            upper[i][j] = 1 / upper[j][i]
    return {
        "respondent_id": "S01",
        "action_pairwise": upper,
        "object_pairwise": [[1, .5, 2], [2, 1, 3], [.5, 1 / 3, 1]],
        "fuzzy_answers": {
            "internal": [5, 3, 5, 1],
            "watching": [5, 4, 2, 1],
            "external": [4, 3, 3, 1],
            "total": [5, 5, 3, 1],
        },
        "provenance": {"synthetic": True},
    }


def test_preserves_s01_saved_ahp_weights_and_legacy_method(knowledge, tmp_path):
    config = make_personalized_risk_config(knowledge)
    # Expected values recorded in REQ-003 S01 saved configs, not recomputed here.
    assert list(config.action_weights.values()) == pytest.approx([
        .24893707192707148, .11637254956747707, .08229835938306326,
        .41935304495132536, .04315838084099919, .06272777628543276,
        .02715281704463081,
    ], abs=1e-14)
    assert config.object_weights == pytest.approx({
        ids.IV_POLE_RISK: .2969613312124974,
        ids.WHEELCHAIR_RISK: .5396145502210746,
        ids.HANDRAIL_DISTANCE_RISK: .16342411856642794,
    }, abs=1e-14)
    for field, weights in (
        ("action_pairwise", config.action_weights),
        ("object_pairwise", config.object_weights),
    ):
        path = tmp_path / f"{field}.csv"
        np.savetxt(path, knowledge[field], delimiter=",")
        assert list(weights.values()) == pytest.approx(
            load_ahp_weights_from_matrix_csv(path), abs=1e-14
        )


@pytest.mark.parametrize("group,node,inputs", [
    ("internal", ids.INTERNAL_RISK, (ids.INTERNAL_STATIC_RISK, ids.INTERNAL_DYNAMIC_RISK)),
    ("watching", ids.EXTERNAL_DYNAMIC_RISK, (ids.STAFF_DISTANCE_RISK, ids.STAFF_NOT_WATCHING_RISK)),
    ("external", ids.EXTERNAL_RISK, (ids.EXTERNAL_STATIC_RISK, ids.EXTERNAL_DYNAMIC_RISK)),
    ("total", ids.TOTAL_RISK, (ids.INTERNAL_RISK, ids.EXTERNAL_RISK)),
])
def test_asymmetric_consequents_at_all_semantic_corners(knowledge, group, node, inputs):
    knowledge["fuzzy_answers"][group] = [5, 4, 2, 1]
    config = make_personalized_risk_config(knowledge)
    aggregator = LegacyLikeFuzzyAggregator(node, config.fuzzy_rule_results[node])
    for (first, second), expected in zip(((1, 1), (1, 0), (0, 1), (0, 0)), (1, .75, .25, 0)):
        assert aggregator.aggregate(dict(zip(inputs, (first, second)))) == expected
    # Also verify interpolation rather than just named corner lookup.
    assert aggregator.aggregate(dict(zip(inputs, (.2, .7)))) == pytest.approx(.325)


@pytest.mark.parametrize("group", ["internal", "watching", "external", "total"])
def test_missing_group_is_error_in_both_entrypoints(knowledge, group):
    del knowledge["fuzzy_answers"][group]
    for entrypoint in (validate_personalized_knowledge, make_personalized_risk_config):
        with pytest.raises(ValueError, match="requires exactly"):
            entrypoint(knowledge)


def test_extra_group_is_error(knowledge):
    knowledge["fuzzy_answers"]["shared"] = [5, 4, 2, 1]
    with pytest.raises(ValueError, match="requires exactly"):
        make_personalized_risk_config(knowledge)


@pytest.mark.parametrize("field", ["respondent_id", "action_pairwise", "object_pairwise", "fuzzy_answers"])
def test_missing_required_top_level_field_is_error(knowledge, field):
    del knowledge[field]
    with pytest.raises(ValueError):
        make_personalized_risk_config(knowledge)


@pytest.mark.parametrize("bad", [None, [], "S01", 1])
def test_nondictionary_knowledge_is_error(bad):
    with pytest.raises(ValueError, match="dictionary"):
        make_personalized_risk_config(bad)


def test_float_precision_reciprocity_tolerance_without_mutation(knowledge):
    knowledge["object_pairwise"][0][1] = 3
    knowledge["object_pairwise"][1][0] = round(1 / 3, 8)
    original = deepcopy(knowledge)
    make_personalized_risk_config(knowledge)
    assert knowledge == original
    knowledge["object_pairwise"][1][0] = .333
    with pytest.raises(ValueError, match="reciprocal"):
        make_personalized_risk_config(knowledge)


@pytest.mark.parametrize("bad", [True, False, 0, 6, -1, 4.5, 4.0, "4", None, float("nan"), float("inf"), 10**1000])
@pytest.mark.parametrize("group", ["internal", "watching", "external", "total"])
def test_invalid_scores_never_reach_engine(knowledge, group, bad):
    knowledge["fuzzy_answers"][group][1] = bad
    with pytest.raises(ValueError):
        make_personalized_risk_config(knowledge)


@pytest.mark.parametrize("field", ["action_pairwise", "object_pairwise"])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 10, .01, "2", None, float("nan"), float("inf"), 10**1000])
def test_invalid_ahp_values_are_rejected(knowledge, field, bad):
    knowledge[field][0][1] = bad
    with pytest.raises(ValueError):
        make_personalized_risk_config(knowledge)


@pytest.mark.parametrize("field", ["action_pairwise", "object_pairwise"])
def test_incomplete_or_nonreciprocal_matrix_is_not_repaired(knowledge, field):
    for invalid in (knowledge[field][:-1], [row[:-1] for row in knowledge[field]]):
        broken = deepcopy(knowledge)
        broken[field] = invalid
        with pytest.raises(ValueError):
            make_personalized_risk_config(broken)
    knowledge[field][0][1] = 1
    with pytest.raises(ValueError, match="reciprocal"):
        make_personalized_risk_config(knowledge)
    knowledge[field][1][0] = 1
    knowledge[field][0][0] = 2
    with pytest.raises(ValueError, match="unit diagonal"):
        make_personalized_risk_config(knowledge)


@pytest.mark.parametrize("bad", [[], None, [5, 4, 1], [5] * 5, "5431"])
def test_malformed_answer_group(knowledge, bad):
    knowledge["fuzzy_answers"]["watching"] = bad
    with pytest.raises(ValueError):
        make_personalized_risk_config(knowledge)


@pytest.mark.parametrize("bad", [None, "", "匿名01", "S 01", "S01\n", True, "../S01"])
def test_identifier_is_ascii_and_safe(knowledge, bad):
    knowledge["respondent_id"] = bad
    with pytest.raises(ValueError, match="ASCII"):
        make_personalized_risk_config(knowledge)


@pytest.mark.parametrize("identifier", ["S01", "S02", "AI-R01"])
@pytest.mark.parametrize("aggregation", ["weighted_sum", "weighted_max"])
def test_valid_metadata_and_aggregation_do_not_mutate_input(knowledge, identifier, aggregation):
    knowledge["respondent_id"] = identifier
    original = deepcopy(knowledge)
    config = make_personalized_risk_config(knowledge, aggregation)
    assert knowledge == original
    assert config.action_aggregation == aggregation
    assert config.model_type == "spatial_context"
    assert config.use_legacy_like_fuzzy
    assert config.fuzzy_profile_name == identifier
    assert len(config.fuzzy_rule_results) == 4
    baseline = RiskConfig()
    for field in ("room_diagonal_m", "height_sigmoid_gain", "height_sigmoid_center", "staff_weights"):
        assert getattr(config, field) == getattr(baseline, field)


def test_unknown_action_aggregation_rejected(knowledge):
    with pytest.raises(ValueError, match="action_aggregation"):
        make_personalized_risk_config(knowledge, "average")


def test_personal_watching_changes_priority_without_changing_lower_risks(knowledge):
    frames = [
        FeatureFrame(
            person_id=person, time_s=0, is_patient_label="yes", is_patient_confidence=1,
            age_group_label="old", age_confidence=1, pose_features=PoseFeatures(0, 0, 0, 0),
            patient_position=Position2D(0, 0), nearest_staff_position=Position2D(distance, 0),
            nearest_staff_velocity=Velocity2D(velocity, 0),
        )
        for person, distance, velocity in (("P01", 8, -1), ("P02", .1, 1))
    ]
    # External and total groups pass through their second (watching/external) input.
    knowledge["fuzzy_answers"]["external"] = [5, 1, 5, 1]
    knowledge["fuzzy_answers"]["total"] = [5, 1, 5, 1]
    runs = []
    for answers in ([5, 5, 1, 1], [5, 1, 5, 1]):
        knowledge["fuzzy_answers"]["watching"] = answers
        engine = RiskEngine(make_personalized_risk_config(knowledge))
        runs.append([engine.evaluate(frame) for frame in frames])
    assert runs[0][0].total_risk > runs[0][1].total_risk
    assert runs[1][0].total_risk < runs[1][1].total_risk
    for old, new in zip(*runs):
        assert old.factor_risks == new.factor_risks
        assert old.upper_risks[ids.INTERNAL_RISK] == new.upper_risks[ids.INTERNAL_RISK]
        assert old.upper_risks[ids.EXTERNAL_STATIC_RISK] == new.upper_risks[ids.EXTERNAL_STATIC_RISK]
        assert old.upper_risks[ids.EXTERNAL_DYNAMIC_RISK] != new.upper_risks[ids.EXTERNAL_DYNAMIC_RISK]
