from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config
from master_thesis_modules.risk_core.schema import node_ids as ids


def test_fuzzy_profiles_match_legacy_questionnaire_values():
    nakamura = make_profile_risk_config(
        ahp_profile_name="中村",
        fuzzy_profile_name="中村",
    )
    hyakutake = make_profile_risk_config(
        ahp_profile_name="中村",
        fuzzy_profile_name="百武",
    )

    assert nakamura.fuzzy_rule_results[ids.EXTERNAL_RISK] == (
        0.75,
        0.0,
        0.25,
        0.0,
    )
    assert nakamura.fuzzy_rule_results[ids.TOTAL_RISK] == (
        1.0,
        0.0,
        0.75,
        0.25,
    )
    assert hyakutake.fuzzy_rule_results[ids.EXTERNAL_RISK] == (
        1.0,
        0.75,
        0.75,
        0.5,
    )
    assert hyakutake.fuzzy_rule_results[ids.TOTAL_RISK] == (
        1.0,
        0.5,
        0.25,
        0.0,
    )


def test_ahp_profiles_load_legacy_action_weights():
    hyakutake = make_profile_risk_config(
        ahp_profile_name="百武",
        fuzzy_profile_name="百武",
    )

    assert hyakutake.action_weights[ids.STANDING_RISK] == 0.1481226337180131
    assert hyakutake.action_weights[ids.LOSING_BALANCE_RISK] == 0.24800886586767873
    assert hyakutake.action_weights[ids.TOUCHING_FACE_RISK] == 0.39001998670236304
