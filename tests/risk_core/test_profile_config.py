import pytest

from master_thesis_modules.risk_core.engine import profile_config
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


# Synthetic fixtures test file ordering without relying on mounted legacy data.


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_config, "LEGACY_COMMON_DIR", tmp_path / "legacy")
    monkeypatch.setattr(profile_config, "DEFAULT_FUZZY_QUESTIONNAIRE_PATH", tmp_path / "answers.csv")
    return tmp_path


def _write_tfn(path, centers):
    # l/r can lie outside [0,1]; only c is constrained for internal consequents.
    path.write_text("".join(f"-0.5,{center},1.5\n" for center in centers), encoding="utf-8")


def _write_answers(path, answers):
    path.write_text("1b,staff\n" + "".join(f"q{i},{answer}\n" for i, answer in enumerate(answers)), encoding="utf-8")


def test_tfn_internal_order_and_existing_external_total_order(isolated_profiles):
    centers = [.9, .7, .3, .1, .2, .4, .6, .8, .15, .35, .55, .75]
    _write_tfn(isolated_profiles / "TFN_staff.csv", centers)
    rules = make_profile_risk_config(fuzzy_profile_name="staff", common_dir=isolated_profiles).fuzzy_rule_results
    assert rules[ids.INTERNAL_RISK] == (.9, .7, .3, .1)
    assert rules[ids.EXTERNAL_RISK] == (.4, .2, .8, .6)
    assert rules[ids.TOTAL_RISK] == (.15, .35, .55, .75)


def test_absent_tfn_uses_questionnaire_in_correct_order(isolated_profiles):
    _write_answers(isolated_profiles / "answers.csv", [5, 4, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4])
    rules = make_profile_risk_config(fuzzy_profile_name="staff", common_dir=isolated_profiles).fuzzy_rule_results
    assert rules[ids.INTERNAL_RISK] == (1., .75, .25, 0.)
    assert rules[ids.EXTERNAL_RISK] == (.5, .25, 1., .75)
    assert rules[ids.TOTAL_RISK] == (0., .25, .5, .75)


@pytest.mark.parametrize("bad", ["nan", "inf", "-0.1", "1.1", "invalid"])
def test_existing_invalid_tfn_never_falls_back_to_valid_questionnaire(isolated_profiles, bad):
    _write_tfn(isolated_profiles / "TFN_staff.csv", [bad] + [.5] * 11)
    _write_answers(isolated_profiles / "answers.csv", [5] * 12)
    with pytest.raises(ValueError):
        make_profile_risk_config(fuzzy_profile_name="staff", common_dir=isolated_profiles)


@pytest.mark.parametrize("rows", [0, 3, 11])
def test_existing_incomplete_tfn_is_error(isolated_profiles, rows):
    _write_tfn(isolated_profiles / "TFN_staff.csv", [.5] * rows)
    with pytest.raises(ValueError):
        make_profile_risk_config(fuzzy_profile_name="staff", common_dir=isolated_profiles)


@pytest.mark.parametrize("model", ["action_only", "patient_context", "action_attribute", "spatial_context"])
@pytest.mark.parametrize("profile", ["unknown", "中村", "百武"])
def test_named_unavailable_profile_is_error_for_every_model(isolated_profiles, model, profile):
    with pytest.raises(ValueError):
        make_profile_risk_config(fuzzy_profile_name=profile, common_dir=isolated_profiles, model_type=model)


def test_unknown_questionnaire_profile_is_error(isolated_profiles):
    _write_answers(isolated_profiles / "answers.csv", [5] * 12)
    with pytest.raises(ValueError):
        make_profile_risk_config(fuzzy_profile_name="unknown", common_dir=isolated_profiles)


def test_incomplete_questionnaire_is_error(isolated_profiles):
    _write_answers(isolated_profiles / "answers.csv", [5] * 3)
    with pytest.raises(ValueError):
        make_profile_risk_config(fuzzy_profile_name="staff", common_dir=isolated_profiles)


def test_unspecified_profile_requires_no_files(isolated_profiles):
    assert make_profile_risk_config(common_dir=isolated_profiles).fuzzy_rule_results == {}


@pytest.mark.parametrize("bad", [4.5, "nan", 0, 6, -1])
@pytest.mark.parametrize("index", range(4))
def test_invalid_internal_questionnaire_answer_is_error(isolated_profiles, bad, index):
    answers = [5] * 12
    answers[index] = bad
    _write_answers(isolated_profiles / "answers.csv", answers)
    with pytest.raises(ValueError):
        make_profile_risk_config(fuzzy_profile_name="staff", common_dir=isolated_profiles)


@pytest.mark.parametrize("row", ["0.2,0.5\n", "0.2,0.5,0.8,1.0\n"])
def test_invalid_tfn_column_count_is_error_without_questionnaire_fallback(isolated_profiles, row):
    (isolated_profiles / "TFN_staff.csv").write_text(row * 12, encoding="utf-8")
    _write_answers(isolated_profiles / "answers.csv", [5] * 12)
    with pytest.raises(ValueError):
        make_profile_risk_config(fuzzy_profile_name="staff", common_dir=isolated_profiles)
