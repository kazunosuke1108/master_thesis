import importlib.util
import json
from pathlib import Path

import pytest
from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config

spec = importlib.util.spec_from_file_location("pilot_knowledge", Path(__file__).parents[2] / "studies/REQ-003/convert_pilot_knowledge.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def knowledge():
    return {"respondent_id":"S01", "synthetic":True,
            "action_pairwise":[[1]*7 for _ in range(7)],
            "object_pairwise":[[1]*3 for _ in range(3)],
            "fuzzy_answers":{"internal":[5,4,3,1], "external":[1,2,3,4], "total":[5,3,2,1]}}


def test_existing_factory_restores_canonical_answers(tmp_path):
    source=tmp_path/"answers.json"; data=knowledge(); source.write_text(json.dumps(data))
    module.convert(source,tmp_path/"common")
    config=make_profile_risk_config("S01","S01",tmp_path/"common")
    assert config.fuzzy_rule_results[20000000] == (1.0,.75,.5,0.0)
    assert config.fuzzy_rule_results[20000001] == (0.0,.25,.5,.75)
    assert config.fuzzy_rule_results[10000000] == (1.0,.5,.25,0.0)
    provenance=json.loads((tmp_path/"common/S01_knowledge_provenance.json").read_text())
    assert provenance["input_group_order"] == ["HH","HL","LH","LL"]
    assert provenance["external_csv_row_order"] == ["HL","HH","LL","LH"]
    assert json.loads(source.read_text()) == data


def test_existing_knowledge_is_not_overwritten(tmp_path):
    source=tmp_path/"answers.json";source.write_text(json.dumps(knowledge()))
    module.convert(source,tmp_path/"common")
    before=(tmp_path/"common/TFN_S01.csv").read_bytes()
    with pytest.raises(FileExistsError): module.convert(source,tmp_path/"common")
    assert (tmp_path/"common/TFN_S01.csv").read_bytes() == before


@pytest.mark.parametrize("invalid",["non_synthetic","non_reciprocal","out_of_scale","fractional_fuzzy"])
def test_bad_answers_are_rejected_before_writing(tmp_path,invalid):
    data=knowledge()
    if invalid=="non_synthetic":data["synthetic"]=False
    if invalid=="non_reciprocal":data["action_pairwise"][0][1]=2
    if invalid=="out_of_scale":data["object_pairwise"][0][1]=10
    if invalid=="fractional_fuzzy":data["fuzzy_answers"]["total"][0]=2.5
    source=tmp_path/"answers.json";source.write_text(json.dumps(data))
    with pytest.raises(ValueError):module.convert(source,tmp_path/"common")
    assert not (tmp_path/"common").exists()
