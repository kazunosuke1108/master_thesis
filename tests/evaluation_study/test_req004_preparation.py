"""Check semantic source ordering and anonymous preservation, not real answers."""
import csv
import importlib.util
import json
from pathlib import Path
import pytest
import yaml
from tests.evaluation_study.test_study import manifest

SPEC=importlib.util.spec_from_file_location('req004_prepare',Path(__file__).resolve().parents[2]/'studies/REQ-004/prepare_inputs.py')
prep=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(prep)


def test_source_order_and_unmapped_answer_preserved(monkeypatch,tmp_path,manifest):
    old=tmp_path/'old';old.mkdir();raw=tmp_path/'raw';raw.mkdir();common=tmp_path/'common';common.mkdir()
    private_name='private_source_name'
    def write_csv(path,rows):
        with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    scores=[1,2,3,4,1,5,2,4,5,4,3,2]
    write_csv(raw/'questionaire_1b.csv',[['1b',private_name]]+[[i+1,x] for i,x in enumerate(scores)])
    write_csv(raw/'questionaire_1a.csv',[['1a',private_name]]+[[i+1,7 if i==24 else 5] for i in range(25)])
    write_csv(common/f'TFN_{private_name}.csv',[[(v-1)/4]*3 for v in scores])
    for key,n in [('30000001',7),('30000010',3)]:write_csv(common/f'comparison_mtx_{key}_{private_name}.csv',[[1]*n for _ in range(n)])
    (old/'existing-profiles-private.yaml').write_text(yaml.safe_dump({'profiles':[{'respondent_id':'R01','ahp_profile':private_name,'fuzzy_profile':private_name,'common_dir':str(common)}]}))
    (old/'synthetic-knowledge').mkdir()
    for identity in ['S01','S02']:(old/'synthetic-knowledge'/f'{identity}.json').write_text(json.dumps({'respondent_id':identity,'synthetic':True,'action_pairwise':[[1]*7]*7,'object_pairwise':[[1]*3]*3,'fuzzy_answers':{'internal':[1,2,3,4],'external':[1,2,3,4],'total':[1,2,3,4]}}))
    monkeypatch.setattr(prep,'OLD',old);monkeypatch.setattr(prep,'RAW',raw)
    output=tmp_path/'new'
    prep.prepare(output,manifest)
    data=json.loads((output/'inputs/AI-R01-base.json').read_text())
    assert data['fuzzy_answers']['external']==[5,1,4,2]
    assert data['unmapped_original_answers'][0]['raw_score']==7
    assert data['unmapped_original_answers'][0]['mapping_status']=='unresolved'
    assert private_name not in json.dumps(data)
    assert data['knowledge_origin']['watching']=='not_yet_collected'
    before=(output/'inputs/AI-R01-base.json').read_bytes()
    with pytest.raises(FileExistsError):prep.prepare(output,manifest)
    assert (output/'inputs/AI-R01-base.json').read_bytes()==before


@pytest.mark.parametrize('bad',[2.5,float('inf'),float('nan'),0,6])
def test_original_scores_not_truncated_or_invented(bad):
    with pytest.raises(ValueError):prep.integer_score(bad,1,5)


def test_view_proxy_presentation_is_explicit_and_legacy_unchanged(manifest):
    from master_thesis_modules.evaluation_study.study import load_cases,_scene
    _,_,cases=load_cases(manifest)
    assert '視線ではありません' in _scene(cases[0])['coordinate_note']
    cases[0]['raw']['view_proxy_assumption']=True
    scene=_scene(cases[0])
    assert '視線ではありません' not in scene['coordinate_note']
    assert '見守り方向' in scene['coordinate_note']
    assert '実際の視線を計測したものではありません' in scene['coordinate_note']
