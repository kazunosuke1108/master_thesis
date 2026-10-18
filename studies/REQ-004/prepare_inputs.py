"""Create anonymized REQ004 inputs without changing any original answers."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import yaml
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from master_thesis_modules.evaluation_study.study import build_bundle, load_cases, _scene
from master_thesis_modules.evaluation_study.questionnaire import render_questionnaire

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / 'outputs/REQ003_pilot_20261017T165939'
from master_thesis_modules.risk_core.engine.profile_config import DEFAULT_FUZZY_QUESTIONNAIRE_PATH
RAW = REPO / DEFAULT_FUZZY_QUESTIONNAIRE_PATH.parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, data):
    with Path(p).open('x', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def integer_score(value, low, high):
    value=float(value)
    if not value.is_integer() or not low <= value <= high:
        raise ValueError('Original answer must be an integer in the documented scale')
    return int(value)

def matrix(path):
    with Path(path).open(newline='') as f:
        return [[float(x) for x in row] for row in csv.reader(f)]

def prepare(root, manifest):
    root=Path(root); manifest=Path(manifest)
    if (root/'inputs').exists(): raise FileExistsError('Prepared inputs are never overwritten')
    (root/'inputs').mkdir(parents=True)
    bundle=build_bundle(manifest,root/'presentation')
    _,_,cases=load_cases(manifest)
    scenes=[_scene(c) for c in cases]
    (bundle/'participant/personalized-questionnaire.html').write_text(render_questionnaire(scenes,include_watching=True),encoding='utf-8')
    # Preserve movement data and explicitly label its assumed role; not measured gaze.
    semantic={'assumption':'移動方向を見守り方向（視野方向）の代理とする研究上の設定。実測視線ではない。', 'scenes':scenes}
    write(root/'inputs/scenes.json',semantic)
    with (RAW/'questionaire_1b.csv').open(newline='',encoding='utf-8-sig') as f: rows_b=list(csv.DictReader(f))
    with (RAW/'questionaire_1a.csv').open(newline='',encoding='utf-8-sig') as f: rows_a=list(csv.DictReader(f))
    private=yaml.safe_load((OLD/'existing-profiles-private.yaml').read_text())['profiles']
    prepared=[]
    for p in private:
        identity='AI-'+p['respondent_id']; name=p['fuzzy_profile']; common=Path(p['common_dir'])
        raw_scores=[integer_score(row[name],1,5) for row in rows_b]
        if len(raw_scores)!=12 or any(x not in range(1,6) for x in raw_scores):raise ValueError('Invalid anonymous 1B input')
        tfns=matrix(common/f'TFN_{name}.csv')
        if len(tfns)!=12 or any(len(t)!=3 for t in tfns) or any(abs(t[1]-(x-1)/4)>1e-12 for t,x in zip(tfns,raw_scores)):raise ValueError('TFN source contract mismatch')
        a=common/f"comparison_mtx_30000001_{p['ahp_profile']}.csv";o=common/f"comparison_mtx_30000010_{p['ahp_profile']}.csv"
        base={'respondent_id':identity,'synthetic':True,'data_kind':'real_answer_derived_ai',
              'action_pairwise':matrix(a),'object_pairwise':matrix(o),
              'fuzzy_answers':{'internal':raw_scores[:4],'external':[raw_scores[i] for i in (5,4,7,6)],'total':raw_scores[8:]},
              'unmapped_original_answers':[{'question_id':'25','raw_score':integer_score(rows_a[24][name],1,9),'mapping_status':'unresolved','note':'Qdへの対応と左右項目は原票未確認。推論に使用しない。'}],
              'knowledge_origin':{'action_pairwise':'real_original','object_pairwise':'real_original','internal':'real_original','external':'real_original','total':'real_original','watching':'not_yet_collected'},
              'source_hashes':{'questionnaire_1a':sha(RAW/'questionaire_1a.csv'),'questionnaire_1b':sha(RAW/'questionaire_1b.csv'),'action_matrix':sha(a),'object_matrix':sha(o),'tfn':sha(common/f'TFN_{name}.csv')},
              'canonical_order':['HH','HL','LH','LL'],'source_1b_external_order':['HL','HH','LL','LH']}
        write(root/'inputs'/f'{identity}-base.json',base);prepared.append(identity)
    for identity in ['S01','S02']:
        src=OLD/'synthetic-knowledge'/f'{identity}.json';base=json.loads(src.read_text())
        base.update({'data_kind':'synthetic_ai','canonical_order':['HH','HL','LH','LL'],
                     'knowledge_origin':{k:'synthetic_original' for k in ['action_pairwise','object_pairwise','internal','external','total']},
                     'source_hashes':{'original_knowledge':sha(src)},'persona_reconstruction':'new prompt based on saved answers/rationale; original prompt unavailable'})
        base['knowledge_origin']['watching']='not_yet_collected'
        write(root/'inputs'/f'{identity}-base.json',base)
    write(root/'inputs/preparation.json',{'respondent_ids':['S01','S02']+prepared,'source_manifest_sha256':sha(manifest),'preparer_sha256':sha(__file__),'private_mapping_sha256':sha(OLD/'existing-profiles-private.yaml'),'personally_identifying_names_exported':False,'source_1b_tfn_centers_checked':len(prepared),'unmapped_question25_count':len(prepared)})
    return root/'inputs'
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('output_root',type=Path);a.add_argument('--manifest',type=Path,default=REPO/'studies/REQ-004/prototype.yaml');v=a.parse_args();print(prepare(v.output_root,v.manifest))
