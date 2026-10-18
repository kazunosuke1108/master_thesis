"""Keep original questionnaire codes distinct from converted AHP ratios."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import yaml
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from master_thesis_modules.risk_core.engine.profile_config import DEFAULT_FUZZY_QUESTIONNAIRE_PATH
REPO=Path(__file__).resolve().parents[2]
SCALE={1:9,2:7,3:5,4:3,5:1,6:1/3,7:1/5,8:1/7,9:1/9}
ORDER=['HH','HL','LH','LL']
def generate(root):
 root=Path(root).resolve();old=REPO/'outputs/REQ003_pilot_20261017T165939';rawdir=(REPO/DEFAULT_FUZZY_QUESTIONNAIRE_PATH).parent
 with (rawdir/'questionaire_1a.csv').open(encoding='utf-8-sig',newline='') as f:a=list(csv.DictReader(f))
 with (rawdir/'questionaire_1b.csv').open(encoding='utf-8-sig',newline='') as f:b=list(csv.DictReader(f))
 mapping={f"AI-{p['respondent_id']}":p['fuzzy_profile'] for p in yaml.safe_load((old/'existing-profiles-private.yaml').read_text())['profiles']}
 count=0
 for path in sorted((root/'participants').glob('*/knowledge.json')):
  k=json.loads(path.read_text());identity=k['respondent_id'];rows=[];source_name=mapping.get(identity)
  for field,n,offset in [('action_pairwise',7,0),('object_pairwise',3,21)]:
   ordinal=offset
   for i in range(n):
    for j in range(i+1,n):
     ratio=k[field][i][j];ordinal+=1
     raw=float(a[ordinal-1][source_name]) if source_name else None
     if source_name and (not raw.is_integer() or abs(SCALE[int(raw)]-ratio)>1e-12):raise ValueError('Original AHP conversion mismatch')
     rows.append({'group':field,'knowledge_pointer':f'/{field}/{i}/{j}','source_question_id':f'1A-{ordinal}' if source_name else None,'source_form_code':int(raw) if raw is not None else None,'converted_pairwise_ratio':ratio,'origin':'real_original' if source_name else 'synthetic_original'})
  original_ids={'internal':[1,2,3,4],'external':[6,5,8,7],'total':[9,10,11,12]}
  for group,answers in k['fuzzy_answers'].items():
   for i,answer in enumerate(answers):
    original=original_ids.get(group,[None]*4)[i]
    if source_name and original and int(float(b[original-1][source_name]))!=answer:raise ValueError('Original Fuzzy ordering mismatch')
    rows.append({'group':group,'knowledge_pointer':f'/fuzzy_answers/{group}/{i}','condition':ORDER[i],'source_question_id':f'1B-{original}' if source_name and original else f'REQ004-watching-{ORDER[i]}' if group=='watching' else None,'raw_score':answer,'normalized_center':(answer-1)/4,'origin':'synthetic_additional' if group=='watching' else 'real_original' if source_name else 'synthetic_original'})
  out=path.parent/'knowledge-lineage.json'
  with out.open('x') as f:json.dump({'respondent_id':identity,'knowledge_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':rows,'unmapped_original_answers':k.get('unmapped_original_answers',[]),'note':'source_form_code and converted_pairwise_ratio are distinct; unresolved Qd not transformed'},f,ensure_ascii=False,indent=2)
  count+=1
 return count
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);v=p.parse_args();print(generate(v.root))
