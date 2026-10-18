"""Validate completed follow-ups; empty packets mean not asked, never accepted."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from master_thesis_modules.evaluation_study.additional_evaluation import validate_followup

def collect(root):
 root=Path(root).resolve();result=[]
 for folder in sorted((root/'analysis-v2').glob('*/*')):
  agg,identity=folder.parent.name,folder.name
  source=folder/'followup-packet.json';packet=json.loads(source.read_text());answer=root/'followup-answers-v2'/agg/f'{identity}.json';out=root/'followup-v2'/agg/identity
  if not packet['cases'] and not answer.exists():
   answer.parent.mkdir(parents=True,exist_ok=True)
   with answer.open('x',encoding='utf-8') as f:f.write((folder/'followup-template.json').read_text())
   with answer.with_suffix('.not-asked.json').open('x') as f:json.dump({'generated_without_ai':True,'reason':'all ranked cases agree or are unable; no extra question asked','positive_acceptance_imputed':False},f,indent=2)
  if not answer.exists():result.append({'respondent_id':identity,'aggregation':agg,'status':'pending'});continue
  if out.exists():
   if (out/'answers.json').read_bytes()!=answer.read_bytes() or (out/'packet.json').read_bytes()!=source.read_bytes():raise ValueError('Immutable follow-up has changed inputs')
  else:validate_followup(source,answer,out,command=['python3','studies/REQ-004/run_pilot.py','followup',str(source),str(answer),str(out)])
  result.append({'respondent_id':identity,'aggregation':agg,'status':'saved' if packet['cases'] else 'not_asked'})
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(collect(a.root),indent=2))
