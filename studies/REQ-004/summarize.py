"""Summarize three distinct descriptive axes without imputing acceptance."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np


def rank_text(values):
    return ' → '.join('＝'.join(sorted(k for k,v in values.items() if v==r)) for r in sorted(set(values.values())))

def summarize(root,destination):
    root=Path(root).resolve();destination=Path(destination)
    if destination.exists():raise FileExistsError('Summary output must be new')
    rows=[];profiles=[];source_hashes={};aggregations=['weighted_sum','weighted_max']
    ids=['S01','S02']+[f'AI-R{i:02}' for i in range(1,15)]
    def read(path):
        source_hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    for identity in ids:
        knowledge=read(root/'participants'/identity/'knowledge.json');ci={}
        base=read(root/'inputs'/f'{identity}-base.json')
        for field in ['action_pairwise','object_pairwise']:
            if knowledge[field]!=base[field]:raise ValueError('Existing pairwise answers changed')
            m=np.asarray(knowledge[field]);ci[field]=(float(np.linalg.eigvals(m).real.max())-len(m))/(len(m)-1)
        for group in ['internal','external','total']:
            if knowledge['fuzzy_answers'][group]!=base['fuzzy_answers'][group]:raise ValueError('Existing Fuzzy answers changed')
        profiles.append({'respondent_id':identity,'cohort':'pilot2' if identity.startswith('S') else 'derived14','CI':ci,'fuzzy_answers':knowledge['fuzzy_answers'],'unmapped_original_answers':knowledge.get('unmapped_original_answers',[])})
        for agg in aggregations:
            folder=root/'analysis-v2'/agg/identity;packet=read(folder/'followup-packet.json')
            follow=read(root/'followup-v2'/agg/identity/'summary.json')
            actual={c['case_id']:c for c in follow['cases']};requested={c['case_id']:c for c in packet['cases']}
            with (folder/'comparison.csv').open(newline='') as handle:comparison=list(csv.DictReader(handle))
            for case in packet['summary_cases']:
                c=case['case_id'];data=actual.get(c);comp=[v for v in comparison if v['case_id']==c]
                initial={v['patient_id']:int(v['response_rank']) for v in comp if v['response_rank']}
                system={v['patient_id']:int(v['system_rank']) for v in comp}
                if case['exact_rank_match'] is False and data is None:raise ValueError('Missing follow-up')
                rows.append({'respondent_id':identity,'cohort':'pilot2' if identity.startswith('S') else 'derived14','aggregation':agg,'case_id':c,'initial_ranks':initial,'initial_rank_text':rank_text(initial) if initial else '判断不能','system_ranks':system,'system_rank_text':rank_text(system),'exact_rank_match':case['exact_rank_match'],'followup_status':'answered' if data else case['followup_status'],'helpfulness':data['helpfulness'] if data else None,'acceptance':data['acceptance'] if data else None,'reason':data['reason'] if data else '', 'pairs':data['pairs'] if data else [],'requested_pairs':len(requested[c]['pairs']) if c in requested else 0})
    summaries=[]
    for cohort in ['pilot2','derived14','all16']:
        for agg in aggregations:
            subset=[r for r in rows if r['aggregation']==agg and (cohort=='all16' or r['cohort']==cohort)]
            asked=[r for r in subset if r['followup_status']=='answered']
            summaries.append({'cohort':cohort,'aggregation':agg,'case_count':len(subset),'exact_matches':sum(r['exact_rank_match'] is True for r in subset),'mismatches':sum(r['exact_rank_match'] is False for r in subset),'followup_case_count':len(asked),'acceptance_counts':dict(Counter(r['acceptance'] for r in asked)),'helpfulness_counts':dict(Counter(str(r['helpfulness']) for r in asked)),'pair_count':sum(len(r['pairs']) for r in asked),'pair_acceptance_counts':dict(Counter(p['acceptance'] for r in asked for p in r['pairs'])),'unasked_cases':len(subset)-len(asked)})
    destination.mkdir(parents=True,exist_ok=False)
    payload={'request':'REQ-004 revision 2','notice':'AI pilot; derived14 are not original human rank/followup responses; conditions are repeated observations, not independent people','explanation_version':2,'primary_metric':None,'combined_success_score':None,'summaries':summaries,'profiles':profiles,'cases':rows,'source_sha256':source_hashes}
    (destination/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fields=['respondent_id','cohort','aggregation','case_id','initial_rank_text','system_rank_text','exact_rank_match','followup_status','helpfulness','acceptance','reason','requested_pairs']
    with (destination/'case-results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    return summaries
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();print(json.dumps(summarize(a.root,a.destination),ensure_ascii=False,indent=2))
