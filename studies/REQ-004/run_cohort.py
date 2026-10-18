"""Resume immutable per-person analyses and save the actual dirty execution code."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from master_thesis_modules.evaluation_study.additional_evaluation import analyze_person
from master_thesis_modules.evaluation_study.additional_questionnaire import render_followup
REPO=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot(destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    files=[]
    for relative in ['master_thesis_modules/risk_core','master_thesis_modules/scenario_sim','master_thesis_modules/evaluation_study','studies/REQ-004']:
        files.extend(p for p in (REPO/relative).rglob('*') if p.suffix in {'.py','.js','.yaml'})
    init=REPO/'master_thesis_modules/__init__.py'
    if init.exists():files.append(init)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(REPO)))
    patch=subprocess.check_output(['git','diff','--binary','HEAD','--','master_thesis_modules','studies','tests','.gitignore'])
    patchfile=destination.with_suffix('.patch');patchfile.write_bytes(patch)
    meta={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'archive_sha256':sha(destination),'dirty_patch_hash':sha(patchfile),'archive_contains_untracked_code':True,'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in sorted(set(files))}}
    destination.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    return meta

def run(root,identities,analysis_subdir="analysis-v2"):
    root=Path(root).resolve();manifest=REPO/'studies/REQ-004/prototype.yaml';results=[]
    for identity in identities:
        p=root/'participants'/identity
        for aggregation in ['weighted_sum','weighted_max']:
            out=root/analysis_subdir/aggregation/identity
            if out.exists():
                saved=json.loads((out/'followup-packet.json').read_text())
                if saved['knowledge_sha256']!=sha(p/'knowledge.json') or saved['initial_responses_sha256']!=sha(p/'initial-responses.csv'):raise ValueError('Existing immutable analysis has different inputs')
                results.append({'identity':identity,'aggregation':aggregation,'status':'already_saved'});continue
            command=['python3','studies/REQ-004/run_pilot.py','analyze',str(manifest),str(p/'knowledge.json'),str(p/'initial-responses.csv'),str(out),'--action-aggregation',aggregation]
            result=analyze_person(manifest,p/'knowledge.json',p/'initial-responses.csv',out,aggregation,command=command)
            snapshot(out/'execution-code.zip')
            packet=out/'followup-packet.json';value=json.loads(packet.read_text())
            with (out/'followup.html').open('x',encoding='utf-8') as handle:handle.write(render_followup(value,packet_sha256=sha(packet)))
            results.append({'identity':identity,'aggregation':aggregation,'summary':result['summary']})
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('identities',nargs='+');p.add_argument('--analysis-subdir',default='analysis-v2');a=p.parse_args();print(json.dumps(run(a.root,a.identities,a.analysis_subdir),ensure_ascii=False,indent=2))
