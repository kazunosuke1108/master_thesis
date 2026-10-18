"""Create future-use forms separately from packets used in the AI evaluation."""
from pathlib import Path
import argparse,hashlib,json,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from master_thesis_modules.evaluation_study.study import build_bundle,load_cases,_scene,_json
from master_thesis_modules.evaluation_study.questionnaire import render_questionnaire
from master_thesis_modules.evaluation_study.additional_questionnaire import render_followup
from master_thesis_modules.evaluation_study.additional_evaluation import _case_bridge
REPO=Path(__file__).resolve().parents[2]
def render(root):
    root=Path(root).resolve();out=root/'presentation-final'
    if out.exists():raise FileExistsError('Final presentation must be new')
    manifest=REPO/'studies/REQ-004/prototype.yaml';build_bundle(manifest,out)
    _,_,cases=load_cases(manifest)
    (out/'participant/personalized-questionnaire.html').write_text(render_questionnaire([_scene(c) for c in cases],include_watching=True),encoding='utf-8')
    ledger=[]
    for source in sorted((root/'analysis-v2').glob('*/*/followup-packet.json')):
        packet=json.loads(source.read_text());changes=0
        for case in packet['cases']:
            revised=_case_bridge(case['pairs'],case['system_ranks'],case['initial_ranks'])
            changes+=revised!=case['case_explanation'];case['case_explanation']=revised
        packet['presentation_revision']='2.1-final-prototype'
        packet['used_in_reported_ai_evaluation']=False
        p=out/'followup'/source.parent.parent.name/source.parent.name;p.mkdir(parents=True)
        target=p/'packet.json';target.write_text(_json(packet))
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        (p/'questionnaire.html').write_text(render_followup(packet,packet_sha256=digest),encoding='utf-8')
        ledger.append({'respondent_id':packet['respondent_id'],'aggregation':source.parent.parent.name,'source_packet_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'future_packet_sha256':digest,'case_sentences_corrected':changes,'future_presentation_only':True})
    (out/'presentation-provenance.json').write_text(_json({'changes':['five-point endpoints aligned with protocol','case-level explanation included in whole-ranking form','zero-difference sentence grammar corrected'],'used_for_new_ai_responses':False,'original_ai_packets_unchanged':True,'ledger':ledger}))
    return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();print(render(a.root))
