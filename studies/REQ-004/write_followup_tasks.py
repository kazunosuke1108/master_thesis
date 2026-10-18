"""Expose only participant-facing explanations to independent condition branches."""
from pathlib import Path
import argparse,hashlib,json

def write(root,identities,analysis_subdir="analysis-v2",protocol_suffix="v2",answers_subdir="followup-answers-v2"):
    root=Path(root).resolve();created=[]
    for identity in identities:
      for agg in ['weighted_sum','weighted_max']:
        analysis=root/analysis_subdir/agg/identity
        packet=json.loads((analysis/'followup-packet.json').read_text())
        dest=root/'protocol'/f'{identity}-{agg}-{protocol_suffix}'
        if dest.exists():raise FileExistsError('Protocol branches never overwritten')
        dest.mkdir()
        display={'respondent_id':identity,'cases':[{'case_id':c['case_id'],'scene':c['scene'],'initial_ranks':c['initial_ranks'],'initial_reasons':c['initial_reasons'],'system_ranks':c['system_ranks'],'case_explanation':c['case_explanation'],'pairs':[{k:p[k] for k in ['pair_id','patient_ids','explanation']} for p in c['pairs']]} for c in packet['cases']]}
        displayfile=dest/'participant-display.json';displayfile.write_text(json.dumps(display,ensure_ascii=False,indent=2)+'\n')
        output=root/answers_subdir/agg/f'{identity}.json';knowledge=root/'participants'/identity/'knowledge.json';initial=root/'participants'/identity/'initial-responses.csv';template=analysis/'followup-template.json'
        task=f'''あなたは研究用の合成AIスタッフ {identity} の説明後回答役です。実スタッフ本人ではありません。
読むファイルは {knowledge}、{initial}、{displayfile}、{template} の4つだけです。別条件や他の回答者の結果は見ないでください。実装コード・研究者用の診断値も読まないでください。
既存の知識回答・最初の順位と理由を持つ人物として、提示された説明を読み、率直な評価を答えます。肯定する必要はなく、受容/部分的受容/拒否/判断保留いずれも可能です。最初の回答を事後的に書き換えないでください。説明の内容が不十分ならそのように答えてください。スタッフの経験・職種・履歴を捏造しないでください。
各場面の4患者全体の算出順位と、各patient pairの説明について、それぞれ以下を答えます。
helpfulness:1全く参考にならない、2あまり参考にならない、3どちらともいえない、4やや参考になる、5とても参考になる。説明だけでは判断できない場合は明示null。
acceptance:accept（受け入れる）、partial（一部のみ受け入れる）、reject（受け入れない）、uncertain（判断できない）。参考になるが受け入れない、も区別してください。
reason:理由を短く記してください。最初の順位判断で重視したこととアンケートで重視したことに、同じ点や違う点があれば教えてください。違いがない場合もそのまま答えてください。研究者が望む返答を作る必要はありません。
雛形のschema_version/respondent_id/packet_sha256/case_id/pair_idを変えず、全指定caseとpairに回答を埋めてください。一つのpairへの受容を場面全体の受容と同一にせず、全体は別に判断してください。
casesが空なら追加質問はありません。テンプレートを空casesのまま保存し、受容を創作しないでください。
所有書込は新規 {output} のみ（親directory作成可）。他担当と共有ワークスペースのためコードや他の成果を書き換えず、再委譲禁止。既存ファイルは上書きしないでください。短い回答理由だけを保存し、内部の思考過程やセッション全文は保存しないでください。
最終は保存したパスだけを簡潔に報告してください。
'''
        taskfile=dest/'task.txt';taskfile.write_text(task)
        invocation=f'Read and execute the isolated follow-up task in {taskfile}. Read only its specified participant inputs and write only its assigned answer file. Do not redelegate.'
        record={'respondent_id':identity,'condition':agg,'phase':'after_explanation','independent_condition_branch':True,'invocation_prompt':invocation,'task_sha256':hashlib.sha256(task.encode()).hexdigest(),'display_sha256':hashlib.sha256(displayfile.read_bytes()).hexdigest(),'original_packet_sha256':hashlib.sha256((analysis/'followup-packet.json').read_bytes()).hexdigest(),'model_identifier':'not exposed','sampling_seed':'unavailable'}
        (dest/'provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');created.append(str(taskfile))
    return created
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('identities',nargs='+');a=p.parse_args();print('\n'.join(write(a.root,a.identities)))
