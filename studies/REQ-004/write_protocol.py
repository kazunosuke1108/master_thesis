"""Save the exact instructions used for isolated AI-persona initial responses."""
from pathlib import Path
import argparse,json,hashlib

def write_protocol(root):
    root=Path(root).resolve()
    for identity in ['S01','S02']+[f'AI-R{i:02}' for i in range(1,15)]:
        base=root/'inputs'/f'{identity}-base.json';scenes=root/'inputs/scenes.json';out=root/'participants'/identity
        text=f'''あなたは研究用の合成AIスタッフ {identity} の回答役です。実スタッフ本人の再現が証明された人物ではありません。
読むファイルは {base} と {scenes} の二つだけです。システムの実装・出力や他回答者のデータを見ないでください。baseの既存判断知識を一貫して用い、保存理由があれば尊重してください。経験年数・職種・履歴を捏造しないでください。unmapped_original_answersは原値の意味と左右項目が未確認なので、判断の根拠にしないでください。
場面ではスタッフvelocityの向きを見守りの方向とする研究上の仮定を用います。これは視線の実測ではありません。
最初に、見守り統合の4問を1低危険〜5高危険で答えます。順番はHH,HL,LH,LL。第一入力はスタッフとの距離リスク（H遠い/L近い）、第二入力は視野外リスク（H患者が視野外/L視野内）。二つを合わせた危険性を各1〜5整数で答えてください。
次にC01,C02,C03について、この瞬間に追加の見守りを優先する患者の順を全4患者に答え、各患者の短い判断理由を記してください。順位が小さいほど優先、同順位可（競争順位1,1,3,4）。判断できなければその場面全患者status=unable、rankは空欄とし理由を答えます。通常status=ranked。
本人役の判断を回答してください。計算結果を推測して一致を作る必要はありません。システム結果はまだ提示していません。
書込担当は新規 {out} のみ。他担当も共有ワークスペースで作業しているため、他のファイルやコードを変更しないでください。再委譲禁止。
knowledge.json: baseの全項目・原値を保持したままfuzzy_answers.watchingへ4整数を追加し、knowledge_origin.watching='synthetic_additional'、watching_rationaleへ短い理由を追加します。
initial-responses.csv: headerはcase_id,respondent_id,patient_id,rank,status,reason。respondent_id={identity}、3場面各4患者=12行を保存します。
新規作成のみ、ファイルが既存なら上書きせずPMへ報告してください。内部の思考過程やセッション全文は保存せず、要求した回答・短い理由のみを記録します。
最終応答は追加4回答と保存パスを簡潔に報告してください。
'''
        target=root/'protocol'/f'{identity}-initial-task.txt'
        with target.open('x',encoding='utf-8') as f:f.write(text)
        invocation=f'Read and execute the isolated AI-persona task in {target}. Read only the task file and its two specified input files; write only its assigned output directory. Do not redelegate.'
        record={'respondent_id':identity,'phase':'initial_before_system','task_file':str(target),'task_sha256':hashlib.sha256(text.encode()).hexdigest(),'tool_invocation_prompt':invocation,'knowledge_base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'scenes_sha256':hashlib.sha256(scenes.read_bytes()).hexdigest(),'generation_model_identifier':'not exposed by agent tool','sampling_seed':'unavailable'}
        with (root/'protocol'/f'{identity}-initial-provenance.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    return root/'protocol'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(write_protocol(a.root))
