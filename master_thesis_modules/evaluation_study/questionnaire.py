"""Render a local full-session questionnaire without system outputs."""

from html import escape
import hashlib
import json

from master_thesis_modules.risk_core.schema import node_ids
from master_thesis_modules.risk_core.schema.node_labels import NODE_LABELS


def render_questionnaire(scenes, *, include_watching=False):
    groups = {
        "action": [NODE_LABELS[node] for node in node_ids.ACTION_RISK_NODES],
        "object": [NODE_LABELS[node] for node in node_ids.OBJECT_RISK_NODES],
    }
    sections = []
    for group, labels in groups.items():
        questions = []
        for i, left in enumerate(labels):
            for j in range(i+1, len(labels)):
                questions.append(
                    f'<label>{escape(left)} ／ {escape(labels[j])} '
                    f'<input data-group="{group}" data-i="{i}" data-j="{j}" '
                    'class="pairwise" type="text" inputmode="decimal" required aria-label="比較値"></label>')
        sections.append('<h2>' + ('動作の一対比較（21問）' if group == 'action' else '物体の一対比較（3問）') + '</h2>' + ''.join(questions))
    fuzzy_labels = {
        'internal': ('患者属性・年齢による危険性', '動作による危険性'),
        'external': ('周囲の物体による危険性', 'スタッフとの距離・見守り状況による危険性'),
        'total': ('患者属性・年齢・動作をまとめた危険性', '周囲の物体・スタッフをまとめた危険性'),
    }
    if include_watching:
        fuzzy_labels = {'internal': fuzzy_labels['internal'],
                        'watching': ('スタッフからの距離による危険性', '患者がスタッフの視野外にいることによる危険性'),
                        'external': fuzzy_labels['external'], 'total': fuzzy_labels['total']}
    for group, (first, second) in fuzzy_labels.items():
        questions = []
        for index, levels in enumerate(('HH', 'HL', 'LH', 'LL')):
            high_low = {'H': '高い', 'L': '低い'}
            label = f'{first}が{high_low[levels[0]]}、{second}が{high_low[levels[1]]}場合、まとめた危険性は？'
            options = '<option value="">選択してください</option>' + ''.join(f'<option value="{i}">{i}</option>' for i in range(1, 6))
            questions.append(f'<label>{escape(label)}<select class="fuzzy" data-group="{group}" data-index="{index}" required>{options}</select></label>')
        sections.append('<h2>組合せ判断（各4問）</h2>' + ''.join(questions))
    for scene in scenes:
        case_id = scene['case_id']
        questions = ''.join(
            f'<label>{escape(patient["patient_id"])}の順位 '
            f'<input class="ranking" data-case="{case_id}" data-patient="{patient["patient_id"]}" type="number" min="1" step="1" required></label>'
            for patient in scene['patients'])
        sections.append(
            f'<section class="case" data-case="{case_id}"><h2>場面 {case_id}</h2>'
            f'<iframe title="場面 {case_id}" src="{case_id}.html"></iframe>'
            f'<label><input type="checkbox" class="unable" data-case="{case_id}">この場面全体を判断できない</label>'
            f'{questions}<label>' + ('順位を決めた理由、または判断不能の理由（必須）' if include_watching else '判断不能の理由（判断不能の場合は必須）') + f'<textarea class="reason" data-case="{case_id}"></textarea></label></section>')
    fingerprint = hashlib.sha256(json.dumps(scenes, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    config = json.dumps({'schema_version': 'REQ004-full-session-trial-v1' if include_watching else 'REQ003-full-session-trial-v1', 'fuzzy_groups': list(fuzzy_labels), 'target_ms': None if include_watching else 300000, 'presentation_sha256': fingerprint, 'groups': groups, 'cases': [{'case_id': s['case_id'], 'patients': [p['patient_id'] for p in s['patients']]} for s in scenes]}, ensure_ascii=False).replace('<', '\\u003c')
    introduction = ('知識回答40問（動作比較21問・物体比較3問・組合せ16問）と順位を回答します。所要時間の上限は設けていません。見守りの組合せでは、距離が遠いほど、また患者がスタッフの視野外にいるほど、それぞれの危険性が高い条件を指します。近い・視野内は低い条件です。順位を決めた理由も、説明を見る前に記入してください。' if include_watching else '知識回答36問（動作比較21問・物体比較3問・組合せ12問）と全場面の順位回答を、開始から完了まで通して測ります。説明を読む時間・途中の休憩も含みます。5分以内に終わるかは未検証です。')
    return '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>知識回答・順位回答の通し試作</title>
<style>body{max-width:1000px;margin:2em auto;font-family:sans-serif}label{display:block;margin:1em 0}input,select,textarea{margin-left:1em}iframe{width:100%;height:760px;border:1px solid #aaa}#error{color:#a00}button{margin:1em;padding:.7em}</style>
<h1>知識回答と瞬間順位の通し試作</h1>
<p>本調査の確定版ではありません。実スタッフへの配布前の試作です。''' + introduction + '''</p>
<p>通信・アップロードはありません。個人名は入力しないでください。開始ボタンで時計が動き、未回答がある場合は完了できません。保存データには本人の判断知識と順位が含まれます。</p>
<button id="start" type="button">始める</button><span id="timer">未開始</span>
<p id="error" role="alert"></p><form id="questionnaire"><fieldset id="answers" disabled>
<label>匿名回答者ID <input id="respondent" type="text" pattern="[A-Za-z0-9_-]+" required></label>
<label>回答の種類 <select id="kind" required><option value="">選択してください</option><option value="synthetic_ai">AIによる予行（synthetic_ai）</option><option value="human_trial">人による試行（human_trial、本調査ではない）</option></select></label>
<p>一対比較は、左の項目が右の項目に比べて見守りの優先度へ寄与する程度を1/9〜9で答えてください。1は同程度、1より大きい値は左を重視、1より小さい値は右を重視します。1/3などの分数または小数で入力できます。逆向きの値は逆数として保存します。組合せ判断は1（低い）〜5（高い）で答えてください。</p>
''' + ''.join(sections) + '''<button id="finish" type="button">回答を完了して時間を記録</button></fieldset></form>
<div id="downloads" hidden><p>回答を確定しました。以下のファイルを保存してください。時間は完了ボタン時点で固定されています。</p><button id="json" type="button">全回答JSONを保存</button><button id="ranks" type="button">順位CSVを保存</button><button id="knowledge" type="button">知識CSVを保存</button></div>
<script id="questionnaire-config" type="application/json">''' + config + '''</script><script src="questionnaire.js"></script></html>'''
