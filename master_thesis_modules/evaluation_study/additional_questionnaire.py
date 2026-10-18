"""Standalone local follow-up form; research diagnostics remain out of its UI."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

from .study import _identifier, _json, _svg


def _questions(scope, label):
    helpfulness = '<option value="">選択してください</option>' + ''.join(
        f'<option value="{i}">{i}：{escape(text)}</option>' for i, text in enumerate(
            ("全く参考にならない", "あまり参考にならない", "どちらともいえない", "やや参考になる", "とても参考になる"), 1)
    ) + '<option value="null">この説明だけでは判断できない</option>'
    acceptance = '<option value="">選択してください</option>' + ''.join(
        f'<option value="{value}">{text}</option>' for value, text in (
            ("accept", "受け入れる"), ("partial", "一部を受け入れる"),
            ("reject", "受け入れない"), ("uncertain", "判断を保留する")))
    return (f'<fieldset class="response" data-scope="{escape(scope)}"><legend>{escape(label)}</legend>'
            '<label>説明は、ご自身の見守りの判断を考える参考になりますか。'
            f'<select class="helpfulness" required>{helpfulness}</select></label>'
            '<label>算出された順を受け入れられますか。'
            f'<select class="acceptance" required>{acceptance}</select></label>'
            '<label>理由を教えてください。アンケートで重視したことと、この場面の順位を決める際に'
            '重視したことに違いがあれば、それも含めてお書きください。違いがない場合も、'
            'そのままお答えください。<textarea class="reason" required></textarea></label></fieldset>')


def render_followup(packet: dict, *, packet_sha256: str | None = None) -> str:
    """Render a pipeline packet; its canonical serialization supplies the binding.

    Pipeline-generated packets use study._json. Callers reading a differently
    formatted packet file must pass that file's SHA256 explicitly (CLI does so).
    """
    if not isinstance(packet, dict) or packet.get("schema_version") != 1:
        raise ValueError("Follow-up packet schema_version must be 1")
    respondent = _identifier(packet.get("respondent_id"), "respondent_id")
    binding = packet_sha256 or hashlib.sha256(_json(packet).encode()).hexdigest()
    if len(binding) != 64 or any(c not in "0123456789abcdef" for c in binding):
        raise ValueError("packet_sha256 must be a lowercase SHA256 digest")
    cases = packet.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Packet cases must be a list")
    sections, contracts, seen = [], [], set()
    for case in cases:
        case_id = _identifier(case.get("case_id"), "case_id")
        if case_id in seen:
            raise ValueError("Duplicate case")
        seen.add(case_id)
        initial, system = case["initial_ranks"], case["system_ranks"]
        if not isinstance(initial, dict) or set(initial) != set(system) or initial == system:
            raise ValueError("Follow-up cases must contain differing complete rankings")
        pairs, pair_ids = [], []
        for pair in case["pairs"]:
            pair_id = _identifier(pair.get("pair_id"), "pair_id")
            if pair_id in pair_ids:
                raise ValueError("Duplicate pair")
            pair_ids.append(pair_id)
            first, second = pair["patient_ids"]
            if first not in initial or second not in initial or first == second:
                raise ValueError("Pair patients must belong to the case")
            explanation = pair.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError("Each patient pair requires the pipeline explanation")
            pairs.append(f'<section><h3>{escape(first)}と{escape(second)}の比較</h3>'
                         + f'<div class="pair-explanation" style="white-space:pre-wrap">{escape(explanation)}</div>'
                         + _questions(pair_id, f'{first}と{second}の2人の順について') + '</section>')
        scene = case.get("scene")
        figure = _svg(scene) if scene else ""
        table = '<table><tr><th>患者</th><th>最初の回答</th><th>算出された順位</th></tr>' + ''.join(
            f'<tr><td>{escape(str(p))}</td><td>{escape(str(initial[p]))}位</td><td>{escape(str(system[p]))}位</td></tr>'
            for p in sorted(initial)) + '</table>'
        sections.append(f'<section class="case" data-case="{escape(case_id)}"><h2>場面 {escape(case_id)}</h2>'
                        + (f'<p>{escape(str(scene.get("description", "")))}</p>' if scene else "")
                        + figure + table
                        + ''.join(pairs) + '<p>' + escape(str(case.get("case_explanation", ""))) + '</p>'
                        + '<p>2人ずつの比較とは別に、この場面の患者全体の順についてお答えください。</p>'
                        + _questions("whole", "この場面の患者全体の順位について") + '</section>')
        contracts.append({"case_id": case_id, "pairs": pair_ids})
    config = _json({"schema_version": 1, "respondent_id": respondent, "packet_sha256": binding, "cases": contracts}).replace("<", "\\u003c")
    introduction = ('算出された順位と最初の回答が異なる場面について、説明を読んでお答えください。'
                    '参考になるかどうかと、算出順位を受け入れるかどうかは、別々にお答えください。'
                    '賛成・反対・判断保留のいずれも、そのまま記入してください。最初の順位回答は変更しません。')
    if not cases:
        introduction = '追加質問の対象となる場面はありません。一致した場面の参考度・受容は未質問として残します。'
    return '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>見守り順位についての追加質問</title>
<style>body{max-width:900px;margin:2em auto;font-family:sans-serif;line-height:1.7}label{display:block;margin:1em}select,textarea{display:block;max-width:95%;min-width:20em}textarea{width:90%;height:6em}fieldset{margin:1em 0}table{border-collapse:collapse}td,th{border:1px solid #999;padding:.5em}svg{max-width:100%;max-height:500px}#error{color:#a00}button{padding:.8em}</style>
<h1>見守り順位についての追加質問</h1><p>''' + escape(introduction) + '''</p>
<p>通信や自動送信はありません。所要時間の上限は設けていません。回答を完了したらJSONファイルを保存してください。</p>
''' + ''.join(sections) + '''<p id="error" role="alert"></p><button id="finish">回答を確定する</button><button id="download" hidden>追加回答JSONを保存</button>
<script id="followup-config" type="application/json">''' + config + '''</script><script>
"use strict";
const contract=JSON.parse(document.getElementById('followup-config').textContent);
let completed=null;
function record(section){
 const selected=section.querySelector('.helpfulness').value;
 const acceptance=section.querySelector('.acceptance').value;
 const reason=section.querySelector('.reason').value.trim();
 if(!selected || !acceptance || !reason)throw Error('各比較と場面全体について、参考度・受容・理由を記入してください。');
 return {helpfulness:selected==='null'?null:Number(selected),acceptance,reason};
}
function finish(){
 if(completed)return completed;
 try{
  const cases=contract.cases.map(c=>{
   const section=Array.from(document.querySelectorAll('.case')).find(x=>x.dataset.case===c.case_id);
   const find=scope=>Array.from(section.querySelectorAll('.response')).find(x=>x.dataset.scope===scope);
   return {case_id:c.case_id,...record(find('whole')),pairs:c.pairs.map(p=>({pair_id:p,...record(find(p))}))};
  });
  completed={schema_version:1,respondent_id:contract.respondent_id,packet_sha256:contract.packet_sha256,cases};
  document.querySelectorAll('.response').forEach(x=>x.disabled=true);
  document.getElementById('finish').disabled=true;document.getElementById('download').hidden=false;
  document.getElementById('error').textContent='';return completed;
 }catch(error){document.getElementById('error').textContent=error.message;return null;}
}
function download(){if(!completed)return;const url=URL.createObjectURL(new Blob([JSON.stringify(completed,null,2)],{type:'application/json;charset=utf-8'}));const anchor=document.createElement('a');anchor.href=url;anchor.download=contract.respondent_id+'-followup.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
document.getElementById('finish').addEventListener('click',finish);
document.getElementById('download').addEventListener('click',download);
window.followupTrial={finish,getResult:()=>completed};
</script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    raw = args.packet.read_bytes()
    html = render_followup(json.loads(raw), packet_sha256=hashlib.sha256(raw).hexdigest())
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
