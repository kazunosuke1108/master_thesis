"""REQ-004 local forms: strict import, packet binding and actual browser export."""
from copy import deepcopy
import csv
from html import unescape
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from master_thesis_modules.evaluation_study.additional_evaluation import analyze_person, validate_followup
from master_thesis_modules.evaluation_study.additional_questionnaire import render_followup
from master_thesis_modules.evaluation_study.questionnaire import render_questionnaire
from master_thesis_modules.evaluation_study.study import _json, _scene, load_cases
from tests.evaluation_study.test_study import manifest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("req004_session_import", ROOT / "studies/REQ-004/import_session.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


@pytest.fixture
def session(manifest):
    scenes = [_scene(case) for case in load_cases(manifest)[2]]
    html = render_questionnaire(scenes, include_watching=True)
    config = json.loads(re.search(r'<script id="questionnaire-config" type="application/json">(.*?)</script>', html).group(1))
    return {"schema_version": importer.SCHEMA, "prototype": True, "data_kind": "synthetic_ai",
            "synthetic": True, "respondent_id": "AI-R01", "presentation_sha256": config["presentation_sha256"],
            "knowledge_item_order": config["groups"], "fuzzy_order": list(importer.ORDER),
            "action_pairwise": [[1] * 7 for _ in range(7)], "object_pairwise": [[1] * 3 for _ in range(3)],
            "fuzzy_answers": {g: [5, 4, 2, 1] for g in ("internal", "watching", "external", "total")},
            "responses": [{"case_id": "C01", "respondent_id": "AI-R01", "patient_id": f"P{i+1:02}",
                           "rank": rank, "status": "ranked", "reason": "説明前の場面判断"} for i, rank in enumerate((3, 2, 1))],
            "timing": {"started_at": "2026-10-18T00:00:00Z", "completed_at": "2026-10-18T00:08:00Z",
                       "elapsed_ms": 480000, "includes_reading_and_breaks": True,
                       "target_ms": None, "target_achieved": None}}


def _import(session, manifest, tmp_path):
    source = tmp_path / "session-input.json"
    source.write_text(json.dumps(session, ensure_ascii=False))
    return source, importer.import_session(source, manifest, tmp_path / "imported")


@pytest.mark.parametrize("kind", ["synthetic_ai", "human_trial"])
def test_import_keeps_original_complete_session(session, manifest, tmp_path, kind):
    session["data_kind"], session["synthetic"] = kind, kind == "synthetic_ai"
    source, output = _import(session, manifest, tmp_path)
    assert source.read_bytes() == (output / "session.json").read_bytes()
    knowledge = json.loads((output / "knowledge.json").read_text())
    assert knowledge["fuzzy_answers"] == session["fuzzy_answers"]
    assert knowledge["synthetic"] == session["synthetic"]
    assert list(csv.DictReader((output / "initial-responses.csv").open()))[0]["rank"] == "3"
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance["timing"]["target_ms"] is None
    with pytest.raises(FileExistsError):
        importer.import_session(source, manifest, output)


@pytest.mark.parametrize("bad", ["old_schema", "synthetic", "kind", "missing_watching", "wrong_fuzzy_order", "wrong_items",
                                     "nan", "boolean", "nonreciprocal", "missing_patient", "extra_respondent", "reason",
                                     "target", "missing_target", "elapsed", "clock", "fingerprint", "bool_rank", "identity"])
def test_invalid_import_never_creates_output(session, manifest, tmp_path, bad):
    if bad == "old_schema": session["schema_version"] = "REQ003-full-session-trial-v1"
    if bad == "synthetic": session["synthetic"] = False
    if bad == "kind": session["data_kind"] = "human"
    if bad == "missing_watching": del session["fuzzy_answers"]["watching"]
    if bad == "wrong_fuzzy_order": session["fuzzy_order"].reverse()
    if bad == "wrong_items": session["knowledge_item_order"]["action"].reverse()
    if bad == "nan": session["fuzzy_answers"]["watching"][0] = float("nan")
    if bad == "boolean": session["fuzzy_answers"]["watching"][0] = True
    if bad == "nonreciprocal": session["action_pairwise"][0][1] = 2
    if bad == "missing_patient": session["responses"].pop()
    if bad == "extra_respondent": session["responses"][0]["respondent_id"] = "Other"
    if bad == "reason": session["responses"][0]["reason"] = " "
    if bad == "target": session["timing"]["target_ms"] = 300000
    if bad == "missing_target": del session["timing"]["target_ms"]
    if bad == "elapsed": session["timing"]["elapsed_ms"] = True
    if bad == "clock": session["timing"]["started_at"] = "2026-10-18T00:00:00"
    if bad == "fingerprint": session["presentation_sha256"] = "0" * 64
    if bad == "bool_rank": session["responses"][0]["rank"] = True
    if bad == "identity": session["respondent_id"] = "../../person"
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(session))
    with pytest.raises(ValueError):
        importer.import_session(source, manifest, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def _browser(html, directory, driver):
    chrome = shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Google Chrome unavailable; real browser behavior unverified")
    path = directory / "form.html"
    path.write_text(html + '<script>' + driver + '</script>', encoding="utf-8")
    result = subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--disable-background-networking",
                             "--no-first-run", f"--user-data-dir={directory / 'chrome'}", "--virtual-time-budget=1500",
                             "--dump-dom", path.as_uri()], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    match = re.search(r'<pre id="exported">(.*?)</pre>', result.stdout, re.S)
    assert match, result.stdout[-5000:]
    return json.loads(unescape(match.group(1)))


def test_req004_browser_exports_40_answers_and_unlimited_session(manifest, tmp_path):
    scenes = [_scene(case) for case in load_cases(manifest)[2]]
    html = render_questionnaire(scenes, include_watching=True)
    shutil.copy(ROOT / "master_thesis_modules/evaluation_study/questionnaire.js", tmp_path / "questionnaire.js")
    driver = r'''
    const assert=(x,m)=>{if(!x)throw Error(m)};
    try{
      assert(document.querySelectorAll('.fuzzy').length===16,'16 fuzzy');
      assert(document.querySelectorAll('.pairwise').length===24,'24 pairwise');
      document.getElementById('start').click();
      assert(questionnaireTrial.finish()===null,'empty blocked');
      document.getElementById('respondent').value='AI-R01';document.getElementById('kind').value='synthetic_ai';
      document.querySelectorAll('.pairwise').forEach(x=>x.value='1/3');
      document.querySelectorAll('.fuzzy').forEach(x=>x.value='4');
      document.querySelectorAll('.ranking').forEach((x,i)=>x.value=String(i+1));
      assert(questionnaireTrial.finish()===null,'initial reason required');
      document.querySelectorAll('.reason').forEach(x=>x.value='説明を見る前の判断です');
      document.querySelector('.fuzzy[data-group="watching"]').value='';
      assert(questionnaireTrial.finish()===null,'watching missing blocked');
      document.querySelector('.fuzzy[data-group="watching"]').value='5';
      const result=questionnaireTrial.finish();assert(result,'complete');
      assert(result.schema_version==='REQ004-full-session-trial-v1','REQ004 schema');
      assert(result.timing.target_ms===null && result.timing.target_achieved===null,'no time limit');
      assert(questionnaireTrial.knowledgeRows(result).length===40,'40 exported answers');
      const blobs=[];URL.createObjectURL=b=>{blobs.push(b);return 'blob:test'};HTMLAnchorElement.prototype.click=function(){};
      document.getElementById('json').click();document.getElementById('knowledge').click();
      Promise.all(blobs.map(b=>b.text())).then(files=>{
        assert(JSON.parse(files[0]).fuzzy_answers.watching.length===4,'download 16 answers');
        assert(files[1].trim().split('\n').length===41,'CSV 40 answers plus header');
        const pre=document.createElement('pre');pre.id='exported';pre.textContent=JSON.stringify({result,files});document.body.appendChild(pre);
      });
    }catch(e){const pre=document.createElement('pre');pre.id='failed';pre.textContent=e.stack;document.body.appendChild(pre)}
    '''
    exported = _browser(html, tmp_path, driver)
    source = tmp_path / "actual-browser-session.json"
    source.write_text(json.dumps(exported["result"]))
    imported = importer.import_session(source, manifest, tmp_path / "browser-import")
    assert json.loads((imported / "knowledge.json").read_text())["fuzzy_answers"]["watching"] == [5, 4, 4, 4]


@pytest.fixture
def packet(session, manifest, tmp_path):
    _, imported = _import(session, manifest, tmp_path)
    analyze_person(manifest, imported / "knowledge.json", imported / "initial-responses.csv", tmp_path / "analysis")
    path = tmp_path / "analysis/followup-packet.json"
    data = json.loads(path.read_text())
    assert data["cases"], "Fixture must exercise a mismatch"
    return path, data


def test_followup_actual_browser_download_validates_and_binds_pair_vs_whole(packet, tmp_path):
    path, data = packet
    html = render_followup(data)
    assert "1：全く参考にならない" in html
    assert "5：とても参考になる" in html
    for case in data["cases"]:
        assert case["case_explanation"] in unescape(html)
        for pair in case["pairs"]:
            assert pair["explanation"] in unescape(html)
            assert "回答" in pair["explanation"]
    for debug in ("hybrid_scores", "local_representatives", "packet_sha256_scope", "Shapley", "fuzzy_rule_results"):
        assert debug not in html
    driver = r'''
    const assert=(x,m)=>{if(!x)throw Error(m)};
    try{
      assert(followupTrial.finish()===null,'missing blocked');
      document.querySelectorAll('.response').forEach(x=>{
        x.querySelector('.helpfulness').value=x.dataset.scope==='whole'?'5':'null';
        x.querySelector('.acceptance').value=x.dataset.scope==='whole'?'reject':'partial';
        x.querySelector('.reason').value='比較と場面全体では判断が異なります';
      });
      const result=followupTrial.finish();assert(result,'complete');
      assert(result.cases[0].helpfulness===5 && result.cases[0].acceptance==='reject','whole independent');
      assert(result.cases[0].pairs.every(p=>p.helpfulness===null && p.acceptance==='partial'),'pair independent');
      const blobs=[];URL.createObjectURL=b=>{blobs.push(b);return 'blob:test'};HTMLAnchorElement.prototype.click=function(){};
      document.getElementById('download').click();
      blobs[0].text().then(text=>{const pre=document.createElement('pre');pre.id='exported';pre.textContent=JSON.stringify({result,file:JSON.parse(text)});document.body.appendChild(pre)});
    }catch(e){const pre=document.createElement('pre');pre.id='failed';pre.textContent=e.stack;document.body.appendChild(pre)}
    '''
    exported = _browser(html, tmp_path, driver)
    assert exported["result"] == exported["file"]
    assert exported["file"]["packet_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    answers = tmp_path / "followup-answers.json"
    answers.write_text(json.dumps(exported["file"]))
    validate_followup(path, answers, tmp_path / "accepted")
    changed = deepcopy(exported["file"])
    changed["packet_sha256"] = "0" * 64
    answers.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="binding"):
        validate_followup(path, answers, tmp_path / "wrong-binding")


def test_render_never_executes_packet_text_as_markup(packet):
    _, data = packet
    malicious = '</script><script>window.evil=true</script><img src=x onerror="window.evil=true">'
    data["cases"][0]["answer_explanations"].append(malicious)
    data["cases"][0]["pairs"][0]["explanation"] += '\n' + malicious
    data["cases"][0]["scene"]["description"] = malicious
    html = render_followup(data)
    assert malicious not in html
    assert '&lt;/script&gt;' in html


def test_agreement_has_no_positive_followup_defaults(tmp_path):
    packet = {"schema_version": 1, "respondent_id": "S01", "cases": [],
              "summary_cases": [{"case_id": "C01", "followup_status": "not_asked_agreement"}]}
    html = render_followup(packet)
    assert 'class="response"' not in html
    assert '未質問' in html
    data = _browser(html, tmp_path, """
      const result=followupTrial.finish();const pre=document.createElement('pre');pre.id='exported';pre.textContent=JSON.stringify(result);document.body.appendChild(pre);
    """)
    assert data["cases"] == []
