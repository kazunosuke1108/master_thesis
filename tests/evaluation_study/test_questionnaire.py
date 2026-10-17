"""Headless Chrome verifies the real local form; automatic answers are fixtures."""

import csv
from html import unescape
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from master_thesis_modules.evaluation_study.study import build_bundle, validate_responses
from tests.evaluation_study.test_study import manifest  # shared synthetic scenario fixture


@pytest.mark.parametrize("unable_trial", [True, False])
def test_full_questionnaire_in_real_browser(manifest, tmp_path, unable_trial):
    chrome = shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Google Chrome is unavailable; browser behavior not verified")
    bundle = build_bundle(manifest, tmp_path / "bundle")
    html_path = bundle / "participant/questionnaire.html"
    # This driver is only inserted into a disposable generated test artifact.
    driver = r'''<script>
    const unableTrial=UNABLE_TRIAL;
    const assert = (ok, message) => {if (!ok) throw Error(message)};
    try {
      assert(document.getElementById('answers').disabled,'initial disabled');
      assert(Array.from(document.querySelectorAll('.pairwise')).length===24,'24 pairs');
      assert(Array.from(document.querySelectorAll('.fuzzy')).length===12,'12 fuzzy');
      assert(Array.from(document.querySelectorAll('input,select,textarea')).filter(x=>!['checkbox'].includes(x.type)).every(x=>x.value===''),'all initially blank');
      document.getElementById('start').click();
      assert(!document.getElementById('answers').disabled,'started');
      assert(questionnaireTrial.finish()===null,'missing blocked');
      document.getElementById('respondent').value='T01';
      document.getElementById('kind').value=unableTrial?'human_trial':'synthetic_ai';
      for(const bad of ['','0','1/10','10','1/0']) {let failed=false;try{questionnaireTrial.ratio(bad)}catch(error){failed=true}assert(failed,'bad ratio rejected')}
      document.querySelectorAll('.pairwise').forEach(x=>x.value='1/3');
      assert(questionnaireTrial.finish()===null,'missing fuzzy blocked');
      document.querySelectorAll('.fuzzy').forEach(x=>x.value='3');
      assert(questionnaireTrial.finish()===null,'missing rankings blocked');
      document.querySelectorAll('.ranking').forEach((x,i)=>x.value=i<2?'1':'2');
      assert(questionnaireTrial.finish()===null,'bad tie blocked');
      document.querySelectorAll('.ranking').forEach((x,i)=>x.value=String(i+1));
      const unable=document.querySelector('.unable');
      if(unableTrial){
        unable.checked=true;unable.dispatchEvent(new Event('change'));
        assert(questionnaireTrial.finish()===null,'missing unable reason blocked');
        document.querySelector('.reason').value='情報不足';
      }
      const result=questionnaireTrial.finish();
      assert(result && result.data_kind===(unableTrial?'human_trial':'synthetic_ai') && result.synthetic===!unableTrial,'human/AI separate');
      assert(result.action_pairwise[0][1]===1/3 && result.action_pairwise[1][0]===3,'reciprocal');
      assert(result.responses.every((x,i)=>unableTrial?(x.status==='unable' && x.rank===''):(x.status==='ranked' && x.rank===i+1)),'rank/unable export');
      assert(result.timing.elapsed_ms>=0 && result.timing.started_at && result.timing.completed_at,'timing');
      assert(questionnaireTrial.finish()===result,'fixed completion');
      assert(document.getElementById('answers').disabled,'frozen');
      assert(questionnaireTrial.knowledgeRows(result).length===36,'36 exported judgements');
      const csv=questionnaireTrial.csv(result.responses,['case_id','respondent_id','patient_id','rank','status','reason']);
      const blobs=[];
      URL.createObjectURL=blob=>{blobs.push(blob);return 'blob:fixture'};
      HTMLAnchorElement.prototype.click=function(){};
      document.getElementById('json').click();document.getElementById('ranks').click();document.getElementById('knowledge').click();
      Promise.all(blobs.map(blob=>blob.text())).then(texts=>{
        assert(texts.length===3,'three downloadable files');
        assert(JSON.parse(texts[0]).respondent_id==='T01','json download');
        assert(texts[1]===csv,'csv download');
        const pre=document.createElement('pre');pre.id='browser-test-result';pre.textContent=JSON.stringify({result,csv,files:texts.length});document.body.appendChild(pre);
      });
    } catch(error) {const pre=document.createElement('pre');pre.id='browser-test-error';pre.textContent=error.stack;document.body.appendChild(pre)}
    </script>'''
    html_path.write_text(html_path.read_text() + driver.replace('UNABLE_TRIAL', str(unable_trial).lower()))
    command = [chrome, "--headless", "--no-sandbox", "--disable-gpu", "--disable-background-networking", "--no-first-run", f"--user-data-dir={tmp_path / 'chrome-profile'}", "--virtual-time-budget=1500", "--dump-dom", html_path.as_uri()]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    match = re.search(r'<pre id="browser-test-result">(.*?)</pre>', result.stdout, re.S)
    assert match, result.stdout[-4000:]
    exported = json.loads(unescape(match.group(1)))
    response = tmp_path / "responses.csv"
    response.write_text(exported["csv"])
    assert len(validate_responses(manifest, response)) == 3
    assert exported["files"] == 3
    assert len(exported["result"]["presentation_sha256"]) == 64
