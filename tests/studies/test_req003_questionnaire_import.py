"""Contract fixtures are synthetic, including the human_trial-labelled fixture."""
import copy
import csv
from html import unescape
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import yaml

from master_thesis_modules.evaluation_study.study import build_bundle, evaluate_profiles, validate_responses
from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config
from tests.evaluation_study.test_study import manifest

SCRIPT = Path(__file__).parents[2] / "studies/REQ-003/import_questionnaire.py"
spec = importlib.util.spec_from_file_location("req003_form_import", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def form_fixture(manifest, tmp_path, kind):
    """Use actual renderer config, not a separately handwritten item order/hash."""
    output = build_bundle(manifest, tmp_path / "bundle")
    text = (output / "participant/questionnaire.html").read_text()
    config = json.loads(re.search(r'<script id="questionnaire-config" type="application/json">(.*?)</script>', text).group(1))
    matrices = []
    for n in (7, 3):
        matrices.append([[1 if i == j else 1/3 if i < j else 3 for j in range(n)] for i in range(n)])
    return {"schema_version": module.SCHEMA, "prototype": True, "data_kind": kind,
            "synthetic": kind == "synthetic_ai", "respondent_id": "Anon_01-test",
            "presentation_sha256": config["presentation_sha256"], "knowledge_item_order": config["groups"],
            "action_pairwise": matrices[0], "object_pairwise": matrices[1], "fuzzy_order": module.ORDER,
            "fuzzy_answers": {"internal": [5,4,3,1], "external": [1,2,3,4], "total": [5,3,2,1]},
            "responses": [{"case_id": "C01", "respondent_id": "Anon_01-test", "patient_id": f"P{i+1:02}", "rank": x, "status": "ranked", "reason": "fixture"} for i, x in enumerate([1,1,3])],
            "timing": {"started_at": "2026-10-17T01:00:00.000Z", "completed_at": "2026-10-17T01:06:00.000Z", "elapsed_ms": 360000, "includes_reading_and_breaks": True, "target_ms": 300000, "target_achieved": False}}


@pytest.mark.parametrize("kind", ["synthetic_ai", "human_trial"])
def test_contract_import_profile_evaluate_roundtrip(manifest, tmp_path, kind):
    data = form_fixture(manifest, tmp_path, kind)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(data))
    original = source.read_bytes()
    output = module.import_session(source, manifest, tmp_path / "imported")
    assert source.read_bytes() == (output / "session.json").read_bytes() == original
    profile = yaml.safe_load((output / "profiles.yaml").read_text())["profiles"][0]
    assert profile["data_kind"] == kind and profile["synthetic"] == (kind == "synthetic_ai")
    config = make_profile_risk_config(data["respondent_id"], data["respondent_id"], output / "knowledge")
    assert config.fuzzy_rule_results[20000000] == (1,.75,.5,0)
    assert config.fuzzy_rule_results[20000001] == (0,.25,.5,.75)
    assert config.fuzzy_rule_results[10000000] == (1,.5,.25,0)
    for name, key in (("30000001", "action_pairwise"), ("30000010", "object_pairwise")):
        with (output / "knowledge" / f'comparison_mtx_{name}_{data["respondent_id"]}.csv').open() as handle:
            assert [[float(x) for x in row] for row in csv.reader(handle)] == data[key]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["timing"] == data["timing"]
    assert provenance["presentation_sha256"] == data["presentation_sha256"]
    assert provenance["data_kind"] == kind
    assert len(validate_responses(manifest, output / "responses.csv")) == 3
    for aggregation in ("weighted_sum", "weighted_max"):
        evaluated = evaluate_profiles(manifest, output / "profiles.yaml", tmp_path / aggregation, aggregation, responses=output / "responses.csv")
        rows = list(csv.DictReader((evaluated / "comparison.csv").open()))
        assert len(rows) == 3 and [r["response_rank"] for r in rows] == ["1","1","3"]
    with pytest.raises(FileExistsError):
        module.import_session(source, manifest, output)
    assert source.read_bytes() == original


@pytest.mark.parametrize("bad", ["schema", "prototype", "kind", "synthetic", "identity", "item_order", "fuzzy_order", "matrix_shape", "matrix_nonreciprocal", "matrix_nan", "matrix_bool", "matrix_range", "fuzzy_missing", "fuzzy_fraction", "timing_nan", "timing_target", "timing_order", "timing_timezone", "hash", "missing_patient", "missing_case", "other_respondent", "duplicate_patient", "bad_tie", "bad_rank", "unknown_status"])
def test_invalid_session_creates_no_output(manifest, tmp_path, bad):
    data = form_fixture(manifest, tmp_path, "synthetic_ai")
    if bad == "schema": data["schema_version"] = "other"
    if bad == "prototype": data["prototype"] = False
    if bad == "kind": data["data_kind"] = "study"
    if bad == "synthetic": data["synthetic"] = False
    if bad == "identity": data["respondent_id"] = "../named"
    if bad == "item_order": data["knowledge_item_order"]["action"].reverse()
    if bad == "fuzzy_order": data["fuzzy_order"] = list(reversed(module.ORDER))
    if bad == "matrix_shape": data["action_pairwise"].pop()
    if bad == "matrix_nonreciprocal": data["object_pairwise"][0][1] = 2
    if bad == "matrix_nan": data["action_pairwise"][0][1] = float("nan")
    if bad == "matrix_bool": data["action_pairwise"][0][0] = True
    if bad == "matrix_range": data["object_pairwise"][0][1] = 10
    if bad == "fuzzy_missing": data["fuzzy_answers"]["external"].pop()
    if bad == "fuzzy_fraction": data["fuzzy_answers"]["internal"][0] = 2.5
    if bad == "timing_nan": data["timing"]["elapsed_ms"] = float("inf")
    if bad == "timing_target": data["timing"]["target_achieved"] = True
    if bad == "timing_order": data["timing"]["completed_at"] = "2026-10-16T01:00:00Z"
    if bad == "timing_timezone": data["timing"]["started_at"] = "2026-10-17T01:00:00"
    if bad == "hash": data["presentation_sha256"] = "0" * 64
    if bad == "missing_patient": data["responses"].pop()
    if bad == "missing_case": data["responses"] = []
    if bad == "other_respondent": data["responses"][0]["respondent_id"] = "Other"
    if bad == "duplicate_patient": data["responses"].append(copy.deepcopy(data["responses"][0]))
    if bad == "bad_tie": data["responses"][2]["rank"] = 2
    if bad == "bad_rank": data["responses"][0]["rank"] = True
    if bad == "unknown_status": data["responses"][0]["status"] = "answered"
    source = tmp_path / "source.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.import_session(source, manifest, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_unable_and_cli(manifest, tmp_path):
    data = form_fixture(manifest, tmp_path, "human_trial")
    for row in data["responses"]:
        row.update(rank="", status="unable", reason="情報不足")
    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(data))
    result = subprocess.run([sys.executable, str(SCRIPT), str(source), str(manifest), str(tmp_path / "cli")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(r["status"] == "unable" for r in validate_responses(manifest, tmp_path / "cli/responses.csv"))


@pytest.mark.parametrize("kind", ["synthetic_ai", "human_trial"])
def test_real_browser_json_to_import_and_evaluate(manifest, tmp_path, kind):
    chrome = shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Chrome unavailable; real-browser export not verified")
    bundle = build_bundle(manifest, tmp_path / "browser-bundle")
    html = bundle / "participant/questionnaire.html"
    driver = '''<script>
    document.getElementById('start').click();
    document.getElementById('respondent').value='Browser01';
    document.getElementById('kind').value=KIND;
    document.querySelectorAll('.pairwise').forEach(x=>x.value='1/3');
    document.querySelectorAll('.fuzzy').forEach((x,i)=>x.value=String(i%5+1));
    document.querySelectorAll('.ranking').forEach((x,i)=>x.value=String(i+1));
    const result=questionnaireTrial.finish();
    const pre=document.createElement('pre');pre.id='session-export';pre.textContent=JSON.stringify(result);document.body.appendChild(pre);
    </script>'''.replace("KIND", json.dumps(kind))
    html.write_text(html.read_text() + driver)
    completed = subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--disable-background-networking", "--no-first-run", f"--user-data-dir={tmp_path / 'chrome-profile'}", "--virtual-time-budget=1000", "--dump-dom", html.as_uri()], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    match = re.search(r'<pre id="session-export">(.*?)</pre>', completed.stdout, re.S)
    assert match, completed.stdout[-4000:]
    data = json.loads(unescape(match.group(1)))
    assert data is not None and data["data_kind"] == kind
    source = tmp_path / "browser-session.json"
    source.write_text(json.dumps(data))
    output = module.import_session(source, manifest, tmp_path / "browser-import")
    evaluated = evaluate_profiles(manifest, output / "profiles.yaml", tmp_path / "browser-evaluation", "weighted_sum", responses=output / "responses.csv")
    rows = list(csv.DictReader((evaluated / "comparison.csv").open()))
    assert len(rows) == 3 and all(row["respondent_id"] == "Browser01" for row in rows)
    config = make_profile_risk_config("Browser01", "Browser01", output / "knowledge")
    for group, node in (("internal",20000000),("external",20000001),("total",10000000)):
        assert config.fuzzy_rule_results[node] == tuple((x-1)/4 for x in data["fuzzy_answers"][group])
