import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from master_thesis_modules.evaluation_study.study import (
    FIELDS, build_bundle, evaluate_profiles, load_cases, main, validate_responses,
)


@pytest.fixture
def manifest(tmp_path):
    scenario = {
        # Deliberately matches a legacy-dispatch name: it must still use YAML.
        "scenario_name": "20260720_4patients",
        "time_range": {"duration_s": 2, "step_s": 1},
        "patients": [
            {"id": "original_a", "position": {"x": 0, "y": 0}, "age_group_label": "old", "action_label": "neutral_sitting"},
            {"id": "original_b", "position": {"x": 2, "y": 0}, "age_group_label": "old", "action_label": "neutral_sitting"},
            {"id": "original_c", "position": {"x": 4, "y": 0}, "age_group_label": "old", "action_label": "neutral_sitting"}],
        "staff": [{"id": "real_staff_source", "position": {"x": 1, "y": 2}, "velocity": {"vx": 0, "vy": -1}}],
        "objects": [{"id": "real_object_source", "type": "iv_pole", "position": {"x": 0, "y": 0}}],
        "events": [
            {"time_s": 0, "event_type": "set_action", "target_id": "original_a", "action_label": "standing_up"},
            {"time_s": 0.25, "event_type": "move_patient", "target_id": "original_b", "position": {"x": 3, "y": 0}},
            {"time_s": 1, "event_type": "set_action", "target_id": "original_c", "action_label": "touch_face"}]}
    source = tmp_path / "scenario.yaml"
    source.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    data = {"schema_version": 1, "cases": [{
        "case_id": "C01", "scenario": "scenario.yaml", "time_s": 0.25,
        "description": "指定時点の場面です。", "patient_ids": {"original_a": "P01", "original_b": "P02", "original_c": "P03"}}]}
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def write_answers(path, ranks=("1", "1", "3"), status="ranked", reason="", respondent="S01"):
    rows = [{"case_id": "C01", "respondent_id": respondent, "patient_id": f"P{i+1:02}", "rank": rank, "status": status, "reason": reason} for i, rank in enumerate(ranks)]
    save_answers(path, rows)
    return rows


def save_answers(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_boundary_and_future_events(manifest):
    _, _, cases = load_cases(manifest)
    case = cases[0]
    assert case["world"].time_s == 0.25
    assert case["frames"][0].action_label == "standing_up"
    assert case["frames"][1].patient_position.x == 3
    assert case["frames"][2].action_label == "neutral_sitting"
    assert all(f.time_s == 0.25 for f in case["frames"])


@pytest.mark.parametrize("time_s,expected_x", [(0, 2), (0.249999, 2), (0.25, 3)])
def test_event_time_not_rounded_to_scenario_step(manifest, time_s, expected_x):
    raw = yaml.safe_load(manifest.read_text())
    raw["cases"][0]["time_s"] = time_s
    manifest.write_text(yaml.safe_dump(raw))
    assert load_cases(manifest)[2][0]["frames"][1].patient_position.x == expected_x


def test_bundle_blind_and_source_tracked(manifest, tmp_path):
    output = build_bundle(manifest, tmp_path / "bundle", ["test-command"])
    public = "\n".join(p.read_text() for p in (output / "participant").iterdir())
    for forbidden in ("original_a", "original_b", "real_staff_source", "real_object_source", "total_risk", "system_rank", "system_score", "fuzzy_profile", "ahp_profile", "comparison_mtx"):
        assert forbidden not in public
    scene = json.loads((output / "participant/C01.json").read_text())
    assert scene["patients"][0]["action"] == "立ち上がる"
    assert scene["staff"][0]["velocity"]["vy"] == -1
    provenance = json.loads((output / "analysis/provenance.json").read_text())
    assert provenance["command_argv"] == ["test-command"]
    assert provenance["source_sha256"][str(manifest)] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(provenance["git_head"]) == 40
    assert "git_status_porcelain" in provenance
    assert any(p.endswith("event_engine.py") for p in provenance["source_sha256"])
    original = (output / "participant/C01.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_bundle(manifest, output)
    assert (output / "participant/C01.json").read_bytes() == original


def test_prompt_instructions_and_html_state_ranked_status(manifest, tmp_path):
    output = build_bundle(manifest, tmp_path / "bundle")
    for filename in ("C01-ai-prompt.txt", "instructions.txt", "C01.html"):
        text = (output / "participant" / filename).read_text()
        assert "全患者のstatusをranked" in text
        assert "全患者のstatusをunable" in text


def test_colocated_object_labels_stack_without_moving_markers(manifest, tmp_path):
    source = tmp_path / "scenario.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["objects"].append({"id": "wheelchair_same_position", "type": "wheelchair", "position": {"x": 0, "y": 0}})
    source.write_text(yaml.safe_dump(raw))
    bundle = build_bundle(manifest, tmp_path / "bundle")
    root = ET.fromstring((bundle / "participant/C01.svg").read_text())
    ns = {"s": "http://www.w3.org/2000/svg"}
    labels = [text for text in root.findall("s:text", ns) if text.get("class") == "object-label"]
    assert [text.text for text in labels] == ["O1 点滴台", "O2 車椅子"]
    assert labels[0].get("x") == labels[1].get("x")
    assert float(labels[1].get("y")) - float(labels[0].get("y")) == 18
    markers = [rect for rect in root.findall("s:rect", ns) if rect.get("fill") == "#777"]
    assert markers[0].attrib == markers[1].attrib
    scene = json.loads((bundle / "participant/C01.json").read_text())
    assert scene["objects"][0]["position"] == scene["objects"][1]["position"] == {"x": 0.0, "y": 0.0}


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_alias", "duplicate_case"])
def test_patient_and_case_set_integrity(manifest, mutation):
    raw = yaml.safe_load(manifest.read_text())
    case = raw["cases"][0]
    if mutation == "missing":
        del case["patient_ids"]["original_a"]
    elif mutation == "extra":
        case["patient_ids"]["nonexistent"] = "P04"
    elif mutation == "duplicate_alias":
        case["patient_ids"]["original_b"] = "P01"
    else:
        raw["cases"].append(dict(case))
    manifest.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_cases(manifest)


@pytest.mark.parametrize("time_s", [-1, 2.1, True, float("nan"), float("inf"), "0"])
def test_bad_time(manifest, time_s):
    raw = yaml.safe_load(manifest.read_text())
    raw["cases"][0]["time_s"] = time_s
    manifest.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_cases(manifest)


def test_pose_override_requires_explicit_description(manifest, tmp_path):
    path = tmp_path / "scenario.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["events"].append({"time_s": 0.1, "event_type": "set_pose_features", "target_id": "original_a", "standing_degree": 0.1, "trunk_tilt": 0.2, "wrist_distance_from_hip": 0.3, "ankle_spread": 0.4})
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="pose overrides"):
        build_bundle(manifest, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_ties_and_unable(manifest, tmp_path):
    response = tmp_path / "responses.csv"
    write_answers(response)
    assert len(validate_responses(manifest, response)) == 3
    write_answers(response, ("", "", ""), "unable", "動作が読み取れない")
    assert validate_responses(manifest, response)[0]["status"] == "unable"


@pytest.mark.parametrize("mutation", ["empty", "partial", "duplicate", "extra", "blank_respondent", "blank_rank", "mixed", "bad_tie", "negative", "unable_rank", "unable_reason", "missing_case"])
def test_invalid_answers(manifest, tmp_path, mutation):
    response = tmp_path / "responses.csv"
    rows = write_answers(response)
    if mutation == "empty":
        rows = []
    elif mutation == "partial":
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["patient_id"] = "P01"
    elif mutation == "extra":
        rows[2]["patient_id"] = "P04"
    elif mutation == "blank_respondent":
        rows[0]["respondent_id"] = ""
    elif mutation == "blank_rank":
        rows[1]["rank"] = ""
    elif mutation == "mixed":
        rows[0]["status"] = "unable"
    elif mutation == "bad_tie":
        rows[2]["rank"] = "2"
    elif mutation == "negative":
        rows[0]["rank"] = "-1"
    elif mutation == "unable_rank":
        for row in rows:
            row.update(status="unable", reason="不明")
    elif mutation == "unable_reason":
        for row in rows:
            row.update(status="unable", rank="")
    elif mutation == "missing_case":
        raw = yaml.safe_load(manifest.read_text())
        raw["cases"].append({**raw["cases"][0], "case_id": "C02"})
        manifest.write_text(yaml.safe_dump(raw))
    save_answers(response, rows)
    with pytest.raises(ValueError):
        validate_responses(manifest, response)


def make_profiles(tmp_path):
    # Synthetic neutral knowledge; never evidence about real staff.
    common = tmp_path / "knowledge"
    common.mkdir()
    for node, size in (("30000001", 7), ("30000010", 3)):
        (common / f"comparison_mtx_{node}_synthetic.csv").write_text(
            "\n".join(",".join("1" for _ in range(size)) for _ in range(size)) + "\n")
    (common / "TFN_synthetic.csv").write_text("0.25,0.5,0.75\n" * 12)
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(yaml.safe_dump({"profiles": [{"respondent_id": "S01", "common_dir": "knowledge", "ahp_profile": "synthetic", "fuzzy_profile": "synthetic"}]}))
    return profiles


@pytest.mark.parametrize("aggregation", ["weighted_sum", "weighted_max"])
def test_evaluation_join_is_separate_and_reproducible(manifest, tmp_path, aggregation):
    profiles = make_profiles(tmp_path)
    response = tmp_path / "responses.csv"
    write_answers(response)
    output = evaluate_profiles(manifest, profiles, tmp_path / "analysis-only", aggregation, ["evaluate-test"], response)
    with (output / "comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["response_rank"] == "1"
    assert all(row["time_s"] == "0.25" for row in rows)
    assert not (output / "participant").exists()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["action_aggregation"] == aggregation
    assert any(p.endswith("TFN_synthetic.csv") for p in provenance["source_sha256"])
    assert provenance["statistical_metrics"] is None
    snapshots = json.loads((output / "S01-snapshots.json").read_text())
    # Compare the exported snapshot against the exact existing risk engine API.
    from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config
    from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine
    frame = load_cases(manifest)[2][0]["frames"][0]
    engine = RiskEngine(make_profile_risk_config("synthetic", "synthetic", tmp_path / "knowledge", action_aggregation=aggregation))
    assert snapshots[0]["result"]["total_risk"] == engine.evaluate(frame).total_risk


def test_missing_profile_never_falls_back(manifest, tmp_path):
    profiles = make_profiles(tmp_path)
    (tmp_path / "knowledge/comparison_mtx_30000001_synthetic.csv").unlink()
    with pytest.raises(ValueError, match="fallback disabled"):
        evaluate_profiles(manifest, profiles, tmp_path / "analysis", "weighted_sum")
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize("bad_matrix", ["1,1\n1,1\n", "1,-1,1\n1,1,1\n1,1,1\n", "1,nan,1\n1,1,1\n1,1,1\n"])
def test_invalid_knowledge_matrix_rejected(manifest, tmp_path, bad_matrix):
    profiles = make_profiles(tmp_path)
    (tmp_path / "knowledge/comparison_mtx_30000010_synthetic.csv").write_text(bad_matrix)
    with pytest.raises(ValueError, match="AHP"):
        evaluate_profiles(manifest, profiles, tmp_path / "analysis", "weighted_sum")
    assert not (tmp_path / "analysis").exists()


def test_output_score_ties_are_not_arbitrarily_ordered(manifest, tmp_path):
    profiles = make_profiles(tmp_path)
    # At symmetric positions with identical state, preserve exact score ties.
    source = tmp_path / "scenario.yaml"
    scenario = yaml.safe_load(source.read_text())
    scenario["events"] = []
    for patient in scenario["patients"]:
        patient["position"] = {"x": 0, "y": 0}
    source.write_text(yaml.safe_dump(scenario))
    output = evaluate_profiles(manifest, profiles, tmp_path / "analysis", "weighted_sum")
    with (output / "system-ranks.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["system_rank"] for r in rows} == {"1"}
    before = (output / "system-ranks.csv").read_bytes()
    with pytest.raises(FileExistsError):
        evaluate_profiles(manifest, profiles, output, "weighted_sum")
    assert (output / "system-ranks.csv").read_bytes() == before


def test_cli_and_template_is_not_valid_response(manifest, tmp_path, capsys):
    main(["bundle", str(manifest), str(tmp_path / "bundle")])
    with pytest.raises(SystemExit) as result:
        main(["validate", str(manifest), str(tmp_path / "bundle/participant/responses-template.csv")])
    assert result.value.code == 2
    assert "respondent_id" in capsys.readouterr().err
