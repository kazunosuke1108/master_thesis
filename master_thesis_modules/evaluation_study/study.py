"""Explicit semantic scenarios, blind presentation, and separate analysis outputs.

This is a questionnaire prototype, not a fixed study design or a statistical
evaluation. Existing scenario/risk implementations are used without modification.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
from html import escape
import io
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

from master_thesis_modules.scenario_sim.encoder.scenario_loader import ScenarioLoader
from master_thesis_modules.scenario_sim.encoder.feature_encoder import FeatureEncoder
from master_thesis_modules.scenario_sim.events.event_engine import EventEngine
from master_thesis_modules.evaluation_study.questionnaire import render_questionnaire


REPO = Path(__file__).resolve().parents[2]
FIELDS = ["case_id", "respondent_id", "patient_id", "rank", "status", "reason"]
NOTICE = "試作・候補。実スタッフ調査の確定版ではありません。"
QUESTION = "この瞬間に、追加の見守りを優先すべき患者の順を答えてください。"
INSTRUCTIONS = (
    "順位を回答するときは、全患者のstatusをrankedにしてください。"
    "rankは1からの整数で、小さいほど優先。同順位は同じ整数とし、"
    "例: 1,1,3（競争順位）。全患者を記入してください。判断できない場面は、"
    "全患者のstatusをunable、rankを空欄にし、reasonを記入してください。"
    "一部患者だけの順位や、空欄を同順位と解釈する回答は受け付けません。"
)
ACTIONS = {
    "neutral_sitting": "座っている", "reach_floor": "床へ手を伸ばす",
    "reach_table": "机へ手を伸ばす", "reach_iv_pole": "点滴台へ手を伸ばす",
    "standing_up": "立ち上がる", "release_brake": "車椅子のブレーキを解除する",
    "move_wheelchair": "車椅子を動かす", "lose_balance": "バランスを崩す",
    "raise_hands": "手を動かす", "cough_up": "むせる", "touch_face": "顔に触れる",
}
OBJECTS = {"iv_pole": "点滴台", "wheelchair": "車椅子", "handrail": "手すり"}
AGES = {"young": "若年", "middle": "中年", "old": "高齢"}


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{label}: use a nonempty anonymous ASCII identifier")
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(rows, fields=FIELDS):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path}")
    return value


def load_cases(manifest_path):
    """Return exact semantic snapshots; no legacy scenario-name dispatch."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != 1 or not manifest.get("cases"):
        raise ValueError("manifest requires schema_version: 1 and nonempty cases")
    cases, seen = [], set()
    for raw in manifest["cases"]:
        case_id = _identifier(raw["case_id"], "case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        time_s = raw["time_s"]
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
            raise ValueError("time_s must be a finite number")
        source = (manifest_path.parent / raw["scenario"]).resolve()
        initial = ScenarioLoader().load(source)
        if not (initial.time_s <= time_s <= initial.duration_s):
            raise ValueError(f"time_s outside scenario time range: {case_id}")
        original_ids = [p.person_id for p in initial.patients]
        if not original_ids or len(set(original_ids)) != len(original_ids):
            raise ValueError("Scenario must contain unique patient IDs")
        for people in (initial.staff, initial.objects):
            ids = [getattr(p, "staff_id", getattr(p, "object_id", None)) for p in people]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate staff/object IDs in scenario")
        mapping = raw["patient_ids"]
        if not isinstance(mapping, dict) or set(mapping) != set(original_ids):
            raise ValueError(f"patient_ids must match the complete scenario patient set: {case_id}")
        aliases = [_identifier(p, "patient_id") for p in mapping.values()]
        if len(set(aliases)) != len(aliases):
            raise ValueError("Duplicate anonymous patient ID")
        description = raw.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Each case requires a nonempty description")
        current = EventEngine().apply(
            replace(initial, events=()),
            tuple(event for event in initial.events if event.time_s <= time_s),
        )
        current = replace(current, time_s=float(time_s))
        frames = FeatureEncoder().encode(current)
        # Serialization also rejects nonfinite source coordinates/pose values.
        _json([asdict(frame) for frame in frames])
        cases.append({"id": case_id, "raw": raw, "source": source,
                      "world": current, "frames": frames, "mapping": mapping})
    return manifest_path, manifest, cases


def _scene(case):
    world, mapping = case["world"], case["mapping"]
    patients = []
    descriptions = case["raw"].get("patient_descriptions", {})
    if not isinstance(descriptions, dict) or set(descriptions) - set(mapping):
        raise ValueError("patient_descriptions contains unknown source patient IDs")
    for patient in world.patients:
        if patient.is_patient_label != "yes" or patient.age_group_label not in AGES:
            raise ValueError("Unsupported patient/age label in presentation")
        if patient.pose_features is not None:
            # A pose override can leave action_label stale in EventEngine.
            # Never infer a human action from a stale label or feature vector.
            action = descriptions.get(patient.person_id)
            if not isinstance(action, str) or not action.strip():
                raise ValueError("Explicit patient_descriptions required for pose overrides")
        else:
            if patient.action_label not in ACTIONS:
                raise ValueError("Explicit supported action_label required for presentation")
            action = ACTIONS[patient.action_label]
            if patient.person_id in descriptions:
                action += "。" + str(descriptions[patient.person_id])
        patients.append({"patient_id": mapping[patient.person_id],
                         "age_group": AGES[patient.age_group_label], "action": action,
                         "position": asdict(patient.position), "height_max": patient.height_max})
    # Original staff/object identifiers are not needed by the respondent.
    staff = [{"staff_id": f"S{i+1}", "position": asdict(p.position),
              "velocity": asdict(p.velocity)} for i, p in enumerate(world.staff)]
    objects = []
    for i, item in enumerate(world.objects):
        if item.object_type not in OBJECTS:
            raise ValueError(f"Unsupported object type for presentation: {item.object_type}")
        objects.append({"object_id": f"O{i+1}", "type": OBJECTS[item.object_type],
                        "position": asdict(item.position)})
    return {"prototype_notice": NOTICE, "case_id": case["id"], "time_s": world.time_s,
            "description": case["raw"]["description"], "question": QUESTION,
            "coordinate_note": "座標はm。xは右、yは上。矢印はスタッフの移動速度(m/s)で、視線ではありません。",
            "state_note": "イベント適用後の瞬間状態。速度による位置の連続補間は行っていません。",
            "patients": patients, "staff": staff, "objects": objects}


def _svg(scene):
    entities = scene["patients"] + scene["staff"] + scene["objects"]
    points = [e["position"] for e in entities]
    min_x = min(p["x"] for p in points) - 1
    max_x = max(p["x"] for p in points) + 1
    min_y = min(p["y"] for p in points) - 1
    max_y = max(p["y"] for p in points) + 1
    scale = min(600 / (max_x-min_x), 420 / (max_y-min_y))
    def xy(position):
        return 50 + (position["x"]-min_x)*scale, 460 - (position["y"]-min_y)*scale
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 520" role="img">',
             '<title>指定時点の配置図</title>', '<rect width="720" height="520" fill="white"/>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#245b32"/></marker></defs>']
    label_slots = {}
    for item in scene["objects"]:
        x, y = xy(item["position"])
        coordinate = (item["position"]["x"], item["position"]["y"])
        slot = label_slots.get(coordinate, 0)
        label_slots[coordinate] = slot + 1
        # Keep the marker at its actual position; stack only text for objects
        # sharing that position so every identifier/type remains readable.
        parts += [f'<rect x="{x-10}" y="{y-10}" width="20" height="20" fill="#777"/>',
                  f'<text class="object-label" data-object-id="{escape(item["object_id"])}" x="{x+14}" y="{y+25+18*slot}" font-size="13">{escape(item["object_id"]+" "+item["type"])}</text>']
    for item in scene["patients"] + scene["staff"]:
        x, y = xy(item["position"])
        staff = "staff_id" in item
        label = item["staff_id"] if staff else item["patient_id"]
        color = "#245b32" if staff else "#244b85"
        parts += [f'<circle cx="{x}" cy="{y}" r="12" fill="{color}"/>',
                  f'<text x="{x+15}" y="{y-15}" font-size="16">{escape(label)}</text>']
        if staff:
            vx, vy = item["velocity"]["vx"], item["velocity"]["vy"]
            if vx or vy:
                # Arrow endpoint uses one second of velocity and is documented.
                parts.append(f'<line x1="{x}" y1="{y}" x2="{x+vx*scale}" y2="{y-vy*scale}" stroke="{color}" marker-end="url(#arrow)"/>')
    parts.append(f'<text x="30" y="505" font-size="13">x→ / y↑、1m = {scale:.1f}px。スタッフ矢印は1秒分の移動速度。</text></svg>')
    return "\n".join(parts)


def _provenance(paths, command):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    modules = REPO / "master_thesis_modules"
    code = [p for name in ("evaluation_study", "scenario_sim", "risk_core")
            for p in (modules / name).rglob("*") if p.suffix in {".py", ".js"}]
    files = sorted(set(Path(p).resolve() for p in [*paths, *code]))
    return {"created_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "request": "REQ-003 revision 7 R-08 / AC-08", "prototype_notice": NOTICE,
            "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "git_status_porcelain": git("status", "--porcelain"),
            "command_argv": list(command), "cwd": str(Path.cwd()), "seed": None,
            "python": sys.version, "platform": platform.platform(), "pyyaml": yaml.__version__,
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "PyYAML")},
            "source_sha256": {str(p): _sha(p) for p in files}}


def _write_new(output_dir, files):
    output_dir = Path(output_dir)
    # mkdir is atomic for the leaf; existing directories/files/symlinks are rejected.
    output_dir.mkdir(parents=True, exist_ok=False)
    for relative, content in files.items():
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return output_dir


def build_bundle(manifest_path, output_dir, command=()):
    manifest_path, manifest, cases = load_cases(manifest_path)
    files, rows, private_cases, scenes = {}, [], [], []
    for case in cases:
        scene = _scene(case)
        scenes.append(scene)
        case_id = case["id"]
        files[f"participant/{case_id}.svg"] = _svg(scene)
        patient_items = "".join(f'<li>{escape(p["patient_id"])}：{escape(p["age_group"])}、{escape(p["action"])}。位置 {escape(str(p["position"]))}</li>' for p in scene["patients"])
        details = _json({"staff": scene["staff"], "objects": scene["objects"]})
        files[f"participant/{case_id}.html"] = (
            '<!doctype html><html lang="ja"><meta charset="utf-8"><title>順位回答の試作</title>'
            f'<p>{NOTICE}</p><h1>{escape(case_id)} / {scene["time_s"]}秒</h1>'
            f'<p>{escape(scene["description"])}</p><p>{QUESTION}</p>'
            f'<img src="{case_id}.svg" alt="配置図"><ul>{patient_items}</ul>'
            f'<p>{escape(scene["coordinate_note"])}</p><p>{escape(scene["state_note"])}</p>'
            f'<pre>{escape(details)}</pre><p>{INSTRUCTIONS}</p></html>')
        files[f"participant/{case_id}.json"] = _json(scene)
        files[f"participant/{case_id}-ai-prompt.txt"] = (
            NOTICE + "\nAIの回答は設問の壁打ち専用で、実スタッフの結果ではありません。\n"
            + QUESTION + "\n場面に明記された情報だけで答え、不足があれば判断不能と理由を示してください。\n"
            + INSTRUCTIONS + "\nCSV列: " + ",".join(FIELDS) + "\n" + _json(scene))
        for alias in case["mapping"].values():
            rows.append(dict(zip(FIELDS, [case_id, "", alias, "", "", ""])))
        private_cases.append({"case_id": case_id, "scenario": str(case["source"]),
                              "time_s": case["world"].time_s, "patient_ids": case["mapping"],
                              "feature_frames": [asdict(frame) for frame in case["frames"]]})
    files["participant/responses-template.csv"] = _csv(rows)
    files["participant/questionnaire.html"] = render_questionnaire(scenes)
    files["participant/questionnaire.js"] = Path(__file__).with_name("questionnaire.js").read_text(encoding="utf-8")
    files["participant/instructions.txt"] = NOTICE + "\n" + QUESTION + "\n" + INSTRUCTIONS + "\n"
    files["analysis/cases.json"] = _json(private_cases)
    files["analysis/manifest.json"] = _json(manifest)
    files["analysis/provenance.json"] = _json(_provenance([manifest_path, *(c["source"] for c in cases)], command))
    return _write_new(output_dir, files)


def validate_responses(manifest_path, response_path):
    """Validate complete case/respondent sets; preserve ties and inability."""
    _, _, cases = load_cases(manifest_path)
    expected = {c["id"]: set(c["mapping"].values()) for c in cases}
    with Path(response_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"CSV columns must be {FIELDS}")
        rows = list(reader)
    if not rows:
        raise ValueError("No answers")
    grouped = {}
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row")
        row = {key: value.strip() for key, value in row.items()}
        case_id = row["case_id"]
        respondent = _identifier(row["respondent_id"], "respondent_id")
        if case_id not in expected:
            raise ValueError(f"Unexpected case: {case_id}")
        grouped.setdefault((respondent, case_id), []).append(row)
    respondents = {r for r, _ in grouped}
    for respondent in respondents:
        if {c for r, c in grouped if r == respondent} != set(expected):
            raise ValueError(f"Missing case for respondent: {respondent}")
    result = []
    for (_, case_id), group in grouped.items():
        ids = [row["patient_id"] for row in group]
        if len(ids) != len(set(ids)) or set(ids) != expected[case_id]:
            raise ValueError(f"Missing, duplicate or unexpected patients: {case_id}")
        statuses = {row["status"] for row in group}
        if statuses == {"unable"}:
            if any(row["rank"] or not row["reason"] for row in group):
                raise ValueError("Unable requires empty rank and reason on every row")
        elif statuses == {"ranked"}:
            if any(not re.fullmatch(r"[1-9][0-9]*", row["rank"]) for row in group):
                raise ValueError("Rank must be a positive integer")
            ranks = [int(row["rank"]) for row in group]
            if any(rank != 1 + sum(other < rank for other in ranks) for rank in ranks):
                raise ValueError("Use competition ranks: 1,1,3 (not 1,1,2)")
        else:
            raise ValueError("Status must be ranked for all patients or unable for all")
        result.extend(group)
    return result


def evaluate_profiles(manifest_path, profiles_path, output_dir, aggregation, command=(), responses=None):
    """Separate analysis command. Require explicit local AHP and TFN sources."""
    from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config
    from master_thesis_modules.risk_core.engine.risk_engine import RiskEngine

    if aggregation not in {"weighted_sum", "weighted_max"}:
        raise ValueError("Explicit action aggregation weighted_sum or weighted_max required")
    manifest_path, _, cases = load_cases(manifest_path)
    profiles_path = Path(profiles_path).resolve()
    profiles = _load(profiles_path).get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles requires a nonempty list")
    source_paths = [manifest_path, profiles_path, *(c["source"] for c in cases)]
    files, all_rows, seen, configs = {}, [], set(), {}
    for profile in profiles:
        respondent = _identifier(profile["respondent_id"], "respondent_id")
        if respondent in seen:
            raise ValueError("Duplicate respondent profile")
        seen.add(respondent)
        directory = (profiles_path.parent / profile["common_dir"]).resolve()
        ahp, fuzzy = profile["ahp_profile"], profile["fuzzy_profile"]
        for name in (ahp, fuzzy):
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ValueError("Profile names must be explicit file-name components")
        paths = [directory / f"comparison_mtx_30000001_{ahp}.csv",
                 directory / f"comparison_mtx_30000010_{ahp}.csv",
                 directory / f"TFN_{fuzzy}.csv"]
        for path in paths:
            if not path.is_file():
                raise ValueError(f"Explicit local knowledge source missing; fallback disabled: {path}")
        for path, size in zip(paths[:2], (7, 3)):
            with path.open(newline="", encoding="utf-8-sig") as handle:
                matrix = list(csv.reader(handle))
            if len(matrix) != size or any(len(row) != size for row in matrix):
                raise ValueError(f"AHP matrix must have shape {size}x{size}: {path}")
            # The existing loader reconstructs diagonal/lower triangle. Validate
            # exactly the upper entries that it actually uses.
            upper = [float(matrix[i][j]) for i in range(size) for j in range(i+1, size)]
            if not all(math.isfinite(value) and value > 0 for value in upper):
                raise ValueError(f"AHP upper-triangle entries must be finite and positive: {path}")
        source_paths.extend(paths)
        config = make_profile_risk_config(ahp, fuzzy, directory, model_type="spatial_context", action_aggregation=aggregation)
        configs[respondent] = asdict(config)
        engine = RiskEngine(config)
        snapshots = []
        for case in cases:
            evaluated = [engine.evaluate(frame) for frame in case["frames"]]
            scores = [result.total_risk for result in evaluated]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError("Nonfinite system score")
            for result in evaluated:
                rank = 1 + sum(bool(score > result.total_risk) for score in scores)
                alias = case["mapping"][result.person_id]
                all_rows.append({"respondent_id": respondent, "case_id": case["id"], "patient_id": alias,
                                 "time_s": result.time_s, "system_score": result.total_risk, "system_rank": rank})
                snapshots.append({"case_id": case["id"], "anonymous_patient_id": alias,
                                  "result": asdict(result), "system_rank": rank})
        files[f"{respondent}-snapshots.json"] = _json(snapshots)
    fields = ["respondent_id", "case_id", "patient_id", "time_s", "system_score", "system_rank"]
    files["system-ranks.csv"] = _csv(all_rows, fields)
    if responses:
        answers = validate_responses(manifest_path, responses)
        if {r["respondent_id"] for r in answers} != seen:
            raise ValueError("Answer respondents must exactly match supplied profiles")
        lookup = {(r["respondent_id"], r["case_id"], r["patient_id"]): r for r in answers}
        comparison = []
        for row in all_rows:
            answer = lookup[(row["respondent_id"], row["case_id"], row["patient_id"])]
            comparison.append({**row, "response_rank": answer["rank"], "response_status": answer["status"], "reason": answer["reason"]})
        files["comparison.csv"] = _csv(comparison, fields + ["response_rank", "response_status", "reason"])
        source_paths.append(Path(responses).resolve())
    files["configs.json"] = _json(configs)
    files["profiles.json"] = _json(_load(profiles_path))
    provenance = _provenance(source_paths, command)
    provenance.update({"action_aggregation": aggregation, "model_type": "spatial_context",
                       "rank_rule": "competition; exact float equality; no tolerance", "statistical_metrics": None})
    files["provenance.json"] = _json(provenance)
    return _write_new(output_dir, files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Share only participant/ with respondents. analysis/ and evaluate outputs contain private mappings and scores.")
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("manifest")
    bundle.add_argument("output_dir")
    validate = commands.add_parser("validate")
    validate.add_argument("manifest")
    validate.add_argument("responses")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("manifest")
    evaluate.add_argument("profiles")
    evaluate.add_argument("output_dir")
    evaluate.add_argument("--action-aggregation", choices=["weighted_sum", "weighted_max"], required=True)
    evaluate.add_argument("--responses")
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "master_thesis_modules.evaluation_study", *(sys.argv[1:] if argv is None else argv)]
    try:
        if args.command == "bundle":
            result = build_bundle(args.manifest, args.output_dir, command)
        elif args.command == "validate":
            result = f"valid: {len(validate_responses(args.manifest, args.responses))} patient answers"
        else:
            result = evaluate_profiles(args.manifest, args.profiles, args.output_dir, args.action_aggregation, command, args.responses)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(result)
