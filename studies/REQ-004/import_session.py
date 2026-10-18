"""Import the REQ-004 full local trial, preserving every original answer."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from master_thesis_modules.evaluation_study.study import (
    FIELDS, _json, _scene, _sha, _write_new, load_cases, validate_responses,
)
from master_thesis_modules.risk_core.engine.personalized_config import validate_personalized_knowledge
from master_thesis_modules.risk_core.schema import node_ids as ids
from master_thesis_modules.risk_core.schema.node_labels import NODE_LABELS

SCHEMA = "REQ004-full-session-trial-v1"
ORDER = ["HH", "HL", "LH", "LL"]


def _timing(timing):
    if not isinstance(timing, dict):
        raise ValueError("Timing record required")
    try:
        times = [datetime.fromisoformat(timing[key].replace("Z", "+00:00"))
                 for key in ("started_at", "completed_at")]
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("Valid timezone-qualified timestamps required") from error
    if any(t.tzinfo is None or t.utcoffset() is None for t in times) or times[1] < times[0]:
        raise ValueError("Timing timestamps must be timezone-qualified and ordered")
    elapsed = timing.get("elapsed_ms")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Nonnegative finite elapsed_ms required")
    if (timing.get("includes_reading_and_breaks") is not True
            or "target_ms" not in timing or timing["target_ms"] is not None
            or "target_achieved" not in timing or timing["target_achieved"] is not None):
        raise ValueError("REQ-004 timing has no time limit or achievement target")


def import_session(source: Path, manifest: Path, output: Path) -> Path:
    source, manifest, output = Path(source), Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError("Output directory must be new")
    raw = source.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA or data.get("prototype") is not True:
        raise ValueError("REQ004-full-session-trial-v1 prototype required")
    kind = data.get("data_kind")
    if kind not in ("synthetic_ai", "human_trial") or type(data.get("synthetic")) is not bool or data["synthetic"] != (kind == "synthetic_ai"):
        raise ValueError("data_kind and synthetic must explicitly agree")
    validate_personalized_knowledge(data)
    groups = {"action": [NODE_LABELS[n] for n in ids.ACTION_RISK_NODES],
              "object": [NODE_LABELS[n] for n in ids.OBJECT_RISK_NODES]}
    if data.get("knowledge_item_order") != groups or data.get("fuzzy_order") != ORDER:
        raise ValueError("Knowledge item order and Fuzzy order must match the form")
    _timing(data.get("timing"))
    manifest_path, _, cases = load_cases(manifest)
    presentation = hashlib.sha256(json.dumps([_scene(c) for c in cases], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if data.get("presentation_sha256") != presentation:
        raise ValueError("Presentation hash does not match the manifest scenes")
    rows = data.get("responses")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Complete initial responses required")
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(FIELDS) or row.get("respondent_id") != data["respondent_id"]:
            raise ValueError("Response fields and respondent must match the session")
        if any(not isinstance(row[k], str) for k in FIELDS if k != "rank"):
            raise ValueError("Response identifiers, status and reason must be strings")
        if not (type(row["rank"]) is int or row["rank"] == "") or isinstance(row["rank"], bool):
            raise ValueError("Rank must be an integer or empty unable value")
        if not row["reason"].strip():
            raise ValueError("Initial reasons required before explanation")
    with tempfile.TemporaryDirectory(prefix="req004-import-") as temporary:
        path = Path(temporary) / "initial-responses.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        validate_responses(manifest_path, path)
        response_text = path.read_text(encoding="utf-8")
    knowledge = {key: data[key] for key in ("respondent_id", "action_pairwise", "object_pairwise", "fuzzy_answers", "data_kind", "synthetic")}
    knowledge["provenance"] = {"source_kind": "full_session_form", "source_sha256": hashlib.sha256(raw).hexdigest(),
                               "fuzzy_order": ORDER, "presentation_sha256": presentation}
    provenance = {"schema_version": SCHEMA, "imported_at": datetime.now(timezone.utc).isoformat(),
                  "respondent_id": data["respondent_id"], "data_kind": kind, "synthetic": data["synthetic"],
                  "source": str(source.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
                  "manifest": str(manifest_path), "manifest_sha256": _sha(manifest_path),
                  "scenario_sha256": {str(case["source"]): _sha(case["source"]) for case in cases},
                  "importer_sha256": _sha(Path(__file__)), "presentation_sha256": presentation,
                  "presentation_sha256_scope": "ordered scenes JSON only; not HTML or JavaScript",
                  "timing": data["timing"], "input_fuzzy_order": ORDER,
                  "knowledge_item_order": groups, "score_conversion": "(answer-1)/4"}
    result = _write_new(output, {"knowledge.json": _json(knowledge), "initial-responses.csv": response_text,
                                 "session.json": raw.decode("utf-8"), "provenance.json": _json(provenance)})
    # Preserve original bytes, including formatting and newline conventions.
    (result / "session.json").write_bytes(raw)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    try:
        print(import_session(args.source, args.manifest, args.output))
    except (ValueError, FileExistsError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
