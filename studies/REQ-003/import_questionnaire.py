"""Import a complete local questionnaire session without changing its answers.

Human-trial labels identify fixture/trial data, not evidence of recruitment.
The existing engine consumes central Fuzzy values; l=c=r does not estimate
uncertainty. This importer does not run evaluation or select an aggregation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

import yaml

# Standalone CLI invoked from the repository checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from master_thesis_modules.evaluation_study.study import FIELDS, _scene, load_cases, validate_responses
from master_thesis_modules.risk_core.schema import node_ids
from master_thesis_modules.risk_core.schema.node_labels import NODE_LABELS

SCHEMA = "REQ003-full-session-trial-v1"
ORDER = ["HH", "HL", "LH", "LL"]


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _matrix(matrix, size):
    if not isinstance(matrix, list) or len(matrix) != size or any(not isinstance(row, list) or len(row) != size for row in matrix):
        raise ValueError(f"Pairwise matrix must be {size}x{size}")
    if any(not _number(x) or not 1/9 <= x <= 9 for row in matrix for x in row):
        raise ValueError("Pairwise values must be finite and in 1/9..9")
    for i in range(size):
        if matrix[i][i] != 1:
            raise ValueError("Pairwise diagonal must equal one")
        for j in range(size):
            if not math.isclose(matrix[i][j] * matrix[j][i], 1, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Pairwise answers must be reciprocal; no automatic repair")


def _timing(timing):
    if not isinstance(timing, dict):
        raise ValueError("Timing record required")
    try:
        times = [datetime.fromisoformat(timing[key].replace("Z", "+00:00")) for key in ("started_at", "completed_at")]
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("Valid timezone-qualified start/completion timestamps required") from error
    if any(t.tzinfo is None or t.utcoffset() is None for t in times) or times[1] < times[0]:
        raise ValueError("Timing timestamps must be timezone-qualified and ordered")
    elapsed = timing.get("elapsed_ms")
    if not _number(elapsed) or elapsed < 0:
        raise ValueError("Nonnegative finite elapsed_ms required")
    if timing.get("includes_reading_and_breaks") is not True or type(timing.get("target_ms")) is not int or timing["target_ms"] != 300000:
        raise ValueError("Full-session timing contract required")
    if type(timing.get("target_achieved")) is not bool or timing["target_achieved"] != (elapsed <= 300000):
        raise ValueError("target_achieved disagrees with elapsed_ms")
    # Monotonic elapsed time is authoritative. Wall-clock adjustments cannot be
    # used to recalculate it; preserve both clocks for provenance.


def import_session(source: Path, manifest: Path, output: Path) -> Path:
    source, manifest, output = Path(source), Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError("Output directory must be new")
    source_bytes = source.read_bytes()
    data = json.loads(source_bytes)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA or data.get("prototype") is not True:
        raise ValueError("Complete prototype questionnaire schema required")
    kind = data.get("data_kind")
    if kind not in ("synthetic_ai", "human_trial") or type(data.get("synthetic")) is not bool or data["synthetic"] != (kind == "synthetic_ai"):
        raise ValueError("data_kind and synthetic must explicitly agree")
    respondent = data.get("respondent_id")
    if not isinstance(respondent, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", respondent):
        raise ValueError("Safe anonymous respondent_id required")
    groups = {"action": [NODE_LABELS[n] for n in node_ids.ACTION_RISK_NODES],
              "object": [NODE_LABELS[n] for n in node_ids.OBJECT_RISK_NODES]}
    if data.get("knowledge_item_order") != groups or data.get("fuzzy_order") != ORDER:
        raise ValueError("Knowledge item order and Fuzzy order must match the form contract")
    matrices = [data.get("action_pairwise"), data.get("object_pairwise")]
    for matrix, size in zip(matrices, (7, 3)):
        _matrix(matrix, size)
    fuzzy = data.get("fuzzy_answers")
    if not isinstance(fuzzy, dict) or set(fuzzy) != {"internal", "external", "total"}:
        raise ValueError("All three Fuzzy groups required")
    for answers in fuzzy.values():
        if not isinstance(answers, list) or len(answers) != 4 or any(type(x) is not int or not 1 <= x <= 5 for x in answers):
            raise ValueError("All twelve integer Fuzzy answers in 1..5 required")
    _timing(data.get("timing"))
    manifest_path, _, cases = load_cases(manifest)
    presentation = hashlib.sha256(json.dumps([_scene(c) for c in cases], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if data.get("presentation_sha256") != presentation:
        raise ValueError("Presentation hash does not match the supplied manifest")
    responses = data.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("Complete responses required")
    for row in responses:
        if not isinstance(row, dict) or set(row) != set(FIELDS) or row.get("respondent_id") != respondent:
            raise ValueError("Every response must have exact fields and the session respondent")
        if any(not isinstance(row[k], str) for k in FIELDS if k != "rank"):
            raise ValueError("Response identifiers, status and reason must be strings")
        if not (type(row["rank"]) is int or row["rank"] == "") or isinstance(row["rank"], bool):
            raise ValueError("Rank must be an integer or the empty unable value")
    # Stage outside the requested destination: invalid data never creates output.
    with tempfile.TemporaryDirectory(prefix="req003-import-") as temporary:
        stage = Path(temporary)
        response_path = stage / "responses.csv"
        with response_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, FIELDS)
            writer.writeheader()
            writer.writerows(responses)
        validate_responses(manifest_path, response_path)
        common = stage / "knowledge"
        common.mkdir()
        answers = fuzzy["internal"] + [fuzzy["external"][i] for i in (1, 0, 3, 2)] + fuzzy["total"]
        rows = matrices + [[[(x-1)/4]*3 for x in answers]]
        names = [f"comparison_mtx_30000001_{respondent}.csv", f"comparison_mtx_30000010_{respondent}.csv", f"TFN_{respondent}.csv"]
        for name, values in zip(names, rows):
            with (common / name).open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(values)
        profile = {"respondent_id": respondent, "common_dir": "knowledge", "ahp_profile": respondent,
                   "fuzzy_profile": respondent, "data_kind": kind, "synthetic": data["synthetic"]}
        (stage / "profiles.yaml").write_text(yaml.safe_dump({"profiles": [profile]}, allow_unicode=True), encoding="utf-8")
        provenance = {"schema_version": SCHEMA, "imported_at": datetime.now(timezone.utc).isoformat(),
                      "respondent_id": respondent, "data_kind": kind, "synthetic": data["synthetic"],
                      "source": str(source.resolve()), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                      "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      "importer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "presentation_sha256": presentation,
                      "presentation_sha256_scope": "ordered scenes JSON only; not questionnaire HTML, questions, or JavaScript",
                      "timing": data["timing"],
                      "knowledge_item_order": groups, "input_fuzzy_order": ORDER,
                      "external_csv_row_order": ["HL", "HH", "LL", "LH"],
                      "score_conversion": "(answer-1)/4", "tfn_width": "l=c=r; uncertainty not estimated"}
        (stage / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "session.json").write_bytes(source_bytes)
        # mkdir is exclusive even if another process created the destination
        # after initial validation; never overwrite a concurrent output.
        output.mkdir(parents=True, exist_ok=False)
        try:
            for path in stage.iterdir():
                if path.is_dir():
                    shutil.copytree(path, output / path.name)
                else:
                    shutil.copyfile(path, output / path.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = import_session(args.source, args.manifest, args.output)
    except (ValueError, FileExistsError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
