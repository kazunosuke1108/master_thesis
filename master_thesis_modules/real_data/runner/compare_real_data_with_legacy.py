"""Compare new evaluated CSVs with legacy evaluated CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from master_thesis_modules.risk_core.schema import node_ids as ids


COMPARE_NODES = (
    ids.TOTAL_RISK,
    ids.INTERNAL_RISK,
    ids.EXTERNAL_RISK,
) + ids.FACTOR_RISK_NODES


def compare_eval_dirs(new: str | Path, legacy: str | Path, output: str | Path) -> Path:
    new = Path(new)
    legacy = Path(legacy)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    new_data = _load_eval_dir(new)
    legacy_data = _load_eval_dir(legacy)
    summary = compare_dataframes(new_data, legacy_data)
    path = output / "legacy_comparison_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(summary["node_errors"]).to_csv(output / "legacy_node_errors.csv", index=False)
    return path


def compare_dataframes(
    new_data: dict[str, pd.DataFrame],
    legacy_data: dict[str, pd.DataFrame],
) -> dict[str, object]:
    node_errors = []
    rank_matches = []
    top_matches = []
    common_patients = sorted(set(new_data) & set(legacy_data))
    for node_id in COMPARE_NODES:
        errors = []
        for person_id in common_patients:
            new_col = _column(new_data[person_id], node_id)
            legacy_col = _column(legacy_data[person_id], node_id)
            if new_col is None or legacy_col is None:
                continue
            n = min(len(new_col), len(legacy_col))
            errors.extend((new_col.iloc[:n] - legacy_col.iloc[:n]).abs().tolist())
        if errors:
            node_errors.append(
                {
                    "node_id": node_id,
                    "mae": float(pd.Series(errors).mean()),
                    "max_error": float(pd.Series(errors).max()),
                }
            )

    if common_patients:
        n_rows = min(len(new_data[patient]) for patient in common_patients)
        for idx in range(n_rows):
            new_ranks = _rank_at(new_data, common_patients, idx)
            legacy_ranks = _rank_at(legacy_data, common_patients, idx)
            rank_matches.append(float(new_ranks == legacy_ranks))
            top_matches.append(float(new_ranks[0] == legacy_ranks[0]))

    return {
        "common_patients": common_patients,
        "node_errors": node_errors,
        "rank_match_rate": float(pd.Series(rank_matches).mean()) if rank_matches else 0.0,
        "top1_match_rate": float(pd.Series(top_matches).mean()) if top_matches else 0.0,
    }


def _load_eval_dir(path: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for csv_path in sorted(path.glob("data_*_eval.csv")):
        person_id = csv_path.name[len("data_") : -len("_eval.csv")]
        data[person_id] = pd.read_csv(csv_path)
    return data


def _column(dataframe: pd.DataFrame, node_id: int) -> pd.Series | None:
    if node_id in dataframe:
        return dataframe[node_id]
    if str(node_id) in dataframe:
        return dataframe[str(node_id)]
    return None


def _rank_at(data: dict[str, pd.DataFrame], patients: list[str], idx: int) -> list[str]:
    scored = []
    for patient in patients:
        col = _column(data[patient], ids.TOTAL_RISK)
        score = float(col.iloc[idx]) if col is not None else 0.0
        scored.append((patient, score))
    return [patient for patient, _ in sorted(scored, key=lambda item: item[1], reverse=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(compare_eval_dirs(args.new, args.legacy, args.output))


if __name__ == "__main__":
    main()

