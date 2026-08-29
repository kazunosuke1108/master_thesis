"""Compare renovated profile-sweep outputs with legacy ``master_v5.py`` CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compare_outputs(
    legacy_dir: str | Path,
    renovated_dir: str | Path,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    legacy_dir = Path(legacy_dir)
    renovated_dir = Path(renovated_dir)
    rows = []
    for legacy_csv in sorted(legacy_dir.glob("data_*_eval.csv")):
        renovated_csv = renovated_dir / legacy_csv.name
        if not renovated_csv.exists():
            rows.append(_missing_file_row(legacy_csv.name))
            continue
        legacy = pd.read_csv(legacy_csv)
        renovated = pd.read_csv(renovated_csv)
        rows.extend(_compare_dataframe(legacy_csv.name, legacy, renovated, tolerance))
    return pd.DataFrame(rows)


def _missing_file_row(filename: str) -> dict[str, object]:
    return {
        "file": filename,
        "column": "",
        "status": "missing_new_file",
        "legacy_rows": None,
        "renovated_rows": None,
        "max_abs_diff": None,
        "mismatch_count": None,
        "first_mismatch_index": None,
    }


def _compare_dataframe(
    filename: str,
    legacy: pd.DataFrame,
    renovated: pd.DataFrame,
    tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if len(legacy) != len(renovated):
        rows.append(
            {
                "file": filename,
                "column": "",
                "status": "row_count_mismatch",
                "legacy_rows": len(legacy),
                "renovated_rows": len(renovated),
                "max_abs_diff": None,
                "mismatch_count": None,
                "first_mismatch_index": None,
            }
        )
    for column in legacy.columns:
        if column not in renovated.columns:
            rows.append(_missing_column_row(filename, column, "missing_new_column"))
            continue
        legacy_values = legacy[column]
        renovated_values = renovated[column]
        legacy_numeric = pd.to_numeric(legacy_values, errors="coerce")
        renovated_numeric = pd.to_numeric(renovated_values, errors="coerce")
        numeric_mask = legacy_numeric.notna() | renovated_numeric.notna()
        if numeric_mask.any():
            diff = (legacy_numeric - renovated_numeric).abs()
            both_nan = legacy_numeric.isna() & renovated_numeric.isna()
            mismatches = numeric_mask & ~both_nan & (diff.fillna(float("inf")) > tolerance)
            status = "ok" if not mismatches.any() else "numeric_mismatch"
            rows.append(
                {
                    "file": filename,
                    "column": column,
                    "status": status,
                    "legacy_rows": len(legacy),
                    "renovated_rows": len(renovated),
                    "max_abs_diff": diff.max(skipna=True),
                    "mismatch_count": int(mismatches.sum()),
                    "first_mismatch_index": _first_index(mismatches),
                }
            )
        else:
            legacy_text = legacy_values.fillna("").astype(str)
            renovated_text = renovated_values.fillna("").astype(str)
            mismatches = legacy_text != renovated_text
            status = "ok" if not mismatches.any() else "text_mismatch"
            rows.append(
                {
                    "file": filename,
                    "column": column,
                    "status": status,
                    "legacy_rows": len(legacy),
                    "renovated_rows": len(renovated),
                    "max_abs_diff": None,
                    "mismatch_count": int(mismatches.sum()),
                    "first_mismatch_index": _first_index(mismatches),
                }
            )
    for column in renovated.columns:
        if column not in legacy.columns:
            rows.append(_missing_column_row(filename, column, "extra_new_column"))
    return rows


def _missing_column_row(filename: str, column: str, status: str) -> dict[str, object]:
    return {
        "file": filename,
        "column": column,
        "status": status,
        "legacy_rows": None,
        "renovated_rows": None,
        "max_abs_diff": None,
        "mismatch_count": None,
        "first_mismatch_index": None,
    }


def _first_index(mask: pd.Series) -> int | None:
    indexes = mask.index[mask].tolist()
    return int(indexes[0]) if indexes else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", required=True)
    parser.add_argument("--renovated-dir", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    report = compare_outputs(args.legacy_dir, args.renovated_dir, args.tolerance)
    failures = report[report["status"] != "ok"]
    if args.output_csv:
        report.to_csv(args.output_csv, index=False)
    if failures.empty:
        print("OK: all compared legacy columns match within tolerance")
    else:
        print(failures.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
