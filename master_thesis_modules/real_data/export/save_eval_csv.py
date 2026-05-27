"""Save evaluated dataframes in legacy-compatible CSV layout."""

from pathlib import Path

import pandas as pd


def save_eval_csvs(evaluated: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for person_id, dataframe in evaluated.items():
        dataframe.to_csv(output_dir / f"data_{person_id}_eval.csv", index=False)

