"""Load `data_<patient>_raw.csv` or `data_<patient>_eval.csv` directories."""

from pathlib import Path

import pandas as pd

from master_thesis_modules.risk_core.features.dataframe_adapter import (
    normalize_legacy_columns,
)


def load_raw_csv_dir(path: str | Path, suffix: str = "_raw.csv") -> dict[str, pd.DataFrame]:
    path = Path(path)
    data_dicts = {}
    for csv_path in sorted(path.glob(f"data_*{suffix}")):
        person_id = csv_path.name[len("data_") : -len(suffix)]
        data_dicts[person_id] = normalize_legacy_columns(pd.read_csv(csv_path))
    return data_dicts

