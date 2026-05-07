"""Load legacy `data_dicts.pickle` files."""

from pathlib import Path
import pickle

import pandas as pd

from master_thesis_modules.risk_core.features.dataframe_adapter import (
    normalize_legacy_columns,
)


def load_data_dicts(path: str | Path) -> dict[str, pd.DataFrame]:
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    return {
        str(person_id): normalize_legacy_columns(dataframe)
        for person_id, dataframe in data.items()
    }

