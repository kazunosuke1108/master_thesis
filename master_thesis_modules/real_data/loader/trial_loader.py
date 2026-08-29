"""Choose an appropriate real-data loader by input path."""

from pathlib import Path

from master_thesis_modules.real_data.loader.data_dicts_loader import load_data_dicts
from master_thesis_modules.real_data.loader.raw_csv_loader import load_raw_csv_dir


def load_trial_input(path: str | Path):
    path = Path(path)
    if path.is_file() and path.suffix == ".pickle":
        return load_data_dicts(path)
    if path.is_dir():
        raw = load_raw_csv_dir(path, suffix="_raw.csv")
        if raw:
            return raw
        return load_raw_csv_dir(path, suffix="_eval.csv")
    raise FileNotFoundError(path)

