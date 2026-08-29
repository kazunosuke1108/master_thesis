"""Compare new scenario outputs with legacy CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from master_thesis_modules.real_data.runner.compare_real_data_with_legacy import (
    compare_eval_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(compare_eval_dirs(Path(args.new), Path(args.legacy), Path(args.output)))


if __name__ == "__main__":
    main()

