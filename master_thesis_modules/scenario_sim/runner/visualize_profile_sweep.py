"""Visualize outputs created by run_profile_sweep."""

from __future__ import annotations

import argparse

from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    visualize_profile_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="run_profile_sweep output directory")
    parser.add_argument(
        "--output",
        default=None,
        help="directory for figures and summary CSVs; defaults to <input>/visualization",
    )
    args = parser.parse_args()
    paths = visualize_profile_sweep(args.input, args.output)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

