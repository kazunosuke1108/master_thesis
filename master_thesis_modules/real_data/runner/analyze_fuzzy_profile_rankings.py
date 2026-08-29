"""Analyze Fuzzy-profile effects on rankings from real-data evaluations."""

from __future__ import annotations

import argparse

from master_thesis_modules.scenario_sim.visualization.plot_fuzzy_profile_rankings import (
    analyze_fuzzy_profile_rankings,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="run_real_data_eval output directory")
    parser.add_argument(
        "--output",
        default=None,
        help="directory for analysis files; defaults to <input>/analysis",
    )
    parser.add_argument(
        "--common-dir",
        default="master_thesis_modules/database/common",
        help="directory containing TFN_<staff>.csv profile files",
    )
    parser.add_argument(
        "--ahp-profile",
        default=None,
        help="AHP profile to analyze; required only when input contains multiple AHP profiles",
    )
    args = parser.parse_args()
    paths = analyze_fuzzy_profile_rankings(
        sweep_dir=args.input,
        output_dir=args.output,
        common_dir=args.common_dir,
        ahp_profile=args.ahp_profile,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
