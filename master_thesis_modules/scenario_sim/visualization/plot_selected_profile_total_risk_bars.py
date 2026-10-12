"""Plot grouped total-risk bars for selected profile-sweep results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PROFILE_LABELS = ("P4", "P6", "P8")
DEFAULT_PATIENT_IDS = ("A", "B", "C", "D")
PATIENT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def build_selected_profile_total_risk_table(
    sweep_dir: str | Path,
    profile_labels: Sequence[str] = DEFAULT_PROFILE_LABELS,
    patient_ids: Sequence[str] = DEFAULT_PATIENT_IDS,
) -> pd.DataFrame:
    """Return mean total risk for each selected profile and patient.

    ``profile_plot_labels.csv`` is used to preserve the P-number mapping of the
    original profile-sweep figure.  Risk values are read directly from each
    profile's ``risk_timeseries.csv``.
    """

    sweep_dir = Path(sweep_dir)
    label_path = sweep_dir / "visualization" / "profile_plot_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Profile label table not found: {label_path}")

    label_table = pd.read_csv(label_path, dtype=str)
    profile_names = dict(
        zip(label_table["plot_label"], label_table["profile_name"], strict=True)
    )
    rows: list[dict[str, str | float]] = []

    for profile_label in profile_labels:
        if profile_label not in profile_names:
            raise ValueError(f"Unknown profile label: {profile_label}")
        profile_name = profile_names[profile_label]
        risk_path = sweep_dir / profile_name / "risk_timeseries.csv"
        if not risk_path.exists():
            raise FileNotFoundError(f"Risk time series not found: {risk_path}")

        risk_data = pd.read_csv(risk_path, dtype={"patient_id": str})
        patient_means = risk_data.groupby("patient_id")["10000000"].mean()
        for patient_id in patient_ids:
            if patient_id not in patient_means.index:
                raise ValueError(
                    f"Patient {patient_id} is missing from profile {profile_label}"
                )
            rows.append(
                {
                    "profile_label": profile_label,
                    "profile_name": profile_name,
                    "patient_id": patient_id,
                    "total_risk": float(patient_means.loc[patient_id]),
                }
            )

    return pd.DataFrame(rows)


def plot_selected_profile_total_risk_bars(
    sweep_dir: str | Path,
    output_png: str | Path,
    profile_labels: Sequence[str] = DEFAULT_PROFILE_LABELS,
    patient_ids: Sequence[str] = DEFAULT_PATIENT_IDS,
    x_tick_labels: Sequence[str] | None = None,
) -> Path:
    """Draw patient bars grouped by selected P-number profile labels."""

    profile_labels = tuple(profile_labels)
    patient_ids = tuple(patient_ids)
    x_tick_labels = tuple(x_tick_labels or profile_labels)
    if not profile_labels:
        raise ValueError("At least one profile label is required")
    if not patient_ids:
        raise ValueError("At least one patient ID is required")
    if len(x_tick_labels) != len(profile_labels):
        raise ValueError("The number of x-axis labels must match the profiles")

    plot_data = build_selected_profile_total_risk_table(
        sweep_dir,
        profile_labels=profile_labels,
        patient_ids=patient_ids,
    )
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    group_centers = list(range(len(profile_labels)))
    group_width = 0.8
    bar_width = group_width / len(patient_ids)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for patient_index, patient_id in enumerate(patient_ids):
        offset = (patient_index - (len(patient_ids) - 1) / 2) * bar_width
        x_positions = [center + offset for center in group_centers]
        patient_data = plot_data[plot_data["patient_id"] == patient_id]
        values_by_profile = patient_data.set_index("profile_label")["total_risk"]
        values = [float(values_by_profile.loc[label]) for label in profile_labels]
        bars = ax.bar(
            x_positions,
            values,
            width=bar_width * 0.9,
            label=patient_id,
            color=PATIENT_COLORS[patient_index % len(PATIENT_COLORS)],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(group_centers, x_tick_labels)
    ax.set_xlabel("Profile")
    ax.set_ylabel("Total risk")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Total risk by selected profile and patient")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title="Patient", ncol=len(patient_ids), loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot grouped total-risk bars for selected profiles."
    )
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILE_LABELS))
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENT_IDS))
    parser.add_argument(
        "--x-labels",
        nargs="+",
        default=None,
        help="Optional display labels for the selected profiles",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_png = args.output or (
        args.sweep_dir / "visualization" / "selected_profile_total_risk_bars.png"
    )
    written_path = plot_selected_profile_total_risk_bars(
        args.sweep_dir,
        output_png,
        profile_labels=args.profiles,
        patient_ids=args.patients,
        x_tick_labels=args.x_labels,
    )
    print(written_path)


if __name__ == "__main__":
    main()
