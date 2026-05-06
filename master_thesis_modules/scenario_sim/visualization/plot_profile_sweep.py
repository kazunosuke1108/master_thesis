"""Visualize AHP/Fuzzy profile sweep outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import pandas as pd


PROFILE_DIR_PATTERN = re.compile(r"^ahp_(?P<ahp>.+)__fuzzy_(?P<fuzzy>.+)$")


@dataclass(frozen=True)
class ProfileRun:
    profile_name: str
    ahp_profile: str
    fuzzy_profile: str
    path: Path
    risk_timeseries: pd.DataFrame
    ranking: pd.DataFrame
    notification_log: pd.DataFrame


def visualize_profile_sweep(
    sweep_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Create profile-sweep figures and summary CSVs.

    Expected input is the directory produced by
    `scenario_sim.runner.run_profile_sweep`, containing subdirectories named
    `ahp_<name>__fuzzy_<name>`.
    """

    sweep_dir = Path(sweep_dir)
    output_dir = Path(output_dir) if output_dir is not None else sweep_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_profile_runs(sweep_dir)
    if not runs:
        raise FileNotFoundError(f"No profile run directories found in {sweep_dir}")

    paths = {
        "profile_summary": output_dir / "profile_summary.csv",
        "profile_ranking_summary": output_dir / "profile_ranking_summary.csv",
        "profile_plot_labels": output_dir / "profile_plot_labels.csv",
        "profile_total_risk_grid": output_dir / "profile_total_risk_grid.png",
        "profile_top_risk_comparison": output_dir / "profile_top_risk_comparison.png",
        "profile_notification_counts": output_dir / "profile_notification_counts.png",
    }
    build_profile_summary(runs).to_csv(paths["profile_summary"], index=False)
    build_ranking_summary(runs).to_csv(paths["profile_ranking_summary"], index=False)
    build_profile_label_table(runs).to_csv(paths["profile_plot_labels"], index=False)
    plot_total_risk_grid(runs, paths["profile_total_risk_grid"])
    plot_top_risk_comparison(runs, paths["profile_top_risk_comparison"])
    plot_notification_counts(runs, paths["profile_notification_counts"])
    return paths


def load_profile_runs(sweep_dir: str | Path) -> list[ProfileRun]:
    sweep_dir = Path(sweep_dir)
    runs = []
    for run_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        match = PROFILE_DIR_PATTERN.match(run_dir.name)
        if match is None:
            continue
        risk_path = run_dir / "risk_timeseries.csv"
        ranking_path = run_dir / "ranking.csv"
        notification_path = run_dir / "notification_log.csv"
        if not risk_path.exists() or not ranking_path.exists():
            continue
        notification_log = (
            pd.read_csv(notification_path)
            if notification_path.exists()
            else pd.DataFrame(
                columns=[
                    "timestamp",
                    "target_patient_id",
                    "total_risk",
                    "rank",
                    "reason",
                    "message",
                    "notification_type",
                ]
            )
        )
        runs.append(
            ProfileRun(
                profile_name=run_dir.name,
                ahp_profile=match.group("ahp"),
                fuzzy_profile=match.group("fuzzy"),
                path=run_dir,
                risk_timeseries=pd.read_csv(risk_path),
                ranking=pd.read_csv(ranking_path),
                notification_log=notification_log,
            )
        )
    return runs


def build_profile_summary(runs: list[ProfileRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        risk_data = run.risk_timeseries
        ranking_data = run.ranking
        top1 = ranking_data[ranking_data["rank"] == 1]
        for patient_id, patient_data in risk_data.groupby("patient_id"):
            top1_count = int((top1["patient_id"] == patient_id).sum())
            rows.append(
                {
                    "profile_name": run.profile_name,
                    "ahp_profile": run.ahp_profile,
                    "fuzzy_profile": run.fuzzy_profile,
                    "patient_id": patient_id,
                    "mean_total_risk": patient_data["10000000"].mean(),
                    "max_total_risk": patient_data["10000000"].max(),
                    "final_total_risk": patient_data.sort_values("timestamp")[
                        "10000000"
                    ].iloc[-1],
                    "top1_count": top1_count,
                    "notice_count": _notification_count(run, "notice", patient_id),
                    "help_count": _notification_count(run, "help", patient_id),
                }
            )
    return pd.DataFrame(rows)


def build_ranking_summary(runs: list[ProfileRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for _, row in run.ranking.iterrows():
            rows.append(
                {
                    "profile_name": run.profile_name,
                    "ahp_profile": run.ahp_profile,
                    "fuzzy_profile": run.fuzzy_profile,
                    "timestamp": row["timestamp"],
                    "patient_id": row["patient_id"],
                    "rank": row["rank"],
                    "total_risk": row["10000000"],
                }
            )
    return pd.DataFrame(rows)


def build_profile_label_table(runs: list[ProfileRun]) -> pd.DataFrame:
    rows = []
    for index, run in enumerate(runs, start=1):
        rows.append(
            {
                "plot_label": _profile_label(index),
                "profile_name": run.profile_name,
                "ahp_profile": run.ahp_profile,
                "fuzzy_profile": run.fuzzy_profile,
            }
        )
    return pd.DataFrame(rows)


def plot_total_risk_grid(runs: list[ProfileRun], output_png: str | Path) -> Path:
    output_png = Path(output_png)
    rows, cols = _grid_shape(len(runs))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), sharey=True)
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for run_index, (ax, run) in enumerate(zip(axes_list, runs), start=1):
        for patient_id, patient_data in run.risk_timeseries.groupby("patient_id"):
            patient_data = patient_data.sort_values("timestamp")
            ax.plot(
                patient_data["timestamp"],
                patient_data["10000000"],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=str(patient_id),
            )
        ax.set_title(_profile_label(run_index))
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
    for ax in axes_list[::cols]:
        ax.set_ylabel("Total risk")
    for ax in axes_list[len(runs) :]:
        ax.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def plot_top_risk_comparison(runs: list[ProfileRun], output_png: str | Path) -> Path:
    output_png = Path(output_png)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for run_index, run in enumerate(runs, start=1):
        top = run.ranking[run.ranking["rank"] == 1].sort_values("timestamp")
        ax.plot(
            top["timestamp"],
            top["10000000"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=_profile_label(run_index),
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Top patient total risk")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def plot_notification_counts(runs: list[ProfileRun], output_png: str | Path) -> Path:
    output_png = Path(output_png)
    labels = [_profile_label(index) for index, _ in enumerate(runs, start=1)]
    notice_counts = [_notification_count(run, "notice") for run in runs]
    help_counts = [_notification_count(run, "help") for run in runs]
    x_positions = range(len(runs))
    fig, ax = plt.subplots(figsize=(max(7, 2.0 * len(runs)), 4.2))
    ax.bar(x_positions, notice_counts, label="notice", color="#c43c39")
    ax.bar(x_positions, help_counts, bottom=notice_counts, label="help", color="#3d6fb6")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Notification count")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def _grid_shape(n_items: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n_items))
    rows = math.ceil(n_items / cols)
    return rows, cols


def _profile_label(index: int) -> str:
    return f"P{index}"


def _notification_count(
    run: ProfileRun,
    notification_type: str,
    patient_id: str | None = None,
) -> int:
    log = run.notification_log
    if log.empty or "notification_type" not in log:
        return 0
    filtered = log[log["notification_type"] == notification_type]
    if patient_id is not None and "target_patient_id" in filtered:
        filtered = filtered[filtered["target_patient_id"] == patient_id]
    return int(len(filtered))
