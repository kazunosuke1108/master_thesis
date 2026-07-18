"""Visualize AHP/Fuzzy profile sweep outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
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
        "profile_total_risk_by_profile_dir": output_dir,
        "profile_hierarchy_timeseries_dir": output_dir,
        "profile_top_risk_comparison": output_dir / "profile_top_risk_comparison.png",
        "profile_notification_counts": output_dir / "profile_notification_counts.png",
    }
    build_profile_summary(runs).to_csv(paths["profile_summary"], index=False)
    build_ranking_summary(runs).to_csv(paths["profile_ranking_summary"], index=False)
    build_profile_label_table(runs).to_csv(paths["profile_plot_labels"], index=False)
    plot_total_risk_grid(runs, paths["profile_total_risk_grid"])
    plot_total_risk_by_profile(runs, output_dir)
    plot_hierarchy_timeseries_by_profile(runs, output_dir)
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
            pd.read_csv(notification_path, dtype={"target_patient_id": str})
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
                risk_timeseries=pd.read_csv(risk_path, dtype={"patient_id": str}),
                ranking=pd.read_csv(ranking_path, dtype={"patient_id": str}),
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
        _plot_total_risk_run(ax, run, _profile_label(run_index))
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


def plot_total_risk_by_profile(
    runs: list[ProfileRun],
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_paths = []
    for run_index, run in enumerate(runs, start=1):
        profile_label = _profile_label(run_index)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        _plot_total_risk_run(ax, run, profile_label)
        ax.set_ylabel("Total risk")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="best", fontsize=9)
        fig.tight_layout()
        output_png = output_dir / f"{profile_label}_total_risk.png"
        fig.savefig(output_png, dpi=180)
        plt.close(fig)
        output_paths.append(output_png)
    return output_paths


def _plot_total_risk_run(
    ax: plt.Axes,
    run: ProfileRun,
    profile_label: str,
) -> None:
    for patient_id, patient_data in run.risk_timeseries.groupby("patient_id"):
        patient_data = patient_data.sort_values("timestamp")
        style = _patient_line_style(patient_data)
        ax.plot(
            patient_data["timestamp"],
            patient_data["10000000"],
            marker="o",
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=style["alpha"],
            label=str(patient_id),
        )
    ax.set_title(profile_label)
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)


def plot_hierarchy_timeseries_by_profile(
    runs: list[ProfileRun],
    output_dir: str | Path,
) -> list[Path]:
    """Plot all available hierarchy nodes and raw features for each profile."""

    output_dir = Path(output_dir)
    output_paths = []
    for run_index, run in enumerate(runs, start=1):
        profile_label = _profile_label(run_index)
        plot_df = _load_hierarchy_plot_data(run)
        plot_items = _hierarchy_plot_items(plot_df)
        if not plot_items:
            continue

        cols = 4
        rows = math.ceil(len(plot_items) / cols)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(5.0 * cols, 2.9 * rows),
            sharex=True,
        )
        axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
        for ax, item in zip(axes_list, plot_items):
            column = item["column"]
            for patient_id, patient_data in plot_df.groupby("patient_id"):
                patient_data = patient_data.sort_values("timestamp")
                style = _patient_line_style(patient_data)
                ax.plot(
                    patient_data["timestamp"],
                    patient_data[column],
                    marker="o",
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    alpha=style["alpha"],
                    label=str(patient_id),
                )
            ax.set_title(item["title"], fontsize=9)
            ax.grid(True, alpha=0.3)
            if item.get("ylim_0_1"):
                ax.set_ylim(-0.03, 1.03)
        for ax in axes_list[len(plot_items) :]:
            ax.axis("off")
        for ax in axes_list[-cols:]:
            ax.set_xlabel("Time [s]")
        handles, labels = axes_list[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
        fig.suptitle(
            f"{profile_label}: hierarchy and feature time series",
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        output_png = output_dir / f"{profile_label}_hierarchy_timeseries.png"
        fig.savefig(output_png, dpi=180)
        plt.close(fig)
        output_paths.append(output_png)
    return output_paths


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


def _load_hierarchy_plot_data(run: ProfileRun) -> pd.DataFrame:
    risk_data = run.risk_timeseries.copy()
    for column in risk_data.columns:
        if column in {"patient_id", "timestamp", "explanation"}:
            continue
        risk_data[column] = risk_data[column].map(_to_plot_value)

    raw_frames = []
    for path in sorted(run.path.glob("data_*_eval.csv")):
        raw_frame = pd.read_csv(path)
        if "person_id" in raw_frame.columns:
            raw_frame = raw_frame.rename(columns={"person_id": "patient_id"})
        elif "patient_id" not in raw_frame.columns:
            raw_frame.insert(0, "patient_id", _patient_id_from_eval_csv(path))
        raw_frames.append(raw_frame)
    if not raw_frames:
        return risk_data

    raw_data = pd.concat(raw_frames, ignore_index=True)
    raw_columns = [
        column
        for column in RAW_FEATURE_COLUMNS
        if column in raw_data.columns and column not in risk_data.columns
    ]
    raw_data = raw_data[["patient_id", "timestamp", *raw_columns]]
    merged = risk_data.merge(raw_data, on=["patient_id", "timestamp"], how="left")
    for column in raw_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def _patient_id_from_eval_csv(path: Path) -> str:
    name = path.name
    prefix = "data_"
    suffix = "_eval.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def _to_plot_value(value: object) -> float:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("(") and text.endswith(")"):
            try:
                values = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return float("nan")
            if isinstance(values, tuple) and values:
                return float(sum(values) / len(values))
        try:
            return float(text)
        except ValueError:
            return float("nan")
    return float(value)


def _patient_line_style(patient_data: pd.DataFrame) -> dict[str, object]:
    if _is_staff_like_series(patient_data):
        return {
            "linestyle": "--",
            "linewidth": 0.8,
            "markersize": 1.8,
            "alpha": 0.55,
        }
    return {
        "linestyle": "-",
        "linewidth": 1.4,
        "markersize": 3.0,
        "alpha": 1.0,
    }


def _is_staff_like_series(patient_data: pd.DataFrame) -> bool:
    if "is_rankable_patient" in patient_data.columns:
        values = patient_data["is_rankable_patient"].dropna()
        if not values.empty:
            return not values.map(_to_bool).mode().iloc[0]
    if "is_patient_label" in patient_data.columns:
        values = patient_data["is_patient_label"].dropna()
        if not values.empty:
            return values.astype(str).str.strip().str.lower().eq("no").mean() >= 0.5
    return False


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text not in {"false", "0", "no", "staff"}


RAW_FEATURE_COLUMNS = (
    "standing_degree",
    "trunk_tilt",
    "wrist_distance_from_hip",
    "ankle_spread",
    "x",
    "y",
    "height_max",
)


PLOT_NODE_GROUPS = (
    (
        "L1",
        (
            ("10000000", "total_risk"),
        ),
    ),
    (
        "L2",
        (
            ("20000000", "internal_risk"),
            ("20000001", "external_risk"),
        ),
    ),
    (
        "L3",
        (
            ("30000000", "internal_static"),
            ("30000001", "internal_dynamic"),
            ("30000010", "external_static"),
            ("30000011", "external_dynamic"),
        ),
    ),
    (
        "L4",
        (
            ("40000000", "patient_attr"),
            ("40000001", "age_attr"),
            ("40000010", "stand_up"),
            ("40000011", "brake_release"),
            ("40000012", "move_wheelchair"),
            ("40000013", "lose_balance"),
            ("40000014", "hand_movement"),
            ("40000015", "coughing"),
            ("40000016", "touch_face"),
            ("40000100", "near_iv_pole"),
            ("40000101", "near_wheelchair"),
            ("40000102", "far_handrail"),
            ("40000110", "staff_distance"),
            ("40000111", "staff_watch_loss"),
        ),
    ),
    (
        "L6/raw",
        (
            ("standing_degree", "standing_degree"),
            ("trunk_tilt", "trunk_tilt"),
            ("wrist_distance_from_hip", "wrist_from_hip"),
            ("ankle_spread", "ankle_spread"),
            ("x", "person_x"),
            ("y", "person_y"),
            ("height_max", "height_max"),
        ),
    ),
)


def _hierarchy_plot_items(plot_df: pd.DataFrame) -> list[dict[str, object]]:
    items = []
    for layer, group_items in PLOT_NODE_GROUPS:
        for column, name in group_items:
            if column not in plot_df.columns:
                continue
            if plot_df[column].notna().sum() == 0:
                continue
            title = f"{layer} {column}: {name}" if column.isdigit() else f"{layer}: {name}"
            items.append(
                {
                    "column": column,
                    "title": title,
                    "ylim_0_1": column not in {"x", "y", "height_max"},
                }
            )
    return items


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
