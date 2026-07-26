"""Relate Fuzzy profile parameters to mean-risk patient rankings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from master_thesis_modules.scenario_sim.visualization.plot_profile_sweep import (
    ProfileRun,
    load_profile_runs,
)


TOTAL_RISK_COLUMN = "10000000"
JAPANESE_FONT = "Noto Sans CJK JP"


def analyze_fuzzy_profile_rankings(
    sweep_dir: str | Path,
    common_dir: str | Path = "master_thesis_modules/database/common",
    output_dir: str | Path | None = None,
    ahp_profile: str | None = None,
) -> dict[str, Path]:
    """Write normalized mean-risk tables and a Fuzzy-profile comparison figure.

    ``C_i`` is calculated from the central (``c``) TFN values at human-facing
    CSV rows 9--12: ``(row9 - row10) + (row11 - row12)``.
    """
    sweep_dir = Path(sweep_dir)
    output_dir = Path(output_dir) if output_dir is not None else sweep_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_profile_runs(sweep_dir)
    if not runs:
        raise FileNotFoundError(f"No profile run directories found in {sweep_dir}")
    runs = _select_ahp_profile(runs, ahp_profile)
    rankings = build_fuzzy_profile_rankings(runs, common_dir)

    paths = {
        "fuzzy_profile_ranking_summary": output_dir / "fuzzy_profile_ranking_summary.csv",
        "fuzzy_profile_ranking_matrix": output_dir / "fuzzy_profile_ranking_matrix.csv",
        "fuzzy_ci_patient_mean": output_dir / "fuzzy_ci_patient_mean.csv",
        "fuzzy_profile_ranking_plot": output_dir / "fuzzy_profile_ranking.png",
    }
    rankings.to_csv(paths["fuzzy_profile_ranking_summary"], index=False)
    build_normalized_risk_matrix(rankings).to_csv(paths["fuzzy_profile_ranking_matrix"])
    build_ci_patient_means(rankings).to_csv(paths["fuzzy_ci_patient_mean"], index=False)
    plot_fuzzy_profile_rankings(rankings, paths["fuzzy_profile_ranking_plot"])
    return paths


def build_fuzzy_profile_rankings(
    runs: list[ProfileRun], common_dir: str | Path
) -> pd.DataFrame:
    """Rank rankable patients by their mean total risk for every Fuzzy profile."""
    rows = []
    common_dir = Path(common_dir)
    for run in runs:
        profile_data = run.risk_timeseries.copy()
        rankable = profile_data["is_rankable_patient"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        mean_risks = (
            profile_data.loc[rankable]
            .groupby("patient_id", as_index=False)[TOTAL_RISK_COLUMN]
            .mean()
            .rename(columns={TOTAL_RISK_COLUMN: "mean_total_risk"})
            .sort_values(["mean_total_risk", "patient_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        minimum_risk = mean_risks["mean_total_risk"].min()
        maximum_risk = mean_risks["mean_total_risk"].max()
        risk_range = maximum_risk - minimum_risk
        mean_risks["normalized_mean_total_risk"] = (
            (mean_risks["mean_total_risk"] - minimum_risk) / risk_range
            if risk_range > 0
            else 0.5
        )
        ci = load_fuzzy_ci(common_dir / f"TFN_{run.fuzzy_profile}.csv")
        for rank, row in enumerate(mean_risks.itertuples(index=False), start=1):
            rows.append(
                {
                    "ahp_profile": run.ahp_profile,
                    "fuzzy_profile": run.fuzzy_profile,
                    "C_i": ci,
                    "rank": rank,
                    "patient_id": row.patient_id,
                    "mean_total_risk": row.mean_total_risk,
                    "normalized_mean_total_risk": row.normalized_mean_total_risk,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["C_i", "fuzzy_profile", "rank"], kind="stable"
    ).reset_index(drop=True)


def load_fuzzy_ci(tfn_path: str | Path) -> float:
    """Return ``(c9 - c10) + (c11 - c12)`` from a headerless TFN CSV."""
    tfn_path = Path(tfn_path)
    if not tfn_path.exists():
        raise FileNotFoundError(f"Fuzzy profile file not found: {tfn_path}")
    tfn = pd.read_csv(tfn_path, header=None, names=["l", "c", "r"])
    if len(tfn) < 12:
        raise ValueError(f"TFN CSV needs at least 12 rows: {tfn_path}")
    return float(
        # (tfn.iloc[8]["c"] - tfn.iloc[9]["c"])
        # + (tfn.iloc[10]["c"] - tfn.iloc[11]["c"])
        (tfn.iloc[10]["c"] - tfn.iloc[9]["c"])
        
    )


def build_normalized_risk_matrix(rankings: pd.DataFrame) -> pd.DataFrame:
    """Create a patient-by-Fuzzy-profile normalized mean-risk table."""
    labels = rankings[["fuzzy_profile", "C_i"]].drop_duplicates().copy()
    labels["label"] = labels.apply(
        lambda row: f"{row['fuzzy_profile']} (C_i={row['C_i']:g})", axis=1
    )
    label_lookup = dict(zip(labels["fuzzy_profile"], labels["label"]))
    matrix = rankings.assign(
        profile_label=rankings["fuzzy_profile"].map(label_lookup)
    ).pivot(
        index="patient_id",
        columns="profile_label",
        values="normalized_mean_total_risk",
    )
    return matrix.reindex(columns=labels["label"])


def build_ci_patient_means(rankings: pd.DataFrame) -> pd.DataFrame:
    """Average normalized risks for each patient within every equal-``C_i`` group."""
    return (
        rankings.groupby(["C_i", "patient_id"], as_index=False)
        .agg(
            mean_normalized_total_risk=("normalized_mean_total_risk", "mean"),
            profile_count=("fuzzy_profile", "nunique"),
        )
        .sort_values(["C_i", "patient_id"], kind="stable")
    )


def plot_fuzzy_profile_rankings(rankings: pd.DataFrame, output_png: str | Path) -> Path:
    """Plot normalized patient risks on a numeric ``C_i`` axis.

    Profiles sharing the same ``C_i`` are given small symmetric offsets so
    their individual profile points remain visible while their shared parameter
    value is apparent from the common tick/grid line.
    """
    patient_ids = sorted(rankings["patient_id"].unique())
    colors = dict(zip(patient_ids, plt.get_cmap("tab10").colors))
    profile_positions = _profile_x_positions(rankings)
    profile_ci = (
        rankings[["fuzzy_profile", "C_i"]]
        .drop_duplicates()
        .set_index("fuzzy_profile")["C_i"]
        .to_dict()
    )
    ci_values = sorted(set(profile_ci.values()))
    figure, axis = plt.subplots(figsize=(max(10, len(profile_positions) * 1.35), 5))

    for patient_id, patient_data in rankings.groupby("patient_id", sort=True):
        patient_data = patient_data.assign(
            x=patient_data["fuzzy_profile"].map(profile_positions)
        ).sort_values("x")
        axis.plot(
            patient_data["x"],
            patient_data["normalized_mean_total_risk"],
            marker="o",
            linewidth=1.25,
            color=colors[patient_id],
            alpha=0.35,
        )

    ci_patient_means = build_ci_patient_means(rankings)
    for patient_id, patient_data in ci_patient_means.groupby("patient_id", sort=True):
        axis.plot(
            patient_data["C_i"],
            patient_data["mean_normalized_total_risk"],
            marker="D",
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5,
            linewidth=2.5,
            color=colors[patient_id],
            label=f"{patient_id} mean",
        )

    for fuzzy_profile, x in sorted(profile_positions.items(), key=lambda item: item[1]):
        axis.text(
            x,
            -0.18,
            f"{fuzzy_profile} (C_i={profile_ci[fuzzy_profile]:g})",
            transform=axis.get_xaxis_transform(),
            rotation=45,
            ha="right",
            va="top",
            fontsize=8,
            fontfamily=JAPANESE_FONT,
        )

    axis.set_xticks(ci_values, [f"{ci:g}" for ci in ci_values])
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(
        min(profile_positions.values()) - 0.1,
        max(profile_positions.values()) + 0.1,
    )
    axis.set_xlabel("C_i", labelpad=62)
    axis.set_ylabel("Normalized mean total risk")
    axis.set_title("Fuzzy profile parameter C_i and normalized patient risk")
    axis.set_axisbelow(True)
    axis.grid(which="major", color="0.85")
    axis.legend(title="Patient ID (C_i group mean)", loc="best")
    figure.tight_layout(rect=(0, 0.22, 1, 1))
    output_png = Path(output_png)
    figure.savefig(output_png, dpi=200)
    plt.close(figure)
    return output_png


def _profile_x_positions(rankings: pd.DataFrame) -> dict[str, float]:
    """Return numeric x positions, separating profiles with equal ``C_i``.

    Within an equal-``C_i`` group, profiles are arranged left-to-right by the
    descending normalized mean-risk difference for patients C and B.
    """
    profiles = rankings[["fuzzy_profile", "C_i"]].drop_duplicates()
    c_minus_b = _normalized_c_minus_b(rankings)
    ci_values = sorted(profiles["C_i"].unique())
    gaps = np.diff(ci_values)
    minimum_gap = float(gaps[gaps > 0].min()) if len(gaps) else 1.0
    maximum_offset = minimum_gap * 0.35
    positions = {}
    for ci, group in profiles.groupby("C_i", sort=True):
        names = sorted(
            group["fuzzy_profile"],
            key=lambda name: (-c_minus_b.get(name, float("-inf")), name),
        )
        offsets = np.linspace(-maximum_offset, maximum_offset, len(names))
        if len(names) == 1:
            offsets = [0.0]
        positions.update(
            {profile_name: float(ci + offset) for profile_name, offset in zip(names, offsets)}
        )
    return positions


def _normalized_c_minus_b(rankings: pd.DataFrame) -> dict[str, float]:
    """Return each Fuzzy profile's normalized mean-risk difference C minus B."""
    required_columns = {
        "fuzzy_profile",
        "patient_id",
        "normalized_mean_total_risk",
    }
    if not required_columns.issubset(rankings.columns):
        return {}
    patient_risks = rankings.pivot(
        index="fuzzy_profile",
        columns="patient_id",
        values="normalized_mean_total_risk",
    )
    if "B" not in patient_risks.columns or "C" not in patient_risks.columns:
        return {}
    return (patient_risks["C"] - patient_risks["B"]).to_dict()


def _select_ahp_profile(runs: list[ProfileRun], ahp_profile: str | None) -> list[ProfileRun]:
    available = sorted({run.ahp_profile for run in runs})
    if ahp_profile is None:
        if len(available) != 1:
            raise ValueError(
                "Multiple AHP profiles found. Specify one with --ahp-profile: "
                + ", ".join(available)
            )
        return runs
    selected = [run for run in runs if run.ahp_profile == ahp_profile]
    if not selected:
        raise ValueError(f"AHP profile not found: {ahp_profile}")
    return selected
