"""Plot total-risk time series."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_risk_timeseries(input_csv: str | Path, output_png: str | Path) -> Path:
    data = pd.read_csv(input_csv)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for patient_id, group in data.groupby("patient_id"):
        style = _patient_line_style(group)
        plt.plot(
            group["timestamp"],
            group["10000000"],
            marker="o",
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=style["alpha"],
            label=patient_id,
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Total risk")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    return output_png


def _patient_line_style(patient_data: pd.DataFrame) -> dict[str, object]:
    if "is_rankable_patient" in patient_data.columns:
        values = patient_data["is_rankable_patient"].dropna()
        if not values.empty and not values.map(_to_bool).mode().iloc[0]:
            return {"linestyle": "--", "linewidth": 0.8, "markersize": 1.8, "alpha": 0.55}
    return {"linestyle": "-", "linewidth": 1.4, "markersize": 3.0, "alpha": 1.0}


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "staff"}
