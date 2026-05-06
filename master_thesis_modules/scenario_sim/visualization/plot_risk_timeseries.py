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
        plt.plot(group["timestamp"], group["10000000"], marker="o", label=patient_id)
    plt.xlabel("Time [s]")
    plt.ylabel("Total risk")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    return output_png

