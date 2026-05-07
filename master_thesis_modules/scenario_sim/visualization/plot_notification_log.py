"""Plot notification events on total-risk time series."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_notification_log(
    risk_timeseries_csv: str | Path,
    notification_log_csv: str | Path,
    output_png: str | Path,
) -> Path:
    risk_data = pd.read_csv(risk_timeseries_csv)
    if Path(notification_log_csv).exists():
        notification_data = pd.read_csv(notification_log_csv)
    else:
        notification_data = pd.DataFrame()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for patient_id, group in risk_data.groupby("patient_id"):
        plt.plot(group["timestamp"], group["10000000"], marker="o", label=patient_id)
    if not notification_data.empty:
        for _, row in notification_data.iterrows():
            color = "tab:red" if row["notification_type"] == "notice" else "tab:blue"
            plt.axvline(row["timestamp"], color=color, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Total risk")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    return output_png

