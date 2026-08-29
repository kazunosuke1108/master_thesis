"""Load legacy AHP weights when CSV files are available locally."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from master_thesis_modules.risk_core.aggregators.ahp import AHPAggregator


def load_ahp_weights_from_matrix_csv(path: str | Path) -> np.ndarray:
    matrix = pd.read_csv(path, header=None).values.astype(float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                matrix[i, j] = 1.0
            elif i > j:
                matrix[i, j] = 1.0 / matrix[j, i]
    eigvals, eigvecs = np.linalg.eig(matrix)
    weights = eigvecs[:, eigvals.real.argmax()].real
    return weights / weights.sum()


class LegacyAHPAggregator(AHPAggregator):
    @classmethod
    def from_csv(cls, node_ids: tuple[int, ...], csv_path: str | Path) -> "LegacyAHPAggregator":
        weights = load_ahp_weights_from_matrix_csv(csv_path)
        return cls(dict(zip(node_ids, weights)))

