"""Independent small implementations used to validate the oracle (scientific tests).

These deliberately avoid ``npbench.vectors`` and the oracle's helpers."""

from __future__ import annotations

import numpy as np
from scipy import stats


def auc_scipy(a: np.ndarray, b: np.ndarray) -> float:
    """AUC via scipy's Mann-Whitney U (ties handled as 1/2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (a.size * b.size))


def contrast_vector_loop(h_a: dict[str, np.ndarray], h_b: dict[str, np.ndarray]) -> np.ndarray:
    """Equal-weight paired mean difference computed with an explicit Python loop."""
    keys = sorted(set(h_a) & set(h_b))
    acc = None
    for k in keys:
        d = np.asarray(h_a[k], dtype=float) - np.asarray(h_b[k], dtype=float)
        acc = d if acc is None else acc + d
    return acc / len(keys)


def slope_polyfit(alpha: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.asarray(alpha, float), np.asarray(y, float), 1)[0])
