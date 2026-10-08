"""Contrast vectors, normalization, projection scales, random controls, and simple readouts.

All functions are pure NumPy so the evaluator, fixtures and reference oracle share one
implementation that is checked against hand calculations in ``tests/unit/test_vectors.py``.

Conventions (registered in docs/protocol.md):
* ``h[i, m, p, l]`` is the activation at matched construction context ``i``, intended mode ``m``,
  matched polarity/persona stratum ``p``, layer ``l``. Here a *group* is the pair ``(i, p)``.
* Mode contrasts use equal weights over the declared matched groups:
    v_ER = mean_g(h[g,E] - h[g,R]);  v_ES = mean_g(h[g,E] - h[g,S]);  v_RS = mean_g(h[g,R] - h[g,S])
  so that, on identical group sets, v_ES == v_ER + v_RS exactly (raw vectors). Unit normalization
  is a separate operation and the identity does not hold for independently normalized vectors.
* Near-zero norms are flagged, never silently divided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODES = ("roleplay", "simulation", "enactment")
MODE_SHORT = {"roleplay": "R", "simulation": "S", "enactment": "E"}
CONTRASTS = {
    "ER": ("enactment", "roleplay"),
    "ES": ("enactment", "simulation"),
    "RS": ("roleplay", "simulation"),
}

NEAR_ZERO_REL = 1e-6


@dataclass
class ContrastVector:
    contrast_id: str
    vector: np.ndarray  # raw, un-normalized
    norm: float
    n_groups: int
    group_ids: list[str]
    near_zero: bool
    per_group_diffs: np.ndarray = field(repr=False)  # (n_groups, dim) for resampling

    def to_record(self) -> dict[str, Any]:
        return {
            "contrast_id": self.contrast_id,
            "norm": self.norm,
            "n_groups": self.n_groups,
            "near_zero": self.near_zero,
            "dim": int(self.vector.shape[0]),
        }


def _as_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("activation must be a 1-D vector per stimulus")
    return x


def paired_mode_contrasts(
    activations: dict[tuple[str, str], np.ndarray],
    contrasts: dict[str, tuple[str, str]] | None = None,
) -> dict[str, ContrastVector]:
    """Compute paired mode contrasts.

    ``activations`` maps ``(group_id, mode)`` to a 1-D activation at one layer. Only groups that
    have *all* modes required by a contrast contribute to that contrast; group ids are reported so
    the identity check can confirm equal group sets.
    """
    contrasts = contrasts or CONTRASTS
    groups = sorted({g for g, _ in activations})
    out: dict[str, ContrastVector] = {}
    for cid, (a, b) in contrasts.items():
        diffs = []
        used = []
        for g in groups:
            if (g, a) in activations and (g, b) in activations:
                diffs.append(_as_matrix(activations[(g, a)]) - _as_matrix(activations[(g, b)]))
                used.append(g)
        if not diffs:
            raise ValueError(f"no matched groups for contrast {cid}")
        d = np.stack(diffs)
        v = d.mean(axis=0)
        scale = float(np.mean([np.linalg.norm(x) for x in d])) or 1.0
        norm = float(np.linalg.norm(v))
        out[cid] = ContrastVector(
            contrast_id=cid,
            vector=v,
            norm=norm,
            n_groups=len(used),
            group_ids=used,
            near_zero=norm <= NEAR_ZERO_REL * scale,
            per_group_diffs=d,
        )
    return out


def check_raw_identity(cv: dict[str, ContrastVector], atol: float = 1e-9) -> tuple[bool, float]:
    """v_ES == v_ER + v_RS holds exactly only when all three used the same group set."""
    if set(cv["ER"].group_ids) != set(cv["ES"].group_ids) or set(cv["RS"].group_ids) != set(
        cv["ES"].group_ids
    ):
        return False, float("nan")
    resid = cv["ES"].vector - (cv["ER"].vector + cv["RS"].vector)
    err = float(np.max(np.abs(resid))) if resid.size else 0.0
    return err <= atol, err


def unit_vector(v: np.ndarray, reference_scale: float | None = None) -> tuple[np.ndarray, bool]:
    """Return ``(u, ok)``. ``ok`` is False for near-zero norms and ``u`` is then all zeros.

    Callers must check ``ok`` and must not manufacture a direction from a near-zero vector."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    scale = reference_scale if reference_scale is not None else 1.0
    if n <= NEAR_ZERO_REL * max(scale, 1e-300):
        return np.zeros_like(v), False
    return v / n, True


def projection_scale(H_construction: np.ndarray, u: np.ndarray) -> float:
    """``s`` = construction-set standard deviation of ``h . u`` (population std, ddof=0)."""
    H = np.asarray(H_construction, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError("H_construction must be (n, dim)")
    proj = H @ np.asarray(u, dtype=np.float64)
    return float(proj.std(ddof=0))


def intervention_displacement(alpha: float, s: float, u: np.ndarray) -> np.ndarray:
    """Intended displacement ``alpha * s * u`` (alpha in projection-std units)."""
    return float(alpha) * float(s) * np.asarray(u, dtype=np.float64)


def random_directions(dim: int, n: int, seed: int, norm: float) -> np.ndarray:
    """``n`` seeded random unit directions scaled to the same injected L2 magnitude ``norm``."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, dim))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * float(norm)


def within_mode_polarity_contrast(
    activations: dict[tuple[str, str, str], np.ndarray], mode: str, pos: str, neg: str
) -> ContrastVector:
    """Control contrast within one mode: mean_ctx(h[ctx, mode, pos] - h[ctx, mode, neg]).

    ``activations`` maps ``(context_id, mode, polarity)`` to a vector."""
    ctxs = sorted({c for c, m, _ in activations if m == mode})
    diffs, used = [], []
    for c in ctxs:
        if (c, mode, pos) in activations and (c, mode, neg) in activations:
            diffs.append(_as_matrix(activations[(c, mode, pos)]) - _as_matrix(activations[(c, mode, neg)]))
            used.append(c)
    if not diffs:
        raise ValueError(f"no contexts with both polarities for mode {mode}")
    d = np.stack(diffs)
    v = d.mean(axis=0)
    norm = float(np.linalg.norm(v))
    scale = float(np.mean([np.linalg.norm(x) for x in d])) or 1.0
    return ContrastVector(
        contrast_id=f"{MODE_SHORT[mode]}_{pos}_minus_{neg}",
        vector=v,
        norm=norm,
        n_groups=len(used),
        group_ids=used,
        near_zero=norm <= NEAR_ZERO_REL * scale,
        per_group_diffs=d,
    )


def mode_by_polarity_interaction(
    activations: dict[tuple[str, str, str], np.ndarray],
    mode_a: str = "enactment",
    mode_b: str = "roleplay",
    pos: str = "positive",
    neu: str = "neutral",
) -> ContrastVector:
    """Preregistered secondary contrast:
    mean_ctx[(h[a,pos]-h[a,neu]) - (h[b,pos]-h[b,neu])]."""
    ctxs = sorted({c for c, _, _ in activations})
    diffs, used = [], []
    for c in ctxs:
        keys = [(c, mode_a, pos), (c, mode_a, neu), (c, mode_b, pos), (c, mode_b, neu)]
        if all(k in activations for k in keys):
            a = _as_matrix(activations[keys[0]]) - _as_matrix(activations[keys[1]])
            b = _as_matrix(activations[keys[2]]) - _as_matrix(activations[keys[3]])
            diffs.append(a - b)
            used.append(c)
    if not diffs:
        raise ValueError("no contexts with the full 2x2 for the interaction")
    d = np.stack(diffs)
    v = d.mean(axis=0)
    norm = float(np.linalg.norm(v))
    scale = float(np.mean([np.linalg.norm(x) for x in d])) or 1.0
    return ContrastVector(
        contrast_id=f"{MODE_SHORT[mode_a]}x{MODE_SHORT[mode_b]}_{pos}_{neu}_interaction",
        vector=v,
        norm=norm,
        n_groups=len(used),
        group_ids=used,
        near_zero=norm <= NEAR_ZERO_REL * scale,
        per_group_diffs=d,
    )


# ----------------------------------------------------------------------------------------------
# Readouts on projections
# ----------------------------------------------------------------------------------------------


def mann_whitney_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(score_pos > score_neg) + 0.5 P(tie); exact O(n*m) for the small sizes used here."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def projection_discrimination(H_a: np.ndarray, H_b: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """Discrimination of two stimulus sets along ``u`` (held-out use only)."""
    pa = np.asarray(H_a, dtype=np.float64) @ u
    pb = np.asarray(H_b, dtype=np.float64) @ u
    return {
        "auc": mann_whitney_auc(pa, pb),
        "mean_diff": float(pa.mean() - pb.mean()),
        "n_a": int(pa.size),
        "n_b": int(pb.size),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def per_dimension_standardizer(
    H_construction: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Optional registered preprocessing for models with massive-activation dimensions (Gemma 2/3):
    per-dimension mean/std fitted on construction data only. Returns (mean, std)."""
    H = np.asarray(H_construction, dtype=np.float64)
    return H.mean(axis=0), H.std(axis=0, ddof=0) + eps
