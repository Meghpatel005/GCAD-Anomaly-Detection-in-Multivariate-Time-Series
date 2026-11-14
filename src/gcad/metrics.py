"""Threshold-independent evaluation metrics (AUROC and AUPRC)."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

LOG = logging.getLogger("gcad.metrics")


def sanitize_scores(scores: np.ndarray, max_nonfinite_fraction: float = 0.01) -> np.ndarray:
    """Replace NaN/inf so ranking metrics stay defined (NaN -> min, +inf -> max finite).

    Raises ValueError if more than ``max_nonfinite_fraction`` of scores are non-finite,
    so a silently diverged scoring pipeline is caught early rather than returning a
    plausible-looking AUROC computed from mostly-NaN scores.
    """
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    if not finite.any():
        raise ValueError("all scores are non-finite")
    pct = (~finite).mean()
    if pct > max_nonfinite_fraction:
        raise ValueError(f"{100 * pct:.1f}% of scores are non-finite")
    if finite.all():
        return s
    lo, hi = s[finite].min(), s[finite].max()
    LOG.warning("%d non-finite scores replaced", int((~finite).sum()))
    return np.nan_to_num(s, nan=lo, posinf=hi, neginf=lo)


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    max_nonfinite_fraction: float = 0.01,
) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    if labels.shape != np.shape(scores):
        raise ValueError(f"labels {labels.shape} and scores {np.shape(scores)} differ in shape")
    if len(np.unique(labels)) < 2:
        LOG.warning("labels contain a single class; metrics are undefined")
        return {"auroc": float("nan"), "auprc": float("nan")}
    s = sanitize_scores(scores, max_nonfinite_fraction)
    return {"auroc": float(roc_auc_score(labels, s)), "auprc": float(average_precision_score(labels, s))}
