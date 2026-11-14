"""Lightweight baselines evaluated alongside GCAD on identical windows.

The six baselines of the paper (DAGMM, USAD, GDN, Anomaly Transformer, GANF, MEMTO) are *not*
re-implemented here; their published numbers are listed in ``paper_results.py`` for reference.
These simple baselines isolate what the causal-deviation score adds on top of the same predictor.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def prediction_error_scores(sq_err: np.ndarray) -> np.ndarray:
    """Total squared one-step prediction error per window, from (M, N) channel errors."""
    return sq_err.sum(axis=1)


def pca_reconstruction_scores(
    train: np.ndarray, test: np.ndarray, indices: np.ndarray, tau: int, variance: float = 0.9
) -> np.ndarray:
    """PCA reconstruction error of the target step ``x[i + tau]`` for every window index ``i``."""
    pca = PCA(n_components=variance, svd_solver="full").fit(train)
    targets = test[indices + tau]
    recon = pca.inverse_transform(pca.transform(targets))
    return ((targets - recon) ** 2).sum(axis=1)
