"""Causal deviation scoring (paper Eqs. 7-12).

* ``normal_pattern``     : mean sparsified causality graph over sampled normal windows (Eqs. 7-8)
* ``deviation_scores``   : causal-pattern deviation Sc (Eq. 10), time-pattern deviation St (Eq. 11),
                           Sc_off (off-diagonal only), and the final score S = Sc + beta*St (Eq. 12)
* ``static_resid_graphs``: control variant — fixed graph x per-window residual (not in the paper)
"""

from __future__ import annotations

import numpy as np

from .causality import sparsify


def normal_pattern(a_normal: np.ndarray, threshold: float, do_sparsify: bool = True) -> np.ndarray:
    """Typical normal causal pattern ``A_norm`` (N, N) from raw graphs (n, N, N)."""
    if a_normal.ndim != 3 or len(a_normal) == 0:
        raise ValueError("a_normal must be a non-empty (n, N, N) array")
    graphs = sparsify(a_normal, threshold) if do_sparsify else a_normal
    return graphs.astype(np.float64).mean(axis=0)


def deviation_scores(
    a_test: np.ndarray,
    a_bar: np.ndarray,
    *,
    threshold: float,
    beta: float,
    eps: float,
    do_sparsify: bool = True,
    chunk: int = 2048,
) -> dict[str, np.ndarray]:
    """Return ``{"Sc", "St", "S", "Sc_off"}`` score sequences for raw test graphs (M, N, N).

    "Sc_off" is Sc restricted to off-diagonal entries (pure inter-series causal deviation).
    """
    if a_test.ndim != 3 or a_test.shape[1:] != a_bar.shape:
        raise ValueError(f"shape mismatch: a_test {a_test.shape}, a_bar {a_bar.shape}")
    m = len(a_test)
    n = a_bar.shape[0]
    sc, st, sc_off = np.empty(m), np.empty(m), np.empty(m)
    denom = a_bar + eps
    diag_bar = np.diag(a_bar)
    diag_denom = diag_bar + eps
    off = ~np.eye(n, dtype=bool)  # (N, N) mask of off-diagonal entries
    for s in range(0, m, chunk):
        block = a_test[s : s + chunk]
        graphs = (sparsify(block, threshold) if do_sparsify else block).astype(np.float64)
        diff = np.abs(graphs - a_bar)
        sc[s : s + chunk] = (diff / denom).sum(axis=(1, 2))  # Eq. 10
        sc_off[s : s + chunk] = ((diff / denom)[:, off]).sum(axis=1)  # off-diagonal only
        d = np.abs(np.diagonal(graphs, axis1=1, axis2=2) - diag_bar)
        st[s : s + chunk] = (d / diag_denom).sum(axis=1)  # Eq. 11
    return {"Sc": sc, "St": st, "S": sc + beta * st, "Sc_off": sc_off}  # Eq. 12


def static_resid_graphs(sq_err: np.ndarray, a_static: np.ndarray) -> np.ndarray:
    """Control variant: (M, N_cause=i, N_effect=j) where column j is scaled by 2*|r_j|.

    Replaces the per-window Jacobian with one fixed graph (mean dyhat/dx over normal windows)
    multiplied by the per-window residual magnitude. Tests whether the per-window Jacobian
    carries information beyond the residual.

    Args:
        sq_err:   (M, N) per-channel squared prediction errors (r_j^2)
        a_static: (N, N) fixed graph = mean prediction-gradient of normal windows

    Returns:
        (M, N, N) array where entry [m, i, j] = 2*|r_j[m]| * a_static[i, j]
    """
    return 2.0 * np.sqrt(sq_err)[:, None, :] * a_static[None, :, :]


def deviation_matrix(
    a_test_single: np.ndarray, a_bar: np.ndarray, threshold: float, do_sparsify: bool = True
) -> np.ndarray:
    """``D = |A~_test - A_norm|`` for one window (used for the case-study heat map)."""
    graph = sparsify(a_test_single, threshold) if do_sparsify else a_test_single
    return np.abs(graph.astype(np.float64) - a_bar)
