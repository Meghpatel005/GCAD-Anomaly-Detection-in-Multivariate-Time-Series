"""Synthetic nonlinear causal benchmark with *causal-break* anomalies.

Mirrors the motivating example of the paper (water pipeline): in normal operation a few channels
are driven by exogenous signals and the remaining channels are nonlinear functions of lagged
parents. Anomalies *decouple* a child from its parents (replace the causal drive by an independent
surrogate with similar variance), so univariate statistics barely change while the causal pattern does.

The ground-truth adjacency is returned so causal-structure recovery can be checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SyntheticData:
    train: np.ndarray  # (T_train, N)
    test: np.ndarray  # (T_test, N)
    labels: np.ndarray  # (T_test,) int8
    true_graph: np.ndarray  # (N, N) bool, [i, j] = i causes j
    meta: dict[str, Any] = field(default_factory=dict)


def _build_graph(n: int, n_roots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((n, n))
    lags = np.ones((n, n), dtype=int)
    for j in range(n_roots, n):
        k = int(rng.integers(1, 3))
        parents = rng.choice(j, size=min(k, j), replace=False)
        for i in parents:
            weights[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.0)
            lags[i, j] = int(rng.integers(1, 4))
    return weights, lags


def _simulate(
    length: int,
    weights: np.ndarray,
    lags: np.ndarray,
    n_roots: int,
    periods: np.ndarray,
    phases: np.ndarray,
    noise: float,
    decouple: np.ndarray | None,
    rng: np.random.Generator,
    burn: int = 50,
    strength: float = 1.0,
) -> np.ndarray:
    n = weights.shape[0]
    total = length + burn
    x = np.zeros((total, n))
    surrogate = np.zeros(n)
    max_lag = int(lags.max())
    # Std of the causal drive of each channel (used to scale the independent surrogate).
    drive_scale = np.sqrt(np.maximum((weights**2).sum(axis=0) * 0.3, 1e-3))
    for t in range(total):
        for r in range(n_roots):
            x[t, r] = np.sin(2 * np.pi * t / periods[r] + phases[r]) + noise * rng.standard_normal()
        surrogate = 0.8 * surrogate + 0.6 * rng.standard_normal(n)
        if t < max_lag:
            continue
        for j in range(n_roots, n):
            parents = np.flatnonzero(weights[:, j])
            broken = decouple is not None and t >= burn and decouple[t - burn, j]
            if broken:
                # Interpolate between normal causal drive and independent surrogate.
                # At strength=1.0 this reduces to surrogate[j]*drive_scale[j] exactly
                # (same as the original code), so bit-for-bit reproducibility is preserved.
                causal = sum(weights[i, j] * np.tanh(x[t - lags[i, j], i]) for i in parents)
                drive = (1.0 - strength) * causal + strength * surrogate[j] * drive_scale[j]
            else:
                drive = sum(weights[i, j] * np.tanh(x[t - lags[i, j], i]) for i in parents)
            x[t, j] = 0.4 * x[t - 1, j] + drive + noise * rng.standard_normal()
    return x[burn:]


def generate_synthetic(
    n_channels: int = 10,
    n_roots: int = 3,
    n_train: int = 3000,
    n_test: int = 3000,
    anomaly_ratio: float = 0.1,
    noise: float = 0.1,
    segment_min: int = 40,
    segment_max: int = 120,
    seed: int = 0,
    anomaly_strength: float = 1.0,
) -> SyntheticData:
    if not 1 <= n_roots < n_channels:
        raise ValueError("need 1 <= n_roots < n_channels")
    if not 0.0 < anomaly_ratio < 0.5:
        raise ValueError("anomaly_ratio must be in (0, 0.5)")
    if segment_min < 2 or segment_max < segment_min or segment_max >= n_test:
        raise ValueError("invalid anomaly segment lengths")
    if not (0.0 < anomaly_strength <= 1.0):
        raise ValueError("anomaly_strength must be in (0, 1]")
    rng = np.random.default_rng(seed)
    weights, lags = _build_graph(n_channels, n_roots, rng)
    periods = rng.uniform(20, 60, size=n_roots)
    phases = rng.uniform(0, 2 * np.pi, size=n_roots)

    train = _simulate(n_train, weights, lags, n_roots, periods, phases, noise, None, rng)

    mask = np.zeros((n_test, n_channels), dtype=bool)
    labels = np.zeros(n_test, dtype=np.int8)
    target = int(anomaly_ratio * n_test)
    children = np.arange(n_roots, n_channels)
    guard = 0
    while labels.sum() < target and guard < 10_000:
        guard += 1
        length = int(rng.integers(segment_min, segment_max + 1))
        start = int(rng.integers(0, n_test - length))
        if labels[max(0, start - 20) : start + length + 20].any():
            continue  # keep segments separated
        k = int(rng.integers(1, max(2, len(children) // 2 + 1)))
        affected = rng.choice(children, size=min(k, len(children)), replace=False)
        mask[start : start + length, affected] = True
        labels[start : start + length] = 1
    test = _simulate(n_test, weights, lags, n_roots, periods, phases, noise, mask, rng, strength=anomaly_strength)

    return SyntheticData(
        train=train.astype(np.float32),
        test=test.astype(np.float32),
        labels=labels,
        true_graph=weights != 0,
        meta={"seed": seed, "n_roots": n_roots, "anomaly_ratio_actual": float(labels.mean())},
    )
