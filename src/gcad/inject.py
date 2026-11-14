"""Synthetic anomaly injection into validation data for label-free hyperparameter tuning.

Injects anomalies into a copy of the (standardised) validation series so that
scoring hyperparameters can be tuned without touching the test set.
"""

from __future__ import annotations

import numpy as np


def inject_anomalies(
    x: np.ndarray,
    rng: np.random.Generator,
    ratio: float = 0.1,
    seg: tuple[int, int] = (20, 100),
) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy of x with synthetic anomalies and step-level 0/1 labels.

    Three anomaly kinds, chosen uniformly per segment:
    - ``shuffle_segment``: replace x[s:e, C] with x[s2:e2, C] from a different time
      (same marginals, cross-channel temporal dependence broken)
    - ``level_shift``: add ±3 std to one channel for the whole segment
    - ``spike``: add ±6 std to one channel for 1-3 steps

    Segments are non-overlapping with a 20-step gap between them. Deterministic given ``rng``.

    Args:
        x:     (T, N) standardised series (already z-scored, so channel std ≈ 1)
        rng:   seeded Generator for reproducibility
        ratio: target fraction of anomalous steps
        seg:   (min_length, max_length) of each anomaly segment

    Returns:
        (x_aug, labels): x_aug is a copy of x with injected anomalies; labels is int8 (T,)
    """
    T, N = x.shape
    x_aug = x.copy()
    labels = np.zeros(T, dtype=np.int8)
    seg_min, seg_max = seg
    target = int(ratio * T)
    guard = 0

    while labels.sum() < target and guard < 10_000:
        guard += 1
        max_seg = min(seg_max + 1, T // 4 + 1)
        if max_seg <= seg_min:
            break
        length = int(rng.integers(seg_min, max_seg))
        start = int(rng.integers(0, T - length))
        end = start + length

        # Enforce a 20-step buffer between segments
        gap_start = max(0, start - 20)
        gap_end = min(T, end + 20)
        if labels[gap_start:gap_end].any():
            continue

        kind = int(rng.integers(0, 3))  # 0=shuffle, 1=level_shift, 2=spike

        if kind == 0:  # shuffle_segment: rearrange values across time (keeps marginal distribution)
            n_ch = max(1, int(rng.integers(1, max(2, N // 2 + 1))))
            channels = rng.choice(N, size=n_ch, replace=False)
            placed = False
            for _ in range(100):
                s2 = int(rng.integers(0, T - length))
                if abs(s2 - start) >= length:  # source segment is at least `length` steps away
                    x_aug[start:end, channels] = x[s2 : s2 + length, channels]
                    placed = True
                    break
            if not placed:
                continue  # could not find a distant non-overlapping source; skip this attempt

        elif kind == 1:  # level_shift: sustained mean shift on one channel
            channel = int(rng.integers(0, N))
            sign = float(rng.choice(np.array([-1.0, 1.0])))
            x_aug[start:end, channel] += sign * 3.0  # 3 std shift

        else:  # spike: brief large-amplitude transient on one channel
            channel = int(rng.integers(0, N))
            sign = float(rng.choice(np.array([-1.0, 1.0])))
            spike_len = int(rng.integers(1, 4))
            spike_start = start + int(rng.integers(0, max(1, length - spike_len)))
            spike_end = min(spike_start + spike_len, end)
            x_aug[spike_start:spike_end, channel] += sign * 6.0  # 6 std spike

        labels[start:end] = 1

    return x_aug, labels

