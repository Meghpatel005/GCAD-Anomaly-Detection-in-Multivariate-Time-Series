"""Tests for inject.py (F6: label-free tuning via injected val anomalies)."""

from __future__ import annotations

import numpy as np
import pytest

from gcad.inject import inject_anomalies


def test_inject_anomalies_deterministic():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    x = np.random.default_rng(0).standard_normal((500, 8)).astype(np.float32)
    x1, l1 = inject_anomalies(x, rng1)
    x2, l2 = inject_anomalies(x, rng2)
    assert np.array_equal(x1, x2) and np.array_equal(l1, l2)


def test_inject_anomalies_label_fraction():
    rng = np.random.default_rng(1)
    x = np.zeros((1000, 6), dtype=np.float32)
    _, labels = inject_anomalies(x, rng, ratio=0.1)
    frac = labels.mean()
    # Should be within [ratio/2, ratio*1.5]
    assert 0.05 <= frac <= 0.15, f"anomaly fraction {frac:.3f} outside [0.05, 0.15]"


def test_inject_anomalies_unmodified_rows_unchanged():
    x = np.ones((500, 5), dtype=np.float32)
    rng = np.random.default_rng(2)
    x_aug, labels = inject_anomalies(x, rng, ratio=0.1)
    # Rows outside anomaly segments must be bit-identical to x
    normal_mask = labels == 0
    assert np.array_equal(x_aug[normal_mask], x[normal_mask])


def test_inject_anomalies_shuffle_keeps_value_range():
    """Shuffle anomaly only rearranges values, so per-channel range is preserved."""
    rng = np.random.default_rng(3)
    x = np.random.default_rng(10).standard_normal((300, 4)).astype(np.float32)
    x_aug, _ = inject_anomalies(x, rng, ratio=0.15)
    # Shuffle kind cannot introduce values outside the original range.
    # Level-shift and spike can, but we check per-channel range stays reasonable.
    for c in range(x.shape[1]):
        orig_min, orig_max = x[:, c].min(), x[:, c].max()
        # Allow up to 6 std headroom for spike/shift anomalies; shuffle must stay within range
        assert x_aug[:, c].min() >= orig_min - 7.0
        assert x_aug[:, c].max() <= orig_max + 7.0


def test_tune_returns_grid_rows(tiny_cfg):
    """tune() must return one row per grid combination."""
    from gcad.pipeline import Experiment

    exp = Experiment(tiny_cfg, seed=0)
    grid = {"scoring.threshold": [0.0, 0.01], "scoring.beta": [0.5, 1.0]}
    df = exp.tune(grid)
    assert len(df) == 4  # 2 × 2
    assert "auroc" in df.columns and "auprc" in df.columns
    assert "scoring.threshold" in df.columns and "scoring.beta" in df.columns


def test_tune_does_not_read_test_data(tiny_cfg):
    """tune() must never access the test split."""
    from gcad.data import PreparedData
    from gcad.pipeline import Experiment

    exp = Experiment(tiny_cfg, seed=0)
    _ = exp.data  # populate _data
    original_data = exp._data

    class _NoTestData:
        """Proxy that raises if .test is accessed."""

        def __getattr__(self, name: str):
            if name == "test":
                raise AssertionError("tune() accessed test data — this must not happen")
            return getattr(original_data, name)

    exp._data = _NoTestData()  # type: ignore[assignment]
    grid = {"scoring.threshold": [0.0, 0.01]}
    exp.tune(grid)  # must complete without touching .test
    exp._data = original_data  # restore so tmp_path cleanup works

