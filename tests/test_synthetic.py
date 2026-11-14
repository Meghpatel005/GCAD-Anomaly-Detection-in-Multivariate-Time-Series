from __future__ import annotations

import numpy as np
import pytest

from gcad.synthetic import generate_synthetic


def test_shapes_labels_and_determinism():
    a = generate_synthetic(n_channels=6, n_roots=2, n_train=300, n_test=400, segment_min=20, segment_max=40, seed=3)
    b = generate_synthetic(n_channels=6, n_roots=2, n_train=300, n_test=400, segment_min=20, segment_max=40, seed=3)
    assert a.train.shape == (300, 6) and a.test.shape == (400, 6) and a.labels.shape == (400,)
    assert np.array_equal(a.train, b.train) and np.array_equal(a.labels, b.labels)
    assert 0.05 < a.labels.mean() < 0.2
    assert np.isfinite(a.train).all() and np.isfinite(a.test).all()


def test_true_graph_is_a_dag_with_root_sources():
    s = generate_synthetic(n_channels=8, n_roots=3, n_train=100, n_test=300, segment_min=20, segment_max=40, seed=0)
    g = s.true_graph
    assert not np.tril(g).any()  # edges only go from lower to higher index -> acyclic
    assert not g[:, :3].any()  # roots have no parents
    assert g.any()


@pytest.mark.parametrize("kwargs", [{"n_roots": 0}, {"anomaly_ratio": 0.9}, {"segment_min": 1}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic(n_train=100, n_test=300, **kwargs)


# ------------------------------------------------------------------ F3: anomaly_strength
def test_anomaly_strength_one_reproduces_original():
    """strength=1.0 must produce bit-for-bit identical output to the unmodified function."""
    kw = {"n_channels": 6, "n_roots": 2, "n_train": 200, "n_test": 400, "segment_min": 20, "segment_max": 40, "seed": 5}
    a = generate_synthetic(**kw, anomaly_strength=1.0)
    b = generate_synthetic(**kw)  # default strength
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)
    assert np.array_equal(a.labels, b.labels)


def test_anomaly_strength_validation():
    kw = {"n_channels": 6, "n_roots": 2, "n_train": 100, "n_test": 200, "segment_min": 20, "segment_max": 40}
    with pytest.raises(ValueError, match="anomaly_strength"):
        generate_synthetic(**kw, anomaly_strength=0.0)
    with pytest.raises(ValueError, match="anomaly_strength"):
        generate_synthetic(**kw, anomaly_strength=1.1)


def test_anomaly_strength_monotone_mse():
    """Higher strength → higher prediction error on anomalous windows (AR(1) baseline)."""
    from sklearn.linear_model import LinearRegression

    kw = {"n_channels": 6, "n_roots": 2, "n_train": 200, "n_test": 600, "segment_min": 30, "segment_max": 60, "seed": 7}
    mses = []
    for strength in (0.25, 0.5, 1.0):
        syn = generate_synthetic(**kw, anomaly_strength=strength)
        anom_idx = np.where(syn.labels == 1)[0]
        # Require at least 5 anomalous points with valid lag-1 predecessors
        valid = anom_idx[anom_idx > 0]
        if len(valid) < 5:
            continue
        X = syn.test[valid - 1]
        y = syn.test[valid]
        pred = LinearRegression().fit(X, y).predict(X)
        mses.append(float(((y - pred) ** 2).mean()))

    assert len(mses) >= 2, "need at least 2 strength levels with anomalous data"
    assert all(mses[i] < mses[i + 1] for i in range(len(mses) - 1)), f"MSE not monotone: {mses}"
