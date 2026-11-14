from __future__ import annotations

import numpy as np
import pytest

from gcad.metrics import evaluate_scores, sanitize_scores
from gcad.scoring import deviation_matrix, deviation_scores, normal_pattern, static_resid_graphs


def test_normal_pattern_is_mean_of_sparsified_graphs():
    g = np.zeros((2, 2, 2))
    g[0, 0, 1], g[1, 0, 1] = 1.0, 3.0
    bar = normal_pattern(g, threshold=0.0)
    assert bar[0, 1] == 2.0 and bar[1, 0] == 0.0


def test_normal_pattern_validates_input():
    with pytest.raises(ValueError):
        normal_pattern(np.zeros((0, 2, 2)), 0.0)


def test_identical_graph_has_zero_deviation():
    rng = np.random.default_rng(0)
    graphs = rng.random((4, 3, 3))
    bar = normal_pattern(graphs, 0.0, do_sparsify=False)
    out = deviation_scores(
        np.broadcast_to(bar, (5, 3, 3)).copy(), bar, threshold=0.0, beta=1.0, eps=1e-3, do_sparsify=False
    )
    assert np.allclose(out["S"], 0.0)


def test_scores_match_hand_computation():
    bar = np.array([[1.0, 0.0], [0.0, 2.0]])
    test = np.array([[[2.0, 1.0], [0.0, 2.0]]])  # one window
    eps = 0.5
    out = deviation_scores(test, bar, threshold=0.0, beta=2.0, eps=eps, do_sparsify=False)
    # Sc = |2-1|/(1+eps) + |1-0|/(0+eps) + 0 + 0 ; St = |2-1|/(1+eps) + 0
    sc = 1 / 1.5 + 1 / 0.5
    st = 1 / 1.5
    assert np.isclose(out["Sc"][0], sc) and np.isclose(out["St"][0], st) and np.isclose(out["S"][0], sc + 2 * st)
    # Sc_off = only off-diagonal (0,1) entry: |1-0|/(0+eps) = 1/0.5 = 2.0
    sc_off_expected = 1 / 0.5
    assert np.isclose(out["Sc_off"][0], sc_off_expected)
    # Sc = Sc_off + diagonal part  (diagonal part = |2-1|/(1+eps))
    assert np.isclose(out["Sc"][0], out["Sc_off"][0] + 1 / 1.5)


def test_chunking_is_consistent():
    rng = np.random.default_rng(1)
    graphs = rng.random((50, 4, 4)).astype(np.float32)
    bar = normal_pattern(graphs[:10], 0.1)
    a = deviation_scores(graphs, bar, threshold=0.1, beta=1.0, eps=1e-2, chunk=7)
    b = deviation_scores(graphs, bar, threshold=0.1, beta=1.0, eps=1e-2, chunk=1000)
    assert np.allclose(a["S"], b["S"])


def test_shape_mismatch():
    with pytest.raises(ValueError):
        deviation_scores(np.zeros((2, 3, 3)), np.zeros((2, 2)), threshold=0, beta=1, eps=1e-3)


def test_deviation_matrix():
    bar = np.zeros((2, 2))
    d = deviation_matrix(np.array([[1.0, 0.5], [0.0, 1.0]]), bar, threshold=0.0)
    assert d[0, 1] == 0.5 and d[1, 0] == 0.0 and d.shape == (2, 2)


def test_metrics_perfect_and_random():
    labels = np.array([0, 0, 1, 1])
    assert evaluate_scores(labels, np.array([0.1, 0.2, 0.8, 0.9])) == {"auroc": 1.0, "auprc": 1.0}
    assert evaluate_scores(labels, np.array([0.9, 0.8, 0.2, 0.1]))["auroc"] == 0.0


def test_metrics_edge_cases():
    assert np.isnan(evaluate_scores(np.zeros(4), np.arange(4.0))["auroc"])
    with pytest.raises(ValueError):
        evaluate_scores(np.array([0, 1]), np.array([0.1]))
    # 50% NaN is above the default 1% threshold — must pass max_nonfinite_fraction=1.0 explicitly
    s = sanitize_scores(np.array([1.0, np.nan, np.inf, 3.0]), max_nonfinite_fraction=1.0)
    assert np.isfinite(s).all() and s[2] == 3.0 and s[1] == 1.0
    with pytest.raises(ValueError):
        sanitize_scores(np.array([np.nan, np.nan]))


def test_sanitize_scores_small_fraction_passes():
    """2 / 1000 non-finite (0.2 %) is below the default 1 % threshold."""
    scores = np.ones(1000)
    scores[0] = np.nan
    scores[1] = np.nan
    result = sanitize_scores(scores)
    assert np.isfinite(result).all()


def test_sanitize_scores_large_fraction_raises():
    """20 / 1000 non-finite (2 %) exceeds the default 1 % threshold."""
    scores = np.ones(1000)
    scores[:20] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sanitize_scores(scores)


def test_static_resid_graphs_shape():
    rng = np.random.default_rng(0)
    sq_err = rng.random((7, 4)).astype(np.float32)
    a_static = rng.random((4, 4)).astype(np.float32)
    out = static_resid_graphs(sq_err, a_static)
    assert out.shape == (7, 4, 4)


def test_static_resid_graphs_scales_column_j():
    """Column j of the output scales linearly with |r_j| = sqrt(sq_err[:, j])."""
    sq_err = np.array([[4.0, 1.0]])  # r = [2, 1]
    a_static = np.ones((2, 2))
    out = static_resid_graphs(sq_err, a_static)
    # out[0, :, 0] = 2 * sqrt(4) * 1 = 4; out[0, :, 1] = 2 * sqrt(1) * 1 = 2
    assert np.allclose(out[0, :, 0], 4.0) and np.allclose(out[0, :, 1], 2.0)


def test_static_resid_graphs_zero_residual():
    sq_err = np.zeros((3, 4))
    a_static = np.ones((4, 4))
    out = static_resid_graphs(sq_err, a_static)
    assert np.allclose(out, 0.0)
