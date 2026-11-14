from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from gcad.causality import (
    causality_from_gradients,
    channel_gradients,
    compute_causality,
    compute_causality_for_indices,
    predictor_jacobian,
    sparsify,
)
from gcad.model import MixerPredictor


class LastStepLinear(nn.Module):
    """yhat_j = sum_i W[j, i] * x[i, last]  -> causal structure known exactly."""

    def __init__(self, w: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(w.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., -1] @ self.weight.T


def test_jacobian_matches_autograd_loop():
    torch.manual_seed(0)
    model = MixerPredictor(4, 3, hidden_dim=8, n_layers=2, dropout=0.0).eval()
    x = torch.randn(5, 4, 3)
    jac = predictor_jacobian(model, x)
    assert jac.shape == (5, 4, 4, 3)
    for b in range(5):
        for j in range(4):
            xi = x[b : b + 1].clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(model(xi)[0, j], xi)
            assert torch.allclose(jac[b, j], grad[0], atol=1e-5)


def test_causality_recovers_known_linear_structure():
    w = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # channel 0 <- channel 1
    model = LastStepLinear(w)
    tau = 4
    x = torch.randn(6, 3, tau)
    y = torch.randn(6, 3)
    a, sq = compute_causality(model, x, y, "loss")
    resid = (x[..., -1] @ w.T) - y
    assert torch.allclose(sq, resid**2, atol=1e-6)
    # A[i, j] = mean_lag |dL_j/dx_i| = |2 r_j W[j, i]| / tau  (only the last lag is non-zero)
    expected = torch.zeros(6, 3, 3)
    expected[:, 1, 0] = (2 * resid[:, 0].abs() * 2.0) / tau
    assert torch.allclose(a, expected, atol=1e-5)
    a_pred, _ = compute_causality(model, x, y, "prediction")
    assert torch.allclose(a_pred[:, 1, 0], torch.full((6,), 2.0 / tau), atol=1e-6)
    assert a_pred[:, 0, 1].abs().max() == 0


def test_model_state_restored_after_extraction():
    model = MixerPredictor(3, 3, hidden_dim=4, n_layers=1)
    model.train()
    compute_causality(model, torch.randn(2, 3, 3), torch.randn(2, 3))
    assert model.training and all(p.requires_grad for p in model.parameters())


def test_unknown_grad_target():
    model = LastStepLinear(torch.eye(2))
    with pytest.raises(ValueError):
        channel_gradients(model, torch.randn(1, 2, 3), torch.randn(1, 2), "nope")


def test_batched_extraction_matches_single_batch():
    torch.manual_seed(1)
    model = MixerPredictor(3, 4, hidden_dim=8, n_layers=1, dropout=0.0)
    series = torch.randn(80, 3)
    idx = np.arange(0, 60, 3)
    a1, e1 = compute_causality_for_indices(model, series, idx, 4, batch_size=4)
    a2, e2 = compute_causality_for_indices(model, series, idx, 4, batch_size=64)
    assert np.allclose(a1, a2, atol=1e-5) and np.allclose(e1, e2, atol=1e-6)
    assert a1.shape == (20, 3, 3) and (a1 >= 0).all()


def test_causality_from_gradients_layout():
    g = torch.zeros(1, 3, 3, 2)
    g[0, 2, 0, :] = torch.tensor([1.0, -3.0])  # effect j=2, cause i=0
    a = causality_from_gradients(g)
    assert a[0, 0, 2] == 2.0 and a.sum() == 2.0


def test_sparsify_properties():
    rng = np.random.default_rng(0)
    a = rng.random((7, 5, 5)).astype(np.float32)
    s = sparsify(a, threshold=0.0)
    off = ~np.eye(5, dtype=bool)
    assert (s >= 0).all()
    assert np.allclose(np.diagonal(s, axis1=1, axis2=2), np.diagonal(a, axis1=1, axis2=2))
    assert (np.minimum(s, np.swapaxes(s, 1, 2))[:, off] == 0).all()  # no bidirectional edges remain
    assert np.allclose(s[:, 0, 1] + (s[:, 1, 0]), np.abs(a[:, 0, 1] - a[:, 1, 0]), atol=1e-6)
    t = sparsify(a, threshold=0.3)
    assert ((t == 0) | (t >= 0.3)).all()


def test_sparsify_symmetric_matrix_has_no_offdiagonal_edges():
    m = np.array([[1.0, 2.0], [2.0, 3.0]])
    s = sparsify(m)
    assert s[0, 1] == 0 and s[1, 0] == 0 and s[0, 0] == 1 and s[1, 1] == 3


def test_sparsify_requires_square():
    with pytest.raises(ValueError):
        sparsify(np.zeros((2, 3)))
