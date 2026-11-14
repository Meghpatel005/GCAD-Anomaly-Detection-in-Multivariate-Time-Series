from __future__ import annotations

import pytest
import torch

from gcad.model import MixerPredictor


@pytest.mark.parametrize("window", [1, 2, 5])
def test_forward_shape_and_gradient(window):
    model = MixerPredictor(n_channels=4, window=window, hidden_dim=8, n_layers=2)
    x = torch.randn(3, 4, window, requires_grad=True)
    out = model(x)
    assert out.shape == (3, 4)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_rejects_wrong_input_shape():
    model = MixerPredictor(n_channels=4, window=3)
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 3))


def test_invalid_construction():
    with pytest.raises(ValueError):
        MixerPredictor(n_channels=0, window=3)


def test_head_receives_sum_of_all_layer_outputs():
    torch.manual_seed(0)
    model = MixerPredictor(n_channels=3, window=4, hidden_dim=8, n_layers=3, dropout=0.0).eval()
    x = torch.randn(2, 3, 4)
    expected, h = torch.zeros_like(x), x
    for layer in model.layers:
        h = layer(h)
        expected = expected + h
    assert torch.allclose(model(x), model.head(expected).squeeze(-1), atol=1e-6)
