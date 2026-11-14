"""Mixer-style predictor (TSMixer-like) used as the gradient generator.

Input  : sliding window ``X_{t-1}`` of shape (B, N, tau)  (N channels, tau time lags)
Output : one-step-ahead prediction ``yhat_t`` of shape (B, N)

Each mixer layer applies a *time-mixing* MLP (shared across the N channels) followed by a
*feature-mixing* MLP (shared across time steps), both with residual connections. A final linear
layer over the time axis (the "fully connected layer" of the paper) maps the sum of all layer outputs
(skip connections) to the prediction. Normalisation is a per-axis LayerNorm, which simplifies TSMixer's 2-D norm.
Only smooth ops are used (GELU, LayerNorm) so input gradients are well defined.
"""

from __future__ import annotations

import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerLayer(nn.Module):
    def __init__(self, n_channels: int, window: int, hidden: int, dropout: float) -> None:
        super().__init__()
        # LayerNorm over a single element is degenerate (always zero), so skip it when window == 1.
        self.time_norm = nn.LayerNorm(window) if window > 1 else nn.Identity()
        self.time_mlp = _MLP(window, hidden, dropout)
        self.feat_norm = nn.LayerNorm(n_channels) if n_channels > 1 else nn.Identity()
        self.feat_mlp = _MLP(n_channels, hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, tau)
        x = x + self.time_mlp(self.time_norm(x))
        z = x.transpose(1, 2)  # (B, tau, N)
        z = z + self.feat_mlp(self.feat_norm(z))
        return z.transpose(1, 2)


class MixerPredictor(nn.Module):
    def __init__(self, n_channels: int, window: int, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if n_channels < 1 or window < 1:
            raise ValueError("n_channels and window must be >= 1")
        self.n_channels, self.window = n_channels, window
        self.layers = nn.ModuleList(MixerLayer(n_channels, window, hidden_dim, dropout) for _ in range(n_layers))
        self.head = nn.Linear(window, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.n_channels, self.window):
            raise ValueError(f"expected (*, {self.n_channels}, {self.window}), got {tuple(x.shape)}")
        skip = torch.zeros_like(x)
        for layer in self.layers:
            x = layer(x)
            skip = skip + x  # every layer's output reaches the head (paper, "Prediction-based Gradient Generator")
        return self.head(skip).squeeze(-1)
