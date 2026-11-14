"""Gradient-based Granger causality discovery and graph sparsification (paper Eqs. 1-6).

For a window ``X_t`` (N x tau) and target ``y_t`` the *channel-separated error detector* gives
``L_{t,j} = (yhat_{t,j} - y_{t,j})^2`` for every channel ``j``. Back-propagating each ``L_{t,j}``
to the input yields a gradient ``G_{t,j}`` of shape (N, tau); stacking over j gives ``G_t`` (N, N, tau).
Since ``dL_j/dx = 2 (yhat_j - y_j) * dyhat_j/dx`` the full Jacobian of the predictor is computed once
(``torch.func.vmap(jacrev)``) instead of running N backward passes per window.

The causality matrix is ``A[i, j] = mean_over_lags |dL_j / dx_i|`` (uniform interest distribution P):
the degree to which series i Granger-causes series j.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from .data import gather_windows

LOG = logging.getLogger("gcad.causality")


# --------------------------------------------------------------------------- gradients
def predictor_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Jacobian of the predictor w.r.t. its input, shape (B, N_out, N_in, tau)."""

    def single(xi: torch.Tensor) -> torch.Tensor:
        return model(xi.unsqueeze(0)).squeeze(0)

    try:
        return torch.func.vmap(torch.func.jacrev(single))(x)
    except RuntimeError as exc:  # pragma: no cover - fallback for ops without vmap support
        LOG.warning("vmap(jacrev) failed (%s); falling back to a per-sample loop", exc)
        return torch.stack([torch.autograd.functional.jacobian(single, xi) for xi in x])


def channel_gradients(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, grad_target: str = "loss"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``G`` of shape (B, N_out=j, N_in=i, tau) and the per-channel squared error (B, N)."""
    jac = predictor_jacobian(model, x)
    with torch.no_grad():
        resid = model(x) - y
    sq_err = resid**2
    if grad_target == "loss":
        g = 2.0 * resid[:, :, None, None] * jac  # dL_j/dx = 2 r_j dyhat_j/dx
    elif grad_target == "prediction":
        g = jac
    else:
        raise ValueError(f"unknown grad_target {grad_target!r}")
    return g, sq_err


def causality_from_gradients(g: torch.Tensor) -> torch.Tensor:
    """Eq. 5 with a uniform P over the lag axis: ``A[b, i, j] = mean_lag |G[b, j, i, lag]|``."""
    return g.abs().mean(dim=-1).transpose(1, 2)


def compute_causality(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, grad_target: str = "loss"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causality matrices (B, N, N) and per-channel squared errors (B, N) for a batch of windows."""
    was_training = model.training
    requires = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        g, sq_err = channel_gradients(model, x, y, grad_target)
        return causality_from_gradients(g).detach(), sq_err.detach()
    finally:
        for p, r in zip(model.parameters(), requires, strict=True):
            p.requires_grad_(r)
        model.train(was_training)


def compute_causality_for_indices(
    model: nn.Module,
    series: torch.Tensor,
    indices: np.ndarray,
    tau: int,
    batch_size: int,
    grad_target: str = "loss",
) -> tuple[np.ndarray, np.ndarray]:
    """Causality matrices (M, N, N) and squared errors (M, N) for the windows starting at ``indices``."""
    device = next(model.parameters()).device
    n = series.shape[1]
    mats = np.empty((len(indices), n, n), dtype=np.float32)
    errs = np.empty((len(indices), n), dtype=np.float32)
    for s in range(0, len(indices), batch_size):
        idx = torch.as_tensor(indices[s : s + batch_size], dtype=torch.long, device=device)
        x, y = gather_windows(series, idx, tau)
        a, e = compute_causality(model, x, y, grad_target)
        mats[s : s + len(idx)] = a.cpu().numpy()
        errs[s : s + len(idx)] = e.cpu().numpy()
    return mats, errs


# --------------------------------------------------------------------------- sparsification
def sparsify(a: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Eq. 6: keep only the unidirectional part of the causal effect, then threshold.

    ``A~[i, j] = max(0, A[i, j] - A[j, i])`` for ``i != j``; ``A~[i, i] = A[i, i]``.
    Entries below ``threshold`` are set to zero. Works on (..., N, N) arrays.
    """
    if a.shape[-1] != a.shape[-2]:
        raise ValueError("last two dimensions must be square")
    out = np.maximum(a - np.swapaxes(a, -1, -2), 0.0)
    diag = np.diagonal(a, axis1=-2, axis2=-1)
    idx = np.arange(a.shape[-1])
    out[..., idx, idx] = diag
    if threshold > 0:
        out = np.where(out < threshold, 0.0, out)
    return out.astype(a.dtype, copy=False)
