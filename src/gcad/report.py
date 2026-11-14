"""Plotting helpers (headless): deviation heat map (paper Fig. 3) and parameter sweeps (Fig. 2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_deviation_matrix(
    d: np.ndarray, names: list[str], path: str | Path, top_k: int = 3
) -> list[tuple[str, str, float]]:
    """Save a heat map of ``D`` and return the ``top_k`` (cause, effect, deviation) pairs."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(d, cmap="YlOrBr", aspect="auto")
    ax.set_xlabel("effect series j")
    ax.set_ylabel("cause series i")
    ax.set_title("Causal pattern deviation |A~_test - A_norm|")
    if len(names) <= 20:
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    order = np.argsort(d, axis=None)[::-1][:top_k]
    return [(names[i], names[j], float(d[i, j])) for i, j in (np.unravel_index(o, d.shape) for o in order)]


def plot_sweep(df: pd.DataFrame, param: str, path: str | Path) -> None:
    """AUROC / AUPRC of GCAD vs. a swept parameter (mean +- std over seeds)."""
    sub = df[df["variant"] == "gcad"]
    grouped = sub.groupby("value")[["auroc", "auprc"]].agg(["mean", "std"]).sort_index()
    xs = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for metric, marker in (("auroc", "o"), ("auprc", "^")):
        ax.errorbar(
            xs,
            grouped[(metric, "mean")],
            yerr=grouped[(metric, "std")].fillna(0.0),
            marker=marker,
            capsize=3,
            label=metric.upper(),
        )
    ax.set_xticks(xs, [str(v) for v in grouped.index])
    ax.set_xlabel(param)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
