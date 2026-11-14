"""Dataset loading, preprocessing and sliding-window utilities.

Real benchmark files are *not* bundled (licences differ); see docs/DATASETS.md for the expected layout.
All loaders return a :class:`RawData`; :func:`prepare_data` then downsamples, splits and standardises.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import paper_results
from .config import DataConfig
from .synthetic import generate_synthetic

LOG = logging.getLogger("gcad.data")

_ENTITY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class DataError(ValueError):
    """Raised when a dataset is malformed."""


class DataNotFoundError(FileNotFoundError):
    """Raised when expected dataset files are missing."""


@dataclass
class RawData:
    train: np.ndarray  # (T_train, N) float32, anomaly-free
    test: np.ndarray  # (T_test, N) float32
    test_labels: np.ndarray  # (T_test,) int8
    channel_names: list[str]
    true_graph: np.ndarray | None = None  # only for synthetic data
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedData:
    train: np.ndarray  # standardised training split
    val: np.ndarray  # standardised validation split (chronologically after train)
    test: np.ndarray
    test_labels: np.ndarray
    channel_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    true_graph: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.train.shape[1]


# --------------------------------------------------------------------------- helpers
def _check_entity(entity: str) -> str:
    if not _ENTITY_RE.fullmatch(entity) or entity in {".", ".."}:
        raise DataError(f"Invalid entity name {entity!r}")
    return entity


def _require(path: Path) -> Path:
    if not path.is_file():
        raise DataNotFoundError(f"Missing dataset file: {path}. See docs/DATASETS.md for the expected layout.")
    return path


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.ffill().bfill().fillna(0.0)


def _downsample(x: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 1:
        return x, labels
    n = (len(x) // k) * k
    if n == 0:
        raise DataError(f"downsample={k} is larger than the series length {len(x)}")
    x = x[:n].reshape(-1, k, x.shape[1]).mean(axis=1).astype(np.float32)
    labels = labels[:n].reshape(-1, k).max(axis=1).astype(np.int8)
    return x, labels


# --------------------------------------------------------------------------- loaders
def _load_synthetic(cfg: DataConfig) -> RawData:
    params = dict(cfg.synthetic)
    syn = generate_synthetic(**params)
    names = [f"x{i}" for i in range(syn.train.shape[1])]
    return RawData(syn.train, syn.test, syn.labels, names, syn.true_graph, syn.meta)


def _load_smd(cfg: DataConfig) -> RawData:
    entity = _check_entity(cfg.entity or "machine-1-1")
    base = Path(cfg.root) / "SMD"
    train = np.loadtxt(_require(base / "train" / f"{entity}.txt"), delimiter=",", dtype=np.float32)
    test = np.loadtxt(_require(base / "test" / f"{entity}.txt"), delimiter=",", dtype=np.float32)
    labels = np.loadtxt(_require(base / "test_label" / f"{entity}.txt"), delimiter=",").astype(np.int8)
    names = [f"c{i}" for i in range(train.shape[1])]
    return RawData(train, test, labels.reshape(-1), names, meta={"entity": entity})


def _load_nasa(cfg: DataConfig, spacecraft: str) -> RawData:
    """Load one telemetry channel. Entities are never concatenated: windows would straddle channel boundaries."""
    base = Path(cfg.root) / "NASA"
    meta = pd.read_csv(_require(base / "labeled_anomalies.csv"))
    meta = meta[meta["spacecraft"] == spacecraft]
    if meta.empty:
        raise DataError(f"No {spacecraft} entries in labeled_anomalies.csv")
    available = list(meta["chan_id"])
    entity = _check_entity(cfg.entity) if cfg.entity else available[0]
    rows = meta[meta["chan_id"] == entity]
    if rows.empty:
        raise DataError(f"Entity {entity!r} is not a {spacecraft} channel; available: {available[:5]}...")
    train = np.load(_require(base / "train" / f"{entity}.npy"), allow_pickle=False).astype(np.float32)
    test = np.load(_require(base / "test" / f"{entity}.npy"), allow_pickle=False).astype(np.float32)
    labels = np.zeros(len(test), dtype=np.int8)
    for seqs in rows["anomaly_sequences"]:
        for start, end in ast.literal_eval(seqs):  # literal_eval: no code execution
            if not (0 <= start <= end < len(test)):
                raise DataError(f"Anomaly range [{start}, {end}] out of bounds for {entity} (len {len(test)})")
            labels[start : end + 1] = 1
    names = [f"c{i}" for i in range(train.shape[1])]
    return RawData(train, test, labels, names, meta={"entity": entity})


def _load_swat(cfg: DataConfig) -> RawData:
    base = Path(cfg.root) / "SWaT"
    train_df = pd.read_csv(_require(base / (cfg.train_file or "SWaT_Dataset_Normal_v1.csv")), skipinitialspace=True)
    test_df = pd.read_csv(_require(base / (cfg.test_file or "SWaT_Dataset_Attack_v0.csv")), skipinitialspace=True)
    for df in (train_df, test_df):
        df.columns = [str(c).strip() for c in df.columns]
    if "Normal/Attack" not in test_df.columns:
        raise DataError("SWaT attack file must contain a 'Normal/Attack' column")
    labels = (test_df["Normal/Attack"].astype(str).str.strip().str.lower() != "normal").to_numpy(np.int8)
    drop = {"Normal/Attack"} | {c for c in train_df.columns if "timestamp" in c.lower()}
    feats = [c for c in train_df.columns if c not in drop]
    if set(feats) - set(test_df.columns):
        raise DataError("SWaT train and attack files do not share the same sensor columns")
    train = _clean_frame(train_df[feats]).to_numpy(np.float32)
    test = _clean_frame(test_df[feats]).to_numpy(np.float32)
    return RawData(train, test, labels, feats)


def _load_psm(cfg: DataConfig) -> RawData:
    base = Path(cfg.root) / "PSM"
    train_df = pd.read_csv(_require(base / (cfg.train_file or "train.csv")))
    test_df = pd.read_csv(_require(base / (cfg.test_file or "test.csv")))
    lab_df = pd.read_csv(_require(base / (cfg.label_file or "test_label.csv")))
    if "label" not in lab_df.columns:
        raise DataError("PSM label file must contain a 'label' column")
    feats = [c for c in train_df.columns if "timestamp" not in c.lower()]
    train = _clean_frame(train_df[feats]).to_numpy(np.float32)
    test = _clean_frame(test_df[feats]).to_numpy(np.float32)
    labels = lab_df["label"].to_numpy(np.int8)
    return RawData(train, test, labels, feats)


def _load_npz(cfg: DataConfig) -> RawData:
    path = _require(Path(cfg.root) / (cfg.train_file or "dataset.npz"))
    with np.load(path, allow_pickle=False) as z:
        missing = {"train", "test", "labels"} - set(z.files)
        if missing:
            raise DataError(f"npz file is missing arrays: {sorted(missing)}")
        train, test, labels = z["train"].astype(np.float32), z["test"].astype(np.float32), z["labels"].astype(np.int8)
    names = [f"c{i}" for i in range(train.shape[1])]
    return RawData(train, test, labels, names)


_LOADERS: dict[str, Callable[[DataConfig], RawData]] = {
    "synthetic": _load_synthetic,
    "smd": _load_smd,
    "msl": lambda c: _load_nasa(c, "MSL"),
    "smap": lambda c: _load_nasa(c, "SMAP"),
    "swat": _load_swat,
    "psm": _load_psm,
    "npz": _load_npz,
}


def load_raw(cfg: DataConfig) -> RawData:
    raw = _LOADERS[cfg.name](cfg)
    if raw.train.ndim != 2 or raw.test.ndim != 2:
        raise DataError("train/test must be 2-D arrays of shape (time, channels)")
    if raw.train.shape[1] != raw.test.shape[1]:
        raise DataError(f"channel mismatch: train {raw.train.shape[1]} vs test {raw.test.shape[1]}")
    if len(raw.test_labels) != len(raw.test):
        raise DataError(f"label length {len(raw.test_labels)} != test length {len(raw.test)}")
    return raw


# --------------------------------------------------------------------------- preprocessing
def _warn_if_unlike_paper(name: str, channels: int, n_train: int, n_test: int, ratio: float) -> None:
    """Flag preprocessing that does not reproduce the dataset statistics in the paper's Table 1."""
    ref = paper_results.TABLE1.get(name)
    if ref is None:
        return
    ref_channels, ref_train, ref_test, ref_ratio = ref
    if (channels, n_train, n_test) != (ref_channels, ref_train, ref_test):
        LOG.warning(
            "%s: channels/train/test = %d/%d/%d but the paper's Table 1 lists %d/%d/%d "
            "(expected for single MSL/SMAP entities or other SMD machines; otherwise check preprocessing)",
            name,
            channels,
            n_train,
            n_test,
            ref_channels,
            ref_train,
            ref_test,
        )
    elif abs(100 * ratio - ref_ratio) > 0.05:
        LOG.warning("%s: anomaly ratio %.2f%% differs from the paper's %.2f%%", name, 100 * ratio, ref_ratio)


def prepare_data(cfg: DataConfig) -> PreparedData:
    raw = load_raw(cfg)
    if cfg.skip_head >= len(raw.train):
        raise DataError(f"data.skip_head={cfg.skip_head} removes the whole training series ({len(raw.train)} rows)")
    raw_train = raw.train[cfg.skip_head :]
    train, _ = _downsample(raw_train, np.zeros(len(raw_train), dtype=np.int8), cfg.downsample)
    test, labels = _downsample(raw.test, raw.test_labels, cfg.downsample)
    train = np.nan_to_num(train.astype(np.float32))
    test = np.nan_to_num(test.astype(np.float32))

    n_val = round(len(train) * cfg.val_fraction)
    split = len(train) - n_val
    train_part, val_part = train[:split], train[split:]

    mean = train_part.mean(axis=0)
    std = train_part.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # constant channels: avoid division by zero

    def scale(a: np.ndarray) -> np.ndarray:
        return ((a - mean) / std).astype(np.float32)

    LOG.info(
        "dataset=%s channels=%d train=%d val=%d test=%d anomalies=%.2f%%",
        cfg.name,
        train.shape[1],
        len(train_part),
        len(val_part),
        len(test),
        100 * float(labels.mean()),
    )
    _warn_if_unlike_paper(cfg.name, train.shape[1], len(train), len(test), float(labels.mean()))
    return PreparedData(
        train=scale(train_part),
        val=scale(val_part),
        test=scale(test),
        test_labels=labels.astype(np.int8),
        channel_names=raw.channel_names,
        mean=mean,
        std=std,
        true_graph=raw.true_graph,
        meta=raw.meta,
    )


# --------------------------------------------------------------------------- windows
def num_windows(length: int, tau: int) -> int:
    """Number of (window, target) pairs in a series of the given length."""
    return max(0, length - tau)


def gather_windows(x: torch.Tensor, idx: torch.Tensor, tau: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows ``X_{t-1} = x[i:i+tau]`` (shape (B, N, tau)) and targets ``x[i+tau]`` (shape (B, N))."""
    offs = idx[:, None] + torch.arange(tau, device=idx.device)[None, :]
    windows = x[offs].transpose(1, 2).contiguous()
    return windows, x[idx + tau]


def window_labels(labels: np.ndarray, tau: int, idx: np.ndarray, mode: str = "target") -> np.ndarray:
    """Label of every window starting at ``idx`` (window covers ``[i, i+tau)``, target at ``i+tau``)."""
    if mode == "target":
        return labels[idx + tau].astype(np.int8)
    csum = np.concatenate([[0], np.cumsum(labels, dtype=np.int64)])
    return ((csum[idx + tau + 1] - csum[idx]) > 0).astype(np.int8)
