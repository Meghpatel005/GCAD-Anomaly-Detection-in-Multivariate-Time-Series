"""Typed, validated configuration loaded from YAML.

Unknown keys are rejected (typos must not be silently ignored) and every value is range-checked.
Individual values can be overridden from the CLI with ``--set section.key=value``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DATASETS = {"synthetic", "swat", "smd", "msl", "smap", "psm", "npz"}


class ConfigError(ValueError):
    """Raised for invalid configuration."""


@dataclass
class DataConfig:
    name: str = "synthetic"  # synthetic | swat | smd | msl | smap | psm | npz
    root: str = "data"
    entity: str | None = None  # SMD machine id, or MSL/SMAP channel id
    downsample: int = 1  # block-mean downsampling factor (SWaT is usually 10)
    skip_head: int = 0  # raw training rows dropped before downsampling (SWaT: 21600 start-up rows)
    val_fraction: float = 0.2  # chronological tail of the normal data used for validation
    label_mode: str = "target"  # window label = label of predicted step ("target") or any step in window
    train_file: str | None = None  # optional file-name overrides (see docs/DATASETS.md)
    test_file: str | None = None
    label_file: str | None = None
    synthetic: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    window: int = 5  # tau: maximum time lag considered for Granger causality
    hidden_dim: int = 64
    n_layers: int = 2  # L stacked mixer layers
    dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    grad_clip: float = 1.0
    device: str = "auto"


@dataclass
class CausalityConfig:
    grad_target: str = "loss"  # "loss": dL_j/dx (paper, Eq. 4/5) | "prediction": dyhat_j/dx
    normal_sample_p: float = 0.1  # Bernoulli rate used to sample training windows (Eq. 7)
    max_normal_windows: int = 2000  # upper bound on sampled windows
    normal_source: str = "train"  # train | val | both
    batch_size: int = 64
    test_stride: int = 1  # evaluate every k-th test window (1 = all)


@dataclass
class ScoringConfig:
    threshold: float = 0.01  # sparsity threshold h
    beta: float = 1.0  # weight of the time-pattern deviation (Eq. 12)
    eps: float = 1e-3  # small constant in Eq. 10 / 11
    sparsify: bool = True


@dataclass
class RunConfig:
    output_dir: str = "runs"
    seeds: list[int] = field(default_factory=lambda: [0])
    log_level: str = "INFO"
    deterministic: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    causality: CausalityConfig = field(default_factory=CausalityConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self) -> Config:
        d, m, t, c, s, r = self.data, self.model, self.train, self.causality, self.scoring, self.run
        _check(d.name in DATASETS, f"data.name must be one of {sorted(DATASETS)}, got {d.name!r}")
        _check(d.downsample >= 1, "data.downsample must be >= 1")
        _check(d.skip_head >= 0, "data.skip_head must be >= 0")
        _check(0.0 < d.val_fraction < 0.5, "data.val_fraction must be in (0, 0.5)")
        _check(d.label_mode in {"target", "any"}, "data.label_mode must be 'target' or 'any'")
        _check(m.window >= 1, "model.window must be >= 1")
        _check(m.hidden_dim >= 1 and m.n_layers >= 1, "model.hidden_dim / n_layers must be >= 1")
        _check(0.0 <= m.dropout < 1.0, "model.dropout must be in [0, 1)")
        _check(t.epochs >= 1 and t.batch_size >= 1, "train.epochs / batch_size must be >= 1")
        _check(t.lr > 0 and t.weight_decay >= 0 and t.patience >= 1, "invalid train lr/weight_decay/patience")
        _check(c.grad_target in {"loss", "prediction"}, "causality.grad_target must be 'loss' or 'prediction'")
        _check(0.0 < c.normal_sample_p <= 1.0, "causality.normal_sample_p must be in (0, 1]")
        _check(c.max_normal_windows >= 1 and c.batch_size >= 1 and c.test_stride >= 1, "invalid causality sizes")
        _check(c.normal_source in {"train", "val", "both"}, "causality.normal_source must be train|val|both")
        _check(s.threshold >= 0 and s.beta >= 0 and s.eps > 0, "scoring.threshold/beta must be >= 0 and eps > 0")
        _check(len(r.seeds) >= 1, "run.seeds must not be empty")
        return self


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "causality": CausalityConfig,
    "scoring": ScoringConfig,
    "run": RunConfig,
}


def config_from_dict(raw: dict[str, Any] | None) -> Config:
    raw = copy.deepcopy(raw or {})
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"Unknown config sections: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        if not isinstance(section, dict):
            raise ConfigError(f"Config section '{name}' must be a mapping")
        valid = {f.name for f in fields(cls)}
        bad = set(section) - valid
        if bad:
            raise ConfigError(f"Unknown keys in '{name}': {sorted(bad)}; valid keys: {sorted(valid)}")
        kwargs[name] = cls(**section)
    return Config(**kwargs).validate()


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (value parsed as YAML) to a raw config dict."""
    raw = copy.deepcopy(raw or {})
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ConfigError(f"Override must look like section.key=value, got {item!r}")
        key, value = item.split("=", 1)
        if "." not in key:
            raise ConfigError(f"Override must look like section.key=value, got {item!r}")
        section, *path, leaf = key.split(".")
        node = raw.setdefault(section, {})
        for part in path:
            node = node.setdefault(part, {}) if isinstance(node, dict) else node
        if not isinstance(node, dict):
            raise ConfigError(f"Cannot override into non-mapping key {key.rsplit('.', 1)[0]!r}")
        node[leaf] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a mapping")
    return config_from_dict(apply_overrides(raw, overrides))


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return asdict(cfg)
