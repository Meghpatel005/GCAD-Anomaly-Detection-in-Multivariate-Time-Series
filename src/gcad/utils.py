"""Small shared helpers: logging, seeding, device selection, JSON IO, hashing."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

LOG = logging.getLogger("gcad")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def source_fingerprint(*objs: Any) -> str:
    """Hash of source code of modules/functions; part of cache keys so code changes invalidate results."""
    digest = hashlib.sha256()
    for obj in objs:
        if isinstance(obj, ModuleType):
            digest.update(Path(inspect.getfile(obj)).read_bytes())
        else:  # function, method, or class: hash only its own source
            digest.update(inspect.getsource(obj).encode())
    return digest.hexdigest()[:16]


def stable_hash(obj: Any) -> str:
    """Short deterministic hash of a JSON-serialisable object (used as a cache key)."""
    blob = json.dumps(obj, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
