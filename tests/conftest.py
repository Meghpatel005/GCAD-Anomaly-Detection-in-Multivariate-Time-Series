from __future__ import annotations

import pytest

from gcad.config import Config, config_from_dict


@pytest.fixture()
def tiny_cfg(tmp_path) -> Config:
    """A very small synthetic configuration that trains in a couple of seconds."""
    return config_from_dict(
        {
            "data": {
                "name": "synthetic",
                "synthetic": {
                    "n_channels": 5,
                    "n_roots": 2,
                    "n_train": 600,
                    "n_test": 600,
                    "anomaly_ratio": 0.15,
                    "segment_min": 20,
                    "segment_max": 50,
                    "seed": 1,
                },
            },
            "model": {"window": 4, "hidden_dim": 16, "n_layers": 1},
            "train": {"epochs": 3, "batch_size": 64, "patience": 3, "device": "cpu"},
            "causality": {"normal_sample_p": 0.3, "max_normal_windows": 60, "batch_size": 32},
            "run": {"output_dir": str(tmp_path / "runs"), "seeds": [0]},
        }
    )
