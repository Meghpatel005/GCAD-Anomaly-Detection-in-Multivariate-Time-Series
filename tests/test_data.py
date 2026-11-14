from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gcad.config import DataConfig
from gcad.data import (
    DataError,
    DataNotFoundError,
    _warn_if_unlike_paper,
    gather_windows,
    load_raw,
    num_windows,
    prepare_data,
    window_labels,
)


def test_gather_windows_alignment():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    w, y = gather_windows(x, torch.tensor([0, 3]), tau=3)
    assert w.shape == (2, 2, 3)
    assert torch.equal(w[0, 0], torch.tensor([0.0, 2.0, 4.0]))  # channel 0, steps 0..2
    assert torch.equal(y[0], x[3]) and torch.equal(y[1], x[6])
    assert num_windows(10, 3) == 7 and num_windows(2, 3) == 0


def test_window_labels_modes():
    labels = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    idx = np.arange(num_windows(len(labels), 2))  # windows [i, i+2), target i+2
    assert window_labels(labels, 2, idx, "target").tolist() == [0, 1, 0, 0, 0]
    assert window_labels(labels, 2, idx, "any").tolist() == [0, 1, 1, 1, 0]


def test_prepare_data_scales_on_train_only_and_handles_constant_channel(tmp_path):
    rng = np.random.default_rng(0)
    train = rng.normal(5, 2, size=(100, 3)).astype(np.float32)
    train[:, 2] = 7.0  # constant channel
    test = rng.normal(5, 2, size=(50, 3)).astype(np.float32)
    np.savez(tmp_path / "dataset.npz", train=train, test=test, labels=np.zeros(50))
    data = prepare_data(DataConfig(name="npz", root=str(tmp_path)))
    assert len(data.train) == 80 and len(data.val) == 20  # chronological 80/20 split
    assert abs(data.train[:, 0].mean()) < 1e-5 and abs(data.train[:, 0].std() - 1) < 1e-5
    assert np.isfinite(data.test).all() and data.std[2] == 1.0


def test_skip_head_drops_startup_rows(tmp_path):
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / "dataset.npz", train=rng.normal(size=(100, 2)), test=rng.normal(size=(30, 2)), labels=np.zeros(30)
    )
    data = prepare_data(DataConfig(name="npz", root=str(tmp_path), skip_head=20, downsample=2))
    assert len(data.train) + len(data.val) == 40
    with pytest.raises(DataError, match="skip_head"):
        prepare_data(DataConfig(name="npz", root=str(tmp_path), skip_head=100))


def test_deviation_from_paper_table_is_logged(caplog):
    with caplog.at_level("WARNING", logger="gcad.data"):
        _warn_if_unlike_paper("swat", 51, 47_520, 44_991, 0.1220)  # exactly the paper's statistics
        _warn_if_unlike_paper("unknown-dataset", 3, 10, 10, 0.1)
    assert caplog.text == ""
    with caplog.at_level("WARNING", logger="gcad.data"):
        _warn_if_unlike_paper("swat", 51, 49_680, 44_991, 0.1220)  # start-up rows not skipped
    assert "47520" in caplog.text and "49680" in caplog.text


def test_missing_files_raise_helpful_error(tmp_path):
    with pytest.raises(DataNotFoundError, match=r"DATASETS\.md"):
        load_raw(DataConfig(name="smd", root=str(tmp_path)))


def test_entity_path_traversal_rejected(tmp_path):
    with pytest.raises(DataError):
        load_raw(DataConfig(name="smd", root=str(tmp_path), entity="../../etc/passwd"))


def test_smd_loader(tmp_path):
    for sub, arr in (("train", np.random.rand(30, 4)), ("test", np.random.rand(20, 4))):
        (tmp_path / "SMD" / sub).mkdir(parents=True, exist_ok=True)
        np.savetxt(tmp_path / "SMD" / sub / "machine-1-1.txt", arr, delimiter=",")
    (tmp_path / "SMD" / "test_label").mkdir()
    np.savetxt(tmp_path / "SMD" / "test_label" / "machine-1-1.txt", np.r_[np.zeros(15), np.ones(5)], fmt="%d")
    raw = load_raw(DataConfig(name="smd", root=str(tmp_path)))
    assert raw.train.shape == (30, 4) and raw.test_labels.sum() == 5


def test_nasa_loader_and_label_ranges(tmp_path):
    base = tmp_path / "NASA"
    (base / "train").mkdir(parents=True)
    (base / "test").mkdir()
    np.save(base / "train" / "P-1.npy", np.random.rand(40, 3))
    np.save(base / "test" / "P-1.npy", np.random.rand(30, 3))
    pd.DataFrame({"chan_id": ["P-1"], "spacecraft": ["SMAP"], "anomaly_sequences": ["[[5, 9], [20, 22]]"]}).to_csv(
        base / "labeled_anomalies.csv", index=False
    )
    raw = load_raw(DataConfig(name="smap", root=str(tmp_path), entity="P-1"))
    assert raw.test_labels.sum() == 5 + 3 and raw.test_labels[5] == 1 and raw.test_labels[10] == 0
    pd.DataFrame({"chan_id": ["P-1"], "spacecraft": ["SMAP"], "anomaly_sequences": ["[[5, 99]]"]}).to_csv(
        base / "labeled_anomalies.csv", index=False
    )
    with pytest.raises(DataError, match="out of bounds"):
        load_raw(DataConfig(name="smap", root=str(tmp_path), entity="P-1"))


def test_nasa_entities_are_not_concatenated(tmp_path):
    base = tmp_path / "NASA"
    (base / "train").mkdir(parents=True)
    (base / "test").mkdir()
    for ent in ("P-1", "P-2"):
        np.save(base / "train" / f"{ent}.npy", np.random.rand(40, 3))
        np.save(base / "test" / f"{ent}.npy", np.random.rand(30, 3))
    pd.DataFrame({"chan_id": ["P-1", "P-2"], "spacecraft": ["SMAP"] * 2, "anomaly_sequences": ["[[1, 2]]"] * 2}).to_csv(
        base / "labeled_anomalies.csv", index=False
    )
    with pytest.raises(DataError, match="not a SMAP channel"):
        load_raw(DataConfig(name="smap", root=str(tmp_path), entity="all"))
    assert load_raw(DataConfig(name="smap", root=str(tmp_path), entity="P-2")).train.shape == (40, 3)


def test_swat_and_psm_loaders_and_downsample(tmp_path):
    swat = tmp_path / "SWaT"
    swat.mkdir()
    cols = {" Timestamp": range(40), "FIT101": np.arange(40.0), "AIT201": np.arange(40.0) * 2}
    pd.DataFrame({**cols, "Normal/Attack": ["Normal"] * 40}).to_csv(swat / "SWaT_Dataset_Normal_v1.csv", index=False)
    pd.DataFrame({**cols, "Normal/Attack": ["Normal"] * 30 + ["Attack"] * 10}).to_csv(
        swat / "SWaT_Dataset_Attack_v0.csv", index=False
    )
    data = prepare_data(DataConfig(name="swat", root=str(tmp_path), downsample=10))
    assert data.channel_names == ["FIT101", "AIT201"]
    assert len(data.train) + len(data.val) == 4 and data.test_labels.tolist() == [0, 0, 0, 1]

    psm = tmp_path / "PSM"
    psm.mkdir()
    pd.DataFrame({"timestamp_(min)": range(20), "f0": np.random.rand(20), "f1": np.random.rand(20)}).to_csv(
        psm / "train.csv", index=False
    )
    pd.DataFrame({"timestamp_(min)": range(10), "f0": np.random.rand(10), "f1": np.random.rand(10)}).to_csv(
        psm / "test.csv", index=False
    )
    pd.DataFrame({"timestamp_(min)": range(10), "label": [0] * 7 + [1] * 3}).to_csv(psm / "test_label.csv", index=False)
    raw = load_raw(DataConfig(name="psm", root=str(tmp_path)))
    assert raw.train.shape == (20, 2) and raw.test_labels.sum() == 3
