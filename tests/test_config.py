from __future__ import annotations

import pytest
import yaml

from gcad.config import ConfigError, apply_overrides, config_from_dict, load_config


def test_defaults_validate():
    assert config_from_dict({}).model.window == 5


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown keys"):
        config_from_dict({"model": {"windw": 3}})


def test_unknown_section_rejected():
    with pytest.raises(ConfigError, match="Unknown config sections"):
        config_from_dict({"modle": {}})


@pytest.mark.parametrize(
    "raw",
    [
        {"model": {"window": 0}},
        {"scoring": {"eps": 0}},
        {"causality": {"normal_sample_p": 0}},
        {"data": {"name": "nope"}},
        {"data": {"val_fraction": 0.9}},
        {"run": {"seeds": []}},
    ],
)
def test_invalid_values_rejected(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_overrides_and_load(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"model": {"window": 3}}))
    cfg = load_config(p, ["scoring.beta=0.5", "run.seeds=[1,2]"])
    assert cfg.model.window == 3 and cfg.scoring.beta == 0.5 and cfg.run.seeds == [1, 2]


def test_bad_override_and_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        apply_overrides({}, ["nokeyvalue"])
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_nested_override_works():
    raw = {"data": {"synthetic": {"seed": 0, "n_channels": 5}}}
    result = apply_overrides(raw, ["data.synthetic.seed=3"])
    assert result["data"]["synthetic"]["seed"] == 3
    assert result["data"]["synthetic"]["n_channels"] == 5  # untouched


def test_nested_override_into_non_mapping_raises():
    with pytest.raises(ConfigError):
        apply_overrides({"model": {"window": 5}}, ["model.window.x=1"])


def test_nested_override_sets_deep_key():
    raw = {"data": {"synthetic": {"seed": 0, "n_channels": 5}}}
    result = apply_overrides(raw, ["data.synthetic.seed=3"])
    assert result["data"]["synthetic"]["seed"] == 3
    assert result["data"]["synthetic"]["n_channels"] == 5


def test_nested_override_into_non_mapping_raises():
    with pytest.raises(ConfigError):
        apply_overrides({"model": {"window": 5}}, ["model.window.x=1"])
