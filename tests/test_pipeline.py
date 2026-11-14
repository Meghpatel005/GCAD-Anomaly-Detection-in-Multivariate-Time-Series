from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from gcad import pipeline
from gcad.cli import main
from gcad.config import config_from_dict, config_to_dict, load_config
from gcad.pipeline import VARIANTS, Experiment, run_experiment, sweep, synthetic_benchmark
from gcad.utils import source_fingerprint

CACHE_FILES = ("model.pt", "a_normal.npy", "a_test.npy", "sq_err.npy", "a_static.npy", "normal_sq_err.npy")


def _mtimes(exp: Experiment) -> dict[str, int]:
    return {name: (exp.dir / name).stat().st_mtime_ns for name in CACHE_FILES}


def test_end_to_end_and_caching(tiny_cfg):
    exp = Experiment(tiny_cfg, seed=0)
    res = exp.evaluate()
    assert set(res["metrics"]) == set(VARIANTS)
    for m in res["metrics"].values():
        assert 0.0 <= m["auroc"] <= 1.0 and 0.0 <= m["auprc"] <= 1.0
    assert "causal_recovery" in res
    assert (exp.dir / "metrics.json").is_file()

    before = _mtimes(exp)
    res2 = Experiment(tiny_cfg, seed=0).evaluate()
    assert _mtimes(exp) == before  # cached weights and graphs were reused, not recomputed
    assert res2["metrics"]["gcad"] == res["metrics"]["gcad"]


def test_same_seed_is_reproducible(tiny_cfg, tmp_path):
    other = config_from_dict({**config_to_dict(tiny_cfg), "run": {"output_dir": str(tmp_path / "again")}})
    a = Experiment(tiny_cfg, seed=0).evaluate(save=False)["metrics"]
    b = Experiment(other, seed=0).evaluate(save=False)["metrics"]
    assert a == b


class TestCacheKeys:
    def test_config_changes_invalidate_model_key(self, tiny_cfg):
        key = Experiment(tiny_cfg, seed=0).model_key
        raw = config_to_dict(tiny_cfg)
        raw["model"]["hidden_dim"] = 8
        assert Experiment(config_from_dict(raw), seed=0).model_key != key
        assert Experiment(tiny_cfg, seed=1).model_key != key

    def test_label_mode_does_not_retrain_but_does_re_extract(self, tiny_cfg):
        raw = config_to_dict(tiny_cfg)
        raw["data"]["label_mode"] = "any"
        other = Experiment(config_from_dict(raw), seed=0)
        base = Experiment(tiny_cfg, seed=0)
        assert other.model_key == base.model_key
        assert other.extract_key != base.extract_key

    def test_code_change_invalidates_cache(self, tiny_cfg, monkeypatch):
        exp = Experiment(tiny_cfg, seed=0)
        key = exp.model_key
        monkeypatch.setattr(pipeline, "_training_code_version", lambda: "edited-source")
        assert exp.model_key != key

    def test_narrow_hashing_ignores_summary_markdown(self, tiny_cfg, monkeypatch):
        exp = Experiment(tiny_cfg, seed=0)
        model_k = exp.model_key
        extract_k = exp.extract_key
        monkeypatch.setattr(pipeline, "summary_markdown", lambda *a, **kw: "patched")
        assert exp.model_key == model_k
        assert exp.extract_key == extract_k


def test_source_fingerprint_discriminates_functions():
    def func_a() -> int:
        return 1

    def func_b() -> int:
        return 2

    assert source_fingerprint(func_a) != source_fingerprint(func_b)
    assert source_fingerprint(func_a) == source_fingerprint(func_a)


def test_artifact_shapes_and_alignment(tiny_cfg):
    art = Experiment(tiny_cfg, seed=0).extract()
    n = tiny_cfg.data.synthetic["n_channels"]
    assert art.a_test.shape == (len(art.labels), n, n)
    assert art.sq_err.shape == (len(art.labels), n)
    assert art.a_normal.shape[1:] == (n, n) and len(art.a_normal) <= tiny_cfg.causality.max_normal_windows
    assert np.isfinite(art.a_test).all() and (art.a_test >= 0).all()
    # F1: new artifact fields
    assert art.a_static.shape == (n, n)
    assert art.normal_sq_err.shape == (len(art.a_normal), n)


def test_old_cache_without_a_static_is_recomputed(tiny_cfg):
    """Old cache dirs missing a_static.npy must trigger recomputation, not crash."""
    exp = Experiment(tiny_cfg, seed=0)
    exp.extract()  # creates all files
    (exp.dir / "a_static.npy").unlink()  # simulate an old-schema cache
    art2 = exp.extract(force=False)  # key mismatch (file missing) → recompute
    assert art2.a_static.shape[0] == tiny_cfg.data.synthetic["n_channels"]


def test_run_experiment_writes_summary(tiny_cfg):
    summary = run_experiment(tiny_cfg, seeds=[0, 1])
    assert list(summary["variant"]) == list(VARIANTS)
    out = next(Path(tiny_cfg.run.output_dir).iterdir())
    assert (out / "summary.csv").is_file() and "AUROC" in (out / "summary.md").read_text()


def test_threshold_sweep_reuses_cached_graphs(tiny_cfg):
    exp = Experiment(tiny_cfg, seed=0)
    exp.evaluate()
    before = _mtimes(exp)
    df = sweep(tiny_cfg, "scoring.threshold", [0.0, 0.01, 1e6], seeds=[0])
    assert _mtimes(exp) == before
    gcad = df[df.variant == "gcad"].set_index("value")
    assert list(gcad.index) == [0.0, 0.01, 1e6]
    # An enormous threshold removes every edge: all scores are 0, so the ranking carries no information.
    assert gcad.loc[1e6, "auroc"] == pytest.approx(0.5)


def test_sweep_rejects_unsupported_parameter(tiny_cfg):
    with pytest.raises(ValueError, match="sweep supports"):
        sweep(tiny_cfg, "train.lr", [0.1])


# ------------------------------------------------------------------ F8: sweep skips unused variants
def test_sweep_skips_pca(tiny_cfg, monkeypatch):
    """sweep() must not call pca_reconstruction_scores (saves time on large datasets)."""
    exp = Experiment(tiny_cfg, seed=0)
    exp.evaluate()  # populate cache

    def boom(*args, **kwargs):
        raise AssertionError("pca_reconstruction_scores must not be called in sweep")

    monkeypatch.setattr(pipeline, "pca_reconstruction_scores", boom)
    sweep(tiny_cfg, "scoring.threshold", [0.0, 0.01], seeds=[0])  # must not raise


def test_evaluate_returns_all_variants_by_default(tiny_cfg):
    res = Experiment(tiny_cfg, seed=0).evaluate(save=False)
    assert set(res["metrics"]) == set(VARIANTS)


# ------------------------------------------------------------------ F2 / F2b: benchmark + run_name
def test_run_name_differs_for_different_synthetic_seeds(tiny_cfg):
    from gcad.pipeline import run_name

    raw1 = config_to_dict(tiny_cfg)
    raw2 = config_to_dict(tiny_cfg)
    raw2["data"]["synthetic"]["seed"] = raw1["data"]["synthetic"]["seed"] + 999
    cfg1 = config_from_dict(raw1)
    cfg2 = config_from_dict(raw2)
    assert run_name(cfg1) != run_name(cfg2)


def test_run_name_same_for_identical_synthetic_configs(tiny_cfg):
    from gcad.pipeline import run_name

    raw = config_to_dict(tiny_cfg)
    cfg2 = config_from_dict(raw)
    assert run_name(tiny_cfg) == run_name(cfg2)


def test_synthetic_benchmark(tiny_cfg, tmp_path):
    df = synthetic_benchmark(tiny_cfg, dataset_seeds=[0, 1], model_seeds=[0], out_dir=str(tmp_path / "bench"))
    assert len(df) == 2 * len(VARIANTS)
    bench_md = (tmp_path / "bench" / "bench.md").read_text()
    assert "wins" in bench_md
    assert set(df.columns) >= {"dataset_seed", "model_seed", "variant", "auroc", "auprc"}


def test_sign_test_helper():
    """8 wins of 8 -> p = 2/256 = 0.0078125 (two-sided sign test)."""
    from scipy.stats import binomtest

    result = binomtest(8, 8, 0.5)
    assert abs(result.pvalue - 0.0078125) < 1e-7


def test_case_study(tiny_cfg):
    res = Experiment(tiny_cfg, seed=0).case_study()
    n = tiny_cfg.data.synthetic["n_channels"]
    assert res["deviation"].shape == (n, n) and np.isfinite(res["score"])


@pytest.mark.slow
def test_causal_score_detects_decoupling_anomalies(tmp_path):
    """Regression guard for the whole chain (sign errors, mis-aligned windows, broken scoring)."""
    cfg = load_config(
        Path(__file__).parents[1] / "configs" / "synthetic.yaml",
        [f"run.output_dir={tmp_path}", "train.device=cpu", "train.epochs=25"],
    )
    res = Experiment(cfg, seed=0).evaluate(save=False)
    assert res["metrics"]["gcad"]["auroc"] > 0.75
    assert res["metrics"]["gcad"]["auprc"] > 2 * res["anomaly_ratio"]
    assert res["causal_recovery"]["raw"] > 0.55  # chance level is 0.5


class TestCli:
    @pytest.fixture()
    def cfg_path(self, tiny_cfg, tmp_path) -> Path:
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(config_to_dict(tiny_cfg)))
        return path

    def test_run(self, cfg_path, capsys):
        assert main(["run", "--config", str(cfg_path), "--seeds", "0"]) == 0
        assert "Results: synthetic" in capsys.readouterr().out

    def test_stage_commands(self, cfg_path, tiny_cfg):
        base = ["--config", str(cfg_path), "--seeds", "0"]
        run_dir = Path(tiny_cfg.run.output_dir) / pipeline.run_name(tiny_cfg)
        assert main(["train", *base]) == 0
        assert (run_dir / "seed0" / "model.pt").is_file()
        assert main(["extract", *base]) == 0
        assert (run_dir / "seed0" / "a_test.npy").is_file()
        assert main(["evaluate", *base]) == 0
        assert main(["case-study", *base]) == 0
        assert (run_dir / "case_study.png").stat().st_size > 0
        assert main(["sweep", *base, "--param", "scoring.eps", "--values", "0.001", "0.1"]) == 0
        assert (run_dir / "sweep_scoring_eps.png").stat().st_size > 0

    def test_errors_return_exit_code_2(self, cfg_path, tmp_path):
        assert main(["run", "--config", str(tmp_path / "nope.yaml")]) == 2
        overrides = ["--set", "data.name=smd", "--set", f"data.root={tmp_path}"]
        assert main(["run", "--config", str(cfg_path), *overrides]) == 2

    def test_too_short_series_is_a_clean_error(self, cfg_path):
        overrides = ["--set", "model.window=700"]
        assert main(["run", "--config", str(cfg_path), *overrides]) == 2
