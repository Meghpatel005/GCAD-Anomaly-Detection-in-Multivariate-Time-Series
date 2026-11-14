"""End-to-end experiment pipeline: train predictor -> extract causal graphs -> score -> evaluate.

Expensive artefacts (model weights, raw causality matrices) are cached on disk under a key derived
from the relevant configuration, so scoring-only changes (threshold, beta, eps, ablations) and
sweeps never re-train or re-compute gradients.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import causality as causality_module
from . import data as data_module
from . import model as model_module
from . import paper_results, synthetic
from . import utils as utils_module
from .baselines import pca_reconstruction_scores, prediction_error_scores
from .causality import compute_causality_for_indices
from .config import Config, ScoringConfig, config_from_dict, config_to_dict
from .data import DataError, PreparedData, gather_windows, num_windows, prepare_data, window_labels
from .metrics import evaluate_scores
from .model import MixerPredictor
from .scoring import deviation_matrix, deviation_scores, normal_pattern, static_resid_graphs
from .utils import (
    ensure_dir,
    load_json,
    resolve_device,
    save_json,
    seed_everything,
    source_fingerprint,
    stable_hash,
)

LOG = logging.getLogger("gcad.pipeline")

VARIANTS = ("gcad", "no_spars", "no_gc", "no_tc", "static_resid", "sc_offdiag", "pred_error", "pca")
VARIANT_DESCRIPTIONS = {
    "gcad": "GCAD (full score S = Sc + beta*St, sparsified graphs)",
    "no_spars": "-Spars: no antisymmetrisation / thresholding",
    "no_gc": "-GC: time-pattern deviation St only (no inter-series causality)",
    "no_tc": "-TC: causal deviation Sc only (beta = 0)",
    "static_resid": "Control: one fixed graph (mean dyhat/dx of normal windows) x per-window residual, same scoring",
    "sc_offdiag": "Sc restricted to off-diagonal entries (pure inter-series causal deviation; not in the paper)",
    "pred_error": "Baseline: total one-step prediction error of the same predictor",
    "pca": "Baseline: PCA reconstruction error of the target step",
}


def _training_code_version() -> str:
    return source_fingerprint(
        model_module, data_module, synthetic,
        utils_module.seed_everything,
        Experiment._new_model, Experiment._eval_loss, Experiment.train,
    )


def run_name(cfg: Config) -> str:
    name = cfg.data.name
    if cfg.data.entity:
        name += "-" + re.sub(r"[^A-Za-z0-9_.-]", "_", cfg.data.entity)
    # F2b: different synthetic params must not collide in the same run dir
    if cfg.data.name == "synthetic" and cfg.data.synthetic:
        name += "-" + stable_hash(cfg.data.synthetic)[:6]
    return f"{name}-w{cfg.model.window}"


@dataclass
class Artifacts:
    a_normal: np.ndarray       # (n, N, N) raw causality graphs of sampled normal windows
    a_test: np.ndarray         # (M, N, N) raw causality graphs of test windows
    sq_err: np.ndarray         # (M, N) per-channel squared prediction errors
    labels: np.ndarray         # (M,) window labels
    indices: np.ndarray        # (M,) window start indices into the test series
    a_static: np.ndarray       # (N, N) fixed graph = mean prediction-gradient of normal windows
    normal_sq_err: np.ndarray  # (n, N) per-channel squared errors of normal windows


class Experiment:
    """One (config, seed) run with on-disk caching."""

    def __init__(self, cfg: Config, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.dir = ensure_dir(Path(cfg.run.output_dir) / run_name(cfg) / f"seed{seed}")
        self.device = resolve_device(cfg.train.device)
        self._data: PreparedData | None = None
        self._model: MixerPredictor | None = None

    # ------------------------------------------------------------------ keys
    @property
    def model_key(self) -> str:
        """Cache key of the trained weights: everything that influences training, including the code."""
        c = config_to_dict(self.cfg)
        data = {k: v for k, v in c["data"].items() if k != "label_mode"}  # labels do not affect training
        train = {k: v for k, v in c["train"].items() if k != "device"}
        return stable_hash(
            {"data": data, "model": c["model"], "train": train, "seed": self.seed, "code": _training_code_version()}
        )

    @property
    def extract_key(self) -> str:
        """Cache key of the causality graphs and window labels."""
        return stable_hash(
            {
                "model": self.model_key,
                "causality": config_to_dict(self.cfg)["causality"],
                "label_mode": self.cfg.data.label_mode,
                "code": source_fingerprint(causality_module, Experiment.extract),
                "schema": 2,  # bump when Artifacts fields change to invalidate old caches
            }
        )

    # ------------------------------------------------------------------ data
    @property
    def data(self) -> PreparedData:
        if self._data is None:
            self._data = prepare_data(self.cfg.data)
            tau = self.cfg.model.window
            if num_windows(len(self._data.train), tau) < 2:
                raise DataError("training split is too short for the chosen window")
            if num_windows(len(self._data.test), tau) < 2:
                raise DataError("test series is too short for the chosen window")
        return self._data

    # ------------------------------------------------------------------ training
    def _new_model(self) -> MixerPredictor:
        m = self.cfg.model
        return MixerPredictor(self.data.n_channels, m.window, m.hidden_dim, m.n_layers, m.dropout).to(self.device)

    def _eval_loss(self, model: MixerPredictor, series: torch.Tensor) -> float:
        tau = self.cfg.model.window
        n = num_windows(len(series), tau)
        if n < 1:
            return float("nan")
        model.eval()
        total = 0.0
        with torch.no_grad():
            for s in range(0, n, 1024):
                idx = torch.arange(s, min(n, s + 1024), device=series.device)
                x, y = gather_windows(series, idx, tau)
                total += F.mse_loss(model(x), y, reduction="sum").item()
        return total / (n * series.shape[1])

    def train(self, force: bool = False) -> MixerPredictor:
        if self._model is not None and not force:
            return self._model
        weights, meta_path = self.dir / "model.pt", self.dir / "train_meta.json"
        if (
            not force
            and weights.is_file()
            and meta_path.is_file()
            and load_json(meta_path).get("key") == self.model_key
        ):
            model = self._new_model()
            model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
            model.eval()
            LOG.info("loaded cached model from %s", weights)
            self._model = model
            return model

        cfg, tau = self.cfg, self.cfg.model.window
        seed_everything(self.seed, cfg.run.deterministic)
        model = self._new_model()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        train_t = torch.as_tensor(self.data.train, device=self.device)
        val_t = torch.as_tensor(self.data.val, device=self.device)
        n_train = num_windows(len(train_t), tau)
        has_val = num_windows(len(val_t), tau) >= 1
        if not has_val:
            LOG.warning("validation split too short; early stopping falls back to the training loss")
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad, history = float("inf"), None, 0, []
        t0 = time.time()
        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            perm = torch.randperm(n_train, generator=gen).to(self.device)
            running = 0.0
            for s in range(0, n_train, cfg.train.batch_size):
                x, y = gather_windows(train_t, perm[s : s + cfg.train.batch_size], tau)
                loss = F.mse_loss(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step()
                running += loss.item() * len(x)
            train_loss = running / n_train
            val_loss = self._eval_loss(model, val_t) if has_val else train_loss
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            LOG.info("epoch %3d train=%.5f val=%.5f", epoch, train_loss, val_loss)
            if not np.isfinite(val_loss):
                raise FloatingPointError("non-finite loss; lower train.lr or check the data")
            if val_loss < best - 1e-7:
                best, bad = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.train.patience:
                    LOG.info("early stopping at epoch %d", epoch)
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        torch.save(model.state_dict(), weights)
        pd.DataFrame(history).to_csv(self.dir / "history.csv", index=False)
        save_json(
            {"key": self.model_key, "best_val_loss": best, "epochs_run": len(history), "seconds": time.time() - t0},
            meta_path,
        )
        self._model = model
        return model

    # ------------------------------------------------------------------ causality extraction
    def extract(self, force: bool = False) -> Artifacts:
        names = ("a_normal", "a_test", "sq_err", "labels", "indices", "a_static", "normal_sq_err")
        files = {n: self.dir / f"{n}.npy" for n in names}
        meta_path = self.dir / "extract_meta.json"
        if (
            not force
            and all(f.is_file() for f in files.values())
            and meta_path.is_file()
            and load_json(meta_path).get("key") == self.extract_key
        ):
            LOG.info("loaded cached causality graphs from %s", self.dir)
            return Artifacts(**{n: np.load(f, allow_pickle=False) for n, f in files.items()})

        model, cfg, tau = self.train(), self.cfg, self.cfg.model.window
        c = cfg.causality
        rng = np.random.default_rng(self.seed)
        train_t = torch.as_tensor(self.data.train, device=self.device)
        val_t = torch.as_tensor(self.data.val, device=self.device)
        sources = {"train": [train_t], "val": [val_t], "both": [train_t, val_t]}[c.normal_source]
        sources = [s for s in sources if num_windows(len(s), tau) >= 1]
        if not sources:
            raise ValueError("no usable windows for the normal causal pattern")
        cap = max(1, -(-c.max_normal_windows // len(sources)))

        n_channels = self.data.n_channels
        normal, normal_err_list = [], []
        pred_sum = np.zeros((n_channels, n_channels), dtype=np.float64)
        pred_n = 0
        for series in sources:
            n_win = num_windows(len(series), tau)
            idx = np.flatnonzero(rng.random(n_win) < c.normal_sample_p)  # Bernoulli sampling (Eq. 7)
            if len(idx) == 0:
                idx = rng.integers(0, n_win, size=1)
            if len(idx) > cap:
                idx = np.sort(rng.choice(idx, size=cap, replace=False))
            LOG.info("normal pattern: %d windows", len(idx))
            # Loss-gradient pass (paper default) — used for the normal causal pattern
            mats_loss, errs = compute_causality_for_indices(model, series, idx, tau, c.batch_size, c.grad_target)
            # Prediction-gradient pass — used to build the static graph (F1 control)
            mats_pred, _ = compute_causality_for_indices(model, series, idx, tau, c.batch_size, "prediction")
            normal.append(mats_loss)
            normal_err_list.append(errs)
            pred_sum += mats_pred.sum(0)
            pred_n += len(idx)

        a_normal = np.concatenate(normal)
        a_static = (pred_sum / pred_n).astype(np.float32)   # (N, N) fixed "graph of the frozen model"
        normal_sq_err = np.concatenate(normal_err_list)      # (n_normal, N)

        test_t = torch.as_tensor(self.data.test, device=self.device)
        indices = np.arange(0, num_windows(len(test_t), tau), c.test_stride)
        LOG.info("extracting causality graphs for %d test windows", len(indices))
        t0 = time.time()
        a_test, sq_err = compute_causality_for_indices(model, test_t, indices, tau, c.batch_size, c.grad_target)
        LOG.info("extraction took %.1fs", time.time() - t0)
        labels = window_labels(self.data.test_labels, tau, indices, cfg.data.label_mode)
        art = Artifacts(a_normal, a_test, sq_err, labels, indices, a_static, normal_sq_err)
        for n, f in files.items():
            np.save(f, getattr(art, n), allow_pickle=False)
        save_json({"key": self.extract_key}, meta_path)
        return art

    # ------------------------------------------------------------------ scoring / evaluation
    def score_variants(
        self,
        art: Artifacts,
        sc: ScoringConfig | None = None,
        variants: Iterable[str] | None = None,
    ) -> dict[str, np.ndarray]:
        sc = sc or self.cfg.scoring
        tau = self.cfg.model.window
        requested = set(variants) if variants is not None else set(VARIANTS)

        result: dict[str, np.ndarray] = {}

        # GCAD-family variants share the sparsified normal pattern and deviation_scores output
        gcad_family = requested & {"gcad", "no_gc", "no_tc", "sc_offdiag"}
        if gcad_family:
            bar = normal_pattern(art.a_normal, sc.threshold, sc.sparsify)
            full = deviation_scores(
                art.a_test, bar, threshold=sc.threshold, beta=sc.beta, eps=sc.eps, do_sparsify=sc.sparsify
            )
            if "gcad" in requested:
                result["gcad"] = full["S"]
            if "no_gc" in requested:
                result["no_gc"] = full["St"]
            if "no_tc" in requested:
                result["no_tc"] = full["Sc"]
            if "sc_offdiag" in requested:
                result["sc_offdiag"] = full["Sc_off"]

        # no_spars needs its own unsparsified pass
        if "no_spars" in requested:
            bar_raw = normal_pattern(art.a_normal, 0.0, False)
            raw = deviation_scores(art.a_test, bar_raw, threshold=0.0, beta=sc.beta, eps=sc.eps, do_sparsify=False)
            result["no_spars"] = raw["S"]

        # static_resid: fixed graph x per-window residual (control for F1)
        if "static_resid" in requested:
            sr_graphs = static_resid_graphs(art.normal_sq_err, art.a_static)
            bar_s = normal_pattern(sr_graphs, sc.threshold, sc.sparsify)
            sr_test = deviation_scores(
                static_resid_graphs(art.sq_err, art.a_static),
                bar_s,
                threshold=sc.threshold, beta=sc.beta, eps=sc.eps, do_sparsify=sc.sparsify,
            )
            result["static_resid"] = sr_test["S"]

        if "pred_error" in requested:
            result["pred_error"] = prediction_error_scores(art.sq_err)

        if "pca" in requested:
            result["pca"] = pca_reconstruction_scores(self.data.train, self.data.test, art.indices, tau)

        return result

    def evaluate(
        self,
        scoring: ScoringConfig | None = None,
        save: bool = True,
        variants: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        sc = scoring or self.cfg.scoring
        art = self.extract()
        scores = self.score_variants(art, sc, variants=variants)
        metrics = {k: evaluate_scores(art.labels, v) for k, v in scores.items()}
        result: dict[str, Any] = {
            "seed": self.seed,
            "n_windows": len(art.labels),
            "anomaly_ratio": float(art.labels.mean()),
            "metrics": metrics,
        }
        if self.data.true_graph is not None:
            result["causal_recovery"] = self._causal_recovery(art, sc)
        if save:
            save_json(result, self.dir / "metrics.json")
            # numpy types **kwargs as the bool `allow_pickle`; the arrays are valid named entries.
            np.savez_compressed(self.dir / "scores.npz", labels=art.labels, **scores)  # type: ignore[arg-type]
        return result

    def _causal_recovery(self, art: Artifacts, sc: ScoringConfig) -> dict[str, float]:
        """AUROC of the mean normal causality matrix against the known synthetic graph (off-diagonal)."""
        truth = self.data.true_graph
        assert truth is not None
        off = ~np.eye(truth.shape[0], dtype=bool)
        out = {}
        for name, mat in (
            ("sparsified", normal_pattern(art.a_normal, sc.threshold, True)),
            ("raw", normal_pattern(art.a_normal, 0.0, False)),
        ):
            out[name] = float(roc_auc_score(truth[off], mat[off])) if len(np.unique(truth[off])) > 1 else float("nan")
        return out

    def tune(self, grid: dict[str, list]) -> pd.DataFrame:
        """Grid-search scoring hyperparameters using injected val anomalies.

        Builds an augmented validation series with synthetic anomalies (inject.py),
        extracts causality graphs with the trained model, and scores every grid
        point for the 'gcad' variant. Never reads test data.

        Args:
            grid: mapping of "section.key" -> list of values, e.g.
                  {"scoring.threshold": [0, 0.001, 0.01], "scoring.beta": [0.5, 1, 2]}

        Returns:
            DataFrame with columns: one per grid key, auroc, auprc.
        """
        from itertools import product

        from . import inject as inject_module

        model = self.train()
        tau = self.cfg.model.window

        # Build augmented val series (inject anomalies; never touches test data)
        rng = np.random.default_rng(self.seed)
        val_arr = self.data.val  # (T_val, N), already standardised
        x_aug, val_labels = inject_module.inject_anomalies(val_arr, rng)
        if len(np.unique(val_labels)) < 2:
            LOG.warning("inject_anomalies produced a single-class label; tune results may be unreliable")

        # Extract graphs for augmented val using the trained model
        val_t = torch.as_tensor(x_aug, device=self.device)
        n_win = num_windows(len(val_t), tau)
        if n_win < 2:
            raise ValueError("validation series too short for tuning")
        indices = np.arange(n_win)
        a_val, sq_err_val = compute_causality_for_indices(
            model, val_t, indices, tau, self.cfg.causality.batch_size, self.cfg.causality.grad_target
        )
        val_win_labels = window_labels(val_labels, tau, indices, self.cfg.data.label_mode)

        # Use a_normal from trained extraction as the normal pattern
        art = self.extract()
        a_norm = art.a_normal

        rows = []
        for combo in product(*[grid[k] for k in grid]):
            raw = config_to_dict(self.cfg)
            for key, val in zip(grid, combo):
                section, *path, leaf = key.split(".")
                node = raw.setdefault(section, {})
                for part in path:
                    node = node.setdefault(part, {}) if isinstance(node, dict) else node
                node[leaf] = val
            sc = config_from_dict(raw).scoring
            bar = normal_pattern(a_norm, sc.threshold, sc.sparsify)
            gcad_scores = deviation_scores(
                a_val, bar, threshold=sc.threshold, beta=sc.beta, eps=sc.eps, do_sparsify=sc.sparsify
            )["S"]
            m = evaluate_scores(val_win_labels, gcad_scores)
            row = dict(zip(grid, combo))
            row.update(m)
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ case study
    def case_study(self) -> dict[str, Any]:
        """Deviation matrix D = |A~_test - A_norm| of the most anomalous labelled window."""
        art, sc = self.extract(), self.cfg.scoring
        bar = normal_pattern(art.a_normal, sc.threshold, sc.sparsify)
        s = deviation_scores(
            art.a_test, bar, threshold=sc.threshold, beta=sc.beta, eps=sc.eps, do_sparsify=sc.sparsify
        )["S"]
        if not art.labels.any():
            raise ValueError("no anomalous windows in the test set")
        k = int(np.argmax(np.where(art.labels == 1, s, -np.inf)))
        d = deviation_matrix(art.a_test[k], bar, sc.threshold, sc.sparsify)
        return {"window": int(art.indices[k]), "score": float(s[k]), "deviation": d}


def run_experiment(cfg: Config, seeds: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    """Run all seeds and return/write the aggregated summary (mean +- std over seeds)."""
    seeds = seeds if seeds is not None else cfg.run.seeds
    rows, recovery = [], []
    out_dir = ensure_dir(Path(cfg.run.output_dir) / run_name(cfg))
    save_json(config_to_dict(cfg), out_dir / "config.json")
    for seed in seeds:
        exp = Experiment(cfg, seed)
        if force:
            exp.train(force=True)
            exp.extract(force=True)
        res = exp.evaluate()
        for variant, m in res["metrics"].items():
            rows.append({"seed": seed, "variant": variant, **m})
        if "causal_recovery" in res:
            recovery.append({"seed": seed, **res["causal_recovery"]})
    df = pd.DataFrame(rows)
    summary = df.groupby("variant", sort=False)[["auroc", "auprc"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reindex([v for v in VARIANTS if v in summary.index]).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    df.to_csv(out_dir / "per_seed.csv", index=False)
    (out_dir / "summary.md").write_text(summary_markdown(summary, cfg, len(seeds), pd.DataFrame(recovery)))
    return summary


def summary_markdown(summary: pd.DataFrame, cfg: Config, n_seeds: int, recovery: pd.DataFrame | None = None) -> str:
    def fmt(m: float, s: float) -> str:
        return f"{m:.4f} ± {0.0 if pd.isna(s) else s:.4f}"

    lines = [
        f"# Results: {cfg.data.name}{'/' + cfg.data.entity if cfg.data.entity else ''} "
        f"(window={cfg.model.window}, {n_seeds} seed(s))",
        "",
        "| Variant | AUROC | AUPRC | Description |",
        "|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['variant']} | {fmt(r['auroc_mean'], r['auroc_std'])} | "
            f"{fmt(r['auprc_mean'], r['auprc_std'])} | {VARIANT_DESCRIPTIONS[r['variant']]} |"
        )
    ref = paper_results.TABLE2["GCAD"].get(cfg.data.name)
    if ref:
        lines += [
            "",
            f"Paper (Table 2, GCAD, {cfg.data.name}): AUROC {ref[0]:.4f}, AUPRC {ref[1]:.4f}. "
            "Published numbers, shown for reference only.",
        ]
        ablation = {v: row[cfg.data.name] for v, row in paper_results.TABLE3.items() if cfg.data.name in row}
        if ablation:
            lines += [
                "Paper ablation (Table 3, AUROC / AUPRC): "
                + "; ".join(f"{v} {a:.4f}/{p:.4f}" for v, (a, p) in ablation.items())
                + "."
            ]
    if recovery is not None and not recovery.empty:
        lines += [
            "",
            "Causal-structure recovery (AUROC of mean normal graph vs. true synthetic graph): "
            f"sparsified {recovery['sparsified'].mean():.4f}, raw {recovery['raw'].mean():.4f}.",
        ]
    return "\n".join(lines) + "\n"


def synthetic_benchmark(
    cfg: Config,
    dataset_seeds: list[int],
    model_seeds: list[int],
    out_dir: str | Path,
) -> pd.DataFrame:
    """Run all (dataset_seed, model_seed) pairs; write bench.csv and bench.md.

    For each dataset seed the config's synthetic.seed is overridden and the
    output_dir is set to a unique subdirectory so run_name collisions are impossible.
    """
    from scipy.stats import binomtest

    out_dir = ensure_dir(Path(out_dir))
    rows = []
    for ds in dataset_seeds:
        raw = config_to_dict(cfg)
        raw["data"]["synthetic"]["seed"] = ds
        raw["run"]["output_dir"] = str(out_dir / f"ds{ds}")
        c = config_from_dict(raw)
        for ms in model_seeds:
            res = Experiment(c, ms).evaluate(save=False)
            for variant, m in res["metrics"].items():
                rows.append({"dataset_seed": ds, "model_seed": ms, "variant": variant, **m})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bench.csv", index=False)

    # Average model seeds within each dataset first, then compute stats over datasets
    per_ds = (
        df.groupby(["dataset_seed", "variant"])[["auroc", "auprc"]]
        .mean()
        .reset_index()
    )
    agg = per_ds.groupby("variant")[["auroc", "auprc"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["variant"] + [f"{a}_{b}" for a, b in agg.columns[1:]]

    baseline = per_ds[per_ds["variant"] == "pred_error"].set_index("dataset_seed")
    n_ds = len(dataset_seeds)

    md_lines = ["# Synthetic Benchmark", "", "## Per-variant summary (mean ± std over dataset seeds)", ""]
    md_lines += ["| Variant | AUROC | AUPRC |", "|---|---|---|"]
    for _, r in agg.iterrows():
        auroc_std = 0.0 if pd.isna(r["auroc_std"]) else r["auroc_std"]
        auprc_std = 0.0 if pd.isna(r["auprc_std"]) else r["auprc_std"]
        md_lines.append(
            f"| {r['variant']} | {r['auroc_mean']:.4f} ± {auroc_std:.4f} | {r['auprc_mean']:.4f} ± {auprc_std:.4f} |"
        )

    md_lines += ["", "## Paired comparison vs pred_error", ""]
    md_lines += ["| Variant | AUROC wins | AUROC p | AUPRC wins | AUPRC p |", "|---|---|---|---|---|"]
    for variant in VARIANTS:
        if variant == "pred_error":
            continue
        var_ds = per_ds[per_ds["variant"] == variant].set_index("dataset_seed")
        joined = var_ds.join(baseline, rsuffix="_base")
        auroc_wins = int((joined["auroc"] > joined["auroc_base"]).sum())
        auprc_wins = int((joined["auprc"] > joined["auprc_base"]).sum())
        auroc_no_tie = int((joined["auroc"] != joined["auroc_base"]).sum())
        auprc_no_tie = int((joined["auprc"] != joined["auprc_base"]).sum())
        auroc_p = binomtest(auroc_wins, auroc_no_tie, 0.5).pvalue if auroc_no_tie > 0 else float("nan")
        auprc_p = binomtest(auprc_wins, auprc_no_tie, 0.5).pvalue if auprc_no_tie > 0 else float("nan")
        md_lines.append(
            f"| {variant} | {auroc_wins}/{n_ds} | {auroc_p:.4f} | {auprc_wins}/{n_ds} | {auprc_p:.4f} |"
        )

    (out_dir / "bench.md").write_text("\n".join(md_lines) + "\n")
    return df


def sweep(cfg: Config, param: str, values: list[Any], seeds: list[int] | None = None) -> pd.DataFrame:
    """Sweep ``section.key`` over ``values``; scoring params reuse cached graphs, others retrain."""
    seeds = seeds if seeds is not None else cfg.run.seeds
    section, key = param.split(".", 1)
    if section not in {"scoring", "model", "causality"}:
        raise ValueError("sweep supports scoring.*, model.* and causality.* parameters")
    rows = []
    for value in values:
        raw = config_to_dict(cfg)
        raw[section][key] = value
        c = config_from_dict(raw)
        for seed in seeds:
            exp = Experiment(c, seed)
            # Only compute the two variants sweep cares about — avoids running PCA for every value x seed
            res = exp.evaluate(save=False, variants=("gcad", "pred_error"))
            for variant in ("gcad", "pred_error"):
                rows.append(
                    {"param": param, "value": value, "seed": seed, "variant": variant, **res["metrics"][variant]}
                )
        LOG.info("sweep %s=%s done", param, value)
    return pd.DataFrame(rows)


__all__ = ["VARIANTS", "Experiment", "run_experiment", "run_name", "sweep", "synthetic_benchmark"]
