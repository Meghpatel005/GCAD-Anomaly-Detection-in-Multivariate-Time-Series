# gcad

Independent re-implementation of **GCAD** – *Anomaly Detection in Multivariate Time Series from the
Perspective of Granger Causality* (Liu, Gao, Jiao; AAAI-25) – with the code to train, evaluate, ablate and
sweep it. It was written from the paper alone; no reference code was used.

**Status:** the method, the metrics and the experiment pipeline are implemented and tested on a built-in
synthetic benchmark. The paper's results on SWaT, SMD, MSL, SMAP and PSM have **not** been reproduced
(the datasets are not redistributable and were not available). See [Status and limitations](#status-and-limitations).

## Method

1. A Mixer-style predictor is trained on anomaly-free data for one-step-ahead prediction.
2. For each sliding window `X` the channel-wise squared errors `L_j = (ŷ_j − y_j)²` are back-propagated to the
   input. The causality matrix is `A[i, j] = mean over lags of |∂L_j / ∂x_i|`.
3. `A` is sparsified: `Ã[i, j] = max(0, A[i, j] − A[j, i])` for `i ≠ j`, then values below a threshold `h` are zeroed.
4. A typical normal pattern `Ā` is the mean of `Ã` over Bernoulli-sampled training windows.
5. The anomaly score is `S = Sc + β·St`: relative deviation of the whole graph (`Sc`) and of its diagonal (`St`)
   from `Ā`, with `ε` in the denominators.

| Paper | Code |
|---|---|
| Mixer predictor | `src/gcad/model.py` |
| Eq. 1, 4, 5 – channel loss, gradients, causality matrix | `src/gcad/causality.py` |
| Eq. 6 – sparsification | `causality.sparsify` |
| Eq. 7–12 – normal pattern, deviation scores | `src/gcad/scoring.py` |
| Training, caching, ablations, sweeps | `src/gcad/pipeline.py` |
| AUROC / AUPRC | `src/gcad/metrics.py` |
| Fig. 2 (parameter sweeps), Fig. 3 (deviation heat map) | `gcad sweep`, `gcad case-study` |
| Published numbers (Tables 1–3) | `src/gcad/paper_results.py` |

The full Jacobian of the predictor is computed once per window with `torch.func.vmap(jacrev)`, using
`∂L_j/∂x = 2 (ŷ_j − y_j) · ∂ŷ_j/∂x`, instead of N backward passes per window.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional, reproduce the tested versions (Python 3.13): pip install -c constraints-tested.txt -e ".[dev]"
make check          # ruff, mypy, pytest (about 15 s on CPU)
```

CI (`.github/workflows/ci.yml`) runs the same checks on Python 3.10–3.13. Only 3.13 has been run by the author.

## Quick start (no downloads)

```bash
make demo           # gcad run --config configs/synthetic.yaml
```

This trains on a built-in nonlinear causal system whose anomalies *break causal links* (a child channel is
decoupled from its parents) and writes `runs/synthetic-w5/summary.md`: AUROC/AUPRC for GCAD, the three
ablations and two baselines, plus causal-structure recovery against the known graph.

Result of the shipped config (3 seeds, mean ± std, CPU):

| Variant | AUROC | AUPRC |
|---|---|---|
| gcad | 0.8646 ± 0.0209 | 0.5029 ± 0.1110 |
| no_spars | 0.8608 ± 0.0268 | 0.5002 ± 0.1119 |
| no_gc (St only) | 0.7707 ± 0.0484 | 0.3314 ± 0.0582 |
| no_tc (Sc only) | 0.8653 ± 0.0206 | 0.5047 ± 0.1116 |
| pred_error (baseline) | 0.8955 ± 0.0299 | 0.7203 ± 0.0884 |
| pca (baseline) | 0.8759 | 0.7018 |

Causal-structure recovery (AUROC of the mean normal graph against the true graph): 0.72 sparsified, 0.75 raw.

## Running the paper's experiments

1. Obtain the datasets and arrange them as in [`docs/DATASETS.md`](docs/DATASETS.md).
2. Run 10 seeds, as in the paper:

```bash
make swat           # or smd | msl | smap | psm  (== gcad run --config configs/<name>.yaml)
```

Output goes to `runs/<dataset>-w<window>/`: `summary.md` and `summary.csv` (all variants, next to the paper's GCAD
and ablation numbers), `per_seed.csv`, and per seed the model, causality graphs, scores and metrics.

Other commands take `--config`, repeatable `--set section.key=value`, `--seeds` and `--force`:

```bash
gcad train|extract|evaluate --config configs/smd.yaml
gcad sweep --config configs/swat.yaml --param scoring.threshold --values 0 0.001 0.01 0.1 0.3   # Fig. 2b
gcad sweep --config configs/swat.yaml --param model.window --values 1 3 5 7 9                    # Fig. 2a, retrains
gcad case-study --config configs/swat.yaml        # Fig. 3-style deviation heat map and top pairs
```

The ablations of Table 3 (`no_spars`, `no_gc`, `no_tc`) are produced by every run. Trained models and graphs are
cached per seed under `runs/`; the cache key includes the configuration and the source of the modules that
produce them, so scoring-only changes (`scoring.*`) and sweeps over them reuse the cache, and code edits do not.
Raw graphs take `windows × N × N × 4` bytes per seed (about 220 MB for PSM); use `causality.test_stride` to thin them.

## Configuration

YAML, validated; unknown keys are rejected (`src/gcad/config.py`). Sections: `data`, `model`, `train`,
`causality`, `scoring`, `run`. The main hyper-parameters are `model.window` (τ), `scoring.threshold` (h),
`scoring.beta` (β), `scoring.eps` (ε) and `causality.normal_sample_p` (Bernoulli p).

## Layout

```
configs/    one YAML per dataset, plus the synthetic demo
docs/       dataset layout
src/gcad/   library: data, model, causality, scoring, metrics, baselines, pipeline, report, cli
tests/      unit and end-to-end tests
```

## Status and limitations

- **Hyper-parameters are not in the paper** (τ, h, β, ε, p, hidden size, layers, learning rate, epochs). The configs
  hold starting values. Tune them without test labels (for example on a held-out part of the data or on injected
  anomalies); tuning on test labels inflates the numbers. The scale of `A`, and therefore `h`, depends on the data.
- **No real-data results.** The loaders are tested on small mock files. When the dataset statistics differ from the
  paper's Table 1, a warning is logged. The paper's SWaT training length (47,520) equals 49,680 minus 2,160 samples,
  so `swat.yaml` skips the first 21,600 raw rows; this is inferred from the numbers and untested on the real file.
  MSL and SMAP sizes in Table 1 refer to one telemetry entity that the paper does not name; set `data.entity`.
- **The causal graph may add little beyond prediction error.** Since `∂L_j/∂x` contains the residual `ŷ_j − y_j`, a
  large part of the score can come from prediction error alone. On the synthetic benchmark the plain
  prediction-error baseline of the same predictor is better than GCAD, differentiating the prediction instead of
  the loss (`causality.grad_target: prediction`) gave an AUROC below 0.5 in a one-seed check, and adding `Sc` and `St`
  to the error in a logistic regression changed AUROC by about 0.002. That benchmark is synthetic and was built for this
  project, so it says nothing certain about real data. On real data, compare `gcad` with `pred_error` in `summary.md`.
- **Interpretation choices** where the paper is ambiguous: the integral in Eq. 5 is a mean over lags (uniform P);
  `-GC` is the time-pattern deviation only; `h` also applies to the diagonal; `Ā` is sampled from the training split
  (`causality.normal_source`); `ε` is configurable; a window is labelled by the step it predicts
  (`data.label_mode: any` labels it anomalous if any step in the window is, which is closer to the paper's wording).
- **Model:** a TSMixer-like predictor with the paper's skip connections into the output layer, but a per-axis
  LayerNorm instead of TSMixer's 2-D normalisation.
- **Baselines:** DAGMM, USAD, GDN, Anomaly Transformer, GANF and MEMTO are not implemented; `paper_results.py` holds
  their published numbers. Included are prediction error (same predictor) and PCA reconstruction.
- Metrics are AUROC and AUPRC per window, without point adjustment.
