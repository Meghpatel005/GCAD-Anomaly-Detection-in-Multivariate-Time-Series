
<div align="center">

# GCAD

**An independent, tested re-implementation of GCAD for Granger-causality-based anomaly detection in multivariate time series.**

A paper-driven implementation, reproducible synthetic experiments, and a critical evaluation of what the learned causal graph contributes.

[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-68%20passing-brightgreen)](#development)
[![License](https://img.shields.io/badge/license-see%20repository-lightgrey)](LICENSE)

[Overview](#overview) · [Method](#method) · [Findings](#experimental-findings) · [Installation](#installation) · [Usage](#usage) · [Development](#development)

</div>

---

## Overview

GCAD is an implementation of the method introduced by Liu, Gao, and Jiao in *GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality* (AAAI 2025).

This repository re-implements the method from the paper alone; no reference implementation was used. It includes the complete experimental pipeline: a Mixer-based predictor, gradient-derived causality graphs, graph sparsification, causal-deviation scoring, ablations, parameter sweeps, and AUROC/AUPRC evaluation.

> **The central result is a cautionary one:** on the built-in synthetic benchmark, GCAD scores below the plain prediction error of its own predictor on all eight independently generated datasets. A fixed graph combined with the per-window residual also outperforms GCAD in the included control experiment.

The repository is organised to make this result inspectable and reproducible—not to obscure it.

### Project status

| Area | Status |
|---|---|
| Core method | Implements the paper's Eqs. 5–12 |
| Automated tests | 68 unit and end-to-end tests |
| Code quality | `ruff` and `mypy` checks pass |
| Synthetic experiments | Reproducible on CPU |
| Published real-world results | Not reproduced; datasets were not available |
| Reference implementation | Not used |

**Important:** Results reported here are from the repository's synthetic benchmark. They do not reproduce or refute the paper's reported results on SWaT, SMD, MSL, SMAP, or PSM.

## Table of contents

- [Overview](#overview)
- [Method](#method)
- [Experimental findings](#experimental-findings)
  - [Shipped demo](#1-shipped-demo)
  - [Across independently generated datasets](#2-results-across-datasets)
  - [Fixed-graph control](#3-fixed-graph-control)
  - [Interpreting the results](#interpreting-the-results)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick start](#quick-start)
  - [Running the paper's experiments](#running-the-papers-experiments)
  - [Command-line interface](#command-line-interface)
  - [Caching](#caching)
- [Configuration](#configuration)
- [Repository structure](#repository-structure)
- [Development and testing](#development-and-testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [References](#references)

## Method

The implementation follows the paper's pipeline:

1. **Prediction.** Train a TSMixer-style model on anomaly-free data to perform one-step-ahead forecasting.
2. **Gradient-based graph extraction.** For each sliding window, back-propagate the channel-wise squared prediction error \(L_j = (\hat{y}_j - y_j)^2\) to the input. The causality matrix is \(A[i,j] = \operatorname{mean}_{\text{lags}} |\partial L_j / \partial x_i|\) (Eqs. 1, 4, and 5).
3. **Sparsification.** Remove symmetric influence and threshold weak edges: \(\tilde{A}[i,j] = \max(0, A[i,j] - A[j,i])\) for \(i \ne j\), then set entries below threshold \(h\) to zero (Eq. 6).
4. **Normal-pattern estimation.** Estimate the normal graph \(\bar{A}\) by averaging sparsified graphs from Bernoulli-sampled training windows (Eqs. 7–8).
5. **Anomaly scoring.** Combine whole-graph deviation \(S_c\) and diagonal time-pattern deviation \(S_t\): \(S = S_c + \beta S_t\), with \(\epsilon\) stabilising the denominators (Eqs. 10–12).

### Implementation details

The full predictor Jacobian is computed once per window using `torch.func.vmap(jacrev)`. The channel-wise loss gradient is then obtained from

\[
\frac{\partial L_j}{\partial x}
=
2(\hat{y}_j-y_j)\frac{\partial \hat{y}_j}{\partial x}.
\]

This avoids running \(N\) separate backward passes for a window. A test compares the result against a brute-force autograd loop.

### Paper-to-code map

| Paper component | Implementation |
|---|---|
| Mixer predictor | [`src/gcad/model.py`](src/gcad/model.py) |
| Eqs. 1, 4, 5 — channel loss, gradients, causality matrix | [`src/gcad/causality.py`](src/gcad/causality.py) |
| Eq. 6 — graph sparsification | `causality.sparsify` |
| Eqs. 7–12 — normal pattern and deviation scores | [`src/gcad/scoring.py`](src/gcad/scoring.py) |
| Training, caching, ablations, and sweeps | [`src/gcad/pipeline.py`](src/gcad/pipeline.py) |
| AUROC and AUPRC | [`src/gcad/metrics.py`](src/gcad/metrics.py) |
| Figures 2 and 3 — parameter sweeps and deviation heat map | `gcad sweep`, `gcad case-study` |
| Published results (Tables 1–3) | [`src/gcad/paper_results.py`](src/gcad/paper_results.py) |

## Experimental findings

All results in this section use the built-in synthetic benchmark: a nonlinear causal system in which anomalies decouple a child channel from its parents. Experiments run on CPU and report window-level AUROC/AUPRC without point adjustment.

Every result can be reproduced using the commands shown below.

### 1. Shipped demo

Run `make demo` to reproduce the default experiment. The reported mean and standard deviation use three model initialisation seeds on a **single dataset** (`data.synthetic.seed = 0`).

| Variant | AUROC | AUPRC |
|---|---:|---:|
| GCAD | 0.8646 ± 0.0209 | 0.5029 ± 0.1110 |
| No sparsification (`no_spars`) | 0.8608 ± 0.0268 | 0.5002 ± 0.1119 |
| No graph component (`no_gc`, \(S_t\) only) | 0.7707 ± 0.0484 | 0.3314 ± 0.0582 |
| No time component (`no_tc`, \(S_c\) only) | 0.8653 ± 0.0206 | 0.5047 ± 0.1116 |
| Prediction error baseline | **0.8955 ± 0.0299** | **0.7203 ± 0.0884** |
| PCA baseline | 0.8759 | 0.7018 |

Causal-structure recovery, measured as AUROC between the mean normal graph and the known ground-truth graph, is 0.72 for the sparsified graph and 0.75 for the raw graph.

### 2. Results across datasets

The script [`scripts/compare_synthetic_datasets.py`](scripts/compare_synthetic_datasets.py) evaluates eight independently generated datasets. Each dataset has a newly sampled causal graph, lag structure, and anomaly segments; one model seed is used per dataset.

| Variant | AUROC | AUPRC |
|---|---:|---:|
| GCAD | 0.856 ± 0.046 | 0.451 ± 0.105 |
| No graph component (`no_gc`) | 0.766 ± 0.065 | 0.383 ± 0.111 |
| Prediction error baseline | **0.936 ± 0.024** | **0.842 ± 0.060** |
| PCA baseline | 0.874 ± 0.037 | 0.690 ± 0.098 |

GCAD performs worse than prediction error on **all eight datasets**. The paired AUROC difference ranges from −0.04 to −0.13; a two-sided sign test gives \(p = 0.008\).

### 3. Fixed-graph control

The script [`scripts/static_graph_control.py`](scripts/static_graph_control.py) tests whether the per-window graph contributes information beyond the prediction residual.

At test time, the GCAD graph is

\[
A[i,j] = \operatorname{mean}_{\text{lags}}
\left|2r_j \cdot \frac{\partial \hat{y}_j}{\partial x_i}\right|.
\]

The Jacobian is a property of the frozen network at the current input; it is not a direct estimate of the data-generating process at time \(t\). The per-window residual \(r_j\) is the component that changes with the observed prediction error.

The control therefore replaces the per-window Jacobian with **one fixed matrix**—the mean prediction gradient over normal windows—while retaining the per-window residual and applying the same Eqs. 6–12 scoring pipeline.

| Model seed | GCAD AUROC / AUPRC | Fixed graph × residual | Prediction error |
|---:|---:|---:|---:|
| 0 | 0.844 / 0.391 | **0.937 / 0.852** | 0.930 / 0.822 |
| 1 | 0.865 / 0.505 | **0.887 / 0.694** | 0.880 / 0.680 |
| 2 | 0.885 / 0.613 | **0.887 / 0.693** | 0.876 / 0.660 |

The fixed-graph control outperforms GCAD on every seed in this experiment. On this benchmark, the instance-specific graph does not improve on a fixed graph scaled by the residual.

*Scope note: this control is currently a standalone diagnostic script, not a pipeline variant. The table uses dataset 0.*

### Interpreting the results

- **The benchmark is deliberately simple.** Breaking a causal link also increases one-step prediction error, and PCA—which ignores temporal structure—already reaches approximately 0.87 AUROC. This benchmark can demonstrate that the graph may be unnecessary; it cannot establish that the graph is useful.
- **These results are limited to the synthetic setting.** They are not a refutation of the paper's results on real-world datasets.
- **Use baselines when evaluating real data.** Each generated `summary.md` reports GCAD alongside `pred_error` and `pca` to make the comparison explicit.

## Installation

### Requirements

- Python 3.10 or later
- PyTorch 2.1 or later

CI is configured for Python 3.10–3.13. The author has run the checks on Python 3.13.

### Install from source

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

To install the tested dependency versions (Python 3.13):

```bash
pip install -c constraints-tested.txt -e ".[dev]"
```

Run the full quality-check suite:

```bash
make check
```

## Usage

### Quick start

The synthetic demo requires no external dataset downloads.

```bash
make demo
```

This runs:

```bash
gcad run --config configs/synthetic.yaml
```

To run the diagnostics supporting the findings in this README:

```bash
make diagnostics
```

Approximate CPU runtime:

| Command | Runtime |
|---|---:|
| `make demo` | 25 seconds |
| `make diagnostics` | 2 minutes |
| `make check` | 20 seconds |

`make demo` writes results to `runs/synthetic-w5/summary.md`, including AUROC/AUPRC for GCAD, its ablations, the two baselines, and causal-structure recovery against the known synthetic graph.

### Running the paper's experiments

1. Obtain the datasets and arrange them according to [`docs/DATASETS.md`](docs/DATASETS.md).
2. Run the desired dataset configuration. The paper uses 10 seeds.

```bash
make swat
# Alternatives:
make smd
make msl
make smap
make psm
```

Each target invokes `gcad run --config configs/<dataset>.yaml`.

Results are written to `runs/<dataset>-w<window>/` and include:

- `summary.md` and `summary.csv` — variant-level metrics alongside the paper's GCAD and ablation numbers.
- `per_seed.csv` — metrics for each seed.
- Per-seed model checkpoints, causality graphs, anomaly scores, and evaluation metrics.

### Command-line interface

| Command | Description |
|---|---|
| `gcad run --config C` | Train, extract graphs, and evaluate across all configured seeds |
| `gcad train --config C` | Train the predictor |
| `gcad extract --config C` | Extract causality graphs |
| `gcad evaluate --config C` | Score and evaluate a single seed |
| `gcad sweep --config C --param P --values V...` | Sweep a hyperparameter |
| `gcad case-study --config C` | Generate a deviation heat map and top channel pairs (Figure 3 style) |

Every command accepts:

- `--seeds` to select model seeds.
- `--force` to bypass cached outputs.
- Repeatable `--set section.key=value` overrides, including nested keys.

Examples:

```bash
# Figure 2b-style threshold sweep
gcad sweep \
  --config configs/swat.yaml \
  --param scoring.threshold \
  --values 0 0.001 0.01 0.1 0.3

# Figure 2a-style window-size sweep (re-trains the model)
gcad sweep \
  --config configs/swat.yaml \
  --param model.window \
  --values 1 3 5 7 9

# Override a nested configuration value
gcad run --config configs/synthetic.yaml --set data.synthetic.seed=3
```

### Caching

Models and extracted graphs are cached per seed under `runs/`.

Cache keys include both the configuration and the source of the modules that produce the cached artefacts. As a result:

- Changes to scoring-only parameters (`scoring.*`) and scoring sweeps reuse existing training and extraction outputs.
- Changes to training or graph-extraction code invalidate the relevant cache.

Raw graph storage is approximately `windows × N × N × 4` bytes per seed (about 220 MB for PSM). Use `causality.test_stride` to reduce the number of stored test windows.

## Configuration

Configuration files use validated YAML. Unknown keys are rejected; see [`src/gcad/config.py`](src/gcad/config.py).

The configuration is organised into six sections:

| Section | Purpose |
|---|---|
| `data` | Dataset and label handling |
| `model` | Predictor architecture and window size |
| `train` | Optimisation and training settings |
| `causality` | Graph extraction and normal-window sampling |
| `scoring` | Sparsification and anomaly-score parameters |
| `run` | Experiment execution and output settings |

### Key parameters

| Parameter | Description |
|---|---|
| `model.window` | \(\tau\), the maximum lag considered |
| `scoring.threshold` | \(h\), the graph sparsification threshold |
| `scoring.beta` | \(\beta\), weight of the time-pattern deviation |
| `scoring.eps` | \(\epsilon\), stabiliser in the scoring denominators |
| `causality.normal_sample_p` | \(p\), Bernoulli sampling probability for normal windows |
| `data.label_mode` | `target`: label the predicted step; `any`: label a window anomalous if any step in it is anomalous |

One configuration file is provided per dataset in [`configs/`](configs/).

## Repository structure

```text
.
├── configs/       Dataset and synthetic-demo YAML configurations
├── docs/          Dataset preparation and layout documentation
├── scripts/       Diagnostic and comparison experiments
├── src/
│   └── gcad/      Core library: data, model, causality, scoring,
│                  metrics, baselines, pipeline, reporting, and CLI
├── tests/         Unit and end-to-end tests
├── runs/          Generated experiment outputs (created at runtime)
└── README.md
```

## Development and testing

Run the complete development checks:

```bash
make check
```

This runs Ruff linting, Ruff formatting checks, mypy, and pytest.

Useful alternatives:

```bash
# Run the test suite
make test

# Skip the slow end-to-end training test
python -m pytest -m "not slow"
```

The test suite covers:

- Numerical correctness, including Jacobian equivalence against autograd.
- Causality recovery on a known linear system.
- Hand-computed anomaly scores.
- Dataset loaders using mock files, including path-traversal rejection.
- Configuration validation and cache-key behaviour.
- An end-to-end regression test in which GCAD detects decoupling anomalies in the synthetic benchmark.

## Limitations

This implementation is intended to be transparent about what has—and has not—been validated.

1. **Real-world datasets have not been evaluated.** Loaders are tested on small mock files only. A warning is logged when dataset statistics differ from Table 1 of the paper. The SWaT preprocessing setting (`skip_head: 21600`) is inferred from the paper's reported numbers and has not been tested against the original file. The MSL and SMAP sizes in Table 1 refer to a telemetry entity that the paper does not identify; set `data.entity` explicitly.
2. **Several hyperparameters are unspecified in the paper.** These include \(\tau\), \(h\), \(\beta\), \(\epsilon\), \(p\), hidden size, layer count, learning rate, and number of epochs. The supplied configurations are starting points. Tune without test labels. The `sweep` command reports test metrics and is descriptive; a validation-based tuning protocol is not yet implemented.
3. **Some paper details require interpretation.** The integral in Eq. 5 is implemented as a uniform mean over lags. \(S_c\) includes the diagonal, as stated in the paper; consequently, the diagonal is counted twice in \(S = S_c + \beta S_t\), and the `no_tc` ablation is nearly identical to the full score. The threshold \(h\) also applies to the diagonal. The normal pattern \(\bar{A}\) is sampled from the training split (`causality.normal_source`), and each window is labelled by the step it predicts.
4. **The predictor is TSMixer-like, not an exact architectural reproduction.** It includes the paper's skip connections into the output layer but uses per-axis LayerNorm instead of TSMixer's 2-D normalisation.
5. **Baseline coverage is limited.** DAGMM, USAD, GDN, Anomaly Transformer, GANF, and MEMTO are not implemented. Their published numbers are stored in `paper_results.py` for reference. The implemented baselines are prediction error from the same predictor and PCA reconstruction.
6. **Metrics are window-level.** Evaluation reports AUROC and AUPRC without point adjustment.
7. **The synthetic benchmark covers one anomaly type.** It is built for this project and currently models causal decoupling at full strength.

## Roadmap

- [ ] Evaluate SMD (machine-1-1) and PSM; compare GCAD against prediction error, PCA, and the fixed-graph control.
- [ ] Integrate the fixed-graph control as a pipeline variant so it appears in every `summary.md`.
- [ ] Add graded anomaly strengths to the synthetic benchmark to identify when, if ever, the graph provides a benefit.
- [ ] Add a label-free tuning protocol for \(h\), \(\beta\), and \(\epsilon\), using anomalies injected into the validation split.

## References

1. Z. Liu, M. Gao, and P. Jiao. [*GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality*](https://ojs.aaai.org/index.php/AAAI/article/view/34096). AAAI, 2025. [arXiv:2501.13493](https://arxiv.org/abs/2501.13493).
2. S.-A. Chen et al. *TSMixer: An All-MLP Architecture for Time Series Forecasting*. TMLR, 2023.

---

<div align="center">

**Reproducibility over claims. Evidence over assumptions.**

</div>
