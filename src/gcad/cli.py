"""Command line interface: ``gcad {train,extract,evaluate,run,sweep,case-study,bench-synthetic,tune}``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .config import ConfigError, load_config
from .data import DataError, DataNotFoundError
from .pipeline import Experiment, run_experiment, run_name, sweep, synthetic_benchmark
from .report import plot_deviation_matrix, plot_sweep
from .utils import ensure_dir, save_json, setup_logging

LOG = logging.getLogger("gcad.cli")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gcad", description="Granger-causality anomaly detection (GCAD)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", required=True, help="YAML config file")
        sp.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="SECTION.KEY=VALUE",
            help="override a config value (repeatable), e.g. --set scoring.beta=0.5 or --set data.synthetic.seed=3",
        )
        sp.add_argument("--seeds", type=int, nargs="+", help="override run.seeds")
        sp.add_argument("--force", action="store_true", help="ignore cached models / graphs")

    for name, text in (
        ("train", "train the predictor"),
        ("extract", "extract causality graphs"),
        ("evaluate", "score and evaluate (single seed)"),
        ("run", "train + extract + evaluate over all seeds"),
        ("case-study", "plot the causal deviation matrix of the most anomalous window"),
    ):
        common(sub.add_parser(name, help=text))

    sw = sub.add_parser("sweep", help="sweep one hyper-parameter (e.g. scoring.threshold)")
    common(sw)
    sw.add_argument("--param", required=True, help="section.key, e.g. scoring.threshold or model.window")
    sw.add_argument("--values", required=True, nargs="+", help="values (parsed as YAML)")

    bench = sub.add_parser(
        "bench-synthetic",
        help="sweep dataset seeds and produce bench.csv + bench.md with sign tests vs pred_error",
    )
    common(bench)
    bench.add_argument(
        "--dataset-seeds",
        nargs="+",
        type=int,
        required=True,
        help="synthetic dataset seeds to sweep over, e.g. --dataset-seeds 0 1 2 3 4 5 6 7 8 9",
    )
    bench.add_argument("--out", default="runs/bench", help="output directory (default: runs/bench)")

    tune_p = sub.add_parser(
        "tune",
        help="label-free scoring hyperparameter tuning via injected val anomalies",
    )
    common(tune_p)
    tune_p.add_argument(
        "--grid",
        nargs="+",
        action="append",
        metavar="KEY V1 V2 ...",
        help="grid axis: KEY followed by values (repeatable), e.g. --grid scoring.threshold 0 0.001 0.01",
    )

    return p


def _dispatch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.run.log_level)
    seeds = args.seeds or cfg.run.seeds
    out = ensure_dir(Path(cfg.run.output_dir) / run_name(cfg))

    if args.command == "run":
        summary = run_experiment(cfg, seeds, force=args.force)
        print((out / "summary.md").read_text())
        return 0 if not summary.empty else 1

    if args.command == "bench-synthetic":
        if cfg.data.name != "synthetic":
            raise ConfigError("bench-synthetic only works with data.name=synthetic")
        df = synthetic_benchmark(cfg, args.dataset_seeds, seeds, str(ensure_dir(Path(args.out))))
        bench_out = Path(args.out)
        print(f"bench.csv: {len(df)} rows written to {bench_out / 'bench.csv'}")
        print((bench_out / "bench.md").read_text())
        return 0

    exp = Experiment(cfg, seeds[0])

    if args.command == "train":
        exp.train(force=args.force)

    elif args.command == "extract":
        exp.extract(force=args.force)

    elif args.command == "evaluate":
        print(yaml.safe_dump(exp.evaluate(), sort_keys=False))

    elif args.command == "case-study":
        res = exp.case_study()
        pairs = plot_deviation_matrix(res["deviation"], exp.data.channel_names, out / "case_study.png")
        save_json(
            {
                "window": res["window"],
                "score": res["score"],
                "top_pairs": [{"cause": a, "effect": b, "deviation": v} for a, b, v in pairs],
            },
            out / "case_study.json",
        )
        print(f"window {res['window']} (score {res['score']:.3f}); top deviating pairs: {pairs}")

    elif args.command == "sweep":
        values = [yaml.safe_load(v) for v in args.values]
        df = sweep(cfg, args.param, values, seeds)
        tag = args.param.replace(".", "_")
        df.to_csv(out / f"sweep_{tag}.csv", index=False)
        plot_sweep(df, args.param, out / f"sweep_{tag}.png")
        print(df[df.variant == "gcad"].groupby("value")[["auroc", "auprc"]].mean().round(4).to_string())

    elif args.command == "tune":
        if not args.grid:
            raise ConfigError("--grid is required for tune; e.g. --grid scoring.threshold 0 0.001 0.01")
        grid: dict[str, list] = {}
        for axis in args.grid:
            if len(axis) < 2:
                raise ConfigError(f"--grid needs KEY followed by at least one value, got: {axis}")
            key = axis[0]
            grid[key] = [yaml.safe_load(v) for v in axis[1:]]
        df = exp.tune(grid)
        df.to_csv(out / "tune.csv", index=False)
        # Best by val AUROC; ties broken by smallest threshold (if present)
        sort_cols = ["auroc"] + (["scoring.threshold"] if "scoring.threshold" in df.columns else [])
        sort_asc = [False] + ([True] if "scoring.threshold" in df.columns else [])
        best = df.sort_values(sort_cols, ascending=sort_asc).iloc[0]
        scoring_overrides = {k.split(".", 1)[1]: best[k] for k in df.columns if k.startswith("scoring.")}
        import yaml as _yaml  # local to avoid shadowing the top-level import at module level
        (out / "tuned_scoring.yaml").write_text(_yaml.safe_dump({"scoring": scoring_overrides}))
        print(f"Best val AUROC: {best['auroc']:.4f}. tuned_scoring.yaml written to {out}")
        print(df.to_string(index=False))

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (ConfigError, DataError, DataNotFoundError) as exc:
        LOG.error("%s", exc)
        return 2
    except FloatingPointError as exc:  # diverged training
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
