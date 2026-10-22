"""Command-line entry point."""

from __future__ import annotations

import argparse

from jax_diffusion.config import DATA_CHOICES, TrainConfig
from jax_diffusion.train import run_experiment


def parse_args(argv=None) -> TrainConfig:
    ap = argparse.ArgumentParser(
        description="Train a JAX score-based diffusion model and evaluate it."
    )
    ap.add_argument("--data", default="points2d", choices=DATA_CHOICES)
    ap.add_argument("--csv", default=None, dest="csv_path")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512, dest="batch_size")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--jac-weight", type=float, default=1e-4, dest="jac_weight")
    ap.add_argument("--n-samples", type=int, default=2000, dest="n_samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--plot", default=None, dest="plot_path")
    ns = ap.parse_args(argv)
    return TrainConfig(**vars(ns))


def main(argv=None) -> None:
    config = parse_args(argv)
    result = run_experiment(config)
    summary = result.summary()
    print("\nsummary:", summary)


if __name__ == "__main__":
    main()
