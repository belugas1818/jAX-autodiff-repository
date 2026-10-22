"""Validated inputs and typed outputs for a diffusion experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

DATA_CHOICES = ("points2d", "signal1d", "csv")


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


@dataclass(frozen=True)
class TrainConfig:
    """Inputs for `run_experiment`.

    Attributes
    ----------
    data:
        Built-in adapter name: ``points2d``, ``signal1d``, or ``csv``.
    csv_path:
        Required when ``data='csv'``. One numeric sample per row.
    steps, batch_size, lr, jac_weight:
        Training hyperparameters. Jacobian weight multiplies the Hutchinson
        estimate of ``||d eps_theta / dx||_F^2 / dim``.
    n_samples:
        Reverse-SDE draws after training. Set to 0 to skip sampling.
    seed:
        JAX PRNG seed.
    hidden, depth, temb:
        MLP width, number of hidden layers, and sinusoidal time-embedding size
        (must be even).
    ema_decay:
        Exponential moving average of weights used for evaluation.
    sample_steps, likelihood_steps:
        Discretization counts for reverse SDE and probability-flow ODE.
    eval_likelihood, eval_denoise:
        Toggle held-out NLL and one-shot denoising reports.
    plot_path:
        Optional PNG of real vs generated samples.
    """

    data: str = "points2d"
    csv_path: str | None = None
    steps: int = 4000
    batch_size: int = 512
    lr: float = 2e-3
    jac_weight: float = 1e-4
    n_samples: int = 2000
    seed: int = 0
    hidden: int = 256
    depth: int = 4
    temb: int = 32
    ema_decay: float = 0.995
    sample_steps: int = 500
    likelihood_steps: int = 200
    nll_test_size: int = 256
    eval_likelihood: bool = True
    eval_denoise: bool = True
    plot_path: str | None = None

    def __post_init__(self) -> None:
        _require(self.data in DATA_CHOICES, f"data must be one of {DATA_CHOICES}")
        if self.data == "csv":
            _require(self.csv_path is not None, "csv_path is required when data='csv'")
            _require(Path(self.csv_path).is_file(), f"csv file not found: {self.csv_path}")
        for name, value in (
            ("steps", self.steps),
            ("batch_size", self.batch_size),
            ("hidden", self.hidden),
            ("depth", self.depth),
            ("temb", self.temb),
            ("sample_steps", self.sample_steps),
            ("likelihood_steps", self.likelihood_steps),
            ("nll_test_size", self.nll_test_size),
        ):
            _require(isinstance(value, int) and value >= 1, f"{name} must be an integer >= 1")
        _require(self.n_samples >= 0, "n_samples must be >= 0")
        _require(self.seed >= 0, "seed must be >= 0")
        _require(self.lr > 0, "lr must be > 0")
        _require(self.jac_weight >= 0, "jac_weight must be >= 0")
        _require(0.0 < self.ema_decay < 1.0, "ema_decay must be in (0, 1)")
        _require(self.temb % 2 == 0, "temb must be even (sin/cos embedding)")


@dataclass
class DenoiseReport:
    """One-shot denoising MSE at a known noise time ``t`` (normalized units)."""

    t: float
    noisy_mse: float
    denoised_mse: float


@dataclass
class ExperimentResult:
    """Outputs of `run_experiment`.

    Arrays are in the **normalized** space used for training unless noted.
    Use ``spec.denormalize`` to return samples to the original data scale.
    """

    spec_name: str
    shape: tuple[int, ...]
    dim: int
    history: list[dict[str, float]]
    params: Any
    ema_params: Any
    samples: jnp.ndarray | None
    real: jnp.ndarray | None
    nll_nats: float | None
    nll_bits_per_dim: float | None
    trace_kind: str | None
    denoise: Sequence[DenoiseReport] = field(default_factory=tuple)

    def summary(self) -> Mapping[str, object]:
        return {
            "data": self.spec_name,
            "shape": self.shape,
            "dim": self.dim,
            "final_loss": self.history[-1]["loss"] if self.history else None,
            "nll_nats": self.nll_nats,
            "nll_bits_per_dim": self.nll_bits_per_dim,
            "trace_kind": self.trace_kind,
            "n_samples": None if self.samples is None else int(self.samples.shape[0]),
        }
