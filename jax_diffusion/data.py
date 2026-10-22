"""Data adapters: each yields a fixed-shape float array per sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
from jax import random

Sampler = Callable[[jnp.ndarray, int], jnp.ndarray]


def _check_finite(name: str, x: jnp.ndarray) -> None:
    if not bool(jnp.all(jnp.isfinite(x))):
        raise ValueError(f"{name} contains non-finite values")


@dataclass
class DataSpec:
    """Holds sample shape, a sampler, and per-feature normalisation.

    Inputs
    ------
    name, shape, sampler:
        ``sampler(key, n) -> (n, *shape)`` float array.

    Outputs
    -------
    sample(key, n):
        Normalized draws ``(x - mean) / std``.
    denormalize(x):
        Inverse of that transform.
    """

    name: str
    shape: tuple[int, ...]
    sampler: Sampler
    mean: Optional[jnp.ndarray] = None
    std: Optional[jnp.ndarray] = None

    def fit_normalizer(self, key, n: int = 20000) -> "DataSpec":
        if n < 2:
            raise ValueError("need at least 2 samples to fit a normalizer")
        x = self.sampler(key, n)
        expected = (n,) + self.shape
        if x.shape != expected:
            raise ValueError(f"sampler returned {x.shape}, expected {expected}")
        _check_finite("raw samples", x)
        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-6
        if bool(jnp.any(std <= 0)):
            raise ValueError("normalizer std must be positive")
        self.mean = mean
        self.std = std
        return self

    def sample(self, key, n: int) -> jnp.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("call fit_normalizer before sample()")
        if n < 1:
            raise ValueError("n must be >= 1")
        return (self.sampler(key, n) - self.mean) / self.std

    def denormalize(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("call fit_normalizer before denormalize()")
        return x * self.std + self.mean


def swiss_roll(key, n: int) -> jnp.ndarray:
    k1, k2 = random.split(key)
    t = 1.5 * jnp.pi * (1 + 2 * random.uniform(k1, (n,)))
    x = jnp.stack([t * jnp.cos(t), t * jnp.sin(t)], axis=1) / 10.0
    return x + 0.05 * random.normal(k2, x.shape)


def sinusoids(key, n: int, length: int = 32) -> jnp.ndarray:
    k1, k2, k3 = random.split(key, 3)
    t = jnp.linspace(0.0, 1.0, length)
    freq = random.uniform(k1, (n, 2), minval=1.0, maxval=4.0)
    phase = random.uniform(k2, (n, 2), minval=0.0, maxval=2 * jnp.pi)
    amp = jnp.array([1.0, 0.5])
    sig = (amp * jnp.sin(2 * jnp.pi * freq[:, None, :] * t[None, :, None]
                         + phase[:, None, :])).sum(-1)
    return sig + 0.02 * random.normal(k3, sig.shape)


def csv_sampler(path: str | Path) -> tuple[tuple[int, ...], Sampler]:
    table_np = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if table_np.ndim != 2 or table_np.shape[0] < 1 or table_np.shape[1] < 1:
        raise ValueError(f"CSV must be a 2D numeric table, got shape {table_np.shape}")
    if not np.isfinite(table_np).all():
        raise ValueError("CSV contains non-finite values")
    table = jnp.asarray(table_np)

    def sampler(key, n: int) -> jnp.ndarray:
        idx = random.randint(key, (n,), 0, table.shape[0])
        return table[idx]

    return table.shape[1:], sampler


def make_spec(name: str, csv_path: str | None = None) -> DataSpec:
    if name == "points2d":
        return DataSpec(name, (2,), swiss_roll)
    if name == "signal1d":
        return DataSpec(name, (32,), sinusoids)
    if name == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required for data type 'csv'")
        shape, sampler = csv_sampler(csv_path)
        return DataSpec(name, shape, sampler)
    raise ValueError(f"unknown data type {name}")
