"""Noise-prediction MLP ``eps_theta(x_t, t)`` and the implied score."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import random

from jax_diffusion.sde import marginal


def time_embedding(t, dim: int = 32):
    if dim < 2 or dim % 2 != 0:
        raise ValueError("time embedding dim must be even and >= 2")
    half = dim // 2
    freqs = jnp.exp(-math.log(1000.0) * jnp.arange(half) / half)
    ang = 1000.0 * t * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)])


def init_params(key, data_dim: int, hidden: int = 256, depth: int = 4, temb: int = 32):
    if data_dim < 1:
        raise ValueError("data_dim must be >= 1")
    if hidden < 1 or depth < 1:
        raise ValueError("hidden and depth must be >= 1")
    sizes = [data_dim + temb] + [hidden] * depth + [data_dim]
    keys = random.split(key, len(sizes) - 1)
    params = []
    for k, (n_in, n_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        w = random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        params.append({"W": w, "b": jnp.zeros(n_out)})
    params[-1]["W"] = params[-1]["W"] * 0.01
    return params


def eps_model(params, x, t):
    """Predict noise for one sample ``x`` (any shape) at scalar time ``t``."""
    shape = x.shape
    temb_dim = params[0]["W"].shape[0] - int(x.size)
    h = jnp.concatenate([x.reshape(-1), time_embedding(t, temb_dim)])
    for layer in params[:-1]:
        h = jax.nn.silu(h @ layer["W"] + layer["b"])
    out = h @ params[-1]["W"] + params[-1]["b"]
    return out.reshape(shape)


def score(params, x, t):
    """Score ``grad_x log p_t(x) = -eps_hat / sigma(t)``."""
    _, s = marginal(t)
    return -eps_model(params, x, t) / s
