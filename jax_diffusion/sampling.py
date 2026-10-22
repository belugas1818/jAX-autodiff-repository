"""Reverse SDE sampling and one-shot Tweedie denoising."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax, random, vmap

from jax_diffusion.model import eps_model, score
from jax_diffusion.sde import T_EPS, beta, marginal


def denoise_one_shot(params, xt, t):
    """Recover ``x_0`` from ``x_t`` in one step (Tweedie's formula)."""
    a, s = marginal(t)
    eps_hat = vmap(eps_model, in_axes=(None, 0, None))(params, xt, t)
    return (xt - s * eps_hat) / a


def sample_reverse_sde(params, key, shape, n, n_steps: int = 500):
    """Euler–Maruyama reverse SDE, then a final Tweedie denoise at ``T_EPS``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    k0, key = random.split(key)
    x = random.normal(k0, (n,) + tuple(shape))
    ts = jnp.linspace(1.0, T_EPS, n_steps + 1)
    batched_score = vmap(score, in_axes=(None, 0, None))

    def step(x, inp):
        t, t_next, k = inp
        dt = t - t_next
        b = beta(t)
        drift = -0.5 * b * x - b * batched_score(params, x, t)
        x_mean = x - drift * dt
        noise = random.normal(k, x.shape)
        return x_mean + jnp.sqrt(b * dt) * noise, None

    keys = random.split(key, n_steps)
    x, _ = lax.scan(step, x, (ts[:-1], ts[1:], keys))
    return denoise_one_shot(params, x, T_EPS)
