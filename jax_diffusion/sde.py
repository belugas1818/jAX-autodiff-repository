"""Variance-preserving forward SDE.

    dx = -1/2 beta(t) x dt + sqrt(beta(t)) dW

Closed form:
    x_t = sqrt(abar(t)) x_0 + sqrt(1 - abar(t)) eps
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

BETA_MIN = 0.1
BETA_MAX = 20.0
T_EPS = 1e-3


def validate_time(t, *, allow_zero: bool = False) -> None:
    t_arr = jnp.asarray(t)
    lo = 0.0 if allow_zero else T_EPS
    if not bool(jnp.all((t_arr >= lo) & (t_arr <= 1.0))):
        raise ValueError(f"t must lie in [{lo}, 1], got {t}")


def beta(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def log_alpha_bar(t):
    return -(BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2)


def marginal(t):
    """Mean coefficient and std of x_t given x_0.

    Returns
    -------
    alpha : array
        ``sqrt(abar(t))``
    sigma : array
        ``sqrt(1 - abar(t))``
    """
    lab = log_alpha_bar(t)
    return jnp.exp(0.5 * lab), jnp.sqrt(-jnp.expm1(lab))


def diffuse(key, x0, t):
    """Forward process: add noise to clean data ``x0`` at time ``t``.

    Returns ``(x_t, eps)`` with the same shape as ``x0``.
    """
    eps = random.normal(key, x0.shape)
    a, s = marginal(t)
    return a * x0 + s * eps, eps
