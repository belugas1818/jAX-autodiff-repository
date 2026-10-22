"""Denoising-score matching loss with optional Jacobian regulariser."""

from __future__ import annotations

import jax.numpy as jnp
from jax import jvp, vmap
from jax import random

from jax_diffusion.model import eps_model
from jax_diffusion.sde import T_EPS, diffuse


def per_example_loss(params, key, x0, jac_weight):
    k_t, k_n, k_v = random.split(key, 3)
    t = random.uniform(k_t, (), minval=T_EPS, maxval=1.0)
    xt, eps = diffuse(k_n, x0, t)

    f = lambda x: eps_model(params, x, t)
    v = random.rademacher(k_v, x0.shape, dtype=x0.dtype)
    eps_hat, jv = jvp(f, (xt,), (v,))

    mse = jnp.mean((eps_hat - eps) ** 2)
    jac_pen = jnp.sum(jv ** 2) / x0.size
    return mse + jac_weight * jac_pen, (mse, jac_pen)


def batch_loss(params, key, x0_batch, jac_weight):
    keys = random.split(key, x0_batch.shape[0])
    losses, (mse, jac) = vmap(per_example_loss, in_axes=(None, 0, 0, None))(
        params, keys, x0_batch, jac_weight)
    return losses.mean(), (mse.mean(), jac.mean())
