"""Adam optimiser in pure JAX, plus a clipped cosine-decay train step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

from jax_diffusion.loss import batch_loss


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "step": jnp.array(0)}


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    step = state["step"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2, state["v"], grads)
    mc = 1 - b1 ** step
    vc = 1 - b2 ** step
    params = jax.tree.map(
        lambda p, m, v: p - lr * (m / mc) / (jnp.sqrt(v / vc) + eps),
        params, m, v)
    return params, {"m": m, "v": v, "step": step}


def make_train_step(sample_fn, batch_size, jac_weight, lr_max, total_steps, ema_decay):
    @jit
    def train_step(params, opt_state, ema, key):
        k_data, k_loss = jax.random.split(key)
        x0 = sample_fn(k_data, batch_size)
        (loss, (mse, jac)), grads = value_and_grad(batch_loss, has_aux=True)(
            params, k_loss, x0, jac_weight)
        gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
        grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, 1.0 / (gnorm + 1e-6)), grads)
        frac = opt_state["step"] / total_steps
        lr = lr_max * 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(frac, 1.0)))
        params, opt_state = adam_update(params, grads, opt_state, lr)
        ema = jax.tree.map(lambda e, p: ema_decay * e + (1.0 - ema_decay) * p, ema, params)
        return params, opt_state, ema, loss, mse, jac

    return train_step
