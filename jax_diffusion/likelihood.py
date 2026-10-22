"""Probability-flow ODE and exact (or Hutchinson) log-likelihood."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax, random

from jax_diffusion.autodiff import hutchinson_trace, jacobian_trace_exact
from jax_diffusion.model import score
from jax_diffusion.sde import T_EPS, beta


def ode_drift(params, x, t):
    return -0.5 * beta(t) * (x + score(params, x, t))


def divergence_exact(params, x, t):
    return jacobian_trace_exact(lambda y: ode_drift(params, y, t), x)


def divergence_hutchinson(params, x, t, key, n_probes: int = 4):
    return hutchinson_trace(lambda y: ode_drift(params, y, t), x, key, n_probes)


def log_likelihood(params, key, x0, n_steps: int = 200, exact: bool = True):
    """``log p_0(x0)`` for one sample by integrating the ODE from ``t=eps`` to 1.

    The divergence of the drift is ``tr(d drift / dx)``. Exact trace uses
    ``jacfwd``; otherwise Hutchinson + VJP.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    d = x0.size
    ts = jnp.linspace(T_EPS, 1.0, n_steps + 1)

    def aug(x, t, k):
        dx = ode_drift(params, x, t)
        div = divergence_exact(params, x, t) if exact else divergence_hutchinson(params, x, t, k)
        return dx, div

    def rk4(state, inp):
        t, t_next, k = inp
        h = t_next - t
        x, lp = state
        k1 = aug(x, t, k)
        k2 = aug(x + 0.5 * h * k1[0], t + 0.5 * h, k)
        k3 = aug(x + 0.5 * h * k2[0], t + 0.5 * h, k)
        k4 = aug(x + h * k3[0], t_next, k)
        x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        lp = lp + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return (x, lp), None

    keys = random.split(key, n_steps)
    (x_t, int_div), _ = lax.scan(rk4, (x0, jnp.float32(0.0)), (ts[:-1], ts[1:], keys))
    log_p_t = -0.5 * jnp.sum(x_t ** 2) - 0.5 * d * jnp.log(2 * jnp.pi)
    return log_p_t + int_div
