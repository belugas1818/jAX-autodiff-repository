"""Risk-neutral geometric Brownian motion, sampled with an explicit JAX key.

The core idea: Ito's formula turns the GBM SDE into an exact normal draw for
log S, so a path is a product of exact steps rather than an Euler guess.
"""

from __future__ import annotations

import jax.numpy as jnp


def terminal_spot(s0, r, q, sigma, maturity, z):
    """Map one standard normal per path into the spot at time T.

    The risk-neutral SDE with continuous dividend yield q is

        dS_t = (r - q) * S_t * dt + sigma * S_t * dW_t.

    Set X_t = log(S_t). Ito's formula for f(S) = log(S) uses

        f'(S) = 1 / S
        f''(S) = -1 / (S * S)

    so

        dX_t = f'(S_t) * dS_t + 0.5 * f''(S_t) * (sigma * S_t) * (sigma * S_t) * dt
             = (1 / S_t) * ((r - q) * S_t * dt + sigma * S_t * dW_t)
               + 0.5 * (-1 / (S_t * S_t)) * (sigma * sigma * S_t * S_t) * dt
             = (r - q) * dt + sigma * dW_t - 0.5 * sigma * sigma * dt
             = (r - q - 0.5 * sigma * sigma) * dt + sigma * dW_t.

    Integrate from 0 to T. W_T - W_0 has law sqrt(T) * Z with Z ~ N(0, 1),
    and X_0 = log(S_0), therefore

        log(S_T) = log(S_0) + (r - q - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z

    and

        S_T = S_0 * exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z).

    ``z`` has shape (n_paths,) and is treated as the Z draws. Parameters are
    cast to ``z.dtype`` so the expression stays in one precision.
    """
    dtype = z.dtype
    s0 = jnp.asarray(s0, dtype=dtype)
    r = jnp.asarray(r, dtype=dtype)
    q = jnp.asarray(q, dtype=dtype)
    sigma = jnp.asarray(sigma, dtype=dtype)
    maturity = jnp.asarray(maturity, dtype=dtype)
    drift = (r - q - 0.5 * sigma * sigma) * maturity
    shock = sigma * jnp.sqrt(maturity) * z
    return s0 * jnp.exp(drift + shock)


def path_spots(s0, r, q, sigma, maturity, z):
    """Build a grid of exact GBM spots, including the spot at time 0.

    ``z`` has shape (n_paths, n_steps). Step i uses the independent normal
    z[:, i] and covers a time slice of length

        dt = T / n_steps.

    The same Ito calculation as in ``terminal_spot``, applied on [t_i, t_{i+1}],
    gives

        S_{i+1} = S_i * exp((r - q - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z_i).

    Unrolling the product from S_0,

        log(S_k) = log(S_0) + (r - q - 0.5 * sigma * sigma) * (k * dt)
                   + sigma * sqrt(dt) * (Z_0 + ... + Z_{k-1}).

    The sum of the normals is a cumulative sum along the step axis. The
    returned array has shape (n_paths, n_steps + 1): column 0 is S_0 and
    column k is S at time k * dt. The last column is S_T.

    When n_steps = 1 there is a single normal and a single slice dt = T, so
    the last column equals ``terminal_spot`` on that same normal.
    """
    n_steps = z.shape[1]
    dtype = z.dtype
    s0 = jnp.asarray(s0, dtype=dtype)
    r = jnp.asarray(r, dtype=dtype)
    q = jnp.asarray(q, dtype=dtype)
    sigma = jnp.asarray(sigma, dtype=dtype)
    maturity = jnp.asarray(maturity, dtype=dtype)
    dt = maturity / n_steps
    log_step = (r - q - 0.5 * sigma * sigma) * dt + sigma * jnp.sqrt(dt) * z
    log_from_start = jnp.cumsum(log_step, axis=1)
    future = jnp.exp(jnp.log(s0) + log_from_start)
    spot0 = jnp.broadcast_to(s0, (z.shape[0], 1))
    return jnp.concatenate([spot0, future], axis=1)
