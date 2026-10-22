"""JVP / VJP helpers used by the diffusion loss and likelihood.

These are the production form of the autodiff experiments in jax_test.py.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, jvp, vjp, vmap
from jax import random


def hessian_vector_product(f: Callable, primals, tangents):
    """Forward-over-reverse Hessian-vector product."""
    return jvp(grad(f), primals, tangents)[1]


def hessian(f: Callable):
    return jacfwd(jacrev(f))


def hutchinson_jacobian_frob_sq(f: Callable, x, v):
    """One-probe Hutchinson estimate of ``||Df(x)||_F^2``.

    Returns ``(f(x), sum((J v)^2))``.
    """
    y, jv = jvp(f, (x,), (v,))
    return y, jnp.sum(jv ** 2)


def hutchinson_trace(f: Callable, x, key, n_probes: int = 4):
    """Unbiased trace estimate ``E_v[v^T J v]`` using one VJP per probe."""
    if n_probes < 1:
        raise ValueError("n_probes must be >= 1")
    _, vjp_fn = vjp(f, x)
    vs = random.rademacher(key, (n_probes,) + x.shape, dtype=x.dtype)
    vj = vmap(lambda v: vjp_fn(v)[0])(vs)
    return jnp.mean(jnp.sum(vj * vs, axis=tuple(range(1, vs.ndim))))


def jacobian_trace_exact(f: Callable, x):
    """``tr(Df)`` via forward-mode Jacobian. Suitable for small dims."""
    flat = x.reshape(-1)
    g = lambda y: f(y.reshape(x.shape)).reshape(-1)
    return jnp.trace(jacfwd(g)(flat))
