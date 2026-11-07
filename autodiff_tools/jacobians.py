"""jacobians.py -- jvp, vjp, jacfwd, jacrev, Hessians and trace estimation.

Core idea
---------
For a vector function f: R^n -> R^m the Jacobian J is the m-by-n matrix
    J[a, b] = d f_a / d x_b.
JAX never has to build J to use it:
    jvp  computes   J * v   (one forward pass, v in R^n)   -> "push a tangent forward"
    vjp  computes   u^T * J (one backward pass, u in R^m)  -> "pull a cotangent back"
Building the full J costs n jvp calls (forward mode, jacfwd) or m vjp calls
(reverse mode, jacrev). So:
    n small (few inputs, many outputs)  -> jacfwd
    m small (many inputs, few outputs)  -> jacrev   (a loss: m = 1)

Definitions used below
----------------------
Directional derivative of f at x in direction v:
    D_v f(x) = lim_{e -> 0} ( f(x + e*v) - f(x) ) / e = J(x) * v
jax.jvp(f, (x,), (v,)) returns the pair ( f(x), J(x) * v ).

Hessian-vector product (HVP), derived
-------------------------------------
For a scalar f, the gradient g(x) = grad f(x) maps R^n -> R^n, and its
Jacobian is the Hessian H(x). Therefore
    H(x) * v = (Jacobian of g at x) * v = jvp(g, (x,), (v,))[1].
This is forward-over-reverse: jvp of grad. It costs a small multiple of one
gradient and never forms the n-by-n matrix H.

Hutchinson trace estimator, derived
-----------------------------------
Let v have independent entries with E[v_i] = 0 and E[v_i^2] = 1
(for example Rademacher: +1 or -1 with probability 1/2 each). Then
    E[ v^T * J * v ] = sum_{a,b} J[a,b] * E[v_a * v_b]
                     = sum_a J[a,a] * E[v_a^2] + sum_{a != b} J[a,b] * E[v_a]*E[v_b]
                     = sum_a J[a,a] * 1 + 0
                     = trace(J).
Each probe needs one jvp (to get J*v) and one dot product with v, so the
trace costs n_probes jvp calls instead of n.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------
# Single-direction products (the two primitives)
# --------------------------------------------------------------------------
def jvp_at(f: Callable, x: jnp.ndarray, v: jnp.ndarray):
    """Return (f(x), J(x) * v). Forward mode; v has the shape of x."""
    return jax.jvp(f, (x,), (v,))


def vjp_at(f: Callable, x: jnp.ndarray, u: jnp.ndarray):
    """Return (f(x), u^T * J(x)). Reverse mode; u has the shape of f(x).

    jax.vjp(f, x) returns f(x) and a function that maps a cotangent u to
    u^T * J. We call that function once with u.
    """
    y, pullback = jax.vjp(f, x)
    (u_times_j,) = pullback(u)
    return y, u_times_j


# --------------------------------------------------------------------------
# Full Jacobians
# --------------------------------------------------------------------------
def jacobian_fwd(f: Callable) -> Callable:
    """Full Jacobian by forward mode: one jvp per input coordinate."""
    return jax.jacfwd(f)


def jacobian_rev(f: Callable) -> Callable:
    """Full Jacobian by reverse mode: one vjp per output coordinate."""
    return jax.jacrev(f)


def jacobian_via_jvp(f: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """Build J column by column, to show what jacfwd does internally.

    Steps
    -----
    1. Let e_b be the b-th standard basis vector (1 in slot b, 0 elsewhere).
    2. J * e_b is column b of J, because (J * e_b)[a] = sum_c J[a,c]*e_b[c] = J[a,b].
    3. vmap over the n basis vectors gives all n columns as rows of an
       (n, m) array; transpose to get the (m, n) Jacobian.
    """
    n = x.size
    basis = jnp.eye(n, dtype=x.dtype).reshape((n,) + x.shape)
    columns = jax.vmap(lambda e: jax.jvp(f, (x,), (e,))[1])(basis)  # (n, m)
    return columns.reshape(n, -1).T


def jacobian_via_vjp(f: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """Build J row by row, to show what jacrev does internally.

    Steps
    -----
    1. Let u_a be the a-th standard basis vector in R^m.
    2. u_a^T * J is row a of J, because (u_a^T * J)[b] = sum_c u_a[c]*J[c,b] = J[a,b].
    3. vmap over the m basis vectors gives all m rows.
    """
    y, pullback = jax.vjp(f, x)
    m = y.size
    basis = jnp.eye(m, dtype=y.dtype).reshape((m,) + y.shape)
    rows = jax.vmap(lambda u: pullback(u)[0])(basis)  # (m, *x.shape)
    return rows.reshape(m, -1)


# --------------------------------------------------------------------------
# Second order
# --------------------------------------------------------------------------
def hessian(f: Callable) -> Callable:
    """Full Hessian of a scalar f: jacfwd of jacrev.

    Reverse mode for the inner gradient (1 output, so 1 backward pass), then
    forward mode over its n outputs-by-n inputs square Jacobian.
    """
    return jax.jacfwd(jax.jacrev(f))


def hvp(f: Callable, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Hessian-vector product H(x) * v without forming H. See module docstring."""
    return jax.jvp(jax.grad(f), (x,), (v,))[1]


# --------------------------------------------------------------------------
# Jacobian trace (divergence), exact and stochastic
# --------------------------------------------------------------------------
def exact_trace(f: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """trace(J(x)) for f: R^n -> R^n by building the full Jacobian. Cost: n jvps."""
    return jnp.trace(jax.jacfwd(f)(x))


def hutchinson_trace(
    f: Callable, x: jnp.ndarray, key: jax.Array, n_probes: int = 64
) -> jnp.ndarray:
    """Unbiased estimate of trace(J(x)) with Rademacher probes. See module docstring.

    Each probe v gives the sample v^T * (J*v); the estimate is their mean.
    The estimator is exact when J is diagonal (v_a^2 = 1 for every a).
    """
    probes = jax.random.rademacher(key, (n_probes,) + x.shape, dtype=x.dtype)

    def one_probe(v):
        _, jv = jax.jvp(f, (x,), (v,))
        return jnp.vdot(v, jv)

    return jnp.mean(jax.vmap(one_probe)(probes))
