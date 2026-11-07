"""batching.py -- vmap and jit helpers.

Core idea
---------
vmap turns a function written for ONE input into a function that runs over a
whole batch, with no Python loop. You write price(s0, strike, ...) for a
single contract and vmap gives you the price of 10,000 contracts. jit then
compiles the batched function with XLA so repeated calls run at compiled speed.

Rules that make jit work
------------------------
* The function must be pure: output depends only on inputs (pass PRNG keys in,
  never use global random state).
* Shapes must not depend on traced VALUES. Python ints that set a shape (such
  as the number of paths) must be marked static or closed over.
* Python `if` on a traced value fails; use jnp.where or jax.lax.cond instead.

in_axes cheat sheet
-------------------
in_axes=0      map over axis 0 of that argument
in_axes=None   do not map; broadcast that argument to every call
"""

from __future__ import annotations

import time
from typing import Callable

import jax
import jax.numpy as jnp


def batch(f: Callable, in_axes=0, compile: bool = True) -> Callable:
    """vmap f over `in_axes`, then jit the result unless compile=False."""
    g = jax.vmap(f, in_axes=in_axes)
    return jax.jit(g) if compile else g


def grid2d(f: Callable, compile: bool = True) -> Callable:
    """Turn a scalar function f(x, y) into G(xs, ys) with G[i, j] = f(xs[i], ys[j]).

    Construction
    ------------
    1. Inner vmap fixes x and maps y:   y -> f(x, y)        in_axes=(None, 0)
       Output shape: (len(ys),).
    2. Outer vmap maps x over the inner function:            in_axes=(0, None)
       Output shape: (len(xs), len(ys)).
    Rows follow xs, columns follow ys. With xs = sigmas and ys = strikes this is
    a price surface.
    """
    inner = jax.vmap(f, in_axes=(None, 0))
    outer = jax.vmap(inner, in_axes=(0, None))
    return jax.jit(outer) if compile else outer


def replicate_over_keys(f: Callable, key: jax.Array, n: int):
    """Run f(key_i) for n independent keys and stack the results.

    jax.random.split(key, n) makes n statistically independent child keys.
    Use it to repeat a whole Monte Carlo experiment n times, for example to
    measure the true spread of an estimator and compare it with its reported
    standard error.
    """
    keys = jax.random.split(key, n)
    return jax.vmap(f)(keys)


def compare_jit_speed(f: Callable, *args, repeats: int = 20) -> dict:
    """Time f eagerly versus jit(f) on the same arguments.

    Returns seconds per call for each, plus the one-off compile time.
    block_until_ready is required because JAX dispatches work asynchronously;
    without it you would time only the launch, not the computation.
    """
    jf = jax.jit(f)

    t0 = time.perf_counter()
    jax.block_until_ready(jf(*args))
    compile_s = time.perf_counter() - t0

    def _time(fn):
        t = time.perf_counter()
        for _ in range(repeats):
            jax.block_until_ready(fn(*args))
        return (time.perf_counter() - t) / repeats

    return {
        "eager_s": _time(f),
        "jit_s": _time(jf),
        "compile_s": compile_s,
    }
