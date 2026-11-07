"""derivatives.py -- first and second partial derivatives with jax.grad.

Core idea
---------
jax.grad(f) takes a function f(x1, ..., xn) that returns ONE scalar and gives
back a new function that returns df/dx1 (or the derivatives you ask for with
`argnums`). It does this by reverse-mode autodiff: one forward pass records
the operations, one backward pass applies the chain rule, so the cost of the
whole gradient is a small multiple of the cost of f itself, no matter how
many inputs f has.

Chain rule, written out (this is what the backward pass does)
-------------------------------------------------------------
Let f(x) = g(h(x)), with h: R -> R and g: R -> R.
    df/dx = g'(h(x)) * h'(x)
Autodiff applies this rule to every elementary operation (+, *, exp, log,
sqrt, ...) in your code, so you never write a derivative by hand.

Why this matters for option pricing
-----------------------------------
A pricer V(s0, strike, r, sigma, maturity) is a scalar function of scalars.
The Greeks are its partial derivatives:
    delta = dV/ds0        gamma = d^2V/ds0^2        vega = dV/dsigma
    rho   = dV/dr         theta = -dV/dmaturity
so `named_partials` and `second_partial` below are all you need to get them.

Conventions
-----------
* f must return a scalar (shape ()).
* Arguments you differentiate with respect to must be floats (not ints).
* Argument order is positional; `names` labels that order.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax


def value_and_partials(f: Callable, argnums: int | Sequence[int] = 0) -> Callable:
    """Return a function giving (f(*args), df/d(args[argnums])) in one pass.

    Why one pass: the forward evaluation already computed f(*args), and the
    backward pass reuses the stored intermediates. Calling f and then grad(f)
    separately would run the forward pass twice.
    """
    return jax.value_and_grad(f, argnums=argnums)


def named_partials(f: Callable, names: Sequence[str], *args) -> dict:
    """First partial derivative of f with respect to EVERY positional argument.

    Steps
    -----
    1. n = len(args).
    2. argnums = (0, 1, ..., n - 1) asks jax.grad for all n partials at once.
    3. jax.grad(f, argnums)(*args) returns the tuple
           (df/dx0, df/dx1, ..., df/dx{n-1})
       evaluated at the point `args`.
    4. Zip that tuple with `names` to get a labelled dict.

    Example
    -------
    >>> f = lambda x, y: x * x * y          # f = x^2 * y
    >>> named_partials(f, ("x", "y"), 3.0, 2.0)
    {'x': 12.0, 'y': 9.0}                   # 2*x*y = 12, x^2 = 9
    """
    if len(names) != len(args):
        raise ValueError(
            f"got {len(names)} names but {len(args)} arguments; they must match"
        )
    argnums = tuple(range(len(args)))
    grads = jax.grad(f, argnums=argnums)(*args)
    return dict(zip(names, grads))


def second_partial(f: Callable, i: int, j: int) -> Callable:
    """Return the function (x0, ..., xn) -> d^2 f / (dx_i dx_j).

    Derivation of the construction
    ------------------------------
        d^2 f / (dx_i dx_j) = d/dx_j ( df/dx_i )
    so we differentiate twice, the inner grad with argnums=i, the outer with
    argnums=j:
        second_partial(f, i, j) = grad(grad(f, argnums=i), argnums=j)

    With i = j = 0 on a pricer this is gamma = d^2 V / ds0^2.
    The mixed partial (i != j) is symmetric for smooth f:
        d^2 f / (dx_i dx_j) = d^2 f / (dx_j dx_i).
    """
    return jax.grad(jax.grad(f, argnums=i), argnums=j)
