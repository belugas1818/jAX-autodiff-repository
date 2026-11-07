"""finite_diff.py -- central finite differences, for checking autodiff and for
payoffs that autodiff cannot differentiate.

Core idea
---------
Autodiff gives exact derivatives of the program you wrote. Finite differences
approximate derivatives from function values only. Use them to
  (1) verify a gradient you wrote or an autodiff result, and
  (2) get Greeks for discontinuous payoffs (digital, barrier), where the
      pathwise derivative is 0 almost everywhere and misses the jump.

Central difference, derived
---------------------------
Taylor expand f around x:
    f(x + h) = f(x) + h*f'(x) + (h^2/2)*f''(x) + (h^3/6)*f'''(x) + ...
    f(x - h) = f(x) - h*f'(x) + (h^2/2)*f''(x) - (h^3/6)*f'''(x) + ...
Subtract:   f(x + h) - f(x - h) = 2*h*f'(x) + (h^3/3)*f'''(x) + ...
Divide by 2*h:
    f'(x) = ( f(x + h) - f(x - h) ) / (2*h) + O(h^2).
Add instead and subtract 2*f(x):
    f(x + h) + f(x - h) - 2*f(x) = h^2*f''(x) + (h^4/12)*f''''(x) + ...
    f''(x) = ( f(x + h) - 2*f(x) + f(x - h) ) / h^2 + O(h^2).

Choosing h
----------
Truncation error shrinks like h^2, rounding error grows like eps/h (first
derivative) or eps/h^2 (second). In float64 (eps ~ 1e-16) h ~ 1e-4 to 1e-3 of
the variable's scale is a good range. In float32 (eps ~ 1e-7) use about 1e-2.
Enable float64 with jax.config.update("jax_enable_x64", True) before any JAX
array is created.

Monte Carlo note: bump the input but keep the SAME random key in both
evaluations (common random numbers). Otherwise sampling noise swamps the
small difference.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp


def _bumped(args: Sequence, argnum: int, delta):
    out = list(args)
    out[argnum] = out[argnum] + delta
    return out


def central_diff(f: Callable, args: Sequence, argnum: int = 0, h: float = 1e-4):
    """( f(x + h) - f(x - h) ) / (2*h) in the argument `argnum`."""
    fp = f(*_bumped(args, argnum, h))
    fm = f(*_bumped(args, argnum, -h))
    return (fp - fm) / (2.0 * h)


def central_second(f: Callable, args: Sequence, argnum: int = 0, h: float = 1e-3):
    """( f(x + h) - 2*f(x) + f(x - h) ) / h^2 in the argument `argnum`."""
    fp = f(*_bumped(args, argnum, h))
    f0 = f(*args)
    fm = f(*_bumped(args, argnum, -h))
    return (fp - 2.0 * f0 + fm) / (h * h)


def check_grad(
    f: Callable,
    args: Sequence,
    argnum: int = 0,
    h: float = 1e-4,
    rtol: float = 1e-4,
    atol: float = 1e-8,
    grad_fn: Callable | None = None,
    raise_on_fail: bool = True,
) -> dict:
    """Compare an analytic gradient with a central difference.

    By default the analytic side is jax.grad(f, argnum). Pass `grad_fn` to test
    a gradient you wrote by hand (it is called as grad_fn(*args)).

    Passes when |analytic - fd| <= atol + rtol * |fd|.
    """
    analytic = (
        grad_fn(*args) if grad_fn is not None else jax.grad(f, argnums=argnum)(*args)
    )
    fd = central_diff(f, args, argnum, h)
    err = jnp.abs(analytic - fd)
    ok = bool(err <= atol + rtol * jnp.abs(fd))
    result = {"analytic": analytic, "finite_diff": fd, "abs_err": err, "ok": ok}
    if raise_on_fail and not ok:
        raise AssertionError(
            f"gradient mismatch: analytic={float(analytic):.10g} "
            f"finite_diff={float(fd):.10g} abs_err={float(err):.3g}"
        )
    return result
