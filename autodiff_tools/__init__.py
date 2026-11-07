"""autodiff_tools -- small, documented JAX autodiff helpers.

Import everything from the package root:
    from autodiff_tools import named_partials, grid2d, greeks_from_price
"""

from .batching import batch, compare_jit_speed, grid2d, replicate_over_keys
from .derivatives import named_partials, second_partial, value_and_partials
from .examples import black_scholes, black_scholes_greeks
from .finite_diff import central_diff, central_second, check_grad
from .greeks import fd_greeks, greeks_from_price
from .jacobians import (
    exact_trace,
    hessian,
    hutchinson_trace,
    hvp,
    jacobian_fwd,
    jacobian_rev,
    jacobian_via_jvp,
    jacobian_via_vjp,
    jvp_at,
    vjp_at,
)

__all__ = [
    "batch",
    "black_scholes",
    "black_scholes_greeks",
    "central_diff",
    "central_second",
    "check_grad",
    "compare_jit_speed",
    "exact_trace",
    "fd_greeks",
    "greeks_from_price",
    "grid2d",
    "hessian",
    "hutchinson_trace",
    "hvp",
    "jacobian_fwd",
    "jacobian_rev",
    "jacobian_via_jvp",
    "jacobian_via_vjp",
    "jvp_at",
    "named_partials",
    "replicate_over_keys",
    "second_partial",
    "value_and_partials",
    "vjp_at",
]
