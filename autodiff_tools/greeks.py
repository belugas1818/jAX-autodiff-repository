"""greeks.py -- option Greeks from any scalar pricer, by autodiff or finite differences.

Core idea
---------
Any pricer with the signature
    price_fn(s0, strike, r, sigma, maturity) -> scalar
is a differentiable function of its five inputs, so the Greeks are partial
derivatives that jax.grad computes for you:

    delta = dV/ds0
    gamma = d^2V/ds0^2
    vega  = dV/dsigma             (per 1.00 of sigma; divide by 100 for per 1%)
    rho   = dV/dr                 (per 1.00 of r;     divide by 100 for per 1%)
    theta = -dV/dmaturity         (value change per year as time passes)

Why theta has a minus sign
--------------------------
Time to maturity is T = T_end - t. As calendar time t moves forward, T falls:
    dT/dt = -1
By the chain rule  dV/dt = dV/dT * dT/dt = -dV/dT, and theta is defined as dV/dt.

Monte Carlo estimators (read this before differentiating one)
-------------------------------------------------------------
A Monte Carlo price with a fixed key is a deterministic function of the five
inputs: the same random draws Z are reused, only s0, sigma, ... change. The
derivative of that function is the PATHWISE estimator:
    d/ds0 of mean( exp(-r*T) * payoff(S_T) ) = mean( exp(-r*T) * payoff'(S_T) * dS_T/ds0 )
This is valid when payoff is continuous (call, put, Asian). It is WRONG for
discontinuous payoffs (digital, barrier knock-out): payoff' is 0 almost
everywhere, so the jump is missed and the estimate collapses toward 0. Use
fd_greeks for those, with the same key in every evaluation and a bump large
enough that a visible fraction of paths crosses the jump.

Gamma is a pitfall even for a call or a put. With the draws fixed,
    S_T = s0 * M,   M = exp((r - 0.5*sigma*sigma)*T + sigma*sqrt(T)*Z)
and M does not depend on s0. Away from the single point S_T = K,
    d/ds0 max(S_T - K, 0)     = 1{S_T > K} * dS_T/ds0 = 1{S_T > K} * M
    d^2/ds0^2 max(S_T - K, 0) = d/ds0 ( 1{S_T > K} ) * M + 1{S_T > K} * dM/ds0
                              = 0 * M + 1{S_T > K} * 0
                              = 0
because the indicator is locally constant in s0 and dM/ds0 = 0. So
grad(grad(price_fn)) of a fixed-draw Monte Carlo call or put is exactly 0,
while the true gamma is positive. The curvature lives in the jump of the
indicator, which only shows up when s0 moves by a finite amount. Use
greeks_from_price(price_fn, gamma="delta_fd"), which takes a central
difference of the pathwise delta, for any Monte Carlo pricer. Keep the
default gamma="autodiff" for smooth closed-form pricers such as
black_scholes, where grad(grad(...)) is exact.

Close the key and any other non-differentiated settings (n_paths, steps, q,
payoff type) over the function, so price_fn takes only the five floats.
"""

from __future__ import annotations

from typing import Callable

import jax


GAMMA_MODES = ("autodiff", "delta_fd")


def greeks_from_price(
    price_fn: Callable,
    gamma: str = "autodiff",
    rel_step: float = 1e-2,
) -> Callable:
    """Return greeks(s0, strike, r, sigma, maturity) -> dict via jax.grad.

    The dict has keys: price, delta, gamma, vega, rho, theta.

    Core idea
    ---------
    One value_and_grad call gives the price and all first-order partials
    (argnums 0, 2, 3, 4 = s0, r, sigma, maturity) together. Gamma is the
    second derivative in s0, and there are two ways to get it, chosen by
    the ``gamma`` argument.

    gamma="autodiff" (default)
        gamma = grad(grad(price_fn, 0), 0)
        Exact for a smooth pricer (closed-form Black-Scholes). WRONG for a
        fixed-draw Monte Carlo call or put: it returns exactly 0.

    gamma="delta_fd"
        gamma = (delta(s0 + h) - delta(s0 - h)) / (2 * h),   h = rel_step * s0
        where delta = grad(price_fn, 0) is the pathwise delta. Use this for
        any Monte Carlo pricer whose pathwise delta is valid (call, put, Asian).

    Any other value of ``gamma`` raises ValueError.

    Why autodiff gamma is 0 on a Monte Carlo call
    ---------------------------------------------
    With the normals Z fixed, every path has
        S_T = s0 * M,   M = exp((r - 0.5*sigma*sigma)*T + sigma*sqrt(T)*Z),
    and M does not contain s0, so dS_T/ds0 = M and dM/ds0 = 0. Away from the
    kink S_T = K, the payoff max(S_T - K, 0) is either 0 or S_T - K, and the
    indicator 1{S_T > K} does not change under a small move of s0, so
        d/ds0 max(S_T - K, 0)     = 1{S_T > K} * M
        d^2/ds0^2 max(S_T - K, 0) = (d/ds0 1{S_T > K}) * M + 1{S_T > K} * dM/ds0
                                  = 0 * M + 1{S_T > K} * 0
                                  = 0.
    The price is the discounted mean of these terms, so its second
    derivative is exactly 0 too. (At the kink itself JAX also returns a
    zero second derivative for maximum.) The put is the same with
    1{S_T < K} and a minus sign.

    Why the central difference of delta recovers gamma
    --------------------------------------------------
    The pathwise delta is
        delta(s0) = exp(-r*T) * mean( 1{s0 * M_i > K} * M_i ).
    As s0 moves from s0 - h to s0 + h, path i flips its indicator from 0
    to 1 exactly when
        s0 - h <= K / M_i < s0 + h,
    so
        delta(s0 + h) - delta(s0 - h)
            = exp(-r*T) * mean( 1{s0 - h <= K / M_i < s0 + h} * M_i ).
    That is a count of paths crossing the strike, weighted by M_i. Its
    expectation is the true delta(s0 + h) - delta(s0 - h) of the Black-Scholes
    price, which is smooth. Taylor-expand a smooth delta about s0:
        delta(s0 + h) = delta(s0) + h * gamma(s0) + (h*h / 2) * delta''(s0)
                        + (h*h*h / 6) * delta'''(s0) + ...
        delta(s0 - h) = delta(s0) - h * gamma(s0) + (h*h / 2) * delta''(s0)
                        - (h*h*h / 6) * delta'''(s0) + ...
    Subtract the second line from the first. The delta(s0) and h*h terms
    cancel:
        delta(s0 + h) - delta(s0 - h) = 2 * h * gamma(s0) + (h*h*h / 3) * delta'''(s0) + ...
    Divide by 2 * h:
        (delta(s0 + h) - delta(s0 - h)) / (2 * h) = gamma(s0) + (h*h / 6) * delta'''(s0) + ...
    So the bias is O(h*h). The noise grows as h shrinks, because only a
    fraction of about 2 * h * density of the paths cross the strike: keep
    rel_step around 1e-2 (h = 1 for s0 = 100) and use many paths.

    Both delta evaluations use the same closed-over draws (common random
    numbers), so the difference measures the crossing paths only, not
    fresh sampling noise.

    Parameters
    ----------
    price_fn : price_fn(s0, strike, r, sigma, maturity) -> scalar.
    gamma : "autodiff" or "delta_fd".
    rel_step : relative bump for gamma="delta_fd"; h = rel_step * s0.
               Ignored for gamma="autodiff".
    """
    if gamma not in GAMMA_MODES:
        raise ValueError(f"gamma must be one of {GAMMA_MODES}, got {gamma!r}")

    first_order = jax.value_and_grad(price_fn, argnums=(0, 2, 3, 4))
    delta_fn = jax.grad(price_fn, argnums=0)
    second_fn = jax.grad(delta_fn, argnums=0)

    def gamma_of(s0, strike, r, sigma, maturity):
        """Gamma by the mode chosen in greeks_from_price (see its docstring)."""
        if gamma == "autodiff":
            return second_fn(s0, strike, r, sigma, maturity)
        h = rel_step * s0
        delta_up = delta_fn(s0 + h, strike, r, sigma, maturity)
        delta_down = delta_fn(s0 - h, strike, r, sigma, maturity)
        return (delta_up - delta_down) / (2.0 * h)

    def greeks(s0, strike, r, sigma, maturity):
        price, (delta, rho, vega, d_maturity) = first_order(
            s0, strike, r, sigma, maturity
        )
        gamma_value = gamma_of(s0, strike, r, sigma, maturity)
        return {
            "price": price,
            "delta": delta,
            "gamma": gamma_value,
            "vega": vega,
            "rho": rho,
            "theta": -d_maturity,
        }

    return greeks


def fd_greeks(
    price_fn: Callable,
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
    rel_step: float = 1e-2,
) -> dict:
    """Greeks by central differences. Use for digital and barrier payoffs.

    Bump sizes (one per variable):
        h_s0 = rel_step * s0          h_sigma = rel_step * sigma
        h_T  = rel_step * maturity    h_r     = rel_step * max(|r|, 0.01)
    Formulas (derived in finite_diff.py):
        delta = ( V(s0 + h) - V(s0 - h) ) / (2*h)
        gamma = ( V(s0 + h) - 2*V(s0) + V(s0 - h) ) / h^2
    and likewise for vega, rho and  theta = -dV/dT.

    Remember: price_fn must use the same key on every call (common random
    numbers), and rel_step must be large enough for noise not to dominate
    but small enough for the O(h^2) truncation error to stay small.
    """
    args = [s0, strike, r, sigma, maturity]
    steps = {
        0: rel_step * s0,
        2: rel_step * max(abs(r), 0.01),
        3: rel_step * sigma,
        4: rel_step * maturity,
    }

    def bump(i, d):
        a = list(args)
        a[i] = a[i] + d
        return price_fn(*a)

    v0 = price_fn(*args)

    def first(i):
        h = steps[i]
        return (bump(i, h) - bump(i, -h)) / (2.0 * h)

    h_s = steps[0]
    gamma = (bump(0, h_s) - 2.0 * v0 + bump(0, -h_s)) / (h_s * h_s)
    return {
        "price": v0,
        "delta": first(0),
        "gamma": gamma,
        "vega": first(3),
        "rho": first(2),
        "theta": -first(4),
    }
