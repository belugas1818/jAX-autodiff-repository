"""examples.py -- reference functions used to validate the autodiff tools.

Core idea
---------
Black-Scholes is a closed-form, differentiable pricer whose Greeks are also
known in closed form. That gives an oracle: autodiff Greeks of
`black_scholes` must equal `black_scholes_greeks`, and a Monte Carlo pricer's
Greeks must land within sampling error of both.

NOTE for the pricer project: this file is a test fixture for these TOOLS. The
pricer's own tests should still write the closed form independently so the
pricer is not validated against code it shares.

Formulas (European option, continuous dividend yield q)
-------------------------------------------------------
    d1 = ( ln(s0/K) + (r - q + 0.5*sigma^2)*T ) / ( sigma*sqrt(T) )
    d2 = d1 - sigma*sqrt(T)
    call = s0*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
    put  = K*exp(-r*T)*N(-d2) - s0*exp(-q*T)*N(-d1)
with N the standard normal CDF and phi its density.

Call Greeks (derivations follow from d(d1)/d(s0) = 1/(s0*sigma*sqrt(T)) and the
identity s0*exp(-q*T)*phi(d1) = K*exp(-r*T)*phi(d2)):
    delta = exp(-q*T) * N(d1)
    gamma = exp(-q*T) * phi(d1) / ( s0 * sigma * sqrt(T) )
    vega  = s0 * exp(-q*T) * phi(d1) * sqrt(T)
    rho   = K * T * exp(-r*T) * N(d2)
    theta = -s0*exp(-q*T)*phi(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2) + q*s0*exp(-q*T)*N(d1)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm


def _d1_d2(s0, strike, r, sigma, maturity, q):
    vol_sqrt_t = sigma * jnp.sqrt(maturity)
    d1 = (jnp.log(s0 / strike) + (r - q + 0.5 * sigma * sigma) * maturity) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def black_scholes(s0, strike, r, sigma, maturity, q=0.0, is_call=True):
    """European option price. Differentiable in all five float inputs."""
    d1, d2 = _d1_d2(s0, strike, r, sigma, maturity, q)
    disc_s = s0 * jnp.exp(-q * maturity)
    disc_k = strike * jnp.exp(-r * maturity)
    if is_call:
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)


def black_scholes_greeks(s0, strike, r, sigma, maturity, q=0.0, is_call=True):
    """Closed-form Greeks (call, or put via parity relations)."""
    d1, d2 = _d1_d2(s0, strike, r, sigma, maturity, q)
    sqrt_t = jnp.sqrt(maturity)
    eq = jnp.exp(-q * maturity)
    er = jnp.exp(-r * maturity)
    pdf1 = norm.pdf(d1)

    gamma = eq * pdf1 / (s0 * sigma * sqrt_t)
    vega = s0 * eq * pdf1 * sqrt_t
    common_theta = -s0 * eq * pdf1 * sigma / (2.0 * sqrt_t)

    if is_call:
        delta = eq * norm.cdf(d1)
        rho = strike * maturity * er * norm.cdf(d2)
        theta = common_theta - r * strike * er * norm.cdf(d2) + q * s0 * eq * norm.cdf(d1)
    else:
        delta = -eq * norm.cdf(-d1)
        rho = -strike * maturity * er * norm.cdf(-d2)
        theta = common_theta + r * strike * er * norm.cdf(-d2) - q * s0 * eq * norm.cdf(-d1)

    return {
        "price": black_scholes(s0, strike, r, sigma, maturity, q, is_call),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "rho": rho,
        "theta": theta,
    }
