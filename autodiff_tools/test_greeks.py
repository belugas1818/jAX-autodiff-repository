import jax
import jax.numpy as jnp
import pytest

from autodiff_tools import (
    black_scholes,
    black_scholes_greeks,
    fd_greeks,
    greeks_from_price,
)

KEYS = ("price", "delta", "gamma", "vega", "rho", "theta")
POINTS = [
    (100.0, 100.0, 0.05, 0.20, 1.0),   # at the money
    (90.0, 100.0, 0.03, 0.35, 0.5),    # out of the money, short dated
    (120.0, 100.0, 0.01, 0.15, 2.0),   # in the money, long dated
]


@pytest.mark.parametrize("args", POINTS)
def test_autodiff_greeks_match_closed_form_call(args):
    auto = greeks_from_price(black_scholes)(*args)
    exact = black_scholes_greeks(*args)
    for k in KEYS:
        assert float(auto[k]) == pytest.approx(float(exact[k]), rel=1e-9, abs=1e-12), k


@pytest.mark.parametrize("args", POINTS)
def test_autodiff_greeks_match_closed_form_put(args):
    put = lambda s0, k, r, sig, t: black_scholes(s0, k, r, sig, t, is_call=False)
    auto = greeks_from_price(put)(*args)
    exact = black_scholes_greeks(*args, is_call=False)
    for k in KEYS:
        assert float(auto[k]) == pytest.approx(float(exact[k]), rel=1e-9, abs=1e-12), k


def test_dividend_yield_greeks():
    q = 0.02
    price = lambda s0, k, r, sig, t: black_scholes(s0, k, r, sig, t, q=q)
    args = POINTS[0]
    auto = greeks_from_price(price)(*args)
    exact = black_scholes_greeks(*args, q=q)
    for k in KEYS:
        assert float(auto[k]) == pytest.approx(float(exact[k]), rel=1e-9, abs=1e-12), k


def test_put_call_parity_for_delta():
    # delta_call - delta_put = exp(-q*T); here q = 0
    args = POINTS[0]
    call = black_scholes_greeks(*args)["delta"]
    put = black_scholes_greeks(*args, is_call=False)["delta"]
    assert float(call - put) == pytest.approx(float(jnp.exp(-0.0 * args[4])), rel=1e-12)


def test_fd_greeks_agree_with_autodiff_on_smooth_pricer():
    args = POINTS[0]
    auto = greeks_from_price(black_scholes)(*args)
    fd = fd_greeks(black_scholes, *args, rel_step=1e-3)
    for k in KEYS:
        assert float(fd[k]) == pytest.approx(float(auto[k]), rel=1e-4, abs=1e-8), k


# --- pathwise Monte Carlo Greeks with a fixed key (common random numbers) ---
N_PATHS = 400_000


def make_mc_call(key):
    z = jax.random.normal(key, (N_PATHS,))

    def price(s0, strike, r, sigma, maturity):
        drift = (r - 0.5 * sigma * sigma) * maturity
        s_t = s0 * jnp.exp(drift + sigma * jnp.sqrt(maturity) * z)
        return jnp.exp(-r * maturity) * jnp.mean(jnp.maximum(s_t - strike, 0.0))

    return price


def test_pathwise_mc_greeks_within_sampling_error():
    args = POINTS[0]
    mc = jax.jit(greeks_from_price(make_mc_call(jax.random.key(7))))
    est = mc(*args)
    exact = black_scholes_greeks(*args)
    # tolerances ~ 5 standard errors at 400k paths
    assert float(est["price"]) == pytest.approx(float(exact["price"]), abs=0.12)
    assert float(est["delta"]) == pytest.approx(float(exact["delta"]), abs=0.006)
    assert float(est["vega"]) == pytest.approx(float(exact["vega"]), abs=0.6)
    assert float(est["rho"]) == pytest.approx(float(exact["rho"]), abs=0.6)


def test_autodiff_gamma_is_exactly_zero_on_fixed_draw_mc_call():
    # S_T = s0 * M with M free of s0, so d^2/ds0^2 max(S_T - K, 0) = 0 on
    # every path. The default gamma="autodiff" therefore returns exactly 0,
    # while the true gamma is clearly positive.
    args = POINTS[0]
    mc = jax.jit(greeks_from_price(make_mc_call(jax.random.key(7))))
    est = mc(*args)
    exact = black_scholes_greeks(*args)
    assert float(est["gamma"]) == 0.0
    assert float(exact["gamma"]) > 0.01


def test_delta_fd_gamma_matches_black_scholes_on_mc_call():
    # gamma = (delta(s0 + h) - delta(s0 - h)) / (2 * h), h = 0.01 * s0 = 1.
    # About 2 * h * density = 4% of the 400k paths cross the strike, so the
    # relative standard error is about 1 / sqrt(0.04 * 400_000) = 0.8%;
    # the O(h*h) bias is far smaller. 5% is about 6 standard errors.
    args = POINTS[0]
    mc = jax.jit(
        greeks_from_price(make_mc_call(jax.random.key(7)), gamma="delta_fd", rel_step=1e-2)
    )
    est = mc(*args)
    exact = black_scholes_greeks(*args)
    assert float(est["gamma"]) == pytest.approx(float(exact["gamma"]), rel=0.05)
    # The first-order Greeks are unchanged by the gamma mode.
    plain = jax.jit(greeks_from_price(make_mc_call(jax.random.key(7))))(*args)
    for k in ("price", "delta", "vega", "rho", "theta"):
        assert float(est[k]) == pytest.approx(float(plain[k]), rel=1e-12), k


def test_invalid_gamma_mode_raises():
    with pytest.raises(ValueError, match="gamma must be one of"):
        greeks_from_price(black_scholes, gamma="hessian")
