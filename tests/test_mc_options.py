"""Monte Carlo prices against an independent Black-Scholes formula."""

import math

import jax.numpy as jnp
import pytest
from jax import random

from autodiff_tools import hessian
from mc_options import price_grid, run_pricing
from mc_options.cli import main, parse_args
from mc_options.gbm import path_spots, terminal_spot
from mc_options.payoffs import asian_call, call, down_out_put, put, up_out_call
from mc_options.pricer import discounted_samples, estimate_greeks, estimate_price
from tests.black_scholes import bs_greeks, bs_price

ATM = dict(s0=100.0, strike=100.0, r=0.05, q=0.0, sigma=0.2, maturity=1.0)


def test_terminal_spot_matches_one_step_path():
    key = random.key(0)
    z = random.normal(key, (32,))
    direct = terminal_spot(100.0, 0.05, 0.01, 0.2, 1.5, z)
    on_path = path_spots(100.0, 0.05, 0.01, 0.2, 1.5, z[:, None])[:, -1]
    assert jnp.allclose(direct, on_path, rtol=1e-5, atol=1e-5)


def test_black_scholes_call_put_and_parity():
    call = bs_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    put = bs_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
    assert call == pytest.approx(10.450583572185565, abs=1e-12)
    assert put == pytest.approx(5.573526022256971, abs=1e-12)
    forward_gap = 100.0 * math.exp(0.0) - 100.0 * math.exp(-0.05)
    assert call - put == pytest.approx(forward_gap, abs=1e-12)


def _within_three_se(payoff, **overrides):
    params = {**ATM, **overrides}
    price, error = estimate_price(
        random.key(0),
        n_paths=100_000,
        payoff=payoff,
        **params,
    )
    closed = bs_price(payoff=payoff, **params)
    assert error > 0
    assert abs(price - closed) < 3.0 * error


def test_mc_call_within_three_standard_errors():
    _within_three_se("call")


def test_mc_put_within_three_standard_errors():
    _within_three_se("put")


def test_mc_put_off_the_money_half_year():
    _within_three_se("put", strike=95.0, maturity=0.5)


def test_mc_call_with_dividend_yield():
    _within_three_se("call", q=0.03, sigma=0.25, strike=105.0)


def test_variance_reduction_cuts_the_standard_error():
    kwargs = dict(n_paths=100_000, payoff="call", **ATM)
    plain_price, plain_se = estimate_price(random.key(0), antithetic=False, control_variate=False, **kwargs)
    anti_price, anti_se = estimate_price(random.key(0), antithetic=True, control_variate=False, **kwargs)
    cv_price, cv_se = estimate_price(random.key(0), antithetic=False, control_variate=True, **kwargs)
    both_price, both_se = estimate_price(random.key(0), antithetic=True, control_variate=True, **kwargs)
    closed = bs_price(payoff="call", **ATM)
    antithetic_factor = (plain_se / anti_se) ** 2
    control_factor = (plain_se / cv_se) ** 2
    both_factor = (plain_se / both_se) ** 2
    # Factors are Var(plain) / Var(method) = (SE_plain / SE_method) ** 2.
    # Antithetics and the control are not ordered against each other: the
    # control already removes most of the spot risk, so stacking the mirror
    # paths does not have to beat the control on its own.
    assert antithetic_factor > 1.2
    assert control_factor > 2.0
    assert both_factor > 1.5
    assert anti_se < plain_se
    assert cv_se < plain_se
    assert both_se < plain_se
    for price, error in (
        (plain_price, plain_se),
        (anti_price, anti_se),
        (cv_price, cv_se),
        (both_price, both_se),
    ):
        assert abs(price - closed) < 3.0 * error


def test_standard_error_shrinks_like_inverse_sqrt_paths():
    small_price, small_se = estimate_price(
        random.key(0), n_paths=20_000, payoff="call", antithetic=False, control_variate=False, **ATM
    )
    large_price, large_se = estimate_price(
        random.key(1), n_paths=320_000, payoff="call", antithetic=False, control_variate=False, **ATM
    )
    # 320_000 / 20_000 = 16, and 1 / sqrt(16) = 0.25.
    assert large_se / small_se == pytest.approx(0.25, rel=0.20)
    closed = bs_price(payoff="call", **ATM)
    assert abs(small_price - closed) < 3.0 * small_se
    assert abs(large_price - closed) < 3.0 * large_se


def test_greeks_with_variance_reduction_still_match_black_scholes():
    greeks, method = estimate_greeks(
        random.key(2),
        n_paths=100_000,
        payoff="call",
        antithetic=True,
        control_variate=True,
        **ATM,
    )
    ref = bs_greeks(payoff="call", **ATM)
    assert method.startswith("pathwise")
    assert greeks["delta"] == pytest.approx(ref["delta"], rel=0.02, abs=0.005)
    assert greeks["gamma"] == pytest.approx(ref["gamma"], rel=0.05, abs=0.001)
    assert greeks["vega"] == pytest.approx(ref["vega"], rel=0.03, abs=0.2)


def _finite_difference(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def test_analytic_greeks_match_finite_differences_of_the_closed_form():
    # The Greek formulas are differentiated by hand. This checks each one
    # against a central difference of bs_price, including a nonzero dividend.
    specs = (
        dict(s0=100.0, strike=100.0, r=0.05, q=0.02, sigma=0.2, maturity=1.0, payoff="call"),
        dict(s0=100.0, strike=90.0, r=0.03, q=0.0, sigma=0.3, maturity=0.5, payoff="put"),
        dict(s0=100.0, strike=100.0, r=0.05, q=0.01, sigma=0.2, maturity=1.0, payoff="digital_call"),
    )
    for spec in specs:
        ref = bs_greeks(**spec)

        def price_of(name, value, spec=spec):
            bumped = dict(spec)
            bumped[name] = value
            return bs_price(**bumped)

        assert ref["delta"] == pytest.approx(_finite_difference(lambda s: price_of("s0", s), spec["s0"], 1e-4), rel=1e-4, abs=1e-6)
        assert ref["vega"] == pytest.approx(_finite_difference(lambda s: price_of("sigma", s), spec["sigma"], 1e-5), rel=1e-4, abs=1e-6)
        assert ref["rho"] == pytest.approx(_finite_difference(lambda s: price_of("r", s), spec["r"], 1e-5), rel=1e-4, abs=1e-6)
        d_maturity = _finite_difference(lambda s: price_of("maturity", s), spec["maturity"], 1e-5)
        assert ref["theta"] == pytest.approx(-d_maturity, rel=1e-4, abs=1e-6)
        h = 1e-3
        spot = spec["s0"]
        gamma_fd = (price_of("s0", spot + h) - 2.0 * ref["price"] + price_of("s0", spot - h)) / (h * h)
        assert ref["gamma"] == pytest.approx(gamma_fd, rel=1e-3, abs=1e-6)


def test_pathwise_hessian_of_the_call_is_zero():
    # autodiff_tools.hessian is jacfwd(jacrev). On max(S_T - K, 0)
    # that second derivative is 0, so it is not the reported gamma.
    z = random.normal(random.key(0), (4096,))

    def price(s0):
        samples = discounted_samples(s0, 100.0, 0.05, 0.0, 0.2, 1.0, z, "call", None)
        return jnp.mean(samples)

    pathwise_gamma = hessian(price)(jnp.float32(100.0))
    analytic = bs_greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["gamma"]
    assert float(pathwise_gamma) == pytest.approx(0.0, abs=1e-5)
    assert analytic > 0.01


@pytest.mark.parametrize("payoff", ["call", "put"])
def test_autodiff_greeks_match_black_scholes(payoff):
    greeks, method = estimate_greeks(
        random.key(1),
        n_paths=200_000,
        payoff=payoff,
        **ATM,
    )
    ref = bs_greeks(payoff=payoff, **ATM)
    assert method.startswith("pathwise")
    assert greeks["delta"] == pytest.approx(ref["delta"], rel=0.02, abs=0.005)
    assert greeks["gamma"] == pytest.approx(ref["gamma"], rel=0.05, abs=0.001)
    assert greeks["vega"] == pytest.approx(ref["vega"], rel=0.03, abs=0.2)
    assert greeks["rho"] == pytest.approx(ref["rho"], rel=0.03, abs=0.3)
    assert greeks["theta"] == pytest.approx(ref["theta"], rel=0.05, abs=0.15)


def test_mc_digital_call_within_three_standard_errors():
    _within_three_se("digital_call")


def test_digital_greeks_use_finite_differences():
    greeks, method = estimate_greeks(
        random.key(3),
        n_paths=200_000,
        payoff="digital_call",
        **ATM,
    )
    ref = bs_greeks(payoff="digital_call", **ATM)
    assert method.startswith("finite")
    assert greeks["delta"] == pytest.approx(ref["delta"], rel=0.15, abs=0.005)
    assert greeks["vega"] == pytest.approx(ref["vega"], rel=0.15, abs=0.05)
    # Gamma is a smoke test only. The true ATM digital gamma is about
    # -0.00033, while the finite-difference estimate at 200k paths has a
    # standard deviation of about 0.0004 across keys. The noise is larger
    # than the signal, so no tolerance can tell a correct gamma from 0:
    # the old check (abs=0.002) passed for gamma = 0. Check only that the
    # number is finite and small.
    assert math.isfinite(greeks["gamma"])
    assert abs(greeks["gamma"]) < 0.01


def test_one_step_asian_matches_the_european_call_on_the_same_paths():
    z = random.normal(random.key(4), (128, 1))
    paths = path_spots(100.0, 0.05, 0.0, 0.2, 1.0, z)
    assert jnp.allclose(asian_call(paths, 100.0, None), call(paths, 100.0, None))


def test_a_far_barrier_never_knocks_out_on_these_paths():
    z = random.normal(random.key(5), (256, 16))
    paths = path_spots(100.0, 0.05, 0.0, 0.2, 1.0, z)
    assert jnp.array_equal(up_out_call(paths, 100.0, 1.0e6), call(paths, 100.0, None))
    assert jnp.array_equal(down_out_put(paths, 100.0, 0.01), put(paths, 100.0, None))


def test_path_dependent_prices_sit_below_the_vanilla_contract():
    vanilla, _ = estimate_price(random.key(6), n_paths=80_000, payoff="call", **ATM)
    asian, _ = estimate_price(
        random.key(6), n_paths=80_000, payoff="asian_call", n_steps=32, **ATM
    )
    knocked, _ = estimate_price(
        random.key(6),
        n_paths=80_000,
        payoff="up_out_call",
        n_steps=32,
        barrier=120.0,
        **ATM,
    )
    put_price, _ = estimate_price(random.key(7), n_paths=80_000, payoff="put", **ATM)
    down, _ = estimate_price(
        random.key(7),
        n_paths=80_000,
        payoff="down_out_put",
        n_steps=32,
        barrier=80.0,
        **ATM,
    )
    assert 0.0 < asian < vanilla
    assert 0.0 < knocked < vanilla
    assert 0.0 < down < put_price


def test_price_grid_matches_a_single_estimate_and_black_scholes():
    strikes = [80.0, 100.0, 120.0]
    sigmas = [0.1, 0.3]
    prices, errors = price_grid(
        random.key(0),
        s0=100.0,
        strikes=strikes,
        sigmas=sigmas,
        r=0.05,
        q=0.0,
        maturity=1.0,
        n_paths=200_000,
        payoff="call",
    )
    assert prices.shape == (2, 3)
    assert errors.shape == (2, 3)
    for i, sigma in enumerate(sigmas):
        for j, strike in enumerate(strikes):
            closed = bs_price(100.0, strike, 0.05, 0.0, sigma, 1.0, "call")
            # Six cells at once: allow 4 standard errors so one unlucky cell
            # does not fail a check that each single price still has to pass
            # at 3 standard errors in the tests above.
            assert abs(float(prices[i, j]) - closed) < 4.0 * float(errors[i, j])

    one_price, one_error = price_grid(
        random.key(8),
        s0=100.0,
        strikes=[100.0],
        sigmas=[0.2],
        r=0.05,
        q=0.0,
        maturity=1.0,
        n_paths=20_000,
        payoff="put",
    )
    single, single_error = estimate_price(
        random.key(8),
        n_paths=20_000,
        payoff="put",
        antithetic=False,
        control_variate=False,
        **ATM,
    )
    assert float(one_price[0, 0]) == pytest.approx(single, rel=0, abs=1e-4)
    assert float(one_error[0, 0]) == pytest.approx(single_error, rel=0, abs=1e-4)
    closed_put = bs_price(payoff="put", **ATM)
    assert abs(float(one_price[0, 0]) - closed_put) < 3.0 * float(one_error[0, 0])


def test_cli_put_command_matches_black_scholes(capsys):
    argv = ["--payoff", "put", "-K", "95", "-T", "0.5", "--paths", "200000"]
    config = parse_args(argv)
    assert config.payoff == "put"
    assert config.strike == 95.0
    assert config.maturity == 0.5
    assert config.n_paths == 200_000
    assert config.antithetic and config.control_variate and config.greeks
    main(argv)
    text = capsys.readouterr().out
    assert "put" in text
    assert "price" in text
    assert "delta" in text
    assert "pathwise" in text
    result = run_pricing(parse_args(argv + ["--no-greeks"]))
    closed = bs_price(100.0, 95.0, 0.05, 0.0, 0.2, 0.5, "put")
    assert abs(result.price - closed) < 3.0 * result.stderr


def test_barrier_requires_a_level_on_the_correct_side_of_the_spot():
    with pytest.raises(ValueError, match="barrier is required"):
        estimate_price(random.key(0), n_paths=10, payoff="up_out_call", **ATM)
    with pytest.raises(ValueError, match="above s0"):
        estimate_price(
            random.key(0), n_paths=10, payoff="up_out_call", barrier=90.0, **ATM
        )
    with pytest.raises(ValueError, match="below s0"):
        estimate_price(
            random.key(0), n_paths=10, payoff="down_out_put", barrier=110.0, **ATM
        )
