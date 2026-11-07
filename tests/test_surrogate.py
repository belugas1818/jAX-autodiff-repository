"""Equinox surrogate pricer against closed-form Black-Scholes.

Skipped unless the ``nn`` extra is installed: pip install -e ".[nn]".
Training runs once per module (about 35 s on CPU).
"""

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("equinox")
pytest.importorskip("optax")

import equinox as eqx  # noqa: E402

from mc_options import surrogate  # noqa: E402
from tests.black_scholes import bs_greeks, bs_price  # noqa: E402

# Tolerances, justified from measured runs (seeds 0, 1, 2, with and without
# jax_enable_x64):
#
# PRICE_RMSE_TOL is in units of price / K, so 1.5e-3 is 0.15 per 100 of
# strike, about 1.6% of the ATM one-year call (price / K ~ 0.095). Measured
# RMSE against Black-Scholes was 5e-4 to 9e-4. The Monte Carlo targets are
# themselves 3e-4 to 1e-3 away from Black-Scholes (one shared normal sample
# per maturity in price_grid makes the cells of a maturity err together),
# so a tolerance much below 1e-3 would test the noise of the training data,
# not the network. An under-trained network (4000 Adam steps) measured 2.4e-3
# and fails, so the check does catch a bad fit.
#
# MC_RMSE_TOL: the RMSE against the Monte Carlo test targets also contains
# the targets' own error (up to 1e-3 above), so it gets that much extra room:
# 1.5e-3 + 1e-3 = 2.5e-3. Measured: 5e-4 to 1.45e-3.
#
# Delta is dimensionless and lies in [0, 1]. Measured RMSE was 0.008 to 0.009
# and the worst test point 0.023 to 0.034. Delta comes from the gradient of a
# fit, which is always noisier than the fit itself, so allow about 2x:
# RMSE < 0.02 and max < 0.06.
PRICE_RMSE_TOL = 1.5e-3
MC_RMSE_TOL = 2.5e-3
DELTA_RMSE_TOL = 0.02
DELTA_MAX_TOL = 0.06


@pytest.fixture(scope="module")
def fitted():
    model, losses, train_set, test_set = surrogate.fit_surrogate(seed=0)
    x_test, y_test = test_set
    report = surrogate.evaluate(model, x_test, y_test, bs_price, bs_greeks)
    print("\nsurrogate report:", {k: round(v, 6) for k, v in report.items()})
    return model, losses, train_set, test_set, report


def test_model_is_a_pytree_with_the_requested_shape():
    model = surrogate.make_model(jax.random.key(0))
    shapes = [leaf.shape for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))]
    assert shapes == [(64, 3), (64,), (64, 64), (64,), (1, 64), (1,)]
    assert model.activation is jax.nn.softplus
    assert "parameter count: 4481" in surrogate.describe_model(model)


def test_greek_conversion_matches_direct_differentiation_of_v():
    # surrogate_greeks converts df/dm and df/dsigma to dV/ds0 and dV/dsigma.
    # Differentiate V(s0, sigma) = K * f(s0 / K, sigma, T) directly and compare.
    model = surrogate.make_model(jax.random.key(1))
    strike, maturity = 95.0, 0.7

    def value(s0, sigma):
        x = jnp.stack([s0 / strike, sigma, jnp.float32(maturity)])
        return strike * surrogate.surrogate_price(model, x)

    s0, sigma = jnp.float32(100.0), jnp.float32(0.25)
    d_s0, d_sigma = jax.grad(value, argnums=(0, 1))(s0, sigma)
    got = surrogate.surrogate_greeks(model, 100.0, strike, 0.25, maturity)
    assert got["delta"] == pytest.approx(float(d_s0), rel=1e-5, abs=1e-7)
    assert got["vega"] == pytest.approx(float(d_sigma), rel=1e-5, abs=1e-5)


def test_training_reduces_the_loss(fitted):
    _, losses, _, _, _ = fitted
    assert losses[-1] < 1e-2 * losses[0]


def test_surrogate_price_matches_black_scholes(fitted):
    _, _, _, _, report = fitted
    assert report["rmse_vs_bs"] < PRICE_RMSE_TOL
    assert report["rmse_vs_mc"] < MC_RMSE_TOL


def test_surrogate_delta_matches_black_scholes(fitted):
    _, _, _, _, report = fitted
    assert report["delta_rmse"] < DELTA_RMSE_TOL
    assert report["delta_max_abs"] < DELTA_MAX_TOL
