import jax.numpy as jnp
import pytest
from jax import random

from jax_diffusion.sde import (
    BETA_MAX,
    BETA_MIN,
    beta,
    diffuse,
    log_alpha_bar,
    marginal,
    validate_time,
)


def test_marginal_at_endpoints():
    a0, s0 = marginal(0.0)
    assert float(a0) == pytest.approx(1.0, abs=1e-6)
    assert float(s0) == pytest.approx(0.0, abs=1e-6)
    a1, s1 = marginal(1.0)
    assert float(a1) < float(s1)
    assert 0.0 < float(a1) < 1.0
    assert 0.0 < float(s1) < 1.0


def test_log_alpha_bar_derivative_is_minus_beta():
    t = jnp.linspace(0.0, 1.0, 16)
    # d/dt log abar = -(beta_min + (beta_max-beta_min) t) = -beta(t)
    analytic = -(BETA_MIN + (BETA_MAX - BETA_MIN) * t)
    numeric = jnp.gradient(log_alpha_bar(t), t)
    assert jnp.allclose(numeric[1:-1], analytic[1:-1], rtol=1e-3, atol=1e-3)
    assert jnp.allclose(analytic, -beta(t))


def test_diffuse_inverts_with_true_eps():
    key = random.key(0)
    x0 = random.normal(key, (8, 2))
    t = 0.37
    xt, eps = diffuse(random.key(1), x0, t)
    a, s = marginal(t)
    recovered = (xt - s * eps) / a
    assert jnp.allclose(recovered, x0, rtol=1e-5, atol=1e-5)


def test_invalid_time_rejected():
    with pytest.raises(ValueError):
        validate_time(0.0)
    with pytest.raises(ValueError):
        validate_time(1.1)
