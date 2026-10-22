import jax.numpy as jnp
from jax import random

from jax_diffusion.likelihood import divergence_exact, divergence_hutchinson, log_likelihood
from jax_diffusion.model import init_params
from jax_diffusion.sampling import denoise_one_shot, sample_reverse_sde
from jax_diffusion.sde import diffuse, marginal


def test_tweedie_formula_with_oracle_noise():
    x0 = random.normal(random.key(0), (7, 2))
    t = 0.2
    xt, eps = diffuse(random.key(1), x0, t)
    a, s = marginal(t)
    x_hat = (xt - s * eps) / a
    assert jnp.allclose(x_hat, x0, rtol=1e-5, atol=1e-5)


def test_reverse_sde_output_shape():
    params = init_params(random.key(0), data_dim=2, hidden=8, depth=1, temb=8)
    samples = sample_reverse_sde(params, random.key(1), (2,), n=5, n_steps=4)
    assert samples.shape == (5, 2)
    assert jnp.all(jnp.isfinite(samples))


def test_denoise_one_shot_shape():
    params = init_params(random.key(0), data_dim=2, hidden=8, depth=1, temb=8)
    xt = random.normal(random.key(2), (6, 2))
    x_hat = denoise_one_shot(params, xt, 0.3)
    assert x_hat.shape == xt.shape


def test_hutchinson_trace_near_exact():
    params = init_params(random.key(0), data_dim=2, hidden=8, depth=1, temb=8)
    x = random.normal(random.key(3), (2,))
    t = 0.5
    exact = divergence_exact(params, x, t)
    est = divergence_hutchinson(params, x, t, random.key(4), n_probes=256)
    assert jnp.isfinite(exact)
    assert abs(float(exact) - float(est)) < 0.75


def test_log_likelihood_is_finite():
    params = init_params(random.key(0), data_dim=2, hidden=8, depth=1, temb=8)
    x0 = random.normal(random.key(5), (2,))
    ll = log_likelihood(params, random.key(6), x0, n_steps=4, exact=True)
    assert jnp.isfinite(ll)
