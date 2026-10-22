import jax.numpy as jnp
from jax import random

from jax_diffusion.loss import batch_loss
from jax_diffusion.model import eps_model, init_params, score, time_embedding
from jax_diffusion.sde import marginal


def test_time_embedding_shape():
    emb = time_embedding(0.3, dim=32)
    assert emb.shape == (32,)
    assert jnp.all(jnp.isfinite(emb))


def test_eps_model_shape_and_score_relation():
    key = random.key(0)
    params = init_params(key, data_dim=2, hidden=16, depth=2, temb=8)
    x = jnp.array([0.1, -0.2])
    t = 0.4
    eps = eps_model(params, x, t)
    assert eps.shape == (2,)
    _, s = marginal(t)
    assert jnp.allclose(score(params, x, t), -eps / s)


def test_batch_loss_finite():
    key = random.key(3)
    params = init_params(key, data_dim=2, hidden=8, depth=1, temb=8)
    x0 = random.normal(random.key(4), (16, 2))
    loss, (mse, jac) = batch_loss(params, random.key(5), x0, jac_weight=1e-4)
    assert jnp.isfinite(loss)
    assert mse >= 0
    assert jac >= 0
