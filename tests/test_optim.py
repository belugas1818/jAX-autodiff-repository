import jax.numpy as jnp
from jax import random, value_and_grad

from jax_diffusion.optim import adam_init, adam_update


def test_adam_reduces_quadratic():
    def loss(params):
        return jnp.sum(params["w"] ** 2)

    params = {"w": jnp.array([3.0, -4.0])}
    state = adam_init(params)
    for _ in range(40):
        g = value_and_grad(loss)(params)[1]
        params, state = adam_update(params, g, state, lr=0.2)
    assert float(loss(params)) < 1.0


def test_adam_init_matches_tree(random_key=None):
    key = random.key(0) if random_key is None else random_key
    params = {"W": random.normal(key, (3, 2)), "b": jnp.zeros(2)}
    state = adam_init(params)
    assert state["m"]["W"].shape == params["W"].shape
    assert int(state["step"]) == 0
