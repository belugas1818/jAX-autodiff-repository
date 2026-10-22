"""Production tests of the autodiff experiments in jax_test.py."""

import jax.numpy as jnp
from jax import jacfwd, jacrev, jvp, random, vjp, vmap
from jax.test_util import check_grads

from jax_diffusion.autodiff import hessian, hessian_vector_product


def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)


def predict(w, b, inputs):
    return sigmoid(jnp.dot(inputs, w) + b)


def _toy():
    inputs = jnp.array(
        [[0.52, 1.12, 0.77],
         [0.88, -1.08, 0.15],
         [0.52, 0.06, -1.30],
         [0.74, -2.49, 1.39]]
    )
    targets = jnp.array([True, True, False, True])
    key = random.key(0)
    _, w_key, b_key = random.split(key, 3)
    w = random.normal(w_key, (3,))
    b = random.normal(b_key, ())

    def loss(w, b):
        preds = predict(w, b, inputs)
        label_probs = preds * targets + (1 - preds) * (1 - targets)
        return -jnp.sum(jnp.log(label_probs))

    return loss, w, b, inputs


def test_check_grads_second_order():
    loss, w, b, _ = _toy()
    check_grads(loss, (w, b), order=2, eps=1e-3)


def test_jacfwd_matches_jacrev():
    _, w, b, inputs = _toy()
    f = lambda w: predict(w, b, inputs)
    assert jnp.allclose(jacfwd(f)(w), jacrev(f)(w), rtol=1e-5, atol=1e-5)


def test_hvp_matches_full_hessian():
    def f(x):
        return jnp.sum(jnp.tanh(x) ** 2)

    key = random.key(1)
    k1, k2 = random.split(key)
    x = random.normal(k1, (8, 6))
    v = random.normal(k2, (8, 6))
    hvp = hessian_vector_product(f, (x,), (v,))
    full = jnp.tensordot(hessian(f)(x), v, 2)
    assert jnp.allclose(hvp, full, rtol=1e-4, atol=1e-4)


def test_vmap_matches_loop_vjp_and_jvp():
    _, w, b, inputs = _toy()
    f = lambda w: predict(w, b, inputs)
    y, vjp_fun = vjp(f, w)
    key = random.key(2)
    u = random.normal(key, (16,) + y.shape)
    loop = jnp.vstack([jnp.asarray(vjp_fun(mi)) for mi in u])
    vmapped, = vmap(vjp_fun)(u)
    assert jnp.allclose(loop, vmapped, rtol=1e-5, atol=1e-5)

    s = random.normal(key, (16,) + w.shape)
    loop_j = jnp.vstack([jvp(f, (w,), (mi,))[1] for mi in s])
    vmap_j = vmap(lambda mi: jvp(f, (w,), (mi,))[1])(s)
    assert jnp.allclose(loop_j, vmap_j, rtol=1e-5, atol=1e-5)
