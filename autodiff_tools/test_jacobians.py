import jax
import jax.numpy as jnp
import pytest

from autodiff_tools import (
    exact_trace,
    hessian,
    hutchinson_trace,
    hvp,
    jacobian_fwd,
    jacobian_rev,
    jacobian_via_jvp,
    jacobian_via_vjp,
    jvp_at,
    vjp_at,
)


# g: R^2 -> R^3,  g(x) = [x0*x1, sin(x0), x1^2]
def g(x):
    return jnp.array([x[0] * x[1], jnp.sin(x[0]), x[1] ** 2])


X = jnp.array([0.9, -1.3])


def analytic_jacobian(x):
    return jnp.array(
        [
            [x[1], x[0]],
            [jnp.cos(x[0]), 0.0],
            [0.0, 2 * x[1]],
        ]
    )


def test_jvp_is_jacobian_times_v():
    v = jnp.array([0.4, -2.0])
    y, jv = jvp_at(g, X, v)
    assert jnp.allclose(y, g(X))
    assert jnp.allclose(jv, analytic_jacobian(X) @ v)


def test_vjp_is_u_times_jacobian():
    u = jnp.array([1.0, -0.5, 2.0])
    y, uj = vjp_at(g, X, u)
    assert jnp.allclose(y, g(X))
    assert jnp.allclose(uj, u @ analytic_jacobian(X))


def test_all_four_jacobian_routes_agree():
    j = analytic_jacobian(X)
    assert jnp.allclose(jacobian_fwd(g)(X), j)
    assert jnp.allclose(jacobian_rev(g)(X), j)
    assert jnp.allclose(jacobian_via_jvp(g, X), j)
    assert jnp.allclose(jacobian_via_vjp(g, X), j)


# h(x) = x0^2 * x1 + x1^3  ->  H = [[2*x1, 2*x0], [2*x0, 6*x1]]
def h(x):
    return x[0] ** 2 * x[1] + x[1] ** 3


def test_hessian_and_hvp():
    big_h = jnp.array([[2 * X[1], 2 * X[0]], [2 * X[0], 6 * X[1]]])
    assert jnp.allclose(hessian(h)(X), big_h)
    v = jnp.array([1.0, -2.0])
    assert jnp.allclose(hvp(h, X, v), big_h @ v)


def test_hutchinson_exact_for_diagonal_jacobian():
    # J = diag(d) -> v^T*J*v = sum(d * v^2) = sum(d) for every Rademacher probe
    d = jnp.array([1.0, -2.0, 3.5, 0.25])
    f = lambda x: d * x
    est = hutchinson_trace(f, jnp.ones(4), jax.random.key(0), n_probes=8)
    assert est == pytest.approx(float(jnp.sum(d)), abs=1e-12)


def test_hutchinson_unbiased_for_dense_jacobian():
    a = jax.random.normal(jax.random.key(1), (4, 4))
    f = lambda x: a @ x
    est = hutchinson_trace(f, jnp.ones(4), jax.random.key(2), n_probes=20_000)
    assert est == pytest.approx(float(exact_trace(f, jnp.ones(4))), abs=0.4)
