import jax.numpy as jnp
import pytest

from autodiff_tools import named_partials, second_partial, value_and_partials


# f(x, y) = x^2 * sin(y)
#   df/dx = 2*x*sin(y)       df/dy = x^2 * cos(y)
#   d2f/dx2 = 2*sin(y)       d2f/dxdy = 2*x*cos(y)
def f(x, y):
    return x * x * jnp.sin(y)


X, Y = 1.7, 0.6


def test_named_partials_match_hand_derivatives():
    p = named_partials(f, ("x", "y"), X, Y)
    assert set(p) == {"x", "y"}
    assert p["x"] == pytest.approx(2 * X * jnp.sin(Y))
    assert p["y"] == pytest.approx(X * X * jnp.cos(Y))


def test_named_partials_rejects_wrong_name_count():
    with pytest.raises(ValueError):
        named_partials(f, ("x",), X, Y)


def test_value_and_partials_returns_value_and_gradient():
    val, (dx, dy) = value_and_partials(f, argnums=(0, 1))(X, Y)
    assert val == pytest.approx(X * X * jnp.sin(Y))
    assert dx == pytest.approx(2 * X * jnp.sin(Y))
    assert dy == pytest.approx(X * X * jnp.cos(Y))


def test_second_partials():
    assert second_partial(f, 0, 0)(X, Y) == pytest.approx(2 * jnp.sin(Y))
    assert second_partial(f, 0, 1)(X, Y) == pytest.approx(2 * X * jnp.cos(Y))
    # mixed partials are symmetric for smooth f
    assert second_partial(f, 0, 1)(X, Y) == pytest.approx(second_partial(f, 1, 0)(X, Y))
    assert second_partial(f, 1, 1)(X, Y) == pytest.approx(-X * X * jnp.sin(Y))
