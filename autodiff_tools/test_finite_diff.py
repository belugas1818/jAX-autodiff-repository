import jax.numpy as jnp
import pytest

from autodiff_tools import central_diff, central_second, check_grad


def f(x, y):
    return jnp.exp(x) * jnp.sin(y)


ARGS = (0.4, 1.1)


def test_central_diff_matches_calculus():
    assert central_diff(f, ARGS, 0) == pytest.approx(float(jnp.exp(0.4) * jnp.sin(1.1)), rel=1e-7)
    assert central_diff(f, ARGS, 1) == pytest.approx(float(jnp.exp(0.4) * jnp.cos(1.1)), rel=1e-7)


def test_central_second_matches_calculus():
    assert central_second(f, ARGS, 1) == pytest.approx(float(-jnp.exp(0.4) * jnp.sin(1.1)), rel=1e-5)


def test_check_grad_passes_for_autodiff():
    res = check_grad(f, ARGS, argnum=1)
    assert res["ok"]


def test_check_grad_catches_a_wrong_hand_gradient():
    wrong = lambda x, y: jnp.exp(x) * jnp.sin(y)  # forgot to differentiate sin
    with pytest.raises(AssertionError):
        check_grad(f, ARGS, argnum=1, grad_fn=wrong)
    res = check_grad(f, ARGS, argnum=1, grad_fn=wrong, raise_on_fail=False)
    assert res["ok"] is False
