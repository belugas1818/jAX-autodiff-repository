import jax
import jax.numpy as jnp

from autodiff_tools import batch, compare_jit_speed, grid2d, replicate_over_keys


def f(x, y):
    return x * x + 3.0 * y


def test_grid2d_matches_python_loops():
    xs = jnp.array([1.0, 2.0, 3.0])
    ys = jnp.array([10.0, 20.0])
    got = grid2d(f)(xs, ys)
    want = jnp.array([[f(x, y) for y in ys] for x in xs])
    assert got.shape == (3, 2)
    assert jnp.allclose(got, want)


def test_batch_matches_vmap_with_broadcast_argument():
    xs = jnp.arange(5.0)
    g = batch(f, in_axes=(0, None))
    assert jnp.allclose(g(xs, 2.0), xs * xs + 6.0)


def test_replicate_over_keys_gives_independent_draws():
    draws = replicate_over_keys(lambda k: jax.random.normal(k), jax.random.key(0), 200)
    assert draws.shape == (200,)
    assert jnp.unique(draws).size == 200          # no repeated key
    assert abs(float(jnp.mean(draws))) < 0.35     # mean ~ 0, std error ~ 0.07


def test_compare_jit_speed_reports_positive_times():
    out = compare_jit_speed(lambda x: jnp.sum(jnp.sin(x) ** 2), jnp.arange(1000.0), repeats=3)
    assert set(out) == {"eager_s", "jit_s", "compile_s"}
    assert all(v > 0 for v in out.values())
