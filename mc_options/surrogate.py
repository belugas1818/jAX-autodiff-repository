"""Equinox surrogate pricer: a small MLP that learns Monte Carlo call prices.

Core idea
---------
A Monte Carlo pricer is accurate but slow: every price needs many simulated
paths. A neural network trained on a table of Monte Carlo prices is a smooth
function of its inputs, so after training it prices in one forward pass and
jax.grad of the network gives the Greeks.

The network learns one function of three inputs,

    f(m, sigma, T) = V / K,      m = s0 / K,

where V is the European call price. r and q are fixed (R and Q below).

Why price / K and moneyness are enough
--------------------------------------
Under GBM the call payoff is homogeneous of degree 1 in (s0, K). Write
S_T = s0 * G with G = exp((r - q - 0.5*sigma*sigma)*T + sigma*sqrt(T)*Z), which
does not involve s0 or K. For any c > 0,

    max(c * s0 * G - c * K, 0) = c * max(s0 * G - K, 0),

so V(c * s0, c * K) = c * V(s0, K). Choose c = 1 / K:

    V(s0, K) / K = V(s0 / K, 1) = f(m, sigma, T).

So one function of (m, sigma, T) covers every (s0, K) pair, and the targets
V / K are all of order 0.01 to 0.3, which suits a network.

Requires the ``nn`` extra (equinox, optax). Run ``python -m mc_options.surrogate``
from the repository root to train, print a report, and (with the ``plot``
extra) save out/surrogate.png.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random

from mc_options.pricer import price_grid

# Fixed market constants. The surrogate is a function of (m, sigma, T) only.
S0 = 100.0
R = 0.03
Q = 0.0

# Input box the network is trained on: (moneyness, sigma, maturity).
INPUT_LOW = (0.80, 0.10, 0.25)
INPUT_HIGH = (1.20, 0.40, 2.00)

# Training grid: 21 * 13 * 8 = 2184 points, covering the whole box.
TRAIN_MONEYNESS = tuple(float(x) for x in jnp.linspace(0.80, 1.20, 21))
TRAIN_SIGMAS = tuple(float(x) for x in jnp.linspace(0.10, 0.40, 13))
TRAIN_MATURITIES = tuple(float(x) for x in jnp.linspace(0.25, 2.00, 8))

# Test grid: 9 * 6 * 4 = 216 points that fall between training nodes and stay
# inside the box (the network is not asked to extrapolate).
TEST_MONEYNESS = tuple(float(x) for x in jnp.linspace(0.83, 1.17, 9))
TEST_SIGMAS = tuple(float(x) for x in jnp.linspace(0.125, 0.375, 6))
TEST_MATURITIES = (0.4, 0.8, 1.1, 1.7)


def make_dataset(key, moneyness, sigmas, maturities, n_paths, s0=S0, r=R, q=Q):
    """Monte Carlo call prices on a (maturity, sigma, moneyness) grid.

    Returns (x, y, y_se):
        x    shape (n, 3), rows (m, sigma, T)
        y    shape (n,),   Monte Carlo price / K
        y_se shape (n,),   standard error of y

    Core idea: ``price_grid`` prices a strike-by-sigma grid at one maturity
    with one shared normal sample, so we loop over maturities and fold the
    loop index into the key. Each maturity then gets its own independent
    sample.

    Derivation of the inputs. s0 is fixed, so a moneyness m gives the strike

        m = s0 / K   =>   K = s0 / m.

    ``price_grid`` returns prices with shape (len(sigmas), len(strikes)),
    row i for sigma_i and column j for K_j. The target is

        y_ij = V_ij / K_j

    and the standard error scales the same way, because dividing every
    sample by the constant K_j divides the sample standard deviation by K_j:

        se(V / K) = s(V / K) / sqrt(N) = (s(V) / K) / sqrt(N) = se(V) / K.

    Use different keys for the training and test sets so the test noise is
    independent of the noise the network was fitted to.
    """
    m_axis = jnp.asarray(moneyness)
    sigma_axis = jnp.asarray(sigmas)
    strikes = s0 / m_axis
    xs, ys, ses = [], [], []
    for i, maturity in enumerate(maturities):
        prices, errors = price_grid(
            random.fold_in(key, i),
            s0=s0,
            strikes=strikes,
            sigmas=sigma_axis,
            r=r,
            q=q,
            maturity=float(maturity),
            n_paths=n_paths,
            payoff="call",
        )
        sigma_mesh, m_mesh = jnp.meshgrid(sigma_axis, m_axis, indexing="ij")
        t_mesh = jnp.full_like(m_mesh, float(maturity))
        xs.append(jnp.stack([m_mesh, sigma_mesh, t_mesh], axis=-1).reshape(-1, 3))
        ys.append((prices / strikes[None, :]).reshape(-1))
        ses.append((errors / strikes[None, :]).reshape(-1))
    x = jnp.concatenate(xs).astype(jnp.float32)
    y = jnp.concatenate(ys).astype(jnp.float32)
    y_se = jnp.concatenate(ses).astype(jnp.float32)
    return x, y, y_se


def make_model(key):
    """MLP with 3 inputs, 1 output, 2 hidden layers of width 64, softplus.

    softplus(u) = log(1 + exp(u)) is smooth, so the network is infinitely
    differentiable in its inputs. That matters here: delta and vega are
    jax.grad of the network, and a ReLU network would have a piecewise
    constant gradient. Parameters are float32 so the model is the same
    whether or not jax_enable_x64 is on.
    """
    return eqx.nn.MLP(
        in_size=3,
        out_size=1,
        width_size=64,
        depth=2,
        activation=jax.nn.softplus,
        dtype=jnp.float32,
        key=key,
    )


def describe_model(model) -> str:
    """Show that an Equinox module is a pytree: its treedef and its array leaves.

    eqx.nn.MLP is a dataclass registered with jax.tree_util. Its leaves are
    the weight and bias arrays of each Linear layer plus non-array fields
    (the activation function, the sizes). jax.tree_util.tree_structure shows
    the nesting; jax.tree_util.tree_leaves on the array part lists the
    parameters. The parameter count of a Linear layer from a inputs to b
    outputs is a * b (weight) + b (bias), so this MLP has

        (3 * 64 + 64) + (64 * 64 + 64) + (64 * 1 + 1) = 256 + 4160 + 65 = 4481.
    """
    params = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    lines = [
        "treedef: " + str(jax.tree_util.tree_structure(model)),
        "array leaves:",
    ]
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        lines.append(f"  {jax.tree_util.keystr(path):24s} {tuple(leaf.shape)}")
    lines.append(f"parameter count: {sum(int(leaf.size) for leaf in leaves)}")
    return "\n".join(lines)


def scale_inputs(x):
    """Map the input box to [-1, 1] in every coordinate.

    For coordinate j with range [low_j, high_j],

        u_j = 2 * (x_j - low_j) / (high_j - low_j) - 1.

    At x_j = low_j this is 2 * 0 - 1 = -1 and at x_j = high_j it is
    2 * 1 - 1 = 1. Without it, sigma (range 0.3) and T (range 1.75) would
    enter the first layer on very different scales. The map is affine, so
    jax.grad through it applies the chain rule automatically:

        df/dx_j = df/du_j * du_j/dx_j = df/du_j * 2 / (high_j - low_j).
    """
    low = jnp.asarray(INPUT_LOW, dtype=x.dtype)
    high = jnp.asarray(INPUT_HIGH, dtype=x.dtype)
    return 2.0 * (x - low) / (high - low) - 1.0


def surrogate_price(model, x):
    """Network estimate of f(m, sigma, T) = V / K at one input x = (m, sigma, T).

    The network sees the scaled input from ``scale_inputs`` and returns a
    shape (1,) array; index 0 makes it a scalar so jax.grad applies.
    """
    return model(scale_inputs(x))[0]


def predict(model, x):
    """``surrogate_price`` on every row of x (shape (n, 3)), via jax.vmap."""
    return jax.vmap(lambda row: surrogate_price(model, row))(x)


def loss_fn(model, x, y):
    """Mean squared error on price / K. A pure function of (model, data).

    loss = (1 / n) * sum_i (f_net(x_i) - y_i) * (f_net(x_i) - y_i).

    Pure means no hidden state: the same model and data always give the
    same number. That is what lets eqx.filter_value_and_grad differentiate
    it with respect to the model's array leaves.
    """
    error = predict(model, x) - y
    return jnp.mean(error * error)


def make_train_step(optimizer):
    """Build one Adam update step, compiled with eqx.filter_jit.

    Why the filter_ variants. An Equinox module is a pytree whose leaves mix
    arrays (weights, biases) with non-array Python objects (the softplus
    function, the integer layer sizes). Plain jax.jit would try to trace
    every leaf as an array and fail on the function, and plain jax.grad
    would try to differentiate it. eqx.filter_jit traces the array leaves
    and treats every other leaf as a static compile-time constant;
    eqx.filter_value_and_grad differentiates with respect to the floating
    point array leaves of the first argument only. eqx.filter(model,
    eqx.is_array) gives the same array-only view for optax, and
    eqx.apply_updates adds the updates back onto those leaves.

    One step:
        loss, grads = value and gradient of loss_fn at the current model
        updates, opt_state = Adam(grads, opt_state)
        model = model + updates     (leaf by leaf)
    """

    @eqx.filter_jit
    def train_step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return train_step


def train(model, x, y, n_steps=15_000, learning_rate=1e-2):
    """Full-batch Adam on (x, y). Returns (model, losses).

    The data set is a few thousand rows, so every step uses all of it. The
    learning rate follows a cosine decay from ``learning_rate`` to 0:

        lr(k) = learning_rate * 0.5 * (1 + cos(pi * k / n_steps)),

    which is learning_rate at k = 0 (cos 0 = 1) and 0 at k = n_steps
    (cos pi = -1). Large early steps find the shape; small late steps let
    the fit settle instead of bouncing around the minimum.
    """
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=n_steps)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer)
    losses = []
    for _ in range(n_steps):
        model, opt_state, loss = train_step(model, opt_state, x, y)
        losses.append(loss)
    return model, [float(v) for v in losses]


def rmse(a, b) -> float:
    """Root mean squared error sqrt((1 / n) * sum_i (a_i - b_i) * (a_i - b_i))."""
    diff = jnp.asarray(a) - jnp.asarray(b)
    return float(jnp.sqrt(jnp.mean(diff * diff)))


def surrogate_greeks(model, s0, strike, sigma, maturity):
    """Price, delta and vega of the trained surrogate, in the units of V.

    jax.grad of ``surrogate_price`` with respect to its input vector
    x = (m, sigma, T) gives (df/dm, df/dsigma, df/dT). Convert to the
    Greeks of V.

    The model of the price is

        V(s0, K, sigma, T) = K * f(m, sigma, T),   m = s0 / K.

    Delta. K does not depend on s0, and dm/ds0 = 1 / K, so by the chain rule

        dV/ds0 = K * df/dm * dm/ds0
               = K * df/dm * (1 / K)
               = df/dm.

    So delta is the moneyness derivative of the network, with no rescaling.

    Vega. K and m do not depend on sigma, so

        dV/dsigma = K * df/dsigma.

    Vega is per 1.00 of sigma, the same convention as tests/black_scholes.py.
    The input scaling inside ``surrogate_price`` is part of the function
    being differentiated, so jax.grad already includes its factor
    2 / (high_j - low_j).
    """
    x = jnp.asarray([s0 / strike, sigma, maturity], dtype=jnp.float32)
    f, grad_x = jax.value_and_grad(lambda row: surrogate_price(model, row))(x)
    price = strike * f
    delta = grad_x[0]
    vega = strike * grad_x[1]
    return {"price": float(price), "delta": float(delta), "vega": float(vega)}


def evaluate(model, x_test, y_test, bs_price, bs_greeks, s0=S0, r=R, q=Q):
    """Compare the surrogate with Monte Carlo targets and closed-form Black-Scholes.

    ``bs_price`` and ``bs_greeks`` are the closed-form functions from
    tests/black_scholes.py (passed in so this package does not import from
    the tests). For each test row (m, sigma, T) the strike is K = s0 / m.

    Returns a dict:
        rmse_vs_mc     RMSE of f_net against the Monte Carlo y_test (price / K)
        rmse_vs_bs     RMSE of f_net against bs_price / K
        mc_vs_bs       RMSE of the Monte Carlo targets against bs_price / K
                       (the noise floor of the targets themselves)
        delta_rmse, delta_max_abs   surrogate delta minus BS delta
        vega_rmse, vega_max_abs     surrogate vega minus BS vega (per 1.00 sigma)
    """
    pred = predict(model, x_test)
    bs_targets, d_err, v_err = [], [], []
    for m, sigma, maturity in x_test.tolist():
        strike = s0 / m
        bs_targets.append(bs_price(s0, strike, r, q, sigma, maturity, "call") / strike)
        ref = bs_greeks(s0, strike, r, q, sigma, maturity, "call")
        got = surrogate_greeks(model, s0, strike, sigma, maturity)
        d_err.append(got["delta"] - ref["delta"])
        v_err.append(got["vega"] - ref["vega"])
    bs_targets = jnp.asarray(bs_targets)
    d_err = jnp.asarray(d_err)
    v_err = jnp.asarray(v_err)
    return {
        "rmse_vs_mc": rmse(pred, y_test),
        "rmse_vs_bs": rmse(pred, bs_targets),
        "mc_vs_bs": rmse(y_test, bs_targets),
        "delta_rmse": float(jnp.sqrt(jnp.mean(d_err * d_err))),
        "delta_max_abs": float(jnp.max(jnp.abs(d_err))),
        "vega_rmse": float(jnp.sqrt(jnp.mean(v_err * v_err))),
        "vega_max_abs": float(jnp.max(jnp.abs(v_err))),
    }


def fit_surrogate(seed=0, n_paths=50_000, n_steps=15_000):
    """Build the data, the model and the trained surrogate in one call.

    Three independent keys come from one seed with random.split: one for
    the training prices, one for the test prices, one for the weights.
    Returns (model, losses, (x_train, y_train), (x_test, y_test)).
    """
    train_key, test_key, model_key = random.split(random.key(seed), 3)
    x_train, y_train, _ = make_dataset(
        train_key, TRAIN_MONEYNESS, TRAIN_SIGMAS, TRAIN_MATURITIES, n_paths
    )
    x_test, y_test, _ = make_dataset(
        test_key, TEST_MONEYNESS, TEST_SIGMAS, TEST_MATURITIES, n_paths
    )
    model = make_model(model_key)
    model, losses = train(model, x_train, y_train, n_steps=n_steps)
    return model, losses, (x_train, y_train), (x_test, y_test)


def save_plot(model, losses, bs_price, path="out/surrogate.png", s0=S0, r=R, q=Q):
    """Loss curve and an error heatmap over (moneyness, sigma) at T = 1.

    Left: training MSE per step on a log scale. Right: the signed error
    100 * (f_net - bs_price / K), i.e. the price error per 100 of strike,
    on a diverging blue-gray-red scale centred on 0 (blue: the surrogate is
    below Black-Scholes, red: above). The colour limits are symmetric, so
    gray always means zero error.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    diverging = LinearSegmentedColormap.from_list(
        "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
    )

    m_axis = jnp.linspace(INPUT_LOW[0], INPUT_HIGH[0], 41)
    sigma_axis = jnp.linspace(INPUT_LOW[1], INPUT_HIGH[1], 31)
    sigma_mesh, m_mesh = jnp.meshgrid(sigma_axis, m_axis, indexing="ij")
    x = jnp.stack([m_mesh, sigma_mesh, jnp.ones_like(m_mesh)], axis=-1).reshape(-1, 3)
    pred = predict(model, x.astype(jnp.float32)).reshape(sigma_mesh.shape)
    exact = jnp.asarray(
        [
            [bs_price(s0, s0 / float(m), r, q, float(sig), 1.0, "call") * float(m) / s0
             for m in m_axis.tolist()]
            for sig in sigma_axis.tolist()
        ]
    )
    error = 100.0 * (pred - exact)
    limit = float(jnp.max(jnp.abs(error)))

    fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surface)
    for ax in (ax_loss, ax_err):
        ax.set_facecolor(surface)
        ax.tick_params(colors=muted, labelcolor=ink, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")

    ax_loss.plot(losses, color="#2a78d6", linewidth=2)
    ax_loss.set_yscale("log")
    ax_loss.grid(True, which="major", color=grid, linewidth=0.6)
    ax_loss.set_title("Training loss (MSE on price / K)", color=ink, fontsize=11, loc="left")
    ax_loss.set_xlabel("Adam step", color=ink, fontsize=9)

    image = ax_err.pcolormesh(
        m_axis, sigma_axis, error, cmap=diverging, vmin=-limit, vmax=limit, shading="nearest"
    )
    ax_err.set_title(
        "Surrogate minus Black-Scholes at T = 1 (per 100 of strike)", color=ink, fontsize=11, loc="left"
    )
    ax_err.set_xlabel("moneyness s0 / K", color=ink, fontsize=9)
    ax_err.set_ylabel("sigma", color=ink, fontsize=9)
    bar = fig.colorbar(image, ax=ax_err)
    bar.ax.tick_params(colors=muted, labelcolor=ink, labelsize=9)
    bar.outline.set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)
    return path


def main() -> None:
    """Train, print the model pytree and the evaluation, and save the plot.

    Run from the repository root: the Black-Scholes oracle is imported from
    tests/black_scholes.py, which is not part of the installed package.
    """
    from tests.black_scholes import bs_greeks, bs_price

    start = time.perf_counter()
    model, losses, (x_train, _), (x_test, y_test) = fit_surrogate()
    elapsed = time.perf_counter() - start
    print(describe_model(model))
    print(f"train rows {x_train.shape[0]}, test rows {x_test.shape[0]}, "
          f"data + training {elapsed:.1f}s, final loss {losses[-1]:.3e}")
    report = evaluate(model, x_test, y_test, bs_price, bs_greeks)
    for name, value in report.items():
        print(f"  {name:14s} {value:.6f}")
    try:
        print("saved", save_plot(model, losses, bs_price))
    except ImportError:
        print("matplotlib not installed (pip install -e '.[plot]'); skipped the plot")


if __name__ == "__main__":
    main()
