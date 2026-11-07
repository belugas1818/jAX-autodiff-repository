"""Average discounted payoffs to estimate a price, its error, and its Greeks.

The core idea: with the random draws held fixed, the Monte Carlo price is an
ordinary function of the market parameters. jax.grad differentiates that
function. A pathwise second derivative does not see a kink, so gamma is a
central difference of the pathwise delta.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import grad, jit, lax, random, vmap

from mc_options.config import OptionConfig, OptionResult
from mc_options.gbm import path_spots, terminal_spot
from mc_options.payoffs import BARRIER_PAYOFFS, PATHWISE, PAYOFFS

# These payoffs read only S_T, so one exact normal is the whole path.
TERMINAL_PAYOFFS = ("call", "put", "digital_call")


def standard_error(samples, antithetic=False):
    """Standard error of the Monte Carlo mean.

    Plain case. Let X_1, ..., X_N be independent discounted payoffs and

        X_bar = (1 / N) * (X_1 + ... + X_N)

        s * s = (1 / (N - 1)) * ((X_1 - X_bar) ** 2 + ... + (X_N - X_bar) ** 2).

    The N - 1 corrects for using X_bar inside s. Independence gives
    Var(X_bar) = Var(X_1) / N, so

        SE = s / sqrt(N).

    Antithetic case. The vector is stored as (X(Z_1), ..., X(Z_M), X(-Z_1),
    ..., X(-Z_M)) with N = 2 * M. X(Z_i) and X(-Z_i) are not independent, so
    feeding all N terms to the plain formula understates the variance. The
    pair means

        P_i = 0.5 * (X(Z_i) + X(-Z_i))

    are independent, and the price we report is their average (which equals
    the average of all N adjusted payoffs). Apply the plain formula to the
    M numbers P_i:

        SE = s_P / sqrt(M).

    Why P_i has the same mean as X. -Z has the same law as Z, so
    E[X(-Z)] = E[X(Z)] and E[P] = E[X].

    Why the variance can fall. With Corr the correlation of X(Z) and X(-Z),

        Var(P) = 0.5 * Var(X) * (1 + Corr)
        Var(mean of M pairs) = Var(P) / M = Var(X) * (1 + Corr) / N.

    The plain estimator on N independent draws has variance Var(X) / N.
    The ratio of the two variances is 1 + Corr. A negative correlation, which
    is what negating a shock tends to produce, makes this ratio smaller than
    1. The variance reduction factor reported in the tests is the reciprocal
    (SE_plain / SE_antithetic) ** 2.
    """
    if antithetic:
        half = samples.shape[0] // 2
        samples = 0.5 * (samples[:half] + samples[half:])
    n = samples.shape[0]
    return jnp.std(samples, ddof=1) / jnp.sqrt(n)


def _paths_for_payoff(s0, r, q, sigma, maturity, z, payoff):
    """Return a path array whose last column is S_T.

    For a terminal payoff the user-facing formula is a single factor

        S_T = S_0 * exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z)

    so ``z`` is shaped (n_paths,) and the path is the column [S_T]. Path
    dependent payoffs pass ``z`` shaped (n_paths, n_steps) and use every step.
    """
    if payoff in TERMINAL_PAYOFFS:
        spot_T = terminal_spot(s0, r, q, sigma, maturity, z)
        return spot_T[:, None]
    return path_spots(s0, r, q, sigma, maturity, z)


def discounted_samples(s0, strike, r, q, sigma, maturity, z, payoff, barrier):
    """Discounted payoff on every path.

    The risk-neutral price is the discounted expectation

        V = exp(-r * T) * E[payoff].

    Replacing the expectation by the sample mean makes the per-path terms

        X_i = exp(-r * T) * payoff(path_i).

    Their average is the Monte Carlo estimate of V. ``barrier`` is forwarded
    to the payoff; vanilla call and put ignore it.
    """
    paths = _paths_for_payoff(s0, r, q, sigma, maturity, z, payoff)
    dtype = paths.dtype
    strike = jnp.asarray(strike, dtype=dtype)
    r = jnp.asarray(r, dtype=dtype)
    maturity = jnp.asarray(maturity, dtype=dtype)
    undiscounted = PAYOFFS[payoff](paths, strike, barrier)
    return jnp.exp(-r * maturity) * undiscounted


def draw_normals(key, n_paths, n_steps, antithetic):
    """Draw the normals for one batch, optionally with the antithetic mirror.

    Without antithetics this is one ``random.normal`` call of shape
    (n_paths,) when n_steps is None (terminal payoff) and (n_paths, n_steps)
    otherwise. The key is the only source of randomness.

    With antithetics, n_paths must be even. Draw M = n_paths / 2 normals
    Z_1, ..., Z_M and stack

        (Z_1, ..., Z_M, -Z_1, ..., -Z_M).

    -Z is standard normal whenever Z is, because the N(0, 1) density
    (1 / sqrt(2 * pi)) * exp(-0.5 * z * z) is even. For a path, every step
    of a scenario is negated together, so the mirror path is the GBM path
    driven by -dW.
    """
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic is True")
    if n_steps is None:
        shape = (n_paths // 2,) if antithetic else (n_paths,)
        z = random.normal(key, shape)
        if antithetic:
            z = jnp.concatenate([z, -z], axis=0)
        return z
    shape = (n_paths // 2, n_steps) if antithetic else (n_paths, n_steps)
    z = random.normal(key, shape)
    if antithetic:
        z = jnp.concatenate([z, -z], axis=0)
    return z


def apply_control_variate(discounted_payoff, spot_T, s0, r, q, maturity):
    """Replace each payoff by a control-variate version with the same mean.

    The control is the discounted terminal spot Y = exp(-r * T) * S_T.
    Under risk-neutral GBM,

        S_T = S_0 * exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z)

    and the lognormal identity E[exp(sigma * sqrt(T) * Z - 0.5 * sigma * sigma * T)]
    = 1, because that exponent is normal with mean -0.5 * v and variance v
    for v = sigma * sigma * T, and E[exp(normal)] = exp(mean + 0.5 * variance)
    = exp(0). Therefore

        E[S_T] = S_0 * exp((r - q) * T)
        E[Y] = exp(-r * T) * E[S_T] = S_0 * exp(-q * T).

    Call this known number e_y. For any constant b,

        X(b) = X - b * (Y - e_y)

    satisfies E[X(b)] = E[X], since E[Y - e_y] = 0. The variance is

        Var(X(b)) = Var(X) + b * b * Var(Y) - 2 * b * Cov(X, Y).

    Differentiate in b and set the derivative to 0:

        0 = 2 * b * Var(Y) - 2 * Cov(X, Y)
        b_star = Cov(X, Y) / Var(Y).

    The sample version, with X_bar and Y_bar the sample means, is

        b = sum((X_i - X_bar) * (Y_i - Y_bar)) / sum((Y_i - Y_bar) ** 2).

    b is wrapped in ``stop_gradient``. It is a number computed from the
    sample; treating it as a constant when differentiating leaves

        dX(b)/d(theta) = dX/d(theta) - b * (dY/d(theta) - de_y/d(theta))

    whose expectation is dE[X]/d(theta), because E[dY/d(theta)] = de_y/d(theta).
    Differentiating through b as well would add a term that estimates zero
    and only adds noise.

    The returned array is X - b * (Y - e_y), using the known e_y and not the
    sample mean of Y. Using the sample mean would force the correction to
    average to 0, so the price would not move. Using e_y shifts the price by
    -b * (Y_bar - e_y), which is the useful part of the control.
    """
    dtype = discounted_payoff.dtype
    r = jnp.asarray(r, dtype=dtype)
    q = jnp.asarray(q, dtype=dtype)
    maturity = jnp.asarray(maturity, dtype=dtype)
    s0 = jnp.asarray(s0, dtype=dtype)
    y = jnp.exp(-r * maturity) * spot_T
    known_mean = s0 * jnp.exp(-q * maturity)
    payoff_gap = discounted_payoff - jnp.mean(discounted_payoff)
    control_gap = y - jnp.mean(y)
    b = jnp.sum(payoff_gap * control_gap) / jnp.sum(control_gap * control_gap)
    b = lax.stop_gradient(b)
    return discounted_payoff - b * (y - known_mean)


def _normals_for(key, payoff, n_paths, n_steps, antithetic):
    """Choose a terminal draw or a path draw, then apply the antithetic mirror.

    Call, put, and digital need one normal per path, so the step count passed
    to ``draw_normals`` is ``None`` and the shape is ``(n_paths,)``. An Asian
    or a barrier needs an independent normal on each slice of length
    ``T / n_steps``, so the shape is ``(n_paths, n_steps)``. In both cases
    the antithetic mirror, when requested, is the second half of that array
    and equals the negative of the first half.
    """
    if payoff in TERMINAL_PAYOFFS:
        return draw_normals(key, n_paths, None, antithetic)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    return draw_normals(key, n_paths, n_steps, antithetic)


def _check_market(payoff, s0, strike, sigma, maturity, n_paths, antithetic, barrier):
    """Reject inputs that make the GBM formulas or the sample variance undefined.

    The spot, strike, volatility, and maturity enter as divisors or as
    ``log(S0 / K)`` and ``sqrt(T)``, so each must satisfy ``value > 0``.
    The sample standard deviation divides by ``N - 1``, so ``N = n_paths``
    must be at least 2. Antithetic pairing uses ``M = n_paths / 2`` pairs,
    which is an integer only when ``n_paths`` is even. An up-and-out is
    already dead at inception unless ``barrier > S0``. A down-and-out is
    already dead unless ``barrier < S0``.
    """
    if payoff not in PAYOFFS:
        raise ValueError(f"payoff must be one of {tuple(PAYOFFS)}")
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2")
    for name, value in (
        ("s0", s0),
        ("strike", strike),
        ("sigma", sigma),
        ("maturity", maturity),
    ):
        if not value > 0:
            raise ValueError(f"{name} must be > 0")
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic is True")
    if payoff in BARRIER_PAYOFFS:
        if barrier is None:
            raise ValueError(f"barrier is required for payoff '{payoff}'")
        if payoff == "up_out_call" and not barrier > s0:
            raise ValueError("up-and-out barrier must be above s0")
        if payoff == "down_out_put" and not barrier < s0:
            raise ValueError("down-and-out barrier must be below s0")


def _sample_function(z, payoff, strike, barrier, control_variate):
    """Build the map from market inputs to the vector of discounted samples.

    When ``control_variate`` is true, each sample is the control-adjusted
    payoff from ``apply_control_variate``. The flag is a Python bool, so it
    is fixed before JAX traces the body.
    """

    def samples(s0, r, q, sigma, maturity):
        """Discounted payoff on each path, then the control adjustment.

        With ``D = exp(-r * T)`` the per-path term before the control is

            X = D * payoff(path).

        When the control is on, ``apply_control_variate`` replaces that with

            X - b * (D * S_T - S0 * exp(-q * T))

        and holds ``b`` fixed under differentiation.
        """
        paths = _paths_for_payoff(s0, r, q, sigma, maturity, z, payoff)
        dtype = paths.dtype
        strike_at = jnp.asarray(strike, dtype=dtype)
        rate = jnp.asarray(r, dtype=dtype)
        time = jnp.asarray(maturity, dtype=dtype)
        undiscounted = PAYOFFS[payoff](paths, strike_at, barrier)
        discounted = jnp.exp(-rate * time) * undiscounted
        if control_variate:
            discounted = apply_control_variate(
                discounted, paths[:, -1], s0, r, q, maturity
            )
        return discounted

    return samples


def estimate_price(
    key,
    s0,
    strike,
    r,
    q,
    sigma,
    maturity,
    n_paths,
    payoff,
    n_steps=1,
    barrier=None,
    antithetic=False,
    control_variate=False,
):
    """Return ``(price, standard_error)`` for one Monte Carlo estimate.

    The key is consumed once, inside ``draw_normals``. Nothing is read from a
    global random generator. For call and put, ``n_steps`` is not used: S_T
    is drawn from the exact terminal formula.

    ``antithetic=True`` stacks each normal with its negative before the
    payoff is evaluated. ``control_variate=True`` subtracts a multiple of
    the discounted terminal spot, centred on its known mean S0 * exp(-q * T).
    The algebra of both adjustments is in ``draw_normals``,
    ``apply_control_variate``, and ``standard_error``. The price is the mean
    of the adjusted samples. The standard error is the plain one, or the
    paired one when antithetics are on.
    """
    _check_market(payoff, s0, strike, sigma, maturity, n_paths, antithetic, barrier)
    z = _normals_for(key, payoff, n_paths, n_steps, antithetic)
    samples = _sample_function(z, payoff, strike, barrier, control_variate)(
        s0, r, q, sigma, maturity
    )
    price = jnp.mean(samples)
    error = standard_error(samples, antithetic)
    return float(price), float(error)


def _fd_step(value: float) -> float:
    """Absolute bump for a central difference.

    The step is one percent of the level, but never smaller than 1e-3, so a
    rate near 0 (for example r = 0.05, where 0.01 * 0.05 = 5e-4) still moves
    by enough to stand above roundoff in float32. The central-difference
    truncation error is (h * h / 6) times the third derivative, from the
    Taylor line

        (f(x + h) - f(x - h)) / (2 * h) = f'(x) + (h * h / 6) * f'''(x) + ...
    """
    return max(0.01 * abs(value), 1e-3)


def _price_function(z, payoff, strike, barrier, control_variate):
    """Build V(S0, r, q, sigma, T), the Monte Carlo mean at a fixed set of draws.

    ``z`` is closed over, so it is the same sample in every bump and in
    ``jax.grad``. The returned value is

        V = (1 / N) * (X_1 + ... + X_N)

    with X_i the discounted payoff on path i, after the control variate when
    that switch is on. Differentiating V is legitimate because the normals
    do not depend on the market inputs.
    """

    def price(s0, r, q, sigma, maturity):
        """Monte Carlo price at these inputs: ``V = (1 / N) * sum(X_i)``."""
        samples = _sample_function(z, payoff, strike, barrier, control_variate)(
            s0, r, q, sigma, maturity
        )
        return jnp.mean(samples)

    return price


def _pathwise_greeks(price_fn, s0, r, q, sigma, maturity):
    """Delta, vega, rho, and theta from jax.grad, gamma from a bumped delta.

    Order of arguments of ``price_fn`` is (s0, r, q, sigma, maturity). q is
    argument 2 and is not a requested Greek. One reverse-mode pass returns

        dV/dS0, dV/dr, dV/d(sigma), dV/dT.

    Theta is the calendar derivative. If T is the time left to expiry and the
    clock advances while the expiry date stays put, T falls, so

        theta = dV/dt = -dV/dT.

    Gamma is not ``jax.hessian``. For a call, S_T = S0 * M with M free of S0,
    and away from the single point S_T = K

        d/dS0 max(S_T - K, 0) = 1_{S_T > K} * M
        d2/dS0/dS0 max(S_T - K, 0) = 0

    because M does not depend on S0 and the indicator is locally constant.
    At the kink JAX's derivative of ``maximum`` is 0 as well. The Hessian of
    the sample mean is therefore 0, while the true Black-Scholes gamma is the
    derivative of an expectation of that indicator and is not 0.

    The pathwise delta is already the smoothed object (it jumps when a path
    crosses the strike, and that jump is what carries gamma). A central
    difference with the same normals reads it off:

        h = max(0.01 * |S0|, 1e-3)
        gamma = (delta(S0 + h) - delta(S0 - h)) / (2 * h).
    """
    first_order = jit(grad(price_fn, argnums=(0, 1, 3, 4)))
    delta, rho, vega, d_maturity = first_order(s0, r, q, sigma, maturity)
    delta_of = jit(grad(price_fn, argnums=0))
    h = _fd_step(float(s0))
    gamma = (
        delta_of(s0 + h, r, q, sigma, maturity) - delta_of(s0 - h, r, q, sigma, maturity)
    ) / (2.0 * h)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "rho": float(rho),
        "theta": float(-d_maturity),
    }


def _finite_difference_greeks(price_fn, s0, r, q, sigma, maturity):
    """Central differences of V for a payoff JAX cannot differentiate pathwise.

    Digital and barrier payoffs are not differentiable pathwise. A digital
    jumps from 0 to 1 when S_T crosses K, and a barrier jumps from the
    vanilla payoff to 0 when the path crosses the barrier. ``jax.grad`` of a
    jump is 0 on both sides and at the jump in JAX's convention, so the
    pathwise Greek misses the mass sitting on that jump and comes out near 0.

    The expectation itself is smooth in (S0, r, sigma, T). Bumping one input
    at a time, with the same normals, is a common-random-number difference:

        delta = (V(S0 + h_s) - V(S0 - h_s)) / (2 * h_s)
        gamma = (V(S0 + h_s) - 2 * V(S0) + V(S0 - h_s)) / (h_s * h_s)
        vega  = (V(sigma + h_v) - V(sigma - h_v)) / (2 * h_v)
        rho   = (V(r + h_r) - V(r - h_r)) / (2 * h_r)
        theta = -(V(T + h_t) - V(T - h_t)) / (2 * h_t)

    with h_x = max(0.01 * |x|, 1e-3). The gamma line is the second-derivative
    Taylor identity

        (f(x + h) - 2 * f(x) + f(x - h)) / (h * h) = f''(x) + (h * h / 12) * f''''(x) + ...
    """
    def value(spot, rate, vol, time):
        """Price at one bumped spot, rate, volatility, and maturity.

        ``q`` stays at the original value. This is ``V(S0, r, sigma, T)``
        in the central-difference formulas of ``_finite_difference_greeks``.
        """
        return price_fn(spot, rate, q, vol, time)

    base = value(s0, r, sigma, maturity)
    h_s = _fd_step(float(s0))
    h_r = _fd_step(float(r))
    h_v = _fd_step(float(sigma))
    h_t = _fd_step(float(maturity))
    delta = (value(s0 + h_s, r, sigma, maturity) - value(s0 - h_s, r, sigma, maturity)) / (
        2.0 * h_s
    )
    gamma = (
        value(s0 + h_s, r, sigma, maturity) - 2.0 * base + value(s0 - h_s, r, sigma, maturity)
    ) / (h_s * h_s)
    vega = (value(s0, r, sigma + h_v, maturity) - value(s0, r, sigma - h_v, maturity)) / (
        2.0 * h_v
    )
    rho = (value(s0, r + h_r, sigma, maturity) - value(s0, r - h_r, sigma, maturity)) / (
        2.0 * h_r
    )
    d_maturity = (value(s0, r, sigma, maturity + h_t) - value(s0, r, sigma, maturity - h_t)) / (
        2.0 * h_t
    )
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "rho": float(rho),
        "theta": float(-d_maturity),
    }


def estimate_greeks(
    key,
    s0,
    strike,
    r,
    q,
    sigma,
    maturity,
    n_paths,
    payoff,
    n_steps=1,
    barrier=None,
    antithetic=False,
    control_variate=False,
):
    """Return ``(greeks, method)`` for one fixed set of normal draws.

    ``greeks`` has keys delta, gamma, vega, rho, theta. ``method`` says
    whether those numbers came from pathwise automatic differentiation or
    from finite differences. Antithetic draws and the control variate, when
    switched on, are inside the function being differentiated, so the Greek
    is the sensitivity of the same estimator that produced the price.
    """
    _check_market(payoff, s0, strike, sigma, maturity, n_paths, antithetic, barrier)
    z = _normals_for(key, payoff, n_paths, n_steps, antithetic)
    price_fn = _price_function(z, payoff, strike, barrier, control_variate)
    if payoff in PATHWISE:
        greeks = _pathwise_greeks(price_fn, s0, r, q, sigma, maturity)
        method = "pathwise AD (gamma: central difference of the pathwise delta)"
    else:
        # Digital and barrier payoffs are not differentiable pathwise, so the
        # Greeks below are finite differences of the price. See
        # ``_finite_difference_greeks``.
        greeks = _finite_difference_greeks(price_fn, s0, r, q, sigma, maturity)
        method = "finite differences (common random numbers)"
    return greeks, method


def _steps_used(payoff, n_steps):
    """How many Brownian steps the estimator actually draws.

    A call, put, or digital depends on the path only through S_T. The exact
    terminal formula is the one-step case, so extra monitoring dates would
    not change the law of the payoff. Asian and barrier contracts use every
    requested step.
    """
    if payoff in TERMINAL_PAYOFFS:
        return 1
    return n_steps


def run_pricing(config: OptionConfig) -> OptionResult:
    """Price one option and, unless switched off, its Greeks.

    The seed on the config becomes ``random.key(seed)``. The same key is
    passed to the price and to the Greeks, so both see the same normals.
    Terminal contracts report ``n_steps`` = 1 even if the config asked for
    a finer grid.
    """
    if not isinstance(config, OptionConfig):
        raise TypeError("config must be an OptionConfig")
    n_steps = _steps_used(config.payoff, config.n_steps)
    key = random.key(config.seed)
    price, error = estimate_price(
        key,
        config.s0,
        config.strike,
        config.r,
        config.q,
        config.sigma,
        config.maturity,
        config.n_paths,
        config.payoff,
        n_steps=n_steps,
        barrier=config.barrier,
        antithetic=config.antithetic,
        control_variate=config.control_variate,
    )
    greeks = {}
    method = None
    if config.greeks:
        greeks, method = estimate_greeks(
            key,
            config.s0,
            config.strike,
            config.r,
            config.q,
            config.sigma,
            config.maturity,
            config.n_paths,
            config.payoff,
            n_steps=n_steps,
            barrier=config.barrier,
            antithetic=config.antithetic,
            control_variate=config.control_variate,
        )
    return OptionResult(
        payoff=config.payoff,
        price=price,
        stderr=error,
        n_paths=config.n_paths,
        n_steps=n_steps,
        greeks=greeks,
        greek_method=method,
    )


def price_grid(key, *, s0, strikes, sigmas, r, q, maturity, n_paths, payoff="call"):
    """Price a strike-by-volatility grid with one shared normal sample.

    ``strikes`` and ``sigmas`` are 1-d. The result arrays have shape
    (len(sigmas), len(strikes)): one row per volatility, one column per
    strike. Only call, put, and digital_call are accepted, because each cell
    uses the terminal formula

        S_T = S_0 * exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z)

    with the same Z vector in every cell. The price in a cell is the mean of
    exp(-r * T) * payoff(S_T), and the standard error is s / sqrt(n_paths)
    with s the sample standard deviation of those discounted payoffs (the
    plain formula in ``standard_error``, with no antithetic pairing and no
    control variate).

    ``vmap`` over strikes, then ``vmap`` over volatilities, turns the scalar
    cell into the whole grid. ``jit`` compiles that batched function, so one
    call prices every cell.
    """
    if payoff not in TERMINAL_PAYOFFS:
        raise ValueError(f"price_grid supports {TERMINAL_PAYOFFS}")
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2")
    z = random.normal(key, (n_paths,))
    strike_axis = jnp.asarray(strikes, dtype=z.dtype)
    sigma_axis = jnp.asarray(sigmas, dtype=z.dtype)
    payoff_fn = PAYOFFS[payoff]

    def one_cell(strike, sigma):
        """Price and standard error at one strike and one volatility.

        The terminal spot is

            S_T = S0 * exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z)

        with the shared normals ``Z``. The cell price is the mean of
        ``exp(-r * T) * payoff(S_T)``, and the cell error is ``s / sqrt(N)``.
        """
        spot_T = terminal_spot(s0, r, q, sigma, maturity, z)
        paths = spot_T[:, None]
        discount = jnp.exp(-jnp.asarray(r, dtype=z.dtype) * jnp.asarray(maturity, dtype=z.dtype))
        discounted = discount * payoff_fn(paths, strike, None)
        price = jnp.mean(discounted)
        error = standard_error(discounted, antithetic=False)
        return price, error

    over_strikes = vmap(one_cell, in_axes=(0, None))
    over_sigmas_and_strikes = vmap(over_strikes, in_axes=(None, 0))
    prices, errors = jit(over_sigmas_and_strikes)(strike_axis, sigma_axis)
    return prices, errors
