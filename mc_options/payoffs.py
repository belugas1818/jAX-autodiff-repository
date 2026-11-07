"""Undiscounted option payoffs on a simulated path.

The core idea: each contract is a pure function of the path, the strike, and
(for barriers) the barrier. Discounting is done by the pricer, not here.
"""

from __future__ import annotations

import jax.numpy as jnp


def call(paths, strike, barrier):
    """European call payoff max(S_T - K, 0).

    ``paths`` has shape (n_paths, n_times). The expiry spot is the last
    column, S_T = paths[:, -1]. The barrier argument is unused: a vanilla
    call does not knock out.

    Piecewise definition, with K = strike:

        if S_T < K, the holder does not exercise, so the payoff is 0
        if S_T >= K, the holder pays K and receives S_T, so the payoff is S_T - K

    Those two lines are the same as

        payoff = max(S_T - K, 0).
    """
    del barrier
    spot_T = paths[:, -1]
    return jnp.maximum(spot_T - strike, 0.0)


def put(paths, strike, barrier):
    """European put payoff max(K - S_T, 0).

    ``paths`` has shape (n_paths, n_times). The expiry spot is the last
    column. The barrier argument is unused.

    Piecewise definition, with K = strike:

        if S_T > K, the holder does not exercise, so the payoff is 0
        if S_T <= K, the holder delivers S_T and receives K, so the payoff is K - S_T

    Those two lines are the same as

        payoff = max(K - S_T, 0).
    """
    del barrier
    spot_T = paths[:, -1]
    return jnp.maximum(strike - spot_T, 0.0)


def digital_call(paths, strike, barrier):
    """Cash-or-nothing digital call: 1 if S_T >= K, else 0.

    The barrier argument is unused. ``paths`` has shape (n_paths, n_times)
    and S_T is the last column. There is no share delivery, so the payoff is
    the indicator itself:

        payoff = 1 when S_T >= K
        payoff = 0 when S_T < K.

    Under GBM the probability of landing exactly on K is 0, so the choice of
    >= versus > does not change the expectation. The indicator jumps, which
    is why the pricer does not use pathwise ``jax.grad`` for this contract.
    """
    del barrier
    spot_T = paths[:, -1]
    return (spot_T >= strike).astype(spot_T.dtype)


def asian_call(paths, strike, barrier):
    """Arithmetic Asian call on the simulated fixings, excluding the spot at 0.

    ``paths`` has columns S_0, S_1, ..., S_n with S_n = S_T. The contract
    averages the n future fixings only:

        A = (1 / n) * (S_1 + S_2 + ... + S_n)

    and pays

        payoff = max(A - K, 0).

    Column 0 is S_0. It is not a fixing, so the average starts at column 1.
    With n = 1 the only fixing is S_T and this payoff equals the European
    call. The map from the path to max(A - K, 0) is Lipschitz, so pathwise
    ``jax.grad`` is valid for the first-order Greeks. The second derivative
    in the spot is still zero away from A = K.
    """
    del barrier
    average = jnp.mean(paths[:, 1:], axis=1)
    return jnp.maximum(average - strike, 0.0)


def up_out_call(paths, strike, barrier):
    """Discrete up-and-out call. Touching the barrier on the grid kills the option.

    Monitor every column of ``paths``, including S_0. Let

        M = max(S_0, S_1, ..., S_n).

    The contract pays the European call if the barrier was never touched, and
    pays 0 otherwise:

        payoff = max(S_T - K, 0) when M < barrier
        payoff = 0               when M >= barrier.

    Written as a product,

        payoff = 1_{M < barrier} * max(S_T - K, 0).

    This is discrete monitoring: a crossing between two grid times that
    returns before the next date is not seen. The indicator jumps when a
    monitored spot crosses the barrier, so the pathwise derivative misses
    that jump.
    """
    highest = jnp.max(paths, axis=1)
    survived = highest < barrier
    vanilla = jnp.maximum(paths[:, -1] - strike, 0.0)
    return survived.astype(vanilla.dtype) * vanilla


def down_out_put(paths, strike, barrier):
    """Discrete down-and-out put. Touching the barrier on the grid kills the option.

    Monitor every column of ``paths``, including S_0. Let

        m = min(S_0, S_1, ..., S_n).

    The contract pays the European put if the barrier was never touched, and
    pays 0 otherwise:

        payoff = max(K - S_T, 0) when m > barrier
        payoff = 0               when m <= barrier.

    Written as a product,

        payoff = 1_{m > barrier} * max(K - S_T, 0).

    The same discrete-monitoring limit as the up-and-out call applies, and
    the indicator is not pathwise differentiable.
    """
    lowest = jnp.min(paths, axis=1)
    survived = lowest > barrier
    vanilla = jnp.maximum(strike - paths[:, -1], 0.0)
    return survived.astype(vanilla.dtype) * vanilla


PAYOFFS = {
    "call": call,
    "put": put,
    "digital_call": digital_call,
    "asian_call": asian_call,
    "up_out_call": up_out_call,
    "down_out_put": down_out_put,
}

# Lipschitz payoffs: jax.grad of the sample mean is an unbiased pathwise Greek.
# The second derivative is still zero almost everywhere, so gamma is not hessian.
PATHWISE = frozenset({"call", "put", "asian_call"})

BARRIER_PAYOFFS = frozenset({"up_out_call", "down_out_put"})
