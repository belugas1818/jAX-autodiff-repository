"""Validated inputs and the printed result of one pricing run.

The core idea: the command line and the library hand the pricer one checked
bundle of numbers, and get back the price, its standard error, and any Greeks.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OptionConfig:
    """Market and contract inputs for ``run_pricing``.

    ``s0`` is the spot, ``strike`` is K, ``r`` the continuous risk-free rate,
    ``q`` the continuous dividend yield, ``sigma`` the GBM volatility, and
    ``maturity`` the year fraction T. ``n_paths`` is the number of simulated
    scenarios. ``n_steps`` is the number of monitoring dates for an Asian or
    a barrier; a call, put, or digital ignores it and uses the one-step
    terminal formula. ``seed`` builds the JAX key ``random.key(seed)``.
    ``antithetic`` mirrors each normal. ``control_variate`` subtracts a
    multiple of the discounted terminal spot. ``greeks`` turns the derivative
    pass on or off. ``barrier`` is required for up-and-out and down-and-out.
    """

    payoff: str = "call"
    s0: float = 100.0
    strike: float = 100.0
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.2
    maturity: float = 1.0
    barrier: float | None = None
    n_paths: int = 100_000
    n_steps: int = 64
    seed: int = 0
    antithetic: bool = True
    control_variate: bool = True
    greeks: bool = True


@dataclass
class OptionResult:
    """Monte Carlo output.

    ``price`` is the sample mean of the discounted (and, when asked,
    control-adjusted) payoffs. ``stderr`` is the standard error of that
    mean. ``n_steps`` is the number of Brownian steps actually used, which
    is 1 for a terminal payoff. ``greeks`` is empty when they were not
    requested. ``greek_method`` names pathwise automatic differentiation or
    finite differences.
    """

    payoff: str
    price: float
    stderr: float
    n_paths: int
    n_steps: int
    greeks: dict = field(default_factory=dict)
    greek_method: str | None = None
