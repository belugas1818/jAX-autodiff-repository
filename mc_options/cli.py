"""Command line for the Monte Carlo pricer.

The core idea: flags become an OptionConfig, ``run_pricing`` does the
simulation, and the process prints the price, the standard error, and the
Greeks.
"""

from __future__ import annotations

import argparse

from mc_options.config import OptionConfig
from mc_options.payoffs import PAYOFFS
from mc_options.pricer import run_pricing


def parse_args(argv=None) -> OptionConfig:
    """Read the flags for one pricing run.

    ``--payoff`` selects the contract. ``-K`` / ``--strike`` is K and
    ``-T`` / ``--maturity`` is the year fraction T. ``--paths`` is the number
    of scenarios and ``--steps`` is the number of monitoring dates for an
    Asian or a barrier. ``--barrier`` is the knock-out level. ``--s0``,
    ``--r``, ``--q``, and ``--sigma`` are the spot, rate, dividend yield, and
    volatility. ``--seed`` is the integer passed to ``random.key``.
    ``--no-antithetic``, ``--no-control-variate``, and ``--no-greeks`` switch
    those three features off; they are on unless the flag is present.
    """
    parser = argparse.ArgumentParser(description="Monte Carlo option pricer in JAX.")
    parser.add_argument("--payoff", default="call", choices=tuple(PAYOFFS))
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--strike", "-K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--q", type=float, default=0.0)
    parser.add_argument("--maturity", "-T", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--barrier", type=float, default=None)
    parser.add_argument("--paths", type=int, default=100_000, dest="n_paths")
    parser.add_argument("--steps", type=int, default=64, dest="n_steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-antithetic", action="store_false", dest="antithetic")
    parser.add_argument("--no-control-variate", action="store_false", dest="control_variate")
    parser.add_argument("--no-greeks", action="store_false", dest="greeks")
    args = parser.parse_args(argv)
    return OptionConfig(
        payoff=args.payoff,
        s0=args.s0,
        strike=args.strike,
        r=args.r,
        q=args.q,
        sigma=args.sigma,
        maturity=args.maturity,
        barrier=args.barrier,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=args.seed,
        antithetic=args.antithetic,
        control_variate=args.control_variate,
        greeks=args.greeks,
    )


def format_result(result) -> str:
    """Render one result as a few plain lines.

    The price line is the Monte Carlo mean and its standard error. When
    Greeks were computed, the next lines name the method (pathwise automatic
    differentiation, or finite differences for a digital or a barrier) and
    list delta, gamma, vega, rho, and theta in that order.
    """
    lines = [
        f"{result.payoff}  paths={result.n_paths}  steps={result.n_steps}",
        f"price   {result.price:.4f}  +/- {result.stderr:.4f} (1 s.e.)",
    ]
    if result.greeks:
        lines.append(f"greeks via {result.greek_method}")
        for name in ("delta", "gamma", "vega", "rho", "theta"):
            lines.append(f"  {name:8s} {result.greeks[name]:10.4f}")
    return "\n".join(lines)


def main(argv=None) -> None:
    """Parse ``argv``, price the option, and print ``format_result``."""
    result = run_pricing(parse_args(argv))
    print(format_result(result))
