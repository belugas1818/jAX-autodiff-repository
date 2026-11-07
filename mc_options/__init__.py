"""Monte Carlo prices for options under risk-neutral geometric Brownian motion.

The core idea: a price is the average of discounted simulated payoffs, and a
Greek is a derivative of that average.
"""

from mc_options.config import OptionConfig, OptionResult
from mc_options.pricer import estimate_greeks, estimate_price, price_grid, run_pricing

__all__ = [
    "OptionConfig",
    "OptionResult",
    "estimate_greeks",
    "estimate_price",
    "price_grid",
    "run_pricing",
]
