# JAX study repo

Three packages, in the order to read them. Install once from this directory:

```bash
pip install -e ".[dev,plot]"
pytest
```

## autodiff_tools

Documented wrappers for `grad`, `jvp`, `vjp`, `jacfwd`, `jacrev`, `vmap`, and `jit`, plus finite-difference checks. The derivations are in the module docstrings. Index: [autodiff_tools/README.md](autodiff_tools/README.md).

```python
from autodiff_tools import named_partials, greeks_from_price, grid2d
```

## mc_options

Monte Carlo prices under risk-neutral geometric Brownian motion. The price is the mean of discounted payoffs. Greeks are `jax.grad` of that mean for call, put, and Asian call, and finite differences for digital and barrier payoffs. The Black-Scholes formula used in the tests is written separately in `tests/black_scholes.py`.

```bash
python -m mc_options
python -m mc_options --payoff put -K 95 -T 0.5 --paths 200000
python -m mc_options --payoff asian_call --steps 252
python -m mc_options --payoff up_out_call --barrier 130
python -m mc_options --no-antithetic --no-control-variate
```

Defaults: `S0 = 100`, `K = 100`, `r = 5%`, `sigma = 20%`, `T = 1` year, 100,000 paths, antithetic variates and a discounted-spot control variate.

## jax_diffusion

Score-based diffusion on toy data (swiss roll, sinusoids, or a CSV). Training, sampling, and likelihood are in [jax_diffusion/README.md](jax_diffusion/README.md).

```bash
python -m jax_diffusion --data points2d --steps 4000 --plot roll.png
```
