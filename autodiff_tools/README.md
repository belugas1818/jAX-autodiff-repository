# autodiff_tools

Small, documented JAX autodiff helpers for the Monte Carlo option pricer project.
Every function carries a docstring with the core idea and the derivation in full
algebraic steps. Multiplication is written `*` throughout.

## Setup

```bash
pip install -e ".[dev]"
pytest autodiff_tools
```

Run from the repository root (the directory that contains `autodiff_tools/`).

## Conventions

- Pricer signature: `price_fn(s0, strike, r, sigma, maturity) -> scalar float`.
  Close over everything you do not differentiate (PRNG key, `n_paths`, `steps`,
  dividend yield `q`, payoff type).
- Use float64 for finite-difference checks:
  `jax.config.update("jax_enable_x64", True)` before creating any array.
- PRNG keys are passed explicitly (`jax.random.key(seed)`); never use global random state.
- Theta is `-dV/dmaturity`. Vega and rho are per 1.00 (divide by 100 for per 1%).

## Index

| File | Function | What it does |
| --- | --- | --- |
| `derivatives.py` | `named_partials(f, names, *args)` | dict of `df/dx_i` for every positional arg (`jax.grad`) |
| | `second_partial(f, i, j)` | `d^2 f / (dx_i dx_j)` as `grad(grad(f, i), j)` |
| | `value_and_partials(f, argnums)` | value and gradient in one pass |
| `jacobians.py` | `jvp_at(f, x, v)` | `(f(x), J * v)`, forward mode |
| | `vjp_at(f, x, u)` | `(f(x), u^T * J)`, reverse mode |
| | `jacobian_fwd`, `jacobian_rev` | full Jacobian via `jacfwd`, `jacrev` |
| | `jacobian_via_jvp`, `jacobian_via_vjp` | the same Jacobian built column by column / row by row, to show how |
| | `hessian(f)`, `hvp(f, x, v)` | full Hessian; Hessian-vector product without forming it |
| | `exact_trace`, `hutchinson_trace` | trace of a Jacobian, exact and stochastic (for the diffusion project later) |
| `batching.py` | `batch(f, in_axes)` | `jit(vmap(f))` |
| | `grid2d(f)` | scalar `f(x, y)` to a matrix over `xs` by `ys` (price surfaces) |
| | `replicate_over_keys(f, key, n)` | repeat a Monte Carlo experiment with `n` independent keys |
| | `compare_jit_speed(f, *args)` | eager vs jit timing and compile time |
| `finite_diff.py` | `central_diff`, `central_second` | central differences, derived in the docstring |
| | `check_grad` | compare autodiff or a hand-written gradient with finite differences |
| `greeks.py` | `greeks_from_price(price_fn, gamma="autodiff", rel_step=1e-2)` | delta, gamma, vega, rho, theta by `jax.grad`; `gamma="delta_fd"` takes gamma as a central difference of the pathwise delta |
| | `fd_greeks(price_fn, ...)` | the same by finite differences, for digital and barrier payoffs |
| `examples.py` | `black_scholes`, `black_scholes_greeks` | differentiable closed-form pricer and analytic Greeks, used as an oracle in tests |

## Things to know before using these on a Monte Carlo pricer

- Fix the key inside `price_fn`. The same draws are then reused for every
  evaluation, so the derivative is the pathwise estimator and finite differences
  see no sampling noise.
- Pathwise delta, vega, rho and theta are valid for continuous payoffs
  (call, put, Asian). Gamma is not: with fixed draws `S_T = s0 * M` with `M`
  free of `s0`, so `d/ds0 max(S_T - K, 0) = 1{S_T > K} * M` and its second
  derivative is exactly 0. The default `gamma="autodiff"` therefore returns 0
  for a Monte Carlo pricer. Use `greeks_from_price(price_fn, gamma="delta_fd")`,
  which computes `(delta(s0 + h) - delta(s0 - h)) / (2 * h)` with
  `h = rel_step * s0` on the same draws. Keep `gamma="autodiff"` for smooth
  closed-form pricers.
- For digital and knock-out barrier payoffs the pathwise derivative misses the
  jump and gives about 0 even for delta. Use `fd_greeks` with a large enough bump.
- `examples.py` is a fixture for these tools. The pricer's own tests should
  write the closed form independently.
