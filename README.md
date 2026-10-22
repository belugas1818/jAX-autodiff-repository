# JAX score-based diffusion

Production package extracted from the exploratory scripts
`.vscode/jax_diffusion.py` (model) and `.vscode/jax_test.py` (autodiff tools:
`grad`, `jvp`, `vjp`, `jacfwd`, `vmap`, `jit`).

The model is a variance-preserving diffusion: an MLP predicts the noise
injected by the forward SDE. Training uses denoising score matching plus an
optional Hutchinson Jacobian penalty. Sampling uses the reverse SDE;
likelihood uses the probability-flow ODE and a Jacobian trace.

## Install

```bash
pip install -e ".[dev,plot]"
```

## Inputs (`TrainConfig`)

| Field | Meaning |
| --- | --- |
| `data` | `points2d` (swiss roll), `signal1d` (length-32 sinusoids), or `csv` |
| `csv_path` | Numeric CSV, one sample per row (required for `csv`) |
| `steps`, `batch_size`, `lr`, `jac_weight` | Training |
| `hidden`, `depth`, `temb` | MLP size; `temb` must be even |
| `n_samples` | Reverse-SDE draws after training (`0` skips sampling) |
| `eval_likelihood`, `eval_denoise` | Held-out NLL and Tweedie denoising |
| `seed` | JAX PRNG seed |

Invalid values raise `ValueError` before any training starts.

## Outputs (`ExperimentResult`)

- `ema_params` — weights used for sampling / NLL / denoising
- `samples`, `real` — normalized arrays, or `None` if sampling skipped
- `nll_nats`, `nll_bits_per_dim` — original-scale NLL, or `None`
- `denoise` — noisy vs denoised MSE at several times `t`
- `history` — logged loss / mse / jacobian penalty
- `summary()` — JSON-friendly dict of the metrics above

Denormalize with `make_spec(...).fit_normalizer(key).denormalize(samples)`.

## Run

```bash
python -m jax_diffusion --data points2d --plot out.png
python -m jax_diffusion --data signal1d --steps 6000
python -m jax_diffusion --data csv --csv my_data.csv
```

Library use:

```python
from jax_diffusion import TrainConfig, run_experiment

result = run_experiment(TrainConfig(data="points2d", steps=4000, seed=0))
print(result.summary())
```

## Tests

```bash
pytest
```

Tests cover schedule identities, Tweedie inversion, autodiff (JVP/VJP/HVP vs
full Jacobian), config/data validation, and a short training smoke run. They
do **not** replace a full-length generative-quality check.
