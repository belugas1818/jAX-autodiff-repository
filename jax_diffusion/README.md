# jax_diffusion

A small score-based diffusion model written in pure JAX: no Flax, no Optax.
It trains a noise-prediction MLP on toy data (a 2D swiss roll, 1D sinusoid
signals, or your own CSV), then draws new samples, measures test
log-likelihood, and checks how well it denoises.

The whole thing is about 800 lines across a dozen modules. Each module does
one job, so you can read it top to bottom in the order below.

---

## Quick start

The modules import each other as `jax_diffusion.<module>`, so run commands
from the folder **above** this one:

```bash
cd "real projs"
pip install jax numpy matplotlib      # requirements.txt is currently empty
python -m jax_diffusion --data points2d --steps 4000 --plot roll.png
```

From Python:

```python
from jax_diffusion import TrainConfig, run_experiment

result = run_experiment(TrainConfig(data="signal1d", steps=2000))
print(result.summary())
```

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `points2d` | `points2d` (swiss roll), `signal1d` (32-sample sinusoids), or `csv` |
| `--csv` | — | Path to a numeric CSV, one sample per row (needed with `--data csv`) |
| `--steps` | 4000 | Training iterations |
| `--batch` | 512 | Batch size |
| `--lr` | 2e-3 | Peak learning rate (cosine-decayed) |
| `--jac-weight` | 1e-4 | Weight of the Jacobian smoothness penalty |
| `--n-samples` | 2000 | Samples to generate after training (0 skips sampling) |
| `--hidden` / `--depth` | 256 / 4 | MLP width and number of hidden layers |
| `--seed` | 0 | PRNG seed |
| `--plot` | — | Save a PNG of real vs. generated samples |

`TrainConfig` has more fields than the CLI exposes (`ema_decay`,
`sample_steps`, `likelihood_steps`, `eval_likelihood`, ...). See
[config.py](config.py).

---

## The idea in four lines

1. **Forward process:** gradually add Gaussian noise to the data until it is
   pure noise at `t = 1`.
2. **Training:** teach a network `eps_theta(x_t, t)` to predict the noise that
   was added.
3. **Score:** that noise prediction gives the gradient of the log density,
   `score = -eps_theta / sigma(t)`.
4. **Generation:** start from noise and run the process backwards, using the
   score to steer toward the data.

---

## Reading order

Read the modules in dependency order, so each file only uses things you've
already seen.

### 1. [sde.py](sde.py): the noise schedule
The variance-preserving SDE `dx = -½β(t)x dt + √β(t) dW`, with `β` rising
linearly from 0.1 to 20.

- `marginal(t)` returns `(alpha, sigma)` so that
  `x_t = alpha·x_0 + sigma·eps`. **Everything else depends on this function.**
- `diffuse(key, x0, t)` noises clean data in a single closed-form step.
- `T_EPS = 1e-3` is the smallest time used, which keeps `sigma` away from zero.

### 2. [model.py](model.py): the network
- `time_embedding`: sin/cos features of `t`, as in transformers.
- `init_params`: an MLP stored as a plain list of `{"W", "b"}` dicts, i.e. a
  pytree. The last layer is scaled by 0.01 so the network starts out
  predicting roughly zero.
- `eps_model(params, x, t)` works on **one** sample. Batching is done later
  with `vmap`.
- `score`: converts the noise prediction into a score.

### 3. [loss.py](loss.py): what training minimizes
For each example: pick a random `t`, noise `x0`, and penalize
`(eps_hat − eps)²`. The same `jvp` call also gives `J·v` for a random ±1
vector `v`, and `|J·v|²` is a cheap (Hutchinson) estimate of how sharp the
network is. That is the `jac` term, weighted by `jac_weight`.

### 4. [autodiff.py](autodiff.py): JVP/VJP helpers
These are reusable forms of the autodiff tricks from `jax_test.py`:
Hessian-vector products, Hutchinson trace estimators, and an exact Jacobian
trace. `likelihood.py` uses the trace functions.

### 5. [optim.py](optim.py): Adam from scratch
- `adam_init` / `adam_update`: Adam written with `jax.tree.map`, so it works
  on any parameter pytree.
- `make_train_step` builds one `@jit` step: sample a batch, take gradients,
  clip them to global norm 1, apply a cosine learning-rate schedule, run
  Adam, and update the EMA weights.

### 6. [data.py](data.py): datasets
A `DataSpec` holds a sample `shape` and a `sampler(key, n)`.
`fit_normalizer` estimates each feature's mean and std, so training happens
in **normalized** units. `denormalize` converts back. `make_spec(name)` picks
the swiss roll, the sinusoids, or the CSV loader.

### 7. [sampling.py](sampling.py): generating data
- `sample_reverse_sde`: starts from `N(0, I)` and steps from `t = 1` down to
  `T_EPS` using Euler–Maruyama inside `lax.scan`.
- `denoise_one_shot`: Tweedie's formula, `x0 ≈ (x_t − sigma·eps_hat) / alpha`.
  It is applied once at the end of sampling and is also used for the
  denoising evaluation.

### 8. [likelihood.py](likelihood.py): exact log-probability
This uses the deterministic *probability-flow ODE* that matches the SDE. It
integrates the ODE from data (`t ≈ 0`) to noise (`t = 1`) with RK4, adding up
the divergence of the drift along the way:

```
log p_0(x0) = log N(x_1; 0, I) + ∫ div(drift) dt
```

The divergence is computed exactly with `jacfwd` when the data has 16
dimensions or fewer, and with a Hutchinson + VJP estimate otherwise.

### 9. [train.py](train.py): the whole pipeline
`run_experiment(config)` connects everything:
build data → init model → train loop (logged about 10 times) → sample with
the EMA weights → test NLL (converted back to original data units, then to
bits/dim) → denoising MSE at `t ∈ {0.05, 0.1, 0.2, 0.4}` → optional plot.
It returns an `ExperimentResult`.

### 10. Wrappers
- [config.py](config.py): `TrainConfig` (validated inputs), `ExperimentResult`, `DenoiseReport`
- [cli.py](cli.py) / [\_\_main\_\_.py](__main__.py): argparse front end for `python -m jax_diffusion`
- [plotting.py](plotting.py): scatter plot for 2D data, line plots for anything else
- [\_\_init\_\_.py](__init__.py): the public API

### Side file: [jax_test.py](jax_test.py)
A scratchpad that follows the JAX autodiff tutorial: `grad`, `value_and_grad`,
finite-difference checks, Hessian-vector products, and `vmap`'d Jacobian
products. The package doesn't import it. If `jvp`/`vjp` are new to you, read
it **before** `loss.py` and `autodiff.py`.

---

## How data flows through it

```
TrainConfig ─► make_spec ─► DataSpec.fit_normalizer
                                   │
             init_params ─► train_step (jit) ×steps ─► params, ema
                                │  uses batch_loss ─► diffuse + eps_model + jvp
                                ▼
           ema ─► sample_reverse_sde ─► samples
           ema ─► log_likelihood     ─► NLL (nats, bits/dim)
           ema ─► denoise_one_shot   ─► denoising MSE
                                ▼
                         ExperimentResult
```

## Reading the output

These numbers only show the format. Your actual values will differ.

```
step   4000  loss 0.3121  mse 0.3120  jac 1.234  (12.3s)
sample mean/std (normalised): 0.002 / 0.991   real: 0.001 / 1.000
test NLL (exact jacfwd trace): 1.85 nats/sample, 1.33 bits/dim
  t=0.10  noisy 0.0712  ->  denoised 0.0301
```

- **mse**: the noise-prediction error. Lower is better.
- **sample vs. real mean/std**: should be close if the model fits the data.
- **NLL**: lower means the model gives held-out data higher probability.
- **denoised < noisy**: the model actually removes noise.

## JAX patterns to look for

- **Explicit PRNG keys:** every random function takes a `key`, and keys are
  `random.split` before each use and never reused.
- **Per-sample functions + `vmap`:** the model and loss handle one example,
  and `vmap` adds the batch dimension.
- **`lax.scan`** replaces Python loops in the samplers and ODE solvers, so
  they compile once.
- **Pytrees:** params, Adam state, and EMA are all nested lists/dicts, and are
  updated with `jax.tree.map`.
