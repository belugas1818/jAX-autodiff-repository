"""Training loop: config in, fitted model and evaluation metrics out."""

from __future__ import annotations

import math
import time

import jax.numpy as jnp
from jax import jit, random, vmap

from jax_diffusion.config import DenoiseReport, ExperimentResult, TrainConfig
from jax_diffusion.data import make_spec
from jax_diffusion.likelihood import log_likelihood
from jax_diffusion.model import init_params
from jax_diffusion.optim import adam_init, make_train_step
from jax_diffusion.sampling import denoise_one_shot, sample_reverse_sde
from jax_diffusion.sde import diffuse, marginal


def run_experiment(config: TrainConfig, *, log=print) -> ExperimentResult:
    """Train a noise-prediction diffusion model and optionally evaluate it.

    Parameters
    ----------
    config:
        Validated `TrainConfig`.
    log:
        Callable used for progress lines (default: ``print``). Pass ``lambda *_: None``
        to silence.

    Returns
    -------
    ExperimentResult
        EMA parameters, optional samples, NLL, and denoising MSE.
    """
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")

    key = random.key(config.seed)
    key, k_norm, k_init = random.split(key, 3)
    spec = make_spec(config.data, config.csv_path)
    spec.fit_normalizer(k_norm)
    dim = int(math.prod(spec.shape))
    log(f"data={spec.name} shape={spec.shape} dim={dim}")

    params = init_params(
        k_init, dim, hidden=config.hidden, depth=config.depth, temb=config.temb
    )
    ema = params
    opt_state = adam_init(params)
    train_step = make_train_step(
        spec.sample,
        config.batch_size,
        config.jac_weight,
        config.lr,
        config.steps,
        config.ema_decay,
    )

    history: list[dict[str, float]] = []
    t0 = time.time()
    report_every = max(1, config.steps // 10)
    for i in range(1, config.steps + 1):
        key, sk = random.split(key)
        params, opt_state, ema, loss, mse, jac = train_step(params, opt_state, ema, sk)
        if i % report_every == 0 or i == 1 or i == config.steps:
            row = {
                "step": float(i),
                "loss": float(loss),
                "mse": float(mse),
                "jac": float(jac),
                "seconds": time.time() - t0,
            }
            history.append(row)
            log(
                f"step {i:6d}  loss {row['loss']:.4f}  mse {row['mse']:.4f}"
                f"  jac {row['jac']:.3f}  ({row['seconds']:.1f}s)"
            )

    samples = None
    real = None
    if config.n_samples > 0:
        key, sk = random.split(key)
        sampler = jit(sample_reverse_sde, static_argnums=(2, 3, 4))
        samples = sampler(ema, sk, spec.shape, config.n_samples, config.sample_steps)
        key, sk = random.split(key)
        real = spec.sample(sk, config.n_samples)
        log(
            "sample mean/std (normalised): "
            f"{float(samples.mean()):.3f} / {float(samples.std()):.3f}"
            f"   real: {float(real.mean()):.3f} / {float(real.std()):.3f}"
        )

    nll_nats = None
    nll_bpd = None
    trace_kind = None
    test = None
    if config.eval_likelihood or config.eval_denoise:
        key, sk = random.split(key)
        test = spec.sample(sk, config.nll_test_size)

    if config.eval_likelihood:
        exact = dim <= 16
        key, sk = random.split(key)
        lk = random.split(sk, test.shape[0])
        ll = jit(vmap(lambda k, x: log_likelihood(
            ema, k, x, n_steps=config.likelihood_steps, exact=exact
        )))(lk, test)
        ll_orig = ll - jnp.sum(jnp.log(spec.std))
        trace_kind = "exact jacfwd trace" if exact else "Hutchinson vjp trace"
        nll_nats = -float(ll_orig.mean())
        nll_bpd = nll_nats / (dim * math.log(2))
        log(
            f"test NLL ({trace_kind}): {nll_nats:.3f} nats/sample, "
            f"{nll_bpd:.3f} bits/dim"
        )

    denoise_rows: list[DenoiseReport] = []
    if config.eval_denoise:
        log("one-shot denoising MSE (normalised units):")
        for t in (0.05, 0.1, 0.2, 0.4):
            key, sk = random.split(key)
            xt, _ = vmap(diffuse, in_axes=(0, 0, None))(
                random.split(sk, test.shape[0]), test, t
            )
            x_hat = denoise_one_shot(ema, xt, t)
            a, _ = marginal(t)
            noisy = float(jnp.mean((xt / a - test) ** 2))
            cleaned = float(jnp.mean((x_hat - test) ** 2))
            denoise_rows.append(DenoiseReport(t=t, noisy_mse=noisy, denoised_mse=cleaned))
            log(f"  t={t:.2f}  noisy {noisy:.4f}  ->  denoised {cleaned:.4f}")

    if config.plot_path:
        if samples is None or real is None:
            raise ValueError("plot_path requires n_samples > 0")
        from jax_diffusion.plotting import save_plot

        save_plot(config.plot_path, spec, real, samples)
        log(f"saved plot to {config.plot_path}")

    return ExperimentResult(
        spec_name=spec.name,
        shape=spec.shape,
        dim=dim,
        history=history,
        params=params,
        ema_params=ema,
        samples=samples,
        real=real,
        nll_nats=nll_nats,
        nll_bits_per_dim=nll_bpd,
        trace_kind=trace_kind,
        denoise=tuple(denoise_rows),
    )
