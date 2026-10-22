"""Score-based diffusion in JAX.

Public surface
--------------
TrainConfig
    Validated experiment inputs.
run_experiment
    Train, sample, and evaluate. Returns ExperimentResult.
make_spec / DataSpec
    Data adapters (swiss-roll, sinusoids, CSV).
sample_reverse_sde, denoise_one_shot, log_likelihood
    Inference primitives.
"""
import os # noqa: F401
import sys # noqa: F401

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jax_diffusion.config import DenoiseReport, ExperimentResult, TrainConfig
from jax_diffusion.data import DataSpec, make_spec
from jax_diffusion.likelihood import log_likelihood
from jax_diffusion.model import eps_model, init_params, score
from jax_diffusion.sampling import denoise_one_shot, sample_reverse_sde
from jax_diffusion.sde import diffuse, marginal
from jax_diffusion.train import run_experiment

__all__ = [
    "DataSpec",
    "DenoiseReport",
    "ExperimentResult",
    "TrainConfig",
    "denoise_one_shot",
    "diffuse",
    "eps_model",
    "init_params",
    "log_likelihood",
    "make_spec",
    "marginal",
    "run_experiment",
    "sample_reverse_sde",
    "score",
]
