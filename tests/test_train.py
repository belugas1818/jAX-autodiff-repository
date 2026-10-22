from jax_diffusion.config import TrainConfig
from jax_diffusion.train import run_experiment


def test_short_training_smoke():
    result = run_experiment(
        TrainConfig(
            data="points2d",
            steps=2,
            batch_size=8,
            hidden=8,
            depth=1,
            temb=8,
            n_samples=4,
            sample_steps=3,
            nll_test_size=4,
            likelihood_steps=2,
            eval_likelihood=True,
            eval_denoise=True,
            seed=0,
        ),
        log=lambda *_: None,
    )
    assert result.dim == 2
    assert result.samples.shape == (4, 2)
    assert result.nll_nats is not None
    assert len(result.denoise) == 4
    assert result.history[-1]["loss"] == result.history[-1]["loss"]  # not NaN
