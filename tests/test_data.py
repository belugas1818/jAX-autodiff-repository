from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jax_diffusion.data import csv_sampler, make_spec


def test_points2d_normalizer_and_roundtrip():
    spec = make_spec("points2d")
    key = random.key(0)
    spec.fit_normalizer(key, n=2000)
    x = spec.sample(random.key(1), 128)
    assert x.shape == (128, 2)
    recovered = spec.denormalize(x)
    assert recovered.shape == x.shape
    assert float(jnp.abs(x.mean()) ) < 0.15
    assert 0.8 < float(x.std()) < 1.2


def test_sample_before_fit_fails():
    spec = make_spec("signal1d")
    with pytest.raises(RuntimeError, match="fit_normalizer"):
        spec.sample(random.key(0), 4)


def test_csv_adapter(tmp_path: Path):
    path = tmp_path / "rows.csv"
    np.savetxt(path, np.arange(20, dtype=np.float32).reshape(10, 2), delimiter=",")
    shape, sampler = csv_sampler(path)
    assert shape == (2,)
    batch = sampler(random.key(0), 5)
    assert batch.shape == (5, 2)

    spec = make_spec("csv", str(path))
    spec.fit_normalizer(random.key(1), n=50)
    assert spec.sample(random.key(2), 3).shape == (3, 2)


def test_csv_rejects_nans(tmp_path: Path):
    path = tmp_path / "bad.csv"
    np.savetxt(path, np.array([[1.0, np.nan], [2.0, 3.0]]), delimiter=",")
    with pytest.raises(ValueError, match="non-finite"):
        csv_sampler(path)
