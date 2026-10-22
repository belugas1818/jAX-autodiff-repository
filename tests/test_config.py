import pytest

from jax_diffusion.config import TrainConfig


def test_default_config_is_valid():
    cfg = TrainConfig()
    assert cfg.data == "points2d"
    assert cfg.temb % 2 == 0


def test_rejects_bad_data_and_hparams():
    with pytest.raises(ValueError, match="data must be"):
        TrainConfig(data="images")
    with pytest.raises(ValueError, match="csv_path"):
        TrainConfig(data="csv")
    with pytest.raises(ValueError, match="lr"):
        TrainConfig(lr=0.0)
    with pytest.raises(ValueError, match="jac_weight"):
        TrainConfig(jac_weight=-1.0)
    with pytest.raises(ValueError, match="even"):
        TrainConfig(temb=31)
    with pytest.raises(ValueError, match="ema_decay"):
        TrainConfig(ema_decay=1.0)
