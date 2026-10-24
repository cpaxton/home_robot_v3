# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from emet.utils.config import get_control_config


def test_control_config_lazy_interpolation_and_composition_boundary(tmp_path, monkeypatch):
    import pytest

    import emet.utils.config as config

    monkeypatch.setattr(config, "CONTROL_CONFIG_DIR", str(tmp_path))
    (tmp_path / "custom.yaml").write_text("speed: 0.2\ncontroller:\n  speed: ${speed}\n")
    cfg = get_control_config("custom")
    assert cfg.controller.speed == 0.2
    cfg.speed = 0.4
    assert cfg.controller.speed == 0.4
    (tmp_path / "composed.yaml").write_text("defaults: [custom]\n")
    with pytest.raises(ValueError, match="standalone"):
        get_control_config("composed")


def test_control_config_preserves_existing_hydra_application():
    # SAM2 initializes Hydra at import time. Exercise that state without a model.
    from contextlib import nullcontext

    context = nullcontext() if GlobalHydra.instance().is_initialized() else hydra.initialize(version_base="1.1")
    with context:
        application = GlobalHydra.instance().hydra
        for name in ("noplan_velocity_sim", "noplan_velocity_hw", "traj_follower"):
            cfg = get_control_config(name)
            assert isinstance(cfg, DictConfig)
            assert cfg == get_control_config(name + ".yaml")
        assert GlobalHydra.instance().hydra is application
