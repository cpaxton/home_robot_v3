# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from emet.utils.config import get_control_config


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
