# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Managed CPU native smoke; no renderer, LLM, or task-success claim."""
import numpy as np
import pytest

from emet.controller.task.tamp.task_contracts import EpisodeSpec
from emet.eval.tamp_benchmark.adapters import RobosuiteAdapter
from emet.eval.tamp_benchmark.runtime import EpisodeRuntime, EpisodeStopped


@pytest.mark.sim
def test_full_robosuite_environment_uses_native_controller_and_horizon():
    robosuite = pytest.importorskip('robosuite')
    env = robosuite.make('Lift', robots='Panda', has_renderer=False,
                        has_offscreen_renderer=False, use_camera_obs=False,
                        horizon=3, ignore_done=False, seed=7)
    episode = EpisodeSpec('native_smoke', 'robosuite', 'Lift', 'seeded_diagnostic_7',
                          7, 'Lift the cube', 3)
    adapter = RobosuiteAdapter(env, suite='robosuite',
                              install_reset=lambda env, spec: env.reset(),
                              success=env._check_success)
    with EpisodeRuntime(adapter, episode) as runtime:
        # Controller holds the initial EE; a failed Lift is the expected control.
        for _ in range(3):
            result = runtime.step(np.zeros(env.action_dim))
        assert result.terminated and result.truncated
        assert env.timestep == runtime.control_steps == 3
        assert runtime.scores(final_goal=None, physical_integrity=None)['upstream_success'] is False
        with pytest.raises(EpisodeStopped):
            runtime.step(np.zeros(env.action_dim))
