from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation.__main__ import Monitor
from emet.rl.manipulation.env import ManipulationConfig


def test_reach_subtype_fractions_use_all_steps_and_clear_between_rollouts(tmp_path):
    monitor = Monitor(tmp_path, ManipulationConfig(), "test", interval=1000, episodes=1)
    monitor.model = SimpleNamespace(logger=SimpleNamespace(name_to_value={}))
    records = []
    monitor.record = records.append
    infos = [
        {"scenario": "reach", "reach_hold_trial": True},
        {"scenario": "reach", "reach_local_trial": True, "reach_local_mode": "translation"},
        {"scenario": "pregrasp"},
        {"scenario": "reach", "reach_local_trial": True, "reach_local_mode": "orientation"},
    ]
    for info in infos:
        info.update(is_success=False, retention_lost=False, unsafe_contact=False, initialization_failed=False, peak_contact_force=0)
        monitor.locals = {"dones": [True], "infos": [info], "rewards": np.zeros(1)}
        monitor._on_step()
    monitor._on_rollout_end()
    assert records[0]["sampling/reach/step_fraction"] == pytest.approx(0.75)
    assert records[0]["sampling/reach_local_translation/step_fraction"] == pytest.approx(0.25)
    assert records[0]["sampling/reach_local_translation/episode_fraction"] == pytest.approx(0.25)
    assert sum(records[0][f"sampling/reach_{mode}/step_fraction"] for mode in ["hold", "local_translation", "local_orientation", "local_combined", "local_unspecified", "ordinary"]) == pytest.approx(0.75)
    monitor._on_rollout_end()
    assert records[1]["sampling/reach_local_translation/step_fraction"] == 0
