from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation import arm_spline_benchmark as benchmark


@pytest.mark.parametrize("terminates", [True, False])
def test_episode_waits_for_decimal_horizon_receipt_and_bounds_nontermination(monkeypatch, terminates):
    class Episode:
        def __init__(self):
            self.unwrapped = self
            self.config = SimpleNamespace(horizon=0.3)
            self.profile = SimpleNamespace(dt=0.1)
            self.data = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1))
            self.steps, self._physics_invalid = 0, False

        def _info(self):
            return defaultdict(float)

        def reset(self, **kwargs):
            return dict.fromkeys(("history", "preview", "motion"), np.zeros(1)), {
                "reference_sha256": "reference",
                "spline_admission": {},
                "spline_rejections": [],
            }

        def step(self, action):
            self.steps += 1
            done = terminates and self.steps >= 3
            return (
                dict.fromkeys(("history", "preview", "motion"), np.zeros(1)),
                0,
                done,
                False,
                {
                    "primitive_success": done,
                    "code": "ok" if done else "running",
                    "spline_peak_path_error": [0, 0],
                    "joint_limit_push_ticks": 0,
                },
            )

    monkeypatch.setattr(benchmark, "stop", lambda _: True)
    row = benchmark.episode(Episode(), {"id": "decimal-horizon"}, policy=lambda _: np.zeros(1))
    assert row["ticks"] == (3 if terminates else 4)
    assert row["success"] == terminates
    assert row["code"] == ("ok" if terminates else "spline_execution_budget_exhausted")
    assert row["stop_confirmed"]
