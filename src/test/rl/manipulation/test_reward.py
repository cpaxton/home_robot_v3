import numpy as np
import pytest

from emet.rl.manipulation.env import running_cost


def test_safe_holding_is_preferred_to_early_failure_and_cost_is_cadence_invariant():
    # An eight-second held load inside all force/retention limits must not be
    # worse than an immediate drop (-20), even if it misses its tracking goal.
    totals = []
    for dt in (0.025, 0.05, 0.1):
        cost = running_cost(
            dt=dt, force=25, jerk=np.full(5, 10), action=np.full(6, 0.2), rotation=0.09, translation=0.015
        )
        totals.append(-round(8 / dt) * cost)
    assert min(totals) > -20
    assert totals == pytest.approx([totals[0]] * 3)


def test_running_cost_orders_loading_and_jerk_without_forbidding_contact():
    args = {"dt": 0.05, "force": 10, "jerk": np.zeros(5), "action": np.zeros(6), "rotation": 0, "translation": 0}
    gentle = running_cost(**args)
    assert gentle > 0
    assert running_cost(**{**args, "force": 25}) > gentle
    assert running_cost(**{**args, "jerk": np.full(5, 100)}) > gentle
