import numpy as np
import pytest

from emet.rl.manipulation.action_audit import grouped_action_rms
from emet.rl.manipulation.profiles import PROFILES


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_action_statistics_follow_base_then_arm_then_grip(robot):
    profile = PROFILES[robot]
    contract = profile.contract("direct_velocity")
    values = np.zeros(profile.action_size)
    offset = 2 if profile.wheel_bodies else 0
    values[offset : offset + profile.arm_size] = 1
    assert grouped_action_rms(values, contract) == {"base_rms": 0, "arm_rms": 1, "grip_rms": 0}
    values[:] = 0
    values[-1] = 3
    assert grouped_action_rms(values, contract) == {"base_rms": 0, "arm_rms": 0, "grip_rms": 3}
    if offset:
        values[:] = 0
        values[:offset] = 2
        assert grouped_action_rms(values, contract) == {"base_rms": 2, "arm_rms": 0, "grip_rms": 0}


def test_action_statistics_reject_unknown_layout_or_invalid_rms():
    contract = PROFILES["innate_mars"].contract("direct_velocity")
    for values in [np.zeros(7), np.full(8, np.nan), np.full(8, -1)]:
        with pytest.raises(ValueError):
            grouped_action_rms(values, contract)
