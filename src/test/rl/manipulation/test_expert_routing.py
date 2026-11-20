from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation.expert_routing import bind_expert


def control():
    return SimpleNamespace(
        profile=SimpleNamespace(robot="innate_mars", speed=[0.5] * 5, action_size=8),
        action_mode="direct_velocity",
    )


@pytest.mark.parametrize(
    "family,expected",
    [
        ("base_only", [1, 1, 0, 0, 0, 0, 0, 0]),
        ("arm_only", [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]),
        ("coupled_hand", [1, 1, 1, 1, 1, 1, 1, 0]),
    ],
)
def test_expert_is_bound_and_executor_limits_authority(family, expected):
    experts = {family: lambda obs: np.ones(8)}
    command = bind_expert(family, control(), experts, arm_speed_limit=0.25)
    experts[family] = lambda obs: -np.ones(8)
    np.testing.assert_array_equal(command(None), expected)


def test_arm_only_refuses_missing_speed_limit():
    with pytest.raises(ValueError, match="speed limit"):
        bind_expert("arm_only", control(), {})


def test_missing_expert_binds_teacher_before_the_first_tick(monkeypatch):
    from emet.rl.manipulation import expert_routing

    plant = control()
    monkeypatch.setattr(
        expert_routing, "task_pose_teacher_action", lambda current: np.ones(8) if current is plant else None
    )
    command = bind_expert("hand_free_base", plant, {})
    np.testing.assert_array_equal(command(None), [1, 1, 1, 1, 1, 1, 1, 0])


def test_bad_action_fails_instead_of_changing_controller():
    command = bind_expert("base_only", control(), {"base_only": lambda obs: [np.nan] * 8})
    with pytest.raises(ValueError, match="finite action"):
        command(None)
