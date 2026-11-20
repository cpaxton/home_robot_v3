"""Command-bound expert selection for the Mars simulator."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from emet.rl.manipulation.command_runtime import task_pose_teacher_action

FAMILIES = frozenset({"base_only", "arm_only", "coupled_hand"})


def tracking_family(task_mode):
    if task_mode in {"coupled", "hand_free_base"}:
        return "coupled_hand"
    if task_mode in FAMILIES:
        return task_mode
    raise ValueError(f"Unknown tracking task mode: {task_mode}")


@dataclass(frozen=True)
class BoundExpert:
    """One policy selection for the lifetime of an unloaded command."""

    family: str
    action_size: int
    policy: Callable
    arm_action_limit: np.ndarray | None

    def __call__(self, observation):
        action = np.asarray(self.policy(observation), dtype=float)
        if action.shape != (self.action_size,) or not np.isfinite(action).all():
            raise ValueError("Expert must return one finite action per command dimension")
        action = np.clip(action, -1, 1)
        action[-1] = 0.0
        if self.family == "arm_only":
            action[:2] = 0.0
            action[2:-1] = np.clip(action[2:-1], -self.arm_action_limit, self.arm_action_limit)
        elif self.family == "base_only":
            action[2:-1] = 0.0
        return action


def bind_expert(task_mode, control, qualified_experts, *, arm_speed_limit=None):
    """Bind a qualified learned actor, or the deterministic teacher, before stepping."""
    family = tracking_family(task_mode)
    if control.profile.robot != "innate_mars" or control.action_mode != "direct_velocity":
        raise ValueError("Expert routing requires the Mars direct-velocity simulator")
    arm_action_limit = None
    if family == "arm_only":
        if arm_speed_limit is None or not np.isfinite(arm_speed_limit) or arm_speed_limit <= 0:
            raise ValueError("Arm-only commands require a positive joint-speed limit")
        arm_action_limit = np.minimum(1, arm_speed_limit / np.asarray(control.profile.speed))
    policy = qualified_experts.get(family)
    if policy is None:

        def policy(_observation):
            return task_pose_teacher_action(control)

    return BoundExpert(family, control.profile.action_size, policy, arm_action_limit)
