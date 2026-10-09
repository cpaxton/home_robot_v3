# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Joint-space Sourccey emotes for the Emet simulation protocol.

These trajectories respect joint limits but are not collision-planned. Execute in
clear space. Vendor hardware requires a calibrated adapter before using them.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np

from emet.core.robot import AbstractRobotClient
from emet.core.task import Operation, Task
from emet.robots.sourccey import SourcceyBackend


@lru_cache(maxsize=1)
def _limits() -> np.ndarray:
    import mujoco

    spec = SourcceyBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    return np.array([model.actuator(name).ctrlrange for name in spec.actuator_names])


def wave_trajectory(q: np.ndarray, side: str = "left", dt: float = 0.05) -> np.ndarray:
    """Dense actuator targets: raise, fan wrist three times, return to starting pose.

    Positions use radians/metres; the three planar base actuators use velocity
    targets and stay zero even when the robot is away from the world origin.
    Quintic ramps bound commanded arm speed to 0.8 rad/s. Other position joints
    hold their measured initial pose (clipped for small solver limit violations).
    """
    spec = SourcceyBackend().get_spec()
    if side not in spec.arm_chains:
        raise ValueError("side must be 'left' or 'right'")
    if not np.isfinite(dt) or not 0.01 <= dt <= 0.1:
        raise ValueError("dt must be between 0.01 and 0.1 seconds")
    q = np.asarray(q, dtype=float)
    if q.shape != (spec.dof,) or not np.isfinite(q).all():
        raise ValueError(f"Expected a finite Sourccey joint state of length {spec.dof}")
    limits = _limits()
    if np.any(q[3:] < limits[3:, 0] - 0.03) or np.any(q[3:] > limits[3:, 1] + 0.03):
        raise ValueError("Measured joints are outside Sourccey limits")
    start = q.copy()
    start[:3] = 0.0
    start[3:] = np.clip(start[3:], limits[3:, 0], limits[3:, 1])
    chain = spec.arm_chains[side]
    indices = [spec.joint_names.index(name) for name in chain.joint_names]
    raised = start.copy()
    raised[indices] = chain.home_arm_q
    wrist = spec.joint_names.index(f"{side}_wrist_roll")
    targets = [raised]
    for _ in range(3):
        for roll in (0.5, -0.5):
            target = raised.copy()
            target[wrist] = roll
            targets.append(target)
    targets.extend([raised, start])
    frames = [start]
    for target in targets:
        previous = frames[-1]
        # Peak derivative of quintic smoothstep is 1.875.
        duration = max(0.5, 1.875 * np.max(np.abs(target - previous)) / 0.8)
        steps = int(np.ceil(duration / dt))
        for u in np.linspace(0, 1, steps + 1)[1:]:
            blend = u**3 * (10 - 15 * u + 6 * u**2)
            frames.append(previous + blend * (target - previous))
    return np.asarray(frames)


class SourcceyWaveOperation(Operation):
    """Feedback-checked wave, preserving the other arm, grippers, and lift."""

    def __init__(self, name: str, agent: Any, side: str = "left") -> None:
        super().__init__(name)
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.robot: AbstractRobotClient = agent.robot
        self._spec = SourcceyBackend().get_spec()
        self.side = side
        self.error: str | None = None
        self._success = False

    def can_start(self) -> bool:
        return bool(self.robot.at_goal())

    def run(self) -> None:
        self._started, self._success, self.error = True, False, None
        last = None
        try:
            if not self.can_start():
                raise RuntimeError("Finish base navigation before waving")
            q, _, _ = self.robot.get_joint_state(timeout=1.0)
            frames = wave_trajectory(q, self.side)
            self.robot.switch_to_manipulation_mode()
            for target in frames:
                self.robot.set_actuator_positions(target)
                last = target
                time.sleep(0.05)
                measured, _, _ = self.robot.get_joint_state(timeout=1.0)
                if measured is None or np.shape(measured) != (self._spec.dof,) or not np.isfinite(measured).all():
                    raise RuntimeError("Missing or invalid joint feedback during wave")
                error = np.abs(np.asarray(measured) - target)
                if error[3] > 0.03 or np.max(error[4:]) > 0.25:
                    raise RuntimeError("Wave tracking error exceeded 0.03 m / 0.25 rad")
                base_error = np.asarray(measured)[:3] - np.asarray(q)[:3]
                yaw_error = np.arctan2(np.sin(base_error[2]), np.cos(base_error[2]))
                if np.linalg.norm(base_error[:2]) > 0.05 or abs(yaw_error) > 0.1:
                    raise RuntimeError("Base moved during wave")
            time.sleep(0.5)
            measured, _, _ = self.robot.get_joint_state(timeout=1.0)
            if measured is None or np.shape(measured) != (self._spec.dof,) or not np.isfinite(measured).all():
                raise RuntimeError("Missing final joint feedback")
            error = np.abs(np.asarray(measured) - frames[-1])
            self._success = bool(error[3] < 0.02 and np.max(error[4:]) < 0.08)
            if not self._success:
                self.error = "Wave did not return to its starting pose within tolerance"
        except (ValueError, RuntimeError, OSError) as exc:
            self.error = str(exc)
            # Stop advancing the gesture; keep the last commanded pose and zero base speed.
            if last is not None:
                try:
                    self.robot.set_actuator_positions(last)
                except Exception:
                    pass

    def was_successful(self) -> bool:
        return self._success


class SourcceyEmoteBackend:
    """EmoteBackend contract, imported without the optional perception stack."""

    def add_named_emote(self, task: Task, name: str, agent: Any) -> None:
        if name in ("wave", "wave_left", "wave_right"):
            side = "right" if name == "wave_right" else "left"
            task.add_operation(SourcceyWaveOperation("emote", agent, side=side))
        else:
            from emet.controller.emotes.backend import GenericEmoteBackend

            GenericEmoteBackend("sourccey").add_named_emote(task, name, agent)
