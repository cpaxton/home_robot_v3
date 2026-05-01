# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Innate Mars emotes live here (not under ``emet.controller``) so imports avoid pulling Dynamem."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import sleep
from typing import Any

import numpy as np

from emet.core.task import Operation, Task

from . import INNATE_MARS_JOINT_NAMES


class EmoteBackend(ABC):
    """Same contract as :class:`emet.controller.emotes.backend.EmoteBackend` (duck-type)."""

    @abstractmethod
    def add_named_emote(self, task: Task, name: str, agent: Any) -> None:
        """Append operation(s) for emote *name*."""


class VoiceOnlyEmote(Operation):
    """Single-step operation that only calls ``agent.robot_say`` (no motion)."""

    def __init__(self, agent: Any, message: str) -> None:
        super().__init__(name="voice_emote")
        self._agent = agent
        self._message = message

    def can_start(self) -> bool:
        return True

    def run(self) -> None:
        self._started = True
        self._agent.robot_say(self._message, discord_action_italic=True)

    def was_successful(self) -> bool:
        return True


def _joint_index(robot: Any, name: str) -> int | None:
    if hasattr(robot, "_spec") and getattr(robot, "_spec", None) is not None:
        names = robot._spec.joint_names
        return names.index(name) if name in names else None
    if name in INNATE_MARS_JOINT_NAMES:
        return INNATE_MARS_JOINT_NAMES.index(name)
    return None


class InnateMarsWaveOperation(Operation):
    """Wave using wrist / gripper joints (joint5 oscillation)."""

    def __init__(self, name: str, agent: Any) -> None:
        super().__init__(name)
        self.robot = agent.robot

    def can_start(self) -> bool:
        return True

    def run(
        self,
        n_waves: int = 3,
        joint5_amplitude: float = 0.35,
        settle_time_s: float = 0.35,
    ) -> None:
        robot = self.robot
        if hasattr(robot, "switch_to_manipulation_mode"):
            robot.switch_to_manipulation_mode()

        q, _, _ = robot.get_joint_state()
        q = np.asarray(q, dtype=float).reshape(-1)
        i5 = _joint_index(robot, "joint5")
        i6 = _joint_index(robot, "joint6")
        if i5 is None or q.size <= i5:
            self._started = True
            return

        base5 = float(q[i5])

        def send_joint5(delta: float, grip_delta: float | None) -> None:
            nonlocal q
            q, _, _ = robot.get_joint_state()
            q = np.asarray(q, dtype=float).reshape(-1)
            pos: dict[str, float] = {"joint5": base5 + delta}
            if grip_delta is not None and i6 is not None and q.size > i6:
                pos["joint6"] = float(np.clip(float(q[i6]) + grip_delta, -1.5, 1.5))
            if hasattr(robot, "set_joint_positions"):
                robot.set_joint_positions(pos)

        for _ in range(n_waves):
            send_joint5(joint5_amplitude, 0.02)
            sleep(settle_time_s)
            send_joint5(-joint5_amplitude, -0.02)
            sleep(settle_time_s)

        send_joint5(0.0, None)
        if i6 is not None and hasattr(robot, "set_joint_positions"):
            q, _, _ = robot.get_joint_state()
            q = np.asarray(q, dtype=float).reshape(-1)
            if q.size > i6:
                robot.set_joint_positions({"joint6": float(q[i6])})

        self._started = True

    def was_successful(self) -> bool:
        return True


class InnateMarsEmoteBackend(EmoteBackend):
    """Innate Mars / Maurice: arm wave; head emotes are speech-only (no pan-tilt DOFs)."""

    def add_named_emote(self, task: Task, name: str, agent: Any) -> None:
        if name == "wave":
            agent.robot_say("I'm waving.", discord_action_italic=True)
            task.add_operation(InnateMarsWaveOperation("emote", agent))
        elif name in ("nod", "nod_head"):
            agent.robot_say(
                "This robot has a fixed stereo head (no nod motion); acknowledging instead.",
                discord_action_italic=True,
            )
            task.add_operation(VoiceOnlyEmote(agent, "(Innate Mars: nod emote has no head joints — acknowledged.)"))
        elif name in ("shake", "shake_head"):
            agent.robot_say(
                "This robot has a fixed stereo head (no shake motion); acknowledging instead.",
                discord_action_italic=True,
            )
            task.add_operation(VoiceOnlyEmote(agent, "(Innate Mars: shake emote has no head joints — acknowledged.)"))
        elif name in ("avert", "avert_gaze"):
            agent.robot_say(
                "This robot has a fixed stereo head (no gaze avert motion); acknowledging instead.",
                discord_action_italic=True,
            )
            task.add_operation(VoiceOnlyEmote(agent, "(Innate Mars: avert emote has no head joints — acknowledged.)"))
        else:
            raise ValueError(f"Invalid emote operation: {name}")
