# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Robot-specific emote implementations (Stretch vs generic mobile manipulators)."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

from emet.controller.operations.emote import (
    AvertGazeOperation,
    NodHeadOperation,
    ShakeHeadOperation,
    WaveOperation,
)
from emet.core.robot import AbstractRobotClient
from emet.core.task import Operation, Task
from emet.robots import ROBOT_REGISTRY
from emet.robots.base import RobotBackend


class EmoteBackend(ABC):
    """Builds emote tasks for a robot family (Stretch head/arm vs dual-arm, etc.)."""

    @abstractmethod
    def add_named_emote(self, task: Task, name: str, agent: Any) -> None:
        """Append operation(s) to *task* for emote *name* (e.g. ``wave``, ``nod_head``)."""


class StretchEmoteBackend(EmoteBackend):
    """Hello Stretch: head pan/tilt + single-arm manipulation (existing operations)."""

    def add_named_emote(self, task: Task, name: str, agent: Any) -> None:
        if name == "nod" or name == "nod_head":
            agent.robot_say("I'm nodding my head.", discord_action_italic=True)
            task.add_operation(NodHeadOperation("emote", agent))
        elif name == "shake" or name == "shake_head":
            agent.robot_say("I'm shaking my head.", discord_action_italic=True)
            task.add_operation(ShakeHeadOperation("emote", agent))
        elif name == "wave":
            agent.robot_say("I'm waving my hand.", discord_action_italic=True)
            task.add_operation(WaveOperation("emote", agent))
        elif name == "avert" or name == "avert_gaze":
            agent.robot_say("I'm looking away.", discord_action_italic=True)
            task.add_operation(AvertGazeOperation("emote", agent))
        else:
            raise ValueError(f"Invalid emote operation: {name}")


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


class GenericEmoteBackend(EmoteBackend):
    """Non-Stretch robots: no Stretch-style arm/head trajectories; speech + noop task step."""

    def __init__(self, robot_id: str) -> None:
        self._robot_id = robot_id

    def add_named_emote(self, task: Task, name: str, agent: Any) -> None:
        msg = (
            f"Gesture {name!r} is not implemented for robot {self._robot_id!r} "
            "(Stretch-only motion). Saying it instead."
        )
        task.add_operation(VoiceOnlyEmote(agent, msg))


def _instantiate_registry_backend(robot_key: str) -> RobotBackend | None:
    """Return a RobotBackend instance for a registry key (same discovery as ``run_agent``)."""
    k = robot_key.lower().replace("-", "_")
    if k not in ROBOT_REGISTRY:
        return None
    mod = importlib.import_module(ROBOT_REGISTRY[k])
    backend_cls = None
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, RobotBackend) and attr is not RobotBackend:
            if hasattr(attr, "get_spec") and attr_name != "RobotBackend":
                backend_cls = attr
                break
    if backend_cls is None:
        return None
    return backend_cls()


def resolve_emote_backend(robot: AbstractRobotClient) -> EmoteBackend:
    """Pick an emote backend from the live ZMQ client (Stretch vs Generic / registry)."""
    from emet.controller.generic_zmq_client import GenericZmqClient
    from emet.controller.zmq_client import StretchZmqClient

    if isinstance(robot, StretchZmqClient):
        return StretchEmoteBackend()
    if isinstance(robot, GenericZmqClient):
        inst = _instantiate_registry_backend(robot.spec.name)
        if inst is not None:
            return inst.get_emote_backend()
        return GenericEmoteBackend(robot.spec.name)
    return GenericEmoteBackend("unknown")
