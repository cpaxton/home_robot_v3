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
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.
#
# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Keep this module light: ``from emet.controller.zmq_client import StretchZmqClient`` runs this
# file first; eager imports of Dynamem / instance / graph controllers pull ultralytics (YOLOE) and
# break ``emet serve mujoco`` on headless OpenCV. Heavy symbols load via PEP 562 ``__getattr__``.

from __future__ import annotations

import importlib
from typing import Any

from .base_controller import BaseController
from .base_robot_agent import BaseRobotAgent
from .zmq_client import HomeRobotZmqClient, StretchZmqClient
from .zmq_client import StretchZmqClient as RobotClient

__all__ = [
    "BaseController",
    "BaseRobotAgent",
    "DynamemController",
    "DynamemRobotAgent",
    "GraphEQAController",
    "RobotAgentGraphEQA",
    "InstanceMemoryController",
    "RobotAgent",
    "StretchZmqClient",
    "HomeRobotZmqClient",
    "RobotClient",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DynamemController": ("emet.controller.controller_dynamem", "DynamemController"),
    "DynamemRobotAgent": ("emet.controller.controller_dynamem", "RobotAgent"),
    "GraphEQAController": ("emet.controller.controller_graph_eqa", "GraphEQAController"),
    "RobotAgentGraphEQA": ("emet.controller.controller_graph_eqa", "RobotAgentGraphEQA"),
    "InstanceMemoryController": ("emet.controller.controller_instance_memory", "InstanceMemoryController"),
    "RobotAgent": ("emet.controller.controller_instance_memory", "RobotAgent"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
