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
# This source code is licensed under the LICENSE file in the root directory of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Any

from emet.controller.emotes.backend import EmoteBackend, resolve_emote_backend
from emet.core.task import Operation, Task


class EmoteTask:
    """
    Creates a task queue with a given emote operation.
    """

    def __init__(self, agent: Any, emote_backend: EmoteBackend | None = None):
        super().__init__()

        # random stuff that has to be synced...
        self.agent = agent
        self.navigation_space = agent.space
        self.parameters = agent.parameters
        self.robot = agent.robot
        self._emote_backend = emote_backend or resolve_emote_backend(agent.robot)

    def get_task(self, emote_operation: Operation | str) -> Task:
        task = Task()
        if isinstance(emote_operation, Operation):
            task.add_operation(emote_operation)
            return task
        if isinstance(emote_operation, str):
            self._emote_backend.add_named_emote(task, emote_operation, self.agent)
            return task
        raise TypeError(f"Expected Operation or str, got {type(emote_operation)}")
