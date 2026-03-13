# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Smoke tests for the controller package: imports and basic instantiation
# (with dummy/mock robot where needed).
"""Smoke tests for emet.controller and DynamemController/InstanceMemoryController."""

import pytest

from emet.controller import (
    BaseRobotAgent,
    DynamemController,
    DynamemRobotAgent,
    StretchZmqClient,
    InstanceMemoryController,
    RobotAgent,
    RobotClient,
)
from emet.controller.base_controller import BaseController as BaseClass
from emet.controller.controller_instance_memory import InstanceMemoryController as IMC, RobotAgent as IMC_alias
from emet.controller.controller_dynamem import DynamemController as DC, RobotAgent as DC_alias
from emet.controller.controller_graph_eqa import GraphEQAController, RobotAgentGraphEQA


def test_controller_exports():
    """Package exports the expected symbols."""
    assert BaseRobotAgent is BaseClass
    assert RobotAgent is IMC_alias  # default export is InstanceMemoryController
    assert InstanceMemoryController is IMC
    assert DynamemController is DC
    assert DynamemRobotAgent is DC_alias
    assert RobotClient is StretchZmqClient


def test_base_robot_agent_abstract():
    """BaseRobotAgent is abstract and cannot be instantiated without get_voxel_map."""
    with pytest.raises(TypeError):
        BaseRobotAgent(robot=None, parameters={})


def test_instance_memory_controller_import():
    """InstanceMemoryController can be imported and has get_voxel_map."""
    assert hasattr(InstanceMemoryController, "get_voxel_map")
    assert hasattr(InstanceMemoryController, "get_robot")


def test_dynamem_controller_import():
    """DynamemController can be imported and has get_voxel_map."""
    assert hasattr(DynamemController, "get_voxel_map")
    assert hasattr(DynamemController, "get_robot")


def test_instance_memory_controller_is_base_subclass():
    """InstanceMemoryController subclasses BaseRobotAgent."""
    assert issubclass(InstanceMemoryController, BaseRobotAgent)


def test_dynamem_controller_is_base_subclass():
    """DynamemController subclasses BaseRobotAgent."""
    assert issubclass(DynamemController, BaseRobotAgent)


def test_graph_eqa_controller_import():
    """GraphEQAController can be imported and has run_eqa and update (overrides)."""
    assert hasattr(GraphEQAController, "run_eqa")
    assert hasattr(GraphEQAController, "run_eqa_one_iter")
    assert hasattr(GraphEQAController, "update")


def test_graph_eqa_controller_is_dynamem_subclass():
    """GraphEQAController subclasses DynamemController."""
    assert issubclass(GraphEQAController, DynamemController)
    assert GraphEQAController is RobotAgentGraphEQA
