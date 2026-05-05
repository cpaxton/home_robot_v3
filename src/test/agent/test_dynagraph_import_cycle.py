# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Regression: agent package must not force an import cycle into controller_dynamem."""

from unittest.mock import MagicMock

from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.task.dynamem import EQAExecuter


def test_env_flags_then_dynagraph_controller_imports():
    """Same order as ``controller_dynamem`` (env_flags) then dynagraph stack."""
    from emet.agent import env_flags

    assert hasattr(env_flags, "env_agent_camera_debug")
    assert issubclass(DynagraphController, GraphEQAController)


def test_run_agent_with_robot_lazy_export():
    """``run_agent_with_robot`` is resolved via ``emet.agent`` module ``__getattr__``."""
    import emet.agent as agent_pkg

    fn = agent_pkg.run_agent_with_robot
    assert callable(fn)
    assert agent_pkg.run_agent_with_robot is fn


def test_eqa_executer_rotate_in_place_for_explore_entry():
    """EQAExecuter.rotate_in_place delegates to the agent (scan / exploration entry)."""
    agent = MagicMock()
    ex = EQAExecuter(agent)
    ex.rotate_in_place()
    agent.rotate_in_place.assert_called_once()
