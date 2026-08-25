# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OVMM find-phase rby1 fast-gate episodes."""

from __future__ import annotations

from emet.eval.ovmm_find_phase import load_find_phase_episodes


def test_rby1_fast_gate_episodes_present():
    eps = load_find_phase_episodes("configs/ovmm/find_phase_episodes.yaml")
    by_id = {e.id: e for e in eps}
    assert "default_table_rby1_s0_distinct_recep" in by_id
    assert "robocasa_rby1_pp_s1" in by_id
    assert by_id["default_table_rby1_s0_distinct_recep"].sim.endswith("default_table_rby1.yaml")
    assert by_id["robocasa_rby1_pp_s1"].sim.endswith("robocasa_pick_place_rby1.yaml")
    assert by_id["robocasa_rby1_pp_s1"].object_gt_body == "obj_main"


def test_prepare_default_table_rby1_mapping_view_moves_and_looks():
    from unittest.mock import MagicMock

    import numpy as np

    from emet.eval.ovmm_find_phase import _prepare_default_table_rby1_mapping_view

    robot = MagicMock()
    robot.get_emet_session.return_value = {
        "emet_robot_id": "rby1",
        "environment": {"kind": "default_table"},
    }
    agent = MagicMock()
    agent.robot = robot
    agent._find_phase_nav_step_timeout = lambda: 12.0

    assert _prepare_default_table_rby1_mapping_view(agent) is True

    robot.move_base_to.assert_called_once()
    goal = robot.move_base_to.call_args[0][0]
    np.testing.assert_allclose(goal, [0.0, 1.5, np.pi], rtol=0, atol=1e-6)
    robot.look_front.assert_called_once()
