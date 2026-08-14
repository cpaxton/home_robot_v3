# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for RoboCasa floor pick/place episodes + the MCTS manip phase (no sim)."""

from __future__ import annotations

import numpy as np

from emet.eval.ovmm_find_phase import load_find_phase_episodes
from emet.eval.ovmm_full import drop_object_to_floor


def test_full_episodes_include_floor_variants():
    eps = load_find_phase_episodes("configs/ovmm/full_episodes.yaml")
    floor = [e for e in eps if e.floor_object]
    assert len(floor) >= 3
    ids = {e.id for e in floor}
    assert "robocasa_rby1_floor_to_counter_mcts" in ids
    assert "robocasa_sourccey_floor_to_cab_sim" in ids
    for e in floor:
        assert e.start_recep == "floor"
        assert e.object_gt_body


def test_floor_episode_schema_roundtrip():
    eps = load_find_phase_episodes("configs/ovmm/full_episodes.yaml")
    mcts = [e for e in eps if e.id == "robocasa_rby1_floor_to_counter_mcts"][0]
    assert mcts.floor_object is True
    assert mcts.start_recep == "floor"
    assert mcts.goal_recep == "counter"
    assert mcts.sim == "configs/sim/robocasa_pick_place_rby1.yaml"


class _FakeRobot:
    """Robot whose sim body-pose writes update the placements dict."""

    def __init__(self, placements: dict):
        self._placements = placements
        self._last_step = 0

    def get_emet_session(self):
        return {"sim_object_placements": self._placements}

    def get_base_pose(self):
        return np.array([0.0, 0.0, 0.0])

    def get_emet_robot_id(self):  # pragma: no cover
        return "rby1"

    def send_action(self, action: dict, **kwargs) -> None:
        from emet.core.zmq_protocol import EMET_ACTION_SIM_SET_BODY_POSE_KEY

        payload = action.get(EMET_ACTION_SIM_SET_BODY_POSE_KEY) or {}
        body = str(payload.get("body") or "")
        pos = payload.get("pos")
        if body in self._placements and pos is not None:
            self._placements[body]["pos"] = list(np.asarray(pos, dtype=np.float64).reshape(3))


def test_drop_object_to_floor_lowers_z():
    placements = {
        "obj_main": {"cat": "obj", "pos": [0.3, -0.5, 1.0]},
    }
    robot = _FakeRobot(placements)
    ok = drop_object_to_floor(robot, "obj_main", placements)
    assert ok
    assert placements["obj_main"]["pos"][2] == 0.02


def test_drop_object_to_floor_missing_body():
    robot = _FakeRobot({})
    assert not drop_object_to_floor(robot, "nope", {})
