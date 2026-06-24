# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import numpy as np
import pytest

from emet.eval.world_fuzz import (
    FuzzAction,
    fuzz_actions_for_cycle,
    random_fuzz_actions,
    scripted_fuzz_actions,
)

PLACEMENTS = {
    "obj_main": {"cat": "apple", "pos": [1.0, 2.0, 0.9], "quat": [1, 0, 0, 0]},
    "vase_1": {"cat": "vase", "pos": [-0.5, 0.5, 0.8], "quat": [1, 0, 0, 0]},
}


def test_scripted_moves_and_doors():
    cycle = {
        "moves": [
            {"body": "obj_main", "delta": [1.5, 0.5, 0.0]},
            {"body": "vase_1", "pos": [0.0, 0.0, 0.8]},
        ],
        "doors": [{"joint": "cab_main_group_leftdoorhinge", "value": 1.4}],
    }
    actions = scripted_fuzz_actions(cycle, PLACEMENTS)
    assert len(actions) == 3
    move = actions[0]
    assert move.kind == "move" and move.target == "obj_main"
    np.testing.assert_allclose(move.pos, [2.5, 2.5, 0.9])
    assert actions[1].pos == (0.0, 0.0, 0.8)
    door = actions[2]
    assert door.kind == "joint" and door.value == 1.4


def test_scripted_move_unknown_body_raises():
    with pytest.raises(RuntimeError, match="not in sim_object_placements"):
        scripted_fuzz_actions({"moves": [{"body": "nope", "delta": [1, 0, 0]}]}, PLACEMENTS)


def test_random_fuzz_deterministic_with_seed():
    spec = {
        "bodies": ["obj_main", "vase_1"],
        "joints": ["door_a", "door_b"],
        "n_moves": 1,
        "n_doors": 1,
        "move_radius_m": 0.8,
        "joint_values": [0.0, 1.2],
    }
    a1 = random_fuzz_actions(spec, PLACEMENTS, np.random.default_rng(7))
    a2 = random_fuzz_actions(spec, PLACEMENTS, np.random.default_rng(7))
    assert a1 == a2
    move = next(a for a in a1 if a.kind == "move")
    base = np.asarray(PLACEMENTS[move.target]["pos"])
    dist = float(np.linalg.norm(np.asarray(move.pos[:2]) - base[:2]))
    assert 0.3 <= dist <= 0.8
    assert move.pos[2] == base[2]
    door = next(a for a in a1 if a.kind == "joint")
    assert door.value in (0.0, 1.2)


def test_random_fuzz_requires_candidates():
    with pytest.raises(RuntimeError, match="candidate 'bodies'"):
        random_fuzz_actions({"n_moves": 1}, PLACEMENTS, np.random.default_rng(0))


def test_fuzz_actions_for_cycle_combines_scripted_and_random():
    cycle = {
        "moves": [{"body": "obj_main", "delta": [0.5, 0.0, 0.0]}],
        "random": {"seed": 3, "joints": ["door_a"], "n_doors": 1, "n_moves": 0},
    }
    actions = fuzz_actions_for_cycle(cycle, PLACEMENTS)
    kinds = [a.kind for a in actions]
    assert kinds == ["move", "joint"]
    assert fuzz_actions_for_cycle(None, PLACEMENTS) == []


def test_apply_fuzz_actions_sends_zmq(monkeypatch):
    sent: list[tuple] = []

    class FakeRobot:
        _last_step = 10

        def send_action(self, action, reliable=False):
            sent.append((action, reliable))

    import emet.eval.world_fuzz as wf

    monkeypatch.setattr(
        "emet.simulation.sim_manipulation.time.sleep",
        lambda *_: None,
    )
    actions = [
        FuzzAction(kind="move", target="obj_main", pos=(1.0, 2.0, 0.9)),
        FuzzAction(kind="joint", target="door_a", value=1.2),
    ]
    applied = wf.apply_fuzz_actions(FakeRobot(), actions)
    assert len(sent) == 2
    assert sent[0][0]["sim_set_body_pose"]["body"] == "obj_main"
    assert sent[1][0]["sim_set_joint_qpos"]["joint"] == "door_a"
    assert all(rel for _, rel in sent)
    assert applied[0]["kind"] == "move" and applied[1]["value"] == 1.2
