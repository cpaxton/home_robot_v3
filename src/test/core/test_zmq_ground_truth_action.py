# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from emet.core.zmq_protocol import EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY, build_mujoco_ground_truth_dump_action


def test_build_mujoco_ground_truth_action_shape():
    a = build_mujoco_ground_truth_dump_action(
        12,
        "/tmp/out.json",
        exclude_robot=False,
        as_json=True,
    )
    assert a["step"] == 12
    assert EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY in a
    inner = a[EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY]
    assert inner["path"] == "/tmp/out.json"
    assert inner["exclude_robot"] is False
    assert inner["json"] is True
