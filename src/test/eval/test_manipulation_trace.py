# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import copy
import json

import numpy as np
import pytest

from emet.eval.manipulation_trace import ManipulationTrace, score_trace


def trace():
    rows = []
    for i in range(56):
        holding = 15 <= i < 30
        placed = i >= 30
        rows.append(
            {
                "sim_time": i / 10,
                "object_pos": [0.2 if placed else 0, 0, 0.6 if holding else 0.5],
                "object_rot": np.eye(3).ravel().tolist(),
                "relative_pos": [0, 0, 0.01] if holding else [0, 0, -0.2],
                "relative_rot": np.eye(3).ravel().tolist(),
                "gripper_contact": holding,
                "support_contact": placed,
                "other_contact": not (holding or placed),
            }
        )
    return rows


def test_sustained_pick_and_released_place():
    result = score_trace(trace())
    assert result["verified"]
    assert result["physical_pick_success"]
    assert result["physical_place_success"]


@pytest.mark.parametrize("failure", ["knocked", "slipping", "rotating", "never_lifted", "supported", "gap"])
def test_false_pick_controls(failure):
    rows = trace()
    for i, row in enumerate(rows[15:30]):
        if failure == "knocked":
            row["gripper_contact"] = False
        elif failure == "slipping":
            row["relative_pos"][0] = i / 100
        elif failure == "rotating":
            angle = i / 10
            row["relative_rot"] = [np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle), 0, 0, 0, 1]
        elif failure == "never_lifted":
            row["object_pos"][2] = 0.52
        elif failure == "supported":
            row["other_contact"] = True
    if failure == "gap":
        rows = [row for i, row in enumerate(rows) if i not in range(19, 25)]
    result = score_trace(rows)
    assert not result["physical_pick_success"]
    assert not result["physical_place_success"]


@pytest.mark.parametrize("failure", ["wrong_support", "still_held", "dropped_later", "sliding"])
def test_false_place_controls(failure):
    rows = trace()
    for i, row in enumerate(rows[30:]):
        if failure == "wrong_support":
            row.update(support_contact=False, other_contact=True)
        elif failure == "still_held":
            row["gripper_contact"] = True
        elif failure == "sliding":
            row["object_pos"][0] = i / 100
    if failure == "dropped_later":
        rows[-1].update(support_contact=False, other_contact=True)
    result = score_trace(rows)
    assert result["physical_pick_success"]
    assert not result["physical_place_success"]


def test_missing_or_invalid_trace_is_unverified():
    assert not score_trace([])["verified"]
    rows = trace()
    rows[3]["sim_time"] = 0
    assert not score_trace(rows)["verified"]
    rows = trace()
    rows[3]["object_pos"][0] = float("nan")
    assert not score_trace(rows)["verified"]


def test_mujoco_recorder_keeps_contact_and_pose_evidence_private(tmp_path):
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="support"><geom type="box" size="1 1 .1"/></body>
      <body name="object" pos="0 0 .2"><freejoint/><geom type="sphere" size=".1"/></body>
      <body name="ee" pos="0 0 1"><body name="finger"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = {"object_body": "object", "support_body": "support", "ee_body": "ee", "gripper_bodies": ["finger"]}
    original_qpos = data.qpos.copy()
    path = tmp_path / "trace.jsonl"
    writer = ManipulationTrace(model, config, path)
    writer.record(model, data)
    writer.record(model, data)  # Rate-limited; no duplicate simulation times.
    writer.close()
    np.testing.assert_array_equal(data.qpos, original_qpos)
    header, row = [json.loads(line) for line in path.read_text().splitlines()]
    assert header["config"] == config
    assert row["support_contact"]
    assert not row["gripper_contact"]
    assert row["contacts"]
    invalid = copy.deepcopy(config)
    invalid["support_body"] = "object"
    with pytest.raises(ValueError):
        ManipulationTrace(model, invalid, tmp_path / "invalid.jsonl")
