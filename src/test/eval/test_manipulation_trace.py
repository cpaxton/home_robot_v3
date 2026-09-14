# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import copy
import json

import numpy as np
import pytest

from emet.eval.manipulation_trace import ManipulationTrace, create_trace, score_recording, score_sequence, score_trace


def trace(pick_start=15, place_start=30, count=56):
    rows = []
    for i in range(count):
        holding = pick_start <= i < place_start
        placed = i >= place_start
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
    data.qvel[:] = np.arange(model.nv) * 0.1
    data.qacc_warmstart[:] = np.arange(model.nv) * 0.01
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
    np.testing.assert_array_equal(row["qvel"], data.qvel)
    np.testing.assert_array_equal(row["qacc_warmstart"], data.qacc_warmstart)
    assert row["act"] == data.act.tolist()
    invalid = copy.deepcopy(config)
    invalid["support_body"] = "object"
    with pytest.raises(ValueError):
        ManipulationTrace(model, invalid, tmp_path / "invalid.jsonl")


def test_ordered_sequence_preserves_all_completed_subgoals():
    result = score_sequence([trace(count=100), trace(pick_start=60, place_start=75, count=100)])
    assert result["verified"] and result["ordered"] and result["physical_place_success"]
    assert result["steps"][0]["first_place_time"] < result["steps"][0]["place_time"]
    assert result["steps"][1]["pick_start_time"] < result["steps"][1]["pick_time"]


@pytest.mark.parametrize("failure", ["reverse_order", "overlap", "lost_first", "failed_second", "different_clock"])
def test_sequence_rejects_individually_plausible_but_incomplete_tasks(failure):
    first = trace(count=100)
    second = trace(pick_start=60, place_start=75, count=100)
    if failure == "reverse_order":
        first, second = second, first
    elif failure == "overlap":
        second = trace(pick_start=35, place_start=60, count=100)
    elif failure == "lost_first":
        first[-1].update(support_contact=False, other_contact=True)
    elif failure == "failed_second":
        for row in second:
            row["gripper_contact"] = False
    else:
        second[-1]["sim_time"] += 0.1
    result = score_sequence([first, second])
    assert not result["ordered"] and not result["physical_place_success"]
    if failure == "different_clock":
        assert not result["verified"]


def test_sequence_recorder_manifest_roundtrip_and_invalid_targets(tmp_path):
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="support"><geom type="box" size="1 1 .1"/></body>
      <body name="one" pos="-.2 0 .2"><freejoint/><geom type="sphere" size=".1"/></body>
      <body name="two" pos=".2 0 .2"><freejoint/><geom type="sphere" size=".1"/></body>
      <body name="ee" pos="0 0 1"><body name="finger"/></body>
    </worldbody></mujoco>""")
    config = {
        "ee_body": "ee",
        "gripper_bodies": ["finger"],
        "steps": [{"object_body": name, "support_body": "support"} for name in ["one", "two"]],
    }
    path = tmp_path / "sequence.jsonl"
    writer = create_trace(model, config, path)
    data = mujoco.MjData(model)
    for index in range(2):
        data.time = index * 0.2
        mujoco.mj_forward(model, data)
        writer.record(model, data)
    writer.close()
    manifest = json.loads(path.read_text())
    assert manifest["schema"] == 2 and len(manifest["traces"]) == 2
    result = score_recording(path)
    assert not result["physical_place_success"]
    # Populate the same recorder-created files with synthetic ordered controls.
    for filename, rows in zip(manifest["traces"], [trace(count=100), trace(60, 75, 100)], strict=True):
        step_path = tmp_path / filename
        header = step_path.read_text().splitlines()[0]
        step_path.write_text(header + "\n" + "\n".join(json.dumps(row) for row in rows) + "\n")
    assert score_recording(path)["physical_place_success"]
    with pytest.raises(FileExistsError):
        create_trace(model, config, path)
    config["steps"][1]["object_body"] = "one"
    with pytest.raises(ValueError, match="distinct"):
        create_trace(model, config, tmp_path / "invalid.jsonl")
    manifest["traces"][1] = manifest["traces"][0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared physical subgoal"):
        score_recording(path)
