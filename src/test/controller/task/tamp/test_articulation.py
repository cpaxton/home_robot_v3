# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import copy
import json
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.task.tamp import agent_bridge as bridge
from emet.controller.task.tamp.articulation import set_receptacle_state
from emet.simulation.articulation import access_precondition, scene_articulations


def fixture():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
      <worldbody><body name="base_link"><joint name="robot_joint" type="hinge" range="0 1"/>
        <geom size=".1"/></body>
      <body name="fixture" pos="2 0 0"><geom type="box" size=".3 .3 .05"/>
        <body name="door"><joint name="door_joint" type="hinge" axis="0 0 1" range="0 1.57"/>
          <geom type="box" pos=".3 0 .3" size=".3 .02 .3"/></body>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def server_fixture():
    from emet.simulation.robosuite_server import RobosuiteZmqServer
    from emet.simulation.sim_object_placements import placements_from_mujoco_model, placements_to_session_dict

    m, d = fixture()
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel, server._mjdata, server._mj_lock = m, d, threading.RLock()
    server._spec = SimpleNamespace(base_link_name="base_link")
    server._objects_info = None
    server._emet_session = {
        "is_simulation": True,
        "environment": {"kind": "molmospaces"},
        "capabilities": {"sim_articulation_state": True, "sim_set_joint_qpos": True},
        "sim_articulations": scene_articulations(m, d, "base_link"),
        "sim_object_placements": placements_to_session_dict(placements_from_mujoco_model(m, d)),
        "articulation_revision": 0,
    }
    return server


class Robot:
    def __init__(self, server):
        self.server = server
        self._state = {"command_protocol": {"version": 2, "server_boot_id": "test-boot"}}
        self.sent = []

    def get_emet_session(self):
        return copy.deepcopy(self.server._attach_emet_session({})["emet_session"])

    def send_action(self, action, **kwargs):
        self.sent.append(action)
        self.server.handle_action(action)


def context(robot):
    return {
        "_tamp_scene_key": bridge.robot_session_key(robot),
        "_tamp_task_refs": {"task:test": bridge.AgentTaskRef("task:test", "payload", "fixture", "payload", "fixture")},
        "_tamp_plans": {"old": {}},
    }


def test_real_server_open_close_refreshes_geometry_and_invalidates_plans():
    robot = Robot(server_fixture())
    ctx = context(robot)
    before = robot.get_emet_session()
    assert access_precondition(before, "fixture") == "receptacle_requires_open"
    closed = np.array(before["sim_object_placements"]["door"]["collision_bounds"])
    result = set_receptacle_state(robot, ctx, "task:test", "open", timeout_s=0.1)
    assert result["status"] == "ok" and result["data"]["verified"]
    assert not ctx["_tamp_plans"]
    after = robot.get_emet_session()
    assert access_precondition(after, "fixture") is None
    assert after["articulation_revision"] == 1
    assert not np.allclose(after["sim_object_placements"]["door"]["collision_bounds"], closed)
    result = set_receptacle_state(robot, ctx, "task:test", "closed", timeout_s=0.1)
    assert result["status"] == "ok"
    np.testing.assert_allclose(robot.get_emet_session()["sim_object_placements"]["door"]["collision_bounds"], closed)
    assert "door_joint" not in json.dumps(result)
    assert result["data"]["assistance"] == ["joint_teleport"]
    assert result["data"]["collision_scope"] == "none"


def test_metadata_refresh_does_not_change_live_dynamics():
    server = server_fixture()
    data = server._mjdata
    data.qpos[-1] = 1.57
    data.qacc[:] = 42
    old_contacts = data.contact.dist.copy()
    server._attach_emet_session({})
    np.testing.assert_array_equal(data.qacc, 42)
    np.testing.assert_array_equal(data.contact.dist, old_contacts)


@pytest.mark.parametrize(
    "change,code",
    [
        ("hardware", "not_simulation"),
        ("capability", "articulation_unsupported"),
        ("stale", "scene_changed_replan"),
        ("unknown", "unknown_task_ref"),
        ("physical", "articulation_unsupported"),
    ],
)
def test_rejects_before_actuation(change, code, monkeypatch):
    robot = Robot(server_fixture())
    ctx = context(robot)
    if change == "hardware":
        robot.server._emet_session["is_simulation"] = False
    if change == "capability":
        robot.server._emet_session["capabilities"]["sim_set_joint_qpos"] = False
    if change == "stale":
        ctx["_tamp_scene_key"] = ("different",)
    if change == "unknown":
        ctx["_tamp_task_refs"] = {}
    if change == "physical":
        monkeypatch.setenv("EMET_PHYSICAL_EXECUTION", "1")
    assert set_receptacle_state(robot, ctx, "task:test", "open")["code"] == code
    assert not robot.sent


def test_stale_receipt_cannot_verify_and_uncertain_command_invalidates_plan():
    robot = Robot(server_fixture())
    ctx = context(robot)
    robot.send_action = lambda *a, **kw: None
    robot.server._emet_session["articulation_result"] = {"request_id": "old", "applied": True}
    result = set_receptacle_state(robot, ctx, "task:test", "open", timeout_s=0.001)
    assert result["status"] == "partial"
    assert result["code"] == "articulation_verification_timeout"
    assert not result["data"]["verified"] and not ctx["_tamp_plans"]


@pytest.mark.parametrize(
    "joint",
    [
        '<joint type="hinge" range="-1 1"/>',
        '<joint type="hinge" limited="false"/>',
        '<joint type="hinge" range="0 1"/><joint type="slide" range="0 1"/>',
    ],
)
def test_ambiguous_fixtures_unsupported_and_robot_excluded(joint):
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body name="fixture">{joint}<geom size=".1"/></body></worldbody></mujoco>'
    )
    groups = scene_articulations(m, mujoco.MjData(m), "base_link")
    assert len(groups) == 1 and not groups[0]["supported"]
    m, d = fixture()
    groups = scene_articulations(m, d, "base_link")
    assert len(groups) == 1 and groups[0]["joint"] == "door_joint"


def test_closed_access_and_revision_reject_stored_snapshot_before_motion():
    robot = Robot(server_fixture())
    task = context(robot)["_tamp_task_refs"]["task:test"]
    session = robot.get_emet_session()
    snapshot = bridge.PlanningSnapshot(
        "test-boot", bridge.robot_session_key(robot), json.dumps(session["capabilities"], sort_keys=True), "{}"
    )
    assert bridge._validate_snapshot(robot, snapshot, task) == "receptacle_requires_open"
    robot.server._emet_session["articulation_revision"] = 1
    assert bridge._validate_snapshot(robot, snapshot, task) == "scene_changed_replan"
