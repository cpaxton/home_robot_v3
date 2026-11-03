# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""A primitive handoff cannot renew a lost, violated or creeping grasp."""
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.eval.contact_stability import TracedNativeAdapter
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker


@pytest.fixture
def held(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="ee" pos="0 0 .1"><joint name="move" type="slide" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
        <body name="rubber_tip_left" pos="-.03 0 0"><geom name="left" type="sphere" size=".012"/></body>
        <body name="rubber_tip_right" pos=".03 0 0"><geom name="right" type="sphere" size=".012"/></body>
      </body>
      <body name="payload" pos="0 0 .1"><freejoint/><geom type="box" size=".02 .02 .02" mass=".1"/></body>
      <body name="table" pos="0 0 -1"><geom type="box" size="1 1 .1"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    def sync(data):
        mujoco.mj_kinematics(model, data)
    sync(data)
    collision = MujocoSceneCollisionChecker(model, robot_body='ee')
    collision.set_payload(model, data, 'payload', 'ee')
    executor = PhysicalPickPlaceExecutor(None, model=model, data=data, ee_body='ee',
        joint_names=('move',), collision=collision, synchronize=sync,
        command_joints=lambda _: pytest.fail('handoff must not command motion'),
        input_mode='privileged', use_sim_time=True, event=Mock())
    executor.payload_body = 'payload'
    adapter = object.__new__(TracedNativeAdapter)
    adapter.model, adapter.data, adapter.synchronize = model, data, sync
    plan = SimpleNamespace(executor=executor, object_body='payload', support_body='table')
    def sleep(seconds):
        data.time += seconds
    monkeypatch.setattr('emet.eval.contact_stability.time', SimpleNamespace(sleep=sleep))
    return adapter, plan


def rotate(adapter, angle):
    addr = int(adapter.model.jnt_qposadr[adapter.model.body('payload').jntadr[0]])
    adapter.data.qpos[addr+3:addr+7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]


def test_stable_measured_payload_is_bound_once_and_requires_place_replanning(held):
    adapter, plan = held
    previous = plan.executor.collision.payload_transform.copy()
    rotate(adapter, .08)  # Inside the unchanged original 0.1 rad bound.
    assert adapter._bind_loaded_payload(plan) is True
    bound = plan.executor.collision.payload_transform.copy()
    assert not np.array_equal(bound, previous)
    adapter.synchronize(adapter.data)
    assert plan.executor.payload_retained() is True
    assert plan.executor.event.call_args.kwargs['placement_replan_required'] is True
    assert adapter._bind_loaded_payload(plan) is False
    np.testing.assert_array_equal(plan.executor.collision.payload_transform, bound)


def test_already_violated_original_pose_bound_cannot_be_renewed(held):
    adapter, plan = held
    previous = plan.executor.collision.payload_transform.copy()
    rotate(adapter, .11)
    assert adapter._bind_loaded_payload(plan) is False
    np.testing.assert_array_equal(plan.executor.collision.payload_transform, previous)
    assert not getattr(plan.executor, '_loaded_binding_done', False)


def test_hold_only_control_checks_stability_but_keeps_original_reference(held):
    adapter, plan = held
    previous = plan.executor.collision.payload_transform.copy()
    rotate(adapter, .08)
    assert adapter._bind_loaded_payload(plan, rebind=False) is True
    np.testing.assert_array_equal(plan.executor.collision.payload_transform, previous)
    assert plan.executor.event.call_args.kwargs['reference_changed'] is False


def test_accumulated_subsample_creep_is_not_hidden_by_a_moving_anchor(held, monkeypatch):
    adapter, plan = held
    previous = plan.executor.collision.payload_transform.copy()
    def sleep(seconds):
        adapter.data.time += seconds
        rotate(adapter, adapter.data.time * .08)
    monkeypatch.setattr('emet.eval.contact_stability.time', SimpleNamespace(sleep=sleep))
    assert adapter._bind_loaded_payload(plan) is False
    np.testing.assert_array_equal(plan.executor.collision.payload_transform, previous)
    assert plan.executor.payload_retained()  # Rejected for stability, before the original bound is lost.


def test_pose_bound_alone_cannot_replace_bilateral_contact(held):
    adapter, plan = held
    previous = plan.executor.collision.payload_transform.copy()
    for name in ('left', 'right'):
        gid = adapter.model.geom(name).id
        adapter.model.geom_contype[gid] = adapter.model.geom_conaffinity[gid] = 0
    assert plan.executor.payload_retained()
    assert adapter._bind_loaded_payload(plan) is False
    np.testing.assert_array_equal(plan.executor.collision.payload_transform, previous)


def test_failed_native_hold_verification_never_reaches_binding(held, monkeypatch):
    adapter, plan = held
    adapter.payload_binding = 'verified_lift'
    adapter._bind_loaded_payload = Mock(side_effect=AssertionError('unverified grasp'))
    monkeypatch.setattr('emet.eval.contact_stability.NativeTableAdapter._verify', lambda *a, **kw: False)
    assert adapter._verify(plan, released=False, seconds=1.) is False
