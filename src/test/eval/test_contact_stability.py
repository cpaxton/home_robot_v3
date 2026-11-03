# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Diagnostic FK/contact reconstruction must not change executing physics."""
import json
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.eval.contact_stability import ContactTrace


def test_trace_captures_rotation_and_contacts_without_mutating_physics(tmp_path):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="ee" pos="0 0 .1"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".01" mass="1"/></body>
      <body name="payload" pos=".2 0 .1"><freejoint/>
        <geom type="box" size=".1 .1 .1" mass="1"/></body>
      <geom name="floor" type="plane" size="1 1 .1"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    addr = int(model.jnt_qposadr[model.body('payload').jntadr[0]])
    data.qpos[addr+3:addr+7] = [np.cos(.13/2), 0, 0, np.sin(.13/2)]
    mujoco.mj_forward(model, data)
    before = [x.copy() for x in (data.qpos, data.qvel, data.qacc, data.contact.dist)]
    expected = np.eye(4)
    expected[:3, 3] = [.2, 0, 0]
    executor = SimpleNamespace(data=data, ee_body='ee', collision=SimpleNamespace(payload_transform=expected))
    path = tmp_path / 'trace.jsonl'
    recorder = ContactTrace(model, path)
    try:
        recorder.record('payload', executor, {'phase': 'lift', 'controller_success': True})
    finally:
        recorder.close()
    row = json.loads(path.read_text())
    assert row['source'] == 'simulator_ground_truth_debug'
    assert row['translation_residual_m'] < 1e-12
    np.testing.assert_allclose(row['rotation_residual_rad'], .13, atol=1e-12)
    assert row['contacts'] and row['qpos'] == before[0].tolist()
    for previous, current in zip(before, (data.qpos, data.qvel, data.qacc, data.contact.dist), strict=True):
        np.testing.assert_array_equal(previous, current)


def test_planning_target_event_does_not_claim_a_measured_snapshot(tmp_path):
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    path = tmp_path / 'trace.jsonl'
    recorder = ContactTrace(model, path)
    try:
        # No body or collision provider is necessary for a planning-only event.
        recorder.record('not_a_measured_body', SimpleNamespace(data=mujoco.MjData(model)),
                        {'phase': 'grasp_replan', 'accepted': True, 'targets': []})
    finally:
        recorder.close()
    row = json.loads(path.read_text())
    assert row['kind'] == 'planning_targets'
    assert 'qpos' not in row and 'contacts' not in row
