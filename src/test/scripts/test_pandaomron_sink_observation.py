# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_pickup_probe_excludes_the_counter_inside_the_payload_selection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from eval_pandaomron_sink_observation import pickup_probe

    y, x = np.indices((20, 20))
    world = np.stack((x * 0.005, y * 0.005, np.full(x.shape, 0.9)), axis=-1)
    world[7:14, 7:14, 2] = 1.0
    patch = SimpleNamespace(plane=np.array([0.0, 0.0, 0.9]), measurement_margin_m=0.005)
    selected = {"polygon": [(1, 1), (18, 1), (18, 18), (1, 18)], "height_offset_m": -0.025}

    target, yaws, evidence = pickup_probe(np.ones(x.shape), world, selected, patch)

    np.testing.assert_allclose(target, [0.05, 0.05, 0.975])
    assert len(yaws) == 4
    assert evidence["points"] == 49
    assert evidence["central_96_percent_bounds_m"][0][2] == pytest.approx(1.0)


def test_transfer_score_requires_verified_tool_receipt_physics_and_native_actions():
    from eval_pandaomron_sink_agent import score_transfer

    physics = {'initial_success': False, 'upstream_success': True, 'basin_contains_tongs': True,
        'lift_with_two_pads_m': .08, 'final_object_speed_mps': 0., 'unintended_contact_frames': 0,
        'unintended_self_contact_frames': 0}
    receipt = {'execute_pick_place_plan': {'status': 'ok', 'code': 'ok'}}
    actions = ['{"accepted": true, "forbidden": []}']
    assert score_transfer(physics, receipt, actions) == (True, True, True)
    assert score_transfer(physics, {}, actions) == (True, False, False)
    assert not score_transfer({**physics, 'unintended_self_contact_frames': 1}, receipt, actions)[2]
    assert not score_transfer(physics, receipt, [actions[0], '{"forbidden": ["teleport"]}'])[2]
    assert not score_transfer(physics, receipt, [])[2]


def test_serial_benchmark_compilation_preserves_physics_and_assets():
    from xml.etree import ElementTree

    from emet.simulation.pandaomron_environment import serial_mesh_compiler

    xml = '<mujoco><compiler angle="radian"/><option timestep=".002"/><asset><mesh name="part" file="part.obj"/></asset></mujoco>'
    result = ElementTree.fromstring(serial_mesh_compiler(xml))
    assert result.find('compiler').attrib == {'angle': 'radian', 'usethread': 'false'}
    assert result.find('option').get('timestep') == '.002'
    assert result.find('asset/mesh').get('file') == 'part.obj'
