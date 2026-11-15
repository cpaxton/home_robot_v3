# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Geometric ordering preserves candidates and respects actual collision frames."""
import mujoco
import numpy as np
import pytest

from emet.motion.grasp_alignment import face_aligned_grasp_order


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rig(geometry):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="degree"/>
      <worldbody><body name="item" euler="0 0 30"><freejoint/>
        <body name="shape" euler="0 0 20">{geometry}</body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, data


def test_face_aligned_height_beats_lower_edge_pinch_and_keeps_all_candidates():
    model, data = rig('<geom type="box" size=".02 .04 .04"/>')
    edge = (yaw(np.deg2rad(30)), 0.)
    face_low = (yaw(np.deg2rad(50)), .012)
    face_high = (face_low[0].copy(), .024)
    candidates = [edge, face_low, face_high]
    q, limits, contacts = data.qpos.copy(), model.jnt_range.copy(), data.ncon
    ordered = face_aligned_grasp_order(model, data, 'item', closing_axis=[1, 0, 0], candidates=candidates)
    assert [id(c) for c in ordered] == [id(face_low), id(face_high), id(edge)]
    assert [id(c) for c in candidates] == [id(edge), id(face_low), id(face_high)]
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(model.jnt_range, limits)
    assert data.ncon == contacts


@pytest.mark.parametrize('geometry', [
    '<geom type="cylinder" size=".02 .04"/>',
    '<geom type="sphere" size=".03"/>',
    '<geom type="box" size=".02 .04 .04"/><geom type="box" pos="0 0 .1" size=".01 .01 .01"/>',
])
def test_unknown_or_compound_contact_shape_keeps_prior_order(geometry):
    model, data = rig(geometry)
    candidates = [(yaw(0), 0.), (yaw(np.deg2rad(50)), .01)]
    result = face_aligned_grasp_order(model, data, 'item', closing_axis=[1, 0, 0], candidates=candidates)
    assert [id(c) for c in result] == [id(c) for c in candidates]


def test_visual_only_geometry_cannot_choose_the_pinch_direction():
    model, data = rig('''<geom type="box" size=".02 .04 .04"/>
      <geom type="box" euler="0 0 -50" size=".1 .1 .1" contype="0" conaffinity="0"/>''')
    candidates = [(yaw(0), 0.), (yaw(np.deg2rad(50)), .01)]
    result = face_aligned_grasp_order(model, data, 'item', closing_axis=[1, 0, 0], candidates=candidates)
    assert result[0] is candidates[1]


def test_missing_closing_axis_cannot_produce_a_geometry_ranking():
    model, data = rig('<geom type="box" size=".02 .04 .04"/>')
    with pytest.raises(ValueError, match='invalid_gripper_closing_axis'):
        face_aligned_grasp_order(model, data, 'item', closing_axis=[float('nan'), 0, 0], candidates=[])


def test_tiny_alignment_difference_preserves_preferred_pitch_height_order():
    model, data = rig('<geom type="box" size=".02 .04 .04"/>')
    preferred = (yaw(np.deg2rad(50.4)), .012)
    fallback = (yaw(np.deg2rad(50)), 0.)
    result = face_aligned_grasp_order(model, data, 'item', closing_axis=[1, 0, 0], candidates=[preferred, fallback])
    assert result[0] is preferred
