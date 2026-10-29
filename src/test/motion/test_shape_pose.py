# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Shape equivalence never substitutes for unobserved material pose or volume."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import CaptureInterval, RobotPoseSample, VisualEffectFrame
from emet.controller.manipulation.shape_pose import ShapePoseEvidence, shape_rotation_distance
from emet.motion.placement_geometry import box_corners


def evidence(model=None):
    return ShapePoseEvidence(
        np.eye(4),
        10.0,
        "rgbd_tracking",
        "capture",
        model or PrimitiveModel("cylinder", (0.02, 0.04)),
        0.001,
        0.01,
        "synthetic-test-only",
    )


def test_equivalence_is_only_the_declared_solid_symmetry():
    cylinder = PrimitiveModel("cylinder", (0.02, 0.04))
    yaw = Rotation.from_euler("z", 1.2).as_matrix()
    flip = Rotation.from_euler("x", np.pi).as_matrix()
    tilt = Rotation.from_euler("x", 0.1).as_matrix()
    assert shape_rotation_distance(cylinder, np.eye(3), yaw) < 1e-7
    assert shape_rotation_distance(cylinder, np.eye(3), flip) < 1e-7
    assert shape_rotation_distance(cylinder, np.eye(3), tilt) == pytest.approx(0.1)
    assert shape_rotation_distance(PrimitiveModel("sphere", (0.02,)), np.eye(3), tilt) == 0
    quarter_turn = Rotation.from_euler("z", np.pi / 2).as_matrix()
    assert shape_rotation_distance(PrimitiveModel("box", (0.02, 0.02, 0.04)), np.eye(3), quarter_turn) < 1e-7
    assert shape_rotation_distance(PrimitiveModel("box", (0.02, 0.03, 0.04)), np.eye(3), quarter_turn) > 1


def test_unique_readers_still_reject_shape_and_evidence_ages_are_preserved():
    pose = evidence()
    with pytest.raises(ObservationUnavailable, match="ambiguous"):
        pose.require("observed", now=10.1)
    with pytest.raises(ObservationUnavailable, match="stale"):
        pose.require_shape(now=13.0)
    with pytest.raises(ValueError):
        pose.world_from_object.setflags(write=True)
    with pytest.raises(ValueError):
        replace(pose, calibration_id=True)


def test_declared_complete_shape_cannot_use_smaller_or_sparse_visible_envelope():
    pose = evidence()
    vertices = pose.model.vertices_reference
    good = PayloadGeometryEvidence("object", "reference", "revision", vertices, pose, "known_model_prior")
    np.testing.assert_array_equal(good.require("object", now=10.1), np.eye(4))
    for small in (vertices * 0.5, vertices[[0, 3, 5, 6]]):
        with pytest.raises(ValueError, match="enclose"):
            replace(good, vertices_reference=small)
    with pytest.raises(ValueError, match="complete_model"):
        replace(good, construction="multiview_bound")


def test_visual_bounds_use_complete_shape_without_dependence_on_representative_yaw():
    pose = evidence()
    payload = PayloadGeometryEvidence(
        "object", "reference", "revision", pose.model.vertices_reference, pose, "known_model_prior"
    )

    def robot(t):
        return RobotPoseSample(np.eye(4), np.zeros(8), t, np.array([[[-0.1] * 3, [0.1] * 3]]))

    frame = VisualEffectFrame(
        payload,
        CaptureInterval("capture", 9.98, 9.99, 10.0, "clock"),
        robot(9.97),
        robot(10.01),
        0.001,
        0.01,
        0.0001,
        0.0001,
    )
    frame.require(now=10.1)
    rotated = np.eye(4)
    rotated[:3, :3] = Rotation.from_euler("z", 0.7).as_matrix()
    other = replace(frame, payload=replace(payload, pose=replace(pose, world_from_object=rotated)))
    np.testing.assert_allclose(frame.bounds(), other.bounds(), atol=1e-12)
    assert np.all(frame.bounds()[0] < -pose.model.half_extents)
    with pytest.raises(ObservationUnavailable, match="ambiguous"):
        replace(frame, position_error_m=0.00001).require(now=10.1)
    # A planning box can be loose; visual support tests concern the explicitly
    # declared complete solid, rather than treating an arbitrary envelope as solid.
    loose = replace(frame, payload=replace(payload, vertices_reference=box_corners([[-0.1] * 3, [0.1] * 3])))
    np.testing.assert_array_equal(frame.bounds(), loose.bounds())
