# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import numpy as np
import pytest

from emet.eval.grapheqa_sensor import command_camera_pose, learned_label_map, validate_packet


def packet():
    return {
        "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        "depth": np.ones((2, 3), dtype=np.float32),
        "camera_K": np.eye(3),
        "camera_pose": command_camera_pose(yaw_deg=0, tilt_deg=-30, sensor_height=1.5),
        "timestamp_ns": np.array(10**9, dtype=np.int64),
        "learned_labels": np.zeros((2, 3), dtype=np.int32),
    }


@pytest.mark.parametrize("field", ["semantic", "objects", "navmesh", "answer", "enrich_labels", "scene_bounds"])
def test_privileged_or_unspecified_packet_fields_rejected(field):
    frame = packet()
    frame[field] = np.array(1)
    with pytest.raises(ValueError, match="exactly"):
        validate_packet(frame, num_labels=92)


def test_background_is_unknown_even_when_first_class_is_detected():
    labels = learned_label_map(np.zeros((1, 3)), np.array([[-1, 0, -1]]), [14, 69])
    np.testing.assert_array_equal(labels, [[0, 14, 0]])


@pytest.mark.parametrize("index", [-1, 2, 0.5, np.nan])
def test_bad_detector_indices_fail_instead_of_becoming_other_classes(index):
    with pytest.raises(ValueError, match="Detector class"):
        learned_label_map(np.array([[index]]), np.array([[0]]), [14, 69])


def test_command_pose_looks_down_and_turns_without_global_pose():
    first = command_camera_pose(yaw_deg=0, tilt_deg=-30, sensor_height=1.5)
    left = command_camera_pose(yaw_deg=90, tilt_deg=-30, sensor_height=1.5)
    np.testing.assert_allclose(first[:3, 2], [np.sqrt(3) / 2, 0, -0.5], atol=1e-10)
    np.testing.assert_allclose(left[:3, 2], [0, np.sqrt(3) / 2, -0.5], atol=1e-10)
    frame = packet()
    frame["camera_pose"] = left
    validate_packet(frame, num_labels=92)


@pytest.mark.parametrize(
    "field,value",
    [
        ("learned_labels", np.full((2, 3), 92, dtype=np.int32)),
        ("depth", np.full((2, 3), np.nan, dtype=np.float32)),
        ("camera_pose", np.diag([-1, 1, 1, 1])),
        ("timestamp_ns", np.array(0)),
    ],
)
def test_invalid_sensor_packets_fail(field, value):
    frame = packet()
    frame[field] = value
    with pytest.raises(ValueError):
        validate_packet(frame, num_labels=92)
