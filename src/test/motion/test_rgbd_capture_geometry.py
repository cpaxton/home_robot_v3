# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Capture-bound visual assembly uses motors, not privileged passive-joint truth."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from prepare_observed_stretch_fixture import prepare

from emet.controller.manipulation.continuous_rgbd import ContinuousRgbdSource, TrackedRgbdCapture
from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.geometry_tracking import GeometryFit
from emet.controller.manipulation.multiview_rgbd_tracking import MultiViewMaskFit, SynchronizedRgbdView
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import evaluate_lift, evaluate_place
from emet.controller.manipulation.rgbd_capture_geometry import RgbdCaptureGeometry, SimGeometryProfile
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.core.rgbd_registration import encode_registration
from emet.simulation.depth_registration import register_depth
from emet.simulation.synchronized_capture import capture_scene_for_sensors, capture_stretch_motors


@pytest.fixture
def sample(tmp_path):
    prepare(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "stretch.xml"))
    data = mujoco.MjData(model)
    probe, packet = capture_scene_for_sensors(
        model, data, threading.Lock(), cameras={"head": ("d435i_camera_rgb", True)}
    )
    motors = capture_stretch_motors(model, probe, packet)
    start, end = packet["interval_ns"]
    obs = SimpleNamespace(
        is_simulation=True,
        camera_K=np.array([[20.0, 0.0, 6.0], [0.0, 20.0, 5.0], [0.0, 0.0, 1.0]]),
        camera_pose=np.asarray(packet["camera_poses"]["head"]),
        received_monotonic=time.monotonic(),
        image_timing={
            "head_cam_left": {
                "source": "renderer_synchronized_snapshot",
                "available": True,
                "clock_domain": "mujoco_render_monotonic",
                "timestamp_ns": start,
                "clock_id": packet["clock_id"],
                "capture_interval_ns": [start, end],
                "depth_capture_interval_ns": [start, end],
                "motor_sample": motors,
                "robot_sample": "POISON: passive truth must never be read",
            }
        },
        rgb=np.zeros((10, 12, 3), np.uint8),
        depth=np.ones((10, 12)),
    )
    profile = SimGeometryProfile("test-simulation-assumptions", 0.003, 0.02, 0.001, 0.005)
    geometry = RgbdCaptureGeometry.from_stretch(obs, model, profile=profile, now=time.monotonic())
    rgb, depth, world = capture_arrays(obs)
    view = SynchronizedRgbdView(rgb, depth, world, geometry.capture, geometry=geometry)
    shape, pose = PrimitiveModel("box", (0.02, 0.03, 0.04)), np.eye(4)
    fit = GeometryFit(
        shape,
        "target",
        np.array([pose]),
        np.array([0.0001]),
        (np.empty((0, 6)),),
        (view.capture.observation_id,),
        view.capture.received_monotonic,
        1,
        False,
    )
    tracked = MultiViewMaskFit(fit, view.capture, (("head", np.ones(depth.shape, bool)),), (), "reference", 0.01)
    return model, obs, view, tracked


def test_visual_frame_keeps_original_capture_and_only_eight_motor_readouts(sample):
    model, _, view, tracked = sample
    frame = view.geometry.visual_frame(tracked, robot_model=model, now=time.monotonic())
    assert frame.capture == view.capture
    assert frame.robot_before.joints.shape == frame.robot_after.joints.shape == (8,)
    assert frame.robot_before.received_monotonic == view.capture.earliest
    assert frame.robot_after.received_monotonic == view.capture.latest
    assert frame.robot_before.gripper_bounds.shape[1:] == (2, 3)
    assert frame.position_error_m == 0.003  # Explicit prior, not the .0001 fit RMS.
    assert frame.alignment_error_m == 0.0  # Ideal co-sampled simulator only.
    for value in (view.geometry.intrinsic, view.geometry.world_from_camera):
        with pytest.raises(ValueError):
            value.setflags(write=True)


@pytest.mark.parametrize("change", ["capture", "world", "camera"])
def test_mixed_frame_or_camera_geometry_rejects_before_use(sample, change):
    _, _, view, _ = sample
    with pytest.raises(ObservationUnavailable):
        if change == "capture":
            replace(view, capture=replace(view.capture, observation_id="different-camera-frame"))
        elif change == "world":
            replace(view, world=view.world + [0.01, 0, 0])
        else:
            pose = view.geometry.world_from_camera.copy()
            pose[0, 3] += 0.01
            replace(view, geometry=replace(view.geometry, world_from_camera=pose))


def test_foreign_fit_model_mutation_and_stale_capture_reject(sample):
    model, _, view, tracked = sample
    with pytest.raises(ObservationUnavailable, match="capture_identity_mismatch"):
        view.geometry.visual_frame(
            replace(tracked, capture=replace(view.capture, observation_id="other")),
            robot_model=model,
            now=time.monotonic(),
        )
    with pytest.raises(ObservationUnavailable, match="stale"):
        view.geometry.require(view, now=view.capture.earliest + 3.0)
    model.geom_size[0, 0] += 0.01
    with pytest.raises(ObservationUnavailable, match="gripper_geometry_changed"):
        view.geometry.visual_frame(tracked, robot_model=model, now=time.monotonic())


def test_duplicate_cannot_change_encoder_or_profile_metadata_without_changing_pixels(sample):
    _, _, view, tracked = sample
    previous = TrackedRgbdCapture(tracked, (("head", view),))
    assert ContinuousRgbdSource._same_packet(previous, {"head": view})
    profile = replace(view.geometry.profile, object_position_error_m=0.004)
    changed = replace(view, geometry=replace(view.geometry, profile=profile))
    assert not ContinuousRgbdSource._same_packet(previous, {"head": changed})
    altered = replace(view.geometry.robot, world_from_base=np.eye(4))
    # Even metadata that does not alter the RGB-grid unprojection is retained.
    other = replace(view, geometry=replace(view.geometry, robot=altered))
    expected = np.array_equal(altered.world_from_base, view.geometry.robot.world_from_base)
    assert ContinuousRgbdSource._same_packet(previous, {"head": other}) == expected


def test_hardware_cannot_claim_ideal_simulator_synchronization(sample):
    model, obs, view, _ = sample
    obs.is_simulation = False
    with pytest.raises(ObservationUnavailable, match="simulation_capture_required"):
        RgbdCaptureGeometry.from_stretch(obs, model, profile=view.geometry.profile, now=time.monotonic())


def test_gripper_bounds_cannot_silently_use_another_error_profile(sample):
    _, _, view, _ = sample
    with pytest.raises(ValueError, match="invalid_rgbd_capture_geometry"):
        replace(view.geometry, profile=replace(view.geometry.profile, robot_position_error_m=0.002))


def test_original_depth_rays_pin_rotation_calibration_and_wire_capture(sample):
    model, obs, view, _ = sample
    native_k = np.array([[20.0, 0, 5.0], [0, 20.0, 5.0], [0, 0, 1.0]])
    raw = np.full((12, 10), 0.9, np.float32)
    rgb = np.zeros((*raw.shape, 3), np.uint8)
    transform = np.eye(4)
    transform[0, 3] = 0.013
    aligned = register_depth(raw, native_k, native_k, transform, raw.shape)
    timing = obs.image_timing["head_cam_left"]
    obs.rgbd_registration = encode_registration(
        rgb=rgb,
        raw_depth=raw,
        registered_depth=aligned,
        depth_k=native_k,
        rgb_k=native_k,
        rgb_from_depth=transform,
        rgb_interval_ns=timing["capture_interval_ns"],
        depth_interval_ns=timing["depth_capture_interval_ns"],
        clock_id=timing["clock_id"],
    )
    obs.rgb = np.rot90(rgb, -1)
    obs.depth = (np.rot90(aligned, -1) * 1000).astype(np.uint16).astype(np.float32) / 1000
    geom = RgbdCaptureGeometry.from_stretch(
        obs, model, profile=view.geometry.profile, now=time.monotonic(), original_rays=True
    )
    measured = SynchronizedRgbdView(obs.rgb, obs.depth, geom.source_points_world, geom.capture, geometry=geom)
    geom.require(measured, now=time.monotonic())
    y, x = np.indices(raw.shape)
    camera = np.stack(((x - 5) * raw / 20, (y - 5) * raw / 20, raw), axis=-1)
    world = camera @ geom.world_from_depth[:3, :3].T + geom.world_from_depth[:3, 3]
    # Every selected registered return is an ORIGINAL depth endpoint.
    valid = np.isfinite(geom.source_points_world).all(-1)
    distances = np.linalg.norm(geom.source_points_world[valid, None] - world.reshape(-1, 3), axis=-1)
    assert distances.min(axis=1).max() < 1e-12
    assert not geom.source_depth.flags.writeable
    obs.depth = obs.depth + 0.001
    with pytest.raises(ObservationUnavailable, match="rgbd_geometry_capture_mismatch"):
        RgbdCaptureGeometry.from_stretch(
            obs, model, profile=view.geometry.profile, now=time.monotonic(), original_rays=True
        )


def test_assembled_reference_validates_at_original_receipt_not_earlier_motor_capture(sample):
    model, _, view, tracked = sample
    reference = view.geometry.visual_frame(tracked, robot_model=model, now=time.monotonic())
    assert reference.robot_after.received_monotonic < reference.capture.received_monotonic
    later = []
    for delta in (0.3, 0.7):
        cap = replace(
            reference.capture,
            observation_id=f"later:{delta}",
            earliest=reference.capture.earliest + delta,
            latest=reference.capture.latest + delta,
            received_monotonic=reference.capture.received_monotonic + delta,
        )
        payload = replace(
            reference.payload,
            pose=replace(
                reference.payload.pose, observation_id=cap.observation_id, received_monotonic=cap.received_monotonic
            ),
        )
        later.append(
            replace(
                reference,
                capture=cap,
                payload=payload,
                robot_before=replace(reference.robot_before, received_monotonic=cap.earliest),
                robot_after=replace(reference.robot_after, received_monotonic=cap.latest),
            )
        )
    now, after = reference.capture.received_monotonic + 0.8, reference.capture.latest + 0.1
    lifted = evaluate_lift(reference, later, support_top_z=0.0, after=after, now=now)
    assert lifted.reason == "visual_lift_not_verified" and not lifted.satisfied
    placed = evaluate_place(
        reference, later, support_bounds=[[-1.0, -1.0, -0.1], [1.0, 1.0, 0.0]], after=after, now=now
    )
    assert placed.reason == "visual_placement_not_verified"
