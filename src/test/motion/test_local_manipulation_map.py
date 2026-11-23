# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from types import SimpleNamespace

import numpy as np
import pytest

from emet.motion.local_manipulation_map import LocalManipulationMap
from emet.motion.observed_solids import WorldSolid


def observation():
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.
    return SimpleNamespace(rgb=np.zeros((41, 41, 3), np.uint8), depth=np.ones((41, 41)),
                           camera_K=np.array([[100., 0., 20.], [0., 100., 20.], [0., 0., 1.]]), camera_pose=pose)


def solid(z):
    pose = np.eye(4)
    pose[2, 3] = z
    return WorldSolid('box', (.01, .01, .01), pose)


def test_local_map_checks_measured_surface_and_refines_nearby_clearance():
    obs = observation()
    scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)})
    assert scene.query(solid(.03)).allowed
    assert not scene.query(solid(.003)).allowed
    assert not scene.query(solid(-.03)).allowed


def test_excluding_self_depth_does_not_clear_occluded_space():
    obs = observation()
    scene = LocalManipulationMap({'camera': obs}, {'camera': np.ones(obs.depth.shape, bool)})
    assert not scene.query(solid(-.03)).allowed
    assert scene.query(solid(-.03), starting_volumes=(solid(-.03),)).allowed


def test_local_update_preserves_previous_obstacles():
    obs = observation()
    first = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)})
    obs.depth = obs.depth + .2
    updated = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)}, previous=first)
    assert not updated.query(solid(.003)).allowed


def observed_lift_rig(offset):
    import time

    import mujoco

    from emet.controller.manipulation.native_scene import model_signature
    from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker
    from emet.controller.manipulation.observed_effects import CaptureInterval
    from emet.motion.observed_admission import CapturedRobotMaterial, ObservedMaterialAdmission, RobotGeometryBound
    from emet.motion.observed_background import ObservedBackground
    from emet.motion.observed_workspace import ObservedWorkspace

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="robot" pos="0 0 .025">
      <joint name="lift" type="slide" axis="0 0 1" range="0 .1"/>
      <geom type="box" size=".02 .02 .02" mass="1"/>
      <body name="ee"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    now = time.monotonic()
    capture = CaptureInterval('sample', now, now, now, 'clock')
    signature = model_signature(model)
    volume = WorldSolid.from_geom(model, data, 0)
    material = CapturedRobotMaterial(capture, signature, 'calibration', 'scene', (volume,), (0.,), 'profile')
    grid = ObservedWorkspace([-.2, -.2, -.1], .05, np.ones((8, 8, 8), np.uint8))
    background = ObservedBackground(grid, np.full(grid.cells.shape, now), 'scene', frozenset(['sample']), capture)
    admission = ObservedMaterialAdmission(background, material, query_robot_geometry=lambda m, d, gid:
        RobotGeometryBound(WorldSolid.from_geom(m, d, gid, offset_m=offset), gid, signature, 'calibration', 'profile', 0.))
    checker = ObservedNativeCollisionChecker(model, robot_body='robot', ee_body='ee',
        scene=background.placement_scene(), payload_provider=SimpleNamespace(input_mode='observed'),
        scene_revision=background.geometry_revision, current_revision=lambda: background.geometry_revision,
        received_monotonic=now, margin_m=0., admission=admission, clock=lambda: now)
    obs = observation()
    checker.local_scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)})
    checker.allow_contact_departure = True
    return model, data, checker


@pytest.mark.parametrize("offset, goal", [(0., .02), (.01, .002)])
def test_observed_arm_can_sample_out_of_local_start_overlap(offset, goal):
    import mujoco

    from emet.motion.arm_rrt import plan_arm_joint_path

    model, data, checker = observed_lift_rig(offset)
    result = plan_arm_joint_path(model, data, joint_names=['lift'], q_start=[0.], q_goal=[goal],
                                 collision=checker, step_size=.002)
    assert result.success, (result, checker.last_departure_diagnostics, checker.local_scene.last_collision)
    assert result.planner == 'departure_rrt'
    assert result.waypoints.departure is result.departure
    if offset:
        data.qpos[:] = goal
        mujoco.mj_forward(model, data)
        # The final pose consumes original-patch padding, not physical space.
        assert checker.configuration_collides(model, data)
        with result.waypoints.clearance_scope():
            assert not checker.configuration_collides(model, data)
        assert checker.configuration_collides(model, data)


def test_isolated_depth_leaves_its_ray_unknown():
    obs = observation()
    obs.depth[20, 20] = .5
    scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)})
    assert not scene.query(solid(.3)).allowed


def test_fine_resolution_retains_the_declared_sensor_uncertainty():
    obs = observation()
    scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)},
                                 resolution_m=.005, depth_error_m=.015)
    assert not scene.query(solid(.02)).allowed
    assert scene.query(solid(.05)).allowed


def test_simulation_unknown_space_policy_still_checks_observed_obstacles():
    obs = observation()
    scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)},
                                 simulation_allow_unknown=True)
    assert scene.query(solid(-.05)).allowed
    assert not scene.query(solid(0.)).allowed


def test_simulation_padding_slack_preserves_measured_obstacles_and_error_floor():
    from dataclasses import replace

    from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker

    obs = observation()
    scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)},
                                 simulation_allow_unknown=True)
    current = [replace(solid(.027), offset_m=.00966)]
    checker = SimpleNamespace(
        target_contact=None, payload=None, _stationary_geometry={}, admission=object(), local_scene=scene,
        _starting_volumes=(), simulation_clearance_slack_m=0.,
        _robot_bound=lambda model, data, gid: (current[0], .001),
    )
    model = SimpleNamespace(geom=lambda gid: SimpleNamespace(name='arm'))
    def collides():
        return ObservedNativeCollisionChecker._robot_scene_collides(checker, model, None, 0, None)
    assert collides()  # 12 mm measured separation is below the 12.66 mm reserve.
    checker.simulation_clearance_slack_m = .002
    assert not collides()
    assert scene.error == .003
    assert len(scene.centers) > 0
    current[0] = replace(solid(.003), offset_m=.00966)
    assert collides()  # Slack never removes actual occupied geometry.
    current[0] = replace(solid(.016), offset_m=.001)
    checker.simulation_clearance_slack_m = .005
    assert collides()  # Keep both the 1 mm model error and 3 mm depth error.


def test_contact_surface_samples_preserve_measured_locations_across_map_updates():
    obs = observation()
    obs.depth[:] = .9983
    first = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)})
    assert np.allclose(first.surface_points[:, 2], .0017)
    assert not np.allclose(first.surface_points[:, 2], first.centers[:, 2])
    obs.depth[:] = .9876
    updated = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)}, previous=first)
    assert np.any(np.isclose(updated.surface_points[:, 2], .0017))
    assert np.any(np.isclose(updated.surface_points[:, 2], .0124))


def test_contact_handoff_keeps_new_obstacles_and_model_uncertainty():
    from emet.motion.surface_contact_patches import SurfaceContactPatches

    obs = observation()
    obs.rgb = np.zeros((81, 81, 3), np.uint8)
    obs.depth = np.ones((81, 81))
    obs.camera_K = np.array([[1000., 0., 40.], [0., 1000., 40.], [0., 0., 1.]])
    mask = np.zeros(obs.depth.shape, bool)
    scene = LocalManipulationMap({'camera': obs}, {'camera': mask})
    patches = SurfaceContactPatches(scene.surface_points, scene.surface_points)
    shape = solid(.02).with_offset(.01)
    assert not scene.query(shape).allowed
    assert scene.query_contact_clearance(shape, patches, minimum_padding_m=.004).allowed
    # The model error and measured free-ray error still need clearance.
    assert not scene.query_contact_clearance(shape, patches, minimum_padding_m=.009).allowed
    obs.depth[30:51, 30:51] = .992  # Separate surface, 8 mm above the original.
    updated = LocalManipulationMap({'camera': obs}, {'camera': mask}, previous=scene)
    result = updated.query_contact_clearance(shape, patches, minimum_padding_m=.004)
    assert not result.allowed and result.reason == 'occupied_local_surface'



def test_executor_carries_clearance_across_phase_plans_from_an_ordinary_clear_start():
    import mujoco

    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    model, data, checker = observed_lift_rig(.01)
    data.qpos[:] = .015
    mujoco.mj_forward(model, data)
    assert not checker.configuration_collides(model, data)
    executor = PhysicalPickPlaceExecutor(SimpleNamespace(), model=model, data=data,
        ee_body='ee', joint_names=('lift',), collision=checker, synchronize=lambda d: None,
        command_joints=lambda q: True, object_pose_provider=checker)
    path, error = executor.plan_joint_target(np.array([.002]))
    assert error is None and callable(path.clearance_scope)
    scope = path.clearance_scope
    assert checker.contact_clearance is None
    # A second phase must retain the measured start's patches rather than
    # resetting them at the hypothetical endpoint of the preceding phase.
    second, error = executor.plan_joint_target(np.array([.004]))
    assert error is None and second.clearance_scope is scope
    assert checker.contact_clearance is None
    with scope():
        assert checker.make_departure(model, data, ('lift',), [.004]) is None


def test_clear_measured_surface_consumes_reserve_but_keeps_error_floor_without_patch_identity():
    from emet.motion.surface_contact_patches import SurfaceContactPatches

    obs = observation()
    obs.depth[:] = .995  # Surface z=5 mm, leaving a 5 mm nominal gap below the box.
    scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)})
    shape = solid(.02).with_offset(.01)
    no_contacts = SurfaceContactPatches([], [])
    assert not scene.query(shape).allowed
    assert scene.query_contact_clearance(shape, no_contacts, minimum_padding_m=.001).allowed
    obs.depth[:] = .992  # A 2 mm gap does not cover model (1 mm) + depth (3 mm) error.
    scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)})
    assert not scene.query_contact_clearance(shape, no_contacts, minimum_padding_m=.001).allowed
