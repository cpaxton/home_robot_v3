# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""PandaOmron native observed-arm binding."""

from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.controller.manipulation.capture_clock import local_sim_capture_interval
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.observed_effects import RobotPoseSample, rotation_distance
from emet.controller.task.tamp.live_transfer import TransferSample
from emet.controller.task.tamp.observed_arm import ObservedArm
from emet.motion.observed_admission import CapturedRobotMaterial, RobotGeometryBound
from emet.motion.observed_solids import WorldSolid
from emet.robots.pandaomron import ARM_JOINTS, BASE_BODY, CONTROL_JOINTS, FINGER_JOINTS, PANDA_ARM_CHAIN

ROOT_BODY = "robot0_base"
BASE_JOINTS = tuple(f"mobilebase0_joint_mobile_{part}" for part in ("forward", "side", "yaw"))


def synchronize_pandaomron(observation, model, data):
    """Populate robot-only FK from capture-aligned joint and base feedback."""
    if observation is None:
        raise ObservationUnavailable("observation_unavailable")
    timing = (observation.image_timing or {}).get("head_cam_left")
    if not isinstance(timing, dict) or timing.get("source") != "renderer_synchronized_snapshot":
        raise ObservationUnavailable("capture_motion_alignment_unknown")
    capture = local_sim_capture_interval(
        timing,
        received_monotonic=observation.received_monotonic,
        observation_id=str(timing.get("timestamp_ns")),
    )
    _synchronize_feedback(observation, model, data)
    return capture


def _synchronize_feedback(observation, model, data):
    joints = np.asarray(observation.joint, dtype=float)
    origin = np.asarray((observation.emet_session or {}).get("navigation_origin_xyt"), dtype=float)
    relative_xy, relative_yaw = np.asarray(observation.gps), np.asarray(observation.compass)
    if (
        joints.shape != (len(CONTROL_JOINTS) + len(FINGER_JOINTS),)
        or origin.shape != (3,)
        or relative_xy.shape != (2,)
        or relative_yaw.shape != (1,)
        or not np.isfinite(np.r_[joints, origin, relative_xy, relative_yaw]).all()
    ):
        raise ObservationUnavailable("invalid_joint_feedback")
    c, s = np.cos(origin[2]), np.sin(origin[2])
    world_xy = origin[:2] + np.array([[c, -s], [s, c]]) @ relative_xy
    world_yaw = origin[2] + relative_yaw[0]
    data.qpos[:] = model.qpos0
    for name, value in zip((*CONTROL_JOINTS, *FINGER_JOINTS), joints, strict=True):
        data.joint(name).qpos[0] = value
    base_id = model.body(BASE_BODY).id
    base_joints = [model.joint(name).id for name in BASE_JOINTS]
    mujoco.mj_kinematics(model, data)
    base = data.body(base_id)
    yaw = np.arctan2(base.xmat[3], base.xmat[0])
    yaw_delta = world_yaw - yaw
    data.qpos[model.jnt_qposadr[base_joints[2]]] += np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jacobian = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacobian, None, base_id)
    slides = base_joints[:2]
    data.qpos[model.jnt_qposadr[slides]] += np.linalg.solve(
        jacobian[:2, model.jnt_dofadr[slides]], world_xy - base.xpos[:2]
    )
    mujoco.mj_kinematics(model, data)


def bind_pandaomron_arm(robot, model):
    def synchronize(data):
        # Servo feedback is independent of camera frame rate. Perception still
        # uses synchronize_pandaomron with the capture-owned encoder sample.
        _synchronize_feedback(robot.get_native_feedback(), model, data)

    def command_joints(goals):
        robot.send_action({"joint": np.asarray(goals).tolist()})
        return True

    return ObservedArm(
        model, ROOT_BODY, PANDA_ARM_CHAIN, synchronize, command_joints, robot.command_gripper_joints, FINGER_JOINTS,
        joint_tolerance=.01, linear_joint_tolerance_m=.001,
    )


def fixed_torso_arm(robot, arm):
    """Plan with seven arm joints while natively holding the measured torso."""
    reference = float(robot.get_native_feedback().joint[len(ARM_JOINTS)])
    synchronize = arm.synchronize

    def measured(data):
        synchronize(data)
        if abs(float(data.joint(CONTROL_JOINTS[-1]).qpos[0])-reference) > .002:
            raise ObservationUnavailable("fixed_torso_drift")

    def command(goals):
        goals = np.asarray(goals, float)
        if goals.shape != (len(ARM_JOINTS),) or not np.isfinite(goals).all():
            raise ValueError("invalid_native_arm_goal")
        robot.send_action({"joint": [*goals, reference]})
        return True

    chain = replace(arm.chain, joint_names=ARM_JOINTS, actuator_names=ARM_JOINTS,
                    home_arm_q=arm.chain.home_arm_q[:len(ARM_JOINTS)])
    return replace(arm, chain=chain, synchronize=measured, command_joints=command)


def panda_sample_reader(robot, arm, profile):
    """Capture synchronized RGB-D views and recompute link geometry from measured joints."""
    data = mujoco.MjData(arm.model)
    signature = model_signature(arm.model)
    calibration = profile.get("robot_geometry")
    if calibration is not None and calibration["model_signature"] != signature:
        raise ObservationUnavailable("robot_material_identity_changed")

    def read(scene_epoch):
        observation = robot.get_observation()
        if profile.get("simulation_visual_self_association_m") and not observation.is_simulation:
            raise ObservationUnavailable("simulation_self_association_requires_simulator")
        capture = synchronize_pandaomron(observation, arm.model, data)
        wrist = (observation.image_timing or {}).get("ee_cam")
        if not isinstance(wrist, dict) or local_sim_capture_interval(
            wrist, received_monotonic=observation.received_monotonic, observation_id=str(wrist.get("timestamp_ns")),
        ) != capture:
            raise ObservationUnavailable("multiview_capture_mismatch")
        sample = RobotPoseSample.from_model(
            arm.model, data, ee_body=arm.chain.ee_body, gripper_bodies=arm.chain.gripper_bodies,
            received_monotonic=capture.earliest,
        )
        target = np.asarray(observation.ee_pose, float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ObservationUnavailable("robot_capture_model_mismatch")
        if (
            np.linalg.norm(sample.world_from_ee[:3, 3] - target[:3, 3]) > arm.position_tolerance_m
            or rotation_distance(sample.world_from_ee[:3, :3], target[:3, :3]) > arm.orientation_tolerance_rad
        ):
            raise ObservationUnavailable("robot_capture_model_mismatch")
        wrist_observation = SimpleNamespace(
            rgb=observation.ee_rgb, depth=observation.ee_depth, camera_K=observation.ee_camera_K,
            camera_pose=observation.ee_camera_pose, received_monotonic=capture.received_monotonic,
        )
        observations = {"scene": observation, "wrist": wrist_observation}
        for name, view in (observation.auxiliary_rgbd_views or {}).items():
            timing = view["timing"]
            if local_sim_capture_interval(timing, received_monotonic=observation.received_monotonic,
                                          observation_id=str(timing.get("timestamp_ns"))) != capture:
                raise ObservationUnavailable("multiview_capture_mismatch")
            observations[name] = SimpleNamespace(
                **{key: view[key] for key in ("rgb", "depth", "camera_K", "camera_pose")},
                received_monotonic=capture.received_monotonic,
            )
        material = None
        query = None
        qualify_surface = None
        if calibration is not None:
            geometry_id, bounds_id = calibration["id"], calibration["configuration_profile_id"]
            sample = replace(
                sample, position_error_m=calibration["ee_position_error_m"],
                rotation_error_rad=calibration["ee_rotation_error_rad"],
            )
            solids, erosion = [], []
            for name, entry in calibration["inner_solids"].items():
                gid = arm.model.geom(name).id
                solids.append(WorldSolid.from_geom(
                    arm.model, data, gid, convex_material_calibration_id=entry.get("convex_material_id"),
                ))
                erosion.append(entry["erosion_m"])
            material = CapturedRobotMaterial(
                capture, signature, geometry_id, scene_epoch, tuple(solids), tuple(erosion), bounds_id,
                surface_tolerance_m=calibration.get("surface_tolerance_m", 0.0),
            )
            if calibration.get("surface_ray_association"):
                from emet.controller.manipulation.robot_surface_rays import measured_robot_surfaces

                material = replace(material, measured_surface_points=np.empty((0, 3)))
                captured_data = mujoco.MjData(arm.model)
                mujoco.mj_copyData(captured_data, arm.model, data)
                captured_material = material

                surface_lock = Lock()
                qualified = []

                def qualify_surface():
                    # This closure owns one immutable capture, including its FK.
                    with surface_lock:
                        if not qualified:
                            qualified.append(build_surface())
                        return qualified[0]

                def build_surface():
                    points = measured_robot_surfaces(
                        arm.model, captured_data, tuple(observations.values()),
                        tolerance_m=calibration.get("surface_tolerance_m", .015),
                        known_material=captured_material,
                        collision_proxy_self_association=calibration.get("collision_proxy_self_association", False),
                        visual_surface_tolerance_m=profile.get("simulation_visual_self_association_m", .003),
                    )
                    return replace(captured_material, measured_surface_points=points)

            def query(model, candidate, geom_id):
                name = model.geom(geom_id).name
                configured = calibration["outer_solids"].get(name)
                if configured is None:
                    raise ObservationUnavailable("robot_collision_envelope_unknown")
                error = configured["error_m"]
                solid = WorldSolid.from_geom(
                    model, candidate, geom_id, offset_m=np.sqrt(3) * arm.collision_margin_m + error,
                    convex_material_calibration_id=configured.get("convex_material_id"),
                )
                return RobotGeometryBound(solid, geom_id, signature, geometry_id, bounds_id, error)

        return TransferSample(
            capture, observations,
            sample, replace(sample, received_monotonic=capture.latest), material, query, {},
            qualify_surface,
        )

    return read
