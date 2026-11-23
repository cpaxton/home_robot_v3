# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bind the shared observed transfer scene into the PandaOmron benchmark."""

import json
import logging
import math
import time

import mujoco
import numpy as np

from emet.controller.manipulation.geometry_models import AxialMeshSurface, KnownMeshSurface
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.task.tamp.live_transfer import LiveTransferScene
from emet.controller.task.tamp.observed_adapter import ObservedNativeTask
from emet.controller.task.tamp.observed_arm import observed_arm_factory
from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports
from emet.controller.task.tamp.placement_confirmation import confirm_placement
from emet.controller.task.tamp.transfer_grounding import select_transfer
from emet.robots.pandaomron.observed import fixed_torso_arm, panda_sample_reader


def panda_transfer_factory(profile_path, robot_model_path, object_name, owner):
    profile = json.loads(profile_path.read_text())
    if profile.get("debug_planning"):
        logging.getLogger("emet.controller.task.tamp.observed_adapter").setLevel(logging.INFO)
        logging.getLogger("emet.controller.manipulation.physical_pick_place").setLevel(logging.INFO)
    departure_options = profile.get("contact_departure_options", {})
    if (not isinstance(departure_options, dict) or set(departure_options)-{
        "new_contact_tolerance_m", "search_timeout_s", "maximum_travel_m", "slack_m", "patch_slack_m"
    } or departure_options and not profile.get("contact_departure")):
        raise ValueError("invalid_contact_departure_options")
    if not 0 < departure_options.get("maximum_travel_m", .03) <= .15:
        raise ValueError("invalid_contact_departure_options")
    if not 0 < departure_options.get("slack_m", .001) <= .005:
        raise ValueError("invalid_contact_departure_options")
    if not 0 <= departure_options.get("patch_slack_m", .005) <= .005:
        raise ValueError("invalid_contact_departure_options")
    visual_self = float(profile.get("simulation_visual_self_association_m", 0.))
    if not math.isfinite(visual_self) or not 0 <= visual_self <= .015:
        raise ValueError("invalid_simulation_visual_self_association")
    local_payload = profile.get("simulation_local_payload_contact", False)
    if type(local_payload) is not bool or (local_payload and not profile.get("local_resolution_m")):
        raise ValueError("invalid_simulation_local_payload_contact")
    support_tolerance = float(profile.get("simulation_support_contact_tolerance_m", 0.))
    if (not math.isfinite(support_tolerance) or not 0 <= support_tolerance <= .01
            or support_tolerance and not local_payload):
        raise ValueError("invalid_simulation_support_contact_tolerance")
    best_fit = profile.get("simulation_best_fit_pose", False)
    if type(best_fit) is not bool:
        raise ValueError("invalid_simulation_best_fit_pose")
    contact_close = profile.get("simulation_contact_terminated_close", False)
    if type(contact_close) is not bool:
        raise ValueError("invalid_simulation_contact_close")
    slack = float(profile.get("simulation_clearance_slack_m", 0.0))
    if not math.isfinite(slack) or not 0 <= slack <= .005 or (slack and not profile.get("local_resolution_m")):
        raise ValueError("invalid_simulation_clearance_slack")
    prior = json.loads((profile_path.parent / profile["payload_prior"]).read_text())
    prior_scale = float(profile.get("payload_prior_scale", 1.))
    if not math.isfinite(prior_scale) or not .5 <= prior_scale <= 2.:
        raise ValueError("invalid_payload_prior_scale")
    geometry = KnownMeshSurface(
        np.asarray(prior["surface_points_reference_m"])*prior_scale, prior["surface_normals_reference"],
        np.asarray(prior["vertices_reference_m"])*prior_scale,
    )
    if profile.get("payload_axial_enclosure") is not None:
        geometry = AxialMeshSurface(geometry)
    task = ObservedNativeTask("payload", "basin-floor", object_name, "sink", source_support_ref="counter")
    grasps = tuple(
        CalibratedGrasp(
            entry["object_from_ee"], tuple(entry["approach_axis_ee"]),
            tuple(map(tuple, entry["close_joints"])),
            approach_m=entry["approach_m"], lift_m=entry["lift_m"], retreat_m=entry["retreat_m"],
            world_aligned=entry.get("world_aligned", False),
            transit_clearance_m=entry.get("transit_clearance_m"),
            keep_current_heading=entry.get("keep_current_heading", False),
        )
        for entry in profile["grasps"]
    )
    mode = "diagnostic_pixel_regions" if any("box" in entry for entry in profile["grounding"].values()) else "semantic"
    lock_torso = profile.get("fixed_torso", False)
    if type(lock_torso) is not bool or lock_torso and (profile.get("approach_joint_routes") or profile.get("guarded_placement")):
        raise ValueError("invalid_fixed_torso_profile")
    probe = profile.get("guarded_placement")
    if probe is not None and (not 0 < probe["half_band_m"] <= .005 or not 0 < probe["hover_gap_m"] <= .010):
        raise ValueError("invalid_probe_contact_band")

    def build_source(robot, arm):
        warm_started = time.monotonic()
        if profile.get("local_resolution_m"):
            from emet.motion.local_geometry_distance import cell_separations
            from emet.motion.observed_solids import WorldSolid

            scratch = mujoco.MjData(arm.model)
            mujoco.mj_kinematics(arm.model, scratch)
            for name, entry in profile["robot_geometry"]["outer_solids"].items():
                solid = WorldSolid.from_geom(arm.model, scratch, arm.model.geom(name).id,
                    convex_material_calibration_id=entry.get("convex_material_id"))
                cell_separations(solid, np.empty((0, 3)), resolution=profile["local_resolution_m"])
        warmup_s = time.monotonic()-warm_started
        if profile.get("contact_departure"):
            if not robot.get_observation().is_simulation:
                raise ValueError("unqualified_contact_departure_requires_simulator")
        if probe is not None:
            session = robot.get_emet_session() or {}
            if (
                session.get("capabilities", {}).get("guarded_placement_probe") is not True
                or session.get("probe_calibration_id") != probe["calibration_id"]
            ):
                raise ObservationUnavailable("probe_unavailable")
        segment = None
        if any("label" in role for role in profile["grounding"].values()):
            from emet.mapping.scene_graph.processor import SceneGraphProcessor

            segment = SceneGraphProcessor(config_name="cpu_scene_graph", device="cpu").segment_current_view
        source = LiveTransferScene(
            arm=arm, task=task, read_sample=panda_sample_reader(robot, arm, profile),
            select=lambda sample, _task: select_transfer(sample, profile=profile, segment=segment),
            payload_model=geometry,
            fingertip_geom_ids=tuple(arm.model.geom(name).id for name in profile["fingertips"]),
            profile=profile,
        )
        source.native_query_warmup_s = warmup_s
        owner.append(source)
        source.source.allow_contact_departure = bool(profile.get("contact_departure"))
        source.source.departure_options = departure_options
        source.source.approach_joint_routes = profile.get("approach_joint_routes", ())
        source.source.joint_planning_timeout_s = profile.get("joint_planning_timeout_s", 10.0)
        source.source.ik_random_starts = profile.get("ik_random_starts", 8)
        source.source.preplace_staging_offsets_m = profile.get("preplace_staging_offsets_m", ())
        source.source.pregrasp_staging_offsets_m = profile.get("pregrasp_staging_offsets_m", ())
        source.source.cartesian_transport = profile.get("cartesian_transport", False)
        source.source.coalesce_execution_waypoints = profile.get("coalesce_execution_waypoints", False)
        source.source.reuse_validated_grasp_paths = profile.get("reuse_validated_grasp_paths", False)
        source.source.planning_max_ik_calls = profile.get("planning_max_ik_calls", 128)
        return source.source

    def confirm(executor, capture_payload, check_surface):
        return confirm_placement(
            executor, calibration_id=probe["calibration_id"], half_band_m=probe["half_band_m"],
            capture_payload=capture_payload, check_surface=check_surface,
        )

    factory = observed_arm_factory(
        robot_name="pandaomron", load_model=lambda: mujoco.MjModel.from_xml_path(str(robot_model_path)),
        build_source=build_source, tasks=(task,),
        candidates=lambda _task, snapshot, region: candidates_from_supports(
            snapshot, grasps, placement_region=region,
            placement_gap_m=probe["hover_gap_m"] if probe is not None else 0.0,
            yaw_options=tuple(math.radians(yaw) for yaw in profile.get("placement_yaw_degrees", (0., 90.))),
        ),
        confirm_placement=confirm if probe is not None else None,
        configure_arm=fixed_torso_arm if lock_torso else None,
    )
    if profile.get("simulation_ignore_environment") or profile.get("simulation_allow_unknown") or profile.get("approach_joint_routes") or slack or contact_close or local_payload or visual_self or best_fit:
        native_factory = factory

        def factory(robot, executor):
            session = robot.get_emet_session() or {}
            observation = robot.get_observation()
            if (profile.get("simulation_ignore_environment") or profile.get("simulation_allow_unknown") or slack or contact_close or local_payload or visual_self or best_fit) and (
                session.get("runtime_kind") != "robocasa_native" or not observation.is_simulation
            ):
                raise ValueError("simulation_collision_override_requires_simulator")
            adapter = native_factory(robot, executor)
            refine = adapter.refine

            def diagnostic_refine(*args, **kwargs):
                result = refine(*args, **kwargs)
                if profile.get("debug_planning"):
                    def event(**fields):
                        if "command" not in fields or "reason" in fields:
                            logging.getLogger(__name__).warning("Native execution: %s", fields)
                    result.event = event
                result.collision.simulation_ignore_environment = bool(profile.get("simulation_ignore_environment"))
                result.collision.simulation_clearance_slack_m = slack
                result.collision.simulation_contact_terminated_close = contact_close
                result.collision.simulation_local_payload_contact = local_payload
                result.collision.simulation_support_contact_tolerance_m = support_tolerance
                return result

            adapter.refine = diagnostic_refine
            normal_data = adapter._data
            assistance = ["simulation_best_fit_pose"] if best_fit else []
            if profile.get("target_surface_association_m", .008) > .008:
                assistance.append("expanded_target_surface_association")
            if visual_self:
                assistance.append("simulation_visual_self_association")
            if support_tolerance:
                assistance.append("simulation_support_contact_tolerance")
            if local_payload:
                assistance.append("simulation_local_payload_contact")
            if contact_close:
                assistance.append("simulation_contact_terminated_close")
            if slack:
                assistance.append("simulation_clearance_slack")
            if departure_options:
                assistance.append("configured_contact_departure_bounds")
            if profile.get("preplace_staging_offsets_m") or profile.get("pregrasp_staging_offsets_m"):
                assistance.append("configured_cartesian_staging_proposals")
            if profile.get("approach_joint_routes"):
                assistance.append("configured_joint_route_proposal")
            if profile.get("simulation_ignore_environment"):
                assistance.append("simulation_environment_and_payload_contact_bypass")
            if profile.get("simulation_allow_unknown"):
                assistance.append("simulation_unknown_space_optimism")
            def diagnostic_data(**extra):
                data = {**normal_data(**extra), "assistance": assistance, "clearance_slack_m": slack,
                        "torso_motion": "held_measured" if lock_torso else "planned"}
                if support_tolerance:
                    data["support_contact_tolerance_m"] = support_tolerance
                if visual_self:
                    data["visual_self_association_m"] = visual_self
                if profile.get("contact_departure"):
                    data["contact_representation"] = "measured_surface_patches"
                if departure_options:
                    data["contact_departure_options"] = dict(departure_options)
                if profile.get("simulation_ignore_environment"):
                    data["collision_scope"] = "robot_self_only"
                return data

            adapter._data = diagnostic_data
            return adapter

    return factory, mode
