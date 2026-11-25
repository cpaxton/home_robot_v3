# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Robot-side bindings for the shared observed pick-and-place executor."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker
from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.motion.mujoco_collision import body_subtree
from emet.robots.base import ArmChain

from .observed_adapter import ObservedNativeAdapter


@dataclass(frozen=True)
class ObservedArm:
    model: mujoco.MjModel
    root_body: str
    chain: ArmChain
    synchronize: Callable
    command_joints: Callable
    command_gripper_joints: Callable
    gripper_control_joint_names: tuple[str, ...]
    coupled_groups: tuple[tuple[str, ...], ...] = ()
    position_tolerance_m: float = 0.003
    orientation_tolerance_rad: float = 0.05
    collision_margin_m: float = 0.005
    allow_contact_departure: bool = False
    joint_tolerance: float = .03
    linear_joint_tolerance_m: float | None = None
    hand_from_ee: np.ndarray | None = None
    maximum_opening_m: float | None = None

    def __post_init__(self):
        if self.hand_from_ee is not None:
            from emet.controller.manipulation.observed_effects import rigid_copy

            object.__setattr__(self, 'hand_from_ee', rigid_copy(self.hand_from_ee))
        if self.maximum_opening_m is not None and (not np.isfinite(self.maximum_opening_m) or self.maximum_opening_m <= 0):
            raise ValueError('invalid_native_gripper_opening')
        bodies = body_subtree(self.model, self.root_body)
        if any(int(body) not in bodies for body in (*self.model.geom_bodyid, *self.model.jnt_bodyid)):
            raise ValueError("observed_robot_only_model_required")
        self.model.body(self.chain.ee_body)
        for body in self.chain.gripper_bodies:
            self.model.body(body)
        for joint in self.chain.joint_names:
            self.model.joint(joint)
        motor_joints = self.gripper_control_joint_names
        open_joints = dict(self.chain.gripper_open_configuration)
        if not motor_joints or len(set(motor_joints)) != len(motor_joints) or not set(motor_joints) <= open_joints.keys():
            raise ValueError("native_gripper_joint_mapping_invalid")
        for joint in motor_joints:
            self.model.joint(joint)
        if not np.isfinite(self.collision_margin_m) or self.collision_margin_m < 0:
            raise ValueError("invalid_observed_collision_margin")
        if any(not callable(f) for f in (self.synchronize, self.command_joints, self.command_gripper_joints)):
            raise ValueError("native_arm_transport_unavailable")


@dataclass(frozen=True)
class TransferCandidate:
    """Metric world-frame targets from task geometry and a calibrated tool frame."""

    grasp: tuple
    place: tuple
    close_joints: tuple[tuple[str, float], ...]

    def __post_init__(self):
        if len(self.grasp) != 3 or len(self.place) != 3 or not self.close_joints:
            raise ValueError("invalid_transfer_candidate")
        names = [name for name, _ in self.close_joints]
        if len(set(names)) != len(names) or not np.isfinite([value for _, value in self.close_joints]).all():
            raise ValueError("invalid_transfer_candidate")
        for position, rotation in (*self.grasp, *self.place):
            p, r = np.asarray(position), np.asarray(rotation)
            if (
                p.shape != (3,)
                or r.shape != (3, 3)
                or not np.isfinite(p).all()
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(r), 1.0, atol=1e-6)
            ):
                raise ValueError("invalid_transfer_candidate")


class _BoundPayload:
    input_mode = "observed"

    def __init__(self, source, task):
        self.source = source
        self.task = task

    def observe_payload(self, object_ref):
        if object_ref != self.task.object_ref:
            raise ObservationUnavailable("payload_identity_mismatch")
        return self.source.snapshot(object_ref, self.task.support_ref).frame.payload


def observed_arm_factory(*, robot_name, load_model, build_source, tasks, candidates, confirm_placement=None, configure_arm=None,
                         grounder_factory=None, initial_payload_state='unknown'):
    """Bind a benchmark's measured source to its declared embodiment in CHAT."""
    from emet.robots import get_robot_backend

    backend = get_robot_backend(robot_name)
    if backend is None:
        raise ValueError("native_robot_unsupported")
    spec = backend.get_spec()

    def bind(robot, _executor):
        bound = getattr(robot, "spec", None)
        if bound is not None and bound.name != spec.name:
            raise ValueError("native_robot_unsupported")
        model = load_model()
        arm = backend.bind_observed_arm(robot, model)
        if configure_arm is not None:
            arm = configure_arm(robot, arm)
        from .execution_phase import ManipulationExecutionPhase

        execution_phase = ManipulationExecutionPhase(initial_payload_state)
        source = build_source(robot, arm)
        adapter = build_observed_arm_adapter(
            robot, robot_name=spec.name, arm=arm, source=source, tasks=tasks, candidates=candidates,
            confirm_placement=confirm_placement, execution_phase=execution_phase,
        )
        if grounder_factory is not None:
            install = getattr(source, 'set_grounder', None)
            if not callable(install):
                raise ValueError('runtime_grounding_source_unavailable')
            install(grounder_factory(execution_phase.can_query))
        return adapter

    return bind


def build_observed_arm_adapter(robot, *, robot_name, arm, source, tasks, candidates, confirm_placement=None,
                               execution_phase=None, clock=time.monotonic):
    """Compose a native arm and measured scene source without a benchmark-specific tool."""

    def refine(task, snapshot, placement_region=None, request=None):
        if snapshot.admission is None:
            raise ObservationUnavailable("robot_material_geometry_unavailable")
        if snapshot.target_contact is None:
            raise ObservationUnavailable("target_contact_geometry_unknown")
        if snapshot.source_support_contact is None or snapshot.support_contact is None:
            raise ObservationUnavailable("support_geometry_unknown")
        propose = getattr(source, 'grasp_proposals', None)
        proposals = ()
        proposed_count, height_adjustments = 0, []
        if (callable(propose) and arm.hand_from_ee is not None
                and getattr(candidates, 'supports_grasp_proposals', False)):
            from emet.perception.grasps.observed import GraspProposal

            proposals = tuple(propose(snapshot))
            if any(not isinstance(p, GraspProposal) for p in proposals):
                raise ValueError('invalid_grasp_proposal')
            proposed_count = len(proposals)
            if arm.maximum_opening_m is not None:
                proposals = tuple(p for p in proposals if p.opening_m <= arm.maximum_opening_m)
        strict_grasp = False
        if request is not None and request.hints.get('grasp'):
            from emet.perception.grasps.observed import GraspProposal

            hint = request.hints['grasp']
            if (set(hint)-{'world_from_hand', 'opening_m', 'observation_id', 'frame_id'}
                    or hint.get('frame_id') != 'world'
                    or hint.get('observation_id') != snapshot.frame.capture.observation_id):
                raise ValueError('invalid_manipulation_hint')
            if arm.hand_from_ee is None or not getattr(candidates, 'supports_grasp_proposals', False):
                raise ValueError('manipulation_options_unsupported')
            if isinstance(hint.get('opening_m'), bool):
                raise ValueError('invalid_manipulation_hint')
            try:
                proposal = GraspProposal(hint['observation_id'], hint['world_from_hand'], hint['opening_m'],
                                        provenance='supervisor_world_hand')
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('invalid_manipulation_hint') from error
            proposed_count, proposals = 1, (proposal,)
            strict_grasp = request.hint_mode == 'strict'
            if arm.maximum_opening_m is not None and proposal.opening_m > arm.maximum_opening_m:
                if strict_grasp:
                    raise ObservationUnavailable('no_observed_transfer_candidate')
                proposals = ()
        if proposals:
            from dataclasses import replace

            from emet.motion.observed_pose_targets import hand_lowest_offset

            measured = mujoco.MjData(arm.model)
            measured.qpos[:] = snapshot.frame.robot_before.joints
            mujoco.mj_kinematics(arm.model, measured)
            bodies = set()
            for body in arm.chain.gripper_bodies:
                bodies.update(body_subtree(arm.model, body))
            geometry = tuple(gid for gid in range(arm.model.ngeom)
                             if int(arm.model.geom_bodyid[gid]) in bodies)
            plane = snapshot.source_support_contact.patch.plane
            adjusted = []
            for proposal in proposals:
                target = proposal.world_from_hand @ arm.hand_from_ee
                lowest = hand_lowest_offset(arm.model, measured, arm.chain.ee_body, geometry, target[:3, :3])
                floor = target[:2, 3] @ plane[:2]+plane[2]
                shift = max(0., floor+snapshot.source_support_contact.patch.measurement_margin_m+.002-lowest-target[2, 3])
                if strict_grasp and shift > 1e-6:
                    raise ObservationUnavailable('no_observed_transfer_candidate')
                height_adjustments.append(float(shift))
                pose = proposal.world_from_hand.copy()
                pose[2, 3] += shift
                adjusted.append(replace(proposal, world_from_hand=pose))
            proposals = tuple(adjusted)
        options = tuple(candidates(task, snapshot, placement_region, grasp_proposals=proposals,
            hand_from_ee=arm.hand_from_ee) if proposals else candidates(task, snapshot, placement_region))
        if not options:
            raise ObservationUnavailable("no_observed_transfer_candidate")
        if len(options) > 16 or any(not isinstance(option, TransferCandidate) for option in options):
            raise ValueError("invalid_transfer_candidate")
        if request is not None and request.hints.get('approach'):
            from .candidate_guidance import guide_approach

            options = guide_approach(options, request.hints['approach'],
                observation_id=snapshot.frame.capture.observation_id)
        opening = arm.chain.gripper_open_configuration
        if not opening:
            raise ObservationUnavailable("unsupported_gripper_open_geometry")
        for option in options:
            if tuple(name for name, _ in option.close_joints) != arm.gripper_control_joint_names:
                raise ValueError("invalid_transfer_candidate")
            for name, target in option.close_joints:
                joint = arm.model.joint(name)
                if not joint.limited[0] or not joint.range[0] <= target <= joint.range[1]:
                    raise ValueError("invalid_transfer_candidate")
        checker = ObservedNativeCollisionChecker(
            arm.model,
            robot_body=arm.root_body,
            ee_body=arm.chain.ee_body,
            scene=snapshot.scene,
            payload_provider=_BoundPayload(source, task),
            scene_revision=snapshot.scene_revision,
            current_revision=source.current_revision,
            received_monotonic=snapshot.frame.capture.received_monotonic,
            margin_m=arm.collision_margin_m,
            admission=snapshot.admission,
            target_contact=snapshot.target_contact,
            source_support_contact=snapshot.source_support_contact,
            support_contact=snapshot.support_contact, clock=clock,
        )
        approach_bodies = set()
        for body in arm.chain.gripper_bodies:
            approach_bodies.update(body_subtree(arm.model, body))
        checker.target_approach_geom_ids = frozenset(
            gid for gid in checker.geom_ids if int(arm.model.geom_bodyid[gid]) in approach_bodies
        )
        executor = PhysicalPickPlaceExecutor(
            robot,
            model=arm.model,
            data=mujoco.MjData(arm.model),
            ee_body=arm.chain.ee_body,
            joint_names=arm.chain.joint_names,
            collision=checker,
            synchronize=arm.synchronize,
            command_joints=arm.command_joints,
            command_gripper_joints=arm.command_gripper_joints,
            coupled_groups=arm.coupled_groups,
            joint_tolerance=arm.joint_tolerance,
            linear_joint_tolerance_m=arm.linear_joint_tolerance_m,
            gripper_open_configuration=opening,
            gripper_close_configuration=options[0].close_joints,
            base_body=arm.root_body,
            position_tolerance_m=arm.position_tolerance_m,
            orientation_tolerance_rad=arm.orientation_tolerance_rad,
            object_pose_provider=checker,
            ik_seed_configurations=(arm.chain.home_arm_q,) if arm.chain.home_arm_q else (),
            approach_joint_routes=getattr(source, "approach_joint_routes", ()),
            joint_planning_timeout_s=getattr(source, "joint_planning_timeout_s", 10.0),
            ik_random_starts=getattr(source, "ik_random_starts", 0),
            preplace_staging_offsets_m=getattr(source, "preplace_staging_offsets_m", ()),
            pregrasp_staging_offsets_m=getattr(source, "pregrasp_staging_offsets_m", ()),
            cartesian_transport=getattr(source, "cartesian_transport", False),
            coalesce_execution_waypoints=getattr(source, "coalesce_execution_waypoints", False),
            execution_shortcut_angle_rad=getattr(source, "execution_shortcut_angle_rad", 0.),
            execution_shortcut_slide_m=getattr(source, "execution_shortcut_slide_m", 0.),
            execution_max_slide_step_m=getattr(source, "execution_max_slide_step_m", .005),
            reuse_validated_grasp_paths=getattr(source, "reuse_validated_grasp_paths", False),
        )
        checker.local_scene = snapshot.local_scene
        if snapshot.local_scene is not None:
            checker.measured_joints = snapshot.frame.robot_before.joints
            # Each segment is checked against the refreshed local snapshot.
            # Publication of an unrelated coarse-map update is not collision.
            checker.current_revision = lambda: checker.scene_revision
        checker.allow_contact_departure = getattr(source, "allow_contact_departure", arm.allow_contact_departure)
        checker.departure_options = getattr(source, "departure_options", {})
        checker.recovery_guides = executor.approach_joint_routes
        executor.grasp_guidance_summary = {
            'source': 'observed_hand_proposals' if proposals else 'calibrated_default',
            'proposed_candidates': proposed_count, 'accepted_candidates': len(proposals),
            'support_height_adjustments_m': height_adjustments,
            'applied_world_from_hand': proposals[0].world_from_hand.tolist()
                if proposals and request is not None and request.hints.get('grasp') else None,
        }

        executor.grasp_targets = list(options[0].grasp)
        executor.place_targets = list(options[0].place)
        executor.transfer_candidates = options
        if confirm_placement is not None:
            executor.release_refiner = lambda: confirm_placement(
                executor, lambda: source.snapshot(task.object_ref, task.support_ref).frame,
                getattr(source, "check_placement_surface", None),
            )
        return executor

    refine.supports_manipulation_request = True
    refine.supports_metric_grasp_hint = arm.hand_from_ee is not None and getattr(candidates, 'supports_grasp_proposals', False)
    return ObservedNativeAdapter(robot, robot_name=robot_name, source=source, refine=refine, tasks=tasks,
                                 max_ik_calls=getattr(source, "planning_max_ik_calls", 128), execution_phase=execution_phase, clock=clock)
