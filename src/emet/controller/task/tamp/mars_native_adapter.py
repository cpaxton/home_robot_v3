"""Privileged Mars simulator adapter for the existing public TAMP tools.

Requires an explicit matching model and a normal arm-reference client. Planning
uses private MuJoCo data; this does not enable the real Mars bridge or observed
perception. Coordinated transport has independent public-tool acceptance gates.
"""

import mujoco
import numpy as np

from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.controller.task.tamp.native_adapter import NativeTableAdapter
from emet.rl.manipulation.profiles import MARS


class MarsPhysicalPickPlaceExecutor(PhysicalPickPlaceExecutor):
    def __init__(self, *args, ik_attempt, **kwargs):
        super().__init__(*args, **kwargs)
        self.ik_attempt = ik_attempt
        self.pose_seed_hints = []

    def _execute_path(self, phase, path):
        if getattr(self.robot, "arm_execution", "joint") == "cartesian":
            from emet.controller.task.tamp.mars_pose_executor import execute_pose_path

            targets = {
                "pregrasp": (self.grasp_targets, 0),
                "grasp": (self.grasp_targets, 1),
                "lift": (self.grasp_targets, 2),
                "preplace": (self.place_targets, 0),
                "place": (self.place_targets, 1),
                "retreat": (self.place_targets, 2),
            }
            choices, index = targets.get(phase, ([], 0))
            goal = None
            if choices:
                position, rotation = choices[index]
                goal = np.r_[position, np.asarray(rotation)[:, :2].ravel()]
            return execute_pose_path(self.robot, self, phase, path, goal_pose=goal)
        return super()._execute_path(phase, path)

    def remember_pose_seed(self, position, rotation, q):
        key = np.r_[position, np.asarray(rotation).ravel()]
        self.pose_seed_hints = [
            item for item in self.pose_seed_hints if not np.allclose(item[0], key, atol=1e-10, rtol=0)
        ]
        self.pose_seed_hints.append((key.copy(), np.asarray(q).copy()))
        self.pose_seed_hints = self.pose_seed_hints[-64:]

    def plan_pose(self, position, rotation, *, ordered_groups=False):
        from emet.motion.mujoco_arm_ik import solve_pose_ik

        initial = self.data.qpos.copy()
        original = initial[self.qadr].copy()
        limits = np.array([self.model.joint(n).range for n in self.joint_names])
        accepted = False
        error = "pose_ik_failed"
        path_error = None
        target_key = np.r_[position, np.asarray(rotation).ravel()]
        hint = None
        if self.pose_seed_hints:
            hint = min(self.pose_seed_hints, key=lambda item: np.linalg.norm(item[0] - target_key))[1]
        candidates = ([hint] if hint is not None else []) + [None, 1.0, 1.5, -1.0, -1.5]
        try:
            for attempt, elbow in enumerate(candidates):
                self.ik_attempt()
                scratch = mujoco.MjData(self.model)
                mujoco.mj_copyData(scratch, self.model, self.data)
                scratch.qpos[:] = initial
                if isinstance(elbow, np.ndarray):
                    scratch.qpos[self.qadr] = elbow
                elif elbow is not None:
                    q = original.copy()
                    difference = elbow - q[2]
                    q[2] = elbow
                    q[1] -= difference / 2
                    q[3] -= difference / 2
                    scratch.qpos[self.qadr] = np.clip(q, limits[:, 0], limits[:, 1])
                result = solve_pose_ik(
                    self.model,
                    scratch,
                    ee_body=self.ee_body,
                    joint_names=self.joint_names,
                    target_pos=position,
                    target_rotation=rotation,
                    coupled_groups=self.coupled_groups,
                    joint_limit_margins=self.joint_limit_margins,
                    tol_m=self.position_tolerance_m,
                    tol_rad=self.orientation_tolerance_rad,
                )
                if not result.success:
                    error = (
                        f"pose_ik_failed:position={result.pos_error_m:.4f},rotation={result.orientation_error_rad:.4f}"
                    )
                    self.event(
                        phase="pose_replan_attempt",
                        attempt=attempt,
                        ik_success=False,
                        position_error_m=result.pos_error_m,
                        orientation_error_rad=result.orientation_error_rad,
                        reason=error,
                    )
                    continue
                goal = scratch.qpos[self.qadr].copy()
                self.data.qpos[:] = initial
                path, error = self.plan_joint_target(goal, ordered_groups=ordered_groups)
                if error is None:
                    from emet.rl.manipulation.grasp_setup import support_margin

                    trial = mujoco.MjData(self.model)
                    mujoco.mj_copyData(trial, self.model, self.data)
                    payload = self.model.body(self.collision.payload_body).id if self.collision.payload_body else None
                    for q in path:
                        trial.qpos[self.qadr] = q
                        if self.collision.configuration_collides(self.model, trial):
                            error = "arm_path_collision"
                            break
                        if support_margin(self.model, trial, MARS, payload)["margin_m"] < 0.005:
                            error = "arm_path_static_support_margin"
                            break
                self.event(
                    phase="pose_replan_attempt",
                    attempt=attempt,
                    ik_success=True,
                    position_error_m=result.pos_error_m,
                    orientation_error_rad=result.orientation_error_rad,
                    reason=error,
                )
                if error is None:
                    self.remember_pose_seed(position, rotation, goal)
                    accepted = True
                    return path, None
                path_error = error
            # A later seed's failed IK must not hide a collision/support failure
            # from a seed that actually solved the requested pose.
            return None, path_error or error
        finally:
            if not accepted:
                self.data.qpos[:] = initial
                mujoco.mj_kinematics(self.model, self.data)

    def _close_gripper_until_still(self, timeout_s=10.0):
        from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult

        start = self._motion_time()
        result = super()._close_gripper_until_still(timeout_s)
        if not result.success:
            return result
        stable = 0.0
        # Match the qualified Mars grasp setup's one-second bilateral gate,
        # inside the public executor's original closing timeout.
        while self._motion_time() - start < timeout_s:
            self.robot.step(MARS.dt)
            self.synchronize(self.data)
            if self.robot.failure:
                return PhysicalMotionResult(False, self.robot.failure, "grasp")
            stable = stable + MARS.dt if self.robot.last_bilateral_contact else 0.0
            if stable >= 1.0:
                return PhysicalMotionResult(True, "bilateral_grip_settled", "grasp")
        return PhysicalMotionResult(False, "grasp_contact_unverified", "grasp")

    def plan_gripper_open(self):
        """Certify both mimic jaws, including bounded measured soft-stop deflection.

        The simulator's compliant stop can measure just beyond the commanded
        open limit. Do not teleport/clamp measured geometry or expand targets:
        collision-check the actual-to-in-range opening path on private data.
        Larger deflections remain rejected. This uses the existing 3 mrad open
        encoder tolerance, not a relaxed arm/retention/contact contract.
        """
        names = tuple(self.gripper_open_configuration)
        addresses = np.array([self.model.joint(n).qposadr[0] for n in names])
        before = self.data.qpos.copy()
        start = before[addresses].copy()
        goal = np.array(list(self.gripper_open_configuration.values()))
        limits = np.array([self.model.joint(n).range for n in names])
        if not np.isfinite(start).all() or np.any(start < limits[:, 0] - 0.003) or np.any(start > limits[:, 1] + 0.003):
            return None, "gripper_open_measured_bounds"
        path = [
            start + t * (goal - start)
            for t in np.linspace(0, 1, max(2, int(np.ceil(np.max(abs(goal - start)) / 0.025)) + 1))
        ]
        try:
            for q in path:
                self.data.qpos[addresses] = q
                if self.collision.configuration_collides(self.model, self.data):
                    return None, "gripper_open_path_collision"
        finally:
            self.data.qpos[:] = before
            mujoco.mj_kinematics(self.model, self.data)
        self.data.qpos[addresses] = goal
        mujoco.mj_kinematics(self.model, self.data)
        return path, None


class MarsNativeTableAdapter(NativeTableAdapter):
    robot_name = "innate_mars"
    finger_bodies = MARS.finger_bodies

    def __init__(self, robot, *, model):
        from emet.controller.manipulation.native_inputs import ObservationUnavailable

        if getattr(robot, "native_mars_task_control", False) is not True:
            raise ObservationUnavailable("native_mars_control_unavailable")
        super().__init__(robot, model=model)
        self._executing = False
        self.execution_ik_calls = 0

    def _data(self, **extra):
        reference_callable = bool(getattr(self.robot.policy, "is_reference_controller", False))
        learned = self.robot.policy is not None and not reference_callable
        controller = (
            self.robot.controller_name
            if reference_callable
            else "mars_checkpoint_policy"
            if learned
            else "mars_task_pose_teacher"
        )
        data = super()._data(robot=self.robot_name, controller=controller, **extra)
        data["policy_calls"] = self.robot.policy_calls if learned else 0
        if reference_callable:
            data["reference_controller_calls"] = self.robot.policy_calls
        data["policy_admission"] = getattr(self.robot.policy, "admission", None)
        data["execution_ik_calls"] = self.execution_ik_calls
        data["stop_confirmed"] = self.robot.stop_confirmed
        data["base_motion"] = "coordinated"
        data["collision_scope"] = "sampled_whole_body_arm_payload; runtime force/wheel monitoring"
        return data

    def _executor(self, obj):
        from emet.eval.physical_tamp import make_scene_checker
        from emet.rl.manipulation.safety import relative_pose

        joints = tuple(name for group in MARS.joint_groups for name in group)
        checker = make_scene_checker(self.model, {"object_body": obj, "gripper_bodies": MARS.finger_bodies})

        def command(q):
            self.synchronize(self.data)
            return self.robot.command_arm_reference(np.asarray(q), blocking=False)

        executor = MarsPhysicalPickPlaceExecutor(
            self.robot,
            ik_attempt=self._take_ik_attempt,
            model=self.model,
            data=self.data,
            ee_body=MARS.ee_body,
            joint_names=joints,
            collision=checker,
            synchronize=self.synchronize,
            command_joints=command,
            gripper_open_configuration={MARS.gripper_joint: MARS.gripper_open, "joint6M": -MARS.gripper_open},
            base_body=MARS.base_body,
            input_mode="privileged",
            joint_tolerance=0.005,
            endpoint_settle_s=0.3,
            use_sim_time=True,
            joint_limit_margins=dict.fromkeys(joints, MARS.tracking_trim_limit + 0.01),
            position_tolerance_m=0.003,
            orientation_tolerance_rad=0.02,
            event=lambda **row: self.events.append({"sim_time": float(self.data.time), **row}),
        )
        original = executor._execute_path

        def execute_stage(phase, path):
            self.synchronize(self.data)
            if np.max(abs(self.data.qpos[executor.qadr] - path[0])) > executor.joint_tolerance:
                from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult

                targets = {
                    "pregrasp": executor.grasp_targets[0] if executor.grasp_targets else None,
                    "grasp": executor.grasp_targets[1] if executor.grasp_targets else None,
                    "lift": executor.grasp_targets[2] if executor.grasp_targets else None,
                    "preplace": executor.place_targets[0] if executor.place_targets else None,
                    "place": executor.place_targets[1] if executor.place_targets else None,
                    "retreat": executor.place_targets[2] if executor.place_targets else None,
                }
                target = targets.get(phase)
                if target is not None:
                    if not executor.payload_retained():
                        return PhysicalMotionResult(False, "payload_not_retained", phase)
                    if self.robot.arm_execution == "cartesian":
                        # Pose acceptance need not reproduce the planner's
                        # particular joint solution. Reconnect measured joints
                        # to the already planned endpoint through the normal
                        # collision planner; the Cartesian executor separately
                        # checks its FK against the unchanged world-pose goal.
                        goal = np.asarray(path[-1]).copy()
                        trial = mujoco.MjData(self.model)
                        mujoco.mj_copyData(trial, self.model, self.data)
                        trial.qpos[executor.qadr] = goal
                        mujoco.mj_kinematics(self.model, trial)
                        body = trial.body(MARS.ee_body)
                        distance = float(np.linalg.norm(body.xpos - target[0]))
                        angle = float(
                            np.arccos(
                                np.clip((np.trace(np.asarray(target[1]).T @ body.xmat.reshape(3, 3)) - 1) / 2, -1, 1)
                            )
                        )
                        reuse = (
                            distance <= executor.position_tolerance_m and angle <= executor.orientation_tolerance_rad
                        )
                        if reuse:
                            path, error = executor.plan_joint_target(goal)
                        else:
                            path, error = executor.plan_pose(target[0], target[1])
                        executor.event(
                            phase=phase,
                            reference_replan="measured_to_joint_witness" if reuse else "measured_world_pose",
                            accepted=error is None,
                            reason=error,
                            execution_ik_added=not reuse,
                            witness_position_error_m=distance,
                            witness_orientation_error_rad=angle,
                        )
                    else:
                        path, error = executor.plan_pose(target[0], target[1])
                    if error:
                        return PhysicalMotionResult(False, error, phase)
            self.robot.phase = {
                "pregrasp": "pregrasp",
                "grasp": "pregrasp",
                "lift": "carry",
                "preplace": "carry",
                "place": "place",
                "retreat": "release",
            }.get(phase, "reach")
            self.robot.payload = self.robot.model.body(obj).id
            if phase == "lift":
                self.robot.retention_reference = relative_pose(self.robot.data, self.robot.servo.ee, self.robot.payload)
            if phase in {"preplace", "place"}:
                # Resume compensation only for the verified free-space arm
                # segment; the robot disables it again at support contact.
                self.robot.servo.set_base_bias_compensation(True)
            result = original(phase, path)
            if phase == "lift" and result.success:
                self.robot.servo.set_base_bias_compensation(True)
            if phase == "place" and result.success and not self.robot.confirm_support_contact():
                from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult

                return PhysicalMotionResult(False, self.robot.failure or "support_contact_unverified", "place")
            return result

        def execute(phase, path):
            if phase == "lift" and self.robot.arm_execution == "cartesian":
                self.synchronize(self.data)
                with checker.payload_support_departure(self.model, self.data) as contacts:
                    executor.event(
                        phase="lift_support_departure",
                        contacts=contacts,
                        maximum_initial_compression_m=0.002,
                        deepening_permitted=False,
                    )
                    return execute_stage(phase, path)
            return execute_stage(phase, path)

        executor._execute_path = execute
        return executor

    def _pinch_center_offset(self):
        scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(scratch, self.model, self.data)
        for name in ("joint6", "joint6M"):
            scratch.qpos[self.model.joint(name).qposadr[0]] = (
                MARS.gripper_open if name == "joint6" else -MARS.gripper_open
            )
        mujoco.mj_kinematics(self.model, scratch)
        # Match the robot-derived open-jaw pinch convention of the qualified
        # fixture. The archived pads, friction and torque remain unchanged.
        points = [
            scratch.body(name).xpos + scratch.body(name).xmat.reshape(3, 3) @ [0.03, 0, 0]
            for name in MARS.finger_bodies
        ]
        ee = scratch.body(MARS.ee_body)
        return ee.xmat.reshape(3, 3).T @ (np.mean(points, axis=0) - ee.xpos)

    def _home_rotation(self, executor):
        return self.data.body(executor.ee_body).xmat.reshape(3, 3).copy()

    def _grasp_frames(self, home_rotation, rotation, height):
        frames = []
        # Favor a supported upper-body pinch, reducing the gravity moment and
        # keeping the wrist out of the support. Every alternative is planned
        # against unchanged geometry and the original public acceptance gates.
        for pitch, fraction in (
            (0.0, -0.075),
            (0.0, 0.0),
            (-np.pi / 4, 0.4),
            (-np.pi / 4, 0.25),
            (0.0, 0.4),
            (np.pi / 6, 0.25),
            (np.pi / 4, 0.25),
            (-np.pi / 6, 0.4),
        ):
            c, sn = np.cos(pitch), np.sin(pitch)
            frames.append((home_rotation @ np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]]), fraction * height))
        return frames

    def _grasp_targets(self, grasp, rotation):
        return [(grasp + delta, rotation) for delta in ([0, 0, 0.01], [0, 0, 0], [0, 0, 0.06])]

    def _take_ik_attempt(self):
        if self._executing:
            if self.execution_ik_calls >= 128:
                raise ValueError("execution_replan_budget_exhausted")
            self.execution_ik_calls += 1
        else:
            if self.planning_ik_calls >= self.max_planning_ik_calls:
                raise ValueError("no_plan_within_budget")
            self.planning_ik_calls += 1

    def _plan_candidate_pose(self, executor, target, rotation, *, ordered_groups=False):
        return executor.plan_pose(target, rotation, ordered_groups=ordered_groups)

    def _prepare_approach(self, executor, obj):
        from emet.controller.task.tamp.mars_transport import plan_transport

        # The fixture starts above the supported object. Preserve that measured
        # approach rather than raising through it and disturbing the grasp.
        path = [self.data.qpos[executor.qadr].copy()]
        yaw = np.arctan2(
            self.data.body(MARS.base_body).xmat.reshape(3, 3)[1, 0],
            self.data.body(MARS.base_body).xmat.reshape(3, 3)[0, 0],
        )
        base_goal = np.r_[self.data.body(MARS.base_body).xpos[:2], yaw]
        move, error = plan_transport(self.model, self.data, executor.collision, None, None, base_target=base_goal)
        if error:
            return None, error
        return {"clearance_path": path, "base_goal": base_goal.tolist(), "base_trajectory": move["trajectory"]}, None

    def _approach(self, plan):
        from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
        from emet.controller.task.tamp.mars_transport import plan_transport

        result = plan.executor._execute_path("approach_clearance", plan.approach["clearance_path"])
        if not result.success:
            return result
        self.synchronize(self.data)
        move, error = plan_transport(
            self.model, self.data, plan.executor.collision, None, None, base_target=plan.approach["base_goal"]
        )
        if error:
            return PhysicalMotionResult(False, error, "approach")
        delta = np.asarray(plan.approach["base_goal"]) - self.robot.servo.base_pose()
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        result = (
            PhysicalMotionResult(True, "stationary_approach", "approach")
            if np.linalg.norm(delta[:2]) < 0.001 and abs(delta[2]) < 0.001
            else self.robot.follow_trajectory(move["trajectory"], loaded=False)
        )
        if result.success:
            self.synchronize(self.data)
            now = self.data.body(plan.object_body)
            translation = now.xpos - plan.object_initial_pose[:3, 3]
            angle = np.arccos(
                np.clip((np.trace(plan.object_initial_pose[:3, :3].T @ now.xmat.reshape(3, 3)) - 1) / 2, -1, 1)
            )
            if np.linalg.norm(translation) > 0.01 or angle > 0.05:
                return PhysicalMotionResult(False, "object_changed_replan", "approach")
            # Re-ground positions after the controller's own clearance motion.
            # Preserve the admitted world grasp orientation and frozen placement
            # goal; all translated targets are re-certified from measured state.
            plan.executor.grasp_targets = [
                (point + translation, rotation) for point, rotation in plan.executor.grasp_targets
            ]
            self.events.append({"phase": "grasp_reground", "translation_m": translation.tolist()})
        return result

    def _prepare_transport(self, executor, obj, goal):
        from emet.controller.task.tamp.mars_transport import plan_transport

        initial = self.data.qpos.copy()
        checker = executor.collision
        attachment = (
            checker.payload_body,
            checker.payload_parent,
            None if checker.payload_transform is None else checker.payload_transform.copy(),
        )
        base = self.data.body(MARS.base_body)
        yaw = np.arctan2(base.xmat.reshape(3, 3)[1, 0], base.xmat.reshape(3, 3)[0, 0])
        short = np.r_[base.xpos[:2] + 0.03 * np.array([np.cos(yaw), np.sin(yaw)]), yaw]
        accepted = False
        error = "no_feasible_transport"
        try:
            for strategy, target in (("short_straight_and_arm", short), ("full_base_transfer", None)):
                self.data.qpos[:] = initial
                checker.payload_body, checker.payload_parent, checker.payload_transform = attachment
                mujoco.mj_kinematics(self.model, self.data)
                move, error = plan_transport(
                    self.model, self.data, checker, self.model.body(obj).id, goal, base_target=target
                )
                if error:
                    self._record_rejection(strategy, error, executor)
                    continue
                endpoint = self.data.qpos.copy()
                rotation = self.data.body(executor.ee_body).xmat.reshape(3, 3).copy()
                targets = self._place_targets(executor, obj, goal, rotation)
                # Select base motion by the complete downstream task, rather
                # than forcing a lateral transfer when the arm can do the work.
                # These are private planning probes within the same IK budget.
                error = None
                for i, (point, orientation) in enumerate(targets):
                    if i == 2:
                        checker.set_payload(self.model, self.data, None)
                        _, error = executor.plan_gripper_open()
                        if error:
                            break
                    _, error = executor.plan_pose(point, orientation)
                    if error:
                        break
                if error:
                    self._record_rejection(strategy, error, executor)
                    continue
                self.data.qpos[:] = endpoint
                checker.payload_body, checker.payload_parent, checker.payload_transform = attachment
                mujoco.mj_kinematics(self.model, self.data)
                move.update(base_goal=move["trajectory"].base[-1].tolist(), strategy=strategy)
                accepted = True
                return move, None
            return None, error
        finally:
            checker.payload_body, checker.payload_parent, checker.payload_transform = attachment
            if not accepted:
                self.data.qpos[:] = initial
                mujoco.mj_kinematics(self.model, self.data)

    def _transport(self, plan):
        from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult

        executor = plan.executor
        self.synchronize(self.data)
        if not executor.payload_retained():
            return PhysicalMotionResult(False, "payload_not_retained", "transport")
        from emet.controller.task.tamp.mars_transport import plan_transport

        transport, error = plan_transport(
            self.model,
            self.data,
            executor.collision,
            self.model.body(plan.object_body).id,
            plan.goal,
            base_target=plan.transport["base_goal"],
        )
        if error:
            return PhysicalMotionResult(False, error, "transport")
        result = self.robot.follow_trajectory(transport["trajectory"])
        if result.success:
            self.synchronize(self.data)
            rotation = self.data.body(executor.ee_body).xmat.reshape(3, 3).copy()
            goal = plan.goal.copy()
            if plan.placement_region is not None:
                from emet.simulation.sim_object_placements import placements_from_mujoco_model

                entry = placements_from_mujoco_model(self.model, self.data)[plan.object_body]
                bounds = np.asarray(entry["bounds"])[:, :2]
                if np.all(bounds[0] >= plan.placement_region[0]) and np.all(bounds[1] <= plan.placement_region[1]):
                    # A region is the public contract. Preserve the achieved
                    # in-region XY and descend vertically instead of twisting
                    # a loaded grasp merely to hit the patch's exact center.
                    goal[:2] = entry["pos"][:2]
                    self.events.append(
                        {"phase": "placement_region_choice", "selected": goal.tolist(), "requested": plan.goal.tolist()}
                    )
            executor.place_targets = self._place_targets(executor, plan.object_body, goal, rotation)
        return result

    def plan(self, *args, **kwargs):
        if self.robot.failure:
            return self._failure("plan_pick_place", "controller_fault_requires_reset")
        result = super().plan(*args, **kwargs)
        if result["status"] == "ok":
            plan = self.plans[result["data"]["plan_ref"]]
            plan.object_initial_pose = np.eye(4)
            plan.object_initial_pose[:3, :3] = self.data.body(plan.object_body).xmat.reshape(3, 3)
            plan.object_initial_pose[:3, 3] = self.data.body(plan.object_body).xpos
            for target, path in zip(plan.executor.grasp_targets, plan.executor.grasp_paths, strict=True):
                plan.executor.remember_pose_seed(target[0], target[1], path[-1])
            transport = plan.transport
            if transport is not None:
                result["data"]["transport_duration_s"] = transport["trajectory"].end
                result["data"]["minimum_static_support_margin_m"] = transport["minimum_static_margin_m"]
                result["data"]["transport_strategy"] = transport["strategy"]
        return result

    def pose_references(self, plan_ref):
        """Read-only debug export; full schedules stay out of agent tool replies."""
        with self.lock:
            plan = self.plans.get(plan_ref)
            if plan is None:
                raise ValueError("unknown_plan")
            return {
                "transport": plan.transport["trajectory"].record(),
                "approach_base": plan.approach["base_trajectory"].record(),
                "arm_targets": [
                    {
                        "stage": stage,
                        "kind": "nominal_planning_endpoint; execution_recertifies_measured_state",
                        "base": plan.transport["trajectory"].base[0 if i < 3 else -1].tolist(),
                        "hand": np.r_[point, orientation[:, :2].ravel()].tolist(),
                        "gripper": MARS.gripper_open
                        if stage in {"pregrasp", "grasp", "retreat"}
                        else MARS.gripper_closed,
                        "base_mask": [True] * 3,
                        "hand_mask": True,
                    }
                    for i, (stage, (point, orientation)) in enumerate(
                        zip(
                            ("pregrasp", "grasp", "lift", "preplace", "place", "retreat"),
                            [*plan.executor.grasp_targets, *plan.executor.place_targets],
                            strict=True,
                        )
                    )
                ],
            }

    def _place_targets(self, executor, obj, goal, rotation):
        from emet.simulation.sim_object_placements import placements_from_mujoco_model

        placement = placements_from_mujoco_model(self.model, self.data)[obj]
        point = np.asarray(placement["pos"])
        joint = self.model.joint(MARS.joint_groups[0][0]).id
        origin = self.data.xanchor[joint].copy()
        axis = self.data.xaxis[joint].copy()
        start, end = point - origin, goal - origin
        start -= axis * (axis @ start)
        end -= axis * (axis @ end)
        yaw = np.arctan2(axis @ np.cross(start, end), start @ end)
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        delta = np.eye(3) + np.sin(yaw) * skew + (1 - np.cos(yaw)) * (skew @ skew)
        target_rotation = delta @ rotation
        offset = delta @ (point - self.data.body(executor.ee_body).xpos)
        # Surface candidates include 20 mm release clearance. Require physical
        # support before release, and preserve the held transform while turning.
        release = goal - [0, 0, 0.02] - offset
        withdraw = release - target_rotation[:, 0] * 0.03 + [0, 0, 0.06]
        return [(release + [0, 0, 0.06], target_rotation), (release, target_rotation), (withdraw, target_rotation)]

    def execute(self, plan_ref):
        with self.lock:
            plan = self.plans.get(plan_ref)
            if plan is not None:
                self.robot.placement_support = plan.support_body
            self._executing = True
            self.execution_ik_calls = 0
            try:
                result = super().execute(plan_ref)
                if result["status"] != "ok" and result["data"].get("execution_started"):
                    self.robot.stop_and_confirm(result["code"])
                    result["data"]["stop_confirmed"] = self.robot.stop_confirmed
                return result
            except Exception:
                self.robot.stop_and_confirm("execution_exception")
                raise
            finally:
                self._executing = False

    def _verify(self, plan, *, released, seconds):
        if released and not self.robot.finish_and_hold():
            return False
        return super()._verify(plan, released=released, seconds=seconds) and not self.robot.failure
