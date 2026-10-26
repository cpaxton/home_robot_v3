"""Synchronous, seeded Mars reaching environment. No transport or realtime sleeps."""

from dataclasses import asdict, dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from emet.rl.contact import solved_contact_force
from emet.rl.control import Goal, MarsControl, baseline_action, invalid_physics
from emet.rl.model import PROXIES, build_model
from emet.rl.profile import MARS


@dataclass(frozen=True)
class EnvConfig:
    task: str = "reach"
    pose_targets: bool = False
    scene: str = "procedural"
    stage: int = 0
    horizon_seconds: float = 10.0
    obstacle_count: int = 3
    contact_training: bool = False
    unsafe_contact_force: float = 30.0  # Experimental simulator threshold, N.
    base_scenario: str = "standard"
    terminal_accuracy_weight: float = 0.0
    arm_only: bool = False
    trajectory_waypoints: int = 1

    def __post_init__(self):
        if self.arm_only and (self.task != "reach" or self.stage != 0):
            raise ValueError("arm_only currently requires reach task at stage 0")
        if not np.isfinite(self.terminal_accuracy_weight) or self.terminal_accuracy_weight < 0:
            raise ValueError("terminal accuracy weight must be finite and nonnegative")
        if self.base_scenario not in {"standard", "mixed", "retarget", "push", "contact_retreat"}:
            raise ValueError("invalid base scenario")
        if self.base_scenario != "standard" and (self.task != "base" or self.stage != 0 or self.scene != "procedural"):
            raise ValueError("base scenarios require base task, procedural scene and stage 0")
        if self.task not in {"base", "reach", "coordinated"}:
            raise ValueError("task must be base, reach, or coordinated")
        if self.pose_targets and self.task != "coordinated":
            raise ValueError("pose_targets requires the coordinated task")
        if self.scene not in {"procedural", "table"} or self.stage not in range(5):
            raise ValueError("scene must be procedural/table and stage must be 0..4")
        if not np.isfinite(self.horizon_seconds) or self.horizon_seconds <= 0 or self.obstacle_count < 0:
            raise ValueError("invalid horizon or obstacle count")
        if (
            isinstance(self.trajectory_waypoints, bool)
            or not isinstance(self.trajectory_waypoints, int)
            or not 1 <= self.trajectory_waypoints <= 20
            or (self.trajectory_waypoints > 1 and self.task != "coordinated")
        ):
            raise ValueError("trajectory_waypoints must be 1..20 and requires coordinated task")
        if not np.isfinite(self.unsafe_contact_force) or self.unsafe_contact_force <= 0:
            raise ValueError("unsafe contact force must be finite and positive")


class MarsReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config=None, render_mode=None):
        self.config = config if isinstance(config, EnvConfig) else EnvConfig(**(config or {}))
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.render_mode = render_mode
        obstacles = self.config.obstacle_count if self.config.stage >= 3 else 0
        if self.config.base_scenario in {"mixed", "contact_retreat"}:
            obstacles = max(obstacles, 1)
        self.model = build_model(self.config.scene, obstacles)
        # Policy uses geometry features, not cameras/rangefinders. Avoid 360 unused
        # sensor raycasts on every training physics step.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.control = MarsControl(self.model, self.data)
        self.action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(
                    -np.inf, np.inf, shape=(MARS.history_steps, 66 + len(PROXIES) * 5), dtype=np.float32
                ),
                "poses": spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32),
                "state": spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32),
                "goal": spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32),
                "obstacles": spaces.Box(-np.inf, np.inf, shape=(len(PROXIES) * 5,), dtype=np.float32),
                **(
                    {
                        "target_tokens": spaces.Box(-np.inf, np.inf, shape=(2, 10), dtype=np.float32),
                        "target_mask": spaces.MultiBinary(2),
                    }
                    if self.config.pose_targets
                    else {}
                ),
            }
        )
        self.substeps = round(MARS.control_dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, MARS.control_dt):
            raise ValueError("control period must be an integer number of physics steps")
        self._renderer = None
        self._needs_reset = True

    def _sample_configuration(self):
        c = self.control
        # Sample close enough to complete under the conservative velocity limits.
        radius = 0.15 if self.config.stage == 0 else 0.5
        for _ in range(300):
            self.data.qpos[c.base_q] = [*self.np_random.uniform(-radius, radius, 2), self.np_random.uniform(-0.5, 0.5)]
            if self.config.arm_only:
                self.data.qpos[c.base_q] = 0
                self.data.qpos[c.arm_q] = self.np_random.uniform(c.limits[:, 0] + 0.08, c.limits[:, 1] - 0.08)
            else:
                self.data.qpos[c.arm_q] = self.np_random.uniform(-0.25, 0.25, 5)
                self.data.qpos[c.arm_q[1]] -= 0.4
            mujoco.mj_forward(self.model, self.data)
            if (
                c.obstacle_features()[1] > 0.0005
                and self._arm_floor_clearance() > 0.0005
                and not self._unsafe_robot_contact()
            ):
                return c.base_pose(), c.arm_positions().copy()
        raise RuntimeError("could not sample a collision-free Mars configuration")

    def _arm_floor_clearance(self):
        """Conservative clearance of active arm collision geometry above floor."""
        model, data = self.model, self.data
        geom_xpos = np.asarray(data.geom_xpos).copy()
        planes = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
        if not planes:
            return float("inf")
        floor_z = min(float(geom_xpos[i, 2]) for i in planes)
        base = self.control.base_body
        arm_geoms = []
        for geom, body in enumerate(model.geom_bodyid):
            body = int(body)
            if body == 0 or body == base or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            ancestor = body
            while ancestor and ancestor != base:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == base:
                arm_geoms.append(geom)
        if not arm_geoms:
            return float("inf")
        return float(np.min(geom_xpos[arm_geoms, 2] - model.geom_rbound[arm_geoms] - floor_z))

    def _unsafe_robot_contact(self):
        """Detect penetrations except the intended mobile-base floor support."""
        data, model = self.data, self.model
        count = min(int(data.ncon), len(data.contact))
        if count == 0:
            return False
        contacts = data.contact
        geom1 = contacts.geom1[:count].copy()
        geom2 = contacts.geom2[:count].copy()
        distance = contacts.dist[:count].copy()
        robot = set(self.control.robot_geoms)
        base = model.geom("rl_base_link").id
        floors = {
            geom for geom in range(model.ngeom) if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE
        }
        for a, b, dist in zip(geom1, geom2, distance, strict=True):
            a, b = int(a), int(b)
            if not (0 <= a < model.ngeom and 0 <= b < model.ngeom):
                continue
            if (a == base and b in floors) or (b == base and a in floors):
                continue
            if dist < -0.0005 and (a in robot or b in robot):
                return True
        return False

    def _sample_goal(self):
        if self.config.task == "coordinated":
            return self._sample_coordinated_goal()
        qpos = self.data.qpos.copy()
        try:
            for _ in range(60):
                base, arm = self._sample_configuration()
                if self.config.arm_only and self.np_random.random() < 0.5:
                    arm = np.clip(
                        qpos[self.control.arm_q] + self.np_random.uniform(-0.25, 0.25, 5),
                        self.control.limits[:, 0] + 0.08,
                        self.control.limits[:, 1] - 0.08,
                    )
                if not self.config.arm_only:
                    break
                # Initial curriculum: reject paths with sampled geometric contact.
                # Discrete checks are not a continuous swept-volume certificate.
                start = qpos[self.control.arm_q]
                count = max(2, int(np.ceil(np.max(np.abs(arm - start)) / 0.04)) + 1)
                clear = True
                for fraction in np.linspace(0, 1, count):
                    self.data.qpos[self.control.arm_q] = start + fraction * (arm - start)
                    mujoco.mj_forward(self.model, self.data)
                    if (
                        self.control.obstacle_features()[1] <= 0.0005
                        or self._arm_floor_clearance() <= 0.0005
                        or self._unsafe_robot_contact()
                    ):
                        clear = False
                        break
                if clear:
                    break
            else:
                raise RuntimeError("could not sample a collision-free arm interpolation")
        finally:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
        mask = np.ones(6)
        if self.config.arm_only:
            mask[0] = 0
            return Goal(self.control.base_pose(), arm, mask)
        # Train masked modes too, holding the omitted dimensions at the start.
        kind = self.np_random.choice(3, p=[0.15, 0.15, 0.70])
        if kind == 0:
            mask[0], base = 0, self.control.base_pose()
        elif kind == 1:
            mask[1:], arm = 0, self.control.arm_positions().copy()
        return Goal(base, arm, mask)

    def _sample_coordinated_goal(self):
        """Sample a nearby SE(3) target from FK and resolve it with arm IK."""
        c = self.control
        start_base = c.base_pose().copy()
        start_arm = c.arm_positions().copy()
        lower, upper = c.limits[:, 0] + 0.04, c.limits[:, 1] - 0.04
        for _ in range(100):
            kind = int(self.np_random.choice(3, p=[0.15, 0.15, 0.70]))
            base_active, hand_active = kind in (0, 2), kind in (1, 2)
            target_base = start_base.copy()
            if base_active:
                target_base += self.np_random.uniform([-0.06, -0.06, -0.16], [0.06, 0.06, 0.16])
                target_base[2] = np.arctan2(np.sin(target_base[2]), np.cos(target_base[2]))
            target_arm = start_arm.copy()
            if hand_active:
                target_arm = np.clip(
                    start_arm + self.np_random.uniform(-0.12, 0.12, 5), lower, upper
                )

            scratch = c._goal_data
            scratch.qpos[:] = self.data.qpos
            axes = self.data.xaxis[c.base_ids]
            delta = target_base[:2] - start_base[:2]
            scratch.qpos[c.base_q[:2]] += np.linalg.solve(axes[:2, :2].T, delta)
            scratch.qpos[c.base_q[2]] += (
                np.arctan2(np.sin(target_base[2] - start_base[2]), np.cos(target_base[2] - start_base[2]))
                / axes[2, 2]
            )
            scratch.qpos[c.arm_q] = target_arm
            mujoco.mj_forward(self.model, scratch)
            rotation = scratch.xmat[c.ee_body].reshape(3, 3)
            target_pose = np.r_[scratch.xpos[c.ee_body], rotation[:, :2].ravel()]

            if hand_active:
                try:
                    resolved_base, resolved_arm = c.solve_ee_pose(
                        target_pose, base=target_base, base_locked=True
                    )
                except ValueError:
                    continue
                if np.max(np.abs(resolved_arm - start_arm)) > 0.25:
                    continue
                target_base, target_arm = resolved_base, resolved_arm

            mask = np.r_[base_active, np.full(5, hand_active)].astype(float)
            return Goal(target_base, target_arm, mask)
        raise RuntimeError("could not sample a nearby, IK-reachable coordinated pose goal")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.control.reset()
        self.steps = 0
        self._needs_reset = False
        for i in range(self.model.nmocap):
            self.data.mocap_pos[i] = [*self.np_random.uniform(-0.8, 0.8, 2), 0.20]
        if self.config.base_scenario != "standard":
            # The base scenario positions its own obstacle after sampling the
            # start. A random box can otherwise cover the entire small start
            # region and make rejection sampling fail on a later reset.
            self.data.mocap_pos[:] = [5, 5, 0.2]
        self._obstacle_origin = self.data.mocap_pos.copy()
        self.sampling_rejected_starts = 0
        for _ in range(30):
            self._sample_configuration()
            self.control.reset()
            try:
                goal = self._sample_goal()
            except RuntimeError:
                if not self.config.arm_only:
                    raise
                self.sampling_rejected_starts += 1
                continue
            if options and "goal" in options:
                goal = Goal.parse(options["goal"], self.control)
            self._trajectory_goals = self._make_trajectory_goals(goal)
            self._trajectory_index = 0
            self.control.set_goal(self._trajectory_goals[0])
            break
        else:
            raise RuntimeError("could not sample connected arm start/goal pair")
        mujoco.mj_forward(self.model, self.data)
        obs, info = self.control.observe()
        if self.config.pose_targets:
            obs = self._with_pose_targets(obs)
        self._previous_cost = self._cost(info)
        return obs, {
            **info,
            "is_success": False,
            "config": asdict(self.config),
            "sampling_rejected_starts": self.sampling_rejected_starts,
            "trajectory_waypoints": self.config.trajectory_waypoints,
        }

    def _make_trajectory_goals(self, final):
        count = self.config.trajectory_waypoints
        if count == 1:
            return [final]
        from emet.rl.waypoints import coordinated_waypoints

        return coordinated_waypoints(self.control.base_pose(), self.control.arm_positions(), final, count)

    def set_goal(self, payload):
        """Explicit goal change; reward is rebased so switches cannot create progress."""
        self.control.set_goal(payload)
        self._previous_cost = self._cost(self.control.observe()[1])

    def expert_action(self):
        return baseline_action(self.control)

    def _with_pose_targets(self, observation):
        """Expose masked base/hand SE(2)/SE(3) goal tokens to coordinated actors."""
        from emet.rl.pose_targets import PoseTargets

        goal = self.control.goal
        pose = self.control.goal_pose
        targets = PoseTargets(
            base=goal.base,
            base_mask=np.full(3, bool(goal.mask[0]), dtype=bool),
            hands=pose[None, :9],
            hand_mask=np.array([bool(goal.mask[1:].any())], dtype=bool),
        )
        tokens, mask = targets.actor_tokens()
        return {**observation, "target_tokens": tokens, "target_mask": mask.astype(np.int8)}

    @staticmethod
    def _cost(metrics):
        return (
            2 * metrics["xy_error"]
            + 0.3 * metrics["yaw_error"]
            + float(metrics.get("hand_active", True)) * (
                5 * metrics["ee_position_error"] + 0.5 * metrics["ee_orientation_error"]
            )
            + 0.1 * metrics["joint_error"]
        )

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset required before step")
        previous_action = self.control.last_action.copy()
        if self.config.arm_only:
            action = np.asarray(action).copy()
            action[:2] = 0
        self.control.accept_action(action)
        goal_changed = self.config.stage >= 2 and self.steps == 60
        if goal_changed:
            self.set_goal(self._sample_goal())
        if self.config.stage >= 2 and self.steps == 100:
            self.data.qvel[self.control.base_v[:2]] += self.np_random.uniform(-0.15, 0.15, 2)
        safe_qpos = self.data.qpos.copy()
        safe_qvel = self.data.qvel.copy()
        safe_time = float(self.data.time)
        goal = self.control.goal
        safe_goal = Goal(goal.base.copy(), goal.arm.copy(), goal.mask.copy())
        collision = False
        peak_contact_force = 0.0
        contact_impulse = 0.0
        invalid = False
        for _ in range(self.substeps):
            if self.config.stage >= 4:
                self.data.mocap_pos[:, 1] = self._obstacle_origin[:, 1] + 0.12 * np.sin(2 * self.data.time)
            self.control.apply()
            mujoco.mj_step(self.model, self.data)
            if invalid_physics(self.data):
                invalid = True
                break
            # Read the forces from the solved step, before recomputing forward dynamics.
            force_sum = 0.0
            contact_view = self.data.contact
            contact_count = min(int(self.data.ncon), len(contact_view))
            invalid |= contact_count != int(self.data.ncon)
            if invalid:
                break
            geom1 = contact_view.geom1[:contact_count].copy()
            geom2 = contact_view.geom2[:contact_count].copy()
            distances = contact_view.dist[:contact_count].copy()
            efc_addresses = contact_view.efc_address[:contact_count].copy()
            dimensions = contact_view.dim[:contact_count].copy()
            excludes = contact_view.exclude[:contact_count].copy()
            efc_force = np.asarray(self.data.efc_force).copy()
            for index in range(contact_count):
                a, b = int(geom1[index]), int(geom2[index])
                if not (0 <= a < self.model.ngeom and 0 <= b < self.model.ngeom):
                    invalid = True
                    break
                if a not in self.control.robot_geoms and b not in self.control.robot_geoms:
                    continue
                # The mobile base's support contact is not an obstacle impact.
                geoms = (a, b)
                if self.model.geom("rl_base_link").id in geoms and any(
                    self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in geoms
                ):
                    continue
                _, force = solved_contact_force(
                    self.model,
                    self.data,
                    index,
                    efc_address=efc_addresses[index],
                    dim=dimensions[index],
                    exclude=excludes[index],
                    efc_force=efc_force,
                )
                force_sum += force
            peak_contact_force = max(peak_contact_force, force_sum)
            contact_impulse += force_sum * self.model.opt.timestep
            mujoco.mj_forward(self.model, self.data)
            invalid |= invalid_physics(self.data)
            if invalid:
                break
            # Latch contacts across substeps, even if the solver separates them.
            robot = self.control.robot_geoms
            contact_view = self.data.contact
            contact_count = min(int(self.data.ncon), len(contact_view))
            geom1 = contact_view.geom1[:contact_count].copy()
            geom2 = contact_view.geom2[:contact_count].copy()
            distances = contact_view.dist[:contact_count].copy()
            for index in range(contact_count):
                a, b = int(geom1[index]), int(geom2[index])
                if distances[index] < -0.001 and (a in robot or b in robot):
                    collision = True
                    break
            if invalid or peak_contact_force > self.config.unsafe_contact_force:
                break
        if invalid:
            # Discard suspect native buffers and report the failed action from
            # the last valid state. Never query geometry on an invalid state.
            self.data = mujoco.MjData(self.model)
            self.data.qpos[:] = safe_qpos
            self.data.qvel[:] = safe_qvel
            self.data.time = safe_time
            mujoco.mj_forward(self.model, self.data)
            self.control = type(self.control)(self.model, self.data)
            self.control.set_goal(safe_goal)
            self.control.stop()
        else:
            mujoco.mj_forward(self.model, self.data)
        obs, metrics = self.control.observe()
        if self.config.pose_targets:
            obs = self._with_pose_targets(obs)
        collision |= metrics["collision"]
        unsafe_contact = peak_contact_force > self.config.unsafe_contact_force
        contact_failure = unsafe_contact or (collision and not self.config.contact_training)
        if contact_failure or invalid:
            self.control.stop()
            obs, metrics = self.control.observe()
            if self.config.pose_targets:
                obs = self._with_pose_targets(obs)
        settled = self.control.settled(metrics, MARS.control_dt)
        # Recovery scenarios must actually encounter their scheduled events.
        success = (
            settled
            and not contact_failure
            and not invalid
            and (self.config.stage < 2 or self.steps >= 100)
            and getattr(self, "_scenario_ready", True)
        )
        waypoint_advanced = False
        if success and self._trajectory_index < len(self._trajectory_goals) - 1:
            self._trajectory_index += 1
            self.control.set_goal(self._trajectory_goals[self._trajectory_index])
            obs, metrics = self.control.observe()
            if self.config.pose_targets:
                obs = self._with_pose_targets(obs)
            self._previous_cost = self._cost(metrics)
            cost = self._previous_cost
            success = False
            waypoint_advanced = True
        cost = self._cost(metrics)
        horizon = self.config.horizon_seconds
        if self.config.base_scenario == "mixed" and getattr(self, "_scenario", "standard") == "standard":
            horizon = min(horizon, 10.0)
        terminal = success or (self.steps + 1) * MARS.control_dt >= horizon
        terminal_accuracy_penalty = (
            self.config.terminal_accuracy_weight * cost
            if terminal and not contact_failure and not invalid
            else 0.0
        )
        smoothness = float(np.sum((self.control.last_action - previous_action) ** 2))
        joint_margin = np.minimum(
            self.control.arm_positions() - self.control.limits[:, 0],
            self.control.limits[:, 1] - self.control.arm_positions(),
        )
        reward = (
            10 * (self._previous_cost - cost)
            - 0.05 * cost
            - 0.002 * smoothness
            - 0.001 * float(np.sum(self.control.last_action**2))
            - 0.01 * float(np.sum(np.maximum(0, 0.1 - joint_margin)))
            + 10 * success
            + 2 * waypoint_advanced
            - (0.0 if self.config.contact_training else 10 * collision)
            - 0.02 * peak_contact_force
            - 0.2 * contact_impulse
            - 10 * unsafe_contact
            - 0.1 * metrics["obstruction"]
            - 10 * invalid
            - terminal_accuracy_penalty
        )
        self._previous_cost = cost
        self.steps += 1
        terminated = bool(success or contact_failure or invalid)
        truncated = bool(self.steps * MARS.control_dt >= horizon and not terminated)
        self._needs_reset = terminated or truncated
        return (
            obs,
            float(reward),
            terminated,
            truncated,
            {
                **metrics,
                "collision": bool(collision),
                "peak_contact_force": peak_contact_force,
                "contact_impulse": contact_impulse,
                "unsafe_contact": bool(unsafe_contact),
                "is_success": bool(success),
                "invalid_physics": bool(invalid),
                "goal_changed": goal_changed,
                "waypoint_advanced": waypoint_advanced,
                "trajectory_waypoint": self._trajectory_index,
                "trajectory_waypoints": len(self._trajectory_goals),
                "terminal_accuracy_penalty": float(terminal_accuracy_penalty),
                "action_change": smoothness,
                "joint_limit_violation": bool(np.any(joint_margin < -0.001)),
                "time": float(self.data.time),
            },
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
