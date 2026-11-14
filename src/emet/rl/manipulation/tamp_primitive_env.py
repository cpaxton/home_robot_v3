"""Step actual public-tool TAMP pregrasp execution in a single-owner child process.

The planner/executor owns task logic, closure and stop confirmation. The actor
gets only copied proprioception, targets and reference twist. Planning failures
remain explicit. This is a primitive environment, never a full pick/place claim.
"""

import multiprocessing as mp
import traceback

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.whole_body_observation import PRIVILEGED_WIDTH


class _Boundary(BaseException):
    def __init__(self, result):
        self.result = result


class _Closed(BaseException):
    pass


def _worker(connection, case, oracle, arm_reference, initial_settle_s):
    import json

    from emet.agent.tools import get_tools
    from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
    from emet.controller.task.tamp.native_adapter import adapter_for
    from emet.eval.mars_tamp_control import fixture
    from emet.rl.manipulation.command_runtime import TaskPoseRuntimeTiming, task_pose_teacher_action
    from emet.rl.manipulation.whole_body_observation import actor_observation, privileged_state
    from emet.rl.manipulation.whole_body_reward import RewardMeter

    env = robot = None
    try:
        env, admission = fixture(case)
        if not admission["accepted"]:
            raise ValueError("fixture_admission:" + str(admission.get("reason")))
        env.control.retention_context_enabled = True
        env.control.history = None
        env.control.base_velocity_feedback_gain = 0.5
        env.control.turn_progress_guard_enabled = True

        class Bridge:
            is_reference_controller = bool(oracle)
            controller_name = "training_oracle" if oracle else "training_actor"

            def __init__(self):
                self.pending, self.ticks = False, 0
                self.reward = np.zeros(3, dtype=np.float32)
                self.meter = RewardMeter(env)
                self.start = float(env.data.time)
                self.motion = np.zeros(9, dtype=np.float32)

            def observation(self):
                return actor_observation(env.control) | {
                    "critic": privileged_state(env, float(env.data.time) - self.start, 30.0)
                }

            def __call__(self, obs):
                from emet.rl.manipulation.whole_body_observation import reference_velocity

                observation = actor_observation(env.control, obs) | {
                    "critic": privileged_state(env, float(env.data.time) - self.start, 30.0)
                }
                if not self.pending:
                    self.meter = RewardMeter(env)
                info = {
                    "case_id": case["id"],
                    "reward_terms": self.meter.last_terms.copy(),
                    "elapsed_control_ticks": max(1, self.ticks),
                    "primitive": "pregrasp",
                    "simulation_only": True,
                    "actor_action_applied": self.pending,
                }
                if oracle:
                    info["oracle_action"] = task_pose_teacher_action(env.control).astype(np.float32)
                connection.send(("decision", observation, self.reward.copy(), info))
                if not connection.poll(60):
                    raise TimeoutError("training learner did not answer within60s")
                message, action = connection.recv()
                if message == "close":
                    raise _Closed()
                if message != "action":
                    raise ValueError("unexpected training IPC command")
                action = np.asarray(action, dtype=float)
                if action.shape != (8,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
                    raise ValueError("invalid training actor action")
                self.reward[:] = 0
                self.ticks = 0
                self.pending = True
                self.motion = reference_velocity(env.control)
                return action

            def after_step(self):
                if self.pending:
                    motion = np.zeros(9) if env.control.holding or env.control.stopped else self.motion
                    self.reward += self.meter.tick(env.control.last_action.copy(), motion)
                    self.ticks += 1

        bridge = Bridge()
        # External PPO updates pause this simulator. These explicit wall budgets
        # are training IPC budgets, never hardware/realtime qualification limits.
        robot = MarsSimRobot(
            env,
            policy=bridge,
            timing=TaskPoseRuntimeTiming(60.0, 61.0),
            arm_execution="cartesian",
            arm_reference=arm_reference,
        )
        robot.step_observer = bridge.after_step
        # Development reset conditioning: physical settling before planning,
        # without a state teleport, grip closure or learned-policy inference.
        # Frozen public acceptance fixtures keep their original initialization.
        if initial_settle_s:
            robot.step(initial_settle_s)
            if robot.failure:
                raise ValueError("initial_settling:" + robot.failure)
        context = {
            "robot": robot,
            "manip_mode": "physical",
            "tamp_inputs": "privileged",
            "_native_tamp_model": env.model,
        }
        tools = {tool.name: tool for tool in get_tools(context)}
        discovery = json.loads(tools["scene_tasks"].func(object_filter="blue"))
        task = next(
            t for t in discovery.get("data", {}).get("tasks", []) if t["receptacle_name"] == "destination support"
        )
        planned = json.loads(
            tools["plan_pick_place"].func(task_ref=task["task_ref"], placement_region=admission["placement_region"])
        )
        if planned["status"] != "ok":
            raise ValueError("public_plan:" + str(planned.get("code")))
        adapter = adapter_for(context, robot)
        plan = adapter.plans[planned["data"]["plan_ref"]]
        object_initial = env.data.xpos[env.payload].copy()
        rotation_initial = env.data.xmat[env.payload].reshape(3, 3).copy()
        original = plan.executor._execute_path
        approach = adapter._approach
        approach_failure = None

        def record_approach(active_plan):
            nonlocal approach_failure
            result = approach(active_plan)
            if not result.success:
                approach_failure = result.message
            return result

        adapter._approach = record_approach

        def primitive(phase, path):
            result = original(phase, path)
            if phase == "pregrasp":
                raise _Boundary(result)
            return result

        plan.executor._execute_path = primitive
        success, reason, executor_result = False, "primitive_not_reached", None
        with robot.stepped_clock():
            try:
                result = json.loads(tools["execute_pick_place_plan"].func(plan_ref=planned["data"]["plan_ref"]))
                executor_result = result
                reason = approach_failure or robot.failure or result.get("code", reason)
            except _Boundary as boundary:
                success, reason = bool(boundary.result.success), boundary.result.message
                if success:
                    success = robot._verify_quiet_hold(2.0, 3.5)
                    reason = "pregrasp_complete" if success else "terminal_motion_unsettled"
        robot.stop_and_confirm("training_primitive_complete" if success else reason)
        success &= robot.stop_confirmed is True
        bridge.reward[0] += 2.0 if success else -10.0
        info = {
            "case_id": case["id"],
            "primitive_success": bool(success),
            "code": reason,
            "stop_confirmed": robot.stop_confirmed,
            "elapsed_control_ticks": max(1, bridge.ticks),
            "actor_action_applied": bridge.pending,
            "reward_terms": bridge.meter.last_terms.copy(),
            "initialization_failed": not bridge.pending,
            "full_pick_place_success": False,
            "policy_calls": robot.inference_counts,
            "commands": robot.commands,
            "guard_failure": robot.failure if not success else None,
            "wheel_lift": bool(env.monitor.wheel_lift),
            "invalid_physics": bool(env._physics_invalid),
            "unsafe_contact": bool(env.monitor.peak_force > env.profile.unsafe_force),
            "retention_lost": bool(getattr(env, "retention_lost", False)),
            "executor_result": executor_result,
            "initial_settle_s": initial_settle_s,
            "object_translation_m": (env.data.xpos[env.payload].copy() - object_initial).tolist(),
            "object_rotation_rad": float(
                np.arccos(
                    np.clip(
                        (np.trace(rotation_initial.T @ env.data.xmat[env.payload].reshape(3, 3).copy()) - 1) / 2, -1, 1
                    )
                )
            ),
        }
        connection.send(("terminal", bridge.observation(), bridge.reward.copy(), info))
    except _Closed:
        if robot is not None:
            robot.stop_and_confirm("training_client_closed")
    except BaseException as exc:
        if robot is not None:
            robot.stop_and_confirm("training_worker_failed")
        try:
            connection.send(("error", str(exc), traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class TampPrimitiveEnv(gym.Env):
    def __init__(self, cases, *, oracle=False, arm_reference="s_curve", timeout=90.0, initial_settle_s=0.0):
        if not cases or arm_reference not in {"s_curve", "jerk_limited"} or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("nonempty cases and a bounded simulator training configuration required")
        if not np.isfinite(initial_settle_s) or not 0 <= initial_settle_s <= 2.0:
            raise ValueError("development settling must be bounded in [0,2] seconds")
        self.cases, self.oracle, self.arm_reference, self.timeout = list(cases), bool(oracle), arm_reference, timeout
        self.initial_settle_s = float(initial_settle_s)
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(-np.inf, np.inf, (20, 101), np.float32),
                "preview": spaces.Box(-np.inf, np.inf, (7, 18), np.float32),
                "motion": spaces.Box(-np.inf, np.inf, (9,), np.float32),
                "critic": spaces.Box(-np.inf, np.inf, (PRIVILEGED_WIDTH,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1, 1, (MARS.action_size,), np.float32)
        self.process = self.connection = None
        self.cursor, self.closed_episode = 0, True

    def _receive(self):
        if not self.connection.poll(self.timeout):
            self.close()
            raise TimeoutError("TAMP training worker timed out")
        message = self.connection.recv()
        if message[0] == "error":
            self.close()
            raise RuntimeError("TAMP training worker: " + message[1] + "\n" + message[2])
        return message

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        if options is not None and (
            set(options) != {"case_id"} or options["case_id"] not in {c["id"] for c in self.cases}
        ):
            raise ValueError("unknown TAMP development case")
        case = (
            next(c for c in self.cases if c["id"] == options["case_id"])
            if options
            else self.cases[self.cursor % len(self.cases)]
        )
        self.cursor += 1
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_worker, args=(child, case, self.oracle, self.arm_reference, self.initial_settle_s), daemon=True
        )
        self.process.start()
        child.close()
        message = self._receive()
        self.closed_episode = message[0] == "terminal"
        self.pending_terminal = message if self.closed_episode else None
        self.last = message[1]
        return self.last, message[3]

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("finite normalized Mars action required")
        if self.pending_terminal is not None:
            message, self.pending_terminal = self.pending_terminal, None
        else:
            if self.closed_episode:
                raise RuntimeError("reset the TAMP primitive before stepping")
            self.connection.send(("action", np.clip(action, -1, 1)))
            message = self._receive()
        self.closed_episode = message[0] == "terminal"
        self.last, rewards, info = message[1:]
        info["reward_groups"] = np.asarray(rewards, dtype=np.float32)
        return self.last, float(np.sum(rewards)), self.closed_episode, False, info

    def close(self):
        if self.connection is not None:
            try:
                self.connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        if self.process is not None:
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None
