"""Unannounced, nominally anchored pose handoffs for general tracking training.

Only reset-time FK constructs references. Neither stage uses the executing
controller's tracking error to choose targets. Runtime preserves history,
measured physics, the reward meter and every native fault/actuator guard.
"""

import copy
from numbers import Real

import numpy as np

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledReference, CoupledTrackingEnv, admit_reference
from emet.rl.manipulation.profiles import PROFILES


def load_curriculum(path):
    """Validate both splits and sampling shares without constructing physics."""
    from emet.rl.manipulation.coupled_tracking import load_suite

    suite = load_suite(path)
    if suite.get("curriculum_format") != "nominal-goal-change-v1" or suite.get("curriculum_robot") != "innate_mars":
        raise ValueError("expected the Mars nominal goal-change curriculum")
    cases = suite["robots"]["innate_mars"]
    changes = suite.get("goal_changes")
    if not isinstance(changes, dict) or set(changes) != {case["id"] for case in cases}:
        raise ValueError("one goal change per declared curriculum case required")
    if any(case["mode"] not in {"coupled", "hand_free_base"} or "retarget" in case for case in cases):
        raise ValueError("curriculum requires nominal hand-active changes")
    for change in changes.values():
        validate_change(change, PROFILES["innate_mars"].arm_size)
    expected = {"general": 0.5, "tamp": 0.5, "within_general": {"hand": 0.6, "base": 0.2, "goal_changes": 0.2}}
    if suite.get("training_mix") != expected:
        raise ValueError("goal-change v1 requires the declared 30/10/10/50 tick mixture")
    return suite


def validate_change(change, arm_size, *, arm_delta_limit=0.1):
    if isinstance(arm_delta_limit, bool) or not np.isfinite(arm_delta_limit) or not 0 < arm_delta_limit <= 0.4:
        raise ValueError("bounded explicit arm change limit required")
    if not isinstance(change, dict) or set(change) != {"travel_m", "turn_rad", "arm_delta", "dwell_s"}:
        raise ValueError("complete nominal goal-change specification required")
    try:
        flat = [change["travel_m"], change["turn_rad"], *change["arm_delta"], change["dwell_s"]]
    except TypeError as exc:
        raise ValueError("finite numeric goal-change geometry required") from exc
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in flat):
        raise ValueError("finite numeric goal-change geometry required")
    values = np.asarray(flat, dtype=float)
    if (
        values.shape != (arm_size + 3,)
        or not np.isfinite(values).all()
        or abs(values[0]) > 0.2
        or abs(values[1]) > 0.5
        or np.any(abs(values[2:-1]) > arm_delta_limit)
        or not 0 <= values[-1] < 2
    ):
        raise ValueError("finite bounded goal-change geometry and dwell in [0,2) required")
    return {
        "travel_m": float(values[0]),
        "turn_rad": float(values[1]),
        "arm_delta": values[2:-1].tolist(),
        "dwell_s": float(values[-1]),
    }


def nominal_successor(control, first, case, change):
    """Read a scratch control view; never call initialize/apply on live state."""
    base, joints = first.coordinates(first.end)
    first.fk(base, joints)
    view = copy.copy(control)
    view.data = first.fk.data
    successor = CoupledReference(
        view,
        {**case, "travel_m": change["travel_m"], "turn_rad": change["turn_rad"], "arm_delta": change["arm_delta"]},
        timing=first.timing,
    )
    before, after = first.sample(first.end), successor.sample(0)
    base_difference = np.asarray(before[0]) - after[0]
    base_difference[2] = np.arctan2(np.sin(base_difference[2]), np.cos(base_difference[2]))
    if not np.allclose(base_difference, 0, atol=1e-9, rtol=0):
        raise ValueError("nominal base handoff is discontinuous")
    for a, b in zip(before[1:], after[1:], strict=True):
        if not np.allclose(a, b, atol=1e-9, rtol=0):
            raise ValueError("nominal hand/gripper handoff is discontinuous")
    for a, b in zip(first.path.sample(first.end)[1:3], successor.path.sample(0)[1:3], strict=True):
        if not np.allclose(a, b, atol=1e-8, rtol=0):
            raise ValueError("nominal handoff derivatives are discontinuous")
    return successor


class GoalChangeTrackingEnv(CoupledTrackingEnv):
    """Two valid pose paths; announce the second before the first quiet hold.

    This trains task-agnostic command replacement, not grasp/lift semantics. The
    second command is installed after a completed control tick, so the returned
    observation contains its goals before the actor chooses the next action.
    """

    def __init__(
        self,
        robot,
        cases,
        *,
        changes,
        suite_path=COARSE_SUITE,
        reward_config=None,
        timing="legacy_v1",
        finish_after_hold=False,
        spread_requirements=None,
        change_arm_limit=0.1,
        settling_budget_s=3.0,
    ):
        if robot not in PROFILES:
            raise ValueError("unsupported goal-change robot")
        if not isinstance(changes, dict):
            raise ValueError("goal-change specifications must be keyed by case ID")
        if not cases or len({c["id"] for c in cases}) != len(cases) or set(changes) != {c["id"] for c in cases}:
            raise ValueError("one goal-change specification per unique case required")
        if any(c["mode"] not in {"coupled", "hand_free_base"} or "retarget" in c for c in cases):
            raise ValueError("nominal handoffs require hand-active cases without measured-state retargets")
        self.ready_for_steps = False
        self.changes = {
            name: validate_change(spec, PROFILES[robot].arm_size, arm_delta_limit=change_arm_limit)
            for name, spec in changes.items()
        }
        super().__init__(
            robot,
            cases,
            suite_path=suite_path,
            reward_config=reward_config,
            timing=timing,
            finish_after_hold=finish_after_hold,
            spread_requirements=spread_requirements,
            settling_budget_s=settling_budget_s,
        )

    def reset(self, *, seed=None, options=None):
        self.ready_for_steps = False
        self.change_delivered = False
        try:
            observation, info = super().reset(seed=seed, options=options)
            e = self.unwrapped
            first = e.control.task_trajectory
            change = self.changes[self.case["id"]]
            if change["dwell_s"] >= e.config.terminal_hold_seconds:
                raise ValueError("goal change must precede the terminal quiet hold")
            self.successor = nominal_successor(e.control, first, self.case, change)
            admission = admit_reference(e, self.successor)
            if not admission["accepted"]:
                raise ValueError("goal_change_reference_rejected:" + str(admission["reason"]))
            self.switch_at = float(np.ceil((first.end + change["dwell_s"]) / e.profile.dt - 1e-10) * e.profile.dt)
            if self.finish_after_hold:
                from dataclasses import replace

                horizon = float(
                    np.ceil((self.switch_at + self.successor.end + self.settling_budget_s) / e.profile.dt)
                    * e.profile.dt
                )
                e.config = replace(e.config, horizon=horizon)
            if self.switch_at + self.successor.end + e.config.terminal_hold_seconds > e.config.horizon:
                raise ValueError("goal_change_exceeds_episode_horizon")
            self.episode_origin = float(e.data.time)
            self.change_record = {
                "first_reference_sha256": info["reference_sha256"],
                "second_reference_sha256": digest(self.successor.record()),
                "switch_at_s": self.switch_at,
                "second_admission": admission,
                "anchor": "first_nominal_endpoint",
                "announced": "at_switch_not_in_prior_preview",
            }
            from emet.rl.manipulation.trajectory_metrics import trajectory_spread

            times = np.r_[first.times, self.switch_at + self.successor.times]
            base = np.vstack([first.base, self.successor.base])
            hand = np.vstack([first.hand, self.successor.hand])
            unique = np.r_[True, np.diff(times) > 1e-10]
            self.reference_spread = trajectory_spread(
                times[unique], hand[unique], base[unique], hand_mask=first.hand_mask, base_mask=first.base_mask
            )
            self.ready_for_steps = True
            return observation, {
                **info,
                "goal_change": self.change_record.copy(),
                "goal_change_delivered": False,
                "trajectory_spread": self.reference_spread,
            }
        except Exception:
            self.unwrapped.control.stop()
            raise

    def step(self, action):
        if not self.ready_for_steps or self.unwrapped.control.stopped:
            raise RuntimeError("reset an active goal-change environment before stepping")
        observation, reward, terminated, truncated, info = super().step(action)
        e = self.unwrapped
        if (
            not (terminated or truncated or self.change_delivered)
            and float(e.data.time) - self.episode_origin >= self.switch_at - 1e-8
        ):
            if e.control.stopped:
                raise RuntimeError("cannot deliver a goal change to stopped control")
            e.set_task_trajectory(self.successor)
            self.change_delivered = True
            # observe() replaces the token at the same timestamp; it neither
            # duplicates a time sample nor discards the measured history.
            observation = self.observe(e.control.observe())
        if terminated or truncated:
            self.ready_for_steps = False
        return (
            observation,
            reward,
            terminated,
            truncated,
            {**info, "goal_change": self.change_record.copy(), "goal_change_delivered": self.change_delivered},
        )

    def may_finish_reference(self):
        return self.change_delivered
