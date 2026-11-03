"""Training rewards tied to the frozen mastery acceptance contract.

Teacher measurements set training-only budgets. They never enter observations
or change execution limits. Evaluation and reward share the same budget math.
"""

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np


def smoothness_budgets(suite, teacher_metrics):
    gate = suite["gate"]
    floors = {
        "peak_force_N": gate["force_floor_N"],
        "base_linear_jerk": gate["base_linear_jerk_floor_m_s3"],
        "base_yaw_jerk": gate["base_yaw_jerk_floor_rad_s3"],
        "joint_jerk": gate["joint_jerk_floor_rad_s3"],
    }
    values = [gate["smoothness_ratio"], *floors.values(), *[teacher_metrics[k] for k in floors]]
    if not np.isfinite(values).all() or any(v <= 0 for v in list(floors.values()) + [gate["smoothness_ratio"]]):
        raise ValueError("finite teacher measurements and positive smoothness budgets required")
    if any(teacher_metrics[k] < 0 for k in floors):
        raise ValueError("negative teacher peak measurement")
    return {k: max(v, gate["smoothness_ratio"] * teacher_metrics[k]) for k, v in floors.items()}


def mastery_metrics(env, info):
    return {
        "peak_force_N": float(info["peak_contact_force"]),
        "base_linear_jerk": float(env.peak_base_axis_jerk[0]),
        "base_yaw_jerk": float(env.peak_base_axis_jerk[1]),
        "joint_jerk": float(np.max(env.peak_joint_jerk)),
        "hand_position_error_m": info["ee_position_error"],
        "hand_orientation_error_rad": info["ee_orientation_error"],
        "base_error_m": info["base_goal_error"],
        "base_heading_error_rad": info["base_heading_error_rad"],
        "hold_drift_m": env.peak_hold_drift,
        "settled_seconds": env.settle_time,
    }


def tracking_contract_passes(info, metrics, gate, *, path_failed):
    return bool(
        not info.get("initialization_failed", False)
        and info.get("is_success", False)
        and metrics["settled_seconds"] + 1e-8 >= gate["terminal_hold_s"]
        and metrics["hold_drift_m"] <= gate["hold_drift_m"]
        and not path_failed
        and not any(info.get(k, False) for k in ("unsafe_contact", "wheel_lift", "invalid_physics", "retention_lost"))
    )


@dataclass(frozen=True)
class ContractRewardConfig:
    jerk_excess_weight: float = 1.0
    peak_excess_weight: float = 2.0
    completion_bonus: float = 10.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("contract reward weights must be finite and nonnegative")


def add_contract_reward_arguments(parser):
    parser.add_argument("--contract-reward", action="store_true")
    parser.add_argument("--jerk-excess-weight", type=float, default=1.0)
    parser.add_argument("--peak-excess-weight", type=float, default=2.0)
    parser.add_argument("--completion-bonus", type=float, default=10.0)


def contract_reward_from_args(args):
    config = ContractRewardConfig(args.jerk_excess_weight, args.peak_excess_weight, args.completion_bonus)
    if not args.contract_reward and config != ContractRewardConfig():
        raise ValueError("reward weight overrides require --contract-reward")
    return config if args.contract_reward else None


class ContractObjective:
    """Dense excess cost, incremental peak cost, and one terminal contract bonus."""

    def __init__(self, suite, teacher_metrics, config=ContractRewardConfig()):
        self.gate, self.budgets, self.config = suite["gate"], smoothness_budgets(suite, teacher_metrics), config
        self.previous_peak_cost = 0.0
        self.bonus_paid = False
        self.passed = False

    @staticmethod
    def excess_cost(ratios):
        excess = np.maximum(np.asarray(ratios, dtype=float) - 1, 0)
        return float(np.sum(np.sqrt(1 + excess**2) - 1))

    def apply(self, reward, info, metrics, *, dt, terminal, path_failed):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite control timestep required")
        old_bonus = float(info.get("reward_terms", {}).get("success_bonus", 0))
        if info.get("initialization_failed") or info.get("invalid_physics"):
            self.passed = False
            return float(reward - old_bonus), {
                "removed_native_success_bonus": old_bonus,
                "contract_jerk_excess_cost": 0.0,
                "contract_peak_excess_cost": 0.0,
                "contract_success_bonus": 0.0,
            }
        jerk = info["control_tick_jerk"]
        base, joints = np.asarray(jerk["base"], dtype=float), np.asarray(jerk["joint"], dtype=float)
        if base.shape != (2,) or joints.ndim != 1 or not len(joints) or not np.isfinite(np.r_[base, joints]).all():
            raise ValueError("finite measured base/joint jerk required")
        if not np.isfinite([metrics[k] for k in self.budgets]).all() or any(metrics[k] < 0 for k in self.budgets):
            raise ValueError("finite nonnegative peak contract measurements required")
        ratios = [metrics[k] / v for k, v in self.budgets.items()]
        peak_cost = self.excess_cost(ratios)
        # Charge only newly accumulated excess, not a stale maximum every tick.
        peak_increment = max(0.0, peak_cost - self.previous_peak_cost)
        self.previous_peak_cost = max(self.previous_peak_cost, peak_cost)
        dense = (
            self.config.jerk_excess_weight
            * dt
            * self.excess_cost(
                [
                    abs(base[0]) / self.budgets["base_linear_jerk"],
                    abs(base[1]) / self.budgets["base_yaw_jerk"],
                    np.max(abs(joints)) / self.budgets["joint_jerk"],
                ]
            )
        )
        peak = self.config.peak_excess_weight * peak_increment
        self.passed = (
            tracking_contract_passes(info, metrics, self.gate, path_failed=path_failed)
            and info["reference_complete"]
            and all(r <= 1 for r in ratios)
        )
        bonus = self.config.completion_bonus if terminal and self.passed and not self.bonus_paid else 0.0
        self.bonus_paid |= bonus > 0
        return float(reward - old_bonus - dense - peak + bonus), {
            "removed_native_success_bonus": old_bonus,
            "contract_jerk_excess_cost": dense,
            "contract_peak_excess_cost": peak,
            "contract_success_bonus": bonus,
        }


class TeacherContractReward(gym.Wrapper):
    """Reset-bound reward wrapper; native observations and commands are untouched."""

    def __init__(self, env, teacher_report, config=ContractRewardConfig()):
        super().__init__(env)
        rows = teacher_report["cases"]
        self.rows = {row["id"]: row for row in rows}
        if len(rows) != len(self.rows):
            raise ValueError("duplicate teacher cases")
        self.reward_config, self.objective = config, None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        native = self.env.unwrapped
        row = self.rows[native.mastery_case["id"]]
        if (
            not row["mastered"]
            or info["initialization_failed"]
            or any(
                row["admission"][k] != native.mastery_admission[k] for k in ("initial_state_sha256", "reference_sha256")
            )
        ):
            raise ValueError("reward teacher/reset identity mismatch")
        self.objective = ContractObjective(native.suite, row["metrics"], self.reward_config)
        return obs, info

    def step(self, action):
        if self.objective is None:
            raise ValueError("reset before contract reward step")
        obs, reward, done, truncated, info = self.env.step(action)
        native = self.env.unwrapped
        shaped, terms = self.objective.apply(
            reward,
            info,
            mastery_metrics(native, info),
            dt=native.profile.dt,
            terminal=done or truncated,
            path_failed=native.path_failed,
        )
        return (
            obs,
            shaped,
            done,
            truncated,
            {
                **info,
                "reward_terms": {
                    **info.get("reward_terms", {}),
                    **terms,
                    "success_bonus": terms["contract_success_bonus"],
                },
                "contract_reward": {
                    "terminal_pass": self.objective.passed if done or truncated else None,
                    "budgets": self.objective.budgets.copy(),
                },
            },
        )
