"""Collect and audit arm-only inward staging and tracking in the Mars simulator."""

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from collect_mars_workspace import FLAGS, launch

from emet.rl.manipulation.arm_spline_benchmark import save, stop
from emet.rl.manipulation.arm_splines import ANCHORS, ArmSplineReference, ArmSplineTrackingEnv, SplineLevel
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.expert_routing import bind_expert
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_observation import ACTOR_KEYS

LEVEL = SplineLevel(0.2, 0.25, 0.065, initial_hold_s=0.2, maximum_duration_s=20.0)
ARM_CAP = 0.25
BASE_DRIFT_LIMIT = 0.01
REQUIRED_INWARD_TRAVEL = 0.06
KEYS = (*ACTOR_KEYS, "action")


class NearLimitEnv(ArmSplineTrackingEnv):
    def validate_case(self, case):
        if (
            case["joint"] not in range(5)
            or case["side"] not in ("lower", "upper")
            or case["split"] not in ("train", "audit")
        ):
            raise ValueError("expected a seeded near-limit Mars arm case")

    def limits_for(self, case):
        simulated = self.unwrapped.control.limits
        published = case["published_limits_rad"]
        if published is None:
            return simulated
        native = np.asarray(published, dtype=float)
        if native.shape != simulated.shape or not np.isfinite(native).all():
            raise ValueError("expected five finite named published arm ranges")
        limits = np.column_stack((np.maximum(simulated[:, 0], native[:, 0]), np.minimum(simulated[:, 1], native[:, 1])))
        if np.any(limits[:, 1] - limits[:, 0] < 0.2):
            raise ValueError("simulator and published joint ranges do not overlap enough for staging")
        return limits

    def sample_posture(self, rng, case, level):
        limits = self.limits_for(case)
        q = np.array(list(ANCHORS.values())[rng.integers(len(ANCHORS))])
        q = np.clip(
            q + rng.uniform(-level.posture_jitter_rad, level.posture_jitter_rad, len(q)),
            limits[:, 0] + 0.045,
            limits[:, 1] - 0.045,
        )
        joint = case["joint"]
        direction = 1 if case["side"] == "lower" else -1
        q[joint] = limits[joint, int(direction < 0)] + direction * rng.uniform(0.018, 0.035)
        return q

    def sample_reference(self, rng, case, level):
        control = self.unwrapped.control
        start = control.positions().copy()
        joint = case["joint"]
        direction = 1 if case["side"] == "lower" else -1
        stage = start.copy()
        stage[joint] += direction * rng.uniform(0.085, 0.13)
        gesture = stage.copy()
        wrist = 3 if joint == 4 else 4
        gesture[wrist] = np.clip(
            stage[wrist] + rng.choice([-1, 1]) * rng.uniform(0.07, 0.14),
            control.limits[wrist, 0] + 0.03,
            control.limits[wrist, 1] - 0.03,
        )
        reference = ArmSplineReference(control, [start, stage, gesture, stage], level, timing=self.timing)
        if case["published_limits_rad"] is not None:
            limits = self.limits_for(case)
            if np.any(reference.certificate["joint_minimum"] < limits[:, 0] + 0.002) or np.any(
                reference.certificate["joint_maximum"] > limits[:, 1] - 0.002
            ):
                raise ValueError("reference leaves the intersection of published and simulator limits")
        return reference

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.unwrapped.fixed_base_trial = True
        self.unwrapped.control.base_xy_tolerance = self.unwrapped.active_base_xy_tolerance
        return obs, info


def rollout(env, case, actor, *, collect=False):
    obs, reset = env.reset()
    plant = env.unwrapped
    control = plant.control
    limits = env.limits_for(case)
    direction = 1 if case["side"] == "lower" else -1
    joint = case["joint"]
    start = control.positions().copy()
    base = control.base_pose()[:2].copy()
    start_margin = float(abs(start[joint] - limits[joint, int(direction < 0)]))
    policy = bind_expert(
        "arm_only",
        control,
        {"arm_only": lambda observation: actor({name: observation[name] for name in ACTOR_KEYS})}
        if actor is not None
        else {},
        arm_speed_limit=ARM_CAP,
    )
    peak_speed = peak_drift = 0.0
    minimum_margin = start_margin
    outward_ticks = 0
    frames = {key: [] for key in KEYS}
    for _ in range(int(np.ceil(plant.config.horizon / plant.profile.dt)) + 1):
        action = policy(obs)
        if collect and np.any(abs(obs["motion"]) > 1e-6):
            for key in ACTOR_KEYS:
                frames[key].append(obs[key].copy())
            frames["action"].append(action.copy())
        q = control.positions()
        margin = direction * (q[joint] - limits[joint, int(direction < 0)])
        outward_ticks += int(margin < 0.025 and direction * action[2 + joint] * plant.profile.speed[joint] < -0.01)
        obs, _, done, truncated, info = env.step(action)
        positions = control.positions()
        minimum_margin = min(
            minimum_margin,
            float(np.min(np.column_stack((positions - limits[:, 0], limits[:, 1] - positions)))),
        )
        peak_speed = max(peak_speed, float(np.max(np.abs(control.velocities()))))
        peak_drift = max(peak_drift, float(np.linalg.norm(control.base_pose()[:2] - base)))
        if done or truncated:
            break
    confirmed = stop(env)
    final = plant._info()
    flags = {key: bool(info.get(key) or final.get(key)) for key in FLAGS}
    inward = float(direction * (control.positions()[joint] - start[joint]))
    gate = (
        minimum_margin >= 0.0
        and outward_ticks == 0
        and peak_speed <= ARM_CAP
        and peak_drift <= BASE_DRIFT_LIMIT
        and inward >= REQUIRED_INWARD_TRAVEL
    )
    success = bool((done or truncated) and info["primitive_success"] and gate and confirmed and not any(flags.values()))
    return {
        "success": success,
        "code": info["code"] if gate or not info["primitive_success"] else "near_limit_gate_failed",
        "stop_confirmed": confirmed,
        "flags": flags,
        "reference_sha256": reset["reference_sha256"],
        "initial_margin_rad": start_margin,
        "minimum_measured_margin_rad": minimum_margin,
        "outward_ticks_near_limit": outward_ticks,
        "final_inward_displacement_rad": inward,
        "peak_arm_speed_rad_s": peak_speed,
        "peak_base_drift_m": peak_drift,
        "peak_hand_error": info["spline_peak_path_error"],
        "reset_rejections": reset["spline_rejections"],
    }, frames


def run_case(case, output):
    env = NearLimitEnv([case], levels=(LEVEL,), maximum_attempts=12, timing="bounded_v2", finish_after_hold=True)
    row = {"case": case, "success": False}
    try:
        teacher, frames = rollout(env, case, None, collect=case["split"] == "train")
        row.update(teacher)
        if row["success"] and frames["action"]:
            path = output / f"{case['id']}.npz"
            np.savez_compressed(path, **{key: np.asarray(value, dtype=np.float32) for key, value in frames.items()})
            row.update(artifact=path.name, frames=len(frames["action"]))
        if row["success"] and case.get("actor"):
            actor = WholeBodyBundle(case["actor"], robot="innate_mars")
            result, _ = rollout(env, case, actor)
            assert result["reference_sha256"] == teacher["reference_sha256"]
            row["actor_result"] = result
    except ValueError as exc:
        row["code"] = str(exc)
        row["stop_confirmed"] = stop(env)
    finally:
        env.close()
    save(output / f"{case['id']}.json", row)


def summarize(rows, output):
    bank = {"families": {name: {"train": {}} for name in ("arm", "base", "coupled")}, "rows": rows}
    training = [r for r in rows if r.get("artifact")]
    if not training:
        return bank
    counts = defaultdict(int)
    for row in training:
        key = (row["case"]["joint"], row["case"]["side"])
        counts[key] += 1
    cursor, spans = 0, []
    arrays = {key: [] for key in KEYS}
    for row in training:
        key = (row["case"]["joint"], row["case"]["side"])
        with np.load(output / row["artifact"], allow_pickle=False) as data:
            for name in KEYS:
                arrays[name].append(data[name])
        end = cursor + row["frames"]
        spans.append({"id": row["case"]["id"], "start": cursor, "stop": end, "weight": 1 / (len(counts) * counts[key])})
        cursor = end
    path = output / "arm-train.npz"
    np.savez_compressed(path, **{key: np.concatenate(value) for key, value in arrays.items()})
    bank["families"]["arm"]["train"].update(
        dataset=path.name, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), training_episodes=spans
    )
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor", type=Path, help="Arm specialist evaluated on teacher-passing audit cases")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--limits-audit", type=Path, help="Use the intersection with the published ranges in this capture audit"
    )
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_case:
        run_case(json.loads(args.worker_case), args.output)
        return
    limits = None
    if args.limits_audit:
        audit = json.loads(args.limits_audit.read_text())["audit"]
        limits = [audit["published_joint_limits"][f"joint{index}"] for index in range(1, 6)]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        {
            "id": f"near-limit:{split}:{joint + 1}:{side}:{rep}",
            "seed": 672_000_000 + split_index * 10_000 + joint * 100 + side_index * 10 + rep,
            "split": split,
            "joint": joint,
            "side": side,
            "level": 0,
            "published_limits_rad": limits,
            "actor": str(args.actor.resolve()) if split == "audit" and args.actor else None,
        }
        for split_index, (split, repeats) in enumerate((("train", 3), ("audit", 2)))
        for joint in range(5)
        for side_index, side in enumerate(("lower", "upper"))
        for rep in range(repeats)
    ]
    sources = source_hashes()
    save(args.output / "manifest.json", {"cases": cases, "sources": sources})
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda case: launch(case, args.output, worker=__file__), cases))
    if source_hashes() != sources:
        raise RuntimeError("Simulator sources changed during the near-limit collection")
    save(args.output / "report.json", summarize(rows, args.output))


if __name__ == "__main__":
    main()
