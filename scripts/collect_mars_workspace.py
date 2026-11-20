"""Collect spatially varied Mars tracking demonstrations with isolated simulator workers."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from measure_mars_workspace import measure
from propose_mars_base_splines import SEED, propose

from emet.rl.manipulation.arm_spline_benchmark import save, stop
from emet.rl.manipulation.arm_splines import ANCHORS, ArmSplineReference, ArmSplineTrackingEnv, SplineLevel
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.command_runtime import TaskPoseTeacherConfig, task_pose_teacher_action
from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledTrackingEnv
from emet.rl.manipulation.whole_body_observation import ACTOR_KEYS
from emet.rl.manipulation.whole_body_splines import WholeBodySplineReference

BASE_PATH_TOLERANCE = 0.035
FAMILIES = ("base", "arm", "coupled")
FLAGS = ("unsafe_contact", "wheel_lift", "invalid_physics", "retention_lost")
BASE_LEVEL = SplineLevel(0.1, 0.4, 0.0, initial_hold_s=0.0, maximum_duration_s=30.0)
ARM_LEVEL = SplineLevel(0.32, 0.28, 0.08, initial_hold_s=0.0, maximum_duration_s=24.0)


class BaseWorkspaceEnv(CoupledTrackingEnv):
    def make_reference(self, control, case):
        wheels = np.asarray(case["wheel_travel_waypoints_m"])
        arm = np.tile(control.positions(), (len(wheels), 1))
        reference = WholeBodySplineReference(control, arm, wheels, BASE_LEVEL)
        reference.hand_mask = False
        return reference


class HandWorkspaceEnv(ArmSplineTrackingEnv):
    def validate_case(self, case):
        if case["family"] not in {"arm", "coupled"} or case["posture"] not in ANCHORS:
            raise ValueError("Invalid hand workspace case")

    def sample_reference(self, rng, case, level):
        control = self.unwrapped.control
        start = control.positions()
        low = np.maximum(start - level.amplitude_rad, control.limits[:, 0] + 0.035)
        high = np.minimum(start + level.amplitude_rad, control.limits[:, 1] - 0.035)
        goals = rng.uniform(low, high, size=(3, len(start)))
        # Spread end configurations across shoulder and elbow sign combinations.
        for joint, sign in enumerate(case["goal_signs"]):
            a, b = sorted((start[joint], high[joint] if sign > 0 else low[joint]))
            goals[-1, joint] = a + (b - a) * rng.uniform(0.45, 0.95)
        arm = np.vstack([start, goals])
        if case["family"] == "arm":
            return ArmSplineReference(control, arm, level, timing=self.timing)
        travel, yaw = np.asarray(case["base_endpoint"]).reshape(2, 1) * np.array([0.0, 0.32, 0.70, 1.0])
        yaw[1:3] += rng.uniform(-0.045, 0.045, 2)
        half_width = control.profile.wheel_separation / 2
        wheels = np.column_stack((travel - yaw * half_width, travel + yaw * half_width))
        return WholeBodySplineReference(control, arm, wheels, level)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        plant = self.unwrapped
        if self.case["family"] == "arm":
            plant.fixed_base_trial = True
        plant.control.base_xy_tolerance = plant.active_base_xy_tolerance
        return obs, info


def make_cases(workspace, proposals_path, hand_counts):
    stored = json.loads(proposals_path.read_text())
    additional = propose(workspace["base"], seed=SEED + 1)
    cases = []
    for index, split in enumerate(("train", "audit")):
        base = stored if split == "train" else additional
        fixture = base["fixture_case"]
        for row in base["accepted"]:
            name = row["id"].split(":")[-1]
            cases.append(
                {
                    **fixture,
                    "id": f"workspace:{split}:base:{name}",
                    "fixture_seed": fixture["seed"],
                    "seed": 671_000_000 + 10_000 * index + int(name),
                    "split": split,
                    "family": "base",
                    "wheel_travel_waypoints_m": row["wheel_travel_waypoints_m"],
                }
            )
        for family_index, family in enumerate(("arm", "coupled"), start=1):
            for i in range(hand_counts[index]):
                seed = 671_000_000 + 10_000 * index + 1_000 * family_index + i
                rng = np.random.default_rng(seed)
                sector = (i // 4 + 2 * (i % 4)) % 8
                signs = [1 if sector & (1 << bit) else -1 for bit in range(3)]
                cases.append(
                    {
                        "id": f"workspace:{split}:{family}:{i}",
                        "family": family,
                        "seed": seed,
                        "split": split,
                        "level": 0,
                        "posture": tuple(ANCHORS)[i % len(ANCHORS)],
                        "goal_signs": signs,
                        "base_endpoint": [
                            float(signs[0] * rng.uniform(0.035, 0.095)),
                            float(signs[2] * rng.uniform(0.07, 0.23)),
                        ],
                    }
                )
    return cases


def run_case(case, output):
    mode = case["family"]
    if mode == "base":
        execution_case = {**case, "seed": case["fixture_seed"]}
        env = BaseWorkspaceEnv(
            "innate_mars", [execution_case], suite_path=COARSE_SUITE, finish_after_hold=True, settling_budget_s=7.0
        )
    else:
        env = HandWorkspaceEnv(
            [case], levels=(ARM_LEVEL,), maximum_attempts=4, timing="bounded_v2", finish_after_hold=True
        )
    plant = env.unwrapped
    plant.config = replace(plant.config, wheel_feedback_base_only_gains=(0.0, 0.5))
    row = {"case": case, "success": False}
    frames = {key: [] for key in (*ACTOR_KEYS, "action")}
    try:
        obs, reset = env.reset()
        control = plant.control
        reference = control.task_trajectory
        peak_base_error = peak_base_heading = 0.0
        teacher = TaskPoseTeacherConfig(coupled_base_tracking=mode == "coupled")
        for _ in range(int(np.ceil(plant.config.horizon / plant.profile.dt)) + 1):
            action = task_pose_teacher_action(control, config=teacher)
            if case["split"] == "train" and np.any(abs(obs["motion"]) > 1e-6):
                for key in ACTOR_KEYS:
                    frames[key].append(obs[key].copy())
                frames["action"].append(action.copy())
            obs, _, done, truncated, info = env.step(action)
            elapsed = control.trajectory_elapsed()
            target = reference.sample(elapsed)[0]
            error = control.base_pose() - target
            peak_base_error = max(peak_base_error, float(np.linalg.norm(error[:2])))
            peak_base_heading = max(peak_base_heading, abs(float(np.arctan2(np.sin(error[2]), np.cos(error[2])))))
            if done or truncated:
                break
        confirmed = stop(env)
        final = plant._info()
        flags = {key: bool(info.get(key) or final.get(key)) for key in FLAGS}
        path_ok = mode != "base" or (
            peak_base_error <= BASE_PATH_TOLERANCE and peak_base_heading <= plant.config.base_heading_tolerance
        )
        passed = bool(
            (done or truncated) and info["primitive_success"] and path_ok and confirmed and not any(flags.values())
        )
        row.update(
            success=passed,
            code=info["code"] if path_ok else "base_path_tracking_failed",
            stop_confirmed=confirmed,
            flags=flags,
            reference_sha256=reset["reference_sha256"],
            duration_s=reference.end,
            peak_base_error_m=peak_base_error,
            peak_base_heading_rad=peak_base_heading,
            peak_hand_position_m=info["peak_hand_position_error"],
            peak_hand_orientation_rad=info["peak_hand_orientation_error"],
        )
        if passed:
            path = output / f"{case['id']}.npz"
            data = {"base": reference.base, "hand": reference.hand}
            if frames["action"]:
                data.update({key: np.asarray(value, dtype=np.float32) for key, value in frames.items()})
            np.savez_compressed(path, **data)
            row.update(
                artifact=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=len(frames["action"])
            )
    except ValueError as exc:
        row.update(code=str(exc), stop_confirmed=stop(env))
    finally:
        env.close()
    save(output / f"{case['id']}.json", row)


def launch(case, output):
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, __file__, "--output", str(output), "--worker-case", json.dumps(case)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            break
        if result.returncode not in {-11, -6}:
            raise RuntimeError(result.stderr)
    path = output / f"{case['id']}.json"
    if not path.exists():
        save(path, {"case": case, "success": False, "code": f"worker_exit_{result.returncode}"})
    row = json.loads(path.read_text())
    print(case["id"], row["success"], row.get("code", "")[:120], flush=True)
    return row


def summarize(workspace, rows, output):
    result = {}
    for family in FAMILIES:
        result[family] = {}
        for split in ("train", "audit"):
            selected = [row for row in rows if row["case"]["family"] == family and row["case"]["split"] == split]
            admitted = [row for row in selected if row["success"]]
            episodes = []
            arrays = {key: [] for key in (*ACTOR_KEYS, "action")}
            spans = []
            for row in admitted:
                with np.load(output / row["artifact"], allow_pickle=False) as data:
                    episodes.append({"id": row["case"]["id"], "base": data["base"], "hand": data["hand"]})
                    if row["frames"]:
                        start = spans[-1]["stop"] if spans else 0
                        spans.append({"id": row["case"]["id"], "start": start, "stop": start + row["frames"]})
                        for key in arrays:
                            arrays[key].append(data[key])
            summary = {
                "attempted": len(selected),
                "admitted": len(admitted),
                "codes": dict(Counter(row["code"] for row in selected)),
            }
            if episodes:
                spec = {**workspace[family], "source_scope": "teacher-admitted generated references"}
                report = measure({**spec, "episodes": episodes})
                combined = measure({**spec, "episodes": spec["episodes"] + episodes}) if split == "train" else None
                summary.update(coverage=report, with_baseline=combined)
                if spans:
                    path = output / f"{family}-train.npz"
                    np.savez_compressed(path, **{key: np.concatenate(values) for key, values in arrays.items()})
                    probability = {weight["id"]: weight["weight"] for weight in report["weights"]}
                    summary.update(dataset=path.name, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    summary["training_episodes"] = [
                        {**span, "weight": probability.get(span["id"], 0.0)} for span in spans
                    ]
                if family == "coupled":
                    base = workspace["coupled_base_visited"]
                    for name, grid in (("coupled_base", base), ("coupled_task_hand", workspace["coupled_task_hand"])):
                        summary[name] = measure({**grid, "source_scope": spec["source_scope"], "episodes": episodes})
            result[family][split] = summary
    for family in FAMILIES:
        train, audit = (result[family][split].get("coverage") for split in ("train", "audit"))
        if train and audit and train["weights"] and audit["weights"]:

            def occupied(report):
                return {tuple(row["index"]) for row in report["goal_coverage"]["bins"]}

            result[family]["overlapping_goal_cells"] = len(occupied(train) & occupied(audit))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hand-counts", type=int, nargs=2, default=(40, 16), metavar=("TRAIN", "AUDIT"))
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_case:
        run_case(json.loads(args.worker_case), args.output)
        return
    if not args.workspace or not args.proposals or args.workers < 1 or min(args.hand_counts) < 1:
        parser.error("workspace, proposals, and positive worker and case counts are required")
    args.output.mkdir(parents=True, exist_ok=False)
    workspace = json.loads(args.workspace.read_text())["families"]
    cases = make_cases(workspace, args.proposals, args.hand_counts)
    assert len({c["seed"] for c in cases}) == len(cases)
    sources = source_hashes()
    save(args.output / "manifest.json", {"sources": sources, "cases": cases})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda case: launch(case, args.output), cases))
    report = summarize(workspace, rows, args.output)
    if source_hashes() != sources:
        raise RuntimeError("Simulator sources changed during collection")
    save(args.output / "report.json", {"families": report, "rows": rows, "hardware_used": False})


if __name__ == "__main__":
    main()
