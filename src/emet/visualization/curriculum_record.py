"""Record actual teacher executions for a 4x4 curriculum gallery (sim only)."""

import argparse
import gzip
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from emet.rl.manipulation.arm_spline_benchmark import save, stop
from emet.rl.manipulation.arm_splines import ArmSplineTrackingEnv
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledTrackingEnv
from emet.rl.manipulation.goal_change_tracking import GoalChangeTrackingEnv, load_curriculum
from emet.visualization.curriculum_cases import FAMILIES, ROOT, TITLES, cases_for


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(e, origin):
    c = e.control
    return (
        float(e.data.time) - origin,
        e.data.qpos.copy(),
        c.base_goal.copy(),
        c.hand_goal.copy(),
        c.positions().copy(),
        c.ee_pose().copy(),
        c.base_pose().copy(),
    )


def archive_case(e, frames, path, row, planned=None):
    if not frames:
        raise ValueError("no recorded physical states")
    arrays = {
        k: np.asarray([f[i] for f in frames])
        for i, k in enumerate(("time_s", "qpos", "base_target", "hand_target", "joints", "hand", "base"))
    }
    if planned is not None:
        arrays.update(planned)
    if any(not np.isfinite(v).all() for v in arrays.values()) or np.any(np.diff(arrays["time_s"]) < 0):
        raise ValueError("finite monotone recorded states required")
    temporary_model = path.with_suffix(".mjb")
    mujoco.mj_saveModel(e.model, str(temporary_model))
    raw_model_sha = sha(temporary_model)
    model_path = path.parent / ("model-" + raw_model_sha + ".mjb.gz")
    if not model_path.exists():
        with temporary_model.open("rb") as source, gzip.open(model_path, "wb", compresslevel=1) as dest:
            shutil.copyfileobj(source, dest)
    temporary_model.unlink()
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    row.update(
        recording=path.with_suffix(".npz").name,
        model=model_path.name,
        model_raw_sha256=raw_model_sha,
        duration_s=float(arrays["time_s"][-1]),
        frames=len(frames),
        model_fingerprint=e.fingerprint,
        measured_joint_span_deg=np.rad2deg(np.ptp(arrays["joints"], axis=0)).tolist(),
    )
    row["assets"] = {name: sha(path.parent / name) for name in (row["recording"], row["model"])}
    return row


def regular_environment(family, cases):
    if family == "dense_arm":
        from emet.rl.manipulation.arm_splines import SplineLevel, load_manifest

        spec = load_manifest(ROOT / "configs/benchmarks/mars_arm_splines_v2.json")
        return ArmSplineTrackingEnv(
            cases,
            levels=tuple(SplineLevel(**v) for v in spec["levels"]),
            timing=spec["reference_timing"],
            finish_after_hold=spec["finish_after_hold"],
            spread_requirements=spec["spread_requirements"],
        )
    if family.startswith("arm_splines_"):
        return ArmSplineTrackingEnv(cases)
    if family == "goal_changes":
        spec = load_curriculum(ROOT / "configs/benchmarks/mars_goal_changes_v1.json")
        env = GoalChangeTrackingEnv(
            "innate_mars", cases, changes={c["id"]: spec["goal_changes"][c["id"]] for c in cases}
        )
    else:
        env = CoupledTrackingEnv("innate_mars", cases, suite_path=COARSE_SUITE)
    env.unwrapped.config = replace(env.unwrapped.config, wheel_feedback_base_only_gains=(0.0, 0.5))
    return env


def record_regular(env, case, path):
    row = {"case": case, "controller": "analytic_teacher", "admitted": False, "success": False}
    e = env.unwrapped
    frames = []
    try:
        _, initial = env.reset(options={"case_id": case["id"]})
        row.update(
            admitted=True,
            reference_sha256=initial["reference_sha256"],
            rejected_proposals=initial.get("spline_rejections", []),
        )
        origin = float(e.data.time)
        reference = e.control.task_trajectory
        indices = np.unique(
            np.r_[np.arange(0, len(reference.times), max(1, len(reference.times) // 150)), len(reference.times) - 1]
        )
        planned = {"planned_hand": reference.hand[indices].copy(), "planned_base": reference.base[indices].copy()}
        if "successor" in vars(env):
            following = env.successor
            indices = np.unique(
                np.r_[np.arange(0, len(following.times), max(1, len(following.times) // 150)), len(following.times) - 1]
            )
            planned["planned_hand"] = np.vstack([planned["planned_hand"], following.hand[indices]])
            planned["planned_base"] = np.vstack([planned["planned_base"], following.base[indices]])
        frames.append(snapshot(e, origin))
        info = {}
        for _ in range(int(np.ceil(e.config.horizon / e.profile.dt))):
            _, _, done, truncated, info = env.step(task_pose_teacher_action(e.control))
            frames.append(snapshot(e, origin))
            if done or truncated:
                break
        row.update(
            success=bool(info.get("primitive_success")),
            code=info.get("code"),
            peak_path_error=info.get("spline_peak_path_error"),
            safety={k: bool(info.get(k)) for k in ("wheel_lift", "unsafe_contact", "invalid_physics")},
            goal_change_delivered=bool(info.get("goal_change_delivered")),
        )
        row["stop_confirmed"] = stop(env)
        if not row["stop_confirmed"]:
            row["success"] = False
        row["stop_tail_in_video"] = False
        return archive_case(e, frames, path, row, planned)
    except ValueError as exc:
        row.update(code=str(exc), stop_confirmed=stop(env))
        return row


def record_tamp(case, path):
    from emet.eval.mars_tamp_control import fixture
    from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv

    raw = {k: v for k, v in case.items() if k != "split"}
    traces = path.parent / (path.name + "-trace")
    env = TampPrimitiveEnv(
        [raw],
        oracle=True,
        primitive="grasp_pose",
        planning_scope="unloaded_prefix",
        wheel_feedback_current_velocity=True,
        wheel_feedback_base_only_gains=(0.0, 0.5),
        trace_dir=traces,
    )
    info = {}
    row = {"case": case, "controller": "analytic_teacher", "admitted": False, "success": False}
    try:
        _, info = env.reset()
        for _ in range(2000):
            action = info.get("oracle_action", np.zeros(8))
            _, _, done, _, info = env.step(action)
            if done:
                break
        else:
            raise ValueError("visualization decision budget exhausted")
    except (ValueError, RuntimeError, EOFError, TimeoutError) as exc:
        row["code"] = str(exc)
    finally:
        env.close()  # Trace is finalized after closing the worker.
    trace_path = traces / "episode-0001.json.gz"
    if not trace_path.exists():
        return row
    with gzip.open(trace_path, "rt") as stream:
        trace = json.load(stream)
    ticks = [r for r in trace.get("ticks", []) if "physics_after" in r]
    row.update(
        admitted=bool(ticks),
        success=bool(info.get("primitive_success")),
        code=info.get("code", row.get("code", trace["outcome"].get("error", "not_recorded"))),
        stop_confirmed=bool(info.get("stop_confirmed")),
        full_pick_place_success=False,
        safety={k: bool(info.get(k)) for k in ("wheel_lift", "unsafe_contact", "invalid_physics")},
        trace_sha256=sha(trace_path),
        stop_tail_in_video=True,
    )
    if not ticks:
        return row
    plant, admission = fixture(raw, wheel_feedback_current_velocity=True)
    try:
        if not admission["accepted"] or plant.fingerprint != info["model_fingerprint"]:
            raise ValueError("render plant differs from recorded TAMP plant")
        origin = ticks[0]["physics_before"]["time"]
        frames = []
        for r in ticks:
            q = np.array(r["physics_after"]["qpos"])
            frames.append(
                (
                    r["physics_after"]["time"] - origin,
                    q,
                    r["base_goal"],
                    r["hand_goal"],
                    r["joint_position"],
                    r["hand"],
                    r["base"],
                )
            )
        return archive_case(plant, frames, path, row)
    finally:
        plant.close()


def record(output, families=FAMILIES):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for family in families:
        directory = output / family
        if directory.exists():
            raise ValueError("fresh family recording required: " + family)
        directory.mkdir()
        cases = cases_for(family)
        sources = source_hashes()
        report = {
            "format": "curriculum-recordings-v1",
            "family": family,
            "title": TITLES[family],
            "controller": "analytic_teacher",
            "hardware_used": False,
            "safety_promoted": False,
            "sources": sources,
            "producer_sha256": sha(__file__),
            "cases_sha256": digest(cases),
            "rows": [],
            "complete": False,
        }
        save(directory / "report.json", report)
        env = None
        try:
            if family != "tamp":
                env = regular_environment(family, cases)
            for index, case in enumerate(cases):
                path = directory / f"case-{index:02d}"
                row = record_tamp(case, path) if family == "tamp" else record_regular(env, case, path)
                report["rows"].append(row)
                save(directory / "report.json", report)
                print("RECORD", family, index, row.get("success"), str(row.get("code"))[:120], flush=True)
            if source_hashes() != sources:
                raise RuntimeError("control sources changed during recording")
            report.update(
                complete=True,
                summary={
                    "attempts": 16,
                    "admitted": sum(r["admitted"] for r in report["rows"]),
                    "successes": sum(r["success"] for r in report["rows"]),
                },
            )
            save(directory / "report.json", report)
        finally:
            if env is not None:
                env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, action="append")
    args = parser.parse_args()
    record(args.output, args.family or FAMILIES)


if __name__ == "__main__":
    main()
