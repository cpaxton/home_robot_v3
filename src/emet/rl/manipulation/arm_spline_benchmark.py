"""Generate, inspect and teacher-validate a seeded arm-spline distribution."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.arm_splines import ArmSplineTrackingEnv, SplineLevel, load_manifest
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.regression_gate import source_state

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / "configs/benchmarks/mars_arm_splines_v1.json"


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def stop(env):
    e = env.unwrapped
    e.control.stop()
    start = float(e.data.time)
    while not e._physics_invalid and not whole_body_stopped(e.control) and e.data.time - start < 0.75 - 1e-9:
        e._tick(np.zeros(e.profile.action_size), score=False)
    return bool(not e._physics_invalid and whole_body_stopped(e.control))


def episode(env, case, policy=None):
    observation, initial = env.reset(options={"case_id": case["id"]})
    e = env.unwrapped
    identity = digest({"qpos": e.data.qpos.tolist(), "qvel": e.data.qvel.tolist()})
    for _ in range(int(e.config.horizon / e.profile.dt)):
        action = (
            task_pose_teacher_action(e.control)
            if policy is None
            else policy({k: observation[k] for k in ("history", "preview", "motion")})
        )
        observation, _, done, truncated, info = env.step(action)
        if done or truncated:
            break
    metrics = e._info()
    confirmed = stop(env)
    return {
        "case": case.copy(),
        "success": bool(info["primitive_success"]),
        "code": info["code"],
        "stop_confirmed": confirmed,
        "ticks": e.steps,
        "reference_sha256": initial["reference_sha256"],
        "initial_state_sha256": identity,
        "admission": initial["spline_admission"],
        "rejections": initial["spline_rejections"],
        "peak_path_error": info["spline_peak_path_error"],
        "joint_limit_push_ticks": info["joint_limit_push_ticks"],
        "metrics": {
            k: metrics[k]
            for k in (
                "ee_position_error",
                "ee_orientation_error",
                "base_goal_error",
                "base_heading_error_rad",
                "peak_base_linear_jerk_m_s3",
                "peak_base_yaw_jerk_rad_s3",
                "peak_contact_force",
                "wheel_lift",
                "unsafe_contact",
                "settled_seconds",
            )
        },
        "invalid_physics": bool(e._physics_invalid),
    }


def require_preflight(path, manifest_path, level):
    report = json.loads(Path(path).read_text())
    spec = load_manifest(manifest_path)
    expected = [c for c in spec["cases"] if c["level"] == level]
    rows = report.get("results", {}).get("teacher", [])
    if (
        report.get("format") != "arm-spline-preflight-v1"
        or report.get("complete") is not True
        or report.get("manifest_sha256") != digest(spec)
        or report.get("sources") != source_hashes()
        or report.get("source_state") != source_state(ROOT)
        or report.get("level") != level
        or [r.get("case") for r in rows] != expected
        or not rows
        or any(not r.get("admitted") or not r.get("stop_confirmed") or r.get("invalid_physics") for r in rows)
    ):
        raise ValueError("complete source/manifest/case-bound spline preflight required")
    train = [r for r in rows if r["case"]["split"] == "train"]
    if sum(r["success"] for r in train) / len(train) < 0.85 or any(
        r["metrics"]["unsafe_contact"] or r["metrics"]["wheel_lift"] for r in train
    ):
        raise ValueError("teacher training distribution is not reliable enough for this curriculum level")
    return report


def run(manifest_path, output, *, level=0, teacher=False, policy_paths=()):
    output = Path(output)
    if output.exists():
        raise ValueError("fresh arm-spline output directory required")
    spec = load_manifest(manifest_path)
    selected = [c for c in spec["cases"] if c["level"] == level]
    if not selected:
        raise ValueError("unknown spline level")
    output.mkdir(parents=True)
    sources = source_hashes()
    bound_state = source_state(ROOT)
    levels = tuple(SplineLevel(**v) for v in spec["levels"])
    env = ArmSplineTrackingEnv(selected, levels=levels)
    report = {
        "format": "arm-spline-preflight-v1",
        "manifest_sha256": digest(spec),
        "sources": sources,
        "source_state": bound_state,
        "level": level,
        "results": {},
        "targets": [],
        "complete": False,
        "hardware_used": False,
        "safety_promoted": False,
        "training_mix": spec["training_mix"],
        "fingerprint": env.unwrapped.fingerprint,
    }
    try:
        for case in selected:
            try:
                env.reset(options={"case_id": case["id"]})
                reference = env.reference
                # The gallery is target geometry, not footage or policy success.
                step = max(1, len(reference.times) // 150)
                indices = np.unique(np.r_[np.arange(0, len(reference.times), step), len(reference.times) - 1])
                report["targets"].append(
                    {
                        "case": case.copy(),
                        "reference_sha256": digest(reference.record()),
                        "admission": env.admission,
                        "rejections": env.rejections.copy(),
                        "certificate": reference.certificate,
                        "time_s": reference.times[indices].tolist(),
                        "hand": reference.hand[indices].tolist(),
                    }
                )
            except ValueError as exc:
                report["targets"].append(
                    {"case": case.copy(), "admitted": False, "reason": str(exc), "rejections": env.rejections.copy()}
                )
            save(output / "targets.json", report["targets"])
        template = Path(__file__).with_name("arm_spline_gallery.html").read_text()
        (output / "gallery.html").write_text(
            template.replace("__DATA__", json.dumps(report["targets"]).replace("<", "\\u003c"))
        )
        controllers = [("teacher", None)] if teacher else []
        for index, path in enumerate(policy_paths):
            policy = WholeBodyBundle(path, robot="innate_mars", fingerprint=env.unwrapped.fingerprint)
            controllers.append(("policy_" + str(index), policy))
        for name, policy in controllers:
            report["results"][name] = []
            for case in selected:
                try:
                    row = episode(env, case, policy)
                    row["admitted"] = True
                except ValueError as exc:
                    row = {
                        "case": case.copy(),
                        "success": False,
                        "code": str(exc),
                        "admitted": False,
                        "stop_confirmed": stop(env),
                        "invalid_physics": bool(env.unwrapped._physics_invalid),
                        "rejections": env.rejections.copy(),
                    }
                report["results"][name].append(row)
                save(output / "report.json", report)
                print(name, case["id"], row["success"], row["code"], flush=True)
        if source_hashes() != sources or source_state(ROOT) != bound_state:
            raise RuntimeError("sources changed during spline benchmark")
        report["complete"] = True
        report["summary"] = {
            k: {
                "attempts": len(v),
                "admitted": sum(r["admitted"] for r in v),
                "successes": sum(r["success"] for r in v),
                "confirmed_stops": sum(r["stop_confirmed"] for r in v),
            }
            for k, v in report["results"].items()
        }
        save(output / "report.json", report)
        print(json.dumps(report["summary"]), flush=True)
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--policy", type=Path, action="append", default=[])
    args = parser.parse_args()
    run(args.manifest, args.output, level=args.level, teacher=args.teacher, policy_paths=args.policy)


if __name__ == "__main__":
    main()
