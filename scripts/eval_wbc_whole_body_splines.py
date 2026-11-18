"""Inspect or teacher-test joint/wheel-generated full-body references (sim only)."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.arm_spline_benchmark import save, stop
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_runtime import TaskPoseTeacherConfig, task_pose_teacher_action
from emet.rl.manipulation.whole_body_splines import LEVELS, WholeBodySplineTrackingEnv, manifest
from emet.rl.regression_gate import file_digest


def gallery(report, output):
    rows = [{**r, "time_s": r.get("times"), "rejections": r.get("rejected_proposals", [])} for r in report["cases"]]
    template = Path(__file__).resolve().parents[1] / "src/emet/rl/manipulation/arm_spline_gallery.html"
    Path(output).write_text(template.read_text().replace("__DATA__", json.dumps(rows).replace("<", "\\u003c")))


def run(output, *, level=0, count=5, teacher=False):
    output = Path(output)
    if output.exists() or type(count) is not int or not 1 <= count <= 1000 or level not in range(len(LEVELS)):
        raise ValueError("fresh output and bounded level/population required")
    output.mkdir(parents=True)
    spec = manifest(max(5, count), max(5, count))
    cases = [c for c in spec["cases"] if c["level"] == level and int(c["id"].rsplit(":", 1)[1]) < count]
    sources = source_hashes()
    report = {
        "format": "whole-body-joint-spline-preflight-v1",
        "manifest": spec,
        "sources": sources,
        "runner_sha256": file_digest(__file__),
        "teacher_enabled": teacher,
        "hardware_used": False,
        "safety_promoted": False,
        "complete": False,
        "cases": [],
    }
    env = WholeBodySplineTrackingEnv(cases)
    report["model_fingerprint"] = env.unwrapped.fingerprint
    try:
        for case in cases:
            row = {"case": case, "admitted": False}
            try:
                _, info = env.reset(options={"case_id": case["id"]})
                reference = env.reference
                row.update(
                    admitted=True,
                    rejected_proposals=env.rejections.copy(),
                    certificate=reference.certificate,
                    admission=env.admission,
                    reference_sha256=digest(reference.record()),
                    trajectory_spread=reference.spread,
                    times=reference.times.tolist(),
                    base=reference.base.tolist(),
                    hand=reference.hand.tolist(),
                    joints=[reference.coordinates(t)[1].tolist() for t in reference.times],
                )
                if teacher:
                    e = env.unwrapped
                    for _ in range(int(np.ceil(e.config.horizon / e.profile.dt)) + 1):
                        action = task_pose_teacher_action(
                            e.control, config=TaskPoseTeacherConfig(coupled_base_tracking=True)
                        )
                        _, _, done, truncated, info = env.step(action)
                        if done or truncated:
                            break
                    row.update(
                        success=bool(info["primitive_success"]),
                        code=info["code"],
                        peak_path_error=info["spline_peak_path_error"],
                        metrics={
                            k: info[k]
                            for k in (
                                "ee_position_error",
                                "ee_orientation_error",
                                "unsafe_contact",
                                "wheel_lift",
                                "invalid_physics",
                                "settled_seconds",
                                "peak_contact_force",
                            )
                        },
                    )
            except ValueError as exc:
                row.update(code=str(exc), rejected_proposals=env.rejections.copy(), success=False)
            row["stop_confirmed"] = stop(env)
            report["cases"].append(row)
            save(output / "report.json", report)
            print(case["id"], row.get("success"), row.get("code", "geometry_admitted"), flush=True)
        if source_hashes() != sources or file_digest(__file__) != report["runner_sha256"]:
            raise RuntimeError("sources changed during whole-body spline evaluation")
        report["complete"] = True
        save(output / "report.json", report)
        gallery(report, output / "gallery.html")
        return report
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--count", type=int, default=5, help="cases per train/audit split; five covers every reset class")
    p.add_argument("--teacher", action="store_true")
    a = p.parse_args()
    run(a.output, level=a.level, count=a.count, teacher=a.teacher)
