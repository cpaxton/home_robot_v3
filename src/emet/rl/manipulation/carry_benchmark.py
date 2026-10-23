"""Freeze, evaluate and visualize Mars carry cases. Simulation only.

python -m emet.rl.manipulation.carry_benchmark freeze --output suite.json
python -m emet.rl.manipulation.carry_benchmark evaluate --suite suite.json --output results
python -m emet.rl.manipulation.carry_benchmark replay --suite suite.json --case-id ID --output case.html
"""

import argparse
import hashlib
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.evaluation import evaluate, paired_comparison
from emet.rl.manipulation.evaluation import summarize as summarize_episodes

STRATA = ("translation", "orientation", "combined", "retarget", "disturbance")
MASSES = (0.02, 0.05, 0.1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob("*.py")) + [
        root / "controller/manipulation/physical_pick_place.py",
        root / "motion/mujoco_collision.py",
        root / "robots/innate_mars/chassis.py",
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def default_config():
    return ManipulationConfig(
        robot="innate_mars",
        scenario="carry",
        action_mode="direct_velocity",
        contact_cone="elliptic",
        contact_impratio=3,
        balanced_grasp_setup=True,
        payload_object="object1",
        payload_mass_kg=0.05,
        randomization_scale=0,
        trajectory_preview=True,
        tracking_reward=True,
        goal_gated_settling=True,
        tamp_carry_trajectories=True,
        grasp_lift_height=0.045,
        support_half_width=0.04,
        base_xy_tolerance=0.005,
        hand_position_tolerance=0.005,
        hand_orientation_tolerance=0.05,
        tracking_precision=4,
        horizon=15,
    )


def make_suite(*, seed=1900000, per_stratum=20):
    if per_stratum < 1:
        raise ValueError("positive stratum size required")
    cases = []
    for mass in MASSES:
        for index, stratum in enumerate(STRATA):
            for repeat in range(per_stratum):
                cases.append(
                    {
                        "id": f"{round(mass * 1000)}g:{stratum}:{repeat}",
                        "seed": 1850001 + repeat
                        if stratum == "combined" and repeat < 10
                        else seed + 1000 * index + repeat,
                        "payload_mass_kg": mass,
                        "stratum": stratum,
                    }
                )
    return {"version": "mars-carry-v2", "config": asdict(default_config()), "cases": cases}


def case_config(suite, case):
    return replace(
        ManipulationConfig(**suite["config"]),
        payload_mass_kg=case["payload_mass_kg"],
        carry_target_mode=case["stratum"] if case["stratum"] in STRATA[:3] else "combined",
        retarget_probability=1.0 if case["stratum"] == "retarget" else 0,
        carry_push_scale=0.1 if case["stratum"] == "disturbance" else 0,
    )


def summarize(rows):
    groups = {}
    for mass in MASSES:
        group = [r for r in rows if r["payload_mass_kg"] == mass]
        admitted = [r for r in group if not r["initialization_failed"]]
        groups[str(mass)] = {
            "cases": len(group),
            "admitted": len(admitted),
            "admission_rate": len(admitted) / len(group) if group else 0,
            "successes": sum(r["is_success"] for r in group),
            "end_to_end_success_rate": sum(r["is_success"] for r in group) / len(group) if group else 0,
            "conditional_execution_success_rate": (
                sum(r["is_success"] for r in admitted) / len(admitted) if admitted else 0
            ),
            "unsafe_contact_count": sum(bool(r.get("unsafe_contact")) for r in group),
            "wheel_lift_count": sum(bool(r.get("wheel_lift")) for r in group),
            "retention_lost_count": sum(bool(r.get("retention_lost")) for r in group),
            "failures": dict(Counter(r["code"] for r in group if not r["is_success"])),
            "candidate_rejections": dict(
                Counter(c["reason"] for r in group for c in r["initial"]["rejected_candidates"])
            ),
        }
    return groups


def ready(report):
    """Small smoke suites cannot qualify a distribution for training."""
    if not report.get("complete") or report.get("controller") != "teacher":
        return False
    rows = report["episodes"]
    if len({r["case_id"] for r in rows}) != len(rows):
        return False
    for mass in MASSES:
        for stratum in STRATA:
            if sum(r["payload_mass_kg"] == mass and r["stratum"] == stratum for r in rows) < 20:
                return False
    return all(
        group["admission_rate"] >= 0.95
        and group["conditional_execution_success_rate"] >= 0.95
        and group["unsafe_contact_count"] == group["wheel_lift_count"] == 0
        for group in summarize(rows).values()
    )


def require_training_readiness(carry_report, pick_place_report, config, fingerprint):
    """Require complete, current teacher evidence before launching PPO workers."""
    if carry_report.get("source_hashes") != source_hashes():
        raise ValueError("carry teacher readiness source hashes are missing or stale")
    if not ready(carry_report):
        groups = summarize(carry_report.get("episodes", []))
        raise ValueError(f"carry teacher readiness failed: complete={carry_report.get('complete')}; per_mass={groups}")
    if carry_report["suite_sha256"] != digest(carry_report["suite"]):
        raise ValueError("carry readiness suite checksum mismatch")
    baseline = carry_report["suite"]["config"]
    for key in (
        "robot",
        "action_mode",
        "contact_cone",
        "contact_impratio",
        "balanced_grasp_setup",
        "payload_asset_set",
        "payload_object",
        "grasp_lift_height",
        "support_half_width",
        "grasp_depth",
        "grasp_pitch_rad",
        "trajectory_preview",
        "tracking_reward",
    ):
        if getattr(config, key) != baseline[key]:
            raise ValueError(f"training does not match qualified carry setup: {key}")
    if any(row["fingerprint"] != fingerprint for row in carry_report["episodes"]):
        raise ValueError("carry readiness model mismatch")
    rows = pick_place_report.get("cases", [])
    expected = {(seed, mass) for seed in range(1860001, 1860011) for mass in MASSES}
    actual = {(r["seed"], r.get("requested_payload_mass_kg")) for r in rows}
    if not pick_place_report.get("complete") or len(rows) != 30 or actual != expected:
        raise ValueError("complete frozen 30-case pick/place evidence required")
    if any(
        not r["success"]
        or r.get("control_source_sha256") != digest(source_hashes())
        or r.get("controller") != "velocity_teacher"
        or r.get("fingerprint") != fingerprint
        for r in rows
    ):
        raise ValueError("pick/place teacher readiness failing or stale")
    if any(s.get("unsafe_contact") or s.get("wheel_lift") for r in rows for s in r["stages"]):
        raise ValueError("pick/place safety gate failed")


def _teacher_case(arguments):
    suite, case = arguments
    return evaluate(config=case_config(suite, case), episodes=1, seed=case["seed"])["episodes"][0]


def run_suite(suite, output, *, policy=None, jobs=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("refusing to overwrite benchmark results; choose a fresh output")
    if jobs not in (1, 2) or (policy is not None and jobs != 1):
        raise ValueError("teacher benchmark supports 1–2 workers; policy evaluation uses one")
    rows = []
    sources = source_hashes()
    pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) if jobs > 1 else None
    results = (
        pool.map(_teacher_case, ((suite, case) for case in suite["cases"]))
        if pool
        else (
            evaluate(policy, config=case_config(suite, case), episodes=1, seed=case["seed"])["episodes"][0]
            for case in suite["cases"]
        )
    )
    try:
        return _write_results(suite, output, results, rows, sources, policy)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)


def _write_results(suite, output, results, rows, sources, policy):
    for case, row in zip(suite["cases"], results, strict=True):
        row.update(case_id=case["id"], payload_mass_kg=case["payload_mass_kg"], stratum=case["stratum"])
        row["episode_id"] = f"innate_mars:carry:{case['id']}:{case['seed']}"
        rows.append(row)
        report = {
            "suite": suite,
            "suite_sha256": digest(suite),
            "episodes": rows,
            "controller": "teacher" if policy is None else policy.manifest["weights_sha256"],
            "complete": len(rows) == len(suite["cases"]),
            "summary": summarize(rows),
            "scope": "simulated carry component; separate continuous pick/place gate required",
            "source_hashes": sources,
        }
        report["ready_for_training"] = ready(report)
        if source_hashes() != sources:
            raise RuntimeError("control sources changed during benchmark; rerun in a stable checkout")
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "report.json")
        print(case["id"], row["code"], row["is_success"], flush=True)
    return report


def replay(suite, case, output):
    """Offline interactive geometry, target axes, COM and rejected candidates.

    Plotly is a visualization-only dependency. No physics steps after reset.
    The adjacent JSON records all evidence even when no reference is admitted.
    """
    import plotly.graph_objects as go

    from emet.eval.physical_tamp import make_scene_checker
    from emet.rl.manipulation.grasp_setup import support_margin
    from emet.rl.manipulation.tracking import TaskFrameFK

    env = ManipulationEnv(case_config(suite, case))
    try:
        env.reset(seed=case["seed"])
        fk = TaskFrameFK(env.model, env.data, env.control)
        checker = make_scene_checker(
            env.model, {"object_body": env.payload_name, "gripper_bodies": env.profile.finger_bodies}
        )
        checker.set_payload(env.model, env.data, env.payload_name, env.profile.ee_body)
        base = env.control.base_pose()
        figure = go.Figure()
        points = env.data.xpos.copy()
        for body in sorted(env.monitor.robot_bodies):
            parent = env.model.body_parentid[body]
            edge = points[[parent, body]]
            figure.add_trace(
                go.Scatter3d(
                    x=edge[:, 0], y=edge[:, 1], z=edge[:, 2], mode="lines", line_color="gray", showlegend=False
                )
            )
        records = []
        path = env.arm_trajectory
        times = np.linspace(0, path.end, 80) if path else [0]
        for time in times:
            hand = fk(base, path.sample(time)[0] if path else env.control.positions()).copy()
            checker.configuration_collides(env.model, fk.data)
            support = support_margin(env.model, fk.data, env.profile, env.payload)
            records.append({"time_s": float(time), "hand": hand.tolist(), "support": support})
        hands = np.array([r["hand"] for r in records])
        figure.add_trace(
            go.Scatter3d(
                x=hands[:, 0],
                y=hands[:, 1],
                z=hands[:, 2],
                mode="lines+markers",
                name="Admitted hand path" if path else "Measured hand",
            )
        )
        for hand in hands[:: max(1, len(hands) // 8)]:
            axes = hand[3:].reshape(3, 2)
            for axis, color in zip(
                np.column_stack((axes, np.cross(axes[:, 0], axes[:, 1]))).T, ("red", "green", "blue"), strict=True
            ):
                edge = np.array([hand[:3], hand[:3] + 0.015 * axis])
                figure.add_trace(
                    go.Scatter3d(
                        x=edge[:, 0], y=edge[:, 1], z=edge[:, 2], mode="lines", line_color=color, showlegend=False
                    )
                )
        for candidate in env.trajectory_rejections:
            if "goal" not in candidate:
                continue
            hand = fk(base, candidate["goal"])
            figure.add_trace(
                go.Scatter3d(
                    x=[hand[0]],
                    y=[hand[1]],
                    z=[hand[2]],
                    mode="markers",
                    name=f"Rejected {candidate['candidate_id']}: {candidate['reason']}",
                )
            )
            if "failure_hand" in candidate:
                point = candidate["failure_hand"]
                figure.add_trace(
                    go.Scatter3d(
                        x=[point[0]],
                        y=[point[1]],
                        z=[point[2]],
                        mode="markers",
                        marker={"color": "red", "symbol": "x", "size": 7},
                        name=f"Failure at {candidate['time_s']:.2f}s",
                    )
                )
        polygon = np.array(records[0]["support"]["support_polygon_m"])
        polygon = np.vstack((polygon, polygon[0]))
        figure.add_trace(
            go.Scatter3d(
                x=polygon[:, 0], y=polygon[:, 1], z=np.zeros(len(polygon)), mode="lines", name="Wheel support polygon"
            )
        )
        com = np.array([r["support"]["loaded_com_m"] for r in records])
        figure.add_trace(
            go.Scatter3d(
                x=com[:, 0], y=com[:, 1], z=np.zeros(len(com)), mode="lines+markers", name="Loaded COM projection"
            )
        )
        for name in (env.payload_name, "table"):
            point = env.data.body(name).xpos
            figure.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers", name=name))
        figure.update_layout(
            title=f"{case['id']} — simulated geometry; not dynamic feasibility",
            scene_aspectmode="data",
            scene={"xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)"},
        )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(output, include_plotlyjs=True)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "suite_sha256": digest(suite),
                    "case": case,
                    "config": asdict(env.config),
                    "fingerprint": env.fingerprint,
                    "qpos": env.data.qpos.tolist(),
                    "admission": env.trajectory_admission,
                    "rejections": env.trajectory_rejections,
                    "samples": records,
                    "grasp_setup": env.randomization.get("grasp_setup"),
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        env.close()


def run_pick_place(regressions, output):
    """Execute every frozen case continuously, retaining failed cases."""
    from emet.rl.manipulation.integrated import run_case

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("refusing to overwrite pick/place results")
    rows = []
    for case in regressions["continuous_pick_place"]:
        row = run_case(
            robot="innate_mars",
            seed=case["seed"],
            payload_object="object1",
            payload_mass_kg=case["payload_mass_kg"],
            action_mode="direct_velocity",
            contact_cone="elliptic",
            contact_impratio=3,
            balanced_grasp_setup=True,
        )
        rows.append(row)
        temporary = output / "report.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "complete": len(rows) == len(regressions["continuous_pick_place"]),
                    "regressions_sha256": digest(regressions),
                    "cases": rows,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        temporary.replace(output / "report.json")
        print(case["seed"], case["payload_mass_kg"], row["success"], row["failure"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--seed", type=int, default=1900000)
    freeze.add_argument("--per-stratum", type=int, default=20)
    freeze.add_argument("--output", type=Path, required=True)
    pick_place = sub.add_parser("pick-place")
    pick_place.add_argument("--regressions", type=Path, default=Path("configs/benchmarks/mars_carry_regressions.json"))
    pick_place.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "replay"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--suite", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if name == "replay":
            cmd.add_argument("--case-id", required=True)
        else:
            cmd.add_argument("--policy", type=Path)
            cmd.add_argument("--teacher-report", type=Path)
            cmd.add_argument("--jobs", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    if args.command == "pick-place":
        run_pick_place(json.loads(args.regressions.read_text()), args.output)
        return
    if args.command == "freeze":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(make_suite(seed=args.seed, per_stratum=args.per_stratum), indent=2) + "\n")
        return
    suite = json.loads(args.suite.read_text())
    if args.command == "replay":
        case = next(c for c in suite["cases"] if c["id"] == args.case_id)
        replay(suite, case, args.output)
        return
    policy = None
    if args.policy:
        from emet.rl.manipulation.policy import ManipulationPolicy

        policy = ManipulationPolicy(args.policy, robot="innate_mars")
    report = run_suite(suite, args.output, policy=policy, jobs=args.jobs)
    if args.teacher_report:
        teacher = json.loads(args.teacher_report.read_text())
        if teacher["suite_sha256"] != report["suite_sha256"] or not teacher["complete"]:
            raise ValueError("paired evaluation requires the same complete teacher suite")
        a, b = dict(teacher), dict(report)
        a["summary"], b["summary"] = summarize_episodes(a["episodes"]), summarize_episodes(b["episodes"])
        (args.output / "paired.json").write_text(json.dumps(paired_comparison(a, b), indent=2) + "\n")


if __name__ == "__main__":
    main()
