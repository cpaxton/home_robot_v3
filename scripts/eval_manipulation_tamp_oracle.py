"""Run real privileged TAMP planning/execution against manipulation controllers.

Requires the native TAMP checkout explicitly because that adapter is not yet in
this branch. No source files in that checkout are modified. Frozen cases are
regression-only, never training examples. Reports remain local.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller", choices=("direct", "feedback", "policy"), default="feedback")
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--cases", nargs="+", choices=("familiar", "left", "right", "generated"), default=["familiar", "left", "right"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=700000)
    args = parser.parse_args()
    if args.repeats < 1 or (args.controller == "policy") != (args.policy is not None):
        parser.error("positive repeats and --policy exactly when --controller policy required")
    native = args.native_root.resolve(strict=True)
    harness_path = native / "scripts/eval_native_table_contact.py"
    sys.path.insert(0, str(native / "src"))
    import emet

    emet.__path__.append(str(Path(__file__).resolve().parents[1] / "src/emet"))
    spec = importlib.util.spec_from_file_location("native_oracle_harness", harness_path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    import mujoco
    import numpy as np

    from emet.rl.manipulation import oracle
    from emet.rl.manipulation.fixtures import evidence, load_case
    from emet.rl.manipulation.oracle import instrument_adapter, oracle_robot_class

    policy = None
    if args.policy:
        from emet.rl.manipulation.policy import ManipulationPolicy

        policy = ManipulationPolicy(args.policy, robot="stretch")
    robot_class = oracle_robot_class(harness.ContactRobot, controller=args.controller, policy=policy)
    harness.ContactRobot = robot_class
    harness.native_adapter.NativeTableAdapter = instrument_adapter(harness.native_adapter.NativeTableAdapter)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        harness_path,
        native / "src/emet/controller/task/tamp/native_adapter.py",
        native / "src/emet/controller/manipulation/physical_pick_place.py",
    ]
    manifest = {
        "scope": "oracle GT planning, in-process physical execution; no hardware/IPC/LLM certification",
        "controller": args.controller,
        "policy": str(args.policy) if args.policy else None,
        "policy_scene_transfer": policy is not None,
        "source_hashes": {str(p.relative_to(native)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "adapter_sha256": hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "retention_translation_m": 0.02,
        "retention_rotation_rad": 0.1,
        "cases": [],
        "promoted": False,
    }
    for case in args.cases:
        for repeat in range(args.repeats):
            if case in {"familiar", "generated"}:
                model = mujoco.MjModel.from_xml_path(harness.default_scene_xml_path)
                if case == "generated":
                    rng = np.random.default_rng(args.seed + repeat)
                    # Fixture initialization only, before oracle planning.
                    for name in ("object1", "object2"):
                        model.body(name).pos[:2] += rng.uniform([-0.02, -0.015], [0.02, 0.015])
                kwargs = {"sequence": ["object2", "object1"], "cluttered": True}
            else:
                model = load_case(case)
                frozen = evidence(case)
                kwargs = {
                    "sequence": ["object2", "object1"],
                    "cluttered": True,
                    "placement_regions": [frozen["goals"][name] for name in ("object2", "object1")],
                }
            result = harness.run_case(model, "object2", **kwargs)
            robot = robot_class.latest
            result.update(
                {
                    "controller_failure": robot.failure,
                    "contact_metrics": robot.monitor.metrics(),
                    "max_payload_translation": robot.max_translation,
                    "max_payload_rotation": robot.max_rotation,
                    "oracle_paths": robot.paths,
                    "control_trace": robot.trace,
                    "split": "frozen_regression"
                    if case in {"left", "right"}
                    else "oracle_generated_test"
                    if case == "generated"
                    else "oracle_control",
                    "seed": args.seed + repeat if case == "generated" else None,
                    "initial_object_positions": {
                        name: model.body(name).pos.tolist() for name in ("object1", "object2")
                    },
                    "historical_trace_replay": False,
                }
            )
            result["passed"] = bool(result["passed"] and not robot.failure)
            artifact = f"{case}-{repeat}.json"
            (args.output / artifact).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            row = {
                "case": case,
                "repeat": repeat,
                "passed": result["passed"],
                "controller_failure": robot.failure,
                "artifact": artifact,
                "executed_paths": len(robot.paths),
            }
            manifest["cases"].append(row)
            (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    return 0 if all(row["passed"] for row in manifest["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
