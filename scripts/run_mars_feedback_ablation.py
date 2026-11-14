"""Run paired simulator pilots through the existing mastery training gates.

The four variants share cases, seeds, budgets, PPO settings and physical gates.
Orientation changes train from scratch. A baseline/velocity pair can instead
start from one checkpoint; reference columns then start with zero influence.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

VARIANTS = {
    "baseline": ("rotation6", False),
    "velocity": ("rotation6", True),
    "rotvec": ("rotvec", False),
    "both": ("rotvec", True),
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "configs/benchmarks/mars_tracking_mastery_hand_accuracy_v1.json"
    )
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--stages", choices=("base", "arm", "coupled"), nargs="+", default=["arm"])
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[90])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--arm-std", type=float, default=0.1)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 256 or len(set(args.variants)) != len(args.variants):
        parser.error("positive steps divisible by 256 and unique variants required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("unique paired seeds required")
    if args.source and any(VARIANTS[v][0] != "rotation6" for v in args.variants):
        parser.error("checkpoint continuation supports the baseline/velocity pair; rotvec requires scratch training")

    from emet.rl.manipulation.mastery import load_suite
    from emet.rl.regression_gate import verify_regressions

    verify_regressions(args.regression_receipt)
    suite = load_suite(args.suite)
    if args.source:
        from emet.rl.manipulation.policy import ManipulationPolicy

        source = ManipulationPolicy(args.source)
        if callable(getattr(source.model.policy, "executor_mask", None)):
            # A continuation must retain the source's executor action masks.
            suite["training_policy"] = "executor_mask_v1"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ, PYTHONPATH=str(ROOT / "src"), MUJOCO_GL="egl", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    result = {
        "scope": "paired feedback architecture pilot; unchanged mastery acceptance gates",
        "steps_per_variant_seed": args.steps,
        "seeds": args.seeds,
        "stages": args.stages,
        "initialization": "scratch" if args.source is None else str(args.source.resolve()),
        "executor_action_mask": suite.get("training_policy") == "executor_mask_v1",
        "complete": False,
        "safety_promoted": False,
        "runs": [],
    }
    for variant in args.variants:
        orientation, velocity = VARIANTS[variant]
        spec = copy.deepcopy(suite)
        spec["config"].update(feedback_orientation_error=orientation, feedback_reference_velocity=velocity)
        suite_path = args.output / f"{variant}-suite.json"
        suite_path.write_text(json.dumps(spec, indent=2) + "\n")
        teacher_directory = args.output / f"{variant}-preflight"
        teacher_path = teacher_directory / "report.json.gz"
        prefix = [sys.executable, "-m", "emet.rl.manipulation.mastery"]
        with (args.output / f"{variant}.log").open("x") as log:
            subprocess.run(
                [
                    *prefix,
                    "preflight",
                    "--suite",
                    str(suite_path),
                    "--stages",
                    *args.stages,
                    "--output",
                    str(teacher_directory),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            for seed in args.seeds:
                destination = args.output / f"{variant}-seed-{seed}"
                command = [
                    *prefix,
                    "train",
                    "--suite",
                    str(suite_path),
                    "--stages",
                    *args.stages,
                    "--preflight",
                    str(teacher_path),
                    "--regression-receipt",
                    str(args.regression_receipt.resolve()),
                    "--output",
                    str(destination),
                    "--steps",
                    str(args.steps),
                    "--seed",
                    str(seed),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--arm-std",
                    str(args.arm_std),
                    "--eval-every",
                    str(args.steps),
                    "--wandb-mode",
                    "disabled",
                ]
                if args.source:
                    command += ["--initialize-from", str(args.source.resolve()), "--freeze-normalization"]
                print(f"Training {variant}, seed {seed}, {args.steps} transitions", flush=True)
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                result["runs"].append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "command": command,
                        "initial": json.loads((destination / "step-0/summary.json").read_text()),
                        "final": json.loads((destination / "last/summary.json").read_text()),
                    }
                )
                (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    result["complete"] = True
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
