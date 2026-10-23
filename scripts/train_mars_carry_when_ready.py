"""One-shot Mars carry PPO supervisor: simulator only, no retries or promotion.

Wait for source-bound teacher qualification and regression tests, then start one
plain PPO run. Invoke with the repository .venv Python. Defaults target this
workstation's two stable CPU cores; override --cpus on a different machine.
"""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carry-report", type=Path, default=Path("logs/manipulation/carry-qualification-v3/report.json")
    )
    parser.add_argument(
        "--pick-place-report",
        type=Path,
        default=Path("logs/manipulation/carry-pick-place-qualification-v3/report.json"),
    )
    parser.add_argument(
        "--regression-xml", type=Path, help="fresh output path for supervisor-run pytest, not existing evidence"
    )
    parser.add_argument("--qualification-pid-file", type=Path, default=Path("/tmp/mars-carry-qualification.pid"))
    parser.add_argument("--output", type=Path, default=Path("logs/manipulation/carry-ppo-v1"))
    parser.add_argument("--cpus", default="14,15")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--timeout-hours", type=float, default=8)
    return parser.parse_args(argv)


def check_regressions(path):
    from emet.rl.regression_gate import run_regressions

    return run_regressions(path)


def training_command(args, *, root=ROOT):
    return [
        str(root / ".venv/bin/python"),
        "-u",
        "-m",
        "emet.rl.manipulation",
        "train",
        "--robot",
        "innate_mars",
        "--scenario",
        "carry",
        "--action-mode",
        "direct_velocity",
        "--contact-cone",
        "elliptic",
        "--contact-impratio",
        "3",
        "--balanced-grasp-setup",
        "--tamp-carry-trajectories",
        "--payload-object",
        "object1",
        "--payload-mass-kg",
        ".05",
        "--grasp-lift-height",
        ".045",
        "--support-half-width",
        ".04",
        "--architecture",
        "history",
        "--randomization-scale",
        ".05",
        "--target-scale",
        "1",
        "--carry-push-scale",
        ".02",
        "--carry-push-probability",
        ".25",
        "--carry-target-mode",
        "mixed",
        "--retarget-probability",
        ".25",
        "--base-xy-tolerance",
        ".005",
        "--base-heading-tolerance",
        ".04",
        "--hand-position-tolerance",
        ".005",
        "--hand-orientation-tolerance",
        ".05",
        "--horizon",
        "20",
        "--goal-gated-settling",
        "--discount",
        ".9975",
        "--initialization-steps",
        "0",
        "--steps",
        str(args.steps),
        "--envs",
        "2",
        "--seeds",
        str(args.seed),
        "--eval-every",
        "100352",
        "--eval-episodes",
        "20",
        "--learning-rate",
        ".0001",
        "--final-learning-rate",
        ".00001",
        "--wandb-mode",
        args.wandb_mode,
        "--output",
        str(args.output),
        "--carry-readiness-report",
        str(args.carry_report),
        "--pick-place-readiness-report",
        str(args.pick_place_report),
        "--regression-receipt",
        str((args.regression_xml or args.output / "regressions.xml").with_suffix(".receipt.json")),
    ]


def supervise(args):
    if args.steps <= 0 or not 0 < args.timeout_hours <= 24:
        raise ValueError("positive training budget and timeout up to 24 hours required")
    cpus = {int(value) for value in args.cpus.split(",")}
    if not 1 <= len(cpus) <= 2:
        raise ValueError("use one or two explicitly selected CPU cores")
    os.sched_setaffinity(0, cpus)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/mpl", MUJOCO_GL="egl")
    args.output.mkdir(parents=True, exist_ok=True)
    args.regression_xml = args.regression_xml or args.output / "regressions.xml"
    lock = (args.output / "supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _supervise_locked(args)
    finally:
        lock.close()


def _supervise_locked(args):
    def status(state, **details):
        path = args.output / "supervisor.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"state": state, "updated_unix": time.time(), **details}, indent=2) + "\n")
        temporary.replace(path)

    try:
        if (args.output / "training.log").exists():
            raise RuntimeError("training was already attempted; choose a fresh output, no automatic retry")
        pid = int(args.qualification_pid_file.read_text())
        status("waiting_for_qualification", qualification_pid=pid)
        deadline = time.monotonic() + args.timeout_hours * 3600
        while True:
            reports = [
                json.loads(p.read_text()) if p.exists() else {} for p in (args.carry_report, args.pick_place_report)
            ]
            if all(r.get("complete") for r in reports):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("qualification timeout")
            if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError("qualification process exited before both complete reports")
            time.sleep(30)
        from emet.rl.manipulation.carry_benchmark import canonical_digests, default_config, require_training_readiness
        from emet.rl.manipulation.env import ManipulationEnv

        config = default_config()
        probe = ManipulationEnv(config)
        try:
            fingerprint = probe.fingerprint
        finally:
            probe.close()
        require_training_readiness(reports[0], reports[1], config, fingerprint, **canonical_digests())
        check_regressions(args.regression_xml or args.output / "regressions.xml")
        for proc in psutil.process_iter(["cmdline"]):
            try:
                command = proc.info["cmdline"] or []
                if "emet.rl.manipulation" in command and "train" in command:
                    raise RuntimeError("another manipulation training job is already active")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        command = training_command(args)
        status("training", command=command)
        with (args.output / "training.log").open("x") as log:
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        status(
            "training_complete_unqualified",
            detail="paired frozen evaluation and public TAMP integration still required",
        )
    except Exception as error:
        status("stopped_without_retry", error=str(error))
        raise


if __name__ == "__main__":
    os.chdir(ROOT)
    supervise(parse_args())
