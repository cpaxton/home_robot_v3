#!/usr/bin/env python3
"""Record independent placement and robot controls under one managed job.

Run via `emet jobs run --cpu-safe --gpu-exclusive --need-mib 8000 -- ...`.
Each case retains its log, process outcome, source commit and available footage.
Failure of one case does not prevent recording the other independent cases.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("open_close", "placement", "rby1", "innate_mars"),
                        default=["open_close", "placement", "rby1", "innate_mars"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    records = []
    for case in args.cases:
        dest = args.output / case
        dest.mkdir(parents=True, exist_ok=True)
        common = ["--start-sim", "--manip-mode", "kinematic", "--cpu-only", "--record-mp4",
                  "--video-overlay", "border"]
        env = dict(os.environ, EMET_SIM_NAV_TELEPORT="1", EMET_PLACEMENT_DIAGNOSTICS_DIR=str(dest / "snapshots"))
        if case in {"open_close", "placement"}:
            cmd = [sys.executable, str(ROOT / "scripts/scripted_sim_pick_place.py"), *common,
                   "--sim", "configs/sim/molmospaces_ithor_train_0.yaml", "--object", "bowl", "--receptacle", "dishwasher",
                   "--video-out", str(dest / "third_person.mp4"),
                   "--video-flags", "ASSISTED | joint teleport | latch | GT | base teleport"]
            if case == "open_close":
                cmd += ["--articulation-cycle"]
            else:
                calls = [{"name": "scene_tasks", "arguments": {"object_filter": "bowl", "robot": "rby1"}}]
                for state in ("open", "closed", "open"):
                    calls.append({"name": "set_receptacle_state", "arguments": {"task_ref": "$task_ref", "state": state}})
                calls += [{"name": "plan_pick_place", "arguments": {"task_ref": "$task_ref"}},
                          {"name": "execute_pick_place_plan", "arguments": {"plan_ref": "$plan_ref"}},
                          {"name": "set_receptacle_state", "arguments": {"task_ref": "$task_ref", "state": "closed"}}]
                cmd += ["--tool-calls-json", json.dumps(calls)]
        else:
            cmd = [sys.executable, str(ROOT / "scripts/scripted_tamp_pick_place.py"), *common,
                   "--sim", "configs/sim/default_table_rby1.yaml", "--robot", case,
                   "--skip-oracle", "--figures-dir", str(dest),
                   "--video-flags", "ASSISTED | latch + placement | GT | base teleport"]
        record = {"case": case, "implementation_commit": commit, "command": cmd, "process_exit_code": None,
                  "status": "running", "artifacts": str(dest), "timeout_s": 900}
        records.append(record)
        manifest = args.output / "suite.json"
        manifest.write_text(json.dumps({"schema_version": 1, "cases": records}, indent=2) + "\n")
        with (dest / "run.log").open("w") as log:
            try:
                # timeout utility also terminates simulator descendants in its process group.
                result = subprocess.run(["timeout", "--kill-after=30", "900", *cmd], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
                record.update(process_exit_code=result.returncode, status="finished")
            except OSError as exc:
                record.update(status="launch_error", error=str(exc))
        manifest.write_text(json.dumps({"schema_version": 1, "cases": records}, indent=2) + "\n")
    return int(any(r["process_exit_code"] != 0 for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
