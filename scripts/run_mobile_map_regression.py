#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Run one sequential CPU lane of a declared assisted mobile-map regression.

Launch each lane through emet jobs run --no-gpu-exclusive on disjoint safe CPUs.
Failures remain in the manifest and do not stop other independent cases.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from emet.utils.process_tree import popen_session, terminate_process_tree

ROOT = Path(__file__).resolve().parents[1]


def prepare_case(fixture, output, case):
    """Change geometry/mass only in declared initial-condition XML, before load."""
    output.mkdir(parents=True, exist_ok=True)
    for source in fixture.glob("*.xml"):
        (output / source.name).write_bytes(source.read_bytes())
    scale = case.get("object_scale", 1.0)
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not 0.5 <= scale <= 1.5:
        raise ValueError("invalid_fixture_scale")
    xml_path = output / "scene_environment.xml"
    tree = ET.parse(xml_path)
    body = tree.getroot().find(f"worldbody/body[@name='{case['object']}']")
    if body is None:
        raise ValueError("fixture_object_missing")
    for geom in body.findall("geom"):
        geom.set("size", " ".join(str(float(v) * scale) for v in geom.get("size").split()))
        geom.set("mass", str(float(geom.get("mass")) * scale**3))
    tree.write(xml_path, encoding="unicode")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.xml")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "configs/eval/mobile_map_regression_20261117.json")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--case", action="append", help="Select exact case IDs; repeatable.")
    parser.add_argument(
        "--refresh-source", action="store_true", help="Enable post-search source reacquisition in task workers."
    )
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    if not 0 <= args.lane < args.lanes or args.timeout_s <= 0:
        parser.error("invalid lane or timeout")
    spec = json.loads(args.suite.read_text())
    ids = [row["id"] for row in spec["cases"]]
    if spec["schema_version"] != 1 or len(ids) != len(set(ids)) or (args.case and set(args.case) - set(ids)):
        parser.error("invalid suite or case selection")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = args.output / f"lane-{args.lane}.json"
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT / "scripts/eval_mobile_mapped_contact.py",
            ROOT / "src/emet/motion/release.py",
            ROOT / "src/emet/controller/task/tamp/native_revalidation.py",
            ROOT / "src/emet/controller/task/tamp/native_dispatch.py",
            ROOT / "src/emet/controller/task/tamp/native_adapter.py",
            ROOT / "src/emet/controller/task/tamp/native_placement.py",
            ROOT / "src/emet/motion/local_scene_store.py",
        ]
    }

    def save():
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scope": spec["scope"],
                    "source_commit": source,
                    "runtime_sha256": sources,
                    "lane": args.lane,
                    "cases": records,
                },
                indent=2,
            )
            + "\n"
        )

    selected = [row for row in spec["cases"] if not args.case or row["id"] in args.case]
    for index, case in enumerate(selected):
        if index % args.lanes != args.lane:
            continue
        out = args.output / case["id"]
        out.mkdir(parents=True, exist_ok=True)
        row = {
            "id": case["id"],
            "kind": "negative_control" if case.get("fault") else "task",
            "status": "running",
            "case": case,
            "result": None,
        }
        records.append(row)
        save()
        fixture = out / "fixture"
        row["fixture_sha256"] = prepare_case(args.fixture.resolve(), fixture, case)
        config = out / "case.json"
        config.write_text(json.dumps(case.get("config", {}), indent=2) + "\n")
        script = "eval_mapped_native_contact.py" if case.get("fault") else "eval_mobile_mapped_contact.py"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--fixture",
            str(fixture.resolve()),
            "--object",
            case["object"],
            "--output",
            str(out.resolve()),
        ]
        cmd += ["--fault", case["fault"]] if case.get("fault") else ["--case-config", str(config.resolve())]
        if args.refresh_source and not case.get("fault"):
            cmd.append("--refresh-source")
        row["command"] = cmd
        row["runtime_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
        save()
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts"),
            "CUDA_VISIBLE_DEVICES": "",
        }
        with (out / "worker.log").open("w") as log:
            process = popen_session(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                row["exit_code"] = process.wait(timeout=args.timeout_s)
                result = out / "result.json"
                if result.exists():
                    row["result"] = json.loads(result.read_text())
                expected = (
                    (row["result"].get("expectation_passed") if case.get("fault") else row["result"].get("passed"))
                    if row["result"]
                    else False
                )
                row["status"] = "passed" if row["exit_code"] == 0 and expected else "failed"
            except subprocess.TimeoutExpired:
                row["status"] = "timeout"
            finally:
                terminate_process_tree(process)
        save()
    return int(any(row["status"] != "passed" for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
