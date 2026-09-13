#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Offline physics replay views, not agent observations or new task executions.

Use the scene from the trace's frozen source checkout. Run serially under the
same GPU lock as experiments. No physics is stepped and no simulator state is
exposed to a live agent. A manifest records exact trace rows and input hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from emet.eval.manipulation_trace import score_trace


def select_frames(rows, result):
    if not rows:
        raise ValueError("Empty physical trace")
    frames = [("initial", min(range(len(rows)), key=lambda i: abs(rows[i]["sim_time"] - 2.0)))]
    if result.get("physical_pick_success"):
        pick = min(range(len(rows)), key=lambda i: abs(rows[i]["sim_time"] - result["pick_time"]))
        frames.append(("picked", pick))
        contacts = [i for i in range(pick, len(rows)) if rows[i]["gripper_contact"]]
        if contacts:
            frames.append(("last_gripper_contact", contacts[-1]))
    frames.append(("final", len(rows) - 1))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text().splitlines()]
    if not records or records[0].get("schema") != 1:
        parser.error("A schema-1 private physical trace is required")
    rows = records[1:]
    result = score_trace(rows)
    if not result["verified"]:
        parser.error(f"Unverified trace: {result['reason']}")
    if args.output.exists():
        parser.error("Use a fresh output directory")

    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    frames = select_frames(rows, result)
    for _, i in frames:
        if len(rows[i]["qpos"]) != model.nq:
            parser.error("Trace configuration dimensions do not match the supplied scene")
    args.output.mkdir(parents=True)
    manifest = {
        "kind": "offline_qpos_replay_not_agent_view",
        "trace": str(args.trace.resolve()),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "scene": str(args.scene.resolve()),
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "physical_result": result,
        "frames": [],
    }
    renderer = mujoco.Renderer(model, height=480, width=640)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # Show robot visuals, not collision meshes.
    try:
        for stage, i in frames:
            row = rows[i]
            data.qpos[:] = row["qpos"]
            mujoco.mj_forward(model, data)
            target = np.asarray(row["object_pos"])
            for name, elevation, distance in (("overview", -25, 2.0), ("top_down", -90, 1.8), ("object", -25, 0.6)):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = target
                camera.azimuth = 130
                camera.elevation = elevation
                camera.distance = distance
                renderer.update_scene(data, camera=camera, scene_option=option)
                filename = f"{stage}-{name}.png"
                Image.fromarray(renderer.render().copy()).save(args.output / filename)
                manifest["frames"].append(
                    {"file": filename, "row": i, "sim_time": row["sim_time"], "wall_time": row["wall_time"]}
                )
    finally:
        renderer.close()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
