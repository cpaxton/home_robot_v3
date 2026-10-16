#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
"""Offline sensor replay at saved poses, not a navigation success experiment.

Run only through emet jobs --cpu-safe --gpu-exclusive. Save actual RGB/depth/K/
optical pose and RGB differences; never claim reconstructed depth is matched
to an old image without checking the render. No semantic sensor is enabled.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Import full stack before GL initialization, matching runtime preflight.
    import emet_habitat.runner  # noqa: F401
    import habitat_sim.utils.common as utils
    from emet_habitat.simulator import HabitatEQASimulator

    from emet.habitat.coordinates import habitat_agent_pose_from_state, habitat_observation_camera_pose

    header = json.loads((args.episode / "observations_history.jsonl").read_text().splitlines()[0])
    height = header["spawn_record"]["init_pose_snapped"]["y"]
    rows = [json.loads(line) for line in (args.episode / "metadata.jsonl").read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    sim = HabitatEQASimulator.from_scene_id(args.scene, use_hm3d_semantics=False)
    results = []
    try:
        for row in rows:
            x, y, yaw = row["pose_xyt"]
            state = sim._agent.get_state()
            state.position = np.array([x, height, y], dtype=np.float32)
            # Recorded XY omits navmesh elevation changes along the route.
            snapped = sim._sim.pathfinder.snap_point(state.position)
            if np.isfinite(snapped).all():
                state.position = np.array([x, snapped[1], y], dtype=np.float32)
            state.rotation = utils.quat_from_angle_axis(-yaw - np.pi / 2, np.array([0.0, 1.0, 0.0]))
            sim._agent.set_state(state)
            frame = sim.get_frame()
            pose = habitat_observation_camera_pose(
                habitat_agent_pose_from_state(frame.agent_state), agent_state=frame.agent_state, floor_y=height
            )
            old = np.asarray(Image.open(args.episode / row["image"]).convert("RGB"))
            error = np.abs(old.astype(float) - frame.rgb.astype(float))
            name = f"frame-{row['frame_idx']:04d}"
            np.savez_compressed(
                args.output / (name + ".npz"), depth=frame.depth, intrinsics=frame.intrinsics, camera_pose=pose
            )
            Image.fromarray(frame.rgb).save(args.output / (name + ".png"))
            results.append(
                {
                    **row,
                    "rgb": name + ".png",
                    "geometry": name + ".npz",
                    "rgb_mae": float(error.mean()),
                    "rgb_exact": bool(np.array_equal(old, frame.rgb)),
                    "base_height_y": float(state.position[1]),
                }
            )
        (args.output / "replay.json").write_text(
            json.dumps(
                {
                    "source": str(args.episode),
                    "scene": args.scene,
                    "sensor_height_m": 1.5,
                    "camera_tilt_deg": -30,
                    "hfov_deg": 120,
                    "frames": results,
                },
                indent=2,
            )
        )
        print(
            json.dumps(
                {
                    "frames": len(results),
                    "exact": sum(r["rgb_exact"] for r in results),
                    "max_rgb_mae": max(r["rgb_mae"] for r in results),
                }
            ),
            flush=True,
        )
    finally:
        sim.close()


if __name__ == "__main__":
    main()
