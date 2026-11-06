#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Scorer-only renderer replay: separate physical motion from odometry drift.

No controller runs and no privileged state returns to an agent. Replaying the
exact recorded RGB is required before associating simulator motion with a trace.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from collect_exploration_history import scene_binding, sha256
from PIL import Image
from probe_grapheqa_sensor_only import habitat_sensor_config
from run_shared_room_pairs import assert_frozen_source, git

from emet.eval.grapheqa_navigation import SensorOnlyHabitat
from emet.habitat.config import hm3d_scene_glb_path
from emet.habitat.datasets import load_scene_init_poses
from emet.utils.job_registry import validated_gpu_lock_fd


def run(args):
    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Use the managed GPU lock for renderer replay")
    root = Path(__file__).resolve().parents[1]
    revision = git(root, "rev-parse", "HEAD")
    assert_frozen_source(root, revision)
    manifest = json.loads((args.source / "manifest.json").read_text())
    rows = [json.loads(line) for line in (args.source / "frames.jsonl").read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    scene, floor = scene_binding(manifest["question_id_for_scene_spawn_only"])
    sensor = SensorOnlyHabitat(
        habitat_sensor_config(hm3d_scene_glb_path(scene)), load_scene_init_poses()[(scene, floor)]
    )
    result = {
        "scorer_only": True,
        "agent_or_model_calls": 0,
        "source_commit": revision,
        "recorded_source_commit": manifest["source_commit"],
        "frames_sha256": sha256(args.source / "frames.jsonl"),
        "driver_sha256": sha256(Path(__file__)),
        "turn_segments": [],
        "matched_rgb_frames": 0,
        "planned_frames": len(rows),
        "status": "running",
    }
    import magnum as mn

    camera = sensor.sim.get_agent(0).scene_node.subtree_sensors["depth"].render_camera
    ray_errors, center_ray_errors = [], []
    for u, v in ((0, 1), (319, 239), (638, 478)):
        ray = camera.unproject(mn.Vector2i(u, v), False)
        gl_direction = np.asarray(camera.camera_matrix)[:3, :3] @ np.asarray(ray.direction)
        cv_direction = gl_direction * [1, -1, -1]
        cv_direction /= cv_direction[2]
        ray_errors.append(float(np.max(np.abs(cv_direction - np.linalg.inv(sensor.k) @ [u, v, 1]))))
        # Native unproject flips viewport Y with height-v-1, then normalizes
        # integer boundaries. Opposite queries (u,v) and (u+1,v-1) therefore
        # enclose the array pixel centre. Rays are affine at unit depth.
        opposite = camera.unproject(mn.Vector2i(u + 1, v - 1), False)
        center_gl = np.asarray(camera.camera_matrix)[:3, :3] @ (
            (np.asarray(ray.direction) + np.asarray(opposite.direction)) / 2
        )
        center_cv = center_gl * [1, -1, -1]
        center_cv /= center_cv[2]
        center_ray_errors.append(float(np.max(np.abs(center_cv - np.linalg.inv(sensor.k) @ [u, v, 1]))))
    result.update(
        projection_matrix=np.asarray(camera.projection_matrix).tolist(),
        intrinsics=sensor.k.tolist(),
        native_boundary_vs_pixel_center_ray_error=max(ray_errors),
        native_raster_center_max_ray_error=max(center_ray_errors),
    )
    previous_true, previous_estimated, segment = None, None, None
    try:
        for row in rows:
            command = row["command"]
            if command is not None:
                action = command["action"]
                limit = 0.1 if action == "move_forward" else 5.0
                amount = limit if command["amount"] is None else command["amount"]
                if action not in sensor.sim.get_agent(0).agent_config.action_space or not 0 < amount <= limit:
                    raise ValueError("Recorded action exceeds the calibrated primitive contract")
                sensor.sim.get_agent(0).agent_config.action_space[action].actuation.amount = amount
                raw = sensor.sim.step(action)
            else:
                raw = sensor.sim.get_sensor_observations()
            captured = np.asarray(Image.open(args.source / f"frame-{row['sequence']:05d}.png"))
            matched = bool(np.array_equal(captured, raw["rgb"][..., :3]))
            result["matched_rgb_frames"] += int(matched)
            # Privileged state is consumed exclusively by this scorer replay.
            true_xy = np.asarray(sensor.sim.get_agent(0).get_state().position)[[0, 2]]
            estimated_xy = np.asarray(row["odometry"]["camera_pose"])[:2, 3]
            if command is not None and command["action"] in {"turn_left", "turn_right"}:
                if segment is None:
                    segment = {
                        "start_sequence": row["sequence"] - 1,
                        "end_sequence": row["sequence"],
                        "actions": 0,
                        "max_true_base_displacement_m": 0.0,
                        "max_estimated_camera_displacement_m": 0.0,
                    }
                    anchor_true, anchor_estimated = previous_true.copy(), previous_estimated.copy()
                    result["turn_segments"].append(segment)
                segment["actions"] += 1
                segment["end_sequence"] = row["sequence"]
                segment["max_true_base_displacement_m"] = max(
                    segment["max_true_base_displacement_m"], float(np.linalg.norm(true_xy - anchor_true))
                )
                segment["max_estimated_camera_displacement_m"] = max(
                    segment["max_estimated_camera_displacement_m"],
                    float(np.linalg.norm(estimated_xy - anchor_estimated)),
                )
            else:
                segment = None
            previous_true, previous_estimated = true_xy.copy(), estimated_xy.copy()
            with (args.output / "scorer_states.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "sequence": row["sequence"],
                            "rgb_matches_recording": matched,
                            "sim_base_xz_m": true_xy.tolist(),
                        }
                    )
                    + "\n"
                )
        result["status"] = (
            "matched_renderer_replay" if result["matched_rgb_frames"] == len(rows) else "renderer_mismatch"
        )
        # Native unproject transforms far-plane points in float32. This unit-
        # depth tolerance is <.02 pixel at this focal length, well below the
        # .5-pixel convention error, and is unrelated to motion tolerances.
        if max(center_ray_errors) > 1e-4:
            result["status"] = "renderer_calibration_mismatch"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        sensor.close()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "matched_renderer_replay" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
