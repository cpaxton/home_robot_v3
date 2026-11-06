#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Save a bounded task-independent RGB-D history; no detector or answer model."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from probe_grapheqa_sensor_only import habitat_sensor_config
from run_shared_room_pairs import assert_frozen_source, git

from emet.eval.exploration_history import CollectionBudgetExceeded, RecordedSensor, collect_history
from emet.eval.grapheqa_navigation import ObservedSpaceNavigation, SensorOnlyHabitat
from emet.habitat.config import hm3d_scene_glb_path, questions_csv_path, scene_init_poses_csv_path
from emet.habitat.datasets import load_scene_init_poses
from emet.utils.job_registry import validated_gpu_lock_fd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_binding(index):
    """Only scene/floor enter the renderer; never parse task/answer fields."""
    with questions_csv_path().open(newline="") as stream:
        for qid, row in enumerate(csv.DictReader(stream)):
            if qid == index:
                return row["scene"], int(row["floor"])
    raise ValueError(f"Unknown development question index: {index}")


def run(args):
    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Rendering requires emet jobs --gpu-exclusive --cpu-safe")
    root = Path(__file__).resolve().parents[1]
    revision = git(root, "rev-parse", "HEAD")
    assert_frozen_source(root, revision)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "mode": "task_independent_development_collection_not_active_agent_acceptance",
        "source_commit": revision,
        "driver_sha256": sha256(Path(__file__)),
        "question_id_for_scene_spawn_only": args.question_id,
        "questions_sha256": sha256(questions_csv_path()),
        "spawns_sha256": sha256(scene_init_poses_csv_path()),
        "budgets": {key: getattr(args, key) for key in ("max_primitives", "max_seconds", "max_legs", "leg_distance_m")},
        "initial_yaw_scan_deg": 360,
        "arrival_yaw_scan_deg": 90,
        "sensor_pose": "command_initialized_rgbd_odometry_no_global_pose_or_loop_closure",
        "footprint_radius_m": 0.1,
        "map_resolution_m": 0.05,
        "semantic_sensor": False,
        "navmesh_route_input": False,
        "renderer_collision_filter": "Habitat internal navmesh motor filter, not a planner input",
        "model_calls": 0,
        "scorer_read": False,
        "status": "running",
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    raw_sensor, navigation, sensor = None, None, None
    frame_count = 0
    traveled, previous_xy = 0.0, None

    def record(frame, command):
        nonlocal frame_count, traveled, previous_xy
        if shutil.disk_usage(args.output).free < 20 * 1024**3 + 4 * 1024**2:
            raise OSError("Keep the shared 20 GiB disk reserve")
        prefix = args.output / f"frame-{frame['sequence']:05d}"
        Image.fromarray(frame["rgb"]).save(prefix.with_suffix(".png"))
        np.savez_compressed(
            prefix.with_suffix(".npz"),
            **{key: frame[key] for key in ("depth", "camera_K", "timestamp_ns")},
            # RGB is already stored losslessly in the PNG; no duplicated array.
        )
        row = {key: frame[key] for key in ("sequence", "timestamp_ns", "collided", "odometry")}
        row.update(
            command=command,
            rgb_sha256=sha256(prefix.with_suffix(".png")),
            depth_sha256=sha256(prefix.with_suffix(".npz")),
        )
        with (args.output / "frames.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        frame_count += 1
        if frame["camera_pose"] is None:
            raise RuntimeError(f"Registration rejected; retain frame and stop: {frame['odometry']}")
        xy = frame["camera_pose"][:2, 3]
        if previous_xy is not None:
            traveled += float(np.linalg.norm(xy - previous_xy))
        previous_xy = xy.copy()

    def event(row):
        row.update(timestamp_ns=time.monotonic_ns(), sequence=raw_sensor.last["sequence"])
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    try:
        if shutil.disk_usage(args.output).free < 20 * 1024**3 + 4 * 1024**2:
            raise OSError("Keep the shared 20 GiB disk reserve")
        scene, floor = scene_binding(args.question_id)
        raw_sensor = SensorOnlyHabitat(
            habitat_sensor_config(hm3d_scene_glb_path(scene)), load_scene_init_poses()[(scene, floor)]
        )
        navigation = ObservedSpaceNavigation()
        sensor = RecordedSensor(raw_sensor, record, max_primitives=args.max_primitives, max_seconds=args.max_seconds)
        manifest.update(
            collect_history(sensor, navigation, event, max_legs=args.max_legs, leg_distance_m=args.leg_distance_m)
        )
    except CollectionBudgetExceeded as exc:
        manifest.update(status="budget_exhausted", reason=str(exc))
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if navigation is not None and navigation.pose is not None:
            try:
                obstacles, explored = navigation.map.get_navigation_map()
                np.savez_compressed(
                    args.output / "navigation.npz",
                    obstacles=obstacles.numpy(),
                    explored=explored.numpy(),
                    pose=navigation.pose,
                    grid_origin=navigation.map.grid_origin.numpy(),
                    grid_resolution_m=0.05,
                )
                manifest.update(observed_cells=int(explored.sum()), obstacle_cells=int(obstacles.sum()))
            except Exception as exc:
                manifest.update(status="failed", map_save_error=f"{type(exc).__name__}: {exc}")
        if raw_sensor is not None:
            try:
                raw_sensor.close()
            except Exception as exc:
                manifest.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        manifest.update(
            frame_count=frame_count,
            attempted_primitives=sensor.primitive_count if sensor else 0,
            estimated_camera_path_m=traveled,
            wall_s=time.monotonic() - started,
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", type=int, required=True, help="Existing development scene/spawn index only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-primitives", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=240)
    parser.add_argument("--max-legs", type=int, default=2)
    parser.add_argument("--leg-distance-m", type=float, default=0.75)
    raise SystemExit(run(parser.parse_args()))
