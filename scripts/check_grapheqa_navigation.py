#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Qualify sensor-only movement before active GraphEQA inference.

No model/detector is loaded. Scene/spawn are renderer-only; routing and arrival
use registered RGB-D and local odometry. This is not an EQA score.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from probe_grapheqa_sensor_only import habitat_sensor_config
from run_shared_room_pairs import assert_frozen_source, git

from emet.eval.grapheqa_navigation import HabitatMotionExecutor, ObservedSpaceNavigation, SensorOnlyHabitat
from emet.habitat.config import hm3d_scene_glb_path
from emet.habitat.datasets import load_hmeqa_questions, load_scene_init_poses
from emet.utils.job_registry import validated_gpu_lock_fd


def run(args):
    if validated_gpu_lock_fd() is None:
        raise RuntimeError("Run through emet jobs --gpu-exclusive; rendering uses the shared GPU")
    root = Path(__file__).resolve().parents[1]
    revision = git(root, "rev-parse", "HEAD")
    assert_frozen_source(root, revision)
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise RuntimeError("Keep the shared 20 GiB disk reserve")
    args.output.mkdir(parents=True, exist_ok=False)
    question = load_hmeqa_questions()[args.question_id]
    spawn = load_scene_init_poses()[(question.scene, question.floor)]
    sensor = SensorOnlyHabitat(habitat_sensor_config(hm3d_scene_glb_path(question.scene)), spawn)
    nav = ObservedSpaceNavigation()
    result = {
        "status": "running",
        "source_commit": revision,
        "question_id": args.question_id,
        "pose_source": "rgbd_odometry_command_initialized_no_simulator_pose",
        "routing": "observed_floor_shared_grid_search_and_sweep",
        "model_calls": 0,
        "tests": [],
    }

    def save(frame):
        Image.fromarray(frame["rgb"]).save(args.output / f"frame-{frame['sequence']:04d}.png")
        if frame["camera_pose"] is not None:
            np.savez_compressed(
                args.output / f"frame-{frame['sequence']:04d}.npz",
                **{key: frame[key] for key in ("rgb", "depth", "camera_K", "camera_pose", "timestamp_ns")},
            )
        row = {key: frame[key] for key in ("sequence", "timestamp_ns", "collided", "odometry")}
        with (args.output / "frames.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if frame["camera_pose"] is None:
            raise RuntimeError(f"RGB-D tracking failed at frame {frame['sequence']}: {frame['odometry']}")

    try:
        save(sensor.last)
        nav.update(sensor.last)
        for _ in range(9):
            frame = sensor.step("turn_left")
            save(frame)
            nav.update(frame)
        # Bounded floor inspection is real camera motion and fresh depth; it
        # neither fills unknown cells synthetically nor moves the base.
        for _ in range(3):
            frame = sensor.step("look_down")
            save(frame)
            nav.update(frame)
        frontiers = nav.frontiers()
        result["frontiers"] = frontiers
        np.savez_compressed(
            args.output / "navigation.npz",
            obstacles=nav.map.get_navigation_map()[0].numpy(),
            explored=nav.map.get_navigation_map()[1].numpy(),
            pose=nav.pose,
        )
        executor = HabitatMotionExecutor(sensor, nav, save)
        # A short reachable position from the sensed map, independent of labels
        # or task answer. Pick the shortest nontrivial observed route.
        proposals = []
        for point in frontiers:
            route, reason = nav.plan(point, object_target=False)
            if route:
                proposals.append((len(route), point, route))
        if not proposals:
            raise RuntimeError("No observed reachable frontier after bounded floor inspection")
        _, point, route = min(proposals, key=lambda item: item[0])
        # Limit the control gate to the first 0.3 m of this admitted route.
        short = []
        origin = nav.pose[:2].copy()
        for waypoint in route:
            short.append(waypoint)
            if np.linalg.norm(np.asarray(waypoint)[:2] - origin) >= 0.3:
                break
        receipt = executor.execute(short)
        result["tests"].append({"case": "reachable_translation", **receipt})
        if not receipt["success"]:
            raise RuntimeError(f"Reachable route failed: {receipt['reason']}")
        # Object/view behind the camera: only heading must change.
        goal = nav.pose.copy()
        goal[2] += np.pi / 2
        receipt = executor.execute([goal.tolist()])
        result["tests"].append({"case": "turn_to_view", **receipt})
        if not receipt["success"]:
            raise RuntimeError(f"Turn-to-view failed: {receipt['reason']}")
        # The fixed local grid edge is unknown/out of range, never a valid goal.
        count = executor.primitive_count
        route, reason = nav.plan([100, 100, 0], object_target=False)
        blocked = not route and count == executor.primitive_count
        result["tests"].append({"case": "unknown_goal_rejected_before_motion", "success": blocked, "reason": reason})
        if not blocked:
            raise RuntimeError("Unknown goal was admitted")
        result.update(status="passed_navigation_gate", primitive_count=executor.primitive_count)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        sensor.close()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
