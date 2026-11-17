#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Live ZMQ camera producer while a tool consumer pauses; no transfer acceptance.

Uses an explicit operator colour seed and known cylinder model. No simulator
object pose, segmentation, contact or passive-joint state enters the tracker.
Camera world localization is still simulator debug evidence. This is a source
cadence diagnostic, not collision qualification or observed task success.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from emet.config.sim_launch_config import load_sim_launch_config_from_path
from emet.controller.manipulation.continuous_rgbd import ContinuousRgbdSource
from emet.controller.manipulation.encoder_capture import read_local_sim_motor_capture
from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.multiview_rgbd_tracking import MultiViewRgbdTracker, SynchronizedRgbdView
from emet.controller.manipulation.rgbd_capture_geometry import RgbdCaptureGeometry, SimGeometryProfile
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.eval.sim_eval_session import (
    connect_benchmark_robot,
    launch_benchmark_sim_server,
    terminate_benchmark_sim_server,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumer-pause-s", type=float, default=3.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--head-tilt", type=float, default=-0.7)
    parser.add_argument(
        "--robot-only-model",
        type=Path,
        help="Bind same-capture motor-only robot geometry; requires --geometry-profile.",
    )
    parser.add_argument(
        "--geometry-profile",
        type=Path,
        help="Explicit simulation geometry/error assumptions JSON; requires --robot-only-model.",
    )
    parser.add_argument(
        "--rgb-codec",
        choices=("jpeg", "png"),
        default="jpeg",
        help="Explicit wire RGB profile; PNG preserves sensor pixels. Depth transport is unchanged.",
    )
    parser.add_argument(
        "--registration-audit",
        action="store_true",
        help="Archive original sensor grids/calibration alongside the ordinary wire observation.",
    )
    args = parser.parse_args()
    if bool(args.robot_only_model) != bool(args.geometry_profile):
        parser.error("--robot-only-model and --geometry-profile must be supplied together")
    profile = SimGeometryProfile(**json.loads(args.geometry_profile.read_text())) if args.geometry_profile else None
    robot_model = mujoco.MjModel.from_xml_path(str(args.robot_only_model.resolve())) if args.robot_only_model else None
    if (
        not np.isfinite([args.consumer_pause_s, args.head_tilt]).all()
        or args.consumer_pause_s <= 1.5
        or args.rounds < 1
    ):
        parser.error("require a finite consumer pause >1.5 s, finite tilt and positive rounds")
    repo, out = Path(__file__).resolve().parents[1], args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = load_sim_launch_config_from_path(str(args.sim.resolve()))
    if config.no_cameras or config.robot != "stretch":
        parser.error("requires simulated Stretch cameras")
    os.environ["EMET_SIM_SYNCHRONIZED_RGBD"] = "1"
    os.environ["EMET_SIM_RGB_CODEC"] = args.rgb_codec
    os.environ["EMET_SIM_RGBD_PROVENANCE"] = "1" if args.registration_audit else "0"
    os.environ["EMET_PHYSICAL_EXECUTION"] = "1"
    os.environ["EMET_PHYSICAL_AUDIT"] = str(out / "actuation.jsonl")
    result = {
        "schema_version": 1,
        "scope": "stationary live head-camera cadence diagnostic; no transfer attempted",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "implementation_sha256": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                repo / "src/emet/controller/manipulation/continuous_rgbd.py",
                repo / "src/emet/controller/manipulation/multiview_rgbd_tracking.py",
                repo / "src/emet/core/rgbd_registration.py",
                repo / "src/emet/simulation/rgb_encoding.py",
                repo / "src/emet/simulation/mujoco_server_stretch.py",
                repo / "src/emet/simulation/stretch_mujoco/mujoco_server_camera_manager.py",
            ]
        },
        "sim_config_sha256": hashlib.sha256(args.sim.read_bytes()).hexdigest(),
        "task_success_assessed": False,
        "execution_qualified": False,
        "robot_interference_guard": "not installed in this candidate-source diagnostic",
        "registration_audit": args.registration_audit,
        "rgb_codec": args.rgb_codec,
        "camera_pose_source": "simulator_localization_debug",
        "seed": "operator red appearance interior; explicit known cylinder radius .02 halfheight .04 m",
        "consumer_pause_s": args.consumer_pause_s,
        "head_pose": [-1.57, args.head_tilt],
        "geometry_profile": json.loads(args.geometry_profile.read_text()) if profile else None,
        "visual_frames": [],
        "rounds": [],
        "captures": [],
        "passed": False,
    }

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    client, source, server = None, None, None
    inputs = deque(maxlen=8)
    registrations = {}
    with (out / "server.log").open("w") as log:
        try:
            server = launch_benchmark_sim_server(config, repo=repo, cpu_only=True, server_stderr=log)
            client = connect_benchmark_robot(config, server.port_offset, enable_rerun_server=False)
            client.head_to(-1.57, args.head_tilt, blocking=True)
            time.sleep(2.0)

            def read():
                observation = client.get_observation()
                if observation is None:
                    return None
                motor = read_local_sim_motor_capture(observation)
                rgb, depth, world = capture_arrays(observation)
                geometry = (
                    RgbdCaptureGeometry.from_stretch(observation, robot_model, profile=profile, now=time.monotonic())
                    if profile
                    else None
                )
                view = SynchronizedRgbdView(rgb, depth, world, motor.capture, geometry=geometry)
                if not inputs or inputs[-1].capture.observation_id != motor.capture.observation_id:
                    inputs.append(view)
                    if args.registration_audit:
                        if observation.rgbd_registration is None:
                            raise RuntimeError("rgbd_registration_unavailable")
                        registrations[motor.capture.observation_id] = {
                            "registration": observation.rgbd_registration,
                            "camera_K": np.asarray(observation.camera_K).tolist(),
                            "camera_pose": np.asarray(observation.camera_pose).tolist(),
                            "image_timing": {
                                key: observation.image_timing["head_cam_left"][key]
                                for key in ("clock_id", "capture_interval_ns", "depth_capture_interval_ns")
                            },
                            "camera_pose_source": "simulator_localization_debug",
                        }
                        retained = {v.capture.observation_id for v in inputs}
                        for key in tuple(registrations):
                            if key not in retained:
                                del registrations[key]
                return {"head": view}

            initial = read()
            if initial is None:
                raise RuntimeError("initial_observation_unavailable")
            ref = initial["head"]
            c = ref.rgb.astype(float)
            mask = binary_erosion(
                (c[..., 0] > 2 * c[..., 1])
                & (c[..., 0] > 2 * c[..., 2])
                & (c[..., 0] > 50)
                & np.isfinite(ref.depth)
                & (ref.depth > 0)
            )
            np.savez_compressed(out / "reference.npz", rgb=ref.rgb, depth=ref.depth, world=ref.world, mask=mask)
            Image.fromarray(ref.rgb).save(out / "reference.png")
            tracker = MultiViewRgbdTracker(
                PrimitiveModel("cylinder", (0.02, 0.04)),
                reference_view="head",
                reference=ref,
                mask=mask,
                scene_epoch="stationary-live-source",
                object_ref="operator-red-instance",
                component_policy="complete_prior_union",
                appearance_model="neutral_specular",
                allow_clipped_views=True,
            )
            source = ContinuousRgbdSource(tracker, initial_views=initial, read_views=read).start()
            for _ in range(args.rounds):
                barrier, before = time.monotonic(), source.processed_captures
                time.sleep(args.consumer_pause_s)
                after_pause_count = source.processed_captures
                current = source.snapshot()
                first, last = source.collect(after=barrier, min_span_s=0.2, timeout_s=2.0)
                if profile:
                    for record in (first, last):
                        measured = dict(record.views)["head"]
                        frame = measured.geometry.visual_frame(
                            record.fit, robot_model=robot_model, now=time.monotonic()
                        )
                        result["visual_frames"].append(
                            {
                                "observation_id": frame.capture.observation_id,
                                "source": "same_capture_rgbd_and_eight_motors",
                                "motor_count": len(frame.robot_before.joints),
                                "gripper_volumes": len(frame.robot_before.gripper_bounds),
                                "position_error_m": frame.position_error_m,
                                "rotation_error_rad": frame.rotation_error_rad,
                                "capture_alignment": "ideal_shared_simulator_snapshot",
                                "execution_qualified": False,
                            }
                        )
                result["rounds"].append(
                    {
                        "processed_during_pause": after_pause_count - before,
                        "processed_during_collection": source.processed_captures - after_pause_count,
                        "snapshot_capture_id": current.fit.capture.observation_id,
                        "snapshot_capture_age_s": time.monotonic() - current.fit.capture.earliest,
                        "collected_ids": [v.fit.capture.observation_id for v in (first, last)],
                        "collected_intervals": [[v.fit.capture.earliest, v.fit.capture.latest] for v in (first, last)],
                        "collection_after": barrier,
                    }
                )
                save()
            result["passed"] = all(r["processed_during_pause"] >= 2 for r in result["rounds"])
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error))
        finally:
            save()  # Preserve the control outcome even if cleanup also fails.
            reader_stopped = True
            if source is not None:
                try:
                    source.close()
                except RuntimeError as error:
                    result.update(shutdown_error=str(error), passed=False)
                    # Kill transport server first to unblock a stuck reader;
                    # do not close the shared client while it is still in use.
                    if server is not None:
                        try:
                            terminate_benchmark_sim_server(server)
                            server = None
                        except Exception as terminate_error:
                            result["server_shutdown_error"] = str(terminate_error)
                    try:
                        source.close(timeout_s=5.0)
                    except RuntimeError as retry_error:
                        reader_stopped = False
                        result["reader_shutdown_error"] = str(retry_error)
                result["processed_captures"] = source.processed_captures
                result["duplicate_packets"] = source.duplicate_packets
                result["captures"] = [
                    {
                        "observation_id": r.fit.capture.observation_id,
                        "earliest": r.fit.capture.earliest,
                        "latest": r.fit.capture.latest,
                        "received_monotonic": r.fit.capture.received_monotonic,
                        "processing_s": r.fit.processing_s,
                    }
                    for r in source.history()
                ]
                result["view_rejections"] = dict(source.tracker.last_view_rejections)
            if client is not None and reader_stopped:
                try:
                    client.stop()
                except Exception as error:
                    result.update(client_shutdown_error=str(error), passed=False)
            if server is not None:
                try:
                    terminate_benchmark_sim_server(server)
                except Exception as error:
                    result.update(server_shutdown_error=str(error), passed=False)
            result["input_captures"] = []
            save()
            for index, v in enumerate(inputs if reader_stopped else ()):
                metadata = {
                    "observation_id": v.capture.observation_id,
                    "earliest": v.capture.earliest,
                    "latest": v.capture.latest,
                    "received_monotonic": v.capture.received_monotonic,
                    "calibration_id": v.capture.calibration_id,
                }
                np.savez_compressed(
                    out / f"input-{index:03}.npz",
                    rgb=v.rgb,
                    depth=v.depth,
                    world=v.world,
                    metadata_json=json.dumps(metadata),
                )
                result["input_captures"].append(metadata)
                if v.capture.observation_id in registrations:
                    (out / f"registration-{index:03}.json").write_text(
                        json.dumps(registrations[v.capture.observation_id]) + "\n"
                    )
            save()
    print(json.dumps({k: result.get(k) for k in ("passed", "error", "processed_captures", "rounds")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
