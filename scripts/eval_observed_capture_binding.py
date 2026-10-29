#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Live RGBD stream -> bounded surface reassociation -> actual scene_tasks tool.

Discovery and optional geometry diagnostics, not full task acceptance. Support
and known-model object candidates are measured separately; no observed execution
provider is installed. Caliban supplies semantic initialization only; later tool calls read measured camera data without model inference. Run via
emet jobs with an available software renderer when the local GPU is reserved.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from emet.agent.tools import get_tools
from emet.config.sim_launch_config import load_sim_launch_config_from_path
from emet.controller.manipulation.encoder_capture import read_local_sim_motor_capture
from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.geometry_tracking import GeometryTracker
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.sim_robot_capture import read_synchronized_robot_capture
from emet.controller.manipulation.stretch_encoder_kinematics import estimate_stretch_ee_from_encoders
from emet.controller.task.tamp.observed_capture import bind_observed_capture_source, capture_arrays
from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider
from emet.eval.observed_readiness import assess_live_readiness, capture_metadata
from emet.eval.sim_eval_session import (
    connect_benchmark_robot,
    launch_benchmark_sim_server,
    terminate_benchmark_sim_server,
)
from emet.llms.openai_vllm_client import OpenaiVLLMClient
from emet.mapping.scene_graph.open_vocab_scene_graph import ObjectObservation
from emet.memory.surface_reacquisition import reacquire_surface
from emet.memory.vlm_region_grounding import select_supported_region, select_vlm_region
from emet.motion.observed_support import fit_observed_support, support_mask_from_seed
from emet.robots.stretch import STRETCH_ARM_CHAIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query", default="red cylinder")
    parser.add_argument(
        "--geometry-model", type=Path, help="Explicit known-model debug prior JSON; does not enable execution."
    )
    parser.add_argument(
        "--synchronized-robot-model",
        type=Path,
        help="Robot-only Stretch MJCF for simulator co-sampling checks; does not enable execution.",
    )
    parser.add_argument(
        "--support-query", default="", help="Optionally ground and measure a support patch; does not enable execution."
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--execution-readiness",
        action="store_true",
        help="Audit live execution prerequisites without attempting manipulation.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    config = load_sim_launch_config_from_path(str(args.sim))
    if config.no_cameras:
        parser.error("this diagnostic requires cameras")
    os.environ["EMET_PHYSICAL_EXECUTION"] = "1"
    os.environ["EMET_PHYSICAL_AUDIT"] = str(out / "actuation.jsonl")
    model = None
    robot_model = (
        mujoco.MjModel.from_xml_path(str(args.synchronized_robot_model.resolve()))
        if args.synchronized_robot_model
        else None
    )
    if args.geometry_model:
        spec = json.loads(args.geometry_model.read_text())
        if spec.get("schema_version") != 1 or spec.get("construction") != "known_model_prior":
            parser.error("geometry model must explicitly declare schema 1 known_model_prior")
        model = PrimitiveModel(spec["kind"], tuple(spec["dimensions_m"]))
    result = {
        "scope": "live ZMQ discovery, support and geometry diagnostics; no task execution acceptance",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_success_assessed": False,
        "queries": [],
        "geometry_tracking": [],
        "robot_alignment": [],
        "motor_feedback": [],
        "geometry_input_tier": "known_model_prior" if model is not None else "not_configured",
        "geometry_model_sha256": hashlib.sha256(args.geometry_model.read_bytes()).hexdigest()
        if model is not None
        else None,
        "vlm_calls": [],
    }

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2))

    save()
    client = None
    with (out / "server.log").open("w") as log:
        server = launch_benchmark_sim_server(config, repo=repo, cpu_only=True, server_stderr=log)
        try:
            client = connect_benchmark_robot(config, server.port_offset, enable_rerun_server=False)
            client.head_to(-1.57, -0.7, blocking=True)
            time.sleep(2.0)
            observation = client.get_observation()
            rgb, depth, world = capture_arrays(observation)
            Image.fromarray(rgb).save(out / "reference.png")
            np.savez_compressed(
                out / "reference.npz",
                rgb=rgb,
                depth=depth,
                world=world,
                K=observation.camera_K,
                pose=observation.camera_pose,
                metadata_json=json.dumps(capture_metadata(observation)),
            )
            provider = RgbdTaskProvider()
            ticket = provider.begin_frame(observation.received_monotonic, observation.image_timing)
            vlm = OpenaiVLLMClient(base_url=args.base_url, model=args.model, max_tokens=256)
            vlm._openai = vlm._openai.with_options(timeout=90, max_retries=0)

            class TimedClient:
                def __call__(self, *a, **kw):
                    try:
                        return vlm.generate_multimodal(*a, **kw)
                    finally:
                        result["vlm_calls"].append(vlm.last_call_metrics)
                        save()

            _, instances, audit = select_supported_region(
                rgb,
                depth,
                args.query,
                "",
                client=TimedClient(),
                min_depth=0.25,
                max_depth=2.5,
                strategy="depth_candidates",
                search_padding=50,
                world_xyz=world,
            )
            result["initialization"] = audit
            save()
            if not audit["valid"]:
                result["reacquisition_passed"] = False
                save()
                return 1
            mask = instances == 0
            np.save(out / "reference-mask.npy", mask)
            ys, xs = np.where(mask)
            reference = ObjectObservation(
                mask=mask,
                bbox_xyxy=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]),
                rgb_crop=rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1],
                points_3d=world[mask],
                points_rgb=None,
                camera_pose=observation.camera_pose,
                label=args.query,
                score=1.0,
                timestep=1,
            )
            provider.update([reference], frame_ticket=ticket, reference_rgb=rgb, reference_world_xyz=world)

            class CaptureClient:
                def get_observation(self):
                    capture = client.get_observation()
                    current_rgb, current_depth, current_world = capture_arrays(capture)
                    np.savez_compressed(
                        out / f"capture-{len(result['queries'])}.npz",
                        rgb=current_rgb,
                        depth=current_depth,
                        world=current_world,
                        K=capture.camera_K,
                        pose=capture.camera_pose,
                        metadata_json=json.dumps(capture_metadata(capture)),
                    )
                    return capture

            bind_observed_capture_source(provider, CaptureClient(), "observed")
            tools = {
                t.name: t
                for t in get_tools(
                    {
                        "robot": client,
                        "manip_mode": "physical",
                        "tamp_inputs": "observed",
                        "observed_tamp_provider": provider,
                    }
                )
            }
            geometry_tracker = (
                GeometryTracker(model, object_ref="semantic-target", max_evaluations=12000)
                if model is not None
                else None
            )
            for index in range(args.samples):
                started = time.perf_counter()
                response = json.loads(tools["scene_tasks"].func())
                result["queries"].append(
                    {
                        "index": index,
                        "seconds": time.perf_counter() - started,
                        "reacquisition_error": provider.last_reacquisition_error,
                        "response": response,
                        "diagnostics": provider.last_reacquisition_diagnostics,
                    }
                )
                if geometry_tracker is not None:
                    current = client.get_observation()
                    current_rgb, current_depth, current_world = capture_arrays(current)
                    stamp = (current.image_timing or {}).get("head_cam_left", {}).get("timestamp_ns")
                    capture_id = str(stamp) if type(stamp) is int and stamp > 0 else ""
                    geometry_started = time.perf_counter()
                    try:
                        surface = reacquire_surface(
                            rgb, world, mask, current_rgb, current_depth, current_world, min_depth=0.25, max_depth=2.5
                        )
                        fit = geometry_tracker.update(
                            surface.points,
                            observation_id=capture_id,
                            received_monotonic=current.received_monotonic,
                            scene_epoch="stationary-diagnostic",
                            now=time.monotonic(),
                        )
                        geometry = fit.as_dict()
                        np.savez_compressed(
                            out / f"geometry-capture-{index}.npz",
                            rgb=current_rgb,
                            depth=current_depth,
                            world=current_world,
                            mask=surface.mask,
                            K=current.camera_K,
                            pose=current.camera_pose,
                            metadata_json=json.dumps(capture_metadata(current)),
                        )
                    except (ObservationUnavailable, ValueError) as error:
                        geometry = {"state": "unknown", "reason": str(error), "execution_qualified": False}
                    result["geometry_tracking"].append(
                        {"index": index, "seconds": time.perf_counter() - geometry_started, **geometry}
                    )
                if robot_model is not None:
                    if geometry_tracker is None:
                        current = client.get_observation()
                    aligned = None
                    try:
                        aligned = read_synchronized_robot_capture(
                            current,
                            robot_model,
                            ee_body=STRETCH_ARM_CHAIN.ee_body,
                            gripper_bodies=STRETCH_ARM_CHAIN.gripper_bodies,
                        )
                        alignment = {
                            "state": "passed",
                            "source": aligned.source,
                            "execution_qualified": aligned.execution_qualified,
                            "observation_id": aligned.capture.observation_id,
                            "capture_span_s": aligned.capture.latest - aligned.capture.earliest,
                            "capture_age_s": time.monotonic() - aligned.capture.earliest,
                            "alignment_error_m": aligned.alignment_error_m,
                            "alignment_error_rad": aligned.alignment_error_rad,
                            "scope": "ideal simulator shared sample, not hardware calibration",
                            "world_from_ee": aligned.robot_after.world_from_ee.tolist(),
                        }
                    except (ObservationUnavailable, ValueError) as error:
                        alignment = {"state": "unknown", "reason": str(error)}
                    result["robot_alignment"].append({"index": index, **alignment})
                    try:
                        motors = read_local_sim_motor_capture(current)
                        estimated = estimate_stretch_ee_from_encoders(current, robot_model)
                        motor_feedback = {
                            "state": "passed",
                            "source": "simulator_stretch_motor_encoders",
                            "observation_id": motors.capture.observation_id,
                            "positions": motors.as_dict(),
                            "units": motors.units,
                            "world_from_ee_estimate": estimated.world_from_ee.tolist(),
                            "inferred_joint_names": list(estimated.inferred_joint_names),
                            "gripper_geometry_qualified": estimated.gripper_geometry_qualified,
                            "execution_qualified": estimated.execution_qualified,
                            "scope": "motor readouts only; no passive joints, EE or gripper-geometry qualification",
                        }
                        if aligned is not None:
                            truth = aligned.robot_after.world_from_ee
                            position_error = float(np.linalg.norm(truth[:3, 3] - estimated.world_from_ee[:3, 3]))
                            rotation_delta = truth[:3, :3].T @ estimated.world_from_ee[:3, :3]
                            angle_error = float(np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1)))
                            motor_feedback["debug_fk_error_m"] = position_error
                            motor_feedback["debug_fk_error_rad"] = angle_error
                            if position_error > 1e-6 or angle_error > 1e-6:
                                motor_feedback.update(state="failed", reason="encoder_fk_calibration_mismatch")
                    except (ObservationUnavailable, ValueError) as error:
                        motor_feedback = {"state": "unknown", "reason": str(error)}
                    result["motor_feedback"].append({"index": index, **motor_feedback})
                save()
                time.sleep(0.2)
            result["reacquisition_passed"] = all(
                q["reacquisition_error"] is None and q["response"]["status"] == "ok" for q in result["queries"]
            )
            # Independent support grounding must not consume the object reference
            # lifetime before its bounded refresh diagnostic. Preserve both ages.
            support_patch, support_color = None, None
            if args.support_query:
                support_observation = client.get_observation()
                support_rgb, _, _ = capture_arrays(support_observation)
                selection, support_audit = select_vlm_region(
                    support_rgb, args.support_query, "", client=TimedClient(), purpose="placement_surface"
                )
                result["support_initialization"] = support_audit
                save()
                if selection.get("verified") is True:
                    try:
                        support_mask = support_mask_from_seed(
                            support_observation, selection, measurement_margin_m=0.005
                        )
                        np.save(out / "support-mask.npy", support_mask)
                        support_color = np.median(support_rgb[support_mask].astype(float), axis=0)
                        support_patch = fit_observed_support(
                            support_observation,
                            support_mask,
                            observation_id="support-reference",
                            measurement_margin_m=0.005,
                        )
                        result["support_reference"] = {
                            "bounds": support_patch.bounds.tolist(),
                            "proposal_source": "semantic_seeded_plane",
                            "scope": "measured footprint candidate; 5mm admission margin is not hardware-calibrated",
                        }
                    except RuntimeError as error:
                        result["support_geometry_error"] = str(error)
                save()
            if args.execution_readiness:
                capture = client.get_observation()
                audit_start = (out / "actuation.jsonl").read_bytes() if (out / "actuation.jsonl").exists() else b""
                result["execution_readiness"] = assess_live_readiness(
                    observation=capture,
                    reference_rgb=rgb,
                    reference_mask=mask,
                    reference_world=world,
                    support_reference=support_patch,
                    support_color=support_color,
                    tools=tools,
                    refreshes=[
                        q["reacquisition_error"] is None and q["response"]["status"] == "ok" for q in result["queries"]
                    ],
                )
                audit_end = (out / "actuation.jsonl").read_bytes() if (out / "actuation.jsonl").exists() else b""
                result["execution_readiness"]["audit_unchanged_during_admission_checks"] = audit_start == audit_end
            result["diagnostic_passed"] = result["reacquisition_passed"]
            if args.execution_readiness:
                readiness = result["execution_readiness"]
                result["diagnostic_passed"] = bool(
                    result["diagnostic_passed"]
                    and readiness["audit_unchanged_during_admission_checks"]
                    and readiness["gates"]["default_execution_boundary"]["state"] == "passed"
                    and readiness["gates"]["live_rgbd"]["state"] == "passed"
                )
            if args.support_query:
                result["support_diagnostic_passed"] = bool(
                    support_patch is not None
                    and (
                        not args.execution_readiness
                        or result["execution_readiness"]["gates"]["support_grounding"]["state"] == "candidate"
                    )
                )
                result["diagnostic_passed"] = bool(result["diagnostic_passed"] and result["support_diagnostic_passed"])
            if geometry_tracker is not None:
                result["geometry_diagnostic_passed"] = all(
                    row["state"] == "candidate" for row in result["geometry_tracking"]
                )
                result["diagnostic_passed"] = bool(result["diagnostic_passed"] and result["geometry_diagnostic_passed"])
            if robot_model is not None:
                result["robot_alignment_passed"] = all(row["state"] == "passed" for row in result["robot_alignment"])
                result["diagnostic_passed"] = bool(result["diagnostic_passed"] and result["robot_alignment_passed"])
                result["motor_feedback_passed"] = all(row["state"] == "passed" for row in result["motor_feedback"])
                result["diagnostic_passed"] = bool(result["diagnostic_passed"] and result["motor_feedback_passed"])
            save()
            return 0 if result["diagnostic_passed"] else 1
        finally:
            try:
                if client is not None:
                    client.stop()
            finally:
                terminate_benchmark_sim_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
