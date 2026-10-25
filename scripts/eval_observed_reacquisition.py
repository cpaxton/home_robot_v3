#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""CPU-rendered RGBD reacquisition diagnostic with remote semantic initialization.

This exercises fresh measured surface positions and rejection, not native task
execution. Camera calibration comes from the simulator sensor; object/body poses
are never provided to the grounding or reacquisition algorithms. Use emet jobs.
"""

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from emet.controller.task.tamp.observed_frames import ObservedFrameGate
from emet.llms.openai_vllm_client import OpenaiVLLMClient
from emet.memory.surface_reacquisition import reacquire_surface
from emet.memory.vlm_region_grounding import select_supported_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query", default="red cylinder")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--split-surface-normals", action="store_true")
    parser.add_argument("--initial-posture", choices=("home", "stow"), default="home")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if not np.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 10:
        parser.error("--settle-seconds must be in 0..10")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "MuJoCo RGBD surface reacquisition; not native manipulation or full run-agent acceptance",
        "query": args.query,
        "vlm_calls": [],
        "frames": [],
    }

    repo = Path(__file__).resolve().parents[1]
    code_paths = [
        "scripts/eval_observed_reacquisition.py",
        "src/emet/memory/surface_reacquisition.py",
        "src/emet/memory/surface_candidates.py",
        "src/emet/memory/vlm_region_grounding.py",
        "src/emet/llms/openai_vllm_client.py",
        "src/emet/controller/task/tamp/observed_frames.py",
    ]
    result["source_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    result["code_sha256"] = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in code_paths}
    result["scene_sha256"] = hashlib.sha256(args.scene.read_bytes()).hexdigest()
    result["surface_normal_splitting"] = args.split_surface_normals

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2))

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    if args.initial_posture == "stow":
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stow")
        if key < 0:
            raise ValueError("stow control targets unavailable")
        # Issue controller targets and physically settle; never assign the arm's
        # qpos to manufacture a clear view. Head pose is initial camera setup.
        data.ctrl[:] = model.key_ctrl[key]
    result["initial_posture"] = args.initial_posture
    for name, value in [("joint_head_pan", -1.57), ("joint_head_tilt", -0.7)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError("Stretch head joint unavailable: " + name)
        data.qpos[model.jnt_qposadr[jid]] = value
        for aid in np.flatnonzero(model.actuator_trnid[:, 0] == jid):
            data.ctrl[aid] = value
    # Reset poses are not physical observations: movable objects can initially
    # be above their supports. Settle before taking the semantic reference.
    for _ in range(int(args.settle_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    result["settle_seconds"] = args.settle_seconds
    result["reference_sim_time"] = float(data.time)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435i_camera_rgb")
    if camera < 0:
        raise ValueError("RGBD camera unavailable")
    height, width = 240, 320
    f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)
    y, x = np.mgrid[:height, :width]
    gate = ObservedFrameGate()
    capture_id = 0
    with mujoco.Renderer(model, height=height, width=width) as renderer:

        def capture():
            nonlocal capture_id
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            camera_xyz = np.stack(
                [(x - (width - 1) / 2) * depth / f, -(y - (height - 1) / 2) * depth / f, -depth], axis=-1
            )
            world = camera_xyz @ data.cam_xmat[camera].reshape(3, 3).T + data.cam_xpos[camera]
            rgb, depth, world = [np.ascontiguousarray(np.rot90(a, -1)) for a in (rgb, depth, world)]
            capture_id += 1
            received = time.monotonic()
            ticket = gate.begin(
                received,
                {
                    "head_cam_left": {
                        "available": True,
                        "timestamp_ns": capture_id,
                        "source": "mujoco-render-capture",
                        "clock_domain": "diagnostic_capture_sequence",
                    }
                },
            )
            return rgb, depth, world, ticket

        reference_rgb, reference_depth, reference_world, reference_ticket = capture()
        Image.fromarray(reference_rgb).save(args.output / "reference.png")
        np.savez_compressed(
            args.output / "reference.npz", rgb=reference_rgb, depth=reference_depth, world=reference_world
        )
        client = OpenaiVLLMClient(base_url=args.base_url, model=args.model, max_tokens=256)
        client._openai = client._openai.with_options(timeout=90, max_retries=0)

        class TimedClient:
            def __call__(self, *a, **kw):
                try:
                    return client.generate_multimodal(*a, **kw)
                finally:
                    result["vlm_calls"].append(client.last_call_metrics)
                    save()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                select_supported_region,
                reference_rgb,
                reference_depth,
                args.query,
                "",
                client=TimedClient(),
                min_depth=0.1,
                max_depth=3.0,
                strategy="depth_candidates",
                search_padding=50,
                world_xyz=reference_world if args.split_surface_normals else None,
            )
            # Keep physics live during remote inference, without using simulator
            # object states to repair semantic selection or reacquisition.
            while not future.done():
                for _ in range(10):
                    mujoco.mj_step(model, data)
                time.sleep(0.02)
            _, instances, audit = future.result()
        result["physics_during_inference"] = True
        result["sim_time_after_initialization"] = float(data.time)
        result["initialization"] = audit
        save()
        if not audit["valid"]:
            result["passed"] = False
            save()
            return 1
        reference_mask = instances == 0
        np.save(args.output / "reference-mask.npy", reference_mask)
        for index in range(args.frames):
            # Advance actual physics between new camera captures. This remains
            # an in-process sensor diagnostic, not a live ZMQ run-agent session.
            for _ in range(10):
                mujoco.mj_step(model, data)
            rgb, depth, world, ticket = capture()
            Image.fromarray(rgb).save(args.output / f"capture-{index}.png")
            np.savez_compressed(args.output / f"capture-{index}.npz", rgb=rgb, depth=depth, world=world)
            diagnostics = {}
            started = time.perf_counter()
            try:
                if ticket.motion_epoch != reference_ticket.motion_epoch:
                    raise ValueError("motion_epoch_changed")
                if time.monotonic() - reference_ticket.received_monotonic > 30.0:
                    raise ValueError("reference_expired")
                selected = reacquire_surface(
                    reference_rgb,
                    reference_world,
                    reference_mask,
                    rgb,
                    depth,
                    world,
                    min_depth=0.1,
                    max_depth=3.0,
                    diagnostics=diagnostics,
                )
                row = {
                    "index": index,
                    "ok": True,
                    "pixels": int(selected.mask.sum()),
                    "position": np.median(selected.points, axis=0).tolist(),
                    "received_monotonic": ticket.received_monotonic,
                    "observation_age_s": time.monotonic() - ticket.received_monotonic,
                }
                Image.fromarray(np.where(selected.mask[..., None], rgb, 0)).save(args.output / f"track-{index}.png")
            except ValueError as exc:
                row = {"index": index, "ok": False, "reason": str(exc)}
            row["reacquire_s"] = time.perf_counter() - started
            row["diagnostics"] = diagnostics
            result["frames"].append(row)
            save()
        # Synthetic missing-depth injection is a separate rejection check; it is
        # not a physically removed object or another successful frame.
        try:
            reacquire_surface(
                reference_rgb,
                reference_world,
                reference_mask,
                rgb,
                np.full_like(depth, np.nan),
                world,
                min_depth=0.1,
                max_depth=3.0,
            )
        except ValueError as exc:
            result["missing_depth_rejected"] = str(exc)
        result["passed"] = all(row["ok"] for row in result["frames"]) and bool(result.get("missing_depth_rejected"))
        result["vlm_query_count"] = len(result["vlm_calls"])
        save()
        print(json.dumps({k: v for k, v in result.items() if k not in ("initialization", "frames")}), flush=True)
        return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
