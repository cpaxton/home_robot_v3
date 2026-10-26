#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Live RGBD stream -> bounded surface reassociation -> actual scene_tasks tool.

Single-object sensor smoke, not full task acceptance: no support seed or native
observed execution provider is installed. Caliban supplies semantic initialization
only; later tool calls read measured camera data without model inference. Run via
emet jobs with an available software renderer when the local GPU is reserved.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from emet.agent.tools import get_tools
from emet.config.sim_launch_config import load_sim_launch_config_from_path
from emet.controller.task.tamp.observed_capture import bind_observed_capture_source, capture_arrays
from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider
from emet.eval.sim_eval_session import (
    connect_benchmark_robot,
    launch_benchmark_sim_server,
    terminate_benchmark_sim_server,
)
from emet.llms.openai_vllm_client import OpenaiVLLMClient
from emet.mapping.scene_graph.open_vocab_scene_graph import ObjectObservation
from emet.memory.vlm_region_grounding import select_supported_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query", default="red cylinder")
    parser.add_argument("--samples", type=int, default=5)
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
    result = {
        "scope": "live ZMQ camera/provider/tool sensor smoke, no support seed or task execution acceptance",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_success_assessed": False,
        "queries": [],
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
                save()
                time.sleep(0.2)
            result["reacquisition_passed"] = all(
                q["reacquisition_error"] is None and q["response"]["status"] == "ok" for q in result["queries"]
            )
            save()
            return 0 if result["reacquisition_passed"] else 1
        finally:
            try:
                if client is not None:
                    client.stop()
            finally:
                terminate_benchmark_sim_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
