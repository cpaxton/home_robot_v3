#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Run authored observation cases through the production CHAT agent and live Stretch.

Invoke serially via emet jobs --gpu-exclusive --cpu-safe. Remote Qwen makes
decisions; only the simulator renderer uses the local GPU. Scores remain manual
until action/evidence qualification, rather than interpreting a zero exit as success.
"""

import argparse
import json
import os
import subprocess
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from emet.eval.chat_gateway import recorded_chat_gateway
from emet.eval.harness_capability_agent import agent_turns, build_stretch_case, initial_gaze
from emet.eval.harness_capability_fixtures import load_development_cases


def run_agent_session(robot, input_reader, endpoint, model):
    from emet.agent.loop import run_agent_with_robot
    from emet.core import get_parameters

    parameters = get_parameters("dynav_config.yaml", robot="stretch")
    parameters["use_instance_memory"] = False
    parameters["force_eqa_siglip_encoder"] = False
    parameters["enable_tts"] = False
    parameters["agent"] = {**(parameters.get("agent") or {}), "tamp_inputs": "observed", "manip_mode": "physical"}
    parameters["motion_planner"] = {**(parameters.get("motion_planner") or {}), "obstacle_map_mode": "physical"}
    # No alternative policy/answer arbiter: actual production CHAT, same tools.
    # The lean diagnostic excludes learned object proposals/segmentation.
    run_agent_with_robot(
        robot="stretch",
        robot_client=robot,
        input_reader=input_reader,
        discord=False,
        use_llm=True,
        llm=f"openai@{endpoint}#{model}",
        device="cpu",
        parameters=parameters,
        memory_backend="dynamem",
        eqa=False,
        vl_include_camera=True,
        visual_servo=False,
        cpu_only=True,
        manipulation_only=True,
        max_tool_rounds=8,
        max_tokens=512,
        debug_llm=True,
        tool_debug=True,
        confirm_nav=False,
        thinking_status=False,
    )


def run_case(case, output, base_url, model, port):
    from emet.config.sim_launch_config import SimLaunchDefaultMujoco
    from emet.eval.sim_eval_session import benchmark_sim_server, connect_benchmark_robot
    from emet.simulation.sim_manipulation import robot_zmq_set_body_pose

    output.mkdir(parents=True, exist_ok=False)
    os.environ["EMET_PHYSICAL_AUDIT"] = str(output / "actuation.jsonl")
    scene, mjmodel = build_stretch_case(case, output / "fixture")
    cfg = SimLaunchDefaultMujoco(robot="stretch", headless=True, scene_path=str(scene), port_offset=port, seed=0)
    repo = Path(__file__).resolve().parents[1]
    turns = agent_turns(case)
    record = {
        "episode_id": case["id"],
        "family": case["family"],
        "turns": turns,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "scope": "Live Stretch, production CHAT; remote VLM, geometry-only memory; no learned localization/manipulation qualification",
        "model": model,
        "quantization": "unknown from endpoint metadata",
        "complete": False,
        "success": None,
        "scoring_status": "manual_evidence_and_action_audit_required",
    }
    (output / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
    measured, intervention = [], None
    robot = None
    with (output / "server.log").open("w") as server_log:
        with benchmark_sim_server(cfg, repo=repo, cpu_only=False, server_stderr=server_log):
            try:
                robot = connect_benchmark_robot(cfg, port, enable_rerun_server=False)
                with recorded_chat_gateway(base_url, output / "model_requests") as endpoint:

                    def capture(tag):
                        obs = robot.get_observation()
                        pose = np.asarray(robot.get_base_pose(), dtype=float).tolist()
                        row = {
                            "tag": tag,
                            "wall_time": time.time(),
                            "base_xyt": pose,
                            "head_pan_tilt": list(robot.get_pan_tilt()),
                            "image_timing": getattr(obs, "image_timing", None),
                        }
                        if hasattr(robot, "get_base_pose_world"):
                            row["base_world_xyt"] = np.asarray(robot.get_base_pose_world(), dtype=float).tolist()
                        if obs is not None and obs.rgb is not None:
                            Image.fromarray(np.asarray(obs.rgb, dtype=np.uint8)).save(output / f"{tag}.png")
                            np.savez_compressed(
                                output / f"{tag}-depth.npz",
                                depth=obs.depth,
                                camera_K=obs.camera_K,
                                camera_pose=obs.camera_pose,
                            )
                        state = getattr(robot, "_state", None) or {}
                        snapshot = state.get("native_tamp_snapshot")
                        if snapshot:
                            row["scorer_only_physics"] = snapshot
                        measured.append(row)
                        (output / "measured.json").write_text(json.dumps(measured, indent=2, default=str))

                    turn_index = 0

                    def next_turn(_prompt):
                        nonlocal turn_index, intervention
                        if turn_index:
                            capture(f"turn-{turn_index}-after")
                        else:
                            pan, tilt = initial_gaze(mjmodel)
                            if robot.head_to(pan, tilt, blocking=True) is not True:
                                raise RuntimeError("Initial head pose unconfirmed")
                            time.sleep(1)
                            capture("initial")
                        if turn_index == len(turns):
                            return None
                        if turn_index == 1 and case["family"] == "changed" and case["variant"] == "moved":
                            # Same command owner, evaluator-only change between
                            # completed turns. No coordinates reach the model.
                            robot_zmq_set_body_pose(robot, "target", [3.5, 0.6, 0.8])
                            intervention = {
                                "type": "target_move",
                                "requested_at": time.time(),
                                "requested_position": [3.5, 0.6, 0.8],
                                "verification": "inspect scorer physics snapshot",
                            }
                            time.sleep(0.5)
                        capture(f"turn-{turn_index + 1}-before")
                        turn = turns[turn_index]
                        turn_index += 1
                        return turn

                    with (output / "agent.log").open("w", buffering=1) as log:
                        with redirect_stdout(log), redirect_stderr(log):
                            run_agent_session(robot, next_turn, endpoint, model)
                    if turn_index != len(turns) or measured[-1]["tag"] != f"turn-{len(turns)}-after":
                        raise RuntimeError("Agent exited before completing the declared turn protocol")
                    record.update(complete=True, exit_code=0)
            except Exception as error:
                record.update(error=f"{type(error).__name__}: {error}")
            finally:
                if robot is not None:
                    robot.stop()
    record["intervention"] = intervention
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/eval/harness_capability_dev.json"))
    p.add_argument("--output", type=Path)
    p.add_argument("--episodes", nargs="+", default=["cap-dev-01", "cap-dev-02"])
    p.add_argument("--base-url")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = p.parse_args()
    if not args.output or not args.base_url:
        p.error("--output and --base-url are required")
    cases = {c["id"]: c for c in load_development_cases(args.config)}
    if set(args.episodes) - cases.keys():
        p.error("Unknown development case")
    args.output = args.output.absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(EMET_PHYSICAL_EXECUTION="1", EMET_SIM_NAV_TELEPORT="0", EMET_MOLMOSPACES_NAV_TELEPORT="0")
    for index, key in enumerate(args.episodes):
        row = run_case(cases[key], args.output / key, args.base_url, args.model, 2700 + index * 10)
        print(json.dumps(row), flush=True)
        if not row["complete"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
