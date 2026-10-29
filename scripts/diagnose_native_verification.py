#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Observe native verification without changing its predicate, timeouts or result.

Reuse an earlier live protocol's exact commands and fixture. This is privileged
stationary Stretch through the production CHAT API, with a read-only scorer.
The process-local method spy is restored on exit and never admits an action.
"""

import argparse
import json
import os
import subprocess
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import mujoco
import numpy as np


@contextmanager
def verification_trace(output):
    from emet.controller.task.tamp.native_adapter import NativeTableAdapter
    from emet.simulation.sim_object_placements import placements_from_mujoco_model

    original = NativeTableAdapter._verify

    def observe(adapter, plan, **kwargs):
        samples = []
        sync = adapter.synchronize
        started, sim_start = time.monotonic(), float(adapter.data.time)
        oid = adapter.model.body(plan.object_body).id
        joint = int(adapter.model.body(plan.object_body).jntadr[0])
        dof = int(adapter.model.jnt_dofadr[joint])

        def capture(data):
            sync(data)
            # No extra synchronization, collision stepping, or time advance.
            samples.append(
                {
                    "sim_time": float(data.time),
                    "body_xyz": data.body(oid).xpos.tolist(),
                    "speed": float(np.linalg.norm(data.qvel[dof : dof + 3])),
                    "spin": float(np.linalg.norm(data.qvel[dof + 3 : dof + 6])),
                }
            )

        adapter.synchronize = capture
        result, error = None, None
        try:
            result = original(adapter, plan, **kwargs)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            adapter.synchronize = sync
            data = mujoco.MjData(adapter.model)
            mujoco.mj_copyData(data, adapter.model, adapter.data)
            mujoco.mj_collision(adapter.model, data)
            contacts = {
                adapter.model.body(
                    adapter.model.geom_bodyid[c.geom2 if adapter.model.geom_bodyid[c.geom1] == oid else c.geom1]
                ).name
                for c in data.contact
                if oid in adapter.model.geom_bodyid[c.geom]
            }
            entry = placements_from_mujoco_model(adapter.model, data)[plan.object_body]
            position = np.asarray(entry["pos"])
            bounds = np.asarray(entry["bounds"])[:, :2]
            row = {
                "object": plan.object_body,
                **kwargs,
                "passed": result,
                "error": error,
                "wall_s": time.monotonic() - started,
                "elapsed_sim_s": float(data.time) - sim_start,
                "goal": plan.goal.tolist(),
                "position": position.tolist(),
                "contacts": sorted(contacts),
                "xy_error": float(np.linalg.norm(position[:2] - plan.goal[:2])),
                "expected_height": float(plan.goal[2] - 0.02),
                "height_error": abs(float(position[2] - (plan.goal[2] - 0.02))),
                "region_inside": None
                if plan.placement_region is None
                else bool(
                    np.all(bounds[0] >= plan.placement_region[0] - 0.005)
                    and np.all(bounds[1] <= plan.placement_region[1] + 0.005)
                ),
                "samples": samples,
            }
            with output.open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    NativeTableAdapter._verify = observe
    try:
        yield
    finally:
        NativeTableAdapter._verify = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--sim", type=Path, required=True)
    p.add_argument("--task-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = p.parse_args()
    from emet.agent.loop import run_agent_with_robot
    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.eval.native_table_live import live_table_robot
    from emet.eval.native_table_scoring import goal_predicates
    from emet.simulation.mujoco_server_stretch import default_scene_xml_path

    protocol = json.loads(args.protocol.read_text())
    cli = protocol["cli"]
    commands = [cli[i + 1] for i, value in enumerate(cli) if value == "--command"]
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    config = output / "agent.yaml"
    config.write_text(
        f"extends: {repo}/configs/emet/default.yaml\nagent:\n  manip_mode: physical\n  tamp_inputs: privileged\nmapping:\n  enable_tts: false\n"
    )
    task = json.loads(args.task_config.read_text())
    sim = args.sim.absolute()
    os.environ.update(
        EMET_PHYSICAL_EXECUTION="1",
        EMET_PHYSICAL_AUDIT=str(output / "actuation.jsonl"),
        EMET_SIM_NAV_TELEPORT="0",
        EMET_MOLMOSPACES_NAV_TELEPORT="0",
    )
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "commands": commands,
                "fixture": str(sim),
                "original_protocol": str(args.protocol),
                "scope": "Current production CHAT API, original task commands; read-only verification spy; privileged stationary",
                "model": args.model,
                "quantization": "not established by endpoint metadata",
            },
            indent=2,
        )
    )
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    with live_table_robot(sim, output, model, initialize=False) as scorer:
        with verification_trace(output / "verification.jsonl"):
            previous = Path.cwd()
            try:
                os.chdir(output)
                with (output / "agent.log").open("w", buffering=1) as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        run_agent_with_robot(
                            robot="stretch",
                            port_offset=load_sim_launch_config_from_path(str(sim)).port_offset,
                            discord=False,
                            use_llm=True,
                            llm=f"openai@{args.base_url}#{args.model}",
                            device="cpu",
                            memory_backend="dynamem",
                            state_only_sim=True,
                            max_tool_rounds=12,
                            commands=commands,
                            agent_config=str(config),
                            native_task_config=task,
                            tool_debug=True,
                            thinking_status=False,
                        )
            finally:
                os.chdir(previous)
        scorer.refresh()
        result = {
            "final_predicates": goal_predicates(scorer, protocol["goals"]),
            "scoring_scope": "Instantaneous final snapshot; continuous stability is reported by verification.jsonl",
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
