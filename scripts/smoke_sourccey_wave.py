#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Exercise Sourccey waves through GenericZmqClient and the production sim handler.

Transport is in-process; physics advances in place of wall-clock sleep. This
checks actuator dynamics, not network delivery or collision-free hardware motion.
Run: MUJOCO_GL=egl uv run python scripts/smoke_sourccey_wave.py
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/emet-mpl")

import cv2
import mujoco
import numpy as np

from emet.controller.generic_zmq_client import GenericZmqClient
from emet.controller.task.emote.emote_task import EmoteTask
from emet.robots.sourccey import SourcceyBackend
from emet.simulation.mujoco_server import _load_default_scene_with_robot
from emet.simulation.robosuite_server import RobosuiteZmqServer


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    spec = SourcceyBackend().get_spec()
    model = _load_default_scene_with_robot("sourccey")
    server = RobosuiteZmqServer(
        robot_spec=spec, scene_model=model, send_port=0, recv_port=0, send_state_port=0, send_servo_port=0
    )
    renderer = video = None
    try:
        server._load_model()
        server._stabilize_physics_state_after_load()
        server._running = True
        # A displaced base guards against accidentally sending base position as velocity.
        server._mjdata.qpos[:3] = [-0.3, 0.2, 0.35]
        mujoco.mj_forward(model, server._mjdata)
        client = GenericZmqClient.__new__(GenericZmqClient)
        client._spec, client._obs_lock = spec, threading.Lock()
        client._obs = client._servo = None
        client.send_action = lambda action, **_: server.handle_action(action)
        qadr = [model.joint(name).qposadr[0] for name in spec.joint_names]

        def feedback():
            client._state = {"joint_positions": server._mjdata.qpos[qadr].copy(), "at_goal": True}

        feedback()
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.15, 0.2, 0.7]
        camera.distance, camera.azimuth, camera.elevation = 2.4, 155, -18
        video = cv2.VideoWriter(str(out / "wave.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
        assert video.isOpened()
        trace = []
        label = ""

        def advance(seconds):
            for _ in range(round(seconds / model.opt.timestep)):
                server._mj_step_once()
            feedback()
            trace.append(server._mjdata.qpos[qadr].copy())
            renderer.update_scene(server._mjdata, camera=camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for _ in range(max(1, round(seconds / 0.05))):
                video.write(frame)

        agent = SimpleNamespace(robot=client, space=None, parameters={})
        results = {}
        for side in ("left", "right"):
            label = f"Sourccey: {side} wave"
            start = server._mjdata.qpos[qadr].copy()
            first = len(trace)
            task = EmoteTask(agent).get_task(f"wave_{side}")
            with patch("emet.robots.sourccey.emote_backend.time.sleep", advance):
                success = task.run()
            motion = np.asarray(trace[first:])
            wrist = spec.joint_names.index(f"{side}_wrist_roll")
            opposite = "right" if side == "left" else "left"
            held = [
                spec.joint_names.index(f"{opposite}_{joint}")
                for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
            ]
            results[side] = {
                "success": bool(success),
                "wrist_excursion_rad": float(np.ptp(motion[:, wrist])),
                "max_base_displacement_m": float(np.max(np.linalg.norm(motion[:, :2] - start[:2], axis=1))),
                "max_other_arm_drift_rad": float(np.max(np.abs(motion[:, held] - start[held]))),
                "final_position_error": float(np.max(np.abs(motion[-1, 3:] - start[3:]))),
            }
            assert success, results[side]
            assert results[side]["wrist_excursion_rad"] > 0.9
            assert results[side]["max_base_displacement_m"] < 0.01
            assert results[side]["max_other_arm_drift_rad"] < 0.05
        (out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        np.save(out / "joint_trace.npy", np.asarray(trace))
        print(json.dumps(results, indent=2))
        return results
    finally:
        if video is not None:
            video.release()
        if renderer is not None:
            renderer.close()
        server._running = False
        server._close_renderers()
        server.close_zmq_resources()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/sourccey-wave"))
    run(parser.parse_args().out)
