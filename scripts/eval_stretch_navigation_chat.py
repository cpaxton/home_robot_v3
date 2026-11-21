#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Exercise production CHAT tools against a loopback Stretch observation-only bridge."""

import argparse
import copy
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from eval_stretch_launch_survey import measured_points, scene_variants, survey_capture
from eval_stretch_slip import build_scene as build_slip_scene
from eval_stretch_slip import reference_from_depth
from PIL import Image
from run_harness_capability_agent import run_agent_session

from emet.eval.chat_gateway import recorded_chat_gateway
from emet.eval.harness_capability_agent import build_stretch_case
from emet.eval.harness_capability_fixtures import load_development_cases
from emet.utils.process_tree import popen_session, terminate_process_tree

PRIVATE_FIELDS = {
    "native_tamp_snapshot",
    "sim_object_placements",
    "sim_articulations",
    "sim_base_pose_query",
    "spawn_floor_map",
    "articulation_result",
    "scene_qpos",
    "scene_qvel",
    "robot_sample",
    "base_xyz",
    "base_up_dot_world_z",
    "ee_pose",
    "actuator_targets",
    "actuator_names",
}
MODEL_NAME = "observed-control-script"
REPO = Path(__file__).resolve().parents[1]
DISCORD_CASE = "model-discord"
DISCORD_DRIVE_CASE = "model-discord-drive"
DISCORD_FULL_CASE = "model-discord-full"
FULL_CHAT_CONTROL = "model-full-chat"
OBJECT_RECOVERY_CONTROL = "model-object-recovery"
CHANGED_OBJECT_RECOVERY = "model-object-recovery-changed"
OBJECT_RECOVERY_CASES = (OBJECT_RECOVERY_CONTROL, CHANGED_OBJECT_RECOVERY)
TABLE_TRANSFER_CASE = "model-table-transfer"
DISCORD_CASES = (DISCORD_CASE, DISCORD_DRIVE_CASE, DISCORD_FULL_CASE)
TERMINAL_FULL_CASES = (FULL_CHAT_CONTROL, *OBJECT_RECOVERY_CASES, TABLE_TRANSFER_CASE)
FULL_CHAT_CASES = (DISCORD_FULL_CASE, *TERMINAL_FULL_CASES)
MODEL_CASES = ("model-read-only", "model-look-around", "model-drive", *TERMINAL_FULL_CASES, *DISCORD_CASES)
WITNESS_CASES = ("witness-clear", "slip", "witness-loss", "model-drive", "operator-stop", DISCORD_DRIVE_CASE,
                 *FULL_CHAT_CASES)
CASES = ("clear", "occupied", "occupied-recovery", "turn", "stale", *WITNESS_CASES,
         "model-read-only", "model-look-around", DISCORD_CASE)
MODEL_DRIVE_METERS = 0.45


def prompts_for(case):
    if case in DISCORD_CASES:
        return ()
    if case == "model-read-only":
        return ("Capture a photo with send_image. From the attached camera view, describe the color and shape "
                "of the small object on top of the round pedestal. Keep the robot stationary.",)
    if case == "model-look-around":
        return ("Can you look around? Tell me what you see on each side.",)
    if case == FULL_CHAT_CONTROL:
        return ("What can you see?", "Which object is to the right of the red cylinder?",
                "Is a red mug visible in the current view?", "Show me your map.")
    if case in OBJECT_RECOVERY_CASES:
        turns = (
            "Look at the table using describe_scene with pan_rad -1.48 and tilt_rad -0.76. "
            "Describe the blue block and remember where you are looking.",
            "Look straight ahead using describe_scene with pan_rad 0 and tilt_rad -0.2. "
            "Is the blue block visible in this new view?",
            "Keep the camera where it is. Use query_memory with question 'blue block' to retrieve its location. "
            "Say whether this is a current sighting or a remembered location.",
            "Look back at the blue block from your earlier table view. Use measured head motion "
            "and tell me whether the block is visible again and what it is resting on.",
        )
        if case == CHANGED_OBJECT_RECOVERY:
            turns += ("Use ground_object to locate the blue block in the current view. Tell me whether "
                      "that agrees with its earlier remembered location. Keep the base stationary.",)
        return turns
    if case == TABLE_TRANSFER_CASE:
        return ("Pick up the red cylinder and place it on the green target area on the table. "
                "Keep the base stationary and leave the blue cube undisturbed. "
                "Report the measured outcome or the reason you stopped.",)
    if case == "model-drive":
        return (f"Try one straight forward drive of {MODEL_DRIVE_METERS} meters using move_forward. Let the robot "
                "check clearance. Report what the tool actually did, including any refusal or stop.",)
    motion = "a 30 degree turn to the left" if case == "turn" else "a 1.5 metre straight forward drive"
    if case in WITNESS_CASES:
        motion = "a 0.45 metre straight forward drive"
    return (
        "Use scene_tasks to check the available transfers.",
        "Check whether a cube-to-table transfer can be planned. Leave the object stationary.",
        f"Inspect the floor before requesting {motion}.",
    )


def bounded_model_tools(get_tools, case, *, navigation_stop_requested=None):
    """Constrain model trials while leaving production perception and drive handlers in place."""
    from emet.agent.tool_outcome import ToolOutcome

    drive_attempted = False

    def bind(context):
        tools = {tool.name: tool for tool in get_tools(context)}
        if case in ("model-read-only", DISCORD_CASE):
            return [tools["send_image"]]
        if case in ("model-look-around", DISCORD_DRIVE_CASE):
            from emet.agent.tools import pending_discord_image_for_send

            look = tools["describe_scene"]
            describe_scene = look.func

            def look_to(pan_rad=None):
                if type(pan_rad) not in (float, int) or not -.60 <= pan_rad <= .60:
                    return ToolOutcome(False, status="outside_trial_bound", tool=look.name,
                                       note="Supply head pan between -0.60 and 0.60 radians.")
                outcome = describe_scene(pan_rad=pan_rad, tilt_rad=-0.20)
                bot = context.get("discord_bot")
                if isinstance(outcome, ToolOutcome) and outcome.ok and bot is not None:
                    image = pending_discord_image_for_send(context)
                    direction = "Left" if pan_rad > 0 else "Right" if pan_rad < 0 else "Forward"
                    bot.push_task_to_all_channels(message=f"{direction} camera view", content=image)
                return outcome

            look.func = look_to
            look.parameters = {"type": "object", "properties": {"pan_rad": {
                "type": "number", "minimum": -.60, "maximum": .60,
            }}, "required": ["pan_rad"]}
            if case == "model-look-around":
                return [tools["send_image"], look]
        tool = tools["move_forward"]
        move_forward = tool.func

        def drive(meters=None):
            nonlocal drive_attempted
            if case == DISCORD_DRIVE_CASE:
                origin, _, request = context.get("operator_turn", "").partition(": ")
                if not origin.startswith("[discord] ") or request.strip().casefold() not in ("drive 0.45 m", "explore"):
                    return ToolOutcome(False, status="operator_authorization_required", tool=tool.name,
                                       note="The operator can authorize one guarded step with `explore` or `drive 0.45 m`.")
            if drive_attempted or type(meters) not in (float, int) or meters != MODEL_DRIVE_METERS:
                return ToolOutcome(False, status="outside_trial_bound", tool=tool.name,
                                   note=f"This trial allows one request for {MODEL_DRIVE_METERS} meters.")
            if navigation_stop_requested is not None and navigation_stop_requested():
                return ToolOutcome(False, status="operator_stop_requested", tool=tool.name,
                                   note="Further driving is disabled for this session.")
            drive_attempted = True
            outcome = move_forward(meters=meters)
            if (case == DISCORD_DRIVE_CASE and navigation_stop_requested is not None
                    and navigation_stop_requested() and isinstance(outcome, dict)):
                from emet.controller.task.tamp.api import response
                from emet.core.command_client import command_receipt

                robot = context["robot"]
                command = getattr(robot, "_last_navigation_command", None)
                receipt = command_receipt(robot, command) if command is not None else None
                if receipt and receipt["status"] == "cancelled" and receipt["result"]["stop_confirmed"]:
                    return response(tool.name, code="operator_cancelled", data={**outcome["data"], "stop_confirmed": True})
            return outcome

        tool.func = drive
        if case == DISCORD_DRIVE_CASE:
            context["bounded_forward_distance_m"] = MODEL_DRIVE_METERS
            return [tools["send_image"], look, tool]
        return [tool]

    return bind


def serve(scene, offset, case, output):
    import mujoco

    import emet.simulation.mujoco_server as runner
    from emet.core.navigation_result import NavigationPoseUncertainty
    from emet.eval.stretch_encoder_server import StretchEncoderNavigationServer

    score_joints = {}
    if case in (TABLE_TRANSFER_CASE, CHANGED_OBJECT_RECOVERY):
        model = mujoco.MjModel.from_xml_path(str(scene))
        names = ("object1", "object2") if case == TABLE_TRANSFER_CASE else ("object1",)
        for name in names:
            joint = int(model.body(name).jntadr[0])
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"scorer requires a free joint for {name}")
            score_joints[name] = (int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]))

    if case == CHANGED_OBJECT_RECOVERY:
        from emet.simulation.sim_manipulation import set_free_body_pose
        from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer

        callback = MujocoServer._ctrl_callback
        intervention = output / "scoring-intervention.request"
        acknowledgement = output / "scoring-intervention.json"
        shifted = False

        def shift_fixture_after_step(server, model, data):
            nonlocal shifted
            callback(server, model, data)
            if not shifted and intervention.is_file():
                shifted = True
                # Only the physics process sees this fixture intervention.
                ok = set_free_body_pose(model, data, "object1", [-.13, -.75, .55])
                acknowledgement.write_text(json.dumps({"ok": ok, "monotonic": time.monotonic()}))

        MujocoServer._ctrl_callback = shift_fixture_after_step

    class ScoredEncoderServer(StretchEncoderNavigationServer):
        def __init__(self, *args, **kwargs):
            self._scoring_stamp = 0
            self._feedback_loss = None
            self._fault_lock = threading.Lock()
            self._inject_loss = case == "stale"
            super().__init__(
                *args,
                encoder_model_path=Path(ET.parse(scene).getroot().find("include").attrib["file"]),
                arrival_pose_uncertainty=NavigationPoseUncertainty(0.04, 0.025),
                motion_reference=self.independent_pose if case in WITNESS_CASES else None,
                **kwargs,
            )

        def independent_pose(self, sample):
            if case == "witness-loss" and self.active and self._encoder_frame.base_xyt[0] > 0.07:
                if self._feedback_loss is None:
                    self._feedback_loss = time.monotonic()
                return None
            return reference_from_depth(sample["rgb"], sample["depth"])

        def _check_independent_motion(self, imagery, frame):
            previous = self._motion_stamp
            check = super()._check_independent_motion(imagery, frame)
            if frame.timestamp_ns != previous:
                row = {"timestamp_ns": frame.timestamp_ns, "encoder_xyt": frame.base_xyt.tolist(), **asdict(check)}
                with (output / "motion-witness.jsonl").open("a") as log:
                    log.write(json.dumps(row) + "\n")
            return check

        def _capture_encoder_frame(self, imagery):
            if self._inject_loss:
                with self._fault_lock:
                    previous = self._encoder_frame
                    if (
                        self._feedback_loss is None
                        and self.active
                        and previous is not None
                        and previous.base_xyt[0] > 0.07
                    ):
                        self._feedback_loss = time.monotonic()
                    if self._feedback_loss is not None and time.monotonic() - self._feedback_loss < 4.0:
                        return None
            return super()._capture_encoder_frame(imagery)

        def cancel_navigation_command(self):
            guard_failed = self._motion_check is not None and not self._motion_check.ok
            if (self._feedback_loss is None and not guard_failed
                    and case not in ("operator-stop", DISCORD_DRIVE_CASE, DISCORD_FULL_CASE)):
                return super().cancel_navigation_command()
            called_at = time.monotonic()
            started = self.robot_sim.pull_status()
            stopped = super().cancel_navigation_command()
            ended = self.robot_sim.pull_status()
            path = output / "fault-stop.json"
            values = {
                "cancel_monotonic": called_at,
                "stop_monotonic": time.monotonic(),
                "wheel_stop_confirmed": stopped,
                "started_forward_m_s": started.base.x_vel,
                "ended_forward_m_s": ended.base.x_vel,
                "ended_yaw_rad_s": ended.base.theta_vel,
                "truth_at_cancel": [started.base.x, started.base.y, started.base.theta],
                "truth_at_stop": [ended.base.x, ended.base.y, ended.base.theta],
            }
            if self._feedback_loss is not None:
                values.update(drop_to_cancel_s=called_at - self._feedback_loss,
                              drop_to_stop_s=time.monotonic() - self._feedback_loss)
            path.write_text(json.dumps(values, indent=2) + "\n")
            return stopped

        def get_full_observation_message(self):
            message = super().get_full_observation_message()
            if message is None or threading.current_thread().name == self._servo_score_thread:
                return message
            stamp = message["head_cam_left/image_timing"]["timestamp_ns"]
            if stamp > self._scoring_stamp:
                self._scoring_stamp = stamp
                status = self._status
                if status is not None:
                    row = {
                        "timestamp_ns": stamp,
                        "wall_time": time.time(),
                        "truth_xyt": [status.base.x, status.base.y, status.base.theta],
                        "estimated_xyt": [*message["gps"], *message["compass"]],
                    }
                    if score_joints and status.scene_qpos is not None and status.scene_qvel is not None:
                        row["objects"] = {
                            name: {"xyz": status.scene_qpos[q:q + 3], "velocity": status.scene_qvel[v:v + 6]}
                            for name, (q, v) in score_joints.items()
                        }
                    with (output / "scoring-only.jsonl").open("a") as log:
                        log.write(json.dumps(row) + "\n")
            return message

        def get_servo_message(self):
            self._servo_score_thread = threading.current_thread().name
            return super().get_servo_message()

        _servo_score_thread = None

    runner._get_stretch_server = lambda: ScoredEncoderServer
    runner.main(
        [
            "--headless",
            "--seed",
            "0",
            "--scene_path",
            scene,
            "--port-offset",
            str(offset),
            "--use_remote_computer",
            "false",
        ],
        standalone_mode=False,
    )


def private_keys(obj):
    if isinstance(obj, dict):
        return set(obj) & PRIVATE_FIELDS | set().union(*(private_keys(x) for x in obj.values()))
    if isinstance(obj, (list, tuple)):
        return set().union(*(private_keys(x) for x in obj))
    return set()


def compact_request(body, index):
    messages = []
    for message in body["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            elements = content
        else:
            elements = [{"type": "text", "text": content or ""}]
        messages.append(
            {
                "role": message["role"],
                "text": "\n".join(str(x.get("text", "")) for x in elements if x.get("type") == "text"),
                "image_count": sum(x.get("type") == "image_url" for x in elements),
            }
        )
    return {"index": index, "messages": messages, "unexpected_keys": sorted(private_keys(body))}


def start_model(output, case, prompts):
    packets = output / "model-requests"
    packets.mkdir()
    motion = {"name": "move_forward", "arguments": {"meters": .45 if case in WITNESS_CASES else 1.5}}
    if case == "turn":
        motion = {"name": "rotate_base", "arguments": {"degrees": 30.0}}
    calls = (
        [{"name": "scene_tasks", "arguments": {}}],
        [{"name": "plan_pick_place", "arguments": {"object_name": "cube", "receptacle_name": "table"}}],
        [{"name": "observe_floor", "arguments": {"target_blocker": False}}, motion],
    )
    if case == "occupied-recovery":
        calls[2].insert(1, {"name": "observe_floor", "arguments": {"pan_rad": 0.26, "tilt_rad": -1.0}})
    rounds = [0, 0, 0]
    replies = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.send_json({"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(replies)
            (packets / f"request-{index:02d}.json").write_text(json.dumps(body) + "\n")
            request = compact_request(body, index)
            (output / f"request-{index:02d}.json").write_text(json.dumps(request, indent=2) + "\n")
            matching = [
                i
                for message in request["messages"]
                if message["role"] == "user"
                for i, prompt in enumerate(prompts)
                if prompt in message["text"]
            ]
            turn = matching[-1] if matching else 0
            pending = calls[turn]
            action = pending[rounds[turn]] if rounds[turn] < len(pending) else None
            rounds[turn] += 1
            replies.append(action)
            message = {"tool_calls": [action] if action else [], "message": "" if action else "Test complete."}
            self.send_json(
                {
                    "id": f"test-{index}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(message)},
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker, replies


def port_check(offset):
    sockets = []
    try:
        for port in range(4401 + offset, 4405 + offset):
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", port))
    finally:
        for sock in sockets:
            sock.close()


def build_table_fixture(output, *, transfer_target=False):
    output.mkdir()
    assets = REPO / "src/emet/assets/robot"
    robot = ET.parse(assets / "stretch.xml")
    robot.find("compiler").set("assetdir", str(assets / "assets"))
    robot.write(output / "stretch.xml")
    scene = ET.parse(assets / "scene_environment.xml")
    scene.getroot().insert(0, ET.Element("include", file="stretch.xml"))
    # Keep the two objects clear of the wrist in the initial head view.
    scene.find('./worldbody/body[@name="object1"]').set("pos", "-.17 -.55 .6")
    if transfer_target:
        ET.SubElement(scene.find("worldbody"), "geom", name="green_target", type="box", pos=".08 -.74 .4805",
                      size=".09 .09 .0005", rgba=".04 .65 .16 1", contype="0", conaffinity="0")
    path = output / "scene.xml"
    scene.write(path)
    return path


def run(case, output, offset, scene, *, model_endpoint=None, model_name=MODEL_NAME, initial_gaze=(0.0, -0.2),
        table_preflight_only=False):
    from emet.agent import loop
    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.core.command_client import command_receipt
    from emet.mapping.voxel.launch_floor_survey import register_launch_floor_survey
    from emet.simulation.sim_subprocess import wait_for_sim_tcp_port

    interactive_discord = case in DISCORD_CASES
    full_chat = case in FULL_CHAT_CASES
    operator_stop = threading.Event() if case in ("operator-stop", DISCORD_DRIVE_CASE, DISCORD_FULL_CASE) else None
    monitor_done = threading.Event()
    monitor = None
    table_robot_model = None
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if scene is None and full_chat:
        scene = build_table_fixture(output / "fixture", transfer_target=case == TABLE_TRANSFER_CASE)
    elif scene is None:
        case_id = "cap-dev-06" if case.startswith("occupied") else "cap-dev-05"
        cases = load_development_cases(REPO / "configs/eval/harness_capability_dev.json")
        scene, _ = build_stretch_case(next(item for item in cases if item["id"] == case_id), output / "fixture")
    scene, survey_path, robot_model_path = scene_variants(scene.resolve(), output)
    survey_rgb, survey_depth, survey_k, survey_pose, surveyed_at = survey_capture(survey_path)
    survey_points, survey_anchors = measured_points(survey_rgb, survey_depth, survey_k, survey_pose)
    np.savez_compressed(
        output / "launch-survey-depth.npz",
        rgb=survey_rgb,
        depth=survey_depth,
        camera_K=survey_k,
        camera_pose=survey_pose,
        anchors=survey_anchors,
    )
    env = {
        **os.environ,
        "EMET_PHYSICAL_EXECUTION": "1",
        "EMET_PHYSICAL_AUDIT": str(output / "actuation.jsonl"),
        "EMET_SIM_NAV_TELEPORT": "0",
        "EMET_MOLMOSPACES_NAV_TELEPORT": "0",
        "EMET_SIM_SYNCHRONIZED_RGBD": "1",
    }
    env.pop("DISCORD_TOKEN", None)
    env.pop("EMET_SIM_FIXED_RGBD_CAMERA", None)
    if case in WITNESS_CASES:
        scene = build_slip_scene(scene, output, slippery=case == "slip")
        env["EMET_SIM_FIXED_RGBD_CAMERA"] = "slip_rgbd"
    port_check(offset)
    started = time.monotonic()
    server = robot = model = model_worker = None
    outcome = {
        "case": case,
        "fixture": str(scene),
        "fixture_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "decisions": ("sensor admission control" if table_preflight_only
                      else "real model via OpenAI-compatible endpoint" if model_endpoint
                      else "scripted local CHAT replies"),
        "model": model_name,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)),
    }

    def save():
        (output / "result.json").write_text(json.dumps(outcome, indent=2) + "\n")

    def request_operator_stop():
        if not operator_stop.is_set():
            requested_at = time.monotonic()
            operator_stop.set()
            (output / "operator-stop.json").write_text(json.dumps({"requested_monotonic": requested_at}) + "\n")

    def record_table_capture(view):
        from emet.eval.stretch_table_observation import assess_live_table_capture

        outcome["table_capture"] = assess_live_table_capture(view, table_robot_model)
        np.savez_compressed(output / "table-capture.npz", rgb=view.rgb, depth=view.depth,
                            camera_K=view.camera_K, camera_pose=view.camera_pose)
        (output / "table-capture-timing.json").write_text(json.dumps({
            "received_monotonic": view.received_monotonic, "image_timing": view.image_timing,
        }, indent=2) + "\n")
        save()

    try:
        with (output / "server.log").open("w") as log:
            server = popen_session(
                [sys.executable, "-u", __file__, "--serve", str(scene), str(offset), case, str(output)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if interactive_discord:
                # Let the operator interrupt reach run() so it can save the session and stop the bridge.
                signal.signal(signal.SIGINT, signal.default_int_handler)
            wait_for_sim_tcp_port("127.0.0.1", 4401 + offset, proc=server, timeout_sec=110)
            robot = create_robot_client_from_cli(
                "stretch",
                "127.0.0.1",
                port_offset=offset,
                enable_rerun_server=False,
                start_immediately=True,
                allow_missing_depth=False,
            )
            if operator_stop is not None:
                robot.navigation_stop_requested = operator_stop.is_set
            robot.move_to_nav_posture()
            assert robot.head_to(0.0, -0.88, blocking=True) is True, "anchor head posture unconfirmed"
            time.sleep(0.5)
            anchor_obs = robot.get_observation()
            _, onboard_anchors = measured_points(
                anchor_obs.rgb, anchor_obs.depth, anchor_obs.camera_K, anchor_obs.camera_pose
            )
            anchor_stamp = anchor_obs.image_timing["head_cam_left"]["timestamp_ns"] / 1e9
            np.savez_compressed(
                output / "launch-onboard-depth.npz",
                rgb=anchor_obs.rgb,
                depth=anchor_obs.depth,
                camera_K=anchor_obs.camera_K,
                camera_pose=anchor_obs.camera_pose,
                anchors=onboard_anchors,
            )
            if not full_chat:
                assert robot.head_to(*initial_gaze, blocking=True) is True, "initial head posture unconfirmed"
            sample = robot.get_observation()
            assert sample is not None and sample.rgb is not None and sample.depth is not None
            with robot._state_lock:
                state = copy.deepcopy(robot._state)
            with robot._obs_lock:
                obs = copy.deepcopy(robot._obs)
            session = robot.get_emet_session()
            outcome["wire_audit"] = {
                "state_fields": sorted(state),
                "observation_fields": sorted(obs),
                "session": session,
                "private_fields": sorted(private_keys((state, obs, session))),
            }
            assert not outcome["wire_audit"]["private_fields"]
            assert session["is_simulation"] is True and session["runtime_kind"] == "stretch_mujoco_sim"
            assert session["capabilities"]["teleport_base"] is False
            assert session["pose_source"] == "stretch_wheel_and_head_encoders"
            assert session["navigation_origin_xyt"] == [0.0, 0.0, 0.0]
            timing = obs["head_cam_left/image_timing"]
            assert timing["wheel_sample"]["interval_ns"] == timing["motor_sample"]["interval_ns"]
            assert timing["wheel_sample"]["interval_ns"] == timing["depth_capture_interval_ns"]
            np.savez_compressed(
                output / "first-head-capture.npz",
                rgb=sample.rgb,
                depth=sample.depth,
                camera_K=sample.camera_K,
                camera_pose=sample.camera_pose,
            )
            (output / "first-head-timing.json").write_text(json.dumps(timing, indent=2) + "\n")

            before = np.asarray(robot.get_base_pose(), float)
            if case == "operator-stop":
                def stop_after_motion():
                    while not monitor_done.wait(0.02):
                        pose = robot.get_base_pose()
                        if (getattr(robot, "_last_navigation_command", None) is not None and pose is not None
                                and np.linalg.norm(pose[:2] - before[:2]) > 0.07):
                            request_operator_stop()
                            return

                monitor = threading.Thread(target=stop_after_motion, daemon=True)
                monitor.start()
            outcome["base_before_xyt"] = before.tolist()
            outcome["ready_wall_s"] = time.monotonic() - started
            save()
            if case == TABLE_TRANSFER_CASE:
                import mujoco

                table_robot_model = mujoco.MjModel.from_xml_path(str(robot_model_path))
            if table_preflight_only:
                from emet.eval.stretch_table_observation import assess_live_table_capture

                assert robot.head_to(*initial_gaze, blocking=True) is True, "table head posture unconfirmed"
                not_before = time.monotonic()
                limit = not_before + 5.0
                while time.monotonic() < limit:
                    view = robot.get_observation()
                    timing = (view.image_timing or {}).get("head_cam_left", {}) if view is not None else {}
                    if timing.get("timestamp_ns", 0) / 1e9 > not_before:
                        break
                    time.sleep(.02)
                else:
                    raise RuntimeError("no synchronized table view after head motion")
                record_table_capture(view)
                # Reuse this exact packet once its original interval has expired.
                expires = view.received_monotonic + 2.02
                time.sleep(max(0.0, expires - time.monotonic()))
                outcome["stale_table_capture"] = assess_live_table_capture(view, table_robot_model)
                outcome["complete"] = True
                return
            prompts = prompts_for(case)
            if not model_endpoint:
                model, model_worker, replies = start_model(output, case, prompts)
                url = f"http://127.0.0.1:{model.server_port}/v1"
            else:
                replies = None
            turns = []
            next_prompt = iter(prompts)

            def read(_prompt):
                prompt = next(next_prompt, None)
                if case == TABLE_TRANSFER_CASE and prompt is None:
                    time.sleep(.5)
                if case == CHANGED_OBJECT_RECOVERY and len(turns) == 2:
                    (output / "scoring-intervention.request").touch()
                    acknowledgement = output / "scoring-intervention.json"
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        if acknowledgement.is_file() and acknowledgement.stat().st_size:
                            if not json.loads(acknowledgement.read_text())["ok"]:
                                raise RuntimeError("fixture intervention failed")
                            break
                        time.sleep(.03)
                    else:
                        raise RuntimeError("fixture intervention was not acknowledged")
                pose = robot.get_base_pose()
                limit = time.monotonic() + 7.0
                while pose is None and time.monotonic() < limit:
                    time.sleep(0.05)
                    pose = robot.get_base_pose()
                if pose is None:
                    raise RuntimeError("missing encoder feedback at turn boundary")
                boundary = {"prompt": prompt, "wall_time": time.time(), "base_xyt": pose.tolist()}
                if case in TERMINAL_FULL_CASES:
                    boundary["completed_model_requests"] = len(list((output / "model-requests").glob("*/receipt.json")))
                if case in OBJECT_RECOVERY_CASES:
                    observation = robot.get_observation()
                    if observation is None or observation.depth is None or observation.rgb is None:
                        raise RuntimeError("missing RGB-D at recovery turn boundary")
                    name = f"recovery-view-{len(turns):02d}.npz"
                    np.savez_compressed(output / name, rgb=observation.rgb, depth=observation.depth,
                                        camera_K=observation.camera_K, camera_pose=observation.camera_pose)
                    boundary["observed_view"] = name
                turns.append(boundary)
                outcome["turn_boundaries"] = turns
                save()
                return prompt

            executor_factory = loop.DynamemTaskExecutor

            def create_with_survey(*args, **kwargs):
                executor = executor_factory(*args, **kwargs)
                vm = executor.agent.voxel_map
                survey = register_launch_floor_survey(
                    survey_points,
                    survey_anchors,
                    onboard_anchors,
                    observation_scope_id=vm.observation_scope_id,
                    survey_captured_at=surveyed_at,
                    anchors_captured_at=anchor_stamp,
                    now=time.monotonic(),
                    start_xy=anchor_obs.gps[:2],
                    radius_m=0.55,
                    max_age_s=90.0,
                    max_anchor_error_m=0.025,
                )
                vm.install_launch_floor_survey(survey)
                executor.agent.update(full_perception=False, observation=anchor_obs)
                if full_chat:
                    from emet.controller.dynamem.head_observation import capture_head_view

                    receipt, first_view = capture_head_view(
                        executor.agent, pan_rad=initial_gaze[0], tilt_rad=initial_gaze[1], require_depth=True
                    )
                    outcome["operator_start_view"] = receipt
                    save()
                    if not receipt["ok"]:
                        raise RuntimeError(f"initial operator view not verified: {receipt['status']}")
                    if case == TABLE_TRANSFER_CASE:
                        record_table_capture(first_view)
                    executor.agent.update(full_perception=case in (*OBJECT_RECOVERY_CASES, TABLE_TRANSFER_CASE),
                                          observation=first_view)
                    np.savez_compressed(
                        output / "operator-start-view.npz", rgb=first_view.rgb, depth=first_view.depth,
                        camera_K=first_view.camera_K, camera_pose=first_view.camera_pose,
                    )
                outcome["launch_survey"] = {
                    "anchor_max_residual_m": survey.max_anchor_residual_m,
                    "points": len(survey.points),
                }
                save()
                return executor

            started_turn = time.monotonic()
            with ExitStack() as stack:
                agent_log = stack.enter_context((output / "agent.log").open("w", buffering=1))
                stack.enter_context(patch.object(loop, "DynamemTaskExecutor", create_with_survey))
                options = {}
                if model_endpoint:
                    url = stack.enter_context(recorded_chat_gateway(model_endpoint, output / "model-requests"))
                    if full_chat:
                        options = {"full_chat": True}
                        if interactive_discord:
                            options["navigation_stop_request"] = request_operator_stop
                        else:
                            registry = loop.get_tools
                            images = []

                            def record_image(message=None, content=None):
                                if content is not None:
                                    name = f"tool-image-{len(images) + 1:02d}.png"
                                    Image.fromarray(content).save(output / name)
                                    images.append({"file": name, "turn": len(turns)})

                            def get_tools_with_image_output(context):
                                context["discord_bot"] = SimpleNamespace(push_task_to_all_channels=record_image)
                                return registry(context)

                            outcome["tool_images"] = images
                            stack.enter_context(patch.object(loop, "get_tools", get_tools_with_image_output))
                    else:
                        model_tools = bounded_model_tools(loop.get_tools, case, navigation_stop_requested=(
                            operator_stop.is_set if operator_stop is not None else None
                        ))
                        stack.enter_context(patch.object(loop, "get_tools", model_tools))
                        options = {"max_tool_rounds": 3, "max_tokens": 160}
                    if case == DISCORD_DRIVE_CASE:
                        options["navigation_stop_request"] = request_operator_stop
                stack.enter_context(redirect_stdout(agent_log))
                stack.enter_context(redirect_stderr(agent_log))
                run_agent_session(robot, None if interactive_discord else read, url, model_name,
                                  measured_floor=True, discord=interactive_discord, **options)
            if interactive_discord:
                outcome.update(complete=True, scoring_status="operator_review_required")
                return
            if len(turns) != len(prompts) + 1 or turns[-1]["prompt"] is not None:
                raise RuntimeError("Agent exited before the final turn boundary")
            if model_endpoint:
                outcome["request_audits"] = [
                    compact_request(json.loads(path.read_text()), index)
                    for index, path in enumerate(sorted((output / "model-requests").glob("request-*/request.json")))
                ]
            command = getattr(robot, "_last_navigation_command", None)
            outcome["navigation_receipt"] = command_receipt(robot, command) if command is not None else None
            after = np.asarray(turns[-1]["base_xyt"], float)
            outcome.update(
                base_after_xyt=after.tolist(),
                displacement_m=float(np.linalg.norm(after[:2] - before[:2])),
                turn_wall_s=time.monotonic() - started_turn,
                model_replies=replies,
                complete=True,
            )

    except KeyboardInterrupt:
        if not interactive_discord:
            raise
        outcome.update(scoring_status="operator_review_required", ended_by="operator_interrupt")
    except Exception as error:
        outcome.update(error=f"{type(error).__name__}: {error}", complete=False)
        raise
    finally:
        monitor_done.set()
        if monitor is not None:
            monitor.join(timeout=1)
        if case in (DISCORD_DRIVE_CASE, DISCORD_FULL_CASE) and robot is not None:
            command = getattr(robot, "_last_navigation_command", None)
            if command is not None:
                try:
                    outcome["navigation_receipt"] = command_receipt(robot, command)
                except RuntimeError as error:
                    outcome["navigation_receipt_error"] = str(error)
        outcome["wall_s"] = time.monotonic() - started
        save()
        try:
            if model is not None:
                model.shutdown()
                model.server_close()
                model_worker.join(timeout=2)
            if robot is not None:
                robot.stop()
        finally:
            terminate_process_tree(server, grace_s=10)


def main():
    if len(sys.argv) == 6 and sys.argv[1] == "--serve":
        serve(sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-offset", type=int, default=2040)
    parser.add_argument("--scene", type=Path, help="Override with an equivalent surveyed fixture source.")
    parser.add_argument("--initial-gaze", nargs=2, type=float, metavar=("PAN", "TILT"),
                        help="Starting head angles in radians, applied after survey registration captures.")
    parser.add_argument("--model-endpoint", help="OpenAI-compatible base URL, including /v1, for model trials.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--table-preflight-only", action="store_true",
                        help="Measure tabletop admission and reject a stale frame without calling the model.")
    args = parser.parse_args()
    if args.table_preflight_only and (args.case != TABLE_TRANSFER_CASE or args.model_endpoint):
        parser.error("--table-preflight-only requires model-table-transfer without a model endpoint")
    if not args.table_preflight_only and bool(args.model_endpoint) != (args.case in MODEL_CASES):
        parser.error("--model-endpoint is required for model cases and unavailable for scripted controls")
    if args.case in DISCORD_CASES:
        channel_id = os.getenv("EMET_DISCORD_CHANNEL_ID", "").strip()
        if not (channel_id.isascii() and channel_id.isdecimal() and int(channel_id) > 0):
            parser.error("model-discord requires the exact private channel ID in EMET_DISCORD_CHANNEL_ID")
        if not os.getenv("DISCORD_TOKEN", "").strip():
            parser.error("model-discord requires DISCORD_TOKEN in the environment")
    initial_gaze = args.initial_gaze
    if initial_gaze is None:
        initial_gaze = (-1.48, -0.76) if args.case in FULL_CHAT_CASES else (0.0, -0.2)
    run(args.case, args.output, args.port_offset, args.scene, model_endpoint=args.model_endpoint,
        model_name=args.model if args.model_endpoint else MODEL_NAME, initial_gaze=initial_gaze,
        table_preflight_only=args.table_preflight_only)
    if args.case in DISCORD_CASES or args.table_preflight_only:
        return
    from score_stretch_navigation_chat import score_case, score_model_case

    score = score_model_case if args.model_endpoint else score_case
    print(json.dumps(score(args.output), indent=2))


if __name__ == "__main__":
    main()
