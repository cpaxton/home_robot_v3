#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Run production CHAT on the archived PandaOmron sink fixture.

The scorer owns simulator state; the client sees the normal RGB-D protocol.
The default model fixture deterministically checks routing and evidence gates.
A real OpenAI-compatible endpoint can be supplied for an agent behavior run.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from emet.robots.pandaomron import SCENE_CAMERAS
from emet.utils.port_utils import get_ports

SCRIPTED_MODEL = "pandaomron-sink-tools"
PORTS = Path(__file__).resolve().parents[1] / "configs/eval/observed_transfer_ports.json"
PANDA_PORT = json.loads(PORTS.read_text())["robocasa365"]
CASES = {PANDA_PORT[split]["case"]: PANDA_PORT[split] for split in ("development", "held_out")}


def scripted_turns(object_name):
    arguments = {"object_name": object_name, "receptacle_name": "sink"}
    return (
        ("Show me a photo of the counter and sink.", "send_image", {}),
        ("Which tasks can you identify in the current view?", "scene_tasks", {}),
        (
            f"Plan the transfer of the {object_name} into the sink and execute it if the plan is admitted.",
            "plan_pick_place",
            arguments,
        ),
        (
            f"If the scene changed before any motion, plan the {object_name} transfer again and execute only a fresh admitted plan.",
            "plan_pick_place",
            arguments,
        ),
    )


def find_tool_receipt(messages, tool):
    for message in reversed(messages):
        if message.get("role") == "system":
            continue
        content = message.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [item.get("text", "") for item in (content or ()) if item.get("type") == "text"]
        )
        for text in reversed(texts):
            for line in reversed(text.splitlines()):
                marker = f"[{tool}] "
                if marker not in line:
                    continue
                raw = line.rsplit(marker, 1)[1].strip()
                try:
                    receipt, _ = json.JSONDecoder().raw_decode(raw)
                except ValueError:
                    continue
                if isinstance(receipt, dict) and receipt.get("tool") == tool:
                    return receipt
    return None


def find_plan_receipt(messages):
    return find_tool_receipt(messages, 'plan_pick_place')


def score_transfer(physics, tool_results, actions):
    physical = bool(
        not physics['initial_success'] and physics['upstream_success'] and physics['basin_contains_tongs']
        and physics['lift_with_two_pads_m'] >= .05 and physics['final_object_speed_mps'] <= .01
        and not physics['unintended_contact_frames'] and not physics['unintended_self_contact_frames'])
    receipt = tool_results.get('execute_pick_place_plan', {})
    observed = receipt.get('status') == 'ok' and receipt.get('code') == 'ok'
    native = bool(actions) and all(not json.loads(action).get('forbidden') for action in actions)
    return physical, observed, bool(physical and observed and native)


@contextmanager
def scripted_model(output, turns):
    requests = []
    dispatched = set()
    attempted_plans = set()
    last_plan_receipt = None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, document):
            serialized = json.dumps(document).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(serialized)))
            self.end_headers()
            self.wfile.write(serialized)

        def do_GET(self):
            self.reply({"object": "list", "data": [{"id": SCRIPTED_MODEL, "object": "model"}]})

        def do_POST(self):
            nonlocal last_plan_receipt
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append(request)
            (output / f"request-{index:02}.json").write_text(json.dumps(request) + "\n")
            calls = []
            user_texts = []
            for message in request.get("messages", ()):
                if message.get("role") != "user":
                    continue
                content = message.get("content", "")
                user_texts.append(
                    content
                    if isinstance(content, str)
                    else " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                )
            match = next(
                (
                    (prompt, tool, arguments)
                    for prompt, tool, arguments in reversed(turns)
                    if any(prompt in text for text in user_texts)
                ),
                None,
            )
            if match is not None and match[0] not in dispatched:
                prompt, tool, arguments = match
                dispatched.add(prompt)
                if prompt != turns[-1][0] or (
                    last_plan_receipt and last_plan_receipt.get("recovery") == "replan"
                    and last_plan_receipt.get("data", {}).get("execution_started") is False
                    and not attempted_plans
                ):
                    calls.append({"name": tool, "arguments": arguments})
            elif match is not None and match[1] == "plan_pick_place":
                receipt = find_plan_receipt(request.get("messages", ())) or {}
                last_plan_receipt = receipt or last_plan_receipt
                plan_ref = receipt.get("data", {}).get("plan_ref") if receipt.get("status") == "ok" else None
                if isinstance(plan_ref, str) and plan_ref and plan_ref not in attempted_plans:
                    attempted_plans.add(plan_ref)
                    calls.append({"name": "execute_pick_place_plan", "arguments": {"plan_ref": plan_ref}})
            completed = find_tool_receipt(request.get("messages", ()), 'execute_pick_place_plan') or {}
            summary = ("The transfer completed and placement was verified." if completed.get('status') == 'ok'
                       else "The sink transfer is incomplete.")
            message = json.dumps({"tool_calls": calls, "message": "" if calls else summary})
            self.reply(
                {
                    "id": f"sink-{index}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": SCRIPTED_MODEL,
                    "choices": [
                        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": message}}
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            )

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            worker.join()


def serve_fixture(fixture, output, port_offset, case, scene_camera, probe_calibration_path=None, torso_offset_m=0.0, scene_camera_fovy_deg=None):
    import mujoco
    from eval_robocasa_panda_sink import Control

    from emet.robots.pandaomron import TORSO_JOINT
    from emet.simulation.pandaomron_server import PandaOmronZmqServer

    probe_calibration = None
    if probe_calibration_path:
        from emet.motion.guarded_placement_probe import ProbeCalibration

        probe_calibration = ProbeCalibration(**json.loads(probe_calibration_path.read_text()))

    spec = CASES[case]
    layout, style, seed = (spec[name] for name in ("layout", "style", "seed"))
    control = Control(fixture, output, layout=layout, style=style, seed=seed)
    if control.report["object_models"][0] != spec["target_asset"] + "/model.xml":
        raise ValueError("target asset differs from the frozen benchmark case")
    control.env.reset()
    if scene_camera_fovy_deg is not None:
        control.model.cam_fovy[control.model.camera(SCENE_CAMERAS[scene_camera]).id] = scene_camera_fovy_deg
    # This source contains native robot links only. The kitchen stays in the scorer.
    robot_model = control.env.robots[0].robot_model.get_xml()
    (output / "robot-only.xml").write_text(robot_model)
    if case == "development-tongs":
        control.reset_stance(0.92, -2.43, np.pi, 0.16)
        control.settle()
    else:
        control.live.qpos[:] = control.initial
        control.live.qvel.fill(0.0)
    if torso_offset_m:
        joint = control.model.joint(TORSO_JOINT)
        address = joint.qposadr[0]
        requested = control.live.qpos[address] + torso_offset_m
        if not joint.range[0] <= requested <= joint.range[1]:
            raise ValueError("diagnostic torso offset is outside its joint limits")
        control.live.qpos[address] = requested
        control.live.qvel[joint.dofadr[0]] = 0.0
    mujoco.mj_forward(control.model, control.live)
    before = control.live.body("obj_main").xpos.copy()
    initial_task_success = bool(control.env._check_success())
    ports = get_ports(port_offset)
    samples = []

    class ScoredServer(PandaOmronZmqServer):
        def _step_physics_tick(self):
            super()._step_physics_tick()
            with self._mj_lock:
                contacts, pads = control.contacts(control.live, allow_finger_contact=True)
                self_contacts = []
                for contact in control.live.contact:
                    if float(contact.dist) >= -.001:
                        continue
                    first, second = int(contact.geom1), int(contact.geom2)
                    if all(int(control.model.geom_bodyid[g]) in control.robot_bodies for g in (first, second)):
                        self_contacts.append([control.model.geom(g).name for g in (first, second)])
                position = control.live.body("obj_main").xpos
                samples.append(
                    {
                        "sim_s": float(control.live.time),
                        "xyz_m": position.tolist(),
                        "pad_contacts": len(pads),
                        "unintended_contacts": contacts,
                        "unintended_self_contacts": self_contacts,
                        "ee_xyz_m": self._ee_pose(control.live)[:3, 3].tolist(),
                        "robot_joint_positions": control.live.qpos[self._qpos_addresses].tolist(),
                    }
                )

    server = ScoredServer(
        env=control.env,
        environment={"kind": "robocasa", "task": "PickPlaceCounterToSink", "layout": layout, "style": style},
        scene_camera=SCENE_CAMERAS[scene_camera],
        probe_calibration=probe_calibration,
        send_port=ports.send,
        recv_port=ports.recv,
        send_state_port=ports.state,
        send_servo_port=ports.servo,
        use_remote_computer=False,
    )
    try:
        server.start(headless=True)
    except KeyboardInterrupt:
        pass
    finally:
        server._running = False
        with server._mj_lock:
            after = control.live.body("obj_main").xpos.copy()
            verdict = {
                "fixture_sha256": control.report["fixture_sha256"],
                "initial_success": initial_task_success,
                "upstream_success": bool(control.env._check_success()),
                "basin_contains_tongs": control.basin_contains_tongs(),
                "object_motion_m": float(np.linalg.norm(after - before)),
                "maximum_ee_displacement_m": max(
                    (float(np.linalg.norm(np.asarray(sample["ee_xyz_m"]) - samples[0]["ee_xyz_m"]))
                     for sample in samples), default=0.0,
                ),
                "lift_with_two_pads_m": max(
                    (sample["xyz_m"][2] - before[2] for sample in samples if sample["pad_contacts"] == 2),
                    default=0.0,
                ),
                "unintended_contact_frames": sum(bool(sample["unintended_contacts"]) for sample in samples),
                "unintended_self_contact_frames": sum(bool(sample["unintended_self_contacts"]) for sample in samples),
                "final_object_speed_mps": float(
                    np.linalg.norm(control.live.qvel[control.model.jnt_dofadr[control.object_joint] :][:3])
                ),
            }
            (output / "physics-trace.json").write_text(json.dumps(samples) + "\n")
            (output / "physics.json").write_text(json.dumps(verdict, indent=2) + "\n")
        server.stop()


def observe(client, output):
    import mujoco
    from PIL import Image

    from emet.controller.manipulation.capture_clock import local_sim_capture_interval
    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    from emet.controller.task.tamp.observed_capture import capture_arrays
    from emet.robots import get_robot_backend
    from emet.robots.pandaomron.observed import synchronize_pandaomron

    deadline = time.monotonic() + 12.0
    rejected = 0
    while True:
        observation = client.get_observation()
        try:
            timing = (observation.image_timing or {}).get("head_cam_left")
            if timing is None:
                raise ObservationUnavailable("capture_interval_unavailable")
            capture = local_sim_capture_interval(
                timing, received_monotonic=observation.received_monotonic, observation_id=str(timing["timestamp_ns"])
            )
            capture.require(now=time.monotonic())
            break
        except ObservationUnavailable:
            if time.monotonic() >= deadline:
                raise
            rejected += 1
            time.sleep(0.08)
    model = mujoco.MjModel.from_xml_path(str(output / "robot-only.xml"))
    arm = get_robot_backend("pandaomron").bind_observed_arm(client, model)
    data = mujoco.MjData(model)
    synchronize_pandaomron(observation, model, data)
    ee = data.body(arm.chain.ee_body)
    published_ee = np.asarray(observation.ee_pose, dtype=float)
    fk_translation_error_m = float(np.linalg.norm(ee.xpos - published_ee[:3, 3]))
    fk_orientation_error_rad = float(
        np.arccos(np.clip((np.trace(ee.xmat.reshape(3, 3).T @ published_ee[:3, :3]) - 1) / 2, -1, 1))
    )
    if fk_translation_error_m > 0.003 or fk_orientation_error_rad > 0.05:
        raise ObservationUnavailable("robot_capture_model_mismatch")
    rgb, depth, points = capture_arrays(observation)
    Image.fromarray(rgb).save(output / "front.png")
    Image.fromarray(observation.ee_rgb).save(output / "wrist.png")
    for name, view in (observation.auxiliary_rgbd_views or {}).items():
        Image.fromarray(view["rgb"]).save(output / f"{name}.png")
    auxiliary_arrays = {f"{name}_{key}": view[key]
                        for name, view in (observation.auxiliary_rgbd_views or {}).items()
                        for key in ("rgb", "depth", "camera_K", "camera_pose")}
    np.savez_compressed(
        output / "sensor.npz",
        **auxiliary_arrays,
        rgb=rgb,
        depth=depth,
        K=observation.camera_K,
        pose=observation.camera_pose,
        joint_positions=observation.joint,
        robot_qpos=data.qpos,
        wrist_rgb=observation.ee_rgb,
        wrist_depth=observation.ee_depth,
        wrist_K=observation.ee_camera_K,
        wrist_pose=observation.ee_camera_pose,
    )
    return {
        "capture_id": capture.observation_id,
        "scene_camera": (observation.emet_session or {}).get("scene_camera"),
        "auxiliary_camera_names": sorted(observation.auxiliary_rgbd_views or {}),
        "rejected_startup_samples": rejected,
        "camera_joint_count": len(observation.joint),
        "depth_points": int(np.count_nonzero(np.isfinite(points).all(axis=-1) & (depth > 0))),
        "wrist_depth_points": int(np.count_nonzero(observation.ee_depth > 0)),
        "robot_only_fk_translation_error_m": fk_translation_error_m,
        "robot_only_fk_orientation_error_rad": fk_orientation_error_rad,
        "head_wrist_capture_match": timing["timestamp_ns"] == observation.image_timing["ee_cam"]["timestamp_ns"],
    }


def run(args, *, observed_native_factory=None):
    from run_harness_capability_agent import run_agent_session

    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.eval.chat_gateway import recorded_chat_gateway
    from emet.simulation.sim_subprocess import wait_for_sim_tcp_port
    from emet.utils.process_tree import popen_session, terminate_process_tree

    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for port in range(4401 + args.port_offset, 4405 + args.port_offset):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    env = {
        **os.environ,
        "EMET_PHYSICAL_EXECUTION": "1",
        "EMET_PHYSICAL_AUDIT": str(output / "actuation.jsonl"),
    }
    if args.sim_pythonpath:
        env["PYTHONPATH"] = args.sim_pythonpath
    client = None
    requests = []
    live_scenes = []
    grounding_mode = None
    if args.observed_profile is not None:
        if observed_native_factory is not None:
            raise ValueError("provide a single observed transfer factory")
        from emet.eval.pandaomron_live_transfer import panda_transfer_factory

        observed_native_factory, grounding_mode = panda_transfer_factory(
            args.observed_profile, output / "robot-only.xml", CASES[args.case]["object_name"], live_scenes,
        )
    probe_args = ["--probe-calibration", str(args.probe_calibration.resolve())] if args.probe_calibration else []
    fovy = getattr(args, "scene_camera_fovy_deg", None)
    camera_args = ["--scene-camera-fovy-deg", str(fovy)] if fovy is not None else []
    torso_offset = getattr(args, "diagnostic_torso_offset_m", 0.0)
    with (output / "server.log").open("w") as log:
        server = popen_session(
            [
                sys.executable,
                __file__,
                str(args.fixture.resolve()),
                str(output),
                "--port-offset",
                str(args.port_offset),
                "--serve-fixture",
                "--case",
                args.case,
                "--scene-camera",
                args.scene_camera,
                "--diagnostic-torso-offset-m",
                str(torso_offset),
                *probe_args,
                *camera_args,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        interrupted = False

        def terminate_run(_signum, _frame):
            nonlocal interrupted
            interrupted = True
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
            raise SystemExit(143)

        previous_termination_handler = signal.signal(signal.SIGTERM, terminate_run)
        try:
            wait_for_sim_tcp_port("127.0.0.1", 4401 + args.port_offset, proc=server, timeout_sec=60)
            client = create_robot_client_from_cli("pandaomron", "127.0.0.1", port_offset=args.port_offset)
            sensor = observe(client, output)
            session = client.get_emet_session()
            if any(key in session for key in ("sim_object_placements", "sim_articulations", "mujoco_ground_truth")):
                raise RuntimeError("benchmark client received simulator scene metadata")
            object_name = CASES[args.case]["object_name"]
            turns = scripted_turns(object_name)
            prompt = (
                f"Pick up the {object_name} from the counter and place it inside the sink. Keep the base stationary."
            )
            prompts = iter([prompt] if args.model_endpoint else [turn[0] for turn in turns])
            with (
                (output / "agent.log").open("w") as transcript,
                redirect_stdout(transcript),
                redirect_stderr(transcript),
            ):
                if args.model_endpoint:
                    with recorded_chat_gateway(args.model_endpoint, output / "model-requests") as endpoint:
                        run_agent_session(
                            client,
                            lambda _: next(prompts, None),
                            endpoint,
                            args.model,
                            robot_name="pandaomron",
                            max_tokens=args.max_tokens,
                            observed_native_factory=observed_native_factory,
                        )
                else:
                    with scripted_model(output, turns) as (endpoint, requests):
                        run_agent_session(
                            client,
                            lambda _: next(prompts, None),
                            endpoint,
                            SCRIPTED_MODEL,
                            robot_name="pandaomron",
                            observed_native_factory=observed_native_factory,
                        )
        except Exception as error:
            if client is None:
                (output / "run-status.json").write_text(json.dumps({
                    "status": "infrastructure_error", "phase": "startup", "execution_started": False,
                    "exception": type(error).__name__, "reason": str(error),
                }, indent=2)+"\n")
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_termination_handler)
            if client is not None and not interrupted:
                from PIL import Image

                final_observation = client.get_observation()
                Image.fromarray(final_observation.rgb).save(output / "final-front.png")
                Image.fromarray(final_observation.ee_rgb).save(output / "final-wrist.png")
                for name, view in (final_observation.auxiliary_rgbd_views or {}).items():
                    Image.fromarray(view["rgb"]).save(output / f"final-{name}.png")
            for index, scene in enumerate(live_scenes):
                scene.invalidate()
                rejection = getattr(scene, "last_tracking_rejection", None)
                if rejection is not None and rejection[0] is not None:
                    seed, rejected_views, reason = rejection
                    arrays = {f"{name}_{key}": getattr(view, key)
                              for name, view in rejected_views
                              for key in ("rgb", "depth", "world", "foreground_mask")
                              if getattr(view, key) is not None}
                    np.savez_compressed(output / f"tracking-rejection-{index:02}.npz",
                                        seed_poses=seed.poses, **arrays)
                    (output / f"tracking-rejection-{index:02}.json").write_text(json.dumps({
                        "reason": reason, "seed": seed.as_dict(),
                        "views": {name: vars(view.capture) for name, view in rejected_views},
                        "diagnostic_only": True,
                    }, indent=2)+"\n")
            if client is not None:
                client.stop()
            try:
                if not interrupted:
                    server.send_signal(signal.SIGINT)
                server.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                terminate_process_tree(server)

    physics = json.loads((output / "physics.json").read_text())
    transcript = (output / "agent.log").read_text()
    audit = output / "actuation.jsonl"
    actions = audit.read_text().splitlines() if audit.exists() else []
    if args.model_endpoint:
        requests = [
            json.loads(path.read_text()) for path in sorted((output / "model-requests").glob("request-*/request.json"))
        ]
    tool_results = {}
    tool_attempts = []
    for line in transcript.splitlines():
        for tool in ("scene_tasks", "plan_pick_place", "pick_place", "execute_pick_place_plan"):
            marker = f"[{tool}] [{tool}] "
            if line.startswith(marker):
                receipt = json.loads(line[len(marker) :])
                tool_results[tool] = receipt
                tool_attempts.append({"tool": tool, "status": receipt["status"], "code": receipt["code"]})
    physical_passed, observed_passed, passed = score_transfer(physics, tool_results, actions)
    report = {
        "passed": bool(passed),
        "physics_passed": physical_passed,
        "observed_execution_passed": observed_passed,
        "case": args.case,
        "evaluation_split": CASES[args.case].get("evaluation_split", "development"),
        "model": args.model if args.model_endpoint else SCRIPTED_MODEL,
        "routing_mode": "real_model" if args.model_endpoint else "scripted_tool_selection",
        "grounding_mode": grounding_mode,
        "diagnostic_torso_offset_m": torso_offset,
        "scene_camera_fovy_deg_override": fovy,
        "observed_source": live_scenes[0].diagnostics if live_scenes else None,
        "observed_source_blockers": live_scenes[0].blockers if live_scenes else [],
        "sensor": sensor,
        "physics": physics,
        "actuation_count": len(actions),
        "saw_model_image": any(
            item.get("type") == "image_url"
            for req in requests
            for msg in req.get("messages", ())
            if isinstance(msg.get("content"), list)
            for item in msg["content"]
        ),
        "tool_results": tool_results,
        "tool_attempts": tool_attempts,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--port-offset", type=int, default=2436)
    parser.add_argument("--case", choices=CASES, default="development-tongs")
    parser.add_argument("--scene-camera", choices=SCENE_CAMERAS, default="right")
    parser.add_argument("--scene-camera-fovy-deg", type=float, help="Explicit simulated lens change; recorded as sensor assistance.")
    parser.add_argument("--observed-profile", type=Path)
    parser.add_argument("--probe-calibration", type=Path)
    parser.add_argument("--diagnostic-torso-offset-m", type=float, default=0.0)
    parser.add_argument("--sim-pythonpath")
    parser.add_argument("--model-endpoint")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, choices=range(16, 513), metavar="16..512", default=128)
    parser.add_argument("--serve-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 <= args.diagnostic_torso_offset_m <= 0.01:
        parser.error("--diagnostic-torso-offset-m must be between 0 and 0.01 m")
    if args.scene_camera_fovy_deg is not None and not 30 <= args.scene_camera_fovy_deg <= 100:
        parser.error("scene camera vertical field of view must be between 30 and 100 degrees")
    if args.serve_fixture:
        serve_fixture(args.fixture, args.output, args.port_offset, args.case, args.scene_camera,
                      args.probe_calibration, args.diagnostic_torso_offset_m, args.scene_camera_fovy_deg)
    else:
        if bool(args.model_endpoint) != bool(args.model):
            parser.error("--model-endpoint and --model must be supplied together")
        sys.exit(run(args))
