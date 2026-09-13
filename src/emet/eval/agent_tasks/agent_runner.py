"""Run the shared agent task mode on camera-observed fixture state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from .fixture import FixtureRobot
from .recording import Recorder, write_json
from .spec import fingerprint, score


class ObservedFixture:
    """Environment skills and an explicit, observation-derived object cache.

    It is neither the learned DynaGraph backend nor a ground-truth inventory.
    Only rendered visible objects enter memory. Ground-truth labels and known
    room topology remain declared assistance for this first policy baseline.
    """

    def __init__(self, robot, recorder, episode):
        self.robot, self.recorder, self.episode = robot, recorder, episode
        self.memory = {}
        self.revisions = []
        self.changes = []
        self.visited = []
        self.observation_ids = []
        self.context = {"robot": robot, "manip_mode": "teleport"}
        self.first_seen = {}
        self.goal_events = {}
        self.response_repairs = 0
        self.pending_plans = {}
        self.deliveries = []

    def room_at(self, position):
        return next(
            (
                r["id"]
                for r in self.robot.scene["rooms"]
                if r["bounds"][0] < position[0] < r["bounds"][2] and r["bounds"][1] < position[1] < r["bounds"][3]
            ),
            "doorway",
        )

    def snapshot(self, kind, policy=None, *, images=True, command_id=None):
        measured = self.robot.positions()
        self.recorder.append(
            kind,
            policy={"memory": copy.deepcopy(self.memory), **(policy or {})},
            evaluator={
                "positions": measured,
                "robot_xyt": self.robot.xyt[:],
                "score": score(self.episode, measured, held=self.robot.held),
                "applied_changes": self.changes[:],
                "cameras": self.robot.camera_metadata(),
            },
            images=self.robot.images() if images else {},
            command_id=command_id,
        )

    def observe(self):
        visible = self.robot.visible_objects()
        oid = f"obs:{self.recorder.step}"
        room = self.room_at(self.robot.xyt)
        if room not in self.visited:
            self.visited.append(room)
        for name, value in visible.items():
            old = self.memory.get(name)
            if old and math.dist(old["position"], value["position"]) > 0.1:
                self.revisions.append(
                    {
                        "object": name,
                        "old": copy.deepcopy(old),
                        "new_position": value["position"],
                        "observation_id": oid,
                    }
                )
            self.memory[name] = {
                "label": self.robot.scene["objects"][name]["label"],
                "room": self.room_at(value["position"]),
                "position": value["position"],
                "observation_id": oid,
            }
            self.first_seen.setdefault(name, self.recorder.step)
        self.observation_ids.append(oid)
        state = {
            "room": room,
            "rooms": [r["id"] for r in self.robot.scene["rooms"]],
            "visited_rooms": self.visited[:],
            "unvisited_rooms": [r["id"] for r in self.robot.scene["rooms"] if r["id"] not in self.visited],
            "visible_objects": [self.robot.scene["objects"][k]["label"] for k in visible],
            "remembered_objects": {v["label"]: v["room"] for v in self.memory.values()},
            "movable_objects": [
                v["label"] for k, v in self.memory.items() if self.robot.scene["objects"][k]["movable"]
            ],
            "receptacles": [
                v["label"] for k, v in self.memory.items() if not self.robot.scene["objects"][k]["movable"]
            ],
            "observation_id": oid,
            "image_observation_ids": [],
            "pending_plans": copy.deepcopy(self.pending_plans),
            "executed_deliveries": copy.deepcopy(self.deliveries),
        }
        self.snapshot(
            "observation",
            {"observation": state, "visible_pixel_counts": {name: value["pixels"] for name, value in visible.items()}},
        )
        return state, None  # This baseline uses a text policy with oracle visible labels.

    def tools(self, skill_interface="plan_execute"):
        from emet.agent.tools import Tool, get_tools

        shared = {t.name: t for t in get_tools(self.context)}

        def navigate_room(room):
            target = next((r for r in self.robot.scene["rooms"] if r["id"] == room), None)
            if target is None:
                return "Unknown room. Choose one of the room names in the observations."
            x = (target["bounds"][0] + target["bounds"][2]) / 2
            self.robot.move_base_to([x, 0, -math.pi / 2])
            state, _ = self.observe()
            return f"Arrived in {room}. Visible objects: {', '.join(state['visible_objects']) or 'none'}."

        def plan_pick_place(object_name, receptacle_name):
            by_label = {v["label"]: (k, v) for k, v in self.memory.items()}
            for label in (object_name, receptacle_name):
                if label not in by_label:
                    return f"Not observed: {label}. Use find_objects(query='{label}') to search camera views before manipulation."
            name, remembered = by_label[object_name]
            if not self.robot.scene["objects"][name]["movable"]:
                return "The source must be a movable object. Use movable_objects from your observations."
            if self.robot.scene["objects"][by_label[receptacle_name][0]]["movable"]:
                return "The destination must be a fixed tray. Use receptacles from your observations."
            if math.dist(remembered["position"], self.robot.positions()[name]) > 0.1:
                return (
                    f"Source observation is stale. Last observed in {remembered['room']}. "
                    f"Use find_objects(query='{object_name}') to locate it again before manipulation."
                )
            before = self.context.get("_tamp_plan_counter", 0)
            response = shared["plan_pick_place"].func(object_name=object_name, receptacle_name=receptacle_name)
            # Keep the full real result in the trace; give the small policy a concise handle receipt.
            self.snapshot("planning_detail", {"result": response}, images=False)
            counter = self.context.get("_tamp_plan_counter", 0)
            ref = f"plan:{counter}"
            if counter > before and ref in self.context.get("_tamp_plans", {}):
                self.pending_plans[ref] = {"object": object_name, "destination": receptacle_name}
                return f"Plan ready: {ref}. Call execute_pick_place_plan with plan_ref={ref}."
            return response

        def execute_pick_place_plan(plan_ref):
            receipt = self.pending_plans.pop(plan_ref, None)
            response = shared["execute_pick_place_plan"].func(plan_ref=plan_ref)
            if receipt and response.startswith("TAMP execution succeeded:"):
                self.deliveries.append({**receipt, "plan_ref": plan_ref})
                return f"{response} Delivered {receipt['object']} to {receipt['destination']}."
            return response

        tools = [
            Tool(
                "navigate_room",
                "Drive to a named room and face its table; reveals a fresh observation.",
                {
                    "type": "object",
                    "properties": {"room": {"type": "string", "enum": [r["id"] for r in self.robot.scene["rooms"]]}},
                    "required": ["room"],
                },
                navigate_room,
                returns_info=True,
            ),
            Tool(
                "plan_pick_place",
                "Plan delivery using remembered object and tray labels, from any room. Returns a plan_ref. "
                "Call this once both names are in remembered_objects; then execute the returned plan_ref.",
                {
                    "type": "object",
                    "properties": {"object_name": {"type": "string"}, "receptacle_name": {"type": "string"}},
                    "required": ["object_name", "receptacle_name"],
                },
                plan_pick_place,
                returns_info=True,
            ),
            Tool(
                "execute_pick_place_plan",
                shared["execute_pick_place_plan"].description,
                shared["execute_pick_place_plan"].parameters,
                execute_pick_place_plan,
                returns_info=True,
            ),
        ]

        def find_objects(query):
            query = str(query).strip().lower()
            if not query:
                return "Search query must name an object."
            rooms = sorted(self.robot.scene["rooms"], key=lambda r: r["id"] in self.visited)
            for room in rooms:
                navigate_room(room["id"])
                visible = self.robot.visible_objects()
                matches = [
                    self.robot.scene["objects"][name]["label"]
                    for name in visible
                    if query in self.robot.scene["objects"][name]["label"].lower()
                ]
                if matches:
                    return f"Observed {', '.join(matches)} in {room['id']}. Ready for manipulation."
            return f"Search finished after {len(rooms)} room views; {query} was not visible."

        search = Tool(
            "find_objects",
            "Search room camera views for a missing or stale object by label. "
            "Stops on a visible match; searches at most all rooms. Use when an object has not been observed.",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            find_objects,
            returns_info=True,
        )
        if skill_interface == "plan_execute":
            return tools + [search]
        if skill_interface != "atomic":
            raise ValueError("unknown skill interface")

        def pick_place(object_name, receptacle_name):
            before = self.context.get("_tamp_plan_counter", 0)
            response = plan_pick_place(object_name, receptacle_name)
            after = self.context.get("_tamp_plan_counter", 0)
            if after <= before:
                return response
            return execute_pick_place_plan(f"plan:{after}")

        return [
            tools[0],
            search,
            Tool(
                "pick_place",
                "Move an observed object onto an observed tray. "
                "Handles pickup, travel and placement from any room. Use remembered object labels.",
                tools[1].parameters,
                pick_place,
                returns_info=True,
            ),
        ]

    def on_event(self, kind, payload):
        if kind == "response_repair":
            self.response_repairs += 1
        command_id = payload.get("command_id")
        self.snapshot(
            kind, payload, images=kind in {"model_input", "tool_result", "agent_finish"}, command_id=command_id
        )
        if kind != "tool_result":
            return
        current = score(self.episode, self.robot.positions(), held=self.robot.held)
        for goal in current["goals"]:
            if goal["passed"]:
                self.goal_events.setdefault(goal["object"], self.recorder.step)
            else:
                self.goal_events.pop(goal["object"], None)
        for change in self.episode.get("changes", []):
            if change not in self.changes and any(
                g["object"] == change["after_goal"] and g["passed"] for g in current["goals"]
            ):
                self.robot.relocate(change["object"], change["position"])
                self.goal_events.pop(change["object"], None)
                self.changes.append(change)
                self.snapshot("world_change")  # no change notification goes to the model


def run_local_agent(
    suite,
    episode,
    output,
    *,
    model="qwen25-3B-Instruct",
    device="cpu",
    max_rounds=24,
    max_tokens=128,
    timeout_s=600,
    client=None,
    skill_interface="atomic",
):
    """No endpoint/provider arguments: only an allowlisted cached local model."""
    if model not in {"qwen35-0.8B", "qwen35-2B", "qwen35-4B", "qwen25-3B-Instruct"}:
        raise ValueError("agent preflight supports cached local models only")
    from emet.agent.task import TASK_INSTRUCTION, run_agent_task

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema_version": 1,
        "suite": suite,
        "episode": episode,
        "source_revision": revision,
        "suite_fingerprint": fingerprint(suite),
        "episode_fingerprint": fingerprint(episode),
        "robot": suite["robot"],
        "seed": suite["seed"],
        "control": "local_agent" if client is None else "contract_mock",
        "mode": "shared_agent_task",
        "integration_level": "shared_agent_task_runtime",
        "assistance": {
            "geometry": "camera_visible_oracle_labels",
            "topology": "known_room_names",
            "navigation": "footprint_checked_kinematic",
            "search": "bounded_camera_visible_room_sweep",
            "manipulation": "object_pose_teleport",
            "memory": "observed_object_cache_not_dynagraph",
            "policy_input": "text_observations",
            "response_parser": "shared_parser_plus_recorded_complete_call_recovery",
            "tool_dispatch": "ordered_batch_with_fresh_navigation_observations",
            "skill_interface": skill_interface,
            "fixture_head_pitch_degrees": suite["scene"]["head_pitch_degrees"],
        },
        "models": {"policy": model, "device": device, "max_tokens": max_tokens, "mock": client is not None},
        "paid_requests_enabled": False,
        "timeout_s": timeout_s,
        "max_rounds": max_rounds,
        "max_actions": 48,
        "max_tools_per_round": 32,
        "implementation_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
        "shared_implementation_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve().parents[2] / name
                for name in ("agent/task.py", "agent/loop.py", "agent/tools.py", "agent/prompt.py", "llms/__init__.py")
            ]
        },
    }
    recorder = Recorder(output, manifest)
    source_dir = output / "source"
    for prefix, hashes in (
        ("eval/agent_tasks", manifest["implementation_sha256"]),
        ("", manifest["shared_implementation_sha256"]),
    ):
        for name in hashes:
            rel = Path(prefix) / name
            destination = source_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((Path(__file__).resolve().parents[2] / rel).read_bytes())
    robot = None
    started = time.monotonic()
    result = {"success": False, "completed": 0, "total": len(episode["goals"])}
    metrics = {"status": "infrastructure_failed", "agent_status": "not_started", "errors": []}
    try:
        robot = FixtureRobot(suite, output, render=True)
        env = ObservedFixture(robot, recorder, episode)
        robot.on_motion = lambda: env.snapshot("motion", images=len(robot.trajectory) % 4 == 0)
        tools = env.tools(skill_interface)
        system_prompt = (
            TASK_INSTRUCTION
            + (
                "\nSkill contract: navigate_room moves only the robot, never an object. "
                "Use navigation to discover missing objects or refresh stale observations. "
                + (
                    "Once the source and destination are remembered, call pick_place to deliver the object. "
                    if skill_interface == "atomic"
                    else "Once the source and destination are remembered, call plan_pick_place, then execute_pick_place_plan. "
                )
                + "The manipulation skill handles travel to both locations. "
                "Do not repeat successful navigation when both labels are already remembered."
            )
            + "\nAvailable tools:\n"
            + json.dumps(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "arguments": t.parameters.get("properties", {}),
                        "required": t.parameters.get("required", []),
                    }
                    for t in tools
                ]
            )
        )
        env.snapshot("start", {"system_prompt": system_prompt, "instruction": episode["instruction"]})
        if client is None:
            import random

            import numpy as np
            import torch

            from emet.llms import get_llm_client

            random.seed(suite["seed"])
            np.random.seed(suite["seed"])
            torch.manual_seed(suite["seed"])
            kwargs = {"quantization": None} if device == "cpu" and model.startswith("qwen35") else {}
            client = get_llm_client(model, prompt=system_prompt, device=device, max_tokens=max_tokens, **kwargs)
            if device == "cpu" and hasattr(client, "model"):
                # AVX2 hosts execute float32 much faster than emulated bfloat16/int4.
                client.model.float()
            config = getattr(getattr(client, "pipe", None), "generation_config", None)
            if config is not None:
                config.do_sample = False
        client.max_tokens = max_tokens
        config = getattr(getattr(client, "model", None), "config", None)
        env.snapshot(
            "model_ready",
            {
                "client_type": type(client).__name__,
                "model_id": getattr(config, "_name_or_path", model),
                "model_revision": getattr(config, "_commit_hash", None),
                "dtype": str(getattr(getattr(client, "model", None), "dtype", "unknown")),
            },
            images=False,
        )
        outcome = run_agent_task(
            episode["instruction"],
            llm_client=client,
            tools=tools,
            robot=robot,
            max_rounds=max_rounds,
            max_actions=48,
            timeout_s=max(1, timeout_s - (time.monotonic() - started)),
            observe=env.observe,
            on_event=env.on_event,
        )
        result = score(episode, robot.positions(), held=robot.held)
        temporal_ok = True
        if episode.get("changes"):
            for change in episode["changes"]:
                obj, trigger = change["object"], change["after_goal"]
                temporal_ok &= (
                    env.first_seen.get(obj, float("inf"))
                    < env.goal_events.get(trigger, -1)
                    < env.goal_events.get(obj, -1)
                )
        metrics.update(
            status="completed"
            if result["success"] and temporal_ok and outcome["status"] == "model_finished"
            else "task_failed",
            agent_status=outcome["status"],
            actions=outcome["actions"],
            model_rounds=outcome["rounds"],
            temporal_order_passed=temporal_ok,
            observed_memory_revisions=env.revisions,
            visited_rooms=env.visited,
            agent_ran=outcome["rounds"] > 0,
            policy_model_executed=not manifest["models"]["mock"] and outcome["rounds"] > 0,
            response_repairs=env.response_repairs,
        )
        env.snapshot("finish", {"agent_outcome": outcome})
    except Exception as exc:
        metrics["errors"].append(f"{type(exc).__name__}: {exc}")
        recorder.append("error", evaluator={"error": metrics["errors"][-1]})
    finally:
        if robot is not None:
            robot.close()
        recorder.close()
    metrics = {
        **result,
        **metrics,
        "success": metrics["status"] == "completed",
        "event_hash": recorder.last_hash,
        "elapsed_s": time.monotonic() - started,
        "evidence_complete": False,
        "paid_cost_usd": 0,
        "physical_success_verified": False,
        "certification": "local_policy_diagnostic",
    }
    write_json(output / "metrics.json", metrics)
    return metrics
