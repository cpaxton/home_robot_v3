"""Versioned task definitions and independent world-state scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def default_suite() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "benchmarks" / "agent_tasks.yaml"


def _vector(value: Any, n: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == n
        and all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
    )


def load_suite(path: str | Path | None = None) -> dict:
    source = Path(path or default_suite())
    suite = yaml.safe_load(source.read_text())
    if not isinstance(suite, dict) or suite.get("schema_version") != 1:
        raise ValueError("expected agent-task schema_version: 1")
    if suite.get("robot") != "rby1":
        raise ValueError("the initial fixture adapter is certified only for the rby1 asset")
    scene = suite["scene"]
    if scene.get("kind") != "three_room_fixture":
        raise ValueError("unsupported scene kind")
    if not _vector(scene.get("start_xyt"), 3):
        raise ValueError("scene.start_xyt must contain three finite numbers")
    rooms = scene["rooms"]
    if len(rooms) < 2:
        raise ValueError("multi-room suite requires at least two rooms")
    for room in rooms:
        bounds = room.get("bounds")
        if not _vector(bounds, 4) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("invalid room bounds")
    objects = scene["objects"]
    for name, obj in objects.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not _vector(obj.get("position"), 3):
            raise ValueError("invalid object ID or position")
    ids = set()
    episodes = suite.get("episodes", [])
    if not episodes:
        raise ValueError("suite requires episodes")
    for episode in episodes:
        eid = episode["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", eid) or eid in ids:
            raise ValueError("episode IDs must be unique safe identifiers")
        ids.add(eid)
        if not episode.get("instruction") or not episode.get("goals"):
            raise ValueError("episode needs instruction and nonempty goals")
        goal_objects = set()
        for goal in episode["goals"]:
            if goal["object"] not in objects or goal["object"] in goal_objects:
                raise ValueError("unknown or repeated goal object")
            goal_objects.add(goal["object"])
            if goal["destination"] not in objects or objects[goal["destination"]]["movable"]:
                raise ValueError("destination must be a fixed receptacle")
            bounds = goal["bounds"]
            if not isinstance(bounds, list) or len(bounds) != 2 or not all(_vector(b, 3) for b in bounds):
                raise ValueError("goal bounds must be two finite XYZ vectors")
            if any(a >= b for a, b in zip(*bounds, strict=False)):
                raise ValueError("goal bounds must have positive volume")
        for event in episode.get("changes", []):
            if event["object"] not in objects or not objects[event["object"]]["movable"]:
                raise ValueError("world changes must name a movable object")
            if not _vector(event["position"], 3):
                raise ValueError("world change needs finite XYZ")
            if event["after_goal"] not in goal_objects:
                raise ValueError("change trigger must name a goal object")
        if any(g["passed"] for g in score(episode, {k: v["position"] for k, v in objects.items()})["goals"]):
            raise ValueError("a goal is already satisfied in the initial scene")
    return suite


def select_episode(suite: dict, episode_id: str) -> dict:
    for episode in suite["episodes"]:
        if episode["id"] == episode_id:
            return episode
    raise ValueError(f"unknown episode {episode_id!r}")


def score(episode: dict, positions: dict, *, held: list[str] | None = None) -> dict:
    """Score measured object centers inside explicit placement regions, never tool text.

    These are placement-region predicates, not a physical grasp/stability metric.
    A held, missing, or nonfinite object cannot satisfy a goal.
    """
    rows = []
    for goal in episode["goals"]:
        position = positions.get(goal["object"])
        lo, hi = goal["bounds"]
        passed = (
            _vector(position, 3)
            and goal["object"] not in (held or [])
            and all(a <= p <= b for a, p, b in zip(lo, position, hi, strict=False))
        )
        rows.append({"object": goal["object"], "destination": goal["destination"], "passed": bool(passed)})
    complete = sum(r["passed"] for r in rows)
    return {"success": bool(rows) and complete == len(rows), "completed": complete, "total": len(rows), "goals": rows}


def policy_task(episode: dict) -> dict:
    """Only intentional task instructions cross this boundary, never solution or goals."""
    return {"episode_id": episode["id"], "instruction": episode["instruction"]}
