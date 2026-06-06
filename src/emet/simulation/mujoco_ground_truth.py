# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Serialize MuJoCo body poses for sanity-checking voxel / graph outputs (especially Robocasa).

The outbound ZMQ recv key is ``emet.core.zmq_protocol.EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class BodyWorldRow:
    body_id: int
    name: str
    x: float
    y: float
    z: float
    yaw_deg: float


def subtree_body_ids(model: mujoco.MjModel, root_body_name: str) -> set[int]:
    """All body IDs whose kinematic chain includes ``root_body_name`` (including the root body)."""
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if rid < 0:
        return set()
    root_body_id = int(rid)
    out: set[int] = set()
    for b in range(model.nbody):
        x = b
        guard = 0
        while x >= 0 and guard < model.nbody + 2:
            guard += 1
            if x == root_body_id:
                out.add(b)
                break
            x = int(model.body_parentid[x])
    return out


def body_yaw_deg_from_xmat(data: mujoco.MjData, bid: int) -> float:
    R = np.asarray(data.body(bid).xmat, dtype=np.float64).reshape(3, 3)
    yaw_rad = float(np.arctan2(R[1, 0], R[0, 0]))
    return float(np.degrees(yaw_rad))


def collect_body_world_rows(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    exclude_body_ids: set[int] | None,
) -> list[BodyWorldRow]:
    mujoco.mj_forward(model, data)
    rows: list[BodyWorldRow] = []
    excluded = exclude_body_ids or set()
    for bid in range(model.nbody):
        if bid in excluded:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not nm or nm == "world":
            continue
        xyz = np.asarray(data.body(bid).xpos, dtype=np.float64)
        yaw = body_yaw_deg_from_xmat(data, bid)
        rows.append(
            BodyWorldRow(
                body_id=int(bid),
                name=str(nm),
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                yaw_deg=yaw,
            )
        )
    rows.sort(key=lambda r: (r.name.lower(), r.body_id))
    return rows


def format_mujoco_ground_truth_text(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    exclude_body_ids: set[int] | None = None,
    title: str = "MuJoCo body snapshot (world frame)",
    header_extra: dict[str, Any] | None = None,
) -> str:
    rows = collect_body_world_rows(model, data, exclude_body_ids=exclude_body_ids)
    lines: list[str] = [
        title,
        f"generated_at_unix={time.time():.6f}",
        "Columns: body_id name x y z yaw_deg (z-up world; yaw around +Z)",
    ]
    if header_extra:
        for k in sorted(header_extra.keys()):
            lines.append(f"{k}={header_extra[k]!r}")
    lines.append("-" * 88)
    for r in rows:
        lines.append(
            f"{r.body_id:4d}  {r.name:48s}  "
            f"x={r.x:8.4f}  y={r.y:8.4f}  z={r.z:8.4f}  yaw_deg={r.yaw_deg:9.4f}"
        )
    lines.append(f"TOTAL_BODIES_LISTED={len(rows)}")
    return "\n".join(lines) + "\n"


def format_mujoco_ground_truth_json(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    exclude_body_ids: set[int] | None = None,
    header_extra: dict[str, Any] | None = None,
) -> str:
    rows = collect_body_world_rows(model, data, exclude_body_ids=exclude_body_ids)
    payload = {
        "generated_at_unix": time.time(),
        "bodies": [
            {"body_id": r.body_id, "name": r.name, "x": r.x, "y": r.y, "z": r.z, "yaw_deg": r.yaw_deg}
            for r in rows
        ],
    }
    if header_extra:
        payload["extras"] = dict(header_extra)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_ground_truth_dump_action_field(raw: Any) -> tuple[str | None, bool, bool]:
    """
    Normalize client payload shapes:

    - ``True`` / ``{}`` → no path (skip)
    - ``"/abs/path.txt"`` → write text
    - ``{"path": "...", "exclude_robot": true, "json": true}``
    """
    if raw is True or raw == {}:
        return None, True, False
    if isinstance(raw, str):
        return raw.strip(), True, raw.strip().lower().endswith(".json")
    if not isinstance(raw, dict):
        return None, True, False
    path_val = raw.get("path") or raw.get("file") or raw.get("filepath")
    path = str(path_val).strip() if path_val is not None else ""
    exclude = bool(raw.get("exclude_robot", True))
    want_json = bool(raw.get("json", False))
    if path.lower().endswith(".json"):
        want_json = True
    return (path if path else None), exclude, want_json


def resolve_exclude_body_ids(
    model: mujoco.MjModel,
    *,
    exclude_robot: bool,
    robot_base_body_name: str | None,
) -> set[int] | None:
    if not exclude_robot or not robot_base_body_name:
        return None
    s = subtree_body_ids(model, robot_base_body_name)
    return s if s else None


def mujoco_ground_truth_write_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    dest: str | Path,
    exclude_robot: bool,
    robot_base_body_name: str | None,
    json: bool,
    extras: dict[str, Any] | None = None,
) -> Path:
    dest_p = Path(dest).expanduser()
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    excl = resolve_exclude_body_ids(model, exclude_robot=exclude_robot, robot_base_body_name=robot_base_body_name)
    if json:
        text = format_mujoco_ground_truth_json(model, data, exclude_body_ids=excl, header_extra=extras)
    else:
        text = format_mujoco_ground_truth_text(model, data, exclude_body_ids=excl, header_extra=extras)
    dest_p.write_text(text, encoding="utf-8")
    return dest_p
