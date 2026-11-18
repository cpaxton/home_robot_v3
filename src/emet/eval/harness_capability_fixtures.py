# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Controlled development snapshots, using the existing observation geometry.

The planar probe is a contact/ray control, not a robot controller. These scenes
do not certify navigation, physical manipulation, or learned-agent performance.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.memory.vlm_region_grounding import region_depth_mask
from emet.motion.navigation_sweep import validate_navigation_sweep


def load_development_cases(path):
    data = json.loads(Path(path).read_text())
    cases = data.get("episodes", [])
    if data.get("schema_version") != 1 or data.get("split") != "development" or len(cases) != 12:
        raise ValueError("Expected twelve version-1 development cases")
    if len({c["id"] for c in cases}) != 12:
        raise ValueError("Duplicate case ID")
    for case in cases:
        rgba = case.get("target_rgba", [0.05, 0.15, 0.9, 1])
        if len(rgba) != 4 or not np.isfinite(rgba).all() or min(rgba) < 0 or max(rgba) > 1:
            raise ValueError("Finite RGBA in [0,1] required")
        if len(case.get("frames", [])) != 2 or not case.get("question"):
            raise ValueError("Two ordered snapshots and a question required")
        for frame in case["frames"]:
            if (
                len(frame["camera_xyz"]) != 3
                or not np.isfinite(
                    [*frame["camera_xyz"], frame["yaw_deg"], frame.get("pitch_deg", 0), frame["target_y"]]
                ).all()
            ):
                raise ValueError("Finite calibrated camera and target position required")
    return cases


def snapshot_model(case, index):
    """Shared simple geometry from OBS-01--04, with explicit counterfactual states."""
    frame = case["frames"][index]
    root = ET.fromstring("""<mujoco model="capability_development">
      <option gravity="0 0 -9.81"/>
      <visual><global offwidth="512" offheight="384"/></visual>
      <worldbody>
        <light pos="0 0 5" directional="true"/>
        <geom name="floor" type="plane" size="8 8 .01" rgba=".5 .5 .5 1" group="0"/>
        <body name="probe_base" pos="0 0 .2">
          <joint name="x" type="slide" axis="1 0 0"/>
          <joint name="y" type="slide" axis="0 1 0"/>
          <joint name="yaw" type="hinge" axis="0 0 1"/>
          <geom name="footprint" type="box" size=".2 .15 .19" group="3" rgba=".2 .2 .2 1"/>
        </body>
      </worldbody></mujoco>""")
    world = root.find("worldbody")
    if frame["target_present"]:
        ET.SubElement(
            world,
            "geom",
            name="target",
            type="box",
            pos=f"3.5 {frame['target_y']} .8",
            size=".1 .2 .15",
            group="1",
            rgba=" ".join(map(str, case.get("target_rgba", [0.05, 0.15, 0.9, 1]))),
        )
    if case.get("decoy_box"):
        ET.SubElement(
            world, "geom", name="decoy", type="box", pos="3.5 .75 .8", size=".1 .2 .15", group="1", rgba=".9 .1 .05 1"
        )
    if case.get("obstacle"):
        ET.SubElement(world, "geom", name="obstacle", type="box", pos="1.5 0 .2", size=".1 .3 .2", group="2")
    yaw = np.radians(frame["yaw_deg"])
    pitch = np.radians(frame.get("pitch_deg", 0))
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0])
    up = np.cross(right, forward)
    ET.SubElement(
        world,
        "camera",
        name="evidence",
        pos=" ".join(map(str, frame["camera_xyz"])),
        xyaxes=" ".join(map(str, [*right, *up])),
        fovy="60",
    )
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, xml


def target_ray(model, data, frame):
    yaw = np.radians(frame["yaw_deg"])
    pitch = np.radians(frame.get("pitch_deg", 0))
    direction = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    hit = np.array([-1], dtype=np.int32)
    groups = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
    depth = mujoco.mj_ray(model, data, np.array(frame["camera_xyz"], dtype=float), direction, groups, True, -1, hit)
    return float(depth), int(hit[0])


def route_control(model, data):
    """Production swept-route check against actual MuJoCo contacts, not wheel execution."""
    footprint = model.geom("footprint").id
    obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
    space = SimpleNamespace(last_validity={})

    def valid(pose):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        collision = obstacle >= 0 and any({c.geom1, c.geom2} == {footprint, obstacle} for c in data.contact)
        space.last_validity = {"reason": "occupied_footprint" if collision else "ok"}
        return not collision

    space.is_valid = valid
    allowed, reason = validate_navigation_sweep(space, [0, 0, 0], [[2.5, 0, 0]])
    return {"allowed": bool(allowed), "reason": reason, "scope": "CPU contact/swept geometry only"}


def qualify_geometry(case):
    distances = []
    positions = []
    for i, frame in enumerate(case["frames"]):
        model, data, _ = snapshot_model(case, i)
        depth, hit = target_ray(model, data, frame)
        distances.append(depth)
        positions.append(data.geom_xpos[model.geom("target").id].copy() if frame["target_present"] else None)
        if not frame["target_present"] and hit != -1:
            raise ValueError("Absent target unexpectedly intersects the sensor ray")
    if case["family"] in ("recall", "search_act"):
        expected_first = case["variant"] != "unseen"
        if (distances[0] > 0) != expected_first or distances[1] >= 0:
            raise ValueError("Recall intervention does not separate earlier/current evidence")
    if case["family"] == "changed":
        moved = not np.array_equal(positions[0], positions[1])
        if moved != (case["variant"] == "moved"):
            raise ValueError("Changed-scene intervention did not move the target")
    if case["family"] == "insufficient":
        count = sum(model.geom(i).name in ("target", "decoy") for i in range(model.ngeom))
        if count != (2 if case["decoy_box"] else 1):
            raise ValueError("Ambiguity intervention has incorrect box count")
    if case["family"] == "approach":
        admitted = []
        for depth in distances:
            try:
                region_depth_mask(
                    np.full((40, 40), depth), [0, 0, 1000, 1000], [500, 500], min_depth=0.25, max_depth=2.5
                )
                admitted.append(True)
            except ValueError:
                admitted.append(False)
        if admitted != [case["initial_grounding"], True]:
            raise ValueError("Range intervention does not exercise grounding limits")
    route = None
    if case["family"] == "floor":
        model, data, _ = snapshot_model(case, 0)
        route = route_control(model, data)
        if route["allowed"] == case.get("obstacle", False):
            raise ValueError("Occupied/clear route control disagrees with contacts")
        model, data, _ = snapshot_model(case, 1)
        origin = np.array(case["frames"][1]["camera_xyz"], dtype=float)
        direction = np.array([2.5, 0, 0]) - origin
        direction /= np.linalg.norm(direction)
        hit = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(model, data, origin, direction, np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8), True, -1, hit)
        name = model.geom(int(hit[0])).name if hit[0] >= 0 else None
        if name != ("obstacle" if case.get("obstacle") else "floor"):
            raise ValueError("Floor inspection ray does not distinguish obstruction from floor")
        route["inspection_ray_hit"] = name
    return {
        "episode_id": case["id"],
        "family": case["family"],
        "ray_distances_m": distances,
        "route_control": route,
        "status": "geometry_control_passed",
        "active_agent_qualified": False,
        "manipulation_qualified": False,
    }
