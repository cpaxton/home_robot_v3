# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Extract pick/place task specs + reachability priors from MolmoSpaces scenes.

Loads a MolmoSpaces ``_physics_metadata.json`` (iTHOR / ProcTHOR) and enumerates
the information an MCTS / TAMP pick-and-place planner needs from a *real scene*:

- **Pickable objects**: non-static bodies with a DROID grasp asset; the GT body is
  the freejoint parent (``*_1_0_0``) used by ``sim_object_placements``.
- **Receptacles**: static bodies exposing ``*Receptacle*`` sites.
- **Task specs**: ``{pick, start_recep, goal_recep, object_gt_body, ...}`` in the
  same schema as ``configs/ovmm/full_episodes.yaml``.
- **Reachability priors**: per object, whether an ``ArmManipProfile``'s gripper can
  reach the object pose (via MuJoCo IK), plus world distances to receptacles --
  ready to seed ``PickPlaceDistancePolicy``.

No VLM, no sim server, no GPU: metadata + grasp assets + vendored robot MJCF only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Freejoint parent suffix: ``<name>_1_0_0`` is the movable root, ``<name>_1_1_0`` the
# mesh child. sim_object_placements keyed by the parent body.
_FREEJOINT_PARENT_SUFFIX = "_1_0_0"
_MESH_CHILD_SUFFIX = "_1_1_0"
_RECEPTACLE_TOKEN = "receptacle"


@dataclass(frozen=True)
class SceneObject:
    """One scene body from MolmoSpaces metadata."""

    body: str  # freejoint parent body name (sim_object_placements key)
    category: str
    asset_id: str
    is_static: bool
    room_id: str
    mesh_child: str  # mesh child body (``*_1_1_0``) or body when absent
    receptacle_sites: tuple[str, ...] = field(default_factory=tuple)
    has_grasps: bool = False


@dataclass(frozen=True)
class SceneTaskSpec:
    """One pick-and-place task, matching ``FindPhaseEpisode`` / ``full_episodes.yaml``."""

    id: str
    tier: str
    sim: str
    object: str
    start_recep: str
    goal_recep: str
    success_radius_m: float = 0.75
    explore_steps: int = 0
    object_gt_body: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tier": self.tier,
            "sim": self.sim,
            "object": self.object,
            "start_recep": self.start_recep,
            "goal_recep": self.goal_recep,
            "success_radius_m": self.success_radius_m,
            "explore_steps": self.explore_steps,
            "object_gt_body": self.object_gt_body,
        }


def default_molmospaces_scenes_dir() -> Path:
    """``$MLSPACES_ASSETS_DIR/scenes`` or ``~/.cache/molmospaces/assets/scenes``."""
    env = os.environ.get("MLSPACES_ASSETS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve() / "scenes"
    return Path.home() / ".cache" / "molmospaces" / "assets" / "scenes"


def _freejoint_parent(body: str) -> str:
    """Coerce a body name to its freejoint parent key (``*_1_0_0``)."""
    if body.endswith(_MESH_CHILD_SUFFIX):
        return body[: -len(_MESH_CHILD_SUFFIX)] + _FREEJOINT_PARENT_SUFFIX
    if body.endswith(_FREEJOINT_PARENT_SUFFIX):
        return body
    return f"{body}{_FREEJOINT_PARENT_SUFFIX}"


def load_scene_metadata(metadata_path: str | Path) -> dict:
    """Load a MolmoSpaces ``*_physics_metadata.json`` into the ``{"objects": {...}}`` dict."""
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"no MolmoSpaces metadata at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, dict) or not objects:
        raise ValueError(f"malformed MolmoSpaces metadata (no objects dict) in {path}")
    return payload


def scene_objects(metadata: dict, *, check_grasps: bool = True) -> list[SceneObject]:
    """Enumerate scene objects from parsed metadata.

    ``check_grasps`` looks up DROID grasp assets (default on; pass False to skip the
    filesystem walk, e.g. for receptacle inventory only).
    """
    from emet.perception.grasps.molmo_grasp_library import has_grasps_for_asset

    objects = metadata.get("objects", {})
    out: list[SceneObject] = []
    for body, info in objects.items():
        if not isinstance(info, dict):
            continue
        parent = _freejoint_parent(body)
        category = str(info.get("category") or "")
        asset_id = str(info.get("asset_id") or "")
        is_static = bool(info.get("is_static", False))
        room_id = str(info.get("room_id") or "")
        name_map = info.get("name_map") or {}
        bodies = name_map.get("bodies") or {}
        sites = name_map.get("sites") or {}
        mesh_child = next((b for b in bodies if b.endswith(_MESH_CHILD_SUFFIX)), parent)
        receptacle_sites = tuple(
            site_name
            for site_name in sites
            if _RECEPTACLE_TOKEN in site_name.lower() or _RECEPTACLE_TOKEN in str(sites[site_name]).lower()
        )
        has_grasps = has_grasps_for_asset(asset_id) if (check_grasps and asset_id) else False
        out.append(
            SceneObject(
                body=parent,
                category=category,
                asset_id=asset_id,
                is_static=is_static,
                room_id=room_id,
                mesh_child=mesh_child,
                receptacle_sites=receptacle_sites,
                has_grasps=has_grasps,
            )
        )
    return out


def pickable_objects(objects: list[SceneObject]) -> list[SceneObject]:
    """Non-static objects with a grasp asset (candidates for pick)."""
    return [o for o in objects if not o.is_static and o.has_grasps]


def receptacle_objects(objects: list[SceneObject]) -> list[SceneObject]:
    """Static objects exposing at least one receptacle site (candidates for place)."""
    return [o for o in objects if o.receptacle_sites]


def category_label(object_: SceneObject) -> str:
    """Lower-cased category as the episode ``object`` / ``start_recep`` label."""
    return object_.category.lower()


def emit_tasks(
    objects: list[SceneObject],
    *,
    sim: str,
    tier: str = "S2",
    success_radius_m: float = 0.75,
    include_self_place: bool = False,
) -> list[SceneTaskSpec]:
    """Emit ``pick X from <its start> to <receptacle>`` tasks from a scene.

    Every pickable object becomes a task whose start receptacle is the object's own
    containing body (or its category label when unknown) and whose goal receptacle
    is each receptacle body in the scene.
    """
    picks = pickable_objects(objects)
    recepts = receptacle_objects(objects)
    tasks: list[SceneTaskSpec] = []
    seen: set[tuple[str, str]] = set()
    for pick in picks:
        start = category_label(pick)
        for recep in recepts:
            if not include_self_place and recep.category.lower() == pick.category.lower():
                continue
            key = (start, category_label(recep))
            if key in seen:
                continue
            seen.add(key)
            tasks.append(
                SceneTaskSpec(
                    id=f"{pick.body.split('_1_0_0')[0].split('_1_0_0')[0]}_to_{recep.body.split('_1_0_0')[0]}",
                    tier=tier,
                    sim=sim,
                    object=category_label(pick),
                    start_recep=start,
                    goal_recep=category_label(recep),
                    success_radius_m=success_radius_m,
                    object_gt_body=pick.body,
                )
            )
    return tasks


@dataclass(frozen=True)
class ObjectReachability:
    """Gripper-contact reachability + geometry for one pickable object."""

    body: str
    category: str
    reachable: bool
    ee_error_m: float
    contact_dist_m: float
    dist_to_start_recep_m: float | None = None
    dist_to_goal_recep_m: float | None = None


def compute_reachability_priors(
    objects: list[SceneObject],
    *,
    robot_id: str = "sourccey",
    arm: str = "left",
    object_pose_fn=None,
) -> dict[str, ObjectReachability]:
    """Per-pickable-object reachability via ArmManipProfile + MuJoCo IK.

    ``object_pose_fn(body) -> np.ndarray(3)`` supplies each object's world position
    (defaults to all zeros when omitted so the *profile wiring* is still verified).
    Returned dict keyed by object body, ready to seed ``PickPlaceDistancePolicy``.
    """
    from emet.motion.arm_manip_profile import ArmManipProfile
    from emet.motion.mujoco_arm_ik import solve_position_ik_multiseed

    profile = ArmManipProfile.for_robot(robot_id, arm=arm)
    import mujoco

    from emet.robots import get_robot_spec

    spec = get_robot_spec(robot_id)
    assert spec is not None and spec.mjcf_path
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    out: dict[str, ObjectReachability] = {}
    for obj in pickable_objects(objects):
        pose = object_pose_fn(obj.body) if object_pose_fn else np.zeros(3, dtype=np.float64)
        pose = np.asarray(pose, dtype=np.float64).reshape(3)
        # Seed from the current EE pose so IK starts near a feasible configuration.
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, profile.ee_body)
        if ee_id >= 0:
            home = np.array(data.body(ee_id).xpos, dtype=np.float64).copy()
            seed = home + np.array([0.0, 0.02, 0.0], dtype=np.float64)
        else:
            seed = pose
        result = solve_position_ik_multiseed(
            model,
            data,
            ee_body=profile.ee_body,
            joint_names=list(profile.joint_names),
            target_pos=seed if np.allclose(pose, 0) else pose,
            max_iters=200,
            tol_m=0.05,
        )
        # Contact distance: nearest gripper contact body to the object after IK.
        contact = float("inf")
        for body in profile.gripper_contact_bodies():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid >= 0:
                contact = min(contact, float(np.linalg.norm(np.array(data.body(bid).xpos) - pose)))
        out[obj.body] = ObjectReachability(
            body=obj.body,
            category=obj.category,
            reachable=result.success and contact <= 0.16,
            ee_error_m=result.pos_error_m,
            contact_dist_m=contact,
        )
    return out
