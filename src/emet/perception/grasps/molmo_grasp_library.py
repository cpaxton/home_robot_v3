# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Load MolmoSpaces grasp NPZ/JSON assets without importing molmo_spaces (numpy-only)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from emet.perception.grasps.asset_id import resolve_asset_id_against_grasps_dir


def default_grasps_dir() -> Path:
    """``$MLSPACES_ASSETS_DIR/grasps`` or ``~/.cache/molmospaces/assets/grasps``."""
    env = os.environ.get("MLSPACES_ASSETS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve() / "grasps"
    try:
        from emet.simulation.molmospaces_config import default_molmospaces_assets_dir

        return default_molmospaces_assets_dir() / "grasps"
    except Exception:
        return Path.home() / ".cache" / "molmospaces" / "assets" / "grasps"


def _tcp_frame_matrix(tcp_frame: str) -> np.ndarray:
    """Gripper TCP correction applied after object-local grasp (Molmo convention)."""
    frame = str(tcp_frame or "droid").lower().strip()
    if frame in ("droid", "identity", "none", ""):
        return np.eye(4, dtype=np.float64)
    if frame == "rum":
        # Match molmo_spaces.utils.grasp_sample: RUM_BASE_TCP @ ROT_Z_90
        rot_z = np.eye(4, dtype=np.float64)
        rot_z[:3, :3] = R.from_euler("z", 90, degrees=True).as_matrix()
        rum = np.eye(4, dtype=np.float64)
        rum[:3, 3] = np.array([0.0, 0.0, 0.12], dtype=np.float64)
        return rum @ rot_z
    raise ValueError(f"unknown tcp_frame={tcp_frame!r} (supported: droid, rum)")


def load_grasp_transforms(asset_id: str, grasps_dir: Path | None = None) -> tuple[str, np.ndarray]:
    """Load object-local grasp transforms for ``asset_id``.

    Returns ``(gripper_kind, transforms)`` with ``transforms`` shape ``(N, 4, 4)``.
    """
    root = Path(grasps_dir) if grasps_dir is not None else default_grasps_dir()
    candidates = [
        (root / "droid" / asset_id / f"{asset_id}_grasps_filtered.npz", "droid"),
        (root / "droid_objaverse" / asset_id / f"{asset_id}_grasps_filtered.npz", "droid"),
        (root / "rum" / asset_id / f"{asset_id}_grasps_filtered.json", "rum"),
    ]
    for path, kind in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".npz":
            data = np.load(path)
            transforms = np.asarray(data.get("transforms", []), dtype=np.float64)
        else:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            transforms = np.asarray(payload.get("transforms", []), dtype=np.float64)
        if transforms.size == 0:
            continue
        if transforms.ndim == 2 and transforms.shape == (4, 4):
            transforms = transforms.reshape(1, 4, 4)
        if transforms.ndim != 3 or transforms.shape[-2:] != (4, 4):
            raise ValueError(f"bad grasp transforms shape {transforms.shape} in {path}")
        return kind, transforms.astype(np.float64)
    raise FileNotFoundError(f"no grasp file for asset_id={asset_id!r} under {root}")


def has_grasps_for_asset(asset_id: str, grasps_dir: Path | None = None) -> bool:
    try:
        load_grasp_transforms(asset_id, grasps_dir=grasps_dir)
        return True
    except FileNotFoundError:
        return False


def resolve_and_load(
    body_name: str,
    *,
    category: str | None = None,
    grasps_dir: Path | None = None,
) -> tuple[str, str, np.ndarray]:
    """Resolve asset id from body/category and load transforms.

    Returns ``(asset_id, gripper_kind, transforms)``.
    """
    root = Path(grasps_dir) if grasps_dir is not None else default_grasps_dir()
    asset_id = resolve_asset_id_against_grasps_dir(body_name, root, category=category)
    if not asset_id:
        raise FileNotFoundError(f"no grasp asset for body={body_name!r} cat={category!r} under {root}")
    kind, transforms = load_grasp_transforms(asset_id, grasps_dir=root)
    return asset_id, kind, transforms


def grasps_to_world(
    T_obj_world: np.ndarray,
    transforms_obj: np.ndarray,
    *,
    tcp_frame: str = "droid",
    include_z_flip: bool = True,
) -> np.ndarray:
    """Object-local grasps → world-frame 4x4 poses (optionally + 180° Z flips)."""
    T_obj = np.asarray(T_obj_world, dtype=np.float64).reshape(4, 4)
    local = np.asarray(transforms_obj, dtype=np.float64)
    if local.ndim == 2:
        local = local.reshape(1, 4, 4)
    tcp = _tcp_frame_matrix(tcp_frame)
    world = np.einsum("ij,njk,kl->nil", T_obj, local, tcp)
    if not include_z_flip:
        return world
    flip = np.eye(4, dtype=np.float64)
    flip[:3, :3] = R.from_euler("z", 180, degrees=True).as_matrix()
    flipped = world.copy()
    flipped[:, :3, :3] = flipped[:, :3, :3] @ flip[:3, :3]
    return np.concatenate([world, flipped], axis=0)


def pose_matrix_from_pos_quat(pos: np.ndarray | list[float], quat_wxyz: np.ndarray | list[float]) -> np.ndarray:
    """Build 4x4 from world position + MuJoCo wxyz quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    # MuJoCo: wxyz; scipy: xyzw
    T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return T
