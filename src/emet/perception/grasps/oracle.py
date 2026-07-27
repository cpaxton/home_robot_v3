# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""In-process MolmoSpaces grasp oracle (fake grasp predictor)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emet.perception.grasps.molmo_grasp_library import (
    default_grasps_dir,
    grasps_to_world,
    has_grasps_for_asset,
    load_grasp_transforms,
    resolve_and_load,
)


@dataclass(frozen=True)
class GraspPose:
    """World-frame grasp candidate."""

    T_world: np.ndarray
    score: float = 1.0
    asset_id: str = ""
    gripper: str = "droid"

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.T_world, dtype=np.float64).reshape(4, 4)[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.T_world, dtype=np.float64).reshape(4, 4)[:3, :3].copy()

    def approach_axis(self) -> np.ndarray:
        """Gripper approach direction in world (default: −Z of grasp frame)."""
        R = self.rotation
        v = -R[:, 2]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


class MolmoGraspOracle:
    """Load Molmo grasp assets and return world-frame poses (robot-agnostic)."""

    def __init__(
        self,
        *,
        grasps_dir: Path | str | None = None,
        tcp_frame: str = "droid",
        include_z_flip: bool = True,
    ) -> None:
        self.grasps_dir = Path(grasps_dir) if grasps_dir is not None else default_grasps_dir()
        self.tcp_frame = str(tcp_frame)
        self.include_z_flip = bool(include_z_flip)

    def has_asset(self, asset_id: str) -> bool:
        return has_grasps_for_asset(asset_id, grasps_dir=self.grasps_dir)

    def predict_from_asset(
        self,
        asset_id: str,
        T_obj_world: np.ndarray,
        *,
        top_k: int | None = None,
        tcp_frame: str | None = None,
    ) -> list[GraspPose]:
        gripper, local = load_grasp_transforms(asset_id, grasps_dir=self.grasps_dir)
        world = grasps_to_world(
            T_obj_world,
            local,
            tcp_frame=tcp_frame or self.tcp_frame,
            include_z_flip=self.include_z_flip,
        )
        poses = [GraspPose(T_world=world[i], score=1.0, asset_id=asset_id, gripper=gripper) for i in range(len(world))]
        if top_k is not None and top_k > 0:
            return poses[: int(top_k)]
        return poses

    def predict_for_body(
        self,
        body_name: str,
        T_obj_world: np.ndarray,
        *,
        category: str | None = None,
        top_k: int | None = None,
        tcp_frame: str | None = None,
    ) -> list[GraspPose]:
        asset_id, gripper, local = resolve_and_load(body_name, category=category, grasps_dir=self.grasps_dir)
        world = grasps_to_world(
            T_obj_world,
            local,
            tcp_frame=tcp_frame or self.tcp_frame,
            include_z_flip=self.include_z_flip,
        )
        poses = [GraspPose(T_world=world[i], score=1.0, asset_id=asset_id, gripper=gripper) for i in range(len(world))]
        if top_k is not None and top_k > 0:
            return poses[: int(top_k)]
        return poses
