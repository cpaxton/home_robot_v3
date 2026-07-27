# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""JSON message schema for the Molmo grasp-oracle ZMQ REP service."""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_pose_4x4(T: np.ndarray) -> list[list[float]]:
    M = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return [[float(M[i, j]) for j in range(4)] for i in range(4)]


def decode_pose_4x4(raw: Any) -> np.ndarray:
    M = np.asarray(raw, dtype=np.float64).reshape(4, 4)
    return M


def make_predict_request(
    *,
    asset_id: str | None = None,
    body_name: str | None = None,
    category: str | None = None,
    object_pose_4x4: np.ndarray | list | None = None,
    top_k: int | None = 32,
    tcp_frame: str | None = None,
) -> dict[str, Any]:
    if object_pose_4x4 is None:
        raise ValueError("object_pose_4x4 required")
    req: dict[str, Any] = {
        "op": "predict",
        "object_pose_4x4": encode_pose_4x4(object_pose_4x4),
    }
    if asset_id:
        req["asset_id"] = str(asset_id)
    if body_name:
        req["body_name"] = str(body_name)
    if category is not None:
        req["category"] = str(category)
    if top_k is not None:
        req["top_k"] = int(top_k)
    if tcp_frame:
        req["tcp_frame"] = str(tcp_frame)
    return req


def make_ok_reply(grasps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"ok": True, "grasps": grasps, "error": None}


def make_error_reply(message: str) -> dict[str, Any]:
    return {"ok": False, "grasps": [], "error": str(message)}
