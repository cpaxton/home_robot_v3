# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ZMQ REP server for :class:`MolmoGraspOracle`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import zmq

from emet.perception.grasps.oracle import MolmoGraspOracle
from emet.perception.grasps.zmq_protocol import (
    decode_pose_4x4,
    encode_pose_4x4,
    make_error_reply,
    make_ok_reply,
)
from emet.utils.logger import Logger

logger = Logger(__name__)

DEFAULT_GRASP_ORACLE_BIND = "tcp://127.0.0.1:5558"


def handle_predict_request(oracle: MolmoGraspOracle, req: dict[str, Any]) -> dict[str, Any]:
    try:
        T = decode_pose_4x4(req.get("object_pose_4x4"))
    except Exception as e:
        return make_error_reply(f"bad object_pose_4x4: {e}")
    top_k = req.get("top_k")
    tcp = req.get("tcp_frame")
    asset_id = req.get("asset_id")
    body = req.get("body_name")
    category = req.get("category")
    try:
        if asset_id:
            poses = oracle.predict_from_asset(str(asset_id), T, top_k=top_k, tcp_frame=tcp)
        elif body:
            poses = oracle.predict_for_body(str(body), T, category=category, top_k=top_k, tcp_frame=tcp)
        else:
            return make_error_reply("need asset_id or body_name")
    except Exception as e:
        return make_error_reply(str(e))
    grasps = [
        {
            "T_world": encode_pose_4x4(p.T_world),
            "score": float(p.score),
            "asset_id": p.asset_id,
            "gripper": p.gripper,
        }
        for p in poses
    ]
    return make_ok_reply(grasps)


def serve_grasp_oracle(
    *,
    bind: str = DEFAULT_GRASP_ORACLE_BIND,
    grasps_dir: Path | str | None = None,
    tcp_frame: str = "droid",
) -> None:
    """Block forever serving predict requests on a ZMQ REP socket."""
    oracle = MolmoGraspOracle(grasps_dir=grasps_dir, tcp_frame=tcp_frame)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(bind)
    logger.info(f"grasp-oracle listening on {bind} grasps_dir={oracle.grasps_dir} tcp_frame={tcp_frame}")
    while True:
        try:
            req = sock.recv_json()
        except Exception as e:
            logger.warning(f"grasp-oracle recv failed: {e}")
            continue
        if not isinstance(req, dict):
            sock.send_json(make_error_reply("request must be a JSON object"))
            continue
        op = str(req.get("op") or "predict").lower()
        if op == "ping":
            sock.send_json({"ok": True, "grasps": [], "error": None, "pong": True})
            continue
        if op != "predict":
            sock.send_json(make_error_reply(f"unknown op={op!r}"))
            continue
        sock.send_json(handle_predict_request(oracle, req))
