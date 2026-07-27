# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ZMQ REQ client for the Molmo grasp-oracle service."""

from __future__ import annotations

from typing import Any

import numpy as np
import zmq

from emet.perception.grasps.oracle import GraspPose
from emet.perception.grasps.zmq_protocol import decode_pose_4x4, make_predict_request
from emet.perception.grasps.zmq_server import DEFAULT_GRASP_ORACLE_BIND


class GraspOracleClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_GRASP_ORACLE_BIND,
        *,
        timeout_ms: int = 5000,
    ) -> None:
        self.endpoint = str(endpoint)
        self.timeout_ms = int(timeout_ms)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

    def close(self) -> None:
        try:
            self._sock.close(0)
        except Exception:
            pass

    def ping(self) -> bool:
        self._sock.send_json({"op": "ping"})
        rep = self._sock.recv_json()
        return bool(isinstance(rep, dict) and rep.get("ok"))

    def predict(
        self,
        *,
        object_pose_4x4: np.ndarray,
        asset_id: str | None = None,
        body_name: str | None = None,
        category: str | None = None,
        top_k: int | None = 32,
        tcp_frame: str | None = None,
    ) -> list[GraspPose]:
        req = make_predict_request(
            asset_id=asset_id,
            body_name=body_name,
            category=category,
            object_pose_4x4=object_pose_4x4,
            top_k=top_k,
            tcp_frame=tcp_frame,
        )
        self._sock.send_json(req)
        rep: dict[str, Any] = self._sock.recv_json()
        if not isinstance(rep, dict) or not rep.get("ok"):
            raise RuntimeError(f"grasp-oracle error: {(rep or {}).get('error')!r}")
        out: list[GraspPose] = []
        for g in rep.get("grasps") or []:
            out.append(
                GraspPose(
                    T_world=decode_pose_4x4(g["T_world"]),
                    score=float(g.get("score", 1.0)),
                    asset_id=str(g.get("asset_id") or ""),
                    gripper=str(g.get("gripper") or "droid"),
                )
            )
        return out
