# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for Molmo grasp library / asset id / oracle / ZMQ."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from emet.perception.grasps.asset_id import (
    candidate_asset_ids,
    resolve_asset_id_against_grasps_dir,
    strip_molmo_instance_suffix,
)
from emet.perception.grasps.molmo_grasp_library import (
    grasps_to_world,
    load_grasp_transforms,
    pose_matrix_from_pos_quat,
)
from emet.perception.grasps.oracle import MolmoGraspOracle
from emet.perception.grasps.zmq_client import GraspOracleClient


def test_strip_and_candidates():
    assert strip_molmo_instance_suffix("Apple_4_1_1_0") == "Apple_4"
    cands = candidate_asset_ids("Apple_4_1_1_0", category="apple")
    assert "Apple_4" in cands
    assert any(c.lower().startswith("apple") for c in cands)


def test_load_and_world_transform(tmp_path: Path):
    asset = "Widget_1"
    d = tmp_path / "droid" / asset
    d.mkdir(parents=True)
    T_local = np.eye(4, dtype=np.float64)
    T_local[:3, 3] = [0.01, 0.02, 0.03]
    np.savez(d / f"{asset}_grasps_filtered.npz", transforms=np.stack([T_local]))
    kind, local = load_grasp_transforms(asset, grasps_dir=tmp_path)
    assert kind == "droid"
    assert local.shape == (1, 4, 4)
    T_obj = pose_matrix_from_pos_quat([1.0, 2.0, 0.5], [1.0, 0.0, 0.0, 0.0])
    world = grasps_to_world(T_obj, local, tcp_frame="droid", include_z_flip=True)
    assert world.shape[0] == 2
    assert np.allclose(world[0, :3, 3], [1.01, 2.02, 0.53], atol=1e-5)


def test_resolve_asset_against_dir(tmp_path: Path):
    (tmp_path / "droid" / "Apple_4").mkdir(parents=True)
    assert resolve_asset_id_against_grasps_dir("Apple_4_1_1_0", tmp_path) == "Apple_4"
    assert resolve_asset_id_against_grasps_dir("something", tmp_path, category="apple") == "Apple_4"


def test_oracle_predict(tmp_path: Path):
    asset = "Cup_2"
    d = tmp_path / "droid" / asset
    d.mkdir(parents=True)
    T_local = np.eye(4, dtype=np.float64)
    np.savez(d / f"{asset}_grasps_filtered.npz", transforms=np.stack([T_local, T_local]))
    oracle = MolmoGraspOracle(grasps_dir=tmp_path, include_z_flip=False)
    T_obj = np.eye(4)
    T_obj[:3, 3] = [0.5, 0.0, 0.8]
    poses = oracle.predict_from_asset(asset, T_obj, top_k=1)
    assert len(poses) == 1
    assert np.allclose(poses[0].position, [0.5, 0.0, 0.8])


def test_zmq_roundtrip(tmp_path: Path):
    import zmq

    from emet.perception.grasps.zmq_server import handle_predict_request

    asset = "Bowl_9"
    d = tmp_path / "droid" / asset
    d.mkdir(parents=True)
    T_local = np.eye(4, dtype=np.float64)
    T_local[:3, 3] = [0.0, 0.0, 0.05]
    np.savez(d / f"{asset}_grasps_filtered.npz", transforms=np.stack([T_local]))
    oracle = MolmoGraspOracle(grasps_dir=tmp_path, include_z_flip=False)

    bind = "tcp://127.0.0.1:19558"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(bind)
    done = threading.Event()

    def _serve_two():
        try:
            for _ in range(2):
                req = sock.recv_json()
                if req.get("op") == "ping":
                    sock.send_json({"ok": True, "grasps": [], "error": None, "pong": True})
                else:
                    sock.send_json(handle_predict_request(oracle, req))
        except zmq.Again:
            pass
        finally:
            done.set()

    t = threading.Thread(target=_serve_two, daemon=True)
    t.start()
    time.sleep(0.1)
    client = GraspOracleClient(bind, timeout_ms=3000)
    assert client.ping()
    T_obj = pose_matrix_from_pos_quat([0.0, 0.0, 1.0], [1, 0, 0, 0])
    poses = client.predict(object_pose_4x4=T_obj, asset_id=asset, top_k=4)
    client.close()
    done.wait(timeout=3)
    sock.close(0)
    t.join(timeout=1)
    assert len(poses) >= 1
    assert poses[0].position[2] == pytest.approx(1.05, abs=1e-4)


def test_cli_grasp_oracle_help():
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-m", "emet.cli", "grasp-oracle", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert r.returncode == 0
    assert "tcp-frame" in r.stdout
    assert "bind" in r.stdout
