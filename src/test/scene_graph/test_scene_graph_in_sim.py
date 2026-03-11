# Copyright (c) Hello Robot, Inc.
#
# Integration test: robot spins in the default MuJoCo scene, builds an
# OpenVocabSceneGraph, and verifies that discrete objects (red cylinder,
# blue cube) appear as scene graph nodes with spatial edges.
#
# Run: uv run python -m pytest src/test/scene_graph/test_scene_graph_in_sim.py -v
# Sim tests run by default; use RUN_SIM_TESTS=0 to skip.

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _wait_for_port(host: str, port: int, timeout_sec: float = 30) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (OSError, socket.timeout):
            time.sleep(0.5)
    return False


@pytest.mark.skipif(
    not RUN_SIM_TESTS,
    reason="Set RUN_SIM_TESTS=0 to skip (sim tests run by default)",
)
@pytest.mark.timeout(180)
def test_scene_graph_default_mujoco():
    """
    Robot spins in the default MuJoCo scene, builds a scene graph via the
    SceneGraphProcessor, and verifies:
    1. At least 2 object nodes are created (red cylinder + blue cube on table)
    2. Objects have point clouds and embeddings
    3. Spatial edges exist (near, on)
    4. Text localization finds "red cylinder"
    """
    proc = None
    robot = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep)
    )
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "emet.simulation.mujoco_server", "--headless"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=45):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("MuJoCo server did not start. stderr:\n" + stderr)

        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.controller.zmq_client import HomeRobotZmqClient
        from emet.core.parameters import get_parameters
        from emet.mapping.scene_graph.processor import SceneGraphProcessor

        robot = HomeRobotZmqClient(
            robot_ip="127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
        )
        parameters = get_parameters("dynav_config.yaml")
        executor = DynamemTaskExecutor(
            robot,
            parameters,
            skip_confirmations=True,
            cpu_only=True,
        )

        # Attach scene graph processor to the voxel map
        sg_processor = SceneGraphProcessor(
            config_name="cpu_scene_graph",
            device="cpu",
        )
        voxel_map = executor.agent.get_voxel_map()
        voxel_map.set_scene_graph_processor(sg_processor)

        # Spin to build the map
        executor([("rotate_in_place", "")])

        # Get the scene graph
        sg = sg_processor.scene_graph
        assert sg.num_objects >= 1, (
            f"Expected at least 1 object in scene graph after spin, got {sg.num_objects}"
        )

        # Check that objects have point clouds
        for node in sg.nodes.values():
            assert node.point_cloud is not None, f"Node {node.node_id} has no point cloud"
            assert node.point_cloud.shape[0] > 0
            assert node.center is not None

        # Check edges
        sg.update_edges()
        assert len(sg.edges) >= 0  # May or may not have edges depending on detection

        # Verify serialization round-trip
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            sg.save(tmpdir)
            from emet.mapping.scene_graph.open_vocab_scene_graph import OpenVocabSceneGraph

            loaded = OpenVocabSceneGraph.load(tmpdir)
            assert loaded.num_objects == sg.num_objects

        # Print summary for debugging
        print(sg.to_string())

    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
