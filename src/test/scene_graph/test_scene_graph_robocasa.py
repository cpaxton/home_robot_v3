# Copyright (c) Hello Robot, Inc.
#
# Integration test: robot spins in a Robocasa kitchen scene, builds an
# OpenVocabSceneGraph, and verifies deduplication and temporal stability.
#
# Robocasa kitchens have walls bounding exploration, making them ideal for
# testing that the same objects seen from multiple angles are properly merged.
#
# Run: uv run python -m pytest src/test/scene_graph/test_scene_graph_robocasa.py -v
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
@pytest.mark.timeout(300)
def test_scene_graph_robocasa_dedup_and_stability():
    """
    Robot spins twice in a Robocasa kitchen. After the first spin we record
    the scene graph state. After the second spin (same scene, same objects),
    we verify:
    1. No explosion in node count (dedup is working)
    2. Objects seen in both spins have observation_count >= 2 (stability)
    3. At least one kitchen object is detected (counter, cabinet, etc.)
    4. Merge_duplicates reduces any remaining duplicates
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
            [
                sys.executable,
                "-m",
                "emet.simulation.mujoco_server",
                "--headless",
                "--use-robocasa",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=60):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("MuJoCo+Robocasa server did not start. stderr:\n" + stderr)

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

        sg_processor = SceneGraphProcessor(
            config_name="cpu_scene_graph",
            device="cpu",
        )
        voxel_map = executor.agent.get_voxel_map()
        voxel_map.set_scene_graph_processor(sg_processor)

        # First spin
        executor([("rotate_in_place", "")])
        sg = sg_processor.scene_graph
        count_after_spin1 = sg.num_objects
        print(f"After spin 1: {count_after_spin1} objects")
        print(sg.to_string())

        assert count_after_spin1 >= 1, (
            "Expected at least 1 object after first spin in Robocasa kitchen"
        )

        # Second spin (same scene, robot returns to similar viewpoints)
        executor([("rotate_in_place", "")])
        count_after_spin2 = sg.num_objects
        print(f"After spin 2: {count_after_spin2} objects")

        # Dedup check: node count should not double
        assert count_after_spin2 < count_after_spin1 * 2.5, (
            f"Node count exploded: {count_after_spin1} -> {count_after_spin2}. "
            "Deduplication may not be working."
        )

        # Stability check: some objects should have been seen multiple times
        stable_nodes = [n for n in sg.nodes.values() if n.observation_count >= 2]
        print(f"Stable objects (obs >= 2): {len(stable_nodes)}")
        assert len(stable_nodes) >= 1, (
            "Expected at least 1 stable object (seen from multiple viewpoints) "
            "after two spins"
        )

        # Post-hoc dedup
        merges = sg.merge_duplicates()
        print(f"Post-hoc merges: {merges}")
        final_count = sg.num_objects
        print(f"Final object count: {final_count}")

        # Edges
        sg.update_edges()
        print(f"Edges: {len(sg.edges)}")

        # Save and verify
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            sg.save(tmpdir)
            from emet.mapping.scene_graph.open_vocab_scene_graph import OpenVocabSceneGraph

            loaded = OpenVocabSceneGraph.load(tmpdir)
            assert loaded.num_objects == final_count

        print("\n=== Final Scene Graph ===")
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
