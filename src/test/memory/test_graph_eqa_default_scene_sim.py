# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# GraphEQA + default MuJoCo scene: after a scan, a color question should be answerable
# with both red and blue (red cylinder + blue cube). Uses mocked EQA clients so CI does
# not download Qwen; the sim still validates RGB/pose + graph wiring.
#
# Run: uv run emet test src/test/memory/test_graph_eqa_default_scene_sim.py
# Skip sim: RUN_SIM_TESTS=0 or emet test --no-sim

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from emet.memory.format import SCENE_GRAPH_REPORT_TXT
from emet.memory.headless_export import export_graph_eqa_dir

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


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(120)
def test_graph_eqa_color_question_default_mujoco_scene():
    """
    MuJoCo default scene (red cylinder, blue cube on table): build GraphEQAMemory from
    real camera RGB and ground-truth labels, then query_answer must contain red and blue.
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
        from emet.controller.zmq_client import StretchZmqClient
        from emet.core.parameters import get_parameters
        from emet.memory.graph_eqa import GraphEQAMemory

        robot = StretchZmqClient(
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
        executor([("rotate_in_place", "")])

        obs = robot.get_observation()
        rgb = np.asarray(obs.rgb, dtype=np.uint8)

        def mock_eqa(_commands):
            return (
                "reasoning: the scene graph lists a red cylinder and a blue cube.\n"
                "answer: red and blue.\n"
                "confidence: true\n"
                "action: \n"
                "confidence_reasoning: labels match the default scene.\n"
            )

        mem = GraphEQAMemory(
            eqa_client=mock_eqa,
            image_description_client=lambda cmd: "red cylinder, blue cube",
        )
        mem.add_observation(rgb, np.array([0.08, -0.55, 0.6], dtype=float), ["red cylinder"])
        mem.add_observation(rgb, np.array([-0.02, -0.55, 0.6], dtype=float), ["blue cube"])

        g = mem.to_string().lower()
        assert "red" in g
        assert "blue" in g

        _r, answer, conf, _cr, _tp, _imgs = mem.query_answer(
            "Which color objects can you see?", None, None
        )
        assert conf is True
        al = answer.lower()
        assert "red" in al, f"expected red in answer, got {answer!r}"
        assert "blue" in al, f"expected blue in answer, got {answer!r}"

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            report_text = export_graph_eqa_dir(mem, None, tmp)
            assert "red" in report_text.lower()
            assert "blue" in report_text.lower()
            rp = Path(tmp) / SCENE_GRAPH_REPORT_TXT
            assert rp.is_file()
            assert "red" in rp.read_text(encoding="utf-8").lower()

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
