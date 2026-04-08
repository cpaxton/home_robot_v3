# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Tests for headless_export: machine-readable save + scene_graph_report.txt

import tempfile
from pathlib import Path

import numpy as np

from emet.memory.format import SCENE_GRAPH_REPORT_TXT, is_memory_directory
from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.headless_export import export_graph_eqa_dir, export_open_vocab_scene_graph_dir


def test_export_graph_eqa_dir_writes_report_and_red_blue_labels():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "reasoning: r\nanswer: ok\nconfidence: true\naction:\nconfidence_reasoning: x",
        image_description_client=lambda x: "a, b",
    )
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.08, -0.55, 0.6]), ["red cylinder"])
    mem.add_observation(rgb, np.array([-0.02, -0.55, 0.6]), ["blue cube"])

    with tempfile.TemporaryDirectory() as tmp:
        text = export_graph_eqa_dir(mem, None, tmp)
        assert "red" in text.lower()
        assert "blue" in text.lower()
        report = Path(tmp) / SCENE_GRAPH_REPORT_TXT
        assert report.is_file()
        body = report.read_text(encoding="utf-8")
        assert "red" in body.lower()
        assert "blue" in body.lower()
        assert is_memory_directory(tmp)


def test_export_open_vocab_scene_graph_dir_writes_report():
    from emet.mapping.scene_graph.open_vocab_scene_graph import OpenVocabSceneGraph

    sg = OpenVocabSceneGraph()
    # Minimal graph: empty still writes save + report
    with tempfile.TemporaryDirectory() as tmp:
        text = export_open_vocab_scene_graph_dir(sg, tmp)
        assert isinstance(text, str)
        report = Path(tmp) / SCENE_GRAPH_REPORT_TXT
        assert report.is_file()
        assert report.read_text(encoding="utf-8") == text
        assert (Path(tmp) / "scene_graph.json").is_file()
