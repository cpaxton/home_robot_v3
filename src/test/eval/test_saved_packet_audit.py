# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import importlib.util
import json
from pathlib import Path

from PIL import Image


def load_script(name):
    path = Path(__file__).resolve().parents[3] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stopping_arms_only_differ_in_confirmation():
    script = load_script("run_view_selection_pairs")
    single = script.stopping_overrides("stop_single")
    corroborated = script.stopping_overrides("stop_corroborated")
    assert {key for key in single if single[key] != corroborated[key]} == {"EMET_EQA_AGENTIC_SINGLE_VIEW_CONFIRM"}
    assert corroborated["EMET_EQA_ANSWERABLE_CONFIRM"] == "1"
    assert script.stopping_overrides("pose") == {}


def test_provenance_arms_only_differ_in_labels():
    script = load_script("run_view_selection_pairs")
    role = script.provenance_overrides("labels_role")
    spatial = script.provenance_overrides("labels_spatial")
    assert {key for key in role if role[key] != spatial[key]} == {"EMET_EQA_VIEW_SPATIAL_LABELS"}
    assert role["EMET_EQA_AGENTIC_SINGLE_VIEW_CONFIRM"] == "1"
    assert spatial["EMET_EQA_RETAIN_APPROACH_VIEWS"] == "1"


def test_gallery_preserves_pixels_escapes_question_and_marks_roles(tmp_path):
    script = load_script("assess_saved_view_packets")
    episode = tmp_path / "input/s0-q12-pose_approach-attempt0/q0012_lazy_graph"
    packet = episode / "selected_views/0001"
    packet.mkdir(parents=True)
    meta = episode / "eqa_decisions/iter_001"
    meta.mkdir(parents=True)
    (meta / "meta.json").write_text(json.dumps({"question": "Is <red> visible?"}))
    Image.new("RGB", (10, 10), "red").save(packet / "image_1.png")
    (packet / "manifest.json").write_text(
        json.dumps(
            {"views": [{"image": "image_1.png", "role": "current", "view_id": "1", "label": "Image 1: current"}]}
        )
    )
    output = tmp_path / "output"
    cases = script.prepare(tmp_path / "input", output)
    assert len(cases) == 1
    assert (output / cases[0]["views"][0]["path"]).read_bytes() == (packet / "image_1.png").read_bytes()
    page = (output / "index.html").read_text()
    assert "Is &lt;red&gt; visible?" in page
    assert "current" in page
    assert "not necessarily final submissions" in page
