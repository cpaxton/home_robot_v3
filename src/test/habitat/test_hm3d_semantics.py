# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from emet.core.interfaces import Observations
from emet.core.parameters import get_parameters
from emet.habitat.hm3d_semantics import (
    Hm3dSemanticLabeler,
    _instance_index_from_object_id,
    hm3d_instance_items_from_obs,
    resolve_hm3d_semantics_enabled,
)


class _FakeCategory:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name

    def index(self) -> int:
        return 0


class _FakeObject:
    def __init__(self, object_id: str, category_name: str):
        self.id = object_id
        self.category = _FakeCategory(category_name)


class _FakeScene:
    def __init__(self, objects):
        self.objects = objects


def test_instance_index_from_object_id():
    assert _instance_index_from_object_id("lamp_42") == 42
    assert _instance_index_from_object_id("Unknown_0") == 0
    assert _instance_index_from_object_id("no_suffix") is None


def test_hm3d_semantic_labeler_skips_structural():
    scene = _FakeScene(
        [
            _FakeObject("lamp_5", "lamp"),
            _FakeObject("wall_6", "wall"),
            _FakeObject("bed_7", "bed"),
        ]
    )
    labeler = Hm3dSemanticLabeler.from_semantic_scene(scene)
    assert labeler is not None
    assert labeler.instance_to_label[5] == "lamp"
    assert labeler.instance_to_label[7] == "bed"
    assert 6 not in labeler.instance_to_label


def test_hm3d_instance_items_from_obs():
    scene = _FakeScene(
        [
            _FakeObject("lamp_5", "lamp"),
            _FakeObject("bed_7", "bed"),
        ]
    )
    labeler = Hm3dSemanticLabeler.from_semantic_scene(scene)
    sem = np.zeros((4, 4), dtype=np.uint32)
    sem[:, :2] = 5
    sem[:, 2:] = 7
    depth = np.ones((4, 4), dtype=np.float32) * 2.0
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        semantic=sem,
        camera_K=np.eye(3),
        camera_pose=np.eye(4),
    )
    items = hm3d_instance_items_from_obs(labeler, obs, min_pixels=1)
    labels = {lab for lab, _ in items}
    assert "lamp" in labels or "bed" in labels


@pytest.mark.parametrize(
    ("requested", "has_semantic_glb", "has_config", "expected"),
    [
        (False, False, False, False),
        (False, True, True, False),
        (None, False, True, False),
        (None, True, False, False),
        (None, True, True, True),
        (True, True, True, True),
    ],
)
def test_hm3d_semantics_resolution_matrix(
    tmp_path,
    requested,
    has_semantic_glb,
    has_config,
    expected,
):
    semantic_glb = tmp_path / "scene.semantic.glb"
    annotated_config = tmp_path / "scene_dataset_config.json"
    if has_semantic_glb:
        semantic_glb.touch()
    if has_config:
        annotated_config.touch()
    assert (
        resolve_hm3d_semantics_enabled(
            requested,
            semantic_glb=semantic_glb,
            annotated_config=annotated_config,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("has_semantic_glb", "has_config"),
    [(False, False), (False, True), (True, False)],
)
def test_explicit_hm3d_semantics_request_fails_closed(
    tmp_path,
    has_semantic_glb,
    has_config,
):
    semantic_glb = tmp_path / "scene.semantic.glb"
    annotated_config = tmp_path / "scene_dataset_config.json"
    if has_semantic_glb:
        semantic_glb.touch()
    if has_config:
        annotated_config.touch()
    with pytest.raises(FileNotFoundError, match="explicitly requested"):
        resolve_hm3d_semantics_enabled(
            True,
            semantic_glb=semantic_glb,
            annotated_config=annotated_config,
        )


@pytest.mark.parametrize(
    ("override", "expected"),
    [("int8", "int8"), ("none", None)],
)
def test_hmeqa_quantization_override_reaches_eqa_parameters(
    monkeypatch,
    override,
    expected,
):
    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project_root / "packages" / "emet_habitat"))
    from emet_habitat.runner import _configure_eqa_parameters

    parameters = get_parameters("dynav_config.yaml")
    _configure_eqa_parameters(
        parameters,
        eqa_vl_family="qwen3_vl",
        eqa_hf_model_id="Qwen/Qwen3-VL-8B-Instruct",
        eqa_vl_quantization=override,
        device="cpu",
    )
    assert parameters.get("eqa/vl_quantization") == expected


def test_hmeqa_batch_semantics_preflight_fails_before_episode_loop(
    monkeypatch,
    tmp_path,
):
    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project_root / "packages" / "emet_habitat"))
    from emet_habitat import runner

    root = tmp_path / "scene_datasets" / "hm3d" / "train"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "hm3d_annotated_scene_dataset_config",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(FileNotFoundError, match="explicitly requested"):
        runner._validate_requested_hm3d_semantics(
            requested=True,
            question_ids=[0],
            questions=[SimpleNamespace(index=0, scene="scene-a")],
            hm3d_root=root,
        )


@pytest.mark.parametrize("use_hm3d_semantics", [False, True])
@pytest.mark.parametrize("use_enrich_labels", [False, True])
def test_hmeqa_semantics_and_enrich_axes_are_independent(
    monkeypatch,
    use_hm3d_semantics,
    use_enrich_labels,
):
    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project_root / "packages" / "emet_habitat"))
    from emet_habitat import runner

    controller_kwargs = {}

    def fake_controller(**kwargs):
        controller_kwargs.update(kwargs)
        return SimpleNamespace(graph_memory=None)

    monkeypatch.setattr(runner, "GraphEQAController", fake_controller)
    robot = SimpleNamespace(uses_hm3d_semantics=use_hm3d_semantics)
    runner._make_controller(
        robot,
        get_parameters("dynav_config.yaml"),
        method="static_graph",
        mock_llm=False,
        mock_llm_explore=False,
        gold_letter="A",
        no_rerun=True,
        use_real_vlm=True,
        device="cpu",
        use_hm3d_semantics=use_hm3d_semantics,
    )
    assert controller_kwargs["use_sensor_perception"] is not use_hm3d_semantics

    seeded = []
    graph_memory = SimpleNamespace(seed_object_hints=seeded.append)
    monkeypatch.setattr(
        runner,
        "enrich_labels_for_dataset_question",
        lambda *_args, **_kwargs: "chair, table.",
    )
    runner._seed_hmeqa_enrich_labels(
        SimpleNamespace(graph_memory=graph_memory),
        question_id=0,
        scene="scene-a",
        questions_path=None,
        enabled=use_enrich_labels,
    )
    assert seeded == (["chair, table."] if use_enrich_labels else [])


def test_hmeqa_controller_rejects_requested_but_ineffective_semantics(monkeypatch):
    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project_root / "packages" / "emet_habitat"))
    from emet_habitat import runner

    with pytest.raises(RuntimeError, match="simulator did not enable"):
        runner._make_controller(
            SimpleNamespace(uses_hm3d_semantics=False),
            get_parameters("dynav_config.yaml"),
            method="static_graph",
            mock_llm=False,
            mock_llm_explore=False,
            gold_letter="A",
            no_rerun=True,
            use_real_vlm=True,
            device="cpu",
            use_hm3d_semantics=True,
        )
