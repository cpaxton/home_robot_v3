# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Scene-task extractor tests against a real MolmoSpaces iTHOR scene (CPU-only)."""

from __future__ import annotations

import pytest

from emet.eval.scene_task_extractor import (
    default_molmospaces_scenes_dir,
    emit_tasks,
    load_scene_metadata,
    pickable_objects,
    receptacle_objects,
    resolve_scene_metadata_for_session,
    scene_objects,
)


@pytest.fixture(scope="module")
def floorplan1_metadata() -> dict:
    scene = default_molmospaces_scenes_dir() / "ithor" / "FloorPlan1_physics_metadata.json"
    if not scene.is_file():
        pytest.skip(f"MolmoSpaces iTHOR FloorPlan1 metadata not installed ({scene})")
    return load_scene_metadata(scene)


def test_load_scene_metadata(floorplan1_metadata):
    objects = floorplan1_metadata["objects"]
    assert isinstance(objects, dict) and len(objects) > 0


def test_scene_objects_enumerated(floorplan1_metadata):
    objs = scene_objects(floorplan1_metadata)
    assert len(objs) >= 10
    static = [o for o in objs if o.is_static]
    movable = [o for o in objs if not o.is_static]
    assert static and movable


def test_pickable_objects_have_grasps(floorplan1_metadata):
    picks = pickable_objects(scene_objects(floorplan1_metadata))
    assert len(picks) >= 1
    for o in picks:
        assert o.has_grasps, f"{o.body} should have a grasp asset"
        assert o.asset_id, f"{o.body} needs an asset_id"
        assert o.body.endswith("_1_1_0"), f"{o.body} should be the mesh child (sim_object_placements key)"


def test_receptacle_objects_have_sites(floorplan1_metadata):
    recepts = receptacle_objects(scene_objects(floorplan1_metadata, check_grasps=False))
    assert len(recepts) >= 1
    for o in recepts:
        assert o.receptacle_sites, f"{o.body} should expose receptacle sites"


def test_emit_tasks_schema(floorplan1_metadata):
    objs = scene_objects(floorplan1_metadata)
    tasks = emit_tasks(objs, sim="configs/sim/molmospaces_ithor_train_0.yaml")
    assert len(tasks) >= 1
    for t in tasks:
        assert t.object
        assert t.start_recep
        assert t.goal_recep
        assert t.object_gt_body
        d = t.to_dict()
        assert d["object_gt_body"] == t.object_gt_body
        assert set(d) >= {"id", "tier", "sim", "object", "start_recep", "goal_recep"}


def test_emit_tasks_gt_bodies_present_in_placements_like(floorplan1_metadata):
    """GT body must be the freejoint parent keyed by sim_object_placements."""
    objs = scene_objects(floorplan1_metadata)
    tasks = emit_tasks(objs, sim="configs/sim/molmospaces_ithor_train_0.yaml")
    bodies = {o.body for o in objs}
    for t in tasks:
        assert t.object_gt_body in bodies, f"GT body {t.object_gt_body} not in scene"


def test_reachability_priors_wire_profile(floorplan1_metadata):
    from emet.eval.scene_task_extractor import compute_reachability_priors

    objs = scene_objects(floorplan1_metadata)
    if not pickable_objects(objs):
        pytest.skip("no pickable objects with grasp assets")

    priors = compute_reachability_priors(objs, robot_id="sourccey", arm="left")
    assert priors, "must compute reachability for at least one object"
    for r in priors.values():
        assert isinstance(r.reachable, bool)
        assert r.ee_error_m >= 0.0
        assert r.contact_dist_m >= 0.0
        assert r.body in {o.body for o in pickable_objects(objs)}


def test_resolve_scene_metadata_for_live_session(tmp_path):
    ithor = tmp_path / "ithor"
    ithor.mkdir()
    (ithor / "FloorPlan1_physics_metadata.json").write_text("{}", encoding="utf-8")
    (ithor / "FloorPlan2_physics_metadata.json").write_text("{}", encoding="utf-8")
    session = {"environment": {"kind": "molmospaces", "scene": "ithor", "index": 1}}
    resolved = resolve_scene_metadata_for_session(session, scenes_dir=tmp_path)
    assert resolved == ithor / "FloorPlan2_physics_metadata.json"

    source_session = {
        "environment": {"kind": "molmospaces", "scene": "ithor", "index": 1},
        "scene_source_basename": "FloorPlan1.xml",
    }
    assert resolve_scene_metadata_for_session(source_session, scenes_dir=tmp_path) == (
        ithor / "FloorPlan1_physics_metadata.json"
    )
