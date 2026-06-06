"""GT scene JSON from live sim_object_placements."""

from __future__ import annotations

import numpy as np

from emet.simulation.mujoco_gt_objects import build_gt_scene_payload_from_session_placements


def test_build_gt_scene_payload_from_session_placements():
    placements = {
        "obj_main": {
            "cat": "mug",
            "pos": np.array([1.0, 2.0, 0.9]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "bounds": np.array([[0.9, 1.9, 0.8], [1.1, 2.1, 1.0]]),
        },
        "_emet_spawn_hint_xyt": {"pos": [0.0, 0.0, 0.0]},
    }
    payload = build_gt_scene_payload_from_session_placements(placements, robot="innate_mars", seed=0)
    assert payload["source"] == "sim_session"
    assert len(payload["objects"]) == 1
    assert payload["objects"][0]["label"] == "mug"
