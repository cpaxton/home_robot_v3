# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agentic-find routing for the Habitat OVMM find runner (no sim).

Verifies ``run_habitat_find_phase_episode`` routes FindObj/FindRec through the
shared AgenticEQA loop (``run_ovmm_agentic_localize``) for dynagraph/static_graph
and falls back to one-shot localize for dynamem / ground_truth.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def _episode() -> object:
    from emet_habitat.ovmm_find_runner import HabitatFindPhaseEpisode

    return HabitatFindPhaseEpisode(
        id="hm3d_lamp_bed_00006",
        scene="00006-HkseAnWCgqk",
        floor=0,
        object="lamp",
        start_recep="bed",
        goal_recep="table",
        success_radius_m=0.75,
        explore_steps=0,
        object_gt_body=None,
    )


def _placement(pos):
    return {"cat": "lamp", "pos": list(pos), "bounds": [[p - 0.1 for p in pos], [p + 0.1 for p in pos]]}


class _FakeSim:
    uses_hm3d_semantics = True
    pathfinder = None
    floor_y = 0.0
    camera_tilt_deg = -30.0
    sensor_height = 1.31

    def __init__(self, scene: str, **kwargs):
        self.last_init_pose_record = {"scene": scene}
        self._sim = SimpleNamespace(semantic_scene=object())

    @classmethod
    def from_scene_id(cls, scene: str, **kwargs) -> _FakeSim:
        return cls(scene)

    def set_init_pose(self, pose) -> None:
        return None

    def close(self) -> None:
        return None


def _run_episode(monkeypatch, backend: str, *, agentic_find: bool | None = None) -> tuple[dict, list[str], list[str]]:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project_root / "packages" / "emet_habitat"))

    placements = {
        "hm3d_lamp_1": _placement([1.0, 1.2, 2.0]),
        "hm3d_table_2": _placement([4.0, 0.0, 4.0]),
        "hm3d_bed_3": _placement([1.0, 0.0, 1.0]),
    }
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig

    run_cfg = FindPhaseRunConfig(
        backend=backend,  # type: ignore[arg-type]
        cpu_only=True,
        not_rotate=True,
        agentic_find=agentic_find,
        agentic_max_rounds=3,
        agentic_max_nav_steps=2,
    )
    if backend == "dynagraph" and agentic_find is None:
        run_cfg.cpu_only = False
    fake_agent = SimpleNamespace(
        stop=lambda: None,
        update=lambda: None,
        voxel_map=SimpleNamespace(encoder=None),
        graph_memory=None,
    )
    agentic_calls: list[str] = []
    oneshot_calls: list[str] = []
    create_kwargs: dict = {}

    def _fake_agentic(agent, question, **kwargs):
        agentic_calls.append(question)
        return SimpleNamespace(
            verified=True,
            verified_obs_id=1,
            xyz=np.asarray([1.0, 1.2, 2.0]),
            n_rounds=2,
            n_nav=1,
            n_explore=0,
            n_retracted_claims=0,
            error=None,
            extra={"xyz_source": "voxel", "voxel_query_used": "lamp"},
        )

    def _fake_oneshot(memory, query, **kwargs):
        oneshot_calls.append(query)
        return np.asarray([1.0, 1.2, 2.0]), True, query, "voxel"

    def _fake_create_agent(*args, **kwargs):
        create_kwargs.update(kwargs)
        return fake_agent

    with (
        patch(
            "emet.eval.ovmm_agentic_find.run_ovmm_agentic_localize",
            side_effect=_fake_agentic,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.set_find_phase_run_seed",
            return_value=None,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.load_scene_init_poses",
            return_value={("00006-HkseAnWCgqk", 0): SimpleNamespace(x=0, y=0, z=0, heading=0.0)},
        ),
        patch(
            "emet_habitat.ovmm_find_runner.default_hm3d_scene_dir",
            return_value="unused",
        ),
        patch(
            "emet_habitat.ovmm_find_runner.HabitatEQASimulator",
            _FakeSim,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.hm3d_placements_from_semantic_scene",
            return_value=placements,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.HabitatRobotClient",
            return_value=SimpleNamespace(
                set_emet_session=lambda s: None,
                get_emet_session=lambda: None,
                stop=lambda: None,
                get_base_pose=lambda: np.zeros(3),
                get_observation=lambda: None,
            ),
        ),
        patch(
            "emet_habitat.ovmm_find_runner.get_parameters",
            return_value={},
        ),
        patch(
            "emet_habitat.ovmm_find_runner.apply_habitat_ovmm_find_parameters",
            return_value={},
        ),
        patch(
            "emet_habitat.runner._configure_habitat_mapping",
            return_value=None,
        ),
        patch(
            "emet_habitat.runner._configure_habitat_nav",
            return_value=None,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.create_find_phase_agent",
            side_effect=_fake_create_agent,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.run_mapping_protocol",
            return_value=1,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.get_memory_backend_for_agent",
            return_value=None,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.resolve_object_query",
            return_value="lamp",
        ),
        patch(
            "emet_habitat.ovmm_find_runner.query_find_phase_localization",
            side_effect=_fake_oneshot,
        ),
        patch(
            "emet_habitat.ovmm_find_runner.compute_find_phase_metrics",
            return_value={
                "find_object_success": True,
                "find_recep_success": True,
                "find_partial_success": 1.0,
                "localization_err_obj_m": 0.0,
                "localization_err_recep_m": 0.0,
            },
        ),
        patch(
            "emet_habitat.ovmm_find_runner.collect_scaling_diagnostics",
            return_value={},
        ),
        patch(
            "emet_habitat.ovmm_find_runner._release_gpu_memory",
            return_value=None,
        ),
    ):
        from emet_habitat.ovmm_find_runner import run_habitat_find_phase_episode

        result = run_habitat_find_phase_episode(_episode(), run_cfg)
    return result, agentic_calls, oneshot_calls, create_kwargs


def test_habitat_agentic_find_routes_dynagraph_through_agentic_loop(monkeypatch) -> None:
    result, agentic_calls, oneshot_calls, create_kwargs = _run_episode(monkeypatch, "dynagraph")
    assert result["agentic_find"] is True
    assert len(agentic_calls) == 2
    assert oneshot_calls == []
    assert result["obj_localize_success"] is True
    assert result["recep_localize_success"] is True
    assert result["obj_localize_source"] == "voxel"
    assert result["obj_agentic_rounds"] == 2
    assert create_kwargs["cpu_only"] is False


def test_habitat_agentic_find_default_off_for_dynamem(monkeypatch) -> None:
    result, agentic_calls, oneshot_calls, _create = _run_episode(monkeypatch, "dynamem")
    assert result["agentic_find"] is False
    assert agentic_calls == []
    assert len(oneshot_calls) == 2


def test_habitat_agentic_find_no_agentic_override_keeps_oneshot(monkeypatch) -> None:
    result, agentic_calls, oneshot_calls, _create = _run_episode(monkeypatch, "dynagraph", agentic_find=False)
    assert result["agentic_find"] is False
    assert agentic_calls == []
    assert len(oneshot_calls) == 2
