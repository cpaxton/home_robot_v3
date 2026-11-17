# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Coordinator-visible current-view grounding uses the existing admission path."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tools import get_tools


@pytest.mark.parametrize("admitted", [True, False])
def test_ground_object_delivers_geometry_or_abstention_to_coordinator(admitted):
    geometry = {"status": "admitted", "xyz": [1, 2, 3], "coordinate_frame": "world", "obs_id": 9}
    if not admitted:
        geometry = {"status": "not_admitted", "depth_support": {"max_depth_m": 2.5}}
    ground = Mock(
        return_value={
            "ok": admitted,
            "status": "grounded" if admitted else "not_grounded",
            "reason": None if admitted else "target outside supported depth range",
            "geometry": geometry,
            "observation": {"source_obs_id": 5, "view_id": "voxel-frame:5"},
        }
    )
    executor = Mock(side_effect=AssertionError("grounding must not move or manipulate"))
    executor.agent = SimpleNamespace(ground_current_view=ground)
    tools = {t.name: t for t in get_tools({"executor": executor})}
    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "ground_object", "arguments": {"query": "microwave", "description": "above the stove"}}],
        tools,
        executor,
    )
    assert ok and has_info and len(results) == 1
    ground.assert_called_once_with("microwave", target_description="above the stove")
    executor.assert_not_called()
    assert '"source_obs_id": 5' in results[0]
    if admitted:
        assert results[0].startswith("[ground_object] ok status=grounded")
        assert '"xyz": [1, 2, 3]' in results[0]
    else:
        assert results[0].startswith("[ground_object] failed status=not_grounded")
        assert "supported depth range" in results[0]
        assert '"xyz"' not in results[0]


def test_ground_object_reports_unsupported_controller_without_retrieval_fallback():
    executor = Mock(side_effect=AssertionError("must not fall back to FIND"))
    executor.agent = SimpleNamespace()
    tools = {t.name: t for t in get_tools({"executor": executor})}
    out = tools["ground_object"].func(query="microwave")
    assert out["ok"] is False and out["status"] == "unavailable"
    executor.assert_not_called()


def test_query_find_profile_explicitly_allows_grounded_instance_admission():
    import yaml

    from emet.config.embodied_agent_config import GraphObjectFusionConfigRef
    from emet.memory.graph_eqa.graph_object_fusion.attach import fusion_config_from_sources

    root = Path(__file__).resolve().parents[3]
    profile = yaml.safe_load((root / "configs/emet/shared_find_acceptance.yaml").read_text())
    assert not GraphObjectFusionConfigRef().enabled
    config = fusion_config_from_sources(parameters=profile["mapping"], fref=GraphObjectFusionConfigRef())
    assert config.enabled and config.use_instance_nodes
