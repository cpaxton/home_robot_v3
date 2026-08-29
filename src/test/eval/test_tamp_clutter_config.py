# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from emet.controller.task.tamp.clutter_chain import plan_clear_clutter
from emet.eval.tamp_clutter import (
    ClutterEpisode,
    clutter_blocks_path,
    clutter_success_flags,
    load_clutter_episodes,
    scatter_ring_targets,
)

REPO = Path(__file__).resolve().parents[3]
EPISODES_YAML = REPO / "configs" / "ovmm" / "clutter_episodes.yaml"


def test_load_clutter_episodes():
    eps = load_clutter_episodes(EPISODES_YAML)
    assert len(eps) >= 6
    modes = {e.mode for e in eps}
    assert modes == {"cleanup", "nav_goal"}
    cleanup = next(e for e in eps if e.mode == "cleanup")
    assert cleanup.bin_query and cleanup.n_objects >= 3
    nav = next(e for e in eps if e.mode == "nav_goal")
    assert nav.goal_landmark or nav.goal_landmark == "auto"


def test_invalid_episode_rejected():
    with pytest.raises(ValueError):
        ClutterEpisode(id="x", tier="S1", sim="s", robot="rby1", mode="bogus", n_objects=2)
    with pytest.raises(ValueError):
        ClutterEpisode(id="x", tier="S1", sim="s", robot="rby1", mode="cleanup", n_objects=2, manip_mode="nope")
    with pytest.raises(ValueError):
        ClutterEpisode(id="x", tier="S1", sim="s", robot="rby1", mode="cleanup", n_objects=-1)


def test_zero_objects_allowed_pure_nav():
    ep = ClutterEpisode(id="x", tier="S1", sim="s", robot="nori", mode="nav_goal", n_objects=0)
    assert ep.n_objects == 0
    assert ep.resolved_manip_mode() == "latch"


def test_battery_episodes_matrix():
    from eval_tamp_clutter import _battery_episodes  # scripts/eval_tamp_clutter.py on sys.path

    eps = _battery_episodes(["nori", "rby1"], [0, 1])
    assert len(eps) == 2 * 2 * 4  # robots x scenes x {pickplace,declutter,navblocked,navclear}
    tests = {str(e.id).split("_")[1] for e in eps}
    assert tests == {"pickplace", "declutter", "navblocked", "navclear"}
    for e in eps:
        assert e.robot in ("nori", "rby1")
        assert e.scene_index in (0, 1)
    # navclear episodes are pure-nav (n_objects == 0).
    navclear = [e for e in eps if "navclear" in e.id]
    assert all(e.n_objects == 0 for e in navclear)


def test_robot_default_manip_mode():
    # rby1 and innate_mars default to kinematic latch; stretch to teleport oracle.
    rby1 = ClutterEpisode(id="a", tier="S1", sim="s", robot="rby1", mode="cleanup", n_objects=3)
    assert rby1.resolved_manip_mode() == "latch"
    mars = ClutterEpisode(id="c", tier="S1", sim="s", robot="innate_mars", mode="nav_goal", n_objects=3)
    assert mars.resolved_manip_mode() == "latch"
    stretch = ClutterEpisode(id="b", tier="S1", sim="s", robot="stretch", mode="cleanup", n_objects=3)
    assert stretch.resolved_manip_mode() == "sim"


def _assert_episode_sim_robot_matches(eps) -> None:
    from emet.config.sim_launch_config import load_sim_launch_config_from_path

    for ep in eps:
        sim_cfg = load_sim_launch_config_from_path(ep.sim)
        assert str(sim_cfg.robot).lower() == str(ep.robot).lower(), (
            f"{ep.id}: episode robot={ep.robot!r} vs sim YAML robot={sim_cfg.robot!r} ({ep.sim})"
        )


def test_small_registry_sim_robot_matches_episode():
    _assert_episode_sim_robot_matches(load_clutter_episodes(EPISODES_YAML))


def test_load_large_clutter_registry():
    large = REPO / "configs" / "ovmm" / "clutter_episodes_large.yaml"
    if not large.exists():
        pytest.skip("large registry not generated (run scripts/generate_tamp_clutter_registry.py)")
    eps = load_clutter_episodes(large)
    assert len(eps) == 200
    robots = {e.robot for e in eps}
    assert robots == {"rby1", "stretch", "innate_mars", "nori"}
    # Landmark variety across nav_goal episodes.
    landmarks = {e.goal_landmark for e in eps if e.mode == "nav_goal" and e.goal_landmark != "auto"}
    assert len(landmarks) >= 4
    nav = [e for e in eps if e.mode == "nav_goal"]
    assert nav and all(e.n_objects == 8 and e.tight_ring for e in nav)
    # rby1 latch; stretch / innate_mars / nori teleport (floor clutter).
    by_robot = {r: next(e.resolved_manip_mode() for e in eps if e.robot == r) for r in robots}
    assert by_robot["rby1"] == "latch"
    assert by_robot["stretch"] == "sim"
    assert by_robot["innate_mars"] == "sim"
    assert by_robot["nori"] == "sim"
    _assert_episode_sim_robot_matches(eps)


def test_scatter_ring_geometry():
    rng = np.random.default_rng(0)
    robot = np.array([0.0, 0.0])
    goal = np.array([2.0, 0.0])
    targets = scatter_ring_targets(robot, goal, 4, radius_m=0.8, rng=rng)
    assert len(targets) == 4
    for t in targets:
        r = float(np.linalg.norm(t - robot))
        assert 0.2 <= r <= 1.2
    # First slot biased toward the goal (+x).
    assert float(targets[0][0]) > 0.0


def test_clutter_blocks_path_ring_blocks_goal():
    robot = np.array([0.0, 0.0])
    goal = np.array([3.0, 0.0])
    # Full ring of objects around the robot -> robot cannot escape without clearing.
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    obj = [np.array([0.55 * np.cos(a), 0.55 * np.sin(a)]) for a in angles]
    blocked, info = clutter_blocks_path(robot, goal, obj, clearance_m=0.22)
    assert blocked is True
    assert info["probe"] == "gt_nav"


def test_clutter_blocks_path_open_route():
    robot = np.array([0.0, 0.0])
    goal = np.array([3.0, 0.0])
    # Objects far off the route -> not blocked.
    obj = [np.array([1.5, 1.5]), np.array([1.5, -1.5])]
    blocked, info = clutter_blocks_path(robot, goal, obj, clearance_m=0.22)
    assert blocked is False


def test_clutter_blocks_path_cleanup_near():
    robot = np.array([0.0, 0.0])
    obj = [np.array([0.5, 0.0]), np.array([0.0, 0.5])]
    blocked, info = clutter_blocks_path(robot, None, obj)
    assert blocked is True
    assert info["probe"] == "cleanup_near"


def test_clutter_success_flags():
    assert clutter_success_flags({"mode": "cleanup", "n_objects": 3, "n_relocated": 3})["task_success"] is True
    assert clutter_success_flags({"mode": "cleanup", "n_objects": 3, "n_relocated": 1})["task_success"] is False
    # Cleanup must not succeed via goal_reached alone.
    assert clutter_success_flags(
        {"mode": "cleanup", "n_objects": 3, "n_relocated": 0, "goal_reached": True}
    )["task_success"] is False
    assert clutter_success_flags({"mode": "nav_goal", "goal_reached": True})["task_success"] is True
    skipped = clutter_success_flags(
        {"mode": "nav_goal", "goal_reached": True, "skipped_invalid": True}
    )
    assert skipped["task_success"] is False
    assert skipped["skipped_invalid"] is True


def test_plan_clear_clutter_missing_bin(monkeypatch):
    """No GT placements -> clean early failure metrics (no sim needed)."""
    import emet.memory.graph_eqa.sim_ground_truth_graph as gt_mod
    from emet.memory import graph_eqa as _ge  # noqa: F401  (module import for patch target)

    class _FakeRobot:
        def get_emet_session(self):
            return {}

    monkeypatch.setattr(gt_mod, "read_sim_object_placements", lambda _session: {})
    out = plan_clear_clutter(
        _FakeRobot(),
        objects=[{"object_query": "apple", "object_gt_body": "apple_1_0_0"}],
        mode="cleanup",
        bin_query="GarbageCan",
    )
    assert out["error"] == "missing_bin_body:GarbageCan"
    assert out["task_success"] is False
