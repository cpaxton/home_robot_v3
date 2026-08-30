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
    first_footprint_hit,
    load_clutter_episodes,
    nav_interpolated_route,
    nav_route_open,
    placement_obstacle_disks,
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
    assert ep.resolved_manip_mode() == "sim"


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
    # rby1 defaults to kinematic latch; innate_mars / nori / stretch to teleport oracle
    # (Nori's model arm bottoms out at z≈0.29 m — it cannot latch-grasp true-floor
    # objects; use --manip-mode latch for a latch experiment).
    rby1 = ClutterEpisode(id="a", tier="S1", sim="s", robot="rby1", mode="cleanup", n_objects=3)
    assert rby1.resolved_manip_mode() == "latch"
    mars = ClutterEpisode(id="c", tier="S1", sim="s", robot="innate_mars", mode="nav_goal", n_objects=3)
    assert mars.resolved_manip_mode() == "sim"
    nori = ClutterEpisode(id="d", tier="S1", sim="s", robot="nori", mode="cleanup", n_objects=3)
    assert nori.resolved_manip_mode() == "sim"
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
    open_ok, _ = nav_route_open(robot, goal, obj, clearance_m=0.22)
    assert open_ok is False


def test_clutter_blocks_path_open_route():
    robot = np.array([0.0, 0.0])
    goal = np.array([3.0, 0.0])
    # Objects far off the route -> not blocked.
    obj = [np.array([1.5, 1.5]), np.array([1.5, -1.5])]
    blocked, info = clutter_blocks_path(robot, goal, obj, clearance_m=0.22)
    assert blocked is False


def test_interpolated_chord_hits_object_on_line():
    robot = np.array([0.0, 0.0])
    goal = np.array([3.0, 0.0])
    disks = [(np.array([1.0, 0.0]), 0.08, "apple")]
    ok, info = nav_interpolated_route(robot, goal, disks, clearance_m=0.22)
    assert ok is False
    assert info["hit"]["hit"] == "apple"
    assert info["n_steps"] >= 150  # 3 m / 0.02 m default
    # Off the chord: same distance, no hit.
    ok_off, _ = nav_interpolated_route(
        robot, goal, [(np.array([1.0, 1.5]), 0.08, "apple")], clearance_m=0.22
    )
    assert ok_off is True
    hit = first_footprint_hit(
        [robot, np.array([1.0, 0.0]), goal],
        disks,
        footprint_r_m=0.22,
        skip_first=True,
    )
    assert hit is not None and hit["hit"] == "apple"


def test_unmoved_tight_ring_blocks_teleport_chord():
    """Battery navblocked geometry: 8 @ 0.5 m, objects never moved → chord is blocked."""
    rng = np.random.default_rng(0)
    robot = np.array([0.0, 0.0])
    goal = np.array([3.0, 0.0])
    targets = scatter_ring_targets(
        robot,
        goal,
        8,
        radius_m=0.5,
        rng=rng,
        radius_jitter=0.02,
        angle_jitter_rad=0.02,
    )
    disks = [(xy, 0.08, f"obj_{i}") for i, xy in enumerate(targets)]
    ok, info = nav_interpolated_route(robot, goal, disks, clearance_m=0.22)
    assert ok is False
    assert info.get("hit"), info
    flags = clutter_success_flags(
        {
            "mode": "nav_goal",
            "n_objects": 8,
            "episode_valid": True,
            "n_relocated": 0,
            "goal_reached": True,  # would-be teleport-to-sofa
            "nav_path_open": False,
        }
    )
    assert flags["task_success"] is False


def test_plan_clear_clutter_fails_when_objects_do_not_move(monkeypatch):
    """Pick/place may report success; if GT XY never changes, nav_goal must fail.

    This is the teleport-to-sofa bug: leftover scatter still on the chord.
    """
    import emet.controller.task.tamp.task_search as task_search
    import emet.memory.graph_eqa.sim_ground_truth_graph as gt_mod
    from emet.controller.task.tamp.task_search import TaskPlan

    rng = np.random.default_rng(0)
    robot_xy = np.array([0.0, 0.0])
    goal = np.array([3.0, 0.0])
    targets = scatter_ring_targets(
        robot_xy,
        goal,
        8,
        radius_m=0.5,
        rng=rng,
        radius_jitter=0.02,
        angle_jitter_rad=0.02,
    )
    bodies = [f"obj_{i}" for i in range(8)]
    pl: dict = {
        "ashcan": {"pos": [4.0, 4.0, 0.1], "cat": "ashcan"},
    }
    for b, xy in zip(bodies, targets, strict=True):
        pl[b] = {"pos": [float(xy[0]), float(xy[1]), 0.02], "cat": "apple"}

    class _FakeRobot:
        def __init__(self) -> None:
            self.moved: list = []

        def get_emet_session(self):
            return {}

        def get_base_pose(self, timeout: float = 2.0):
            return np.array([0.0, 0.0, 0.0])

        def move_base_to(self, xyt, **kwargs):
            self.moved.append(np.asarray(xyt, dtype=np.float64))
            return True

    def _ok_plan(*_a, **_k):
        return TaskPlan(
            steps=[],
            object_body="obj_0",
            receptacle_body="ashcan",
            success=True,
            message="claimed_ok",
        )

    robot = _FakeRobot()
    robot._state = {"base_xyz": np.array([0.0, 0.0, 0.0])}
    monkeypatch.setattr(gt_mod, "read_sim_object_placements", lambda _session: dict(pl))
    monkeypatch.setattr(task_search, "plan_pick_place_mcts", _ok_plan)
    monkeypatch.setattr(task_search, "execute_task_plan", _ok_plan)

    out = plan_clear_clutter(
        robot,
        objects=[{"object_query": "apple", "object_gt_body": b} for b in bodies],
        mode="nav_goal",
        bin_query="ashcan",
        goal_xy=goal,
        manip_mode="sim",
        robot_move_goal=goal,  # would count as arrived if the snap were issued
    )
    assert out["n_relocated"] == 0
    assert robot.moved == []
    assert out["nav_path_open"] is False
    assert out["goal_reached"] is False
    assert out["task_success"] is False
    assert clutter_success_flags({**out, "episode_valid": True})["task_success"] is False
    hit = (out.get("nav_probe_after") or {}).get("hit") or {}
    assert hit.get("hit") in set(bodies)


def test_placement_obstacle_disks_skips_high_bodies():
    pl = {
        "apple": {"pos": [1.0, 0.0, 0.02], "cat": "apple"},
        "lamp": {"pos": [1.0, 0.0, 2.0], "cat": "lamp"},
        "table": {"pos": [2.0, 0.0, 0.4], "bounds": [[1.5, -0.4, 0.0], [2.5, 0.4, 0.8]]},
    }
    disks = {name: r for _xy, r, name in placement_obstacle_disks(pl, skip_bodies=("apple",))}
    assert "apple" not in disks
    assert "lamp" not in disks
    assert disks["table"] > 0.2


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
    # Pure-nav (n=0) still succeeds on goal_reached.
    assert clutter_success_flags({"mode": "nav_goal", "goal_reached": True})["task_success"] is True
    skipped = clutter_success_flags(
        {"mode": "nav_goal", "goal_reached": True, "skipped_invalid": True}
    )
    assert skipped["task_success"] is False
    assert skipped["skipped_invalid"] is True
    # Blocked nav_goal: teleport-to-sofa without an open route is not success.
    snap = clutter_success_flags(
        {
            "mode": "nav_goal",
            "n_objects": 8,
            "episode_valid": True,
            "n_relocated": 0,
            "goal_reached": True,
            "nav_path_open": False,
        }
    )
    assert snap["task_success"] is False
    assert snap["nav_path_open"] is False
    cleared = clutter_success_flags(
        {
            "mode": "nav_goal",
            "n_objects": 8,
            "episode_valid": True,
            "n_relocated": 8,
            "goal_reached": True,
            "nav_path_open": True,
        }
    )
    assert cleared["task_success"] is True
    # Missing nav_path_open + blocked + nothing relocated → do not count as success.
    legacy = clutter_success_flags(
        {"mode": "nav_goal", "n_objects": 8, "episode_valid": True, "n_relocated": 0, "goal_reached": True}
    )
    assert legacy["task_success"] is False


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


def test_plan_clear_clutter_refuses_teleport_through_closed_ring(monkeypatch):
    """Failed relocates must not snap the base through leftover clutter."""
    import emet.controller.task.tamp.task_search as task_search
    import emet.memory.graph_eqa.sim_ground_truth_graph as gt_mod
    from emet.controller.task.tamp.task_search import TaskPlan

    robot_xy = np.array([0.0, 0.0])
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    bodies = [f"obj_{i}" for i in range(8)]
    pl: dict = {
        "ashcan": {"pos": [4.0, 4.0, 0.1], "cat": "ashcan"},
    }
    for b, a in zip(bodies, angles, strict=True):
        pl[b] = {
            "pos": [0.55 * float(np.cos(a)), 0.55 * float(np.sin(a)), 0.02],
            "cat": "apple",
        }

    class _FakeRobot:
        def __init__(self) -> None:
            self.moved: list = []

        def get_emet_session(self):
            return {}

        def get_base_pose(self, timeout: float = 2.0):
            return np.array([float(robot_xy[0]), float(robot_xy[1]), 0.0])

        def move_base_to(self, xyt, **kwargs):
            self.moved.append(np.asarray(xyt, dtype=np.float64))
            return True

    robot = _FakeRobot()
    robot._state = {"base_xyz": np.array([0.0, 0.0, 0.0])}
    monkeypatch.setattr(gt_mod, "read_sim_object_placements", lambda _session: dict(pl))
    fail = TaskPlan(steps=[], object_body="", receptacle_body=None, success=False, message="nope")
    monkeypatch.setattr(task_search, "plan_pick_place_mcts", lambda *a, **k: fail)

    out = plan_clear_clutter(
        robot,
        objects=[{"object_query": "apple", "object_gt_body": b} for b in bodies],
        mode="nav_goal",
        bin_query="ashcan",
        goal_xy=np.array([3.0, 0.0]),
        manip_mode="sim",
    )
    assert robot.moved == []
    assert out["nav_path_open"] is False
    assert out["goal_reached"] is False
    assert out["task_success"] is False
    assert out["n_relocated"] == 0


def test_plan_clear_clutter_snaps_when_route_is_open(monkeypatch):
    """Once clutter is off the route, a single teleport snap is allowed."""
    import emet.controller.task.tamp.task_search as task_search
    import emet.memory.graph_eqa.sim_ground_truth_graph as gt_mod
    from emet.controller.task.tamp.task_search import TaskPlan

    bodies = [f"obj_{i}" for i in range(3)]
    pl: dict = {
        "ashcan": {"pos": [4.0, 4.0, 0.1], "cat": "ashcan"},
    }
    for b in bodies:
        pl[b] = {"pos": [4.0, 4.0, 0.05], "cat": "apple"}

    class _FakeRobot:
        def __init__(self) -> None:
            self.moved: list = []

        def get_emet_session(self):
            return {}

        def get_base_pose(self, timeout: float = 2.0):
            return np.array([0.0, 0.0, 0.0])

        def move_base_to(self, xyt, **kwargs):
            self.moved.append(np.asarray(xyt, dtype=np.float64))
            self._state = {"base_xyz": np.array([float(xyt[0]), float(xyt[1]), 0.0])}
            return True

    robot = _FakeRobot()
    robot._state = {"base_xyz": np.array([0.0, 0.0, 0.0])}
    monkeypatch.setattr(gt_mod, "read_sim_object_placements", lambda _session: dict(pl))
    fail = TaskPlan(steps=[], object_body="", receptacle_body=None, success=False, message="nope")
    monkeypatch.setattr(task_search, "plan_pick_place_mcts", lambda *a, **k: fail)

    out = plan_clear_clutter(
        robot,
        objects=[{"object_query": "apple", "object_gt_body": b} for b in bodies],
        mode="nav_goal",
        bin_query="ashcan",
        goal_xy=np.array([3.0, 0.0]),
        manip_mode="sim",
        robot_move_goal=np.array([3.0, 0.0]),
    )
    assert len(robot.moved) == 1
    assert out["nav_path_open"] is True
    assert out["goal_reached"] is True
    assert out["task_success"] is True
