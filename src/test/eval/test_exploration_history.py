# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace

import numpy as np
import pytest

from emet.eval.exploration_history import CollectionBudgetExceeded, RecordedSensor, collect_history, route_prefix


def test_route_prefix_counts_path_length_not_endpoint_distance_and_keeps_turns():
    route = [[0, 0, 1], [0.3, 0, 1], [0.3, 0.3, 1], [0, 0.3, 1]]
    prefix, distance = route_prefix(route, [0, 0, 0], 0.65)
    assert prefix[:3] == route[:3]
    np.testing.assert_allclose(prefix[-1], [0.25, 0.3, 1])
    assert distance == pytest.approx(0.65)
    assert route_prefix([], None, 1) == ([], 0.0)
    with pytest.raises(ValueError):
        route_prefix(route, [0, 0, 0], np.nan)


class Sensor:
    def __init__(self):
        self.last = {"sequence": 1}
        self.actions = []

    def step(self, action, amount=None):
        self.actions.append(action)
        return self.capture()

    def capture(self):
        self.last = {"sequence": self.last["sequence"] + 1}
        return self.last


def test_all_inspection_primitives_consume_budget_and_last_frame_is_retained():
    raw, records = Sensor(), []
    sensor = RecordedSensor(
        raw, lambda frame, command: records.append((frame, command)), max_primitives=2, max_seconds=10
    )
    sensor.step("look_down")
    sensor.step("turn_left")
    with pytest.raises(CollectionBudgetExceeded):
        sensor.step("move_forward")
    assert raw.actions == ["look_down", "turn_left"]
    assert records[-1] == ({"sequence": 3}, {"action": "turn_left", "amount": None})
    assert sensor.primitive_count == 2


def test_elapsed_budget_rejects_observation_and_motion_before_any_sensor_call():
    now = [0.0]
    raw = Sensor()
    sensor = RecordedSensor(raw, lambda *args: None, max_primitives=100, max_seconds=2, clock=lambda: now[0])
    now[0] = 2.0
    with pytest.raises(CollectionBudgetExceeded):
        sensor.capture()
    with pytest.raises(CollectionBudgetExceeded):
        sensor.step("turn_left")
    assert raw.last["sequence"] == 1 and not raw.actions


def test_exceptional_attempt_still_counts_without_inventing_a_frame():
    raw, records = Sensor(), []

    def failed(*args):
        raise OSError("sensor disconnected")

    raw.step = failed
    sensor = RecordedSensor(raw, lambda *args: records.append(args), max_primitives=1, max_seconds=10)
    with pytest.raises(OSError):
        sensor.step("move_forward")
    assert sensor.primitive_count == 1 and len(records) == 1
    with pytest.raises(CollectionBudgetExceeded):
        sensor.step("move_forward")


def navigation(frontiers=()):
    return SimpleNamespace(
        pose=None,
        update=lambda frame: None,
        calibrate_stationary_floor=lambda frame: {"reference_z_m": 0.0},
        frontiers=lambda: list(frontiers),
        plan=lambda *args, **kwargs: ([], "no_valid_goals"),
    )


def test_fixed_scan_and_no_frontier_failure_are_retained_without_forced_motion():
    raw, events = Sensor(), []
    sensor = RecordedSensor(raw, lambda *args: None, max_primitives=100, max_seconds=10)
    result = collect_history(sensor, navigation([[1, 0, 0]]), events.append)
    assert result == {"status": "no_admitted_frontier", "completed_legs": 0}
    assert raw.actions.count("turn_left") == 72
    assert raw.actions.count("look_down") == raw.actions.count("look_up") == 3
    assert "move_forward" not in raw.actions
    assert events[-1]["reason"] == "no_valid_goals"


def test_failed_measured_route_stops_collection_without_an_automatic_retry(monkeypatch):
    raw, events = Sensor(), []
    sensor = RecordedSensor(raw, lambda *args: None, max_primitives=100, max_seconds=10)
    nav = navigation([[1, 0, 0]])
    nav.pose = np.zeros(3)
    nav.plan = lambda *args, **kwargs: ([[0.5, 0, 0]], "ok")
    calls = []

    class Executor:
        def __init__(self, *args):
            pass

        def execute(self, route):
            calls.append(route)
            return {"success": False, "reason": "collision"}

    monkeypatch.setattr("emet.eval.exploration_history.HabitatMotionExecutor", Executor)
    result = collect_history(sensor, nav, events.append)
    assert result["status"] == "motion_failed" and result["reason"] == "collision"
    assert len(calls) == 1
    assert events[-1]["stage"] == "motion_finished"


def test_frontier_distance_ranking_is_independent_of_waypoint_count(monkeypatch):
    raw, events = Sensor(), []
    sensor = RecordedSensor(raw, lambda *args: None, max_primitives=100, max_seconds=10)
    nav = navigation([[1, 0, 0], [0.5, 0, 0]])
    nav.pose = np.zeros(3)
    nav.plan = lambda point, **kwargs: (
        [[1, 0, 0]] if point[0] == 1 else [[x, 0, 0] for x in np.linspace(0.05, 0.5, 10)],
        "ok",
    )

    class Executor:
        def __init__(self, *args):
            pass

        def execute(self, route):
            return {"success": False, "reason": "test_stop"}

    monkeypatch.setattr("emet.eval.exploration_history.HabitatMotionExecutor", Executor)
    collect_history(sensor, nav, events.append)
    selected = next(event for event in events if event["stage"] == "motion_started")
    assert selected["target"] == [0.5, 0, 0]


def test_interrupted_route_keeps_its_leg_and_measured_pose(monkeypatch):
    raw, events = Sensor(), []
    sensor = RecordedSensor(raw, lambda *args: None, max_primitives=100, max_seconds=10)
    nav = navigation([[1, 0, 0]])
    nav.pose = np.zeros(3)
    nav.plan = lambda *args, **kwargs: ([[0.5, 0, 0]], "ok")

    class Executor:
        def __init__(self, *args):
            pass

        def execute(self, route):
            raise CollectionBudgetExceeded("test limit")

    monkeypatch.setattr("emet.eval.exploration_history.HabitatMotionExecutor", Executor)
    with pytest.raises(CollectionBudgetExceeded):
        collect_history(sensor, nav, events.append)
    assert events[-1] == {"stage": "motion_interrupted", "leg": 0, "reason": "test limit", "current_pose": [0, 0, 0]}


def test_scene_selection_never_reads_task_or_scorer_columns(monkeypatch, tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("collection_driver", scripts / "collect_exploration_history.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    path = tmp_path / "questions.csv"
    path.write_text("placeholder")
    monkeypatch.setattr(module, "questions_csv_path", lambda: path)

    class Restricted(dict):
        def __getitem__(self, key):
            assert key in {"scene", "floor"}, "Task/scorer data reached the collector"
            return super().__getitem__(key)

    monkeypatch.setattr(module.csv, "DictReader", lambda stream: iter([Restricted(scene="scene-a", floor="1")]))
    assert module.scene_binding(0) == ("scene-a", 1)
