"""Observation boundaries and stale-source guards without native rendering."""

import copy
from types import SimpleNamespace

from emet.eval.agent_tasks.agent_runner import ObservedFixture
from emet.eval.agent_tasks.spec import load_suite


class Recording:
    def __init__(self):
        self.step, self.events = 0, []

    def append(self, kind, **payload):
        self.events.append((kind, copy.deepcopy(payload)))
        self.step += 1


def environment():
    suite = load_suite()
    positions = {k: v["position"][:] for k, v in suite["scene"]["objects"].items()}
    visible = {"red_cylinder": {"position": positions["red_cylinder"], "pixels": 20}}
    robot = SimpleNamespace(
        scene=suite["scene"],
        xyt=[0, 0, 0],
        held=[],
        positions=lambda: positions,
        visible_objects=lambda: visible,
        images=lambda: {},
        camera_metadata=lambda: {},
    )
    return ObservedFixture(robot, Recording(), suite["episodes"][0]), positions, visible


def test_hidden_objects_never_enter_policy_inventory():
    env, _, _ = environment()
    state, image = env.observe()
    assert state["visible_objects"] == ["red cylinder"]
    assert state["remembered_objects"] == {"red cylinder": "kitchen"}
    assert "blue cube" not in str(state) and "goals" not in state
    assert image is None


def test_unseen_change_preserves_belief_until_observation():
    env, positions, visible = environment()
    env.observe()
    before = copy.deepcopy(env.memory)
    positions["red_cylinder"] = [0.45, -1.2, 0.56]
    visible.clear()
    env.observe()
    assert env.memory == before and env.revisions == []
    visible["red_cylinder"] = {"position": positions["red_cylinder"], "pixels": 25}
    env.observe()
    assert env.memory["red_cylinder"]["position"][0] == 0.45
    assert env.revisions[0]["old"]["position"][0] == 0
    assert env.recorder.events[0][1]["policy"]["memory"]["red_cylinder"]["position"][0] == 0


def test_stale_plan_guard_does_not_reveal_new_position(monkeypatch):
    from emet.agent import tools

    env, positions, visible = environment()
    visible["red_tray"] = {"position": positions["red_tray"], "pixels": 25}
    env.observe()
    positions["red_cylinder"] = [0.45, -1.2, 0.56]
    monkeypatch.setattr(
        tools,
        "get_tools",
        lambda _: [SimpleNamespace(name="execute_pick_place_plan", description="execute", parameters={})],
    )
    plan = next(t for t in env.tools() if t.name == "plan_pick_place")
    response = plan.func("red cylinder", "red tray")
    assert "stale" in response and "kitchen" in response and "0.45" not in response
    assert env.memory["red_cylinder"]["position"][0] == 0


def test_failed_planning_cannot_return_previous_handle(monkeypatch):
    from emet.agent import tools

    env, positions, visible = environment()
    visible["red_tray"] = {"position": positions["red_tray"], "pixels": 25}
    env.observe()
    env.context.update(_tamp_plan_counter=1, _tamp_plans={"plan:1": {}})
    monkeypatch.setattr(
        tools,
        "get_tools",
        lambda _: [
            SimpleNamespace(name="execute_pick_place_plan", description="execute", parameters={}),
            SimpleNamespace(name="plan_pick_place", func=lambda **_: "planning failed"),
        ],
    )
    plan = next(t for t in env.tools() if t.name == "plan_pick_place")
    assert plan.func("red cylinder", "red tray") == "planning failed"


def test_search_uses_visible_pixels_and_stops_at_room_budget(monkeypatch):
    from emet.agent import tools

    env, positions, visible = environment()
    visits = []

    def move(goal):
        env.robot.xyt = goal
        visits.append(env.room_at(goal))
        visible.clear()
        if visits[-1] == "living":
            visible["blue_cube"] = {"position": positions["blue_cube"], "pixels": 10}

    env.robot.move_base_to = move
    monkeypatch.setattr(
        tools,
        "get_tools",
        lambda _: [SimpleNamespace(name="execute_pick_place_plan", description="execute", parameters={})],
    )
    search = next(t for t in env.tools() if t.name == "find_objects")
    assert "Observed blue cube in living" in search.func("blue cube")
    assert "dining" not in visits
    visits.clear()
    assert "not visible" in search.func("nonexistent teapot")
    assert len(visits) == len(env.robot.scene["rooms"])
    assert "nonexistent teapot" not in env.memory


def test_temporal_scorer_allows_redelivery_after_world_change():
    env, positions, _ = environment()
    env.episode = load_suite()["episodes"][2]
    env.robot.relocate = lambda name, position: positions.update({name: position[:]})
    env.observe()
    positions["red_cylinder"] = [7.65, -1.2, 0.56]
    env.on_event("tool_result", {})
    positions["green_marker"] = [4.35, -1.2, 0.56]
    env.on_event("tool_result", {})
    assert "red_cylinder" not in env.goal_events
    assert positions["red_cylinder"][0] == 0.45
    positions["red_cylinder"] = [7.65, -1.2, 0.56]
    env.on_event("tool_result", {})
    assert env.first_seen["red_cylinder"] < env.goal_events["green_marker"] < env.goal_events["red_cylinder"]
