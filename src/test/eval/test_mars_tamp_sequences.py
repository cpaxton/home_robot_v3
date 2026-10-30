import json

import numpy as np
import pytest

from emet.eval.mars_tamp_agent import AgentToolDriver
from emet.eval.mars_tamp_control import CANONICAL_SUITE
from emet.eval.mars_tamp_sequences import run_sequence, stage_goals, validate_program


def program():
    return {
        "id": "shuttle",
        "instruction": "Deliver the blue object and return it.",
        "stages": [{"object": "blue", "support": "destination"}, {"object": "blue", "support": "source"}],
    }


def test_programs_are_declarative_and_use_existing_objects_and_distinct_supports():
    assert validate_program(program()) == program()
    for key, value in [
        ("stages", [{"object": "blue", "support": "source"}] * 2),
        ("stages", [{"object": "red", "support": "destination"}] * 2),
        ("instruction", ""),
        ("id", "../../commands"),
    ]:
        with pytest.raises(ValueError):
            validate_program({**program(), key: value})
    with pytest.raises(ValueError):
        validate_program({**program(), "python": "robot.reset()"})
    goals = stage_goals(program(), {"source_support": [0.1, 0.2, 0.3], "placement_region": [[1, 2], [3, 4]]})
    assert goals[0]["support_body"] == "destination" and goals[1]["support_body"] == "table"
    np.testing.assert_allclose(goals[1]["placement_region"], [[0.07, 0.17], [0.13, 0.23]])


def test_single_delivery_is_supported_without_changing_the_multi_transfer_contract():
    single = {**program(), "stages": program()["stages"][:1]}
    assert validate_program(single) == single
    assert len(validate_program(program())["stages"]) == 2
    with pytest.raises(ValueError):
        validate_program({**single, "stages": []})


@pytest.mark.sim
@pytest.mark.parametrize("record", [False, True])
def test_opt_in_diagnostics_align_actor_input_with_pre_step_physics(record):
    from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
    from emet.eval.mars_tamp_control import fixture

    case = json.loads(CANONICAL_SUITE.read_text())["cases"][0]
    env, admission = fixture(case)
    try:
        assert admission["accepted"]
        initial = env.data.qpos.copy()
        start = float(env.data.time)
        robot = MarsSimRobot(env, record_observations=record)
        robot.step(env.profile.dt)
        row = robot.trace[-1]
        assert ("physics_before" in row) is record
        assert ("physics_after" in row) is record
        if record:
            np.testing.assert_array_equal(row["physics_before"]["qpos"], initial)
            np.testing.assert_array_equal(row["physics_after"]["qpos"], env.data.qpos)
            assert row["physics_before"]["time"] == start
            assert row["physics_after"]["time"] == pytest.approx(start + env.profile.dt)
    finally:
        env.close()


def test_multistep_agent_does_not_stop_after_first_transfer():
    from emet.agent.tools import Tool

    names = ["scene_tasks", "plan_pick_place", "execute_pick_place_plan"] * 2
    replies = [json.dumps({"tool_calls": [{"name": n, "arguments": {}}]}) for n in names]
    registry = {n: Tool(n, n, {"type": "object", "properties": {}}, lambda: None) for n in set(names)}
    calls = []

    def call(name, **kwargs):
        calls.append(name)
        return {"status": "ok", "code": "ok", "data": {}}

    driver = AgentToolDriver(
        lambda system, prompt: replies.pop(0),
        input_mode="privileged",
        instruction="Complete two ordered transfers.",
        executions_required=2,
    )
    driver(call, {}, registry)
    assert calls == names
    assert len(driver.trace) == 6
    assert not replies


@pytest.mark.sim
def test_agent_cannot_replace_frozen_stage_region_or_reset_physics():
    case = json.loads(CANONICAL_SUITE.read_text())["cases"][0]
    turn = 0
    selected = None

    def infer(system, prompt):
        nonlocal turn, selected
        turn += 1
        if turn == 1:
            return json.dumps({"tool_calls": [{"name": "scene_tasks", "arguments": {"object_filter": "blue"}}]})
        # Handles are discovered at runtime; the model sees the real response.
        for line in prompt.splitlines():
            if "] " in line and '"tasks"' in line:
                result = json.loads(line.split("] ", 1)[1])
                selected = next(
                    x["task_ref"] for x in result["data"]["tasks"] if x["receptacle_name"] == "destination support"
                )
        assert selected is not None
        return json.dumps(
            {
                "tool_calls": [
                    {
                        "name": "plan_pick_place",
                        "arguments": {"task_ref": selected, "placement_region": [[-100, -100], [100, 100]]},
                    }
                ]
            }
        )

    row = run_sequence(case, program(), infer=infer, max_turns=2)
    assert not row["success"] and row["completed_stages"] == 0
    assert row["commands"] == [] and row["trace"] == []
    assert row["initial_state"]["qpos"] == row["final_state"]["qpos"]
    assert row["initial_state"]["time"] == row["final_state"]["time"]
    assert row["calls"][-1]["result"]["code"] == "frozen_stage_region_required"
    assert row["calls"][-1]["dispatched"] is False
