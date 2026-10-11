"""Physical controller bindings must preserve distinct wrist axes."""

import importlib.util
from pathlib import Path

import pytest


def test_arm_binding_uses_client_roll_pitch_yaw_order():
    path = Path(__file__).resolve().parents[3] / "scripts/eval_physical_tamp.py"
    spec = importlib.util.spec_from_file_location("eval_physical_tamp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {
        "joint_lift": 0.6,
        "joint_arm_l0": 0.01,
        "joint_arm_l1": 0.02,
        "joint_arm_l2": 0.03,
        "joint_arm_l3": 0.04,
        "joint_wrist_roll": -0.3,
        "joint_wrist_pitch": -0.7,
        "joint_wrist_yaw": 1.2,
    }
    command = module.arm_client_command(values)
    # arm_to() and MujocoZmqServer.manip_to() both use this wire order.
    assert command == [0, 0.6, 0.1, -0.3, -0.7, 1.2]


def test_trial_deadline_escapes_task_recovery():
    from types import SimpleNamespace

    from emet.controller.task.tamp.task_search import TaskPlan, TaskPlanStep, execute_task_plan

    path = Path(__file__).resolve().parents[3] / "scripts/eval_physical_tamp.py"
    spec = importlib.util.spec_from_file_location("eval_physical_tamp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def deadline(*args, **kwargs):
        raise module.TrialWallTimeout("trial_wall_timeout")

    plan = TaskPlan([TaskPlanStep("approach", {"xyt": [0, 0, 0]})], "object", None)
    with pytest.raises(module.TrialWallTimeout):
        execute_task_plan(SimpleNamespace(move_base_to=deadline), plan, executor=None, grasp_poses=[])
