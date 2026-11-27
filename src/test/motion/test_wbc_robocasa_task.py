# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""RoboCasa task-target bridge checks."""

from __future__ import annotations

import numpy as np
import pytest

from emet.rl.manipulation.wbc_robocasa_task import (
    _GRIPPER_CLOSE,
    _GRIPPER_OPEN,
    _PHASE_INTENT,
    RoboCasaPandaTask,
)


def test_cabinet_accessible_accepts_one_open_door():
    class _Fixture:
        name = "cab"
        door_joint_names = ("left", "right")

        def get_joint_state(self, env, names):
            del env, names
            return {"left": 0.90, "right": 0.73}

    task = object.__new__(RoboCasaPandaTask)
    task.env = object()
    task.cabinet_fixture = lambda: _Fixture()
    assert task.door_states() == (0.90, 0.73)
    assert task.cabinet_accessible()


def test_cabinet_accessible_rejects_barely_open_doors():
    class _Fixture:
        name = "cab"
        door_joint_names = ("left", "right")

        def get_joint_state(self, env, names):
            del env, names
            return {"left": 0.55, "right": 0.40}

    task = object.__new__(RoboCasaPandaTask)
    task.env = object()
    task.cabinet_fixture = lambda: _Fixture()
    assert not task.cabinet_accessible()


def test_phase_intent_labels_cover_task_phases():
    task = object.__new__(RoboCasaPandaTask)
    for phase, intent in _PHASE_INTENT.items():
        task.curr_phase = phase
        assert task._intent() == intent


def test_gripper_commands_close_both_mirrored_fingers():
    # The two finger command limits are mirrored: first (0, 0.04), second
    # (-0.04, 0). Normalized +1 opens the first and -1 opens the second.
    assert tuple(_GRIPPER_OPEN) == (1.0, -1.0)
    assert tuple(_GRIPPER_CLOSE) == (-1.0, 1.0)


@pytest.mark.sim
def test_build_task_poses_from_a_robocasa_scene():
    pytest.importorskip("robosuite")
    pytest.importorskip("robocasa")
    import robosuite
    from robosuite import load_part_controller_config

    from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend
    from emet.rl.manipulation.wbc_robocasa_task import RoboCasaPandaTask

    env = robosuite.make(
        env_name="PickPlaceCounterToCabinet",
        robots="PandaOmron",
        controller_configs=load_part_controller_config(default_controller="OSC_POSE"),
        translucent_robot=False,
        layout_and_style_ids=[[1, 1]],
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    try:
        backend = RoboCasaPandaBackend(env, base_backoff=0.0, target_mode="offset")
        task = RoboCasaPandaTask(env, backend)
        poses = task.build_poses(seed=1)
        assert poses.object_category
        assert poses.cabinet_name
        assert np.isfinite(poses.object_pose).all()
        assert np.isfinite(poses.place_pose).all()
        for q in (poses.pregrasp_q, poses.grasp_q, poses.lift_q, poses.place_q, poses.retreat_q):
            assert q.shape == (7,)
            assert np.isfinite(q).all()
            assert np.all(q >= backend._arm_low - 1e-9)
            assert np.all(q <= backend._arm_high + 1e-9)
        assert poses.grasp_pose[2] > poses.object_pose[2]
        assert poses.lift_pose[2] > poses.grasp_pose[2]

        # DAgger labels must be aligned to the observation that produced the
        # action, not to the post-step state.
        task._set_phase("pregrasp", poses.pregrasp_q, poses.pregrasp_pose)
        observation = task.layout.encode(task.backend._observation())
        expected_teacher = np.asarray(task.backend.teacher_action(), dtype=np.float64)
        expected_teacher[-2:] = _GRIPPER_OPEN
        step, _terminated, next_observation = task._step_action(observation, _GRIPPER_OPEN)
        assert step.observation is observation
        assert np.array_equal(next_observation, task.layout.encode(task.backend._observation()))
        assert np.allclose(step.teacher_action, expected_teacher)
        assert tuple(step.action[-2:]) == tuple(_GRIPPER_OPEN)
    finally:
        env.close()
