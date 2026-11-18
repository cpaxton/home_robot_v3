import json

import numpy as np
import pytest

from emet.rl.test_drive import Simulation, checkpoints, training_status

pytestmark = pytest.mark.sim


def test_only_simulation_modes_are_accepted():
    for mode in ("herman", "hardware", "innate_mars", "tcp://192.168.1.43"):
        with pytest.raises(ValueError, match="local simulation"):
            Simulation(mode)


def test_test_drive_changes_reference_without_teleporting_and_steps_physics():
    sim = Simulation("mars-base", seed=700001)
    try:
        before = sim.env.data.qpos.copy()
        target = sim.env.control.base_pose().copy()
        target[0] += 0.1
        sim.goal(target)
        np.testing.assert_array_equal(before, sim.env.data.qpos)
        before_time = sim.env.data.time
        sim.step()
        assert sim.env.data.time > before_time
        assert sim.state()["simulation_only"]
        assert "unsupported" in sim.state()["wheel_support"]
        with pytest.raises(ValueError):
            sim.goal([np.nan, 0, 0])
        with pytest.raises(ValueError, match="configured local roots"):
            sim.load_policy("/arbitrary/path", [])
    finally:
        sim.close()


def test_interactive_goal_nudge_moves_the_base_goal_without_teleporting():
    sim = Simulation("mars-base", seed=700002)
    try:
        before_pose = sim.env.data.qpos.copy()
        before_goal = sim.env.control.goal.base.copy()
        sim.nudge_goal([0.02, -0.01, 0], rotation_z=0.05)
        np.testing.assert_array_equal(before_pose, sim.env.data.qpos)
        np.testing.assert_allclose(sim.env.control.goal.base[:2], before_goal[:2] + [0.02, -0.01])
        assert sim.env.control.goal.base[2] == pytest.approx(before_goal[2] + 0.05)
    finally:
        sim.close()


def test_coordinated_test_drive_switches_masked_base_and_ee_goals():
    sim = Simulation("mars-coordinated", seed=700003)
    try:
        sim.control_mode = 1
        assert sim.state()["fields"][:3] == ["Base X (m)", "Base Y (m)", "Base yaw (rad)"]
        pose = sim.env.control.pose_features()[:9].copy()
        sim.control_mode = 2
        sim.goal(pose)
        assert sim.env.control.goal.mask[0] == 0
        assert sim.obs["target_mask"].tolist() == [False, True]
        sim.control_mode = 3
        sim.goal(pose)
        assert sim.env.control.goal.mask[0] == 1
        assert sim.obs["target_mask"].tolist() == [True, True]
        sim.nudge_goal([0.005, 0, 0], rotation_z=0.01)
        assert sim.state()["fields"][0] == "EE X (m)"
        with pytest.raises(ValueError, match="reachable"):
            sim.goal([2, 2, 2, *pose[3:]])
    finally:
        sim.close()


@pytest.mark.parametrize("action_mode", ["residual", "direct_velocity"])
def test_checkpoint_discovery_filters_incomplete_and_wrong_robot(tmp_path, action_mode):
    manifest = tmp_path / "manifest.json"
    from emet.rl.manipulation.profiles import STRETCH

    manifest.write_text(
        json.dumps(
            {
                "format": "emet-manipulation-v1",
                "robot": "stretch",
                "contract": STRETCH.contract(action_mode),
                "action_mode": action_mode,
                "steps": 1024,
            }
        )
    )
    assert not checkpoints([tmp_path], "stretch-arm")
    (tmp_path / "policy.zip").write_bytes(b"placeholder")
    assert len(checkpoints([tmp_path], "stretch-arm")) == 1
    assert not checkpoints([tmp_path], "mars-base")


def test_training_status_separates_initialization_from_ppo_and_partial_writes(tmp_path):
    seed = tmp_path / "seed-0"
    seed.mkdir()
    (seed / "run.json").write_text(json.dumps({"seed": 0, "initialization_steps": 32768, "steps": 262144}))
    (seed / "metrics.jsonl").write_text('{"timesteps": 40000}\n{"timesteps":')
    row = training_status([tmp_path])[0]
    assert row["steps"] == 40000 and row["ppo_steps"] == 7232 and row["phase"] == "PPO"


def test_stretch_test_drive_uses_physical_base_and_cartesian_hand_targets():
    sim = Simulation("stretch-arm", seed=700004)
    try:
        control = sim.env.control
        starting_qpos = sim.env.data.qpos.copy()
        pose = control.ee_pose().copy()
        sim.control_mode = 2
        sim.goal(pose)
        np.testing.assert_array_equal(sim.env.data.qpos, starting_qpos)
        assert control.base_mask.all()
        assert control.hand_mask
        sim.control_mode = 1
        base_target = control.base_pose().copy()
        base_target[0] += 0.04
        sim.goal(base_target)
        assert control.base_mask.all()
        assert not control.hand_mask
        assert sim.state()["fields"][0] == "Base X (m)"
        with pytest.raises(ValueError, match="unreachable"):
            sim.control_mode = 2
            sim.goal([2, 2, 2, 1, 0, 0, 1, 0, 0])
    finally:
        sim.close()


def test_mars_physical_test_drive_uses_wheels_and_reports_support():
    sim = Simulation("mars-arm", seed=610001)
    try:
        assert sim.env.profile.robot == "innate_mars"
        assert sim.state()["wheel_support"] == "evaluated in simulator"
        sim.control_mode = 2
        before = sim.env.data.qpos.copy()
        sim.goal(sim.env.control.ee_pose())
        np.testing.assert_array_equal(sim.env.data.qpos, before)
        sim.step()
        assert len(sim.state()["metrics"]["min_wheel_load"]) == 3
        applied = []
        original_step = sim.env.step

        def capture_push(action):
            applied.append(
                (
                    float(sim.env.data.xfrc_applied[sim.env.control.ee, 0]),
                    float(sim.env.data.xfrc_applied[sim.env.payload, 0]),
                )
            )
            return original_step(action)

        sim.env.step = capture_push
        sim.push()
        sim.step()
        assert applied == [(1.0, 0.0)]
        if sim.done:
            # A safety termination must remove the external force immediately.
            assert sim.env.data.xfrc_applied[sim.env.control.ee, 0] == 0
        else:
            assert sim.env.data.xfrc_applied[sim.env.control.ee, 0] > 0
        assert sim.env.data.xfrc_applied[sim.env.payload, 0] == 0
    finally:
        sim.close()


@pytest.mark.parametrize("mode", ["mars-arm", "stretch-arm"])
def test_physical_base_nudges_keep_se2_target_shape_and_preserve_state(mode):
    sim = Simulation(mode, scenario="reach", seed=610001)
    try:
        sim.control_mode = 1
        before = sim.env.data.qpos.copy()
        goal = sim.env.control.base_goal.copy()
        sim.nudge_goal([0.01, -0.02, 0.03], rotation_z=0.04)
        assert sim.env.control.base_goal.shape == (3,)
        np.testing.assert_allclose(sim.env.control.base_goal, goal + [0.01, -0.02, 0.04])
        np.testing.assert_array_equal(sim.env.data.qpos, before)
    finally:
        sim.close()


@pytest.mark.parametrize("mode", ["mars-arm", "stretch-arm"])
@pytest.mark.parametrize("offset", [(0, 0.06, 0), (0, -0.06, 0), (0.06, 0, 0), (-0.06, 0, 0)])
def test_feedback_physically_reaches_base_targets_in_all_directions(mode, offset):
    sim = Simulation(mode, scenario="reach", seed=610001)
    try:
        sim.control_mode = 1
        target = sim.env.control.base_pose() + offset
        sim.goal(target)
        for _ in range(400):
            sim.step()
            if sim.done:
                break
        error = sim.env.control.base_pose() - target
        assert sim.info["is_success"], (sim.info, error)
        assert np.linalg.norm(error[:2]) < 0.02
        assert abs(error[2]) < 0.04
        assert not sim.info["wheel_lift"]
        assert not sim.info["unsafe_contact"]
        if offset[0] != 0:
            # Stretch's non-unit wheel gear used to reduce wheel speed ninefold.
            assert sim.info["time"] < 6
    finally:
        sim.close()


@pytest.mark.parametrize("mode", ["mars-arm", "stretch-arm"])
@pytest.mark.parametrize("dy", [0.04, -0.04])
def test_whole_body_hand_tracking_converges_without_large_lateral_excursion(mode, dy):
    sim = Simulation(mode, scenario="reach", seed=610001)
    try:
        sim.control_mode = 3
        target = sim.env.control.ee_pose().copy()
        target[1] += dy
        before = sim.env.data.qpos.copy()
        sim.goal(target)
        np.testing.assert_array_equal(sim.env.data.qpos, before)
        peak_error = 0
        for _ in range(400):
            sim.step()
            peak_error = max(peak_error, sim.env._hand_pose_error()[0])
            if sim.done:
                break
        assert sim.info["is_success"], sim.info
        position, rotation = sim.env._hand_pose_error()
        assert position < 0.015
        assert rotation < 0.10
        assert peak_error < 0.06
        assert not sim.info["wheel_lift"]
        assert not sim.info["unsafe_contact"]
        # Continue real physics after the UI episode ends to expose wandering
        # that a frozen success frame would hide.
        base_at_stop = sim.env.control.base_pose().copy()
        hand_at_stop = sim.env.control.ee_pose().copy()
        import mujoco

        for _ in range(500):
            sim.env.control.apply()
            mujoco.mj_step(sim.env.model, sim.env.data)
        assert np.linalg.norm(sim.env.control.base_pose()[:2] - base_at_stop[:2]) < 0.005
        assert np.linalg.norm(sim.env.control.ee_pose()[:3] - hand_at_stop[:3]) < 0.01
    finally:
        sim.close()


@pytest.mark.parametrize("mode", ["mars-arm", "stretch-arm"])
def test_whole_body_mid_motion_retarget_preserves_state_history_and_command_continuity(mode):
    sim = Simulation(mode, scenario="reach", seed=610001)
    try:
        sim.control_mode = 3
        start = sim.env.control.ee_pose().copy()
        target = start.copy()
        target[1] += 0.04
        sim.goal(target)
        for _ in range(10):
            sim.step()
        assert not sim.done
        qpos = sim.env.data.qpos.copy()
        history = sim.env.control.history.copy()
        command = sim.env.control.base_command.copy()
        timestamp = float(sim.env.data.time)
        target = start.copy()
        target[1] -= 0.04
        sim.goal(target)
        np.testing.assert_array_equal(sim.env.data.qpos, qpos)
        np.testing.assert_array_equal(sim.env.control.history[:-1], history[:-1])
        np.testing.assert_array_equal(sim.env.control.base_command, command)
        assert sim.env.data.time == timestamp
        sim.step()
        p = sim.env.profile
        assert np.all(
            abs(sim.env.control.base_command - command)
            <= np.array([p.base_acceleration, p.base_yaw_acceleration]) * p.dt + 1e-9
        )
        for _ in range(400):
            if sim.done:
                break
            sim.step()
        assert sim.info["is_success"], sim.info
        assert not sim.info["wheel_lift"]
        assert not sim.info["unsafe_contact"]
    finally:
        sim.close()


@pytest.mark.parametrize("mode", ["mars-arm", "stretch-arm", "mars-coordinated"])
def test_target_selection_updates_masks_consistently_and_reset_can_keep_selection(mode):
    sim = Simulation(mode, scenario="reach", seed=610001)
    try:
        qpos = sim.env.data.qpos.copy()
        for selected in (1, 2, 3):
            sim.select_target_mode(selected)
            assert sim.control_mode == selected
            assert len(sim.state()["goal"]) == (3 if selected == 1 else 9)
            np.testing.assert_array_equal(sim.env.data.qpos, qpos)
            if mode == "mars-coordinated":
                assert bool(sim.env.control.goal.mask[0]) == (selected != 2)
                assert bool(sim.env.control.goal.mask[1:].any()) == (selected != 1)
            else:
                assert sim.env.control.base_mask.all()
                assert sim.env.control.hand_mask == (selected != 1)
        sim.obs, sim.info = sim.env.reset(seed=610002)
        sim.select_target_mode(1, sampled=True)
        assert sim.state()["control_mode"] == 1
        assert len(sim.state()["goal"]) == 3
    finally:
        sim.close()


def test_completed_physical_reach_can_resume_without_teleporting_or_clearing_faults():
    sim = Simulation("mars-arm", scenario="reach", seed=610001)
    try:
        sim.select_target_mode(2)
        sim.goal(sim.env.control.ee_pose())
        for _ in range(100):
            sim.step()
            if sim.done:
                break
        assert sim.info["is_success"]
        qpos, timestamp = sim.env.data.qpos.copy(), sim.env.data.time
        history = sim.env.control.history.copy()
        assert sim.resume_completed_reach()
        np.testing.assert_array_equal(sim.env.data.qpos, qpos)
        np.testing.assert_array_equal(sim.env.control.history[:-1], history[:-1])
        assert sim.env.data.time == timestamp
        sim.nudge_goal([0.01, 0, 0])
        for _ in range(100):
            sim.step()
            if sim.done:
                break
        assert sim.info["is_success"]
        sim.env.monitor.wheel_lift = True
        sim.env.control.stop()
        assert not sim.resume_completed_reach()
        assert sim.env.control.stopped
    finally:
        sim.close()


def test_preview_policy_contract_survives_simulation_replacement_and_feedback_switch(monkeypatch):
    from emet.rl.test_drive import Simulation

    sim = Simulation("mars-arm", "reach", seed=2100002)
    replacement = None
    try:

        class PreviewPolicy:
            robot = "innate_mars"
            action_mode = "direct_velocity"
            normalization = {"history": None, "preview": None}
            manifest = {
                "fingerprint": sim.env.fingerprint,
                "pose_only_targets": True,
                "environment": {
                    "tracking_reward": True,
                    "wheel_feedback_gain": 0.5,
                    "retention_context": True,
                    "whole_body_settling": True,
                    "base_xy_tolerance": 0.05,
                    "hand_position_tolerance": 0.005,
                    "hand_orientation_tolerance": 0.05,
                },
            }

            def __call__(self, observation):
                assert set(observation) == {"history", "preview"}
                assert observation["preview"].shape == (7, 18)
                assert observation["history"].shape == (20, 101)
                return np.zeros(8, dtype=np.float32)

        policy = PreviewPolicy()
        sim.use_policy(policy, "preview-test")
        assert sim.env.control.base_velocity_feedback_gain == 0.5

        def no_ik(*args, **kwargs):
            raise AssertionError("learned Cartesian targets must not run UI IK")

        monkeypatch.setattr(sim, "_solve_manipulation_pose", no_ik)
        original_q = sim.env.data.qpos.copy()
        sim.select_target_mode(2)
        sim.goal(sim.env.control.ee_pose())
        assert sim.env.control.base_mask.all()
        sim.select_target_mode(3)
        assert not sim.env.control.base_mask.any()
        goal = sim.env.control.ee_pose().copy()
        goal[0] += 0.02
        sim.goal(goal)
        assert not sim.env.control.base_mask.any()
        np.testing.assert_array_equal(sim.env.data.qpos, original_q)
        assert sim.env.observation_space.contains(sim.obs)
        sim.step()
        replacement = Simulation("mars-arm", "reach", seed=2100003)
        replacement.use_policy(sim.policy, sim.policy_name)
        replacement.select_target_mode(sim.control_mode, sampled=True)
        assert replacement.env.observation_space.contains(replacement.obs)
        replacement.step()
        replacement.use_policy(None, "Feedback controller")
        assert replacement.env.control.base_velocity_feedback_gain == 2.0
        assert not replacement.env.control.retention_context_enabled
        assert set(replacement.obs) == {"history"}
        assert replacement.env.observation_space.contains(replacement.obs)
        replacement.step()
    finally:
        sim.close()
        if replacement is not None:
            replacement.close()


def navigation_policy(sim, tmp_path, **execution):
    from dataclasses import replace

    from emet.rl.manipulation.base_navigation_feedback import BaseNavigationFeedback
    from emet.rl.manipulation.whole_body_actor import WholeBodyActor
    from emet.rl.manipulation.whole_body_bundle import export_actor

    cfg = replace(
        sim.env.config, action_mode="direct_velocity", retention_context=True,
        trajectory_preview=True, tracking_reward=True, physical_wheel_slew=True,
        wheel_feedback_gain=0.5, wheel_feedback_current_velocity=True,
        tracking_precision=4, terminal_hold_seconds=2,
        base_xy_tolerance=0.05, base_heading_tolerance=np.deg2rad(5), **execution,
    )
    return export_actor(
        WholeBodyActor(sim.env.profile.robot, hidden=16, base_rate_feedback=False,
                       base_navigation_feedback=BaseNavigationFeedback(reference_tracking=True, heading_gain_scale=2)),
        tmp_path / "actor", cfg, sim.env.fingerprint, steps=0,
    )


def test_navigation_ui_replans_preserve_physics_and_show_endpoint_not_intermediate_reference(tmp_path, monkeypatch):
    sim = Simulation("mars-arm", "reach", seed=610001)
    try:
        sim.use_policy(navigation_policy(sim, tmp_path), "navigation-test")
        assert sim.env.control.base_velocity_feedback_current
        assert sim.env.control.physical_wheel_slew_enabled
        assert sim.env.tracking_precision == 4
        assert sim.env.config.terminal_hold_seconds == 2
        sim.select_target_mode(1)
        target = sim.env.control.base_pose() + [0.08, 0, 0]
        sim.goal(target)
        np.testing.assert_allclose(sim.state()["goal"], target)
        assert sim.state()["metrics"]["base_endpoint_xy_error_m"] == pytest.approx(0.08)
        assert np.linalg.norm(sim.env.control.base_goal[:2] - target[:2]) > 0.07
        sim.nudge_goal([0.01, 0, 0])
        target[0] += 0.01
        np.testing.assert_allclose(sim.state()["goal"], target)
        for _ in range(10):
            sim.step()
        c, d = sim.env.control, sim.env.data
        previous = c.task_trajectory
        boundary = previous.state(c.trajectory_elapsed())
        qpos, qvel, ctrl, history, stamp = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), c.history.copy(), d.time
        target = c.base_pose() + [0.02, 0.06, 0.1]
        sim.goal(target)
        assert c.task_trajectory is not previous
        for index in (0, 2, 3):
            np.testing.assert_allclose(c.task_trajectory.state(0)[index], boundary[index], atol=1e-9)
        for measured, before in ((d.qpos, qpos), (d.qvel, qvel), (d.ctrl, ctrl), (c.history[:-1], history[:-1])):
            np.testing.assert_array_equal(measured, before)
        assert d.time == stamp
        np.testing.assert_allclose(sim.state()["goal"], target)
        installed = c.task_trajectory
        monkeypatch.setattr("emet.rl.manipulation.coupled_tracking.admit_reference", lambda *args: {"accepted": False, "reason": "test_collision"})
        with pytest.raises(ValueError, match="test_collision"):
            sim.goal(target + [0.01, 0, 0])
        assert c.task_trajectory is installed
        np.testing.assert_allclose(sim.state()["goal"], target)
        sim.select_target_mode(3)
        assert c.task_trajectory is None
        sim.goal(c.ee_pose())
        sim.step()
        assert c.task_trajectory is None
        c.stop()
        sim.select_target_mode(1)
        with pytest.raises(ValueError, match="nonfaulted"):
            sim.goal(target)
        assert c.stopped
    finally:
        sim.close()


def test_adjacent_drag_updates_are_coalesced_without_dropping_pause_or_mode_commands():
    import queue

    from emet.rl.test_drive import _drain_goal_updates

    pending = queue.Queue()
    edits = [{"op": "nudge", "translation": [0.01, 0, 0]}, {"op": "goal", "values": [0, 0.1, 0]}]
    pause, later = {"op": "pause"}, {"op": "target_mode", "mode": 3}
    for item in [*edits[1:], pause, later]:
        pending.put(item)
    updates, deferred = _drain_goal_updates(pending, edits[0])
    assert updates == edits
    assert deferred == pause
    assert pending.get_nowait() == later


def test_batched_base_nudges_accumulate_with_one_admission_and_invalid_batches_are_atomic(tmp_path, monkeypatch):
    sim = Simulation("mars-arm", "reach", seed=610001)
    try:
        sim.use_policy(navigation_policy(sim, tmp_path), "navigation-test")
        sim.select_target_mode(1)
        start = sim.base_endpoint()
        from emet.rl.manipulation import coupled_tracking
        original = coupled_tracking.admit_reference
        calls = []
        def admit(*args):
            calls.append(1)
            return original(*args)
        monkeypatch.setattr(coupled_tracking, "admit_reference", admit)
        sim.update_base_goals([
            {"op": "nudge", "translation": [0.01, 0, 0]},
            {"op": "nudge", "translation": [0.02, 0, 0], "rotation_z": 0.03},
        ])
        np.testing.assert_allclose(sim.base_endpoint(), start + [0.03, 0, 0.03])
        assert len(calls) == 1
        installed = sim.env.control.task_trajectory
        with pytest.raises(ValueError, match="target movement"):
            sim.update_base_goals([
                {"op": "nudge", "translation": [0.01, 0, 0]},
                {"op": "nudge", "translation": [1, 0, 0]},
            ])
        assert len(calls) == 1 and sim.env.control.task_trajectory is installed
    finally:
        sim.close()


@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4, 5])
def test_initial_navigation_reference_requires_all_six_base_velocities_to_be_quiet(tmp_path, axis):
    sim = Simulation("mars-arm", "reach", seed=610001)
    try:
        sim.use_policy(navigation_policy(sim, tmp_path), "navigation-test")
        sim.select_target_mode(1)
        c = sim.env.control
        sim.env.data.qvel[c.base_dof + axis] = 0.1
        with pytest.raises(ValueError, match="measured rest"):
            sim.goal(c.base_pose() + [0.05, 0, 0])
        assert c.task_trajectory is None
    finally:
        sim.close()


def test_live_viewer_goal_command_resumes_paused_simulation_without_run_button():
    import time

    from emet.rl.test_drive import TestDrive

    app = TestDrive([], "mars-arm")
    def wait_for(predicate):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with app.lock:
                state = dict(app.snapshot)
            assert state.get("status") != "error", state
            if predicate(state):
                return state
            time.sleep(0.02)
        raise AssertionError(f"viewer command did not complete: {state}")
    try:
        wait_for(lambda state: state.get("status") == "running")
        app.commands.put({"op": "target_mode", "mode": 1})
        wait_for(lambda state: state.get("control_mode") == 1)
        app.commands.put({"op": "pause"})
        paused = wait_for(lambda state: state.get("status") == "paused")
        target = np.array(paused["current"]) + [0.02, 0, 0]
        app.commands.put({"op": "goal", "values": target.tolist()})
        resumed = wait_for(lambda state: state.get("status") == "running" and state.get("metrics", {}).get("time", 0) > paused["metrics"]["time"])
        np.testing.assert_allclose(resumed["goal"], target)
        assert app.frame and resumed["simulation_only"]
    finally:
        app.closed.set()
        app.thread.join(timeout=10)
        assert not app.thread.is_alive()


def test_scoped_execution_survives_viewer_selection_and_mode_changes(tmp_path):
    sim=Simulation("mars-arm", "reach", seed=35874001)
    try:
        bundle=navigation_policy(sim,tmp_path,wheel_feedback_base_only_gains=(0.,.5))
        assert bundle.manifest["format"] == "emet-whole-body-v4"
        sim.use_policy(bundle,"scoped candidate")
        sim.select_target_mode(1)
        np.testing.assert_array_equal(sim.env.control.wheel_feedback_gains(),[0,.5])
        sim.select_target_mode(2)
        assert sim.env.control.wheel_feedback_gains()==.5
        sim.use_policy(None,"Feedback controller")
        assert sim.env.control.base_only_velocity_feedback_gains is None
        assert sim.env.control.wheel_feedback_gains()==2.
    finally:
        sim.close()
