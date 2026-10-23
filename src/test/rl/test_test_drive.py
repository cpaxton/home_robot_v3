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
                    "base_xy_tolerance": 0.05,
                    "hand_position_tolerance": 0.005,
                    "hand_orientation_tolerance": 0.05,
                },
            }

            def __call__(self, observation):
                assert set(observation) == {"history", "preview"}
                assert observation["preview"].shape == (7, 18)
                return np.zeros(8, dtype=np.float32)

        policy = PreviewPolicy()
        sim.use_policy(policy, "preview-test")

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
        assert set(replacement.obs) == {"history"}
        assert replacement.env.observation_space.contains(replacement.obs)
        replacement.step()
    finally:
        sim.close()
        if replacement is not None:
            replacement.close()
