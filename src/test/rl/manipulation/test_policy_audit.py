from types import SimpleNamespace

import numpy as np
import pytest

from emet.eval.mars_policy_audit import describe_input, phase_predictions
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import MARS, PROFILES


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
@pytest.mark.parametrize("context", [False, True])
def test_named_history_layout_matches_actual_encoding(robot, context):
    env = ManipulationEnv(ManipulationConfig(robot=robot, trajectory_preview=True, retention_context=context))
    try:
        obs = env.control.observe()["history"]
        fields, width = history_layout(PROFILES[robot], retention_context=context)
        assert obs.shape == (PROFILES[robot].history_steps, width)
        np.testing.assert_array_equal(obs[-1, fields["phase"]], [1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(obs[-1, fields["hand_pose"]], env.control.ee_pose())
        np.testing.assert_allclose(obs[-1, fields["base_pose"]], env.control.base_pose())
        np.testing.assert_array_equal(obs[-1, fields["base_mask"]], env.control.base_mask)
        assert fields["target_tokens"].stop - fields["target_tokens"].start == 20
        if context:
            assert fields["retain_object_command"].start == width - 2
    finally:
        env.close()


def test_input_audit_names_clipped_phase_and_keeps_raw_values():
    fields, width = history_layout(MARS, retention_context=True)
    values = np.zeros((MARS.history_steps, width))
    values[:, fields["phase"].start + 1] = 1
    variance = np.ones_like(values)
    variance[:, fields["phase"].start + 1] = 0.002
    policy = SimpleNamespace(
        manifest={"environment": {"retention_context": True}, "epsilon": 1e-8, "clip": 10},
        normalization={"history": (np.zeros_like(values), variance)},
        mean=np.zeros_like(values),
    )
    before = values.copy()
    report = describe_input(policy, {"history": values})
    assert report["history"]["clipped_fraction"] == pytest.approx(1 / width)
    assert report["history"]["largest_columns"][0]["name"] == "phase/pregrasp"
    assert report["history"]["max_abs_z"] > 20
    np.testing.assert_array_equal(values, before)


def test_phase_counterfactuals_do_not_mutate_original_history():
    fields, width = history_layout(MARS)
    observations = []

    class Actor:
        manifest = {"environment": {}}

        def __call__(self, value):
            observations.append(value["history"].copy())
            return np.zeros(8)

    values = np.zeros((MARS.history_steps, width))
    original = values.copy()
    result = phase_predictions(Actor(), {"history": values})
    assert len(result) == len(observations) == 7
    for index, value in enumerate(observations):
        np.testing.assert_array_equal(value[:, fields["phase"]].sum(axis=1), np.ones(MARS.history_steps))
        assert np.all(value[:, fields["phase"].start + index] == 1)
    np.testing.assert_array_equal(values, original)
