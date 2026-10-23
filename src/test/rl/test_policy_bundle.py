import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")

import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from emet.rl.env import MarsReachEnv
from emet.rl.policy import PPOPolicy, export_policy
from emet.rl.server import ReactiveController

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("transformer", [False, True])
def test_export_keeps_observation_normalization_and_rejects_wrong_contract(tmp_path, transformer):
    torch.set_num_threads(1)
    vec = VecNormalize(DummyVecEnv([MarsReachEnv]), norm_reward=False)
    try:
        from emet.rl.transformer import HistoryTransformer

        kwargs = {"features_extractor_class": HistoryTransformer} if transformer else {}
        model = sb3.PPO(
            "MultiInputPolicy", vec, device="cpu", seed=4, n_steps=16, batch_size=16, n_epochs=1, policy_kwargs=kwargs
        )
        model.learn(32)
        normalized = vec.reset()
        raw = vec.get_original_obs()
        expected = model.predict(normalized, deterministic=True)[0][0]
        export_policy(model, vec, tmp_path, {"seed": 4})
        policy = PPOPolicy(tmp_path)
        actual = policy({k: v[0] for k, v in raw.items()})
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        env = vec.venv.envs[0]
        controller = ReactiveController(env.model, env.data, policy=policy)
        controller.start({"base": [0.8, 0, 0]})
        expected_action = policy(controller.control.observe()[0])
        controller.step()
        assert controller.outcome is None
        np.testing.assert_allclose(controller.control.last_action[:2], expected_action[:2], atol=1e-6)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["profile"]["arm_joints"].reverse()
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="profile/model"):
            PPOPolicy(tmp_path)
    finally:
        vec.close()
