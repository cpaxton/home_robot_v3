"""Arm feedback initialization for history-conditioned PPO refinement."""

import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from emet.rl.env import EnvConfig, MarsReachEnv
from emet.rl.policy import export_policy
from emet.rl.profile import MARS


class ArmHistoryFeatures(BaseFeaturesExtractor):
    """Measured state, recent motion and an editable joint feedback prior."""

    fixed_normalization = True

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=25 + 15 + 45 + 5 * 25 + 7)

    def forward(self, observations):
        state, goal = observations["state"], observations["goal"]
        arm = (goal[:, 4:9] / (MARS.max_joint_speed * MARS.control_dt) - state[:, 10:15]).clamp(-1, 1)
        prior = torch.cat([torch.zeros_like(arm[:, :2]), arm], dim=1)
        return torch.cat(
            [state, goal, observations["obstacles"], observations["history"][:, -5:, :25].flatten(1), prior], dim=1
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = EnvConfig(arm_only=True, horizon_seconds=6)
    env = VecNormalize(DummyVecEnv([lambda: MarsReachEnv(config)]), norm_reward=False, clip_obs=100)
    env.training = False  # Identity observation scaling is fixed across export/resume.
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            device="cpu",
            seed=12,
            n_steps=256,
            batch_size=256,
            learning_rate=3e-5,
            ent_coef=0,
            policy_kwargs={
                "features_extractor_class": ArmHistoryFeatures,
                "net_arch": {"pi": [], "vf": [128, 128]},
                "log_std_init": -4,
            },
        )
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.action_net.weight[:, -7:] = torch.eye(7)
        from dataclasses import asdict

        export_policy(model, env, args.output, {"env": asdict(config), "steps": 0, "algorithm": "arm feedback prior"})
    finally:
        env.close()


if __name__ == "__main__":
    main()
