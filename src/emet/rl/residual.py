"""Initialize an unrestricted residual policy at the exact base feedback controller."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from emet.rl.base import BASE, MarsBaseEnv
from emet.rl.env import EnvConfig
from emet.rl.model import PROXIES
from emet.rl.networks import FixedScaleBaseHistoryFeatures
from emet.rl.policy import PPOPolicy, export_policy


class ResidualBaseFeatures(FixedScaleBaseHistoryFeatures):
    """Analytic feedback prior plus learned history/clearance corrections.

    The final linear actor starts with identity weights on the last two features.
    PPO can change every coefficient; there is no penalty tying it to the teacher.
    """

    def __init__(self, observation_space, raw_mean, raw_std):
        super().__init__(observation_space)
        self._features_dim += 4
        self.register_buffer("raw_mean", torch.tensor(raw_mean, dtype=torch.float32))
        self.register_buffer("raw_std", torch.tensor(raw_std, dtype=torch.float32))

    def forward(self, observations):
        raw = observations["history"][:, -1] * self.raw_std + self.raw_mean
        x, y = raw[:, 0], raw[:, 1]
        distance = torch.linalg.vector_norm(raw[:, :2], dim=1)
        bearing = torch.atan2(y, x)
        reverse = bearing.abs() > np.pi / 2
        bearing = torch.where(reverse, torch.atan2(torch.sin(bearing + np.pi), torch.cos(bearing + np.pi)), bearing)
        v = torch.where(reverse, -1.0, 1.0) * (1.5 * distance).clamp(max=BASE.max_v) * bearing.cos().clamp(min=0)
        arrived = distance <= BASE.xy_tolerance * 0.6
        v = torch.where(arrived, 0.0, v) / BASE.max_v
        yaw = torch.atan2(raw[:, 2], raw[:, 3])
        w = 2.5 * torch.where(arrived, yaw, bearing) / BASE.max_w
        prior = torch.stack([v, w], dim=1).clamp(-1, 1)
        clearance = raw[:, 9 : 9 + len(PROXIES)].min(dim=1).values
        proximity = ((0.20 - clearance) / 0.20).clamp(0, 1)
        return torch.cat([super().forward(observations), prior * proximity[:, None], prior], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = PPOPolicy(args.reference_bundle)
    if reference.task != "base":
        parser.error("base reference required")
    torch.set_num_threads(1)
    config = EnvConfig(**reference.manifest["training"]["env"])
    env = VecNormalize(DummyVecEnv([lambda: MarsBaseEnv(config)]), norm_reward=False)
    try:
        for key, stats in reference.manifest["normalization"].items():
            rms = env.obs_rms[key]
            rms.mean, rms.var, rms.count = np.asarray(stats["mean"]), np.asarray(stats["var"]), stats["count"]
        env.training = False
        rms = env.obs_rms["history"]
        model = PPO(
            "MultiInputPolicy",
            env,
            device="cpu",
            n_steps=256,
            batch_size=256,
            learning_rate=0.0001,
            ent_coef=0,
            seed=11,
            policy_kwargs={
                "features_extractor_class": ResidualBaseFeatures,
                "features_extractor_kwargs": {
                    "raw_mean": rms.mean[-1].tolist(),
                    "raw_std": np.sqrt(rms.var[-1] + env.epsilon).tolist(),
                },
                "net_arch": {"pi": [], "vf": [128, 128]},
                "log_std_init": -3.5,
            },
        )
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.action_net.weight[:, -2:] = torch.eye(2)
        export_policy(
            model,
            env,
            args.output,
            {
                "env": reference.manifest["training"]["env"],
                "algorithm": "exact feedback prior, unrestricted linear residual",
                "reference": str(Path(args.reference_bundle)),
                "steps": 0,
            },
        )
        print(
            json.dumps(
                {
                    "output": args.output,
                    "actor_parameters": sum(p.numel() for p in model.policy.action_net.parameters()),
                }
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
