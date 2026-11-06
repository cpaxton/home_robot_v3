"""Seven motion requests with masked exploration; executor retains the gripper.

Training uses the same history, pose masks and preview as inference. Extra
latent mask columns only control sampling/log probability; they do not change
the exported ordinary eight-output actor's deterministic mean.
"""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.distributions import StateDependentNoiseDistribution

from emet.rl.manipulation.base_component import BasePolicy
from emet.rl.manipulation.feedback_policy import FeedbackLatent


class MotionNoise(StateDependentNoiseDistribution):
    def proba_distribution_net(self, latent_dim, log_std_init=-2, latent_sde_dim=None):
        # Base errors 6, arm errors 18, free-base forward/yaw errors 18 each,
        # and a small stationary floor. No measured contact or task ID.
        return super().proba_distribution_net(latent_dim, log_std_init, latent_sde_dim=61)

    def sample(self):
        return super().sample() * self.mask

    def mode(self):
        return super().mode() * self.mask

    def log_prob(self, actions):
        return (self.distribution.log_prob(actions) * self.mask).sum(dim=-1)

    def entropy(self):
        return (self.distribution.entropy() * self.mask).sum(dim=-1)


class MotionLatent(FeedbackLatent):
    def __init__(self, *args, fields, history_width, steps, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields, self.history_width, self.steps = fields, history_width, steps
        self.mask_start = self.context_width + int(self.gate_fn is not None)
        self.latent_dim_pi += 4

    def forward_actor(self, features):
        latent = super().forward_actor(features)
        history = features[:, (self.steps - 1) * self.history_width : self.steps * self.history_width]
        masks = torch.cat([history[:, self.fields["base_mask"]], history[:, self.fields["hand_mask"]]], dim=1)
        return torch.cat([latent[:, : -self.feedback_width], masks, latent[:, -self.feedback_width :]], dim=1)


class WholeBodyPolicy(BasePolicy):
    motion_outputs = 7
    noise_type = MotionNoise

    def __init__(self, *args, arm_exploration_coefficient_std=0.03, **kwargs):
        if not np.isfinite(arm_exploration_coefficient_std) or not 0 < arm_exploration_coefficient_std <= 1:
            raise ValueError("arm exploration coefficient std must be in (0,1]")
        self.arm_exploration_coefficient_std = arm_exploration_coefficient_std
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        features = self.features_extractor
        self.mlp_extractor = MotionLatent(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            feedback_width=features.feedback_width,
            gate_fn=lambda x: features.turn_gate(x, self.turn_xy_tolerance),
            fields=features.fields,
            history_width=features.history_width,
            steps=features.steps,
        )

    def noise_configuration(self):
        mask = torch.zeros_like(self.log_std)
        mask[[0, 3, 60], 0] = 1
        mask[[1, 2, 4, 5, 60], 1] = 1
        mask[24:42, 0] = 1
        mask[42:60, 1] = 1
        mask[6:24, 2:] = 1
        mask[60, 2:] = 1
        coefficients = torch.full_like(mask, self.exploration_coefficient_std)
        coefficients[:, 2:] = self.arm_exploration_coefficient_std
        return mask, coefficients

    def _get_action_dist_from_latent(self, latent_pi):
        errors = latent_pi[:, -self.features_extractor.feedback_width :]
        masks = latent_pi[:, self.mlp_extractor.mask_start : self.mlp_extractor.mask_start + 4]
        base_mask, hand = masks[:, :3] > 0.5, masks[:, 3:4] > 0.5
        hand_errors = errors[:, [*range(3, 12), *range(15, 24)]]
        free_forward = (~base_mask[:, :2].any(dim=1, keepdim=True)) & hand
        free_yaw = (~base_mask[:, 2:3]) & hand
        noise = torch.cat(
            [
                errors[:, [0, 1, 2, 12, 13, 14]],
                hand_errors,
                hand_errors * free_forward,
                hand_errors * free_yaw,
                errors.new_full((len(errors), 1), 0.002),
            ],
            dim=1,
        )
        distribution = self.action_dist.proba_distribution(self.action_net(latent_pi), self.log_std, noise)
        base = base_mask.any(dim=1, keepdim=True) | hand
        distribution.mask = torch.cat([base.expand(-1, 2), hand.expand(-1, 5)], dim=1).to(latent_pi.dtype)
        return distribution


class FrozenGripActions(gym.Wrapper):
    def __init__(self, env, source):
        super().__init__(env)
        native = env.unwrapped
        if (
            native.profile.robot != "innate_mars"
            or source.robot != "innate_mars"
            or source.manifest["fingerprint"] != native.fingerprint
        ):
            raise ValueError("whole-body source robot/model mismatch")
        self.source, self.last_observation = source, None
        self.action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

    def reset(self, **kwargs):
        self.last_observation, info = self.env.reset(**kwargs)
        return self.last_observation, info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if (
            self.last_observation is None
            or action.shape != (7,)
            or not np.isfinite(action).all()
            or np.any(abs(action) > 1)
        ):
            raise ValueError("reset then provide seven finite normalized motion requests")
        full = np.r_[action, self.source(self.last_observation)[-1]]
        self.last_observation, reward, done, truncated, info = self.env.step(full)
        return self.last_observation, reward, done, truncated, info
