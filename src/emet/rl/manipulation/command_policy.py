"""One feedback actor with executor-visible action masks and command units.

Masking only removes requests the executor already ignores. Active means are
unchanged; no temporal action filter, teacher residual or task manager is added.
The same mask governs sampling, deterministic means, likelihood and entropy.
"""

import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution

from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures
from emet.rl.manipulation.normalization import CommandVecNormalize
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import PROFILES


def component_zero_command_columns(profile, *, retention_context):
    """These actor-unit channels are identically zero in the arm component."""
    fields, _ = history_layout(profile, retention_context=retention_context)
    previous = fields["proprio/previous_action"]
    return [
        *range(fields["proprio/base_command"].start, fields["proprio/base_command"].stop),
        *range(previous.start, previous.start + (2 if profile.wheel_bodies else 0)),
        previous.stop - 1,
    ]


class ExecutorCommandVecNormalize(CommandVecNormalize):
    """Keep known normalized wheel/grip requests in [-1, 1] actor units.

    An arm-only producer observes constant zero wheel/grip requests. Empirical
    variance then collapses and ordinary new requests saturate frozen history.
    Keep arm-command statistics unchanged for the learned component transfer.
    Exported ordinary bundle statistics implement these units at runtime.
    """

    @property
    def component_zero_columns(self):
        return component_zero_command_columns(self.profile, retention_context=self.retention_context)

    def pin_command_features(self):
        super().pin_command_features()
        stats = self.obs_rms["history"]
        stats.mean[..., self.component_zero_columns] = 0
        stats.var[..., self.component_zero_columns] = 1 - self.epsilon


class ExecutorGaussian(DiagGaussianDistribution):
    def __init__(self, action_dim, mask):
        super().__init__(action_dim)
        self.mask = mask

    def sample(self):
        return super().sample() * self.mask

    def mode(self):
        return super().mode() * self.mask

    def log_prob(self, actions):
        return (self.distribution.log_prob(actions) * self.mask).sum(dim=-1)

    def entropy(self):
        return (self.distribution.entropy() * self.mask).sum(dim=-1)


class ExecutorFeedbackPolicy(FeedbackActorCriticPolicy):
    """The existing actor/trunk, with zero probability mass on ignored requests.

    Visible target masks permit the same base/arm requests as control.accept().
    Closed grip correction additionally requires the executor's retention intent.
    Independent executor holds/stops still override the actor unconditionally.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("use_sde", False):
            raise ValueError("executor action masking currently requires diagonal Gaussian exploration")
        super().__init__(*args, **kwargs)
        features = self.features_extractor
        if not isinstance(features, GoalFeedbackFeatures) or "retain_object_command" not in features.fields:
            raise ValueError("executor action masking requires feedback features and retention context")

    def executor_mask(self, obs):
        history = obs["history"][:, -1]
        fields = self.features_extractor.fields
        hand = history[:, fields["hand_mask"]] > 0.5
        base = (history[:, fields["base_mask"]] > 0.5).any(dim=1, keepdim=True) | hand
        phase = history[:, fields["phase"]]
        closed = (phase[:, [3, 4, 6]] > 0.5).any(dim=1, keepdim=True)
        retain = history[:, fields["retain_object_command"]] > 0.5
        profile = PROFILES[self.features_extractor.robot]
        blocks = ([base.expand(-1, 2)] if profile.wheel_bodies else []) + [
            hand.expand(-1, profile.arm_size),
            closed & retain,
        ]
        return torch.cat(blocks, dim=1).to(history.dtype)

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        actor, critic = features
        return self.mlp_extractor.forward_actor(actor), self.mlp_extractor.forward_critic(critic)

    def _executor_distribution(self, obs, latent):
        distribution = ExecutorGaussian(self.action_space.shape[0], self.executor_mask(obs))
        return distribution.proba_distribution(self.action_net(latent), self.log_std)

    def forward(self, obs, deterministic=False):
        actor, critic = self._latents(obs)
        distribution = self._executor_distribution(obs, actor)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape)), self.value_net(critic), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        actor, critic = self._latents(obs)
        distribution = self._executor_distribution(obs, actor)
        return self.value_net(critic), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        if not self.share_features_extractor:
            features = features[0]
        return self._executor_distribution(obs, self.mlp_extractor.forward_actor(features))
