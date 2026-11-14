# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Opt-in arm adaptation with an unchanged learned base/gripper function.

One ordinary feedback actor remains at inference. Only arm gain rows and arm
exploration are optimized; the critic remains trainable. Base/grip requests keep
their learned deterministic means instead of being zeroed or filtered.
"""

import torch

from emet.rl.manipulation.command_policy import ExecutorFeedbackPolicy, ExecutorGaussian


class ArmAdaptationGaussian(ExecutorGaussian):
    """Arm marginal with deterministic executor-masked base/grip commands."""

    def __init__(self, action_dim, mask):
        super().__init__(action_dim, mask)
        self.learning_mask = torch.zeros_like(mask)
        self.learning_mask[:, 2:-1] = mask[:, 2:-1]

    def sample(self):
        samples = self.distribution.sample()
        return torch.where(self.learning_mask.bool(), samples, self.distribution.mean) * self.mask

    def log_prob(self, actions):
        return (self.distribution.log_prob(actions) * self.learning_mask).sum(dim=-1)

    def entropy(self):
        return (self.distribution.entropy() * self.learning_mask).sum(dim=-1)


class ArmAdaptationPolicy(ExecutorFeedbackPolicy):
    """Freeze actor context/base/grip; learn five arm rows of the same head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.features_extractor.robot != "innate_mars" or self.action_space.shape != (8,):
            raise ValueError("arm adaptation requires the eight-command Mars feedback actor")
        if any(True for _ in self.features_extractor.parameters()):
            raise ValueError("arm adaptation requires parameter-free feedback features")

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if any(group.get("weight_decay", 0) != 0 for group in self.optimizer.param_groups):
            raise ValueError("arm adaptation requires zero weight decay to preserve frozen gain rows")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith(("mlp_extractor.value_net.", "value_net."))
                or name in {"action_net.gain.weight", "action_net.gain.bias", "log_std"}
            )
        for parameter in (self.action_net.gain.weight, self.action_net.gain.bias):
            mask = torch.zeros_like(parameter).reshape(self.action_net.motion_size, self.action_net.feedback_width, -1)
            mask[2:] = 1
            parameter.register_hook(lambda gradient, mask=mask.reshape(parameter.shape): gradient * mask)
        noise_mask = torch.zeros_like(self.log_std)
        noise_mask[2:-1] = 1
        self.log_std.register_hook(lambda gradient: gradient * noise_mask)
        self.optimizer = self.optimizer_class(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _executor_distribution(self, obs, latent):
        distribution = ArmAdaptationGaussian(self.action_space.shape[0], self.executor_mask(obs))
        return distribution.proba_distribution(self.action_net(latent), self.log_std)

    def frozen_function_state(self):
        """Exact protected tensors; capture after checkpoint transfer/setup."""
        protected = {}
        for name, value in self.state_dict().items():
            if name.startswith(("mlp_extractor.value_net.", "value_net.")):
                continue
            if name in {"action_net.gain.weight", "action_net.gain.bias"}:
                value = value.reshape(self.action_net.motion_size, self.action_net.feedback_width, -1)[:2]
            elif name == "log_std":
                value = value[[0, 1, -1]]
            protected[name] = value.detach().cpu().clone()
        return protected

    def assert_frozen_function(self, expected):
        current = self.frozen_function_state()
        if current.keys() != expected.keys():
            raise RuntimeError("frozen base/context/grip layout changed")
        for name in current:
            if current[name].dtype != expected[name].dtype or not torch.equal(current[name], expected[name]):
                raise RuntimeError("frozen base/context/grip function changed: " + name)
