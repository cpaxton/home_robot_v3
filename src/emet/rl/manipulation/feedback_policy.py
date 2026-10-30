"""Learned gain scheduling with an explicit stationary motion equilibrium.

The mean motion is a learned matrix applied to current/future pose errors and
measured velocities. Context/history can change every gain but cannot produce
constant motion when all those signals vanish. No teacher action is evaluated.
The gripper keeps a separate bounded head and executor-only release authority.
"""

import math

import torch
import torch.nn.functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn

from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.policy import ManipulationMLPFeatures
from emet.rl.manipulation.profiles import PROFILES


class GoalFeedbackFeatures(ManipulationMLPFeatures):
    """Ordinary four-sample history/preview context plus an error signal tail."""

    def __init__(self, observation_space, *, robot, retention_context=False, orientation_scale=1.0):
        super().__init__(observation_space)
        if not self.has_preview:
            raise ValueError("feedback features require trajectory preview")
        self.fields, width = history_layout(PROFILES[robot], retention_context=retention_context)
        self.preview_fields, _ = preview_layout()
        if observation_space["history"].shape[-1] != width:
            raise ValueError("feedback history contract mismatch")
        self.feedback_width = 24 + PROFILES[robot].arm_size + 6
        if not math.isfinite(orientation_scale) or not 0.05 <= orientation_scale <= 1:
            raise ValueError("invalid feedback angular scale")
        self.orientation_scale = float(orientation_scale)
        self.robot = robot
        self.history_width = observation_space["history"].shape[-1]
        self.preview_shape = observation_space["preview"].shape
        self._features_dim += self.feedback_width

    def turn_gate(self, features, tolerance):
        """Smooth mode weight from announced targets/masks, never case IDs.

        Native feedback normalization fixes XY error scale at 0.1 m. Fully
        select restoring heading feedback inside the accepted XY region;
        transition smoothly to general feedback between one and two radii.
        Inspect every preview target so upcoming travel retains general control.
        """
        start = self.steps * self.history_width
        preview = features[:, start : start + math.prod(self.preview_shape)].reshape(-1, *self.preview_shape)
        xy = preview[:, :, self.preview_fields["base_error"]][:, :, :2] * 0.1
        distance = torch.linalg.vector_norm(xy, dim=-1).amax(dim=-1)
        x = ((2 * tolerance - distance) / tolerance).clamp(0, 1)
        smooth = x**3 * (10 - 15 * x + 6 * x**2)
        eligible = (preview[:, :, self.preview_fields["base_mask"]][:, :, 2] > 0.5).all(dim=1)
        eligible &= (preview[:, :, self.preview_fields["hand_mask"]] < 0.5).all(dim=1).all(dim=1)
        return (smooth * eligible)[:, None]

    def forward(self, obs):
        context = super().forward(obs)
        history = obs["history"][:, -1]
        samples = []
        for index in (1, 4):  # announced offsets 0 and +0.2 s in preview_v1
            preview = obs["preview"][:, index]
            base = preview[:, self.preview_fields["base_error"]]
            hand = preview[:, self.preview_fields["hand_error"]]
            identity = hand.new_tensor([1, 0, 0, 1, 0, 0])
            mask = preview[:, self.preview_fields["hand_mask"]]
            samples += [base, hand[:, :3], (hand[:, 3:] - identity * mask) / self.orientation_scale]
        samples += [history[:, self.fields["proprio/velocities"]], history[:, self.fields["base_velocity"]]]
        return torch.cat([context, *samples], dim=1)


class FeedbackLatent(MlpExtractor):
    def __init__(self, *args, feedback_width, gate_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_width = feedback_width
        self.context_width = self.latent_dim_pi
        self.gate_fn = gate_fn
        self.latent_dim_pi += feedback_width + int(gate_fn is not None)

    def forward_actor(self, features):
        blocks = [self.policy_net(features)]
        if self.gate_fn is not None:
            blocks.append(self.gate_fn(features))
        return torch.cat([*blocks, features[:, -self.feedback_width :]], dim=1)


class FeedbackAction(nn.Module):
    def __init__(self, context_width, feedback_width, action_size):
        super().__init__()
        self.feedback_width, self.motion_size = feedback_width, action_size - 1
        self.gain = nn.Linear(context_width, self.motion_size * feedback_width)
        self.grip = nn.Linear(context_width, 1)
        self.zero_output()

    def zero_output(self):
        for layer in (self.gain, self.grip):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, latent):
        context, errors = latent[:, : -self.feedback_width], latent[:, -self.feedback_width :]
        gain = self.gain(context).reshape(-1, self.motion_size, self.feedback_width)
        motion = torch.einsum("bae,be->ba", gain, errors) / math.sqrt(self.feedback_width)
        return torch.cat([torch.tanh(motion), torch.tanh(self.grip(context))], dim=1)


class FeedbackActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, restoring_yaw_feedback=False, turn_xy_tolerance=0.005, **kwargs):
        if type(restoring_yaw_feedback) is not bool or not math.isfinite(turn_xy_tolerance) or turn_xy_tolerance <= 0:
            raise ValueError("invalid restoring yaw feedback settings")
        self.restoring_yaw_feedback = restoring_yaw_feedback
        self.turn_xy_tolerance = turn_xy_tolerance
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        if self.restoring_yaw_feedback and self.features_extractor.robot != "innate_mars":
            raise ValueError("restoring yaw head currently supports Mars only")
        self.mlp_extractor = FeedbackLatent(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            feedback_width=self.features_extractor.feedback_width,
            gate_fn=(lambda features: self.features_extractor.turn_gate(features, self.turn_xy_tolerance))
            if self.restoring_yaw_feedback
            else None,
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        head = RestoringYawAction if self.restoring_yaw_feedback else FeedbackAction
        self.action_net = head(
            self.mlp_extractor.context_width, self.features_extractor.feedback_width, self.action_space.shape[0]
        ).to(self.device)
        # The replacement head must belong to the optimizer created for PPO.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class RestoringYawAction(FeedbackAction):
    """Learned positive restoring gains and opposing damping for base-only turns.

    General arm/translation feedback is retained outside the selected mode. This
    is a policy parameterization, not a teacher residual or an action filter.
    It guarantees restoring mean action at rest only when the mode weight is 1
    and current/future yaw errors agree; it is not a closed-loop certificate.
    """

    def __init__(self, context_width, feedback_width, action_size):
        super().__init__(context_width, feedback_width, action_size)
        self.context_width = context_width
        self.turn_gain = nn.Linear(context_width, 3)
        nn.init.zeros_(self.turn_gain.weight)
        with torch.no_grad():
            self.turn_gain.bias.copy_(torch.log(torch.expm1(torch.tensor([2.0, 2.0, 1.0]))))

    def forward(self, latent):
        context = latent[:, : self.context_width]
        gate = latent[:, self.context_width : self.context_width + 1]
        errors = latent[:, -self.feedback_width :]
        general = super().forward(torch.cat([context, errors], dim=1))
        gains = F.softplus(self.turn_gain(context))
        yaw = (gains[:, 0] * errors[:, 2] + gains[:, 1] * errors[:, 14] - gains[:, 2] * errors[:, -1]) / math.sqrt(
            self.feedback_width
        )
        yaw = (1 - gate[:, 0]) * general[:, 1] + gate[:, 0] * torch.tanh(yaw)
        return torch.cat([general[:, :1], yaw[:, None], general[:, 2:]], dim=1)
