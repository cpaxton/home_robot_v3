"""One masked actor with a full-window GRU and three independent privileged critics."""

import math

import torch
from torch import nn

from emet.rl.manipulation.base_navigation_feedback import BaseNavigationFeedback, navigation_feedback_gain
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.whole_body_observation import PRIVILEGED_WIDTH


class SensorEncoder(nn.Module):
    def __init__(self, robot):
        super().__init__()
        self.profile = PROFILES[robot]
        self.fields, self.width = history_layout(self.profile, retention_context=True)
        self.preview_fields, _ = preview_layout()
        p = self.profile
        names = (
            "positions",
            "velocities",
            "target_error",
            "grip_position",
            "grip_error",
            "grip_reference",
            "previous_action",
        )
        self.names = tuple("proprio/" + n for n in names) + ("phase", "retain_object_command", "gripper_velocity")
        scales = (
            [math.pi] * p.arm_size
            + list(p.speed)
            + [0.04] * p.arm_size
            + [max(abs(p.gripper_open - p.gripper_closed), 0.001)] * 3
            + [1.0] * p.action_size
            + [1.0] * 9
        )
        self.token_width = len(scales) + 9  # base velocity6 + projected hand gravity3
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer(
            "motion_scale",
            torch.tensor([max(p.base_speed, 0.1)] * 2 + [max(p.base_yaw_speed, 0.5)] + [0.1] * 3 + [1.0] * 3),
        )

    def forward(self, obs):
        h = obs["history"]
        selected = torch.cat([h[..., self.fields[name]] for name in self.names], dim=-1) / self.scales
        velocity = h[..., self.fields["base_velocity"]].clone()
        yaw = h[..., self.fields["base_pose"]][..., 2]
        co, si = torch.cos(yaw), torch.sin(yaw)
        for start in (0, 3):
            x, y = velocity[..., start].clone(), velocity[..., start + 1].clone()
            velocity[..., start] = co * x + si * y
            velocity[..., start + 1] = -si * x + co * y
        velocity = velocity / velocity.new_tensor(
            [max(self.profile.base_speed, 0.1)] * 3 + [max(self.profile.base_yaw_speed, 0.5)] * 3
        )
        hand_columns = h[..., self.fields["hand_pose"]][..., 3:].reshape(*h.shape[:-1], 3, 2)
        third = torch.linalg.cross(hand_columns[..., 0], hand_columns[..., 1], dim=-1)
        rotation = torch.cat([hand_columns, third[..., None]], dim=-1)
        gravity = -rotation[..., 2, :]
        history = torch.cat([selected, velocity, gravity], dim=-1).clamp(-20, 20)
        preview = obs["preview"].clone()
        preview[..., self.preview_fields["base_error"]] /= preview.new_tensor([0.1, 0.1, 0.5])
        preview[..., self.preview_fields["hand_error"]][..., :3] /= 0.02
        # All temporal samples are real observations; no hidden state survives a call.
        motion = obs["motion"].clone()
        motion[..., :2] = torch.stack(
            [co[:, -1] * motion[:, 0] + si[:, -1] * motion[:, 1], -si[:, -1] * motion[:, 0] + co[:, -1] * motion[:, 1]],
            dim=-1,
        )
        for start in (3, 6):
            motion[:, start : start + 3] = torch.einsum(
                "bij,bj->bi", rotation[:, -1].transpose(1, 2), motion[:, start : start + 3].clone()
            )
        return history, preview.clamp(-20, 20), (motion / self.motion_scale).clamp(-20, 20)


class WholeBodyActor(nn.Module):
    def __init__(self, robot="innate_mars", hidden=64, *, base_rate_feedback=True, base_navigation_feedback=None):
        super().__init__()
        if type(base_rate_feedback) is not bool:
            raise ValueError("base_rate_feedback must be a boolean")
        self.robot, self.hidden = robot, hidden
        self.base_rate_feedback = base_rate_feedback
        if base_navigation_feedback is not None and not isinstance(base_navigation_feedback, BaseNavigationFeedback):
            raise ValueError("expected an explicit BaseNavigationFeedback contract")
        self.base_navigation_feedback = base_navigation_feedback
        self.encoder = SensorEncoder(robot)
        p = self.encoder.profile
        if base_navigation_feedback is not None and not p.wheel_bodies:
            raise ValueError("base navigation feedback requires a wheeled profile")
        self.gru = nn.GRU(self.encoder.token_width, hidden, batch_first=True)
        self.context = nn.Sequential(nn.Linear(hidden + 126 + 9, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU())
        self.feedback_width = 24 + p.arm_size + 6 + 9
        self.gain = nn.Linear(128, (p.action_size - 1) * self.feedback_width)
        self.grip = nn.Linear(128, 1)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        nn.init.zeros_(self.grip.weight)
        nn.init.zeros_(self.grip.bias)

    def action_mask(self, obs):
        h, f = obs["history"][:, -1], self.encoder.fields
        hand = h[:, f["hand_mask"]] > 0.5
        base = (h[:, f["base_mask"]] > 0.5).any(dim=-1, keepdim=True) | hand
        retain = h[:, f["retain_object_command"]] > 0.5
        phase = h[:, f["phase"]][:, [3, 4, 6]].any(dim=-1, keepdim=True)
        blocks = [base.expand(-1, 2)] if self.encoder.profile.wheel_bodies else []
        return torch.cat([*blocks, hand.expand(-1, self.encoder.profile.arm_size), retain & phase], dim=-1).float()

    def forward(self, obs):
        history, preview, motion = self.encoder(obs)
        context = self.gru(history)[0][:, -1]
        context = self.context(torch.cat([context, preview.flatten(1), motion], dim=-1))
        f, pf = self.encoder.fields, self.encoder.preview_fields
        errors = []
        for index in (1, 4):
            row = preview[:, index]
            hand = row[:, pf["hand_error"]].clone()
            hand[:, 3:] -= hand.new_tensor([1, 0, 0, 1, 0, 0]) * row[:, pf["hand_mask"]]
            errors += [row[:, pf["base_error"]], hand]
        velocities = obs["history"][:, -1, f["proprio/velocities"]] / history.new_tensor(self.encoder.profile.speed)
        base_velocity = history[:, -1, -9:-3]
        feedback = torch.cat([*errors, velocities, base_velocity, motion], dim=-1)
        gain = self.gain(context).reshape(-1, self.encoder.profile.action_size - 1, self.feedback_width)
        if self.base_navigation_feedback is not None:
            current = obs["history"][:, -1]
            eligible = (current[:, f["hand_mask"]].squeeze(-1) < 0.5) & (current[:, f["base_mask"]] > 0.5).all(-1)
            gain = navigation_feedback_gain(gain, preview, motion, eligible, pf, self.base_navigation_feedback)
        if not self.base_rate_feedback and self.encoder.profile.wheel_bodies:
            # The native base servo already closes its velocity loop. This
            # opt-in ablation removes only the base head's instantaneous linear
            # rate path; history, announced twist and hand-driven base authority
            # remain. Arm rows retain measured rates for reactive compensation.
            gain = gain.clone()
            gain[:, :2, 24 : 24 + self.encoder.profile.arm_size + 6] = 0
        mean = torch.cat(
            [
                torch.tanh(torch.einsum("bae,be->ba", gain, feedback) / math.sqrt(self.feedback_width)),
                torch.tanh(self.grip(context)),
            ],
            dim=-1,
        )
        if self.base_navigation_feedback is not None and self.base_navigation_feedback.reference_tracking:
            # Direct unit conversion of the announced reference twist. This is
            # base-only: coupled hand control retains its original learned map.
            # There is no teacher action or temporal output filter here; native
            # wheel speed/slew/stop guards still own actuator commands.
            base = (mean[:, :2] + motion[:, [0, 2]]).clamp(-1, 1)
            mean = torch.cat([torch.where(eligible[:, None], base, mean[:, :2]), mean[:, 2:]], dim=-1)
        return mean * self.action_mask(obs)


class MultiCriticPolicy(nn.Module):
    def __init__(
        self,
        robot="innate_mars",
        hidden=64,
        privileged=True,
        stationary_base_std_scale=1.0,
        *,
        base_rate_feedback=True,
        base_navigation_feedback=None,
        quiet_reference_std_scale=1.0,
    ):
        super().__init__()
        if isinstance(stationary_base_std_scale, bool) or not 0 < stationary_base_std_scale <= 1:
            raise ValueError("stationary base exploration scale must be finite and in (0,1]")
        self.stationary_base_std_scale = float(stationary_base_std_scale)
        if isinstance(quiet_reference_std_scale, bool) or not 0 < quiet_reference_std_scale <= 1:
            raise ValueError("quiet reference exploration scale must be finite and in (0,1]")
        self.quiet_reference_std_scale = float(quiet_reference_std_scale)
        self.actor = WholeBodyActor(
            robot, hidden, base_rate_feedback=base_rate_feedback, base_navigation_feedback=base_navigation_feedback
        )
        self.critic_encoder = SensorEncoder(robot)
        self.privileged = bool(privileged)
        width = self.critic_encoder.token_width + 126 + 9 + PRIVILEGED_WIDTH
        self.critics = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(width, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))
                for _ in range(3)
            ]
        )
        self.log_std = nn.Parameter(torch.full((PROFILES[robot].action_size,), math.log(0.1)))

    def values(self, obs):
        h, preview, motion = self.critic_encoder(obs)
        privileged = obs["critic"] if self.privileged else torch.zeros_like(obs["critic"])
        # Critic gradients never update the actor's history encoder or action head.
        features = torch.cat([h[:, -1], preview.flatten(1), motion, privileged.clamp(-100, 100)], dim=-1)
        return torch.cat([critic(features) for critic in self.critics], dim=-1)

    def distribution(self, obs):
        mean = self.actor(obs)
        std = self.log_std.clamp(-6, 0).exp()
        factors = torch.ones_like(mean)
        if self.stationary_base_std_scale != 1.0 and self.actor.encoder.profile.wheel_bodies:
            # Retain this profile guard even though the condition has one too:
            # the condition is also queried directly for training telemetry.
            # Training distribution only: retain mean/base authority, even when
            # a stationary announced reference still requires base correction.
            factor = torch.where(self.base_exploration_condition(obs), self.stationary_base_std_scale, 1.0)
            # Profile.action_size and actor.action_mask put forward/yaw first,
            # followed by arm joints and grip, whenever wheel_bodies is set.
            factors[:, :2] = factor[:, None]
        if self.quiet_reference_std_scale != 1.0:
            factor = torch.where(self.quiet_reference_condition(obs), self.quiet_reference_std_scale, 1.0)
            # Train motion corrections with less noise during announced holds.
            # Means, gripper exploration and actuator authority are unchanged.
            # Overlapping noise options use the smaller scale, not their product.
            factors[:, :-1] = torch.minimum(factors[:, :-1], factor[:, None])
        if self.stationary_base_std_scale != 1.0 or self.quiet_reference_std_scale != 1.0:
            std = std * factors
        return torch.distributions.Normal(mean, std)

    def quiet_reference_condition(self, obs):
        """Constant current/future motion targets, using sensor-visible inputs.

        Preview row zero is past reference; rows one onward include the present.
        Exact zero announced twist and constant targets/masks identify a hold.
        Measured error/velocity never disable correction. A free base may still
        assist an active hand. Gripper intent does not gate motion exploration.
        """
        fields, pf = self.actor.encoder.fields, self.actor.encoder.preview_fields
        history, preview = obs["history"][:, -1], obs["preview"][:, 1:]
        constant = torch.ones(len(history), dtype=torch.bool, device=history.device)
        for name in ("base_error", "base_mask", "hand_error", "hand_mask"):
            values = preview[:, :, pf[name]]
            constant &= (values == values[:, :1]).all(dim=(1, 2))
        for name in ("base_mask", "hand_mask"):
            constant &= (history[:, fields[name]] == preview[:, 0, pf[name]]).all(dim=-1)
        active = (history[:, fields["base_mask"]] > 0.5).any(dim=-1)
        active |= (history[:, fields["hand_mask"]] > 0.5).any(dim=-1)
        return active & constant & (obs["motion"] == 0).all(dim=-1)

    def base_exploration_condition(self, obs):
        """An explicitly constrained, constant base preview with an active hand.

        Zero announced velocity alone does not imply a held base goal. A free
        base and an upcoming moving reference retain their exploration. Actual
        base motion/error is deliberately not used to mask correction authority.
        """
        fields, preview_fields = self.actor.encoder.fields, self.actor.encoder.preview_fields
        history, preview = obs["history"][:, -1], obs["preview"][:, 1:]
        hand = history[:, fields["hand_mask"]].squeeze(-1) > 0.5
        constrained = (history[:, fields["base_mask"]] > 0.5).all(dim=-1)
        constrained &= (preview[:, :, preview_fields["base_mask"]] > 0.5).all(dim=(1, 2))
        targets = preview[:, :, preview_fields["base_error"]]
        # The producer reuses one measured pose across preview rows and emits
        # exact zero twist for holds. Future producer jitter deliberately keeps
        # full exploration; do not silently widen this into an approximate hold.
        constant = (targets == targets[:, :1]).all(dim=(1, 2))
        stationary = (obs["motion"][:, :3] == 0).all(dim=-1)
        return hand & constrained & constant & stationary & bool(self.actor.encoder.profile.wheel_bodies)

    def evaluate(self, obs, actions):
        dist = self.distribution(obs)
        mask = self.actor.action_mask(obs)
        return (dist.log_prob(actions) * mask).sum(-1), (dist.entropy() * mask).sum(-1), self.values(obs)
