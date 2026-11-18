"""Experimental motion heads for a matched, frozen-backbone diagnostic.

These are in-memory simulator experiments. The direct head has no deployment
bundle format and must not be substituted into an existing actor-only bundle.
"""

import copy
import math
from numbers import Real

import torch
from torch import nn

from emet.rl.manipulation.whole_body_actor import WholeBodyActor


class DirectMotionActor(WholeBodyActor):
    """Same measured history/context/masks; bounded direct motion outputs."""

    def __init__(self, source):
        super().__init__(source.robot, source.hidden)
        self.encoder = copy.deepcopy(source.encoder)
        self.gru = copy.deepcopy(source.gru)
        self.context = copy.deepcopy(source.context)
        self.grip = copy.deepcopy(source.grip)
        del self.gain
        self.motion_head = nn.Sequential(
            nn.Linear(128, 256), nn.ELU(), nn.Linear(256, self.encoder.profile.action_size - 1)
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

    def forward(self, obs):
        history, preview, motion = self.encoder(obs)
        context = self.gru(history)[0][:, -1]
        context = self.context(torch.cat([context, preview.flatten(1), motion], dim=-1))
        action = torch.cat([torch.tanh(self.motion_head(context)), torch.tanh(self.grip(context))], dim=-1)
        return action * self.action_mask(obs)


def matched_heads(source, *, seed=20261118, backbone_frozen=True):
    """Clone one backbone and zero both motion means with explicit fitting mode.

    Disable base-only gain priors in this comparison: a direct head does not
    possess them. The retained, unchanged source remains a separate baseline.
    Head parameter counts differ and are reported rather than called equal.
    """
    if type(backbone_frozen) is not bool:
        raise ValueError("explicit boolean backbone fitting mode required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gain = copy.deepcopy(source)
        gain.base_navigation_feedback = None
        gain.base_rate_feedback = True
        nn.init.zeros_(gain.gain.weight)
        nn.init.zeros_(gain.gain.bias)
        direct = DirectMotionActor(gain)
    for actor in (gain, direct):
        actor.requires_grad_(not backbone_frozen)
        actor.grip.requires_grad_(False)
        (
            actor.gain
            if isinstance(actor, WholeBodyActor) and not isinstance(actor, DirectMotionActor)
            else actor.motion_head
        ).requires_grad_(True)
    return {"gain": gain, "direct": direct}


def recovery_reference(env, *, seed, offset_rad=0.08, acquisition_s=1.5):
    """At reset, announce a nearby admitted goal with an explicit capture hold.

    Only a scratch FK view is offset. The measured plant is never teleported or
    pushed, and the original evaluation cases/precision gates stay unchanged.
    """
    from dataclasses import replace

    import numpy as np

    from emet.rl.manipulation.arm_splines import ArmSplineReference, SplineLevel
    from emet.rl.manipulation.coupled_tracking import admit_reference
    from emet.rl.manipulation.tracking import TaskFrameFK

    if (
        isinstance(offset_rad, (bool, np.bool_))
        or isinstance(acquisition_s, (bool, np.bool_))
        or not 0 < offset_rad <= 0.1
        or not 0.5 <= acquisition_s <= 2
        or env.unwrapped.steps != 0
    ):
        raise ValueError("bounded reset-only acquisition reference required")
    e, old = env.unwrapped, env.reference
    rng = np.random.default_rng(seed)
    original_level = SplineLevel(**old.certificate["level"])
    # Acquisition is an explicit extra hold, not a larger motion-time budget.
    level = replace(
        original_level,
        initial_hold_s=acquisition_s,
        maximum_duration_s=original_level.maximum_duration_s + acquisition_s - original_level.initial_hold_s,
    )
    low = np.maximum(-offset_rad, e.control.limits[:, 0] + 0.01 - np.asarray(old.certificate["joint_minimum"]))
    high = np.minimum(offset_rad, e.control.limits[:, 1] - 0.01 - np.asarray(old.certificate["joint_maximum"]))
    rejections = []
    for _ in range(16):
        offset = rng.uniform(low, high)
        points = np.asarray(old.certificate["joint_waypoints"]) + offset
        scratch = TaskFrameFK(e.model, e.data, e.control)
        scratch(e.control.base_pose(), points[0])
        view = copy.copy(e.control)
        view.data = scratch.data
        try:
            reference = ArmSplineReference(view, points, level, timing="bounded_v2")
            admission = admit_reference(e, reference)
            if not admission["accepted"]:
                raise ValueError(str(admission["reason"]))
            break
        except ValueError as exc:
            rejections.append({"joint_offset_rad": offset.tolist(), "reason": str(exc)})
    else:
        raise ValueError("bounded recovery reference sampling exhausted: " + str(rejections))
    initial_error = float(np.linalg.norm(reference.hand[0, :3] - e.control.ee_pose()[:3]))
    e.set_task_trajectory(reference)
    env.reference = reference
    env.admission = admission
    e.config = replace(e.config, horizon=float(np.ceil((reference.end + 3) / e.profile.dt) * e.profile.dt))
    return env.observe(e.control.observe()), {
        "joint_offset_rad": offset.tolist(),
        "initial_hand_error_m": initial_error,
        "acquisition_seconds": acquisition_s,
        "admission": admission,
        "rejected_proposals": rejections,
    }


def fit_head(actor, observations, targets, *, updates=1000, batch_size=128, seed=20261118, learning_rate=3e-3):
    """Teacher-action fitting isolates the head; this is not a PPO result."""
    if type(updates) is not int or updates <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("positive integer diagnostic fitting budget required")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.01
    ):
        raise ValueError("bounded positive diagnostic learning rate required")
    if set(observations) != {"history", "preview", "motion"}:
        raise ValueError("sensor-only fitting observations required")
    tensors = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in observations.items()}
    targets = torch.as_tensor(targets, dtype=torch.float32)
    count = len(targets)
    if (
        targets.shape != (count, actor.encoder.profile.action_size)
        or not count
        or any(len(v) != count or not torch.isfinite(v).all() for v in tensors.values())
        or not torch.isfinite(targets).all()
        or torch.any(targets.abs() > 1)
    ):
        raise ValueError("finite bounded teacher actions and aligned sensor observations required")
    optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(updates):
        indices = torch.randint(count, (min(batch_size, count),), generator=generator)
        batch = {k: v[indices] for k, v in tensors.items()}
        mask = actor.action_mask(batch)[:, :-1]
        loss = (((actor(batch)[:, :-1] - targets[indices, :-1]) ** 2) * mask).sum() / mask.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 100 == 0 or step + 1 == updates:
            losses.append({"update": step + 1, "action_mse": float(loss.detach())})
    return losses
