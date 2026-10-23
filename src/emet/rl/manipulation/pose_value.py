"""Pose-conditioned value inputs and conservative candidate selection.

Candidates are world-frame end-effector SE(3) targets. The value model sees
them relative to the current end effector and the same proprioceptive history
used by the reactive controller. A candidate mask permits variable-size TAMP
sets to be padded for batched inference without scoring padding entries.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _pose(value, *, name):
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError(f"{name} must be an SE(3) transform")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-3, rtol=0
    ):
        raise ValueError(f"{name} rotation must be in SO(3)")
    return pose


def pose_candidate_features(history, current_pose, candidates, candidate_mask):
    """Encode candidate poses in the current EE frame, alongside state history.

    The six rotation coordinates are the first two columns of the relative
    rotation matrix (the continuous 6D rotation representation). Padded poses
    are deliberately not inspected and receive an all-zero feature row.
    """
    history = np.asarray(history, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float64)
    mask = np.asarray(candidate_mask)
    if history.ndim != 2 or not history.shape[0] or not history.shape[1] or not np.isfinite(history).all():
        raise ValueError("history must be a nonempty finite [time, feature] array")
    if candidates.ndim != 3 or candidates.shape[1:] != (4, 4):
        raise ValueError("candidates must have shape [count, 4, 4]")
    if mask.shape != (len(candidates),) or mask.dtype != np.bool_:
        raise ValueError("candidate_mask must be a boolean vector matching candidates")
    current = _pose(current_pose, name="current_pose")
    if not mask.any():
        raise ValueError("at least one real candidate is required")

    result = np.zeros((len(candidates), history.size + 9), dtype=np.float32)
    state = history.reshape(-1)
    inverse_rotation = current[:3, :3].T
    for index in np.flatnonzero(mask):
        target = _pose(candidates[index], name=f"candidate[{index}]")
        relative_rotation = inverse_rotation @ target[:3, :3]
        relative_position = inverse_rotation @ (target[:3, 3] - current[:3, 3])
        result[index] = np.r_[state, relative_position, relative_rotation[:, :2].T.reshape(-1)]
    return result


@dataclass(frozen=True)
class CandidateValueDecision:
    """An ensemble value decision; unknown candidates are never accepted."""

    accepted_index: int | None
    means: np.ndarray
    standard_deviations: np.ndarray
    lower_confidence_bounds: np.ndarray
    decisions: tuple[str, ...]


class PoseConditionedValue(nn.Module):
    """Shared critic for variable-size batches of robot-relative SE(3) goals.

    Train multiple independent instances as an ensemble. Each candidate is
    scored from the same encoded proprioceptive history and robot identity;
    candidate order and padding do not affect valid scores. ``relative_pose``
    stores translation followed by the 6D rotation representation.
    """

    def __init__(self, history_features, robot_count=2, hidden=128):
        super().__init__()
        if min(history_features, robot_count, hidden) < 1:
            raise ValueError("value network dimensions must be positive")
        self.history_encoder = nn.GRU(history_features, hidden, batch_first=True)
        self.robot_embedding = nn.Embedding(robot_count, 16)
        self.pose_encoder = nn.Sequential(nn.Linear(9, 48), nn.Tanh(), nn.Linear(48, 32), nn.Tanh())
        self.value_head = nn.Sequential(
            nn.Linear(hidden + 16 + 32, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, history, relative_pose, candidate_mask, robot_ids):
        if history.ndim != 3 or relative_pose.ndim != 3 or relative_pose.shape[-1] != 9:
            raise ValueError("expected history [batch,time,features] and poses [batch,candidates,9]")
        batch, count = relative_pose.shape[:2]
        if history.shape[0] != batch or candidate_mask.shape != (batch, count):
            raise ValueError("candidate batch dimensions do not agree")
        if robot_ids.shape != (batch,) or candidate_mask.dtype != torch.bool:
            raise ValueError("robot_ids and boolean candidate_mask must match the batch")
        if torch.any(~candidate_mask.any(dim=1)):
            raise ValueError("every batch row needs at least one candidate")
        state = self.history_encoder(history)[0][:, -1]
        robot = self.robot_embedding(robot_ids.long())
        # Padding may contain arbitrary sentinels; make it harmless before the
        # MLP and explicitly zero its output below.
        poses = torch.where(candidate_mask[..., None], relative_pose, torch.zeros_like(relative_pose))
        pose = self.pose_encoder(poses)
        state = state[:, None, :].expand(-1, count, -1)
        robot = robot[:, None, :].expand(-1, count, -1)
        values = self.value_head(torch.cat([state, robot, pose], dim=-1)).squeeze(-1)
        return values.masked_fill(~candidate_mask, 0.0)


def decide_pose_candidates(
    member_values,
    candidate_mask,
    *,
    minimum_value,
    maximum_disagreement,
    confidence_multiplier=2.0,
):
    """Select by a conservative ensemble lower bound, abstaining on uncertainty.

    ``member_values`` contains one expected-return estimate per ensemble member
    and candidate. A value below ``minimum_value`` rejects a candidate. High
    ensemble disagreement, nonfinite values, and padded rows are unknown and
    cannot pass the gate. Thresholds must be calibrated on held-out episodes.
    """
    values = np.asarray(member_values, dtype=np.float64)
    mask = np.asarray(candidate_mask)
    if values.ndim != 2 or values.shape[1] == 0 or values.shape[0] < 2:
        raise ValueError("member_values must have shape [at least two members, candidates]")
    if mask.shape != (values.shape[1],) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("candidate_mask must select at least one candidate")
    if (
        not np.isfinite([minimum_value, maximum_disagreement, confidence_multiplier]).all()
        or maximum_disagreement < 0
        or confidence_multiplier < 0
    ):
        raise ValueError("invalid value-selection thresholds")

    known = mask & np.isfinite(values).all(axis=0)
    means = np.full(values.shape[1], np.nan)
    deviations = np.full(values.shape[1], np.nan)
    bounds = np.full(values.shape[1], np.nan)
    means[known] = values[:, known].mean(axis=0)
    deviations[known] = values[:, known].std(axis=0)
    bounds[known] = means[known] - confidence_multiplier * deviations[known]
    decisions = []
    accepted = []
    for index in range(values.shape[1]):
        if not mask[index] or not known[index]:
            decisions.append("unknown")
        elif deviations[index] > maximum_disagreement:
            decisions.append("unknown")
        elif bounds[index] < minimum_value:
            decisions.append("rejected_low_value")
        else:
            decisions.append("accepted")
            accepted.append(index)
    selected = max(accepted, key=lambda index: bounds[index]) if accepted else None
    return CandidateValueDecision(selected, means, deviations, bounds, tuple(decisions))
