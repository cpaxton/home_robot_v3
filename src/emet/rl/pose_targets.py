"""Masked variable-size world-frame pose targets for coordinated robot control.

A target batch has one optional planar base pose and zero or more optional hand
poses. It is a data contract shared by environments, policy adapters and test
drive; it does not solve IK or grant the policy permission to move masked parts.
"""

from dataclasses import dataclass

import numpy as np

TARGET_TOKEN_WIDTH = 10
BASE_MASK_COLUMNS = slice(7, 10)
HAND_MASK_COLUMN = 9


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


@dataclass(frozen=True)
class PoseTargets:
    """One SE(2) base target and a padded set of SE(3) hand targets.

    ``base_mask`` is component-wise [x, y, yaw]. It specifies target
    constraints, not actuator availability. The controller contract determines
    whether an unconstrained base may move to meet an active hand target. Hand poses use position plus the first two
    columns of a rotation matrix (9 values), with one boolean mask per hand.
    False hand entries are padding/disabled targets and must be ignored by the
    actor and controller.
    """

    base: np.ndarray
    base_mask: np.ndarray
    hands: np.ndarray
    hand_mask: np.ndarray

    def __post_init__(self):
        base = _finite_array(self.base, (3,), "base target")
        base_mask = np.asarray(self.base_mask)
        hands = np.asarray(self.hands, dtype=np.float32)
        hand_mask = np.asarray(self.hand_mask)
        if base_mask.shape != (3,) or base_mask.dtype != np.bool_:
            raise ValueError("base_mask must be a boolean vector of shape (3,)")
        if hands.ndim != 2 or hands.shape[1:] != (9,) or not np.isfinite(hands).all():
            raise ValueError("hand targets must be a finite (N, 9) array")
        if hand_mask.shape != (len(hands),) or hand_mask.dtype != np.bool_:
            raise ValueError("hand_mask must be a boolean vector matching hand targets")
        if not base_mask.any() and not hand_mask.any():
            raise ValueError("at least one target must be active")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "base_mask", base_mask.copy())
        object.__setattr__(self, "hands", hands.copy())
        object.__setattr__(self, "hand_mask", hand_mask.copy())

    @classmethod
    def from_values(cls, *, base, base_mask, hands, hand_mask):
        """Construct targets from JSON-like sequences with strict mask typing."""
        return cls(base, np.asarray(base_mask, dtype=bool), hands, np.asarray(hand_mask, dtype=bool))

    def actor_tokens(self, *, max_hands=None):
        """Return padded (N+1, 10) target tokens and a boolean token mask.

        The base token is [x, y, sin(yaw), cos(yaw), 6 zeros, 3 mask bits].
        Hand tokens are [position, 6D rotation, 3 zero values, active bit].
        Target masks are repeated in the token payload for observability and
        also returned separately for attention/pooling masks.
        """
        count = len(self.hands) if max_hands is None else int(max_hands)
        if count < len(self.hands) or count < 0:
            raise ValueError("max_hands cannot be smaller than the supplied hand set")
        tokens = np.zeros((count + 1, TARGET_TOKEN_WIDTH), dtype=np.float32)
        tokens[0, :4] = [self.base[0], self.base[1], np.sin(self.base[2]), np.cos(self.base[2])]
        tokens[0, BASE_MASK_COLUMNS] = self.base_mask
        tokens[1 : len(self.hands) + 1, :9] = self.hands
        tokens[1 : len(self.hands) + 1, HAND_MASK_COLUMN] = self.hand_mask
        token_mask = np.zeros(count + 1, dtype=bool)
        token_mask[0] = bool(self.base_mask.any())
        token_mask[1 : len(self.hands) + 1] = self.hand_mask
        return tokens, token_mask
