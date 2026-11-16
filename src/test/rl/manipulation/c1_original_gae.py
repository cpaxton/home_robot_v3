# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Frozen executable GAE from e0d8df02 for the source-recovery boundary test."""

import numpy as np


def grouped_gae(rewards, values, next_values, terminated, episode_end, ticks, *, gamma=0.9975, lam=0.95):
    arrays = [np.asarray(x) for x in (rewards, values, next_values)]
    rewards, values, next_values = arrays
    n = len(rewards)
    if (
        any(x.shape != (n, 3) or not np.isfinite(x).all() for x in arrays)
        or n < 2
        or np.asarray(ticks).shape != (n,)
        or not np.isfinite(ticks).all()
        or np.any(np.asarray(ticks) <= 0)
        or np.any(np.asarray(ticks) != np.floor(ticks))
        or np.asarray(terminated).shape != (n,)
        or np.asarray(episode_end).shape != (n,)
        or not np.isin(terminated, [False, True]).all()
        or not np.isin(episode_end, [False, True]).all()
        or np.any(np.asarray(terminated, dtype=bool) & ~np.asarray(episode_end, dtype=bool))
        or not 0 < gamma <= 1
        or not 0 <= lam <= 1
    ):
        raise ValueError("finite grouped rewards/values and positive elapsed control ticks required")
    advantage = np.zeros_like(rewards, dtype=np.float64)
    following = np.zeros(3)
    for t in reversed(range(n)):
        discount = gamma ** float(ticks[t])
        delta = rewards[t] + discount * (not terminated[t]) * next_values[t] - values[t]
        following = delta + discount * lam ** float(ticks[t]) * (not episode_end[t]) * following
        advantage[t] = following
    return advantage.astype(np.float32), (advantage + values).astype(np.float32)
