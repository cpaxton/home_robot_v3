"""Training wrappers; task selection never runs inside the deployed actor."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.whole_body_observation import (
    PRIVILEGED_WIDTH,
    actor_observation,
    privileged_state,
    reference_velocity,
)
from emet.rl.manipulation.whole_body_reward import RewardMeter


class GeneralTrackingEnv(gym.Wrapper):
    """Keep the native goals, reset admission, horizon and hard stop guards."""

    def __init__(self, env):
        super().__init__(env)
        p = env.unwrapped.profile
        if not env.unwrapped.config.retention_context or not env.unwrapped.config.trajectory_preview:
            raise ValueError("whole-body training requires history retention context and trajectory preview")
        _, width = history_layout(p, retention_context=True)
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(-np.inf, np.inf, (p.history_steps, width), np.float32),
                "preview": spaces.Box(-np.inf, np.inf, (7, 18), np.float32),
                "motion": spaces.Box(-np.inf, np.inf, (9,), np.float32),
                "critic": spaces.Box(-np.inf, np.inf, (PRIVILEGED_WIDTH,), np.float32),
            }
        )
        self.meter = None

    def observation(self, raw):
        e = self.env.unwrapped
        return actor_observation(e.control, raw) | {
            "critic": privileged_state(e, e.steps * e.profile.dt, e.config.horizon)
        }

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        if info.get("initialization_failed"):
            raise ValueError("general development reset rejected: " + str(info.get("initialization_failure_reason")))
        self.meter = RewardMeter(self.env.unwrapped)
        return self.observation(raw), info

    def step(self, action):
        from emet.rl.manipulation.mastery_objective import mastery_metrics, tracking_contract_passes

        e = self.env.unwrapped
        motion = reference_velocity(e.control)
        raw, _, done, truncated, info = self.env.step(action)
        rewards = self.meter.tick(action, motion)
        success = False
        if done or truncated:
            success = bool(
                tracking_contract_passes(info, mastery_metrics(e, info), e.suite["gate"], path_failed=e.path_failed)
            )
            rewards[0] += 2.0 if success else -2.0
        info = {
            **info,
            "reward_groups": rewards,
            "reward_terms": self.meter.last_terms.copy(),
            "elapsed_control_ticks": 1,
            "primitive_success": success,
            "full_pick_place_success": False,
        }
        return self.observation(raw), float(rewards.sum()), done, truncated, info


class ControlStepMixture(gym.Env):
    """Deficit scheduling by actual simulated control ticks, at episode boundaries.

    Episodes remain intact. A long TAMP episode is followed by enough general
    ticks to recover the requested proportion; equal episode counts would not.
    The counters are included in every info so imbalance remains visible.
    """

    def __init__(self, environments, weights=None):
        if not environments:
            raise ValueError("nonempty environment mapping required")
        self.environments = dict(environments)
        self.names = tuple(self.environments)
        w = np.array([1.0] * len(self.names) if weights is None else weights, dtype=float)
        if w.shape != (len(self.names),) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("one positive finite mixture weight per environment required")
        self.weights = w / w.sum()
        self.ticks = np.zeros(len(w), dtype=np.int64)
        self.episodes = np.zeros(len(w), dtype=np.int64)
        self.active = None
        self.finished = True
        first = next(iter(self.environments.values()))
        self.action_space, self.observation_space = first.action_space, first.observation_space
        if any(
            e.action_space != self.action_space or e.observation_space != self.observation_space
            for e in self.environments.values()
        ):
            raise ValueError("mixture environments must have identical observation/action contracts")

    def ledger(self):
        return {
            name: {
                "control_ticks": int(self.ticks[i]),
                "episodes": int(self.episodes[i]),
                "requested_fraction": float(self.weights[i]),
            }
            for i, name in enumerate(self.names)
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self.finished:
            raise RuntimeError("finish the current primitive before selecting a new mixture episode")
        self.active = int(np.argmin(self.ticks / self.weights))
        obs, info = self.environments[self.names[self.active]].reset(seed=seed, options=options)
        self.finished = False
        self.episodes[self.active] += 1
        return obs, {**info, "training_family": self.names[self.active], "mixture": self.ledger()}

    def step(self, action):
        if self.finished or self.active is None:
            raise RuntimeError("reset mixture before stepping")
        obs, reward, terminated, truncated, info = self.environments[self.names[self.active]].step(action)
        ticks = info.get("elapsed_control_ticks", 1)
        if isinstance(ticks, bool) or int(ticks) != ticks or ticks <= 0:
            raise ValueError("positive integer physical control ticks required")
        self.ticks[self.active] += int(ticks)
        self.finished = terminated or truncated
        return (
            obs,
            reward,
            terminated,
            truncated,
            {**info, "training_family": self.names[self.active], "mixture": self.ledger()},
        )

    def close(self):
        for env in self.environments.values():
            env.close()
