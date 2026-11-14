"""PPO with independent return/advantage estimates and simulator-only value inputs."""

from dataclasses import dataclass

import numpy as np
import torch


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


def combine_advantages(advantages, weights=(1.0, 1.0, 1.0)):
    a = torch.as_tensor(advantages)
    weights = torch.as_tensor(weights, dtype=a.dtype, device=a.device)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2 or weights.shape != (3,) or not torch.isfinite(a).all():
        raise ValueError("at least two finite three-group advantages required")
    if not torch.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError("nonnegative finite advantage weights required")
    normalized = (a - a.mean(0)) / a.std(0, unbiased=False).clamp_min(1e-6)
    return (normalized * weights).sum(-1), normalized


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.9975
    gae_lambda: float = 0.95
    clip: float = 0.1
    epochs: int = 5
    batch_size: int = 128
    entropy_weight: float = 0.001
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    advantage_weights: tuple = (1.0, 1.0, 1.0)

    def __post_init__(self):
        vals = [self.learning_rate, self.critic_learning_rate, self.clip, self.max_grad_norm, self.target_kl]
        if any(not np.isfinite(x) or x <= 0 for x in vals) or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid PPO rates/discounts")
        if type(self.epochs) is not int or self.epochs <= 0 or type(self.batch_size) is not int or self.batch_size < 2:
            raise ValueError("positive PPO epochs and batch size at least two required")
        if not np.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError("nonnegative entropy weight required")
        combine_advantages(torch.zeros((2, 3)), self.advantage_weights)


class MultiCriticPPO:
    def __init__(self, policy, config=PPOConfig()):
        self.policy, self.config = policy, config
        self.optimizer = torch.optim.Adam(
            [
                {"params": [*policy.actor.parameters(), policy.log_std], "lr": config.learning_rate},
                {"params": policy.critics.parameters(), "lr": config.critic_learning_rate},
            ]
        )

    def update(self, rollout):
        cfg = self.config
        obs = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in rollout["observations"].items()}
        actions = torch.as_tensor(rollout["actions"], dtype=torch.float32)
        old_log = torch.as_tensor(rollout["log_prob"], dtype=torch.float32)
        advantages, targets = grouped_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["next_values"],
            rollout["terminated"],
            rollout["episode_end"],
            rollout["ticks"],
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )
        advantage, normalized = combine_advantages(torch.from_numpy(advantages), cfg.advantage_weights)
        targets = torch.from_numpy(targets)
        logs, stop = [], False
        for _ in range(cfg.epochs):
            for indices in torch.randperm(len(actions)).split(cfg.batch_size):
                if len(indices) < 2:
                    continue
                batch = {k: v[indices] for k, v in obs.items()}
                log, entropy, values = self.policy.evaluate(batch, actions[indices])
                ratio = (log - old_log[indices]).exp()
                surrogate = torch.minimum(
                    ratio * advantage[indices], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage[indices]
                )
                actor_loss = -surrogate.mean() - cfg.entropy_weight * entropy.mean()
                value_losses = ((values - targets[indices]) ** 2).mean(0)
                loss = actor_loss + 0.5 * value_losses.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite multi-critic PPO loss")
                with torch.no_grad():
                    kl = ((ratio - 1) - (log - old_log[indices])).mean().item()
                logs.append([actor_loss.item(), *value_losses.detach().tolist(), kl])
                if kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.policy.actor.parameters(), self.policy.log_std], cfg.max_grad_norm
                )
                for critic in self.policy.critics:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
            if stop:
                break
        avg = np.mean(logs, axis=0)
        return {
            "actor_loss": float(avg[0]),
            "critic_losses": avg[1:4].tolist(),
            "approx_kl": float(avg[4]),
            "advantage_std": np.std(advantages, axis=0).tolist(),
            "advantage_correlation": torch.nan_to_num(torch.corrcoef(normalized.T)).tolist(),
        }
