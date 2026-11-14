"""PPO with independent return/advantage estimates and simulator-only value inputs."""

from dataclasses import dataclass

import numpy as np
import torch


def discount_tick_rewards(rewards, *, gamma):
    """Return sum(gamma**j * r[j]) for one variable-duration decision.

    Rewards are earned after each physical control tick. Continuation starts
    after all ticks and is discounted separately by gamma**ticks in GAE.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if (
        rewards.ndim != 2
        or rewards.shape[1] != 3
        or len(rewards) == 0
        or not np.isfinite(rewards).all()
        or not 0 < gamma <= 1
    ):
        raise ValueError("finite per-tick three-group rewards and gamma in (0,1] required")
    if len(rewards) == 1:
        # Preserve existing one-tick values bitwise, including signed zeros.
        return rewards[0].astype(np.float32)
    discounts = gamma ** np.arange(len(rewards), dtype=np.float64)
    return np.sum(discounts[:, None] * rewards, axis=0).astype(np.float32)


def grouped_gae(rewards, values, next_values, terminated, episode_end, ticks, *, gamma=0.9975, lam=0.95):
    """GAE over decisions whose rewards are already discounted within ticks."""
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


def combine_advantages(advantages, weights=(1.0, 1.0, 1.0), *, normalization="shared"):
    """Preserve reward coefficients by whitening the weighted sum once.

    GAE is linear across heads with shared clocks and termination. Thus the
    raw weighted sum represents the specified total reward. Whitening each
    head first instead changes that objective and removes positive within-head
    reward scales. Keep that historical mode explicit for paired experiments.
    The second return contains the weighted contribution from each head.
    """
    a = torch.as_tensor(advantages)
    weights = torch.as_tensor(weights, dtype=a.dtype, device=a.device)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2 or weights.shape != (3,) or not torch.isfinite(a).all():
        raise ValueError("at least two finite three-group advantages required")
    if not torch.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError("nonnegative finite advantage weights required")
    if normalization not in {"shared", "per_head"}:
        raise ValueError("advantage normalization must be shared or per_head")
    centered = a - a.mean(0)
    if normalization == "per_head":
        components = centered / a.std(0, unbiased=False).clamp_min(1e-6) * weights
    else:
        combined = (a * weights).sum(-1)
        components = centered * weights / combined.std(unbiased=False).clamp_min(1e-6)
    return components.sum(-1), components


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
    advantage_normalization: str = "shared"

    def __post_init__(self):
        vals = [self.learning_rate, self.critic_learning_rate, self.clip, self.max_grad_norm, self.target_kl]
        if any(not np.isfinite(x) or x <= 0 for x in vals) or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid PPO rates/discounts")
        if type(self.epochs) is not int or self.epochs <= 0 or type(self.batch_size) is not int or self.batch_size < 2:
            raise ValueError("positive PPO epochs and batch size at least two required")
        if not np.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError("nonnegative entropy weight required")
        combine_advantages(torch.zeros((2, 3)), self.advantage_weights, normalization=self.advantage_normalization)


class MultiCriticPPO:
    def __init__(self, policy, config=PPOConfig()):
        self.policy, self.config = policy, config
        self.optimizer = torch.optim.Adam(
            [
                {"params": [*policy.actor.parameters(), policy.log_std], "lr": config.learning_rate},
                {"params": policy.critics.parameters(), "lr": config.critic_learning_rate},
            ]
        )

    def actor_gradient_diagnostics(self, ratio, components):
        """First minibatch: unclipped actor gradients per head, excluding entropy.

        Includes exploration log_std. This observes derivatives without writing
        parameter.grad or changing the subsequent PPO clipping/update.
        """
        parameters = [p for p in [*self.policy.actor.parameters(), self.policy.log_std] if p.requires_grad]
        losses = -(ratio[:, None] * components).mean(0)
        gradients = []
        for index in range(3):
            derivative = torch.autograd.grad(losses[index], parameters, retain_graph=True, allow_unused=True)
            gradients.append(
                torch.cat(
                    [
                        torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                        for p, g in zip(parameters, derivative, strict=True)
                    ]
                )
            )
        matrix = torch.stack(gradients)
        norms = matrix.norm(dim=1)
        cosines = (matrix @ matrix.T) / (norms[:, None] * norms[None, :]).clamp_min(1e-12)
        return {
            "actor_gradient_norm_by_head": norms.tolist(),
            "actor_gradient_cosines": cosines.tolist(),
            "gradient_diagnostics": "first_minibatch_unclipped_no_entropy",
        }

    def update(self, rollout):
        cfg = self.config
        if rollout.get("reward_discount", cfg.gamma) != cfg.gamma:
            raise ValueError("rollout tick reward discount must match PPO gamma")
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
        advantage, normalized = combine_advantages(
            torch.from_numpy(advantages), cfg.advantage_weights, normalization=cfg.advantage_normalization
        )
        targets = torch.from_numpy(targets)
        logs, stop, gradient_stats, grad_norms = [], False, {}, []
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
                clipped = ((ratio - 1).abs() > cfg.clip).float().mean().item()
                logs.append([actor_loss.item(), *value_losses.detach().tolist(), kl, clipped])
                if kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
                if not gradient_stats:
                    gradient_stats = self.actor_gradient_diagnostics(ratio, normalized[indices])
                self.optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [*self.policy.actor.parameters(), self.policy.log_std], cfg.max_grad_norm
                )
                grad_norms.append(float(norm))
                for critic in self.policy.critics:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
            if stop:
                break
        avg = np.mean(logs, axis=0)
        rewards = np.asarray(rollout["rewards"])
        raw_values = np.asarray(rollout["values"])
        discount = cfg.gamma ** np.asarray(rollout["ticks"], dtype=float)
        td = (
            rewards
            + discount[:, None] * (~np.asarray(rollout["terminated"], dtype=bool))[:, None] * rollout["next_values"]
            - raw_values
        )
        return {
            "actor_loss": float(avg[0]),
            "critic_losses": avg[1:4].tolist(),
            "approx_kl": float(avg[4]),
            "clip_fraction": float(avg[5]),
            "actor_gradient_norm_pre_clip": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "optimizer_steps": len(grad_norms),
            "raw_reward_mean": rewards.mean(0).tolist(),
            "raw_reward_std": rewards.std(0).tolist(),
            "raw_value_mean": raw_values.mean(0).tolist(),
            "raw_value_std": raw_values.std(0).tolist(),
            "raw_td_mean": td.mean(0).tolist(),
            "raw_td_std": td.std(0).tolist(),
            "advantage_mean": np.mean(advantages, axis=0).tolist(),
            "advantage_std": np.std(advantages, axis=0).tolist(),
            "zero_reward_heads": np.all(rewards == 0, axis=0).tolist(),
            "combined_advantage_mean": float(advantage.mean()),
            "combined_advantage_std": float(advantage.std(unbiased=False)),
            "head_contribution_std": normalized.std(0, unbiased=False).tolist(),
            "advantage_correlation": torch.nan_to_num(torch.corrcoef(normalized.T)).tolist(),
            "advantage_normalization": cfg.advantage_normalization,
            "combined_tracking_correlation": float(
                torch.nan_to_num(torch.corrcoef(torch.stack([advantage, normalized[:, 0]]))[0, 1])
            ),
            **gradient_stats,
        }
