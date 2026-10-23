"""Value-only warm-up on policy trajectories before PPO actor updates."""

import numpy as np
import torch


def warmup_critic(model, env, steps, *, epochs=20, seed=0, run=None):
    rng = np.random.default_rng(seed)
    pending = [[] for _ in range(env.num_envs)]
    histories, returns = [], []
    obs = env.reset()
    for _ in range(max(1, steps // env.num_envs)):
        raw = env.get_original_obs()["history"].copy()
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(action)
        for index in range(env.num_envs):
            pending[index].append((raw[index], float(rewards[index])))
            if dones[index]:
                value = 0.0
                for history, reward in reversed(pending[index]):
                    value = reward + model.gamma * value
                    histories.append(history)
                    returns.append(value)
                pending[index].clear()
    if not histories:
        raise ValueError("critic warm-up requires enough steps for complete episodes")
    x = torch.as_tensor(env.normalize_obs({"history": np.stack(histories)})["history"], dtype=torch.float32)
    y = torch.as_tensor(returns, dtype=torch.float32)[:, None]
    parameters = list(model.policy.mlp_extractor.value_net.parameters()) + list(model.policy.value_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(len(x))
        for offset in range(0, len(x), 512):
            ids = order[offset : offset + 512]
            value = model.policy.predict_values({"history": x[ids]})
            loss = (value - y[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = {
            "critic_warmup/epoch": epoch + 1,
            "critic_warmup/mse": float(np.mean(losses)),
            "critic_warmup/examples": len(x),
        }
        if run is not None:
            run.log(metrics)
    print(f"Critic warm-up: {metrics}", flush=True)
    # Reset rollout state after collecting separate warm-up trajectories.
    model._last_obs = None
    return metrics
