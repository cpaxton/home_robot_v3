"""Dataset aggregation initializes an independent velocity actor; no runtime teacher."""

import numpy as np
import torch


def initialize_direct_actor(model, env, steps, *, rounds=5, epochs=20):
    """Label learner-visited states, with a decreasing teacher intervention rate.

    Small imitation error on teacher trajectories alone does not establish
    feedback stability. DAgger includes drift/recovery histories from the actor.
    Collect across full target coverage, then restore the PPO curriculum scale.
    Each fit renormalizes all stored raw histories using current frozen stats.
    """
    if steps < env.num_envs * rounds:
        raise ValueError("direct initialization needs at least one vector step per aggregation round")
    histories, targets, records = [], [], []
    scales = env.get_attr("target_scale")
    env.env_method("set_target_scale", 1.0)
    obs = env.reset()
    try:
        for iteration in range(rounds):
            beta = 1 - iteration / max(1, rounds - 1)
            for _ in range(max(1, steps // (env.num_envs * rounds))):
                teacher = np.clip(np.asarray(env.env_method("teacher_action"), dtype=np.float32), -1, 1)
                histories.append({key: value.copy() for key, value in env.get_original_obs().items()})
                targets.append(teacher)
                predicted = model.predict(obs, deterministic=True)[0]
                intervention = np.random.random((env.num_envs, 1)) < beta
                actions = np.where(intervention, teacher, predicted)
                obs, _, _, _ = env.step(actions)
            raw = {key: np.concatenate([row[key] for row in histories]) for key in histories[0]}
            count = len(next(iter(raw.values())))
            actions = torch.as_tensor(np.concatenate(targets), device=model.device)
            observations = {}
            for key, values in raw.items():
                stats = env.obs_rms[key]
                normalized = np.clip(
                    (values - stats.mean) / np.sqrt(stats.var + env.epsilon), -env.clip_obs, env.clip_obs
                )
                observations[key] = torch.as_tensor(normalized, dtype=torch.float32, device=model.device)
            losses = []
            for _ in range(epochs):
                for indices in np.array_split(np.random.permutation(count), max(1, count // 256)):
                    prediction = model.policy.get_distribution(
                        {key: value[indices] for key, value in observations.items()}
                    ).distribution.mean
                    # Motion commands are rare relative to masked and holding
                    # dimensions; do not let predicting zero dominate the fit.
                    weight = 1 + 4 * (actions[indices].abs() > 0.02)
                    loss = torch.mean(weight * (prediction - actions[indices]) ** 2)
                    model.policy.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                    model.policy.optimizer.step()
                    losses.append(float(loss.detach()))
            records.append(
                {
                    "round": iteration,
                    "teacher_probability": beta,
                    "samples": count,
                    "final_batch_weighted_mse": losses[-1],
                }
            )
            print("Imitation round", records[-1], flush=True)
            # The last step's observation was normalized with the collection
            # stats, which are unchanged by fitting.
    finally:
        for index, scale in enumerate(scales):
            env.env_method("set_target_scale", scale, indices=index)
    return {
        "method": "dagger_full_target_coverage_v1",
        "samples": count,
        "rounds": records,
        "runtime_teacher": False,
    }
