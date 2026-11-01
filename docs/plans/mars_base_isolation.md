# Isolate base optimization while preserving the learned arm

PR #224 at `a7accb4b` is the validated parent. Its latest mixed actor retains
16/16 expanded arm tracking but only 4/16 full arm passes, with 38/50 base
tracking and 32/50 full base passes. Do not extend that run unchanged.

1. Audit return direction on the existing forward/reverse training cases from
   the same reset posture. Scale the announced smooth reference velocity by
   -0.1, 0, 0.02, 0.1, 0.25, 0.5, 0.75 and 1. Preserve native safety/path guards
   and every failed prefix. This diagnostic may use reference derivatives; it
   is not a runtime controller or a qualification result.
2. Initialize a two-output base training component from the verified pre-PPO
   whole-body arm transfer. Keep the actor context, arm gain rows, grip and
   normalization fixed. Train base gain rows, restoring-yaw gains, exploration
   coefficients and critic only. Apply the frozen learned arm means during
   training. This is training decomposition; the exported runtime remains one
   eight-output actor with the original shared context and learned arm rows.
3. Use standard SB3 generalized state-dependent exploration (gSDE), resampling
   every eight 50 ms control ticks (0.4 s). Noise features are current/future
   announced base errors plus a small constant floor; noise decays naturally
   near a stationary target. Start coefficient std0.5. Restrict forward noise
   to forward-error features and yaw noise to lateral/heading features; this
   affects exploration, never the deterministic mean. Freeze forbidden noise
   coefficients. Native velocity, acceleration, contact and wheel guards remain.
4. Verify actual PPO updates, frozen context/arm rows, checkpoint reload,
   temporally coherent sampling, finite/consistent likelihoods, export into
   one actor and prediction equivalence. The source bundle remains immutable;
   model/environment/normalization binding and fresh teacher/regression receipts
   are required before training. Evaluate the initial merged actor on all16
   arm cases before the run. If this fails, stop before optimization.
5. One 32,768-step seed90 pilot: one CPU14 world, one Torch/BLAS thread, CUDA
   hidden, offline scalar W&B, LR1e-4, gamma0.9975, rollout/batch256, five epochs,
   clip0.1 and target KL0.01. Keep the existing75% arm /25% base transition
   budgets and80% translation share inside base, all reference poses, horizons,
   rewards and acceptance gates. Compare all18 canonical cases at step0, every
   8192 and final; evaluate all66 expanded cases at final. No automatic extension,
   validation-selected checkpoint, teacher action loss or hardware promotion.

gSDE follows Raffin, Kober and Stulp, [Smooth Exploration for Robotic
Reinforcement Learning](https://proceedings.mlr.press/v164/raffin22a.html).
The installed SB3 implementation is inspected directly. The error-based noise
features and frozen-arm decomposition are declared experiment choices, not a
claim that the paper establishes safety or solves this particular benchmark.

Raw component weights, merged bundles and traces stay in ignored `logs/`.
Track compact case metrics, identities, exposure and update diagnostics. Loaded
hold/lift, agent pick/place and Herman remain separate acceptance stages.
