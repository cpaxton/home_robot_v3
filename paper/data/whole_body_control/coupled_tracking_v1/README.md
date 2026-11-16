# Coupled tracking build and policy diagnostics

See [the experiment report](../../../../docs/experiments/coupled_tracking_20261116.md).
`results.json` retains native teacher outcomes, source/archive/XML/receipt hashes,
PPO and supervised-bootstrap metadata, fixed case rows, and diagnostic boundaries.
The SVGs show private reference paths and actual teacher tracking for both robots.
Raw rollout files, weights, source archives and W&B files stay under ignored logs.

The runtime for this v1 evidence is `db6b6964`. The subsequent coarse-base v2
benchmark and serialized actor feedback flag are a separate source/readout. Do
not reuse the 1,421-test v1 receipt to qualify that runtime. Script/protocol text
is archived for reproduction; outputs and initializers use local ignored paths.

Keep outcome producers distinct: teacher, raw learned actor, fixed-weight base
feedback ablation and explicit executor hold/reacquisition. Repeated case/seed
replays and development-selected variants are not independent reliability trials.
Open-jaw grasp-pose success is not grasp retention or pick-and-place success.
No policy is safety-promoted and no hardware is used.
