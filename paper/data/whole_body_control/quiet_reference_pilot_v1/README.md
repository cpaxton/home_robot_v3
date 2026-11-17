# Stationary-reference exploration pilot

Exact source-bound protocol/producers for runtime `9d0620f9`. The two arms share
actor/critic initialization, reward, task mixture, three independent critic epochs
and fixed final training budget. Exploration differs, so first actions/rollouts
are intentionally not identical. `status.json` is a dated preparation snapshot,
not a completed evaluation; raw weights/traces and live status remain ignored
under `logs/manipulation/quiet-reference-pilot-v1/` and `logs/jobs/`.

The fresh complete source gate and native admission must both pass before PPO.
The previous 1,574-test receipt does not qualify this source. The experiment
restores the retained localhost viewer on completion/failure; no automatic policy
promotion or hardware motion. See the [protocol report](../../../../docs/experiments/quiet_reference_training_20261117.md).
