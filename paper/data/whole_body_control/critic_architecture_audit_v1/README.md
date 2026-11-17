# Critic / actor architecture audit

See [the architecture review](../../../../docs/rl/WHOLE_BODY_ARCHITECTURE_REVIEW.md).
These exact, small producers and reports come from frozen runtime `5f0eb3c2`;
no actor is trained or exported by the study. Sixteen native stochastic episodes
produce 7,097 transitions, with five training and three held-out cases and two
noise seeds per case. Native arrays/weights remain ignored under
`logs/eval/critic-architecture-audit-v1/`. The reward alternatives share physical
traces; trained critic values on retained-actor trajectories are off-policy
stress diagnostics. Finite-return fits are split by cases and do not certify
expected state values or safety.

The six-case region probe preserves weights and raw observations/context.
It improves selected successes but adds unsafe contact, so it is not promoted.
`region-probe.json` keeps compact physical metrics and the full raw-report hash.
No compressed traces or checkpoints belong in this directory.
