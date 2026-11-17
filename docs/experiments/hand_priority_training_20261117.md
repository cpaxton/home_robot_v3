# Isolated hand-pose priority training

This experiment follows the [completed exploration comparison](quiet_reference_training_20261117.md).
The reduced-noise candidate regresses deterministic accuracy/contact despite
more successful stochastic holds, and stays disabled. The next hypothesis is
narrower than the previous combined reward pilot: increase only hand-pose
reward priority, with ordinary exploration and all other reward terms fixed.

`hand_priority_only` sets `hand_pose_weight=4`, compared with the completed
`ordinary_noise` control's weight 1. Position/rotation scales remain 0.01 m /
0.1 rad, fine-pose blend and progress remain zero, velocity gating remains off,
and hand velocity weight remains 0.5. Jerk, command-delta, effort, force and
terminal coefficients are unchanged. Raising tracking priority does change
its relative importance; unchanged coefficients do not guarantee unchanged
physical jerk or contact behavior. Those are measured independently.

Both use source `9d0620f9`, seed 107, identical initial actor/fresh critics,
ordinary std 0.02, three independent critic epochs, 16,384 decisions with
1,024 critic-only warmup, the original decaying LR/clip/KL settings and the same
50/50 general/TAMP physical-tick mix. The same twelve training cases comprise
eight hand-active and four base-only cases. Each hand case completes only two
episodes and each base case one in the prior control, so this is a bounded
diagnostic, not enough exposure to establish general mastery.

The completed same-source control is reused explicitly rather than retrained.
The new runner verifies identical initial policy, first observation and first
rollout observations/actions **before its first update**, including warmup.
Reward groups are expected to differ. It reuses the source-bound teacher/physical
admission with explicit original report/checksums and `fresh_physics_replay=false`;
there is no new teacher rollout or renamed historical evidence. Goals, guards,
acceptance thresholds, model/source identities and case ordering remain fixed.
The existing 1,585-test receipt is verified against the unchanged source.

After the previous temporary runtime disappears, the archived source checksum
is verified and restored under ignored `logs/runtimes/hand-priority-9d0620f9/`.
All receipt-bound inputs match the canonical source. Job
`20261117_082059_5a7777` runs the one native worker on CPU14, numerical threads
one, CUDA hidden and W&B offline (`qte8xtkt`). The initial pairing passes and
training starts. Raw outputs live under ignored
`logs/manipulation/hand-priority-pilot-v1/`; exact runners/protocol are under
`logs/recovery/hand-priority-pilot-v1/`. The viewer is paused and restored
automatically on completion/failure using the retained actor.

Fixed final mean-policy evaluation covers the same 24 development cases, ten
existing coupled cases and six unloaded TAMP prefixes. Existing audits have
been inspected repeatedly; they now support development comparison, and fresh
confirmatory cases are required before promotion. No intermediate checkpoint
selection, automatic extensions, teacher actions in loss, loaded carry or
hardware use. Preserve all contact failures and report precision, drift, quiet
settling, measured jerk, contact and stop confirmation separately.
