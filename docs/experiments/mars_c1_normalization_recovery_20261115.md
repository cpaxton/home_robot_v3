# C1 recovery: better hand tracking, unresolved whole-body settling

The interrupted comparison is complete. The historical per-head control retains
its original 4,096 decisions; the recovered shared-normalization candidate
completes exactly 4,096 decisions and 16 rollouts at immutable `afeef3d0`.
The previous 256 unoptimized candidate ticks remain a separate failed attempt.
No control retraining, extra seed, intermediate selection or threshold change
is hidden in this recovery. The retained source checkpoint is unchanged.

## Fixed-final results

| Measurement | Identical initial zero-head actor | Per-head final | Shared final |
|---|---:|---:|---:|
| Full contract, all twelve cases |2/12|0/12|2/12|
| Tracking and settling, all twelve |2/12|1/12|2/12|
| Smoothness, all twelve |12/12|2/12|3/12|
| Arm position within 3 mm at termination |3/10|0/10|10/10|
| Completed arm reference with position ≤3 mm and rotation ≤0.02 rad |1/10|0/10|5/10|

Both measured initial reports have exactly identical outcomes and continuous
metrics. The shared final masters **the same two cases** as that initial actor:
base hold and `supported_A_up@amplitude:0.25`. This preserves initial mastery
and recovers from the per-head regression; it does not establish new full-task
mastery or beat the retained controller.

Hand following does improve. All ten shared-final arm references complete,
with termination position errors **0.262–1.992 mm** and rotation errors
**0.0064–0.0333 rad**. Five meet both endpoint hand tolerances, versus one at
initialization. Errors for earlier-stopped runs are relative to their active
reference at termination, so the compact record includes reference completion
and execution duration rather than treating every stop as a full endpoint.

The two trained quarter-pitch cases finish their references at approximately
**0.950/1.249 mm** and **0.0163/0.0197 rad**, but do not settle. Their base
heading errors are **0.0125/0.0158 rad**, outside the unchanged 0.01-rad goal
tolerance. Measured base creep and nonzero commands remain after the reference
ends. Improving hand endpoints alone does not satisfy quiet whole-body control.

Five shared arm cases stop on the fixed-base contract, versus two per-head.
Nine of ten shared arm cases fail terminal hold and base-linear jerk; three
fail yaw jerk. The per-head final has ten terminal-hold and nine base-linear
jerk failures. Shared normalization helps tracking credit without solving the
base/quiet/smoothness tradeoff.

No wheel-lift, unsafe-contact, invalid-physics or retention flag trips in either
fixed-final evaluation or the logged stochastic training episodes. Those
episodes still have **0/27** successes for shared and **0/40** for per-head.
Different post-update episode lengths produce different later case exposure;
the compact record reports actual control ticks per case. These short guarded
simulator observations are not hardware safety qualification.

## Credit and implementation evidence

The median correlation between the combined actor advantage and tracking is
**0.999915** for shared normalization and **0.376466** for per-head. Shared
normalization preserves the intended raw reward priorities; the first-minibatch
gradient diagnostics and per-head scales remain recorded. The comparison also
changes aggregate advantage scale, so it does not isolate relative weighting
from that scale change or prove that normalization alone solves learning.

Fresh current-source checks pass **1,249/1,249**, with zero failures, errors or
skips. Teacher admission remains **164/164 general** and **5/6 TAMP pregrasp**,
retaining the declared object-motion challenge. The explicit recovery proof
verifies every declared source change, both actual teacher manifests and all
164 physical identities. It replays the 256 saved actions and the original
first update in disposable copies, with identical policy and optimizer state.
No extra simulator transitions or actor export occur in that offline proof.

The candidate independently reproduces every initial numeric-state and complete
first-rollout hash before its first optimizer update. Old source/preflight
identities are preserved, not renamed. The proof is scoped to this one-tick,
general-only diagnostic; it does not claim equivalence for bundled TAMP returns.

Fresh admission through recovered final evaluation takes **1,089.2 seconds
(18 min 9 s)**. The recovery supervisor itself takes 951.9 seconds, excluding
the earlier source implementation/review. Raw traces, tensor states, weights
and offline W&B stay in ignored storage.

## Follow-through

The independently bounded frozen-base/context arm trial received CPU14 after
C1 completed and its native children exited. It uses the same immutable gated
snapshot and its own teacher admission before optimization. Its earlier queue
timeout with zero PPO steps remains preserved. No automatic C1 extension,
policy promotion or hardware motion follows these results.

The next general-controller work should address active terminal settling,
base authority and exploration while retaining the improved hand following.
Small errors do not justify discarding base-motion, jerk or stopped cases.

[Complete compact outcomes, all cases, update diagnostics and artifact hashes](../../paper/data/whole_body_control/mars_c1_normalization_recovery_v1/results.json)
and the [recovery protocol](../plans/mars_c1_recovery_20261115.md) retain both
initial anchors, every fixed-final failure and the explicit source bridge.
