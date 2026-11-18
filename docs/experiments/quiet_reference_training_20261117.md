# Stationary-reference exploration training — complete

This fixed-budget pair follows the [completed critic pilot](finite_critic_pilot_20261117.md)
and [architecture review](../rl/WHOLE_BODY_ARCHITECTURE_REVIEW.md). The retained
policy is insufficient; the experiment is designed to improve physical tracking
and settling, rather than merely preserve its current score.

Both arms start from the same retained actor and fresh critics, seed 107,
16,384 decisions with 1,024 critic-only warmup, unchanged legacy rewards,
corrected finite-contract deadlines, three independent critic epochs, std 0.02,
and the previous actor LR 5e-5 → 1e-5 / clip 0.05 / target KL 0.01 settings.
Outer general/TAMP exposure is 50/50 by control ticks; general is 80/20 hand/base.
The identical twelve train and twelve audit cases, ten existing coupled cases,
and six unloaded TAMP prefixes retain their goals and thresholds.

`ordinary_noise` uses the original distribution. `quiet_reference_noise` scales
only motion exploration by 0.125 when present/future pose targets and masks are
constant and announced twist is exactly zero. Upcoming motion/mask changes keep
full noise. Measured error or motion never suppresses correction commands.
Gripper noise stays unchanged; sampling, likelihood and entropy use the same
conditioned Gaussian. No output smoothing, teacher loss or actor inference change.

The initializer, critic parameters and first observation must match exactly.
The first sampled rollouts are expected to diverge because exploration differs;
this pair must not claim identical first actions. Audit cases never enter PPO,
no intermediate checkpoints are selected, and no automatic extensions occur.
Evaluate fixed final actor means on identical physics/reference fingerprints,
reporting hand accuracy, quiet settling, base drift, jerk, contact, wheel support
and confirmed stops. All case failures remain in the denominator.

The smaller frozen-actor probe is diagnostic: 0/8 → 3/8 stochastic successes,
all 16 stops confirmed, same one unsafe-contact identity. It is not a trained
policy improvement. This new source requires its own complete regression receipt
and fresh native teacher admission before PPO; the prior 1,574-test receipt
cannot qualify it. The source-bound runner and exact protocol remain under
ignored `logs/recovery/quiet-reference-pilot-v1/`; raw native evidence,
checkpoints and offline W&B records go to ignored
`logs/manipulation/quiet-reference-pilot-v1/`. No hardware motion or promotion.

## Launch record

Runtime `9d0620f9c3d3e198494b0a02ca3ce78513b4d04c` is frozen from the committed
source, with archive/model/test inputs verified against the canonical worktree.
The 99 focused checks pass. Fresh full gate `20261117_015047_2d92b1` runs on
CPU15; native admission `20261117_015047_4e1b10` runs on CPU14. PPO supervisor
`20261117_015136_781d98` waits for both, verifies the complete new receipt and
runner hashes, then trains the ordinary/quiet pair. At this launch snapshot,
PPO has not started. A paper build waits until the test worker finishes.
The viewer is paused while the native worker is reserved, then automatically
restored on localhost:8766. Numerical threads are one and CUDA is hidden.
[Compact launch evidence](../../paper/data/whole_body_control/quiet_reference_pilot_v1/README.md)
contains exact producers and identities; live status is in ignored job logs.

Fresh native teacher admission passes 12/12 training and 6/6 TAMP prefixes;
all 24 teacher cases remain recorded (23/24 overall). The retained deterministic
baseline replay continues while the fresh full regression gate runs. This is
admission evidence, not new PPO progress or a learned-policy success claim.

## Completed result

Fresh full gate passes **1,585 tests** in 1,025.80 s. Both arms finish 16,384
decisions (15,360 actor-update decisions): ordinary 19,662 physical ticks;
quiet 19,713. The paper builds to 61 pages. Offline W&B IDs are `2wswbine`
and `9hcd3mka`. Initial policy/first observations match; first sampled actions
intentionally differ because noise changes. The ordinary arm exactly reproduces
the preceding independent-critic actor weights, providing a deterministic control.

| Controller | New train | Reused audit | Existing coupled | Unloaded TAMP |
| --- | ---: | ---: | ---: | ---: |
| Retained | 11/12 | 9/12 | 6/10 | 6/6 |
| Ordinary exploration | 11/12 | 9/12 | 7/10 | 6/6 |
| Quiet-reference exploration | 11/12 | 8/12 | 6/10 | 6/6 |

**Reject the reduced-noise candidate.** It adds contact failures on new audit
hand 2 and existing coupled training case 4. Across new/existing coupled cases,
unsafe-contact episodes increase 3 → 5. All 120 actor stops confirm; there is no
evaluation wheel lift, invalid physics or retention loss. Worst active-hand
endpoint error increases **13.10 → 19.59 mm**, base linear jerk
**5.89 → 22.26 m/s³**, while yaw jerk falls **53.36 → 38.35 rad/s³**.
Lower yaw jerk alone is not an overall improvement. All failed cases remain.

The training hypothesis was partly confirmed: ordinary exploration completes
0/20 general episodes; quiet completes 13/20. Ordinary has one training wheel
lift, quiet none. Quiet lowers noise on 8,971 decisions, but receives only 472
actor optimizer steps versus 624 ordinary steps under the same KL limits.
Actor KL early-stops occur in 32/60 updates versus 15/60 ordinary updates.
Median tracking-head gradient norm doubles (42.40 versus 21.23). Shrinking
Gaussian variance also tightens the KL constraint on mean changes; these
measurements suggest an exploration/trust-region interaction, but do not
identify a unique cause of poor generalization. A
successful noisy hold is insufficient evidence for deterministic pose accuracy
or contact safety. The optional setting stays disabled by default.

The next experiment isolates **hand-pose reward weight alone, 1 → 4**, keeping
ordinary exploration, original remaining reward terms, independent critics and
physical guards fixed. The earlier combined reward pilot changed several terms
and cannot answer this narrower question. Reuse the completed ordinary control
only after exact initial-state/first-action pairing before learning. These audit
cases have been inspected repeatedly and now serve development comparison;
fresh confirmatory cases are required before any promotion claim.

After completion the retained viewer restarts. The temporary runtime subsequently
disappears; raw evidence and archived source remain intact. Any follow-up restores
the checksummed source into ignored workspace storage and revalidates the same
source-bound receipt, rather than treating missing files as qualified inputs.
