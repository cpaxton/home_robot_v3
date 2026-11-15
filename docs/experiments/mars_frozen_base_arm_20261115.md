# Frozen-base/context arm adaptation: a small velocity gain, incomplete coordination

The declared pair is complete: seed90, 16,384 transitions per variant and
32,768 total. Reference velocity improves full arm passes from its own initial
60/64 to **61/64**; baseline regresses from 59/64 to 57/64. Both retain 64/64
arm tracking and settling. This is a modest positive result for the velocity
recipe on one development seed, not a general improvement or policy promotion.
Keep the retained source as the production baseline.

## Fixed initial/final outcomes

| Measurement | Baseline initial | Baseline final | Velocity initial | Velocity final |
| --- | ---: | ---: | ---: | ---: |
| Full arm contract | 59/64 | 57/64 | 60/64 | 61/64 |
| Arm tracking and settling | 64/64 | 64/64 | 64/64 | 64/64 |
| Arm base-linear jerk failures | 5/64 | 7/64 | 4/64 | 3/64 |
| Full base contract | 42/50 | 42/50 | 43/50 | 43/50 |
| Base tracking and settling | 49/50 | 49/50 | 49/50 | 49/50 |
| Full coupled contract | 2/8 | 2/8 | 2/8 | 2/8 |
| Full contract, all declared cases | 103/122 | 101/122 | 105/122 | 106/122 |

Arm validation remains 4/4 full in every readout. Arm training full passes are
55/60→53/60 for baseline and 56/60→57/60 for velocity. Coupled validation
remains 0/2; base validation remains 3/5 full and 4/5 tracking. All canonical
eight-case arm initial/final evaluations also pass. The 122-case expanded
population is the declared admission subset, not the later 164-case contact
population or an actual pregrasp/loaded task benchmark.

Velocity gains three full passes: half-amplitude `roll_left`, half-amplitude
`elbow_up`, and quarter-amplitude `near_straight_A_translate_down`. It loses
quarter-amplitude `translate_up` and `near_straight_A_translate_up` to
base-linear jerk. Baseline gains two and loses four; all losses are base-linear
jerk. No arm yaw-jerk or joint-jerk failure appears in either fixed final.

All 64 arm references complete in every readout. At termination, velocity's
worst arm position/orientation errors improve from **0.992 mm / 0.00970 rad**
to **0.648 mm / 0.00648 rad**. Baseline's worst errors increase from
0.986 mm / 0.00972 rad to 1.435 mm / 0.01365 rad. These are errors at the
reported stop after reference completion, not a claim about maximum transient
error along the full trajectory. Full acceptance still includes the independent
tracking, settling and smoothness checks.

## What freezing did and did not establish

Both serialized actors preserve actor context, base/grip function, restoring
yaw and normalization exactly, while arm gains and critic parameters change.
Every base-only case's outcome and continuous metrics also match its own
initial readout exactly. Preserving the base function does not make the base
physically stationary during arm motion: arm-induced motion and changed live
observations still produce three velocity-final and seven baseline-final arm
base-jerk failures. Six coupled cases still fail tracking/settling.

The historical unfrozen seed90 finals had 38/64 full arm passes for baseline
and 25/64 for velocity. The new recipe retains much more prior capability, but
simultaneously freezes context/base/grip and suppresses wheel/grip exploration.
It is not a contemporaneous four-arm factorial comparison isolating those
changes. Initial baseline/velocity scores also differ, so final 61 versus 57
alone is not the effect of adding velocity. No new seed, intermediate checkpoint
selection, automatic extension or default feature change follows this pair.

The separate [C1 zero-base diagnostic](mars_c1_zero_base_diagnostic_20261115.md)
also points to base interference, but uses a different GRU actor and an initial
zero base function. Its two-policy intervention must not be described as the
single legacy actor or retained learned base used here.

## Admission, provenance and completion

Managed job `20261115_012253_ef3fda` completed successfully on immutable
`afeef3d0b2c24492f492c00efceddde21d420ce5`. It reused that identical source's
passing **1,249/1,249** full regression receipt, and each feature variant
independently passed **122/122** teacher cases before PPO. The retained source
weights hash is
`e7b425699510cc635485bdd8f59a5b25692fb165faf248aaecbc2707c0dbb883`.
The [predeclared protocol](../plans/mars_frozen_base_arm_followthrough.md)
records LR 1e-5, arm std 0.03, arm-only training, deterministic learned wheel/grip
means and unchanged references, rotation6 representation, rewards and limits.

All **244 initial/final case pairs** match reset/reference identities, as do
all 122 cross-variant identities. The archive preserves all **488** expanded
outcomes, failure reasons, separate jerk/error metrics, training updates,
serialized checks and artifact hashes. Teacher admission, training and all
readouts take **2,006.8 seconds (33 min 27 s)** after the verified CPU14
handoff; this excludes the reused regression gate and implementation time.
There is no measured world-model or alternative-host speedup.

The run uses one CPU14 worker, one numerical thread, CUDA hidden and rendering
disabled. Host-visible native children exited and the managed job reports
`done`; CPU14 is released. No GPU, Caliban or hardware motion was used. The
previous queue timeout with zero PPO transitions remains a separate failed
attempt, not an earlier successful run.

[Complete compact evidence](../../paper/data/whole_body_control/mars_frozen_base_arm_followthrough_v1/results.json)
includes the immutable prelaunch protocol and final supervisor state. The
adjacent extraction and supervision text files preserve the exact executed
scripts; original traces and weights remain in ignored artifact storage.
