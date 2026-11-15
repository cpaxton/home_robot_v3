# Reference-conditioned base exploration: better settling, unchanged full mastery

The bounded fresh GRU comparison is complete. Reducing base sampling standard
deviation for explicitly constrained, stationary base previews improves arm
tracking and settling from **1/10 to 7/10**. Full arm passes remain **1/10**:
nine cases still fail base-linear jerk. Keep the retained production source and
default scale 1.0. This is a useful single-seed diagnostic result, not controller
qualification or evidence that the robot is motionless.

## Fixed outcomes

| Measurement | Identical initial actors | Control, scale 1.0 | Candidate, scale 0.1 |
| --- | ---: | ---: | ---: |
| Full contract, all twelve cases | 2/12 | 2/12 | 2/12 |
| Tracking and settling, all twelve | 2/12 | 2/12 | 8/12 |
| Arm tracking and settling | 1/10 | 1/10 | 7/10 |
| Arm smoothness / full contract | 10/10 / 1/10 | 1/10 / 1/10 | 1/10 / 1/10 |
| Arm fixed-base contract stops | 0/10 | 5/10 | 0/10 |
| Arm terminal-hold failures | 9/10 | 9/10 | 3/10 |
| Arm base-linear jerk failures | 0/10 | 9/10 | 9/10 |
| Arm base-yaw jerk failures | 0/10 | 3/10 | 0/10 |
| Completed arm endpoint within 3 mm and 0.02 rad | 1/10 | 5/10 | 7/10 |

Both final actors complete all ten arm references and keep every termination
position error within 3 mm. Candidate median arm position/orientation errors
are **0.201 mm / 0.00902 rad**, versus control **1.524 mm / 0.02137 rad**.
Worst orientation worsens from 0.0333 to 0.0449 rad: both roll directions still
time out, as does full-amplitude `supported_A_up`. Position accuracy alone
does not establish orientation or full-contract mastery.

The six new arm tracking/settling passes are full-amplitude pitch up/down and
their quarter/half-amplitude variants. All six still fail base-linear jerk.
Both actors retain exactly the initial full-success IDs: base hold and
quarter-amplitude `supported_A_up`. Base forward fails in every readout.
There are no new full-success cases and no coupled, broader contact, loaded,
public-tool or hardware evaluation in this twelve-case diagnostic population.

## What the physical traces show

The two trained quarter-pitch endpoints improve from control 0.950/1.249 mm
and 0.0163/0.0197 rad to candidate **0.059/0.039 mm** and
**0.00270/0.00313 rad**. Candidate base heading errors are
0.00130/0.00113 rad, versus control 0.01250/0.01576 rad; native settling time
is nine seconds instead of zero.

The archive also measures running samples whose announced base/hand targets
match the final reference, using their last one-second window and excluding
terminal stop samples. Candidate median absolute yaw commands in those two
windows fall to approximately 0.00029/0.00017 rad/s, versus control
0.00265/0.00313 rad/s. Measured forward motion remains: candidate median
absolute forward speeds are 1.37/1.74 mm/s, versus control 1.11/0.30 mm/s.
Passing the existing settling band and reducing commands do not eliminate
physical motion or jerk.

A read-only acceptance audit reconstructs all 48 recorded base-linear/yaw
jerk peaks from saved base poses, with maximum error **2.22e-16**. All nine
candidate arm linear-jerk failures peak during ordinary `running` samples,
not fault stopping. Candidate median/maximum arm-case linear jerk are
0.353/0.660 m/s³, versus control 0.347/1.057 m/s³: the worst peak falls, but
the median and failure count do not improve. No threshold or metric is changed.

This audit exposes a measurement difference to investigate next. The unchanged
[v2 reward](../../src/emet/rl/manipulation/whole_body_reward.py) differentiates
Jacobian-derived velocity vectors, while the
[acceptance measurement](../../src/emet/rl/manipulation/env.py) differentiates
scalar planar displacement speed and wrapped heading rate. The existing legacy
contract objective uses the native acceptance channels. This difference is not
proof of the cause of these failures; compare the reward and acceptance signals
before another weight increase or training extension. Preserve the original
gate and report both measurements if their interpretation is revised.

A separate signed/unsigned diagnostic confirms why these definitions cannot be
treated as interchangeable. For the synthetic straight trajectory
`x(t) = 0.05*t*t`, acceleration is constant at 0.1 m/s² and vector jerk is zero.
Excluding the two derivative warmup samples, the unsigned-speed formula peaks
at **1, 2 and 4 m/s³** for 100, 50 and 25 ms sampling, respectively. Signed
displacement derivatives stay below 2.7e-12; constant-velocity and non-reversing
constant-acceleration controls also have negligible jerk in both formulas.
An unsigned-speed reversal spike is not by itself evidence of harsh physical
jerk.

This does not clear the candidate. Its ten saved arm traces have signed
displacement-vector jerk peaks of 0.462–0.701 m/s³, each larger than that case's
unsigned peak. These are diagnostic measurements with no new teacher budgets
or pass labels, and cannot reconstruct the exact instantaneous-velocity reward.
Both existing implementations use the same base body-frame origin in world XY,
not different axle/center-of-mass points, and the same 50 ms cadence. Acceptance
uses interval-average displacement speed with zero initial derivative history;
the reward uses instantaneous Jacobian velocity initialized from actual qvel.
Record both velocity channels and reset state before claiming exact alignment.
The [signed/unsigned audit](../../paper/data/whole_body_control/mars_reference_base_exploration_v1/signed-jerk-audit.json) and exact script retain all controls and
48 replayed cases; no native physics, policy call or training step is added.

## Controlled change and actual exposure

Only the training distribution changes. The hand must be active, every base
coordinate constrained in history and current/future preview, the future base
targets constant, and announced base velocity zero. Free-base goals and moving
previews retain normal noise. Deterministic means, base correction authority,
actor inputs, rewards, limits and exported single-actor inference are unchanged.
This is not an arm-reachability test; a static far target can still need base
correction. Sampling, PPO likelihood and entropy share the same distribution.

Both fresh runs use seed90, shared advantage normalization, LR 3e-4, five epochs,
batch128 and rollout256, with exactly **4,096 decisions each / 8,192 total**.
The candidate verifies source, preflight, case set, complete PPO settings,
initial actor/policy/optimizer, initial observation and RNG against the control
before sampling. All 48 evaluation identities and both initial outcome/metric
records match. Rollouts may diverge once the changed distribution is active;
there is no exact-rollout or old-source recovery exemption for this comparison.

The condition holds on 2,074 control decisions and **2,063 candidate decisions**;
only the latter have reduced noise. Actual case exposure (control/candidate):
base hold 2,022/2,033, quarter pitch up 793/792, quarter pitch down 1,281/1,271.
The stochastic training episodes record 0/27 and 0/25 successes. Deterministic
fixed evaluation is a distinct measurement, not a relabeling of those episodes.

The default control independently reproduces the earlier shared-normalization
C1 run: all 24 initial/final outcomes and every continuous metric, complete
first-rollout numeric hashes, and final actor-weight bytes are identical.
This is a post-run audit; the control was freshly trained, and admission was
not borrowed across changed source identities.

## Validation and provenance

[PR #238](https://github.com/cpaxton/home_robot_v3/pull/238) implements
the opt-in setting. Immutable revision
`3ac722e1096dc901d68192b803dc9596471c46c7` passes **1,260/1,260** fresh full
regressions, including the 48 focused checks. Teacher admission is **164/164 general**
and **5/6 pregrasp**, preserving the object-motion challenge; all six measured
stops confirm. Full checks take 977.8 seconds; paired training and readouts take
203.7 seconds after admission. This is not a world-model or host speedup claim.
The complete paper builds successfully at 57 pages with no undefined citations.

Managed jobs `20261115_101800_b63738` (checks), `20261115_101800_2b8ed4`
(teacher) and `20261115_102028_c99a87` (pair) all report `done`.
Host-visible native children have exited. CPU14/15 reservations are released,
except any separately recorded document build. Runs use one numerical thread,
CUDA hidden and ordinary rendering disabled. No GPU training, Caliban or
hardware motion, extra seed, extension, intermediate selection or promotion.

The [predeclared protocol](../plans/mars_reference_base_exploration.md) and
[complete compact evidence](../../paper/data/whole_body_control/mars_reference_base_exploration_v1/results.json)
retain every case, failure, update, exposure, active endpoint measurement and
artifact binding. Adjacent text files archive exact supervision/extraction and
jerk-replay scripts; detailed traces, checkpoints and offline artifacts remain
in ignored storage. Experiment completion does not close the broader
coordination, smoothness or task-qualification TODOs.
