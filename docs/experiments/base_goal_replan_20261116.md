# Continuous base-goal replans and reverse-motion diagnosis

Simulator-only follow-up to the [rest-to-rest endpoint report](base_goal_reference_20261116.md).
The prior actor parameter tensors, 25-second horizon, 5 cm / 5-degree base
contract, two-second measured hold and existing force/support guards stay fixed.
No new PPO updates, hardware motion or policy promotion are involved.

## Implemented reference and viewer behavior

`BaseGoalReference` can replace its own unretimed announced reference during
acceleration, cruise, deceleration or a previous braking segment. A one-axis
jerk-limited braking prefix matches the old nominal pose, forward/yaw velocity
and acceleration, then joins a new turn–drive–turn S-curve at rest. Impossible
boundaries, unrelated moving schedules and external phase retiming are rejected;
they are not silently restarted at zero velocity. Reference derivative bounds
are kinematic and do not certify measured acceleration, force or wheel support.

The sampler retains a bounded 50 ms tail of previous primitives. This preserves
the negative preview row and the central-difference announced twist at the
installation instant. Clamping negative time at the new boundary would halve
feedforward on the first tick despite a mathematically matched forward path.
The implementation does not retain a chain of prior models/FK state.

`install_base_goal` screens initial measured rest across all six free-base
velocities and arm velocities, then admits the full nominal geometry before
installation. Moving replans match the existing reference, not a newly measured
zero-velocity boundary. Installation preserves physics, past command history,
gripper authority and fault state. Failed updates preserve the prior valid
schedule; they do not clear a stop or retry through a fault.

The viewer opts in only for a selected reference-tracking actor in unloaded
base-only reach mode. Its blue marker, nudge anchor and tracking-error display
represent the final endpoint, rather than the intermediate moving reference.
Adjacent drag/key edits are combined into one admission/replan; relative nudges
accumulate and pause/hold/mode commands retain queue order. Hand/mode changes
clear the old base schedule. The selected checkpoint's native velocity-feedback,
wheel-slew, tracking-precision and quiet-hold settings are applied explicitly.

A live worker/renderer test catches an unintended pause/resume regression, now
fixed: a valid goal resumes a paused simulation without another Run command.
The same test exposes a cold-worker MuJoCo 3.5 numeric-XML loading failure on a
valid large inline Mars vertex attribute. The captured finite mesh XML compiles
unchanged outside the failing startup path. Viewer construction now converts
generated vertex/face arrays into in-memory native binary MSH assets. Training
and benchmark factories keep their default inline representation.

For both robots, tests compare more than 400 compiled model arrays, preserving
all numerical arrays except source-path offsets, and compare ten physics steps
bitwise. Source XML identity, geometry, transforms, mesh coordinates, inertia,
actuators and contact settings remain unchanged. Unsupported mesh encodings
fail closed. No dependency, driver, global allocator or robot-model tuning is
used as a workaround. The live renderer test includes startup, queued pause,
goal-driven resume, a rendered frame and clean worker shutdown.

## First complete native readout: source e30a7dfc, feedback gain 0.5

Job `20261116_144906_3388c2` completes all **101** cases. The unchanged 86-case
cohorts reproduce 24/24 base trajectories, 10/10 small general, 6/10 larger
general, 6/6 unloaded TAMP, 4/4 moving handoffs, 2/8 raw static endpoints,
8/8 compiled development endpoints and 16/16 compiled endpoint audits.

The new fixed moving-goal audit passes **10/12**. All **36** requested changes
are admitted and delivered, including repeated changes during braking and ten
consecutive 50 ms updates. Boundary tests verify pose/velocity/acceleration and
announced-twist continuity, with no physics/history reset. Three actual
`Simulation`/public-bundle integration cases pass **3/3**, including 55 logical
goal edits combined into twelve admitted replans. These direct integration cases
are distinct from the later cold-worker/renderer regression.

The two failures are reverse-drive cases `mars:base-moving-audit-v1:5` and `:7`,
seeds 35870005 and 35870007. Both request a new lateral goal at tick 30 (1.5 s).
They stop at 1.708 s and 1.678 s on the unchanged support guard: caster gaps
0.1141 mm and 0.1093 mm with zero caster load. Both drive wheels remain loaded.
There is no excessive-contact flag. The small gap does not waive the failure.

Paired replay job `20261116_145325_e28df4` repeats each exact reset with and
without the goal change. Both unchanged references also fail, with exactly the
same recorded caster gaps. Thus these are existing reverse-motion failures,
not an improvement/regression estimate attributable to replan construction.
Repeated deterministic replays are not counted as independent reliability
trials. Physics-rate traces preserve current base/wheel velocities, requested
and applied commands and actuator-space forces without an extra forward solve.
Actuator-space force is not universally a Cartesian force in newtons.

## Native feedback calibration diagnostic

Job `20261116_150733_00c390` changes only the additional body-velocity
proportional feedback gain after the same reset, retaining learned tensors and
native wheel velocity servos. Gain 0.25 resolves one of the two failed resets;
gain zero resolves both, with and without retargeting (4/4 reused replays).
The original gain-0.5 failures stay in their original tables. This is
post-failure control-loop diagnosis, not a fresh held-out or learning result.

`TampPrimitiveEnv` and `primitive_rollout` now accept and record the same
explicit gain, defaulting to their previous 0.5. Spawned-worker tests check both
0 and 0.5 through the actual pose boundary and preserve executor-only closure.
This permits the broad comparison to use the declared gain consistently in
general, viewer and TAMP execution instead of silently mixing runtime settings.

## Final-source verification and fixed gain-zero comparison

Source **13c53634** includes the cold-viewer repair and explicit TAMP gain option.
Its snapshot contains **3,486** checksummed inputs, including the HTML asset.
The compressed source archive SHA-256 is
`01d33ac94b1600aa1b522bdfc90bda76c75be5d437fe889785fec97604437f7f`;
the uncompressed tar SHA-256 is
`1283c90af4f92a64eafdebbd8abce58c723aa0da7e8551bbc6948b65bac38a38`.
The final runtime is `/tmp/emet-base-goal-replan-13c53634`.

Final regression job `20261116_151952_bd45a2` completes **1,520/1,520**
source-bound tests, with zero failures, errors or skips (JUnit 1,205.10 s;
receipt interval 1,206.78 s). The receipt is verified against the canonical
source, including the HTML asset. Gain-zero job `20261116_151955_512625`
completes all **109** cases. Both use one numerical thread, CUDA hidden and
CPU15/CPU14 respectively. No PPO steps run.

The fixed gain-zero comparison repeats the complete 101-case readout and adds
eight fresh reverse retargets (seeds 35872000–35872007, 12/18 cm reverse targets
and signed lateral changes). These cases are not added to training or used to
retune the candidate. Configuration, weight and source identities are kept per
producer; the original gain-0.5 results retain their earlier source identity.

| Cohort | Retained gain 0.5 | Global gain zero |
| --- | --- | --- |
| Base trajectories | 24/24 | 24/24 |
| Small general | 10/10 | 4/10 |
| Larger general | 6/10 | 2/10 |
| Unloaded TAMP | 6/6 | 5/6 |
| Moving handoffs | 4/4 | 0/4 |
| Raw static endpoints | 2/8 | 2/8 |
| Compiled development endpoints | 8/8 | 7/8 |
| Compiled endpoint audit | 16/16 | 15/16 |
| Moving-goal audit | 10/12 | 12/12 |
| Viewer integration | 3/3 | 2/3 |
| Fresh reverse retargets | Not run | 8/8 |

**Reject gain zero as a global replacement.** The remaining viewer failure
finishes at 25.37 mm / 0.09357 rad at 25 s, outside the unchanged heading and
quiet-hold contract. The candidate and all failures are retained. Native
coupled cases use randomization scale zero; viewer cases use the existing scale
one. Paired viewer comparisons use identical seeds. A template-model identity
check does not certify equality of every randomized runtime parameter.

## Scoped diagnostic and remaining decision

Job `20261116_153146_db79d9` completes **15/15**: two failed reverse replays,
three viewer cases and ten small-general cases. It uses linear gain zero/yaw
gain 0.5 only for fully constrained base-only commands; active-hand commands
retain gain 0.5 on both axes. This is a temporary controller override, **not a
deployable checkpoint or supported production configuration**. It preserves
learned tensors and tests whether existing hand control can be retained while
addressing reverse support loss. It does not establish larger-general, TAMP,
fresh-audit or mask-transition/handoff performance. One reverse replay reaches
0.09692 mm caster gap, close to the unchanged 0.1 mm threshold.

Measured motion is still less smooth than the reference. Maximum recorded
base yaw jerk is 150.09 rad/s³ in the gain-0.5 moving audit, 112.43 with global
gain zero, and 109.63 across the scoped diagnostic. Corresponding maximum
linear jerk is 20.68, 23.44 and 18.52 m/s³. These finite-difference measurements
are not force or damage ratings. Synchronous reference construction takes
87.74 ms median / 201.07 ms maximum in the original moving audit and
89.10 / 191.49 ms with gain zero. It is not hard-real-time 50 ms compilation;
actor inference and native tracking are separate operations.

The next decision is to implement a supported, versioned per-axis execution
contract, then compare it on broader coupled cases, TAMP, fresh reverse audits
and live mask transitions/handoffs before further PPO. Do not expose the
unsupported diagnostic override through the checkpoint selector. General/TAMP
solver settings still differ and need explicit per-plant checkpoint admission.
Loaded grasp/lift/carry, arbitrary disturbances and Herman remain separate work.

## Viewer, artifacts and verification

Finalizer job `20261116_153518_be7428` verifies the receipt and passes a fresh
subprocess test of actual cold `TestDrive` startup, public checkpoint selection,
automatic resume on a goal, five accumulated nudges, ordered pause, rendering
and clean shutdown. This is separate from direct main-thread `Simulation`
integration. The paper builds to **60 pages**. Evaluation is logged to offline
W&B run `5ew43iu9`; no upload, training or automatic promotion occurs.

The live simulator is on localhost **8766**, managed job
`20261116_154539_54ba7e`, using the immutable final source and only the retained
gain-0.5 actor root. It does not connect to hardware. Over SSH, forward port
8766 and open `http://localhost:8766`. Reset if the scene has finished/faulted
before selecting the checkpoint, then press `1` for base targets. Accepted goal
edits resume execution automatically. A fault is never cleared to load a policy.

Compact complete outcomes, producer scripts, source/receipt hashes and a figure
are in [base_goal_replan_v1](../../paper/data/whole_body_control/base_goal_replan_v1/README.md).
Raw physics traces, actor weights and compressed archives remain ignored.
The earlier partial regression run is cancelled after the live-viewer issue;
its verified orphan process group is terminated and its XML is never reused.
A disk-full snapshot attempt is retained. Three owned completed/unused duplicate
runtimes are removed only after checking completion and archive checksums;
source archives, logs, weights and results are preserved.

## Stop-reporting limitation

The legacy evaluation helper retries 16 scored ticks after a support fault.
Because a latched wheel fault ends each scored tick after one 2 ms physics
substep, those retries advance only **32 ms**, not 0.8 s. Its original
`stop_confirmed=false` fields are preserved; they do not establish a failure to
stop within a full physical stopping window. Production cancellation already
uses unscored stepping with separate 0.75 s wall-clock and simulation deadlines.
Separate offline diagnostic `20261116_155434_94880d` repeats the two exact
faults with an explicit 0.75 s simulation budget. Both meet the existing
all-six-base-DOF/arm quiet thresholds after **0.15 s** of unscored physics,
with wheel and stop latches preserved. This is a physical-time result, not a
wall-clock cancellation-deadline test. It does not replace or waive either
original wheel-unloading failure.
