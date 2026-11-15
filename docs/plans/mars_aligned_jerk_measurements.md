# Align base-jerk observations before changing the objective

The completed exploration experiment improves arm tracking/settling to 7/10,
but nine arm cases still fail the frozen linear-jerk gate. The prior offline
audit establishes different definitions, not an excuse to relabel failures.
This next diagnostic records those definitions on the same native steps.

## Fixed scope

Replay the control and candidate final actors from immutable `3ac722e1` on all
twelve predeclared C1 evaluation cases each: **24 episodes**, no new cases,
seeds, PPO updates or exported policies. Use the same physical models, references,
teacher-relative budgets and independent guards. Preserve failed or incomplete
diagnostic attempts in new output directories; never overwrite frozen results.

The completed 24-episode readout exposes large instantaneous-velocity peaks
even during base hold. A separately recorded **two-episode stationary anchor**
therefore replays the same base-hold case with the original zero-head actor
and analytic teacher. Both must reproduce their archived trajectories and
metrics exactly. This addresses the plant/sampling question without another
training variant or any change to the original 24-case record.

The next two-anchor replay enables `--physics-substeps` and records every valid
2-ms physics step: copied generalized velocities, actuator commands/forces,
requested base motion and the applied velocity after inner-loop feedback.
The observer is detached before reset/fixture conditioning. It never calls
`mj_forward` mid-step: generalized velocities are post-integration, whereas
actuator forces describe the preceding solve. Actuator-space force units depend
on the transmission. These raw samples diagnose plant/servo oscillation and
control-cadence aliasing without treating endpoint velocity as an interval mean.

The observer records the existing v2 RewardMeter on the deterministic evaluation
states. This is the exact implementation on those states, not a reconstruction
of the earlier stochastic training trajectories or a changed evaluation reward.

## Channels and invariants

- Native acceptance: backward differences of unsigned interval displacement
  speed and wrapped heading rate, with its original zero derivative history.
- Signed displacement-vector velocity/acceleration/jerk on the same poses.
- Actual instantaneous Jacobian velocity and the v2 meter's acceleration/jerk,
  initialized from measured qvel as in training.
- The same instantaneous samples with zero reset history, isolating startup
  differences, and trapezoidal adjacent-sample averages approximating the
  displacement interval. Neither diagnostic substitutes for a frozen metric.

All use the base body origin in world XY and the configured 50-ms cadence.
Record actual clock increments too. Interval and instantaneous velocities have
different nominal timestamps; their backward-difference jerk centers differ
by half a control tick. Angular-Z velocity is not universally the exact Euler
heading derivative under tilt; the original yaw channel stays unchanged.

The optional evaluation observer never changes an action, target, reward,
acceptance label or physics step. Every v2 meter call checks copied qpos,
qvel, ctrl, act and simulator time for mutation. Native gate velocity and
acceleration must equal the observer's reconstructed values at every step.
Both final actors must reproduce all twelve original physical identities,
outcomes, continuous metrics, actions, poses and reference sequences exactly.
Failure aborts interpretation, with raw and partial evidence retained.

## Readout and next decision

Report each channel's peak, peak after the first two samples, peak time/code,
reward proxy at the gate peak, and integrated reward costs. Separate running,
startup and stop events. Keep the original pass counts even if a diagnostic
channel would give another answer. Additional native instrumentation changes
the producer identity; it does not retroactively qualify that source using the
older training receipt.

Use one CPU14 native replay, one numerical thread, CUDA hidden and no ordinary
rendering. CPU15 is available for focused checks. After the readout, decide
whether the objective needs timing/definition alignment, stronger peak costs
or another bounded change. No new training or reward/gate modification is
part of this diagnostic protocol.

## Bounded plant follow-up

The completed substep anchors show a roughly 50-Hz oscillation in chassis
velocity and wheel commands while the requested base command remains zero.
Predeclare twelve teacher episodes: hold, forward and quarter-amplitude pitch-up,
each with legacy cached COM velocity, proportional feedback disabled, current
qvel at the same COM, and current qvel at the axle. Apply interventions only
after unchanged reset/admission and restore them before the next reset. The
legacy variant must exactly reproduce archived teacher outcomes/actions/poses;
every variant must preserve reset/reference identities. Keep all outcomes and
raw archives, including errors. This isolates feedback gain, cache timing and
measurement point; no new filter, reward, policy update or production default.

Current-qvel Jacobians reuse the latest native position-stage geometry; this
probe does not claim to refresh that geometry. Measure the native contract,
instantaneous proxy, high-rate chassis velocity and wheel-actuator force on
all variants before deciding whether a production opt-in correction is justified.

The first twelve-episode feedback attempt is invalid for this comparison:
`code=ok` was incorrectly treated as terminal, restoring feedback during a
continuing hold. Preserve its raw archives under `feedback/`; the corrected
attempt uses `feedback-v2/` and explicit Gym termination/truncation flags.
Regression tests cover nonterminal `ok`, actual termination and truncation.
