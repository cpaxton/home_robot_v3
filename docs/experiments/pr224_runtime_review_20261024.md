# PR 224 runtime review repairs, 2026-10-24

The four runtime findings were checked against the current source. These changes
repair stopping and feedback checks and expose existing teacher/timing choices;
they do not tune the tracking policy, relax qualification thresholds, or enable
hardware motion.

## Spatial teacher units

`TaskPoseTeacherConfig` separates position and orientation feedback gains in
inverse seconds. Position error produces m/s and rotation-vector error produces
rad/s. A declared characteristic length converts angular rows to linear units
for the damped least-squares task metric. Defaults retain both legacy gains of
4/s and the legacy 1 m/rad metric. This documents the heuristic instead of
claiming that its position/orientation tradeoff is optimal. Changing the metric
or gains requires separate paired trajectory validation.

## Measured whole-body stopping

The shared `whole_body_stopped` check requires finite arm velocities below the
existing 0.03 per-joint SI cutoff, the norm of all three base translation rates
below 0.005 m/s, and the norm of all three angular rates below 0.02 rad/s. Vertical bounce,
roll and pitch can no longer be ignored by cancellation, terminal quiet hold
or independent public-task success scoring. The simulator facade's explicit
stop handshake also uses this same check. Wall-time and simulation-time stop
timeouts remain bounded at 0.75 s, and an unconfirmed stop retains ownership.

## Explicit simulator timing limits

`TaskPoseRuntimeTiming` configures observation/inference, total cycle and
physics-state freshness ceilings. Defaults remain 2*control_dt, 2*control_dt
and 0.25 s. Invalid/nonfinite limits reject; a cycle budget smaller than its
inference budget rejects. The constructor takes these trusted settings, not the
remote goal payload. Advertisements and receipts expose the active limits, and
runtime diagnostics retain observed durations.

Limits never widen automatically after a slow step. Command lifetime and state
freshness are checked again after inference, before another action reaches
`env.step`. A policy may meet its inference ceiling but still lose permission
to actuate because the command expired or its state became stale. Cycle
overruns continue to fail and request measured stopping. These simulator
settings do not change the hardware odometry boundary or robot watchdog.

For an explicitly slower simulator host, for example:

```python
from emet.rl.manipulation.command_runtime import TaskPoseCommandRuntime, TaskPoseRuntimeTiming

runtime = TaskPoseCommandRuntime(
    env,
    timing=TaskPoseRuntimeTiming(
        inference_budget_s=0.15,
        cycle_budget_s=0.30,
        max_state_age_s=0.25,
    ),
)
```

These are failure ceilings, not a requested control cadence or hardware setting.
The simulation timestep, speed/slew limits and absolute command deadline remain
independent.

## Guarded driver feedback

The disabled-by-default `GuardedControlSession` now requires a stationary arm
before session startup and rejects measured arm overspeed before dispatching even
a zero requested velocity. Missing, malformed or nonfinite measurements fault.
Feedback is copied before use. Command slew remains based on the last accepted
command and elapsed wall time; disturbed/noisy feedback is not treated as a new
command. Initial motion, invalid feedback and observed overspeed latch a fault
and request a bounded measured stop. The guard never restarts/recenters itself.

This is tested against a loopback driver. Herman's actual units, watchdog,
measured-stop implementation and force calibration remain unverified; the
existing hardware driver boundary stays disabled.

## Evidence

The focused runtime/control-session/reactive/public-tool tests pass, including
each formerly omitted base DOF, combined-axis motion, invalid feedback, separate
timing limits, stale state after inference and command expiry during inference.
Deterministic test clocks exercise timing rather than relying on host contention.

**86 focused tests and 642 full regression tests pass**, with no failures, errors
or skips. The full regression run has a verified source-bound
[receipt](../../paper/data/whole_body_control/pr224-review-regressions.receipt.json)
and adjacent JUnit XML. Runtime package parity and relevant lint pass.

The unchanged-default teacher was also compared with the pre-review function
at `8cb85269` on 24 independently reset, matched Mars/Stretch states, with base
constraints enabled/disabled and timed/static goals. Maximum normalized action
difference was **zero**. The first exploratory probe reused mutable controller
state across calls and was invalid as a parity comparison; the archived result
uses independent matched resets.

The frozen `left:20g:0` Mars public transfer also passes with all independent
predicates true, including the stronger robot-still predicate. Its peak contact
load remains 3.32 N with no wheel-unloading termination. This is one positive
control, not a rerun of the complete carry/public/agent qualification suites;
their historical source-bound reports remain unchanged. No training or hardware
trial was started by these review repairs.

The [summary](../../paper/data/whole_body_control/pr224_runtime_review_v1.json)
and adjacent compressed full report preserve the exact runtime source hashes,
positive-control trace, default-teacher comparison and test-receipt identities.
