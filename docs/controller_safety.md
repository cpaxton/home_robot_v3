# Controller safety review and visualization

**Historical checkout review:** the subsequent
[learned-safety curriculum and agent coordination plan](plans/2026-11-13_whole_body_safety_curriculum.md)
records the agreed policy-learning priority and newer active worktrees. The
active manipulation branch now has a physical wheel/chassis model; the planar
limitation below describes the archived base evaluations. The options below
remain the original review, not the current implementation priority.

Review of checkout `0795f470` on 2026-11-13. The numbers below come from archived
local reports, not new rollouts. Their file hashes are embedded in the generated
viewer; historical reports do not consistently identify the evaluated source and
model revisions. The checkout has distinct Mars coupled base/arm, Mars base-only,
and stationary manipulation controllers. Component results do not establish
combined base/arm safety.

## What the existing evidence says

| Evaluation | Feedback | Learned | Scope / limitation |
| --- | --- | --- | --- |
| Mars unloaded arm reach | 100/100 completed | 100/100 completed | Collision-free curriculum; no payload or wheel dynamics |
| Mars base standard | 18/20 completed | 18/20 completed | Planar simulator |
| Mars base retarget | 20/20 completed | 19/20 completed | Planar simulator |
| Mars base push | 20/20 completed | 20/20 completed | Planar simulator |
| Mars base contact-retreat | 3/20 unsafe-force cutoffs | 3/20 unsafe-force cutoffs | Teacher approach, then tested retreat controller |
| Mars base approach-retreat | 8/20 unsafe-force cutoffs | 8/20 unsafe-force cutoffs | Tested controller handles approach as well |
| Stretch physical-v2 carry | 11/100 completed | 31/100 completed | Retention loss 49/100 vs 39/100 |
| Stretch physical-v2 place | 1/100 completed | 1/100 completed | Unsafe contact 6/100 vs 10/100 |

Sources: `logs/mars-arm/test-{teacher,learned}.json`,
`logs/mars-base-extended/test-{teacher,learned}/*-<scenario>.json`, and
`logs/manipulation/physical-v2-final-audit/{teacher,learned}.json`.
The latter audit's `status.json` reports an oracle-harness failure; it has no
`acceptance.json`. These Stretch results are diagnostic and do not establish a
promoted controller. Newer experiment directories exist; this is an explicitly
selected baseline snapshot, not an automatic selection of the latest policy.

All 3 contact-retreat unsafe episodes terminate before the retreat goal/event
and policy handoff. All 8 approach-retreat unsafe episodes also terminate before
the retreat event. Equal end-to-end rates therefore do not establish equal
learned and feedback retreat safety. Inspect `teacher_approach`, `event_count`,
and per-episode outcomes together. The report's internal scenario is
`contact_retreat` for both protocols; label the input with the external protocol.

The Mars planar base cannot represent wheel lift, roll, or pitch. The stationary
Stretch monitor evaluates simulated support, but that does not validate Mars
support or the physical robot. The experimental 30 N ceiling is not a measured
hardware damage or tipping threshold.

## Controller options, in recommended order

1. **Predictive command filter around every controller.** Start with a short
   rollout and braking check using measured position/velocity, tracking error,
   actuator limits, and inflated robot/payload geometry. Cover self-collision,
   arm-floor contact, base/arm swept motion, and stale perception. Include the
   stopping trajectory and sensing/control delay, not just the next endpoint.
   Run on a private state copy and initially record interventions in shadow mode.
   Put it after policy/reference generation and before actuator writes in
   `rl/server.py` and manipulation execution. Keep monitoring at feedback cadence;
   a filter called only when inference returns cannot protect against hung inference.
   [Predictive safety filters](https://arxiv.org/abs/1812.05506) provide a formal
   framework, but safety claims require validated uncertainty bounds and a feasible
   backup controller. A finite sampled rollout alone supplies no such guarantee.

2. **Consistent acceleration/jerk limits and explicit stop behavior.**
   `MarsBaseControl` bounds base command acceleration and `ManipulationControl`
   bounds arm target acceleration. Coupled `MarsControl` currently bounds speeds
   and target lead, with a 0.20 s obstruction latch, but has no equivalent
   acceleration limiter. Add coordinated retiming across base and arm, and
   separate ordinary deceleration, obstruction hold, and urgent protective stop.
   Test loaded holds so stopping does not drop the object or collapse a gravity
   loaded joint. [Ruckig](https://arxiv.org/abs/2105.04830) is an implementation
   option for jerk-limited synchronized references; it does not check collisions,
   measured impact force, or wheel stability.

3. **QP/CBF projection for coordinated avoidance.** Minimize deviation from the
   proposed base/arm command under joint stopping margins, velocity/acceleration
   bounds, collision-distance constraints, and the base's nonholonomic constraints.
   Match the optimization to the actual position/velocity servo interface.
   Define behavior for infeasibility, numerical failure, and missed deadlines;
   do not silently relax safety constraints to meet a task goal. A validated
   fallback must handle states where stopping alone cannot preserve safety.
   [CBF-QP](https://arxiv.org/abs/1609.06408),
   [sampled control with delays](https://arxiv.org/abs/2005.06418), and
   [mobile-manipulator whole-body avoidance](https://arxiv.org/abs/2409.14775)
   are useful references. Local avoidance can stall; route planning still matters.

4. **Contact-aware compliance and load limits.** Distinguish intended finger/object
   or placement contact from forbidden body contact. Add approach-speed, peak-force,
   sustained-load, and impulse limits by phase; use calibrated effort feedback or
   a validated observer where available. Tracking residuals alone cannot establish
   contact force. A generic position hold can keep pushing after a collision.
   Preserve the existing explicit release authorization and retention checks.

5. **Support-aware speed and reach envelope.** Obtain authoritative Mars support
   geometry and mass/COM/payload parameters; model vertical/roll/pitch dynamics and
   real wheel contacts. Derate acceleration, turning, and reach jointly with arm
   extension, payload, slope, and braking direction. Static COM projection is a
   diagnostic, not a dynamic tipping certificate. This model work can proceed
   alongside the command filter and is required to evaluate wheel-loss prevention.

Keep `manipulation/risk.py` outcome probabilities and `pose_value.py` rankings
advisory. Neither is a substitute for enforcing known physical constraints.

## Offline viewer

`emet.visualization.controller_safety` reads explicit episode reports and writes
standalone HTML with per-scenario outcome counts, 95% Wilson intervals,
force-versus-duration plots, and a seed selector exposing recorded measurements
and evaluation configuration. Inputs stay separate. Missing labels stay unknown;
failed initialization counts against task completion and is excluded from
executed safety denominators. Wheel-lift labels require explicit support evaluation.
There is no simulator launch, network service, or robot connection.

Reports with optional null metadata remain readable; malformed flag types and
nonfinite JSON values are rejected. Force-definition metadata is retained and
shown as unspecified when absent. Intervals assume independent trials: repeated
seed/case identities and missing identities are flagged without deleting rows
or claiming that record counts establish independent evidence.

Verified from the repository root using the existing environment:

```bash
PYTHONPATH=src .venv/bin/python -m emet.visualization.controller_safety \
  --report 'Mars feedback approach-retreat=logs/mars-base-extended/test-teacher/teacher-approach_retreat.json' \
  --report 'Mars learned approach-retreat=logs/mars-base-extended/test-learned/learned-approach_retreat.json' \
  --output logs/controller-safety-review/approach-retreat.html
```

The initial `logs/controller-safety-review/index.html` includes the 14 explicitly
selected input files underlying the table above: 22 cohorts and 1,400 episodes,
including five Stretch scenarios per controller. Local logs are not committed.

## Next visualization work

Extend the existing `rl/test_drive.py` for live exploration and
`visualization/rerun.py` for synchronized recorded inspection:

* **3D replay:** actual/reference robot, collision proxies and payload volume,
  nearest-point segments, contact normals/forces, support contacts and predicted
  braking motion. Show absent sensors or unsupported model outputs as unknown.
* **Linked traces:** scrub pose and timelines together for measured/commanded
  velocity, force/impulse, tracking error, target lead, joint stopping margins,
  wheel clearance/load, tilt, inference latency and state/perception age. Mark
  retarget, handoff, obstruction, filter intervention, and stop reason.
* **Capability atlas:** bin workspace position with separate orientation, payload,
  speed, arm extension and obstacle-clearance slices. Show task completion,
  unsafe outcomes, trials and uncertainty separately. Include rejected starts and
  unsampled bins; avoid interpolating untested regions into apparent safe space.
* **Paired failure comparison:** load the exact same initial snapshot, scene,
  disturbance schedule and time budget for baseline and filtered control. Inspect
  whether an intervention prevented a violation or merely caused a timeout.

Log monotonic timestamps, controller/model/config hashes, seed and complete initial
state, references, measured state, proposed/applied commands, hazard margins and
stop reasons. Episode summaries cannot reconstruct these traces. Prioritize thin
obstacles between ticks, combined loaded base/arm braking, arm-floor levering,
contact before handoff, retarget near limits, dropped/stale feedback, inference
overrun, and failed initialization in the regression battery. Report worst-case
and tail severity plus completion; keep component and whole-body evidence separate.
