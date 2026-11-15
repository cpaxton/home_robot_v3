# Current-COM feedback: actual reset and frozen-controller validation

This is step 1 of [the ordered WBC plan](mars_wbc_next_steps.md), after the
matched-reset [diagnosis](../experiments/mars_aligned_jerk_measurements_20261115.md).
No PPO, policy export, hardware motion, reward/gate revision or default change.
Integrate reviewed main before freezing executable sources and running fresh
full source-bound regressions on CPU15. Native validation uses CPU14, one
numerical thread, CUDA hidden and no MuJoCo rendering.

## Fixed first batch: 24 episodes

`scripts/validate_mars_current_feedback.py --phase loop` crosses two actual
reset configurations (cached/current COM) with twelve existing conditions:

- Five training base primitives: hold, forward, reverse, yaw-left and yaw-right.
- The same five with a declared world +X end-effector push: 0.5 N, no torque,
  six control ticks beginning at tick10, using the existing two-tick rise/fall
  envelope. No random force direction, altered inertia or friction.
- Mid-motion cancellation at tick20 for base-forward and quarter-pitch-up.

Use original seeds, references, geometry and limits. Rename condition IDs
explicitly; these are paired diagnostic cases, not independent random trials.
Enable the option in the environment configuration before initialization and
verify that it survives each reset. Record all native substeps and nominal
scores before stop verification. End the finite external test wrench before
testing a stop; gravity and contact remain live. Never claim sustained-force
stopping from that check. Stop verification retains the existing measured arm
and all-six-base-DOF limits, with at most0.75 simulated seconds and no reset,
rearming or grip release. Record failure, force/support flags and all stop traces.

Broad validation requires all twelve current-COM admissions and stops, no
unsafe-contact/wheel-lift/invalid-physics result, and successful nominal base
tracking. Disturbance tracking is reported separately. Preserve every raw
attempt; never overwrite an output directory or relax a guard after a failure.

## Fixed second batch: 364 episodes

`--phase baseline --loop-report PATH` requires the matching completed loop
producer, script and canonical condition manifests. It enables current-COM
feedback through the actual initialization path for every evaluated environment.

| Frozen controller | General population | Actual TAMP pregrasp primitives |
| --- | ---: | ---: |
| Analytic teacher | Existing164 admission cases | Existing6 development/audit cases |
| Retained legacy actor | Same164 | Same6 |
| Frozen GRU control | Existing12 C1 evaluation cases | None |
| Frozen GRU candidate | Same12 | None |

Keep every actor-weight checksum and its original bundle environment metadata.
Report the explicit environment-option override without retagging/exporting a
bundle. Use the real public-tool primitive environment; closure/pick/place
remain out of scope. Reconstruct and verify the bundle's original full-scene
fingerprint, then use the existing conservative compiled-robot identity for
TAMP scene transfer before inference. An extra fixed support may differ; robot
geometry, dynamics, actuators, couplings and solver settings must match.
Teacher/legacy TAMP reset hashes must match wherever both initialize.

The teacher's current-source report is fresh experimental admission, preserving
all six TAMP denominators and the known object-motion challenge's new measured
outcome. It does not become an archived teacher report by renaming its hash.
Score all actors against fresh teacher-relative budgets and also against the
original `3ac722e1` teacher budget matrix as a separately labeled comparison.
New reset/reference hashes are not claimed identical to old-plant hashes.
Retain full per-case tracking, orientation, drift, jerk, contact/support flags,
measured stops and TAMP failure codes. Save completed general cases atomically
every ten episodes so a machine crash preserves partial evidence without
claiming a complete run.

Fresh full regressions and the above validation are prerequisites for deciding
whether to run the separately declared direction-coverage curriculum pair.
The strongest frozen actor must be selected from these results; no assumption
that the GRU wins, no automatic PPO launch and no implicit budget extension.
The existing238/239 results and their producer identities stay unchanged.

The first source (`24c14b27`) completed its 24-case loop check but its broad run
and full regression run were interrupted after identifying an overly strict
new adapter check against the whole-scene fingerprint. This would have rejected
the second fixed TAMP support despite identical robots. Retain their partial
evidence under `mars-current-feedback-validation-v1`; restart both phases and
checks under a fresh v2 output root with the compiled-robot compatibility guard.
Do not relabel partial results as complete or reuse their receipts.

## Next decision after the completed v2 baseline:24 TAMP episodes

The completed corrected-reset readout gives retained legacy12/12 on the common
C1 cases and151/164 general, versus GRU candidate8/12. Retained legacy still
passes0/6 actual pregrasps, teacher5/6; the known case now reports a different
`operation_failed`/lift-stage failure and does not pass the experimental
admission gate. Do not train past this unclassified change or infer TAMP ability
from the small general benchmark.

Predeclare one further batch,4groups×the same6cases=24episodes:

| Group | Existing frozen weights | Extra physical settling before planning |
| --- | --- | ---: |
| Teacher | None | 2 s |
| Retained legacy | Original `e7b42569…` | 2 s |
| Velocity-feature legacy final | Frozen arm comparison `38c291b8…` | 0 s |
| GRU candidate | Original exploration candidate | 0 s |

Keep the completed zero-settle teacher/legacy runs as anchors. Settling uses the
existing development-only API before task discovery/planning: no teleport,
grip closure or actor inference. Goals, thresholds, limits and geometry remain
unchanged; this is a changed initialization condition, never a retagged frozen
acceptance result. The worker's initial-state hash precedes optional settling;
do not call it a post-settling identity.

Use `scripts/probe_mars_tamp_tracking.py`, verify the existing same-runtime
full regression receipt, and preserve raw outcomes after every case. The
velocity final is bound to the existing checked-in frozen-arm report and exact
weight SHA; no fitting, new checkpoint choice or adaptive extension. Preserve
every failure/stop and compare actual TAMP outcomes across controller families
before deciding whether to add trajectory/phase exposure or change an objective.
