# Active Mars controller work

Updated 2026-11-16. This is the operational index for ownership, branches,
validation and the next milestone. The
[technical next-steps plan](../plans/mars_wbc_next_steps.md) owns the detailed
implementation sequence; experiment reports own measured results. Older
snapshots and threads are historical evidence, not competing current plans.

## One controller owner and development worktree

Continue controller implementation, training decisions and progress updates in
**Mars controller — canonical implementation, training and TAMP integration**,
thread `01a1373d-ef6f-7a83-a81f-ac02e6edd5c0`.

Use the existing worktree:

```text
/home/cpaxton/src/home_robot_v4/.worktrees/wbc-grasp-primitive
branch: feat/wbc-grasp-primitive
```

The root checkout is the older `review-214` branch with pre-existing uncommitted
safety-viewer/package/documentation work. It is not the development or training
checkout. Those edits, historical worktrees, actor weights, raw logs and source
archives are preserved. Do not reset, clean or stage the root checkout to make
the controller branches look clean.

## Linear review stack

| Order | PR / branch | Scope |
| --- | --- | --- |
| 1 | [#239](https://github.com/cpaxton/home_robot_v3/pull/239), `feat/wbc-jerk-measurements` → `main` | Current-COM feedback, bounded nonfault hold reacquisition, task diagnostics and semantic failure reporting. Ready for review. |
| 2 | [#240](https://github.com/cpaxton/home_robot_v3/pull/240), `feat/wbc-directional-curriculum` → #239 | Seventeen prepared direction cases, twenty-four audit cases and strict loader. Ready for review; not admitted or trained. |
| 3 | [#241](https://github.com/cpaxton/home_robot_v3/pull/241), `feat/wbc-grasp-primitive` → #240 | Unloaded grasp-pose execution boundary, teacher admission, and this consolidated development line. Draft while stage admission is blocked. |

#241 includes both parent branches; use it for the next controller change.
Keep each PR's diff scoped to its parent. When a parent is squash-merged,
restack descendants semantically and verify that already-merged guards/tests
are preserved. Do not resolve a stack conflict by reverting the parent's newer
files. No PR is merged or actor promoted by this consolidation.

## Current state and evidence

**No PPO run is active.** Recent progress repaired execution and expanded the
unloaded primitive boundary with unchanged retained policy weights. No full
pick-and-place or hardware controller is qualified.

| Evidence | Scope / result |
| --- | --- |
| [Task-stage diagnosis](../experiments/mars_task_arrival_20261115.md) | Same actor reaches declared grasp region and executor closure after one bounded hold reacquisition. Lift support reserve, loaded retention and fixed-world-hand/base coordination remain gaps. Twelve development attempts are not independent reliability trials. |
| [Unloaded-stage admission](../experiments/mars_unloaded_grasp_admission_20261116.md) | Strict original contracts: teacher 5/6 at both pregrasp and grasp-pose boundaries, all twelve stops confirmed, no closure/loaded commands. Both stage admissions remain false. |
| #239 runtime `2cbf70d8` | 1,330 passing source-bound canonical regressions, verified against that runtime. |
| Pre-consolidation #241 runtime `9cd439bf` | 1,367 passing source-bound canonical regressions, exactly bound to the twelve-episode teacher source; four offline readout tests pass. |
| Combined runtime `c2d82bb7` | **1,389/1,389 passing canonical regressions**, zero failures/errors/skips, 744.87 s. Fresh receipt verified against the consolidated source; [compact validation record](../../paper/data/whole_body_control/mars_controller_consolidation_v1/validation.json). Parent receipts are not reused. |

The combined check ran as managed job `20261116_014840_fbac92` on CPU15.
Its full XML/receipt remain ignored under
`logs/eval/mars-controller-unification-v1/`; the compact record binds their
hashes and the preserved source archive. This is software validation only,
with no additional teacher episodes, PPO steps or hardware use.

Keep EE pose-tracking rotation distinct from object rotation relative to the
hand. Successful teacher unloaded grasp errors are 0.000295–0.001423 rad;
loaded retention failures cross the separate 0.1 rad object-rotation limit.
Rotvec features exist but have not received a trained ablation; the old C1
quarter-pitch experiment does not establish broad direction coverage.

## Current implementation awaiting fresh readouts

The explicit native unloaded planning scope is implemented, retaining the
historical full-task default. Coupled Mars/Stretch reference generation,
geometry admission, teacher readout/plots and a PPO-compatible general environment
are implemented in [the current run instructions](../plans/general_whole_body_tracking.md#coupled-simulator-implementation-2026-11-16).
Focused tests cover native scope isolation, both embodiments, mask invariance,
private-witness geometry and grouped PPO collection/update without critic leakage.
Fresh full-source regressions and physical readouts are required for this new
runtime; the 1,389-test consolidation receipt above is historical. No new policy
is trained or promoted by implementing these interfaces. Preserve old failures.

## Next milestone, in order

1. **Scope unloaded planning to requested motion.** The current full-task
   executor replans a prospective loaded lift before either unloaded boundary.
   Its support rejection blocks one case even though lift is not requested.
   Add explicit unloaded-prefix planning/admission at the TAMP/reference seam.
   Preserve collision, support and geometry checks for every requested segment,
   default full-task loaded validation, closure/release authority and all frozen
   failures. Do not fabricate a lift witness, drop the case or borrow the old
   expected-failure exception. Obtain a fresh stage-specific readout before PPO.
2. **Add reachable coupled whole-body coverage for Mars and Stretch.** The
   teacher can execute a 20 mm forward preparation while holding the world hand; the actor advances
   about 2.15 mm. Include this motion, actual unloaded grasp trajectories,
   hand-only goals with authorized free-base motion, jointly feasible masked
   base/hand targets, ordinary reach/base goals, goal changes and reset/hold
   transitions. Sample/admit full-robot paths, not only stationary-arm poses. Keep
   #240's frozen train/audit identities separate from new coverage.
3. **Register and run bounded mixed training only after admission.** Preserve
   the outer 50/50 general/task exposure by physical control ticks, report
   per-stage exposure and compare unchanged initial/final checkpoints with the
   teacher on identical cases. Keep reward/representation ablations isolated,
   retain smoothness/force/support/stop checks, log source/config/seed/case
   identities and outcomes to W&B, and do not select checkpoints post hoc.
4. **Qualify loaded progression separately.** Plan support reserve and examine
   grasp fit/wrench stability and angular creep using the existing evidence.
   Do not increase torque/friction, relax retention or replay closed-loop holds
   as a substitute for evidence. Herman remains observation-only until a
   separately authorized, qualified hardware test.

Coupled control is the framework for both first robots; limited stationary-arm
DoF does not establish failure of general whole-body control. The
[embodiment comparison plan](../plans/general_whole_body_tracking.md#coupled-control-and-embodiment-comparison)
records MABEL as an optional, unintegrated redundant-robot diagnostic if model,
reference and learning investigations leave a suspected embodiment limitation.
This does not displace the unloaded-prefix fix or authorize another training run.

The controller owns smooth reactive pose/trajectory tracking. TAMP owns task
choice, feasible stance/reference construction and planning scope. Neither
side should conceal the other's failures.

## Partner work and archived threads

The TAMP benchmark/agent effort in thread
`01a0ddca-cf1f-7a43-ab7f-30a48ba6dcd0` continues its already-running validation
in `/home/cpaxton/src/home_robot_v2/.worktrees/tamp-benchmark-runtime`.
Its completed handoff is [PR #242](https://github.com/cpaxton/home_robot_v3/pull/242),
branch `feat/tamp-native-base-approach`, head `855348a2`, stacked on #234.
It reports privileged native 8/8 tasks and 18/18 transfers, actual-agent 4/4
tasks and 10/10 transfers, four rejection checks and 407 affected tests.
It is a planning/agent validation partner, not a second controller-training
owner. Reuse its supported planning interfaces where appropriate; coordinate
changes to shared TAMP code before editing. Its native/agent results are not
learned-policy or Mars hardware qualification. Its completed benchmark package is separate from the controller review stack;
do not infer its branch from old #231 or merge planner changes without semantic
review. Confirm any new job reservations at the next handoff.

These completed overlapping conversations were preserved and archived:

| Thread | Handoff |
| --- | --- |
| `01a1d761-e1a7-7780-9155-35fa16093810` | Feedforward/representation and #241 unloaded-boundary work; clean at `f78ef902`, no jobs or reservations. Latest orientation findings and next prefix-planning scope are reflected above. |
| `01a1b669-9813-7f70-9e37-4e3c7ac104e2` | Older controller research/coordination and completed diagnostic handoffs; evidence retained in current reports and historical TODO. |
| `01a1da91-9b35-75b1-96ec-3e1c0f7da2fa` | Completed #227/#233 review: bundled reward discounting and repeated-trial confidence findings. Both PRs are merged; historical review remains recoverable. |

Unrelated game, graph/agent and other-robot conversations are unchanged.

## Resource and launch rules

One controller native worker on CPU14 and at most one test/offline worker on
CPU15; one numerical thread each, CUDA hidden. Disable ordinary rendering;
the existing canonical renderer-test exception may use EGL. Coordinate any
overlap with the TAMP partner, whose current native runner uses CPU0–7.

Use immutable source snapshots and managed job records. Verify fresh source
receipts instead of trusting stale XML. Report actual job status; a prepared
manifest, queued launch or passing software-only PPO test is not training.
Avoid environment upgrades on this unstable machine. Keep raw compressed
traces and weights ignored, and retain source/archive hashes in compact reports.
Only remove owned completed duplicate runtimes after verifying job completion
and preserving source archives/logs. Do not kill unrelated or unidentified jobs.
