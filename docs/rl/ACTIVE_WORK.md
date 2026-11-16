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

## Review stack after parent merges

| Order | PR / branch | Scope |
| --- | --- | --- |
| 1 | [#239](https://github.com/cpaxton/home_robot_v3/pull/239), `feat/wbc-jerk-measurements` → `main` | Current-COM feedback, bounded nonfault hold reacquisition, task diagnostics and semantic failure reporting. Merged to main. |
| 2 | [#240](https://github.com/cpaxton/home_robot_v3/pull/240), `feat/wbc-directional-curriculum` → main | Seventeen prepared direction cases, twenty-four audit cases and strict loader. Merged to main; that separate suite remains untrained. |
| 3 | [#241](https://github.com/cpaxton/home_robot_v3/pull/241), `feat/wbc-grasp-primitive` → main | Unloaded grasp-pose execution boundary, teacher admission, and this consolidated development line. Draft while learned-policy stability and task progression remain incomplete. |

#239 and #240 were squash-merged on 2026-11-16. #241 now targets main, which
is merged into the canonical branch with an unchanged controller source tree.
Use #241 for the next controller change. Keep its diff scoped to current main. When a parent is squash-merged,
restack descendants semantically and verify that already-merged guards/tests
are preserved. Do not resolve a stack conflict by reverting the parent's newer
files. Parent merges do not promote an actor or qualify hardware.

## Current state and evidence

**No PPO run is active.** The 32,768-decision joint fine-tuning run completes
with scores unchanged: v1 9/10, v2 5/10, fresh base audits 0/16 and TAMP 6/6.
It is not promoted. The earlier retained checkpoint remains the control anchor.
The fixed-weight signed-feedback intervention passes 24/24 base trajectory
cases and is implemented as an explicit opt-in actor parameterization. The safely versioned v3 export at source `823b3d1d` has a complete native
readout and **1,451 passing source-bound regressions** (990.82 s). The paper
builds successfully. Current work proceeds toward static-goal references and
broader coupled trajectory training. It is not a learned-policy or
hardware qualification.
The earlier 4,096-decision pilot and both 1,024-update imitation diagnostics are
complete. Static lateral endpoint convergence and broader coupled coverage remain gaps. No full
pick-and-place or hardware controller is qualified.

Earlier endpoint source **`0a682deb`** adds a continuous rest-to-rest base
endpoint compiler and opt-in reference feedforward/XY-region/heading calibration.
Its complete native readout preserves **24/24 base paths, 10/10 small general,
6/10 larger general, 6/6 unloaded TAMP and 4/4 moving handoffs**. Raw static goals
remain **2/8**; compiled development endpoints pass **8/8** and fresh compiled
audits **16/16**. Fresh source-bound regressions pass **1,480/1,480**, with a
verified canonical receipt; the paper builds (59 pages), and evaluation is
logged to offline W&B `ybaxsdnt`. Measured jerk and moving-start retarget continuity
remain explicit gaps. No UI/service default or checkpoint is promoted. See the
[endpoint report](../experiments/base_goal_reference_20261116.md).

Current implementation source **`13c53634`** adds motion-matched base replans,
past-preview continuity and atomic opt-in viewer goal batching. Native source
`e30a7dfc` preserves the old 86-case cohorts and passes 10/12 new moving cases
and 3/3 actual viewer/public-bundle cases. Two reverse caster events reproduce
without retargeting. Global gain zero resolves those reverse replays but
regresses coupled accuracy, so it is rejected as a global replacement. The
axis/mask-scoped diagnostic passes 15/15; it remains unsupported for deployment.
Final source-bound regressions pass 1,520/1,520, the cold viewer passes and the
paper builds (60 pages); no PPO is active. See the
[replan/servo report](../experiments/base_goal_replan_20261116.md).

| Evidence | Scope / result |
| --- | --- |
| [Current complete readout](../experiments/coupled_tracking_20261116.md) | Explicit unloaded-prefix teacher 6/6 at pregrasp and 6/6 at open-jaw grasp-pose; Mars and Stretch coupled v1 teachers each 10/10. |
| `db6b6964` full regressions | 1,421 passing source-bound tests, zero failures/errors/skips, 863.76 s; exact producer of those native readouts. |
| Direct PPO pilot | 4,096 decisions / 4,314 physical ticks, approximately 53/47 general/TAMP. General 0/10 → 0/10, grasp-pose 0/6 → 0/6. General case/phase exposure is incomplete. |
| Teacher-action bootstrap | 1,024 supervised updates, train labels only. Actor-only TAMP grasp-pose 5/6; raw general 0/10 with settling/self-contact failures. |
| Fixed-weight base-rate ablation | Full-horizon general 9/10 versus raw 0/10; arm rate feedback retained. Base-only becomes worse; not a promoted replacement. |
| Explicit completion-hold diagnostic | 10/10 using existing executor hold/reacquisition. Keep separate from raw actor stability. |
| Pre-navigation runtime `57e8e73b` | New v2 coarse base acceptance, unchanged hand precision, serialized opt-in base feedback mode. Physical readouts: teacher Mars 8/10, Stretch 9/10; fixed-weight Mars actor v1 9/10, v2 5/10, original TAMP grasp-pose 6/6. Fresh current-source regressions pass 1,433/1,433 in 913.38 s; paper builds (59 pages). Old receipts do not qualify this source. |

Retained baseline runtime **`823b3d1d`**: base trajectories **24/24**, small general
**10/10**, larger general **6/10**, TAMP grasp-pose **6/6**, moving handoffs
**4/4**, and static endpoint goals **2/8**. Fresh source-bound tests are
**1,451/1,451**; v3 loader compatibility and paper build pass.

The [base-direction diagnostic](../experiments/base_direction_bootstrap_20261116.md)
identifies wrong-sign base-only yaw correction. Eight new training references
and six unchanged TAMP references pass teacher admission. Output-only fitting
regresses to v1 4/10, v2 2/10 and TAMP 1/6, with fresh base audits still 0/16;
it is rejected. The retained initializer remains v1 9/10, v2 5/10, TAMP 6/6.
The fresh base-audit teacher is 15/16; its heading failure remains counted.

The preceding [unloaded-stage report](../experiments/mars_unloaded_grasp_admission_20261116.md)
remains historical 5/6 at both boundaries, under full-task planning scope.
Consolidation runtime `c2d82bb7` remains historical 1,389/1,389 software evidence.
Neither is silently relabeled as the new prefix/coupled result. Source archives,
failed setup attempts, all case outcomes and offline W&B runs are retained.

Keep hand pose rotation distinct from object rotation relative to the hand.
The failed bootstrap audit-right case is `object_changed_replan`, not a successful
grasp. Open-jaw pose operations do not establish loaded retention.

## Next milestone, in order

1. **Complete current readouts.** Preserve v1 and frozen TAMP contracts. V2 uses
   a 5 cm / 5° general base region, 15 cm required base travel and unchanged
   3 mm / 0.02 rad hand targets. Explicitly label small nullspace motion as
   hand-hold coverage. Compare teacher and actor without pooling repeated seeds.
2. **Fix base-only stability and final approach.** The learned base's direct
   rate path causes substantial hunting; disabling it helps nine hand/coupled
   cases but harms base-only. Retain native servo limits and reactive arm
   compensation. The opt-in signed/reference-tracking modes now pass the
   existing paths and rest-to-rest endpoint audit. Moving replans now have
   complete readouts. Next implement and validate a supported per-axis feedback contract and diagnose measured jerk, while
   retaining the larger coupled hand misses and contact stops as failures.
3. **Improve training exposure, then run a bounded comparison.** Preserve outer
   50/50 general/task physical ticks, train/audit separation and fixed-final
   evaluation. Use the [SuperMimic handoff](../plans/supermimic_controller_training.md)
   to design physically valid phase/reset sampling; that sampler and a learned
   dynamics backend are not yet implemented. Log per-phase/case exposure.
4. **Qualify loaded progression separately.** Repair support reserve, grasp
   fit/retention and continuous grasp/lift transitions before broader agent
   tasks. Do not increase torque/friction, relax retention or clear stop latches.
   Herman remains observation-only pending separate hardware qualification.

The current prefix endpoint is terminal and needs simulator reset; continuous
prefix-to-loaded handoff remains work. Stretch static support includes its
caster, but dynamic unloading metrics currently cover only its drive wheels.

Coupled control remains the framework for both first robots. The
[embodiment comparison plan](../plans/general_whole_body_tracking.md#coupled-control-and-embodiment-comparison)
keeps MABEL as an optional, unintegrated diagnostic. TAMP owns task choice and
feasible reference construction; the controller owns smooth reactive tracking.

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

The signed-feedback prototype also passes 10/10 small general, 6/10 larger
general, 6/6 TAMP and 4/4 live base-to-hand handoffs. Static endpoint goals remain
2/8 despite reaching the XY region. See the
[complete direction/formulation report](../experiments/base_direction_bootstrap_20261116.md).
Completed jobs: native `20261116_122228_f4f79a` and regressions
`20261116_122228_e25e20`, immutable runtime `/tmp/emet-base-navigation-823b3d1d`.
The previous source's canceled partial suite is not reused.

The retained reference-tracking v3 test candidate is
`logs/eval/base-goal-reference-native-v1/actor` (gain 0.5). The global gain-zero
candidate at `logs/eval/base-goal-replan-feedback0-v1/actor` is a rejected
diagnostic replacement; do not switch service/UI defaults to it. No actor or
hardware interface is promoted. Final source runtime is
`/tmp/emet-base-goal-replan-13c53634`, with its verified compressed source archive
under ignored `logs/recovery/`. Completed duplicate runtimes `0a682deb`,
`adc5d168` and `e30a7dfc` were pruned after process/job/archive verification;
their archives, logs and results remain intact.

Completed jobs: native gain-zero `20261116_151955_512625`, full regressions
`20261116_151952_bd45a2`, scoped diagnostic `20261116_153146_db79d9` and final
verification/evidence `20261116_153518_be7428`. The only live controller UI is
job `20261116_154539_54ba7e` on CPU14, localhost port8766, retained gain-0.5
checkpoint root. Reset a finished/faulted scene before selecting the actor;
accepted goal edits then resume automatically. The completed offline stopping
check `20261116_155434_94880d` confirms both failed reverse cases stop after
0.15 s of physical stepping; original wheel failures remain counted. The
superseded partial regression run is cancelled and its verified orphan group terminated; its XML is never reused.
