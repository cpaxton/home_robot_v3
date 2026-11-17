# Active Mars controller work

Updated 2026-11-17. This is the operational index for ownership, branches,
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

**Architecture/critic audit and corrected PPO are complete (2026-11-17).**
The [architecture review](WHOLE_BODY_ARCHITECTURE_REVIEW.md) keeps coupled
pose inputs/history and fixes a concrete critic-target mismatch: finite tracking
deadlines bootstrapped physically stopped states. The opt-in independent critic
budget also separates value fitting from actor KL stopping. Frozen source
`2acc3b52` passes **1,574 source-bound tests** and both corrected arms complete
16,384 decisions. Retained/joint/independent are **20/24, 19/24, 20/24** on new
cases, **6/10, 6/10, 7/10** on existing coupled and **6/6** TAMP prefixes each.
All 120 actor stops confirm; known contact identities persist, with no wheel lift.
Independent critic fitting reduces worst active-hand error but increases worst
yaw jerk. Neither candidate is promoted. Exact results and the preserved failed
first preflight are in the [completed report](../experiments/finite_critic_pilot_20261117.md).

Retained is a comparison anchor, not a satisfactory controller. A frozen-actor
probe finds stationary-reference exploration is a concrete remaining obstacle:
0/8 → 3/8 stochastic successes when motion noise falls during announced holds.
The mean commands and gripper noise are unchanged. Implement and test this as
an optional training distribution, then run a fixed-budget pair with independent
critics and unchanged rewards/guards. Source changes require fresh regression
and teacher admission; the completed 2acc receipt cannot qualify them.

**Next training pair is queued at frozen source `9d0620f9`.** The optional
stationary-reference motion-noise distribution passes 99 focused checks. Full
gate `20261117_015047_2d92b1` and fresh native admission
`20261117_015047_4e1b10` precede PPO supervisor `20261117_015136_781d98`.
Both arms use independent critics, original rewards and fixed 16,384-decision
budgets; only hold exploration differs. At launch PPO has not started. The
viewer is paused for the sole native worker and restored automatically after
completion/failure. [Protocol/live-log locations](../experiments/quiet_reference_training_20261117.md).

**Wrapper reward work:** the [reward specification](../experiments/wrapper_reward_20261116.md)
records the legacy 20 Hz objective and an opt-in, versioned contract-tracking
profile shared by general/coupled/TAMP wrappers. It adds fine hand priority,
pose-gated velocity credit and measured progress against the same active goal.
Legacy defaults remain unchanged. The bounded paired reward pilot below is
complete with independent coupled/mask cases and the existing 50/50 general/TAMP
physical-tick mixture. Older runtime receipts are historical; the fresh
1,563-test receipt qualifies the pilot source.


**Reward PPO pilot is complete; neither candidate is promoted.** Frozen runtime
`5f0eb3c2` passes 1,563 fresh complete regressions. Both reward arms train for
16,384 decisions with fresh critics and matched initial states/first actions.
Retained/new-reward/legacy results are **20/24 → 18/24 → 19/24** on new cases,
**6/10 → 6/10 → 7/10** on existing coupled cases, and **6/6** TAMP prefixes for
all three. All 120 actor stops confirm; three existing contact-failure identities
remain per controller, and some jerk/contact maxima worsen. Teacher is 12/12
training and 11/12 audit. Legacy resolves the old reverse 10.09 mm false
equilibrium to 0.0154 mm but regresses other cases. This is a tradeoff, not a
general controller improvement. Keep the retained actor. That historical PPO pair is complete. Subsequent critic and exploration work
is tracked above; its viewer/job state supersedes this older snapshot. See the
[completed paired report](../experiments/wrapper_reward_training_20261116.md).

**Earlier PPO:** The 32,768-decision joint fine-tuning run completes
with scores unchanged: v1 9/10, v2 5/10, fresh base audits 0/16 and TAMP 6/6.
It is not promoted. The earlier retained checkpoint remains the control anchor.
The fixed-weight signed-feedback intervention passes 24/24 base trajectory
cases and is implemented as an explicit opt-in actor parameterization. The safely versioned v3 export at source `823b3d1d` has a complete native
readout and **1,451 passing source-bound regressions** (990.82 s). The paper
builds successfully. Subsequent endpoint, moving-goal and scoped-feedback results below supersede
that milestone. It is not a learned-policy or
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

Previous implementation source **`13c53634`** adds motion-matched base replans,
past-preview continuity and atomic opt-in viewer goal batching. Native source
`e30a7dfc` preserves the old 86-case cohorts and passes 10/12 new moving cases
and 3/3 actual viewer/public-bundle cases. Two reverse caster events reproduce
without retargeting. Global gain zero resolves those reverse replays but
regresses coupled accuracy, so it is rejected as a global replacement. The
axis/mask-scoped diagnostic passes 15/15; it remains unsupported for deployment.
Final source-bound regressions pass 1,520/1,520, the cold viewer passes and the
paper builds (60 pages); no PPO is active. See the
[replan/servo report](../experiments/base_goal_replan_20261116.md).

Current source **`4b3c4059`** implements the opt-in v4 scoped execution contract.
Fresh source-bound regressions pass **1,538/1,538**; cold v4 viewer checks pass.
The matched 113-case-per-contract readout improves moving goals **10/12 → 12/12**
and preserves base 24/24, compiled endpoints 24/24, small coupled 10/10,
larger coupled 6/10, unloaded TAMP 6/6 and original handoffs 4/4. New repeated
mask handoffs remain 2/4. All 226 completed cases confirm a stop. Moving linear
jerk worsens despite improved yaw jerk, so no default/hardware promotion. No PPO
is active. See the [scoped-feedback report](../experiments/scoped_wheel_feedback_20261116.md).

The [hand-accuracy deep dive](../experiments/hand_tracking_accuracy_20261116.md)
completes **24 diagnostic episodes**, all stops confirmed, with unchanged source
and actor tensors. All nine actor replays reproduce the prior primary metrics
exactly. Four quiet misses are false equilibria: opposing learned base/hand
feedback cancels while the hand remains wrong. Native actuator lag, force limits
and joint limits do not explain them. Teacher and actor also share two modeled
link3/base self-contact failures despite safe nominal witnesses. Fixed gain and
feature probes are not exported: stronger hand feedback improves final error
but worsens some transient error/jerk, while removing base terms creates a new
contact stop. The current mixed wrappers replace native reward with a fixed
RewardMeter; tune the objective the learner actually receives before more PPO.

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

Completed jobs: earlier gain-zero `20261116_151955_512625`, earlier full
regressions `20261116_151952_bd45a2`, scoped diagnostic
`20261116_153146_db79d9`, and prior verification/evidence `20261116_153518_be7428`.
The stopping check `20261116_155434_94880d` confirms both failed reverse cases
stop after 0.15 s of physical stepping; wheel failures remain counted.

Current immutable runtime: `/tmp/emet-scoped-wheel-feedback-4b3c4059`, with
3,504 checksummed inputs and a compressed source archive. Completed jobs:
full gate `20261116_164218_a4b9d1`, native completion `20261116_164653_17158b`
and finalizer `20261116_164739_e113e5`. Offline W&B: `w2ld2dmi`. Failed partial
attempts and producer corrections are preserved in the experiment report;
source bytes and physical contracts remain unchanged. The snapshot's local
Git view pins the frozen revision for report tests.

The v4 candidate is `logs/eval/scoped-wheel-feedback-v2/candidate/actor`.
It is an experimental simulator option, not a promoted replacement for the
retained v3 actor. Keep known larger-coupled/mask failures and the measured
linear-jerk regression visible. Next diagnose that motion and generate related
training trajectories while preserving ordinary tracking and the outer mix.
Per-plant agent admission, loaded manipulation and Herman remain unqualified.

Disk cleanup removed disposable package cache and four completed duplicate
runtimes after checking archives, Git references and absence of active jobs.
Generated extras were archived separately. Checkpoints, traces, original source
archives, models, datasets and unrelated work are preserved. Compatibility
runtimes `823b3d1d`, `57e8e73b` and `13c53634` remain for reproducibility.

A single simulator viewer is running as `20261116_170847_aa8444` on CPU14,
localhost8766, exposing the retained v3 and experimental v4 roots for explicit
selection. The earlier viewer was stopped during native evaluation. Reset a finished/faulted scene
before selecting an actor. Goal edits then resume automatically. No hardware.

Completed hand-audit job: `20261116_175205_240a6d`, one offline worker on CPU15;
static private-geometry checks add no rollouts. Offline W&B: `bjvggd38`. No workers remain active for training/evaluation; only the
existing CPU14 viewer remains. The next implementation is explicit wrapper
reward/contract alignment, analogous non-audit coupled/mask training cases and
bounded direct-PPO mastery with trajectory/force/jerk comparisons. Preserve
ordinary tracking and outer 50/50 general/TAMP physical ticks.
