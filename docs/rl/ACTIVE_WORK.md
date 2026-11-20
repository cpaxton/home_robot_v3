# Active Mars controller work

Updated 2026-11-18. This is the operational index for ownership, branches,
validation and the next milestone. The
[technical next-steps plan](../plans/mars_wbc_next_steps.md) owns the detailed
implementation sequence; experiment reports own measured results. Older
snapshots and threads are historical evidence, not competing current plans.

The maintained [training mixture](TRAINING_MIXTURE.md) distinguishes the actual
completed 40/10/50 hand-active/base/TAMP tick split from the completed 30/10/10/50
completed split adding goal transitions. [Hardware readiness](MARS_HARDWARE_READINESS.md) keeps
robot-side feedback/executor/stop work separate from policy progress.

**Latest decision (Nov 18):** corrective goal labels substantially improve arm
spline tracking, but do not yet yield a general controller. Fresh confirmation
is **19/32 actor vs 31/32 teacher**, with one actor contact stop; all 110 stops
across that spline/general/TAMP check confirm. Balanced fitting improves hand
and goal-change regressions to 14/16 each, but worsens base/spline behavior and
leaves two stops unconfirmed. A gain-only follow-up preserves the retained
actor's declared base feedback. No new PPO budget or hardware motion is enabled.
Details and the unchanged physical-tick mixture are below.

**Herman passive audit is current:** reachable, Zenoh/domain 0, arm/odom ~200/30 Hz.
The active robot source retains the known velocity/failed-read bugs; a patch for
its exact current revision passes offline C++ checks and an isolated native
ROS build, but is not deployed or executed.
Current reported joint posture is outside the recent small training neighborhood.
No motion, driver/service startup or active software deployment. [Audit](../experiments/herman_feedback_audit_20261117.md).

**Captured-scene rehearsal is complete and visible on localhost:8771.** Actual
parked posture is blocked by two published-limit discrepancies and a conservative
link3/chassis proxy overlap; no physics rollout or hardware motion from that
state. A separate explicitly synthetic staged start admits all six nearby cases:
teacher **6/6**, both actors **0/6**, all **18/18** simulator stops confirmed.
The new actor has sub-1.6 mm hand errors in four arm-active cases but does not
settle; both yaw cases exceed the measured translation boundary. Partial room
geometry and camera calibration are observed, not a physical dynamics fit.
[Report, viewer and next steps](../experiments/herman_rehearsal_20261117.md).

**18:19 follow-up and wave preflight:** Herman's reported folded posture and
feedback-source defects are unchanged; stereo/TF are live, wrist remains absent,
no WBC bridge listens. A tiny base-held wrist wave admits from a separately
synthetic start: teacher 1/1, both actors 0/1 (translation boundary at 4.5 s), all
3/3 simulator stops confirmed. Fifty focused regressions pass. The restored
8771 wave replay and 8770 passive console are transient user services. No motor
commands or robot deployment. [Report](../experiments/herman_wave_20261117.md).

**Visible arm-wave correction:** the previous case was only a small wrist roll.
The checked v2 video now shows forearm raise, elbow/wrist wave and return.
Measured teacher elbow/pitch/roll travel is **37.2° / 29.6° / 35.0°**; peak hand
error 1.61 mm. Both learned actors move but fail (base boundary / unsafe contact),
all stops confirm. Current 8771 service: `emet-herman-arm-wave`.
[Checked video and joint outputs](../experiments/herman_elbow_wave_20261117.md).

**Random smooth arm distribution is implemented:** 20% of all training ticks,
with 10/10/10% existing hand/base/goal changes and 50% TAMP. Seeded FK-witness
splines cover legal folded and reaching postures, continuous motion through
interior knots, bounded derivatives and quiet holds. Four exploratory teacher
cases pass below 2 mm; the complete level-zero teacher and source-bound tests
have now passed for frozen `614dfcd6`: see the current launch/restart status
below. No new actor improvement is claimed. [Protocol](../experiments/mars_arm_splines_20261117.md).

**Baseline curriculum visualization is complete:** seven checked 4×4 teacher videos,
112 distinct displayed trials, all 144 spline targets plus existing families in
a 208-path 3D coverage gallery. Small splines 16/16, medium 14/16, full 12/16;
these are teacher outcomes, not new actor scores. [Viewer and measured results](../experiments/mars_curriculum_visualization_20261117.md).

An eighth, larger-motion v2 candidate video now passes **16/16 teacher trials**.
Median local hand excursion is 80.5 mm and reference-motion duty is 86.2%.
Shorter successes replay with a visible reset at real-time speed; failures remain
visible. The expanded coverage gallery contains 224 paths (208 hand-active).
Media, raw trial tables and model snapshots remain ignored under `logs/`.

**Completed spline baseline:** frozen `614dfcd6` passed 1,679 regressions and
48/48 teacher preflight cases. The 65,536-decision restart completed; its paired
actor outcome is summarized below. Launch/interruption records remain under
`logs/recovery/arm-spline-development-v1`. Do not freeze live TAMP IPC.

**Current decision: fix motion distribution before more PPO.** The spline
pilot and paired evaluation are complete: both actors 0/48, with 47/48 and
46/48 confirmed stops. Neither is promoted. Raw outputs remain ignored under
`logs/eval/arm-spline-paired-v1-restart`. Within-episode auditing found only
0.4 mm median local hand excursion in the ordinary hand pool and 3.9/25 s
of target motion for ordinary episodes. Opt-in bounded v2 timing, quiet-hold
completion and minimum-spread sampling are implemented. V2 also widens ordinary
hand and goal-change arm offsets rather than letting base travel supply all hand
motion. Full-family candidate preflight precedes further training.
[Metrics](CURRICULUM_VISUALIZATION.md#per-trajectory-spread-and-dead-time-audit).

**Main integration:** main `f1b7bb5b` is integrated with semantic
preservation of unloaded TAMP scopes through grasp alignment/fallback search,
checked base approaches, observed verification and measured stopping. Runtime
`command_client` copies are synchronized. Integration/recheck runs cover 765
distinct passing cases; this is not a new full frozen training receipt. The
session-start abort now preserves `SystemExit` before guard creation.
The newer centralized agent tool bindings retain the unloaded scope; 78 agent
import/dispatch/session/discovery tests pass after that merge.
The subsequent CHAT-handler extraction preserves that scope in `chat_tamp.py`;
124 agent runtime, capture, dispatch and task-session tests pass after this merge.

**Expanded-curriculum debugging:** general reset now synchronizes the servo's
base tolerance with the task after leaving the fixed-base fixture (previously
5 mm vs a declared 5 cm). The v2 general teacher opts into coupled base/arm
feedback so relaxing base precision does not strand an underactuated arm. Its
finite six-second settling ceiling includes the unchanged two-second measured
quiet hold; episodes end as soon as that hold passes. Both-robot tolerance,
coupled teacher, source-bound preflight and media regressions cover these
changes. The corrected candidate teacher rerun passes **70/70**, all admitted
with confirmed stops: splines 24/24, ordinary hand 16/16, base 8/8, goal changes
16/16, unloaded TAMP 6/6. All 170 current focused tests pass, including the local
HTTP test rerun with loopback permission. These do not replace a fresh full
training regression receipt. The first frozen `e412cc47` full-suite attempt
crashed natively while slicing MuJoCo contact objects for critic features; the
gate stopped before PPO. That failed log is retained. The follow-up snapshots
active contact arrays instead, documents finite-deadline termination and derives
actor feedback offsets without changing its action law. A fresh full suite and
teacher receipt are required before the bounded 65,536-decision pilot. No
promotion or hardware motion is implied. The [architecture comparison protocol](WHOLE_BODY_ARCHITECTURE_REVIEW.md#required-comparison-before-actor-promotion)
separates distribution repair from the still-unresolved gain-head question.

**Current frozen run:** `b958fc62` passes 104 contact/actor/coupled regressions;
the layout refactor gives bit-identical actions for 128 observations across
eight Mars/Stretch feedback configurations. Managed service
`emet-motion-density-reviewed` is running a fresh full suite on this immutable
snapshot, then a fresh teacher preflight, source evaluation, 65,536-decision PPO
pilot and paired evaluations. Stage failures prevent PPO. State/logs are under
`logs/recovery/motion-density-v2/pipeline-current.json`. This snapshot precedes
the later main tool-binding merge; its eventual receipt must not be described
as full validation of that newer combined tip. The frozen full gate has now
passed **1,833 tests**, and its fresh teacher preflight passes **70/70** again.
The first PPO launch exhausted the service's 64 thread slots in W&B before any
update. It is preserved. Service `emet-motion-density-training` restarts the
same fixed budget in `logs/manipulation/motion-density-v2-restart`, with one
compute core, one Go worker and 128 thread slots. Restart state is
`logs/recovery/motion-density-v2/restart.json`. That pipeline is now complete:
65,536 decisions / 78,440 physical ticks. Matched splines remain **0/24 → 0/24**,
with 24/24 stops each; median final hand position error improves 24.0 → 21.1 mm,
but contact stops increase 15 → 17 and wheel-lift stops stay at two. Hand cases
stay 1/16, base 8/8, handoffs 0/16. Final unloaded TAMP is 6/6; source has 5/6
plus one execution error with no terminal stop evidence, so that difference
cannot be attributed to learned tracking. No promotion or further budget
extension. Detailed raw outputs remain under ignored `logs/eval/motion-density-v2-*`.

**Broader references implemented separately:** experimental joint/wheel cubic
paths integrate nonholonomic axle motion and use FK for the same pose interface.
Three bands use 25/55/85% of configured speed caps, four anchor neighborhoods
plus full-joint-space reset proposals, bounded sampling and retained rejections.
First teacher checks: **9/10 slow, 4/10 medium, 0/10 fast**, all 30 admitted and
stopped, no unsafe-force/wheel-lift/invalid-physics flags. The faster cases fail
the unchanged path-precision gate. This family is not in PPO. [Protocol](TRAINING_MIXTURE.md#jointwheel-sampler-implementation-and-first-validation).

**Controlled head diagnostics completed:** common backbone and zero initial
motion means; matched teacher-action fitting, not PPO. Frozen backbone: both
heads 0/8. Trainable context: gain 2/8, direct 0/8. Adding admitted reset-time
capture errors gives gain 3/8 and direct 1/8, all successes on train cases;
held-out cases remain 0/4. All actor stops confirm. One recovery audit expert
hits an unsafe-contact guard; it is retained, never used for fitting, and not
called qualified. The direct head cannot be exported through existing bundle
formats. [Interpretation and reproduction](WHOLE_BODY_ARCHITECTURE_REVIEW.md#small-motion-head-diagnostics).

**Broader fitting diagnostic completed:** `emet-gain-recovery-population` used
16 existing train identities and eight audit identities, nominal plus admitted
capture-error references, the coupled teacher, and 4,000 fitting updates on the
gain controller. Training teacher failures prevent fitting; audit failures stay
reported and never enter the optimizer. All 48 coupled-teacher trials passed,
but the fitted controller reaches only **1/16 train, 0/8 audit**, with 24/24
confirmed stops. This is supervised error-coverage diagnosis, not another PPO
budget extension or a deployed actor. Its immutable
overlay uses the validated `b958fc62` baseline; the pre-existing working-tree
reward edit is excluded and preserved. Output:
`logs/eval/gain-recovery-population-v1/report.json`.

**Corrective-data diagnostic completed:** `emet-gain-local-response` adds two nearby virtual-goal
labels every five physical ticks at measured teacher states. Scratch FK checks
joint-path collision/support; copied control/history views supply sensor-only
inputs and expert responses. No virtual goal is executed, and the measured
plant, real command and history stay unchanged. The actor, 4,000-update budget
and original evaluation cases match the previous diagnostic. This tests whether
local corrective directions absent from nearly perfect rollouts improve closed
loop behavior. It reaches **16/16 train and 7/8 audit**, with **24/24 confirmed
stops**, no unsafe-contact, wheel-lift or invalid-physics flags. Median terminal
hand error is 0.094 mm and 0.000535 rad. The remaining folded-right audit case
settles but exceeds path bounds (5.26 mm / 0.0445 rad peak); its good endpoint
does not erase that failure. This is supervised fitting, not PPO improvement
or qualification. Output:
`logs/eval/gain-local-response-v1/report.json`.

**Independent-seed check completed:** `emet-gain-local-confirmation` freezes
that actor and evaluates 32 fresh spline identities (16 at level zero, eight
each at levels one/two), paired with the same coupled teacher. It then reruns
the existing hand/base/goal-change/unloaded-TAMP pools, explicitly labeled as
reused regression cases. Nothing is fitted or pruned. The experimental weights
remain outside deployment bundles. Results:

| Fresh spline level | Actor | Coupled teacher |
| --- | ---: | ---: |
| Nearby | 12/16 | 16/16 |
| Medium | 6/8 | 8/8 |
| Faster/wider | 1/8 | 7/8 |

All 64 spline stops confirm; the actor has eleven path misses, one timeout and
one unsafe-contact stop (30.2 N reported peak), no wheel lift. Thirty of its 32
episodes satisfy the two-second settling check, so accurate endpoints alone
are insufficient. Reused pools yield hand 3/16, base 4/8, goal changes 7/16,
unloaded TAMP 6/6, with all 46 stops confirmed. These are unloaded grasp-pose
boundaries, not full pick/place or carry. The original base actor's 8/8 is not
preserved by this zero-initialized motion-head experiment. Output:
`logs/eval/gain-local-confirmation-v1/report.json`.

**Mixed diagnostic completed:** `emet-gain-mixed-response` retains corrective
labels and the 4,000-update budget, with explicit sampling probabilities: 50%
spline examples and one sixth each ordinary hand, base and goal-change examples.
It adds four canonical train/four audit identities per general family; audit
labels never enter fitting. This tests missing cross-family exposure, without
adding a residual controller, changing deployment architecture or relaxing
acceptance bounds. All 72 collection teacher trials pass. Confirmation yields
hand **14/16**, base **2/8**, goal changes **14/16**, unloaded TAMP **6/6**, but
only **13/32** spline successes. Three actor wheel-lift episodes occur across
splines/base; only **76/78 actor stops** confirm (teacher 32/32). This is an
unqualified tradeoff, not a replacement for either retained actor. The original
report includes one false `running` outcome caused by the horizon bug below;
it remains a failed episode. Confirmation seeds are reused regression cases,
not another independent holdout. Outputs:
`logs/eval/gain-mixed-response-v1` and `logs/eval/gain-mixed-confirmation-v1`.

**Current follow-up:** `emet-gain-source-feedback` keeps the mixed data/budget
but restores both declared source base settings: signed reference-tracking
navigation feedback and `base_rate_feedback=False`. Earlier head comparisons
deliberately disabled those settings to match a direct head; this gain-only
follow-up must not be presented as that matched comparison. The actor and
executor interfaces are unchanged. A frozen fitting/evaluation pipeline is
running on one CPU. Outputs: `logs/eval/gain-source-feedback-v1` and
`logs/eval/gain-source-feedback-confirmation-v1`.

**Evaluator correction:** spline rollouts now use ceiling-plus-one tick budgets
and explicit exhaustion failure. Flooring a decimal horizon could exit one
tick before the environment's terminal receipt. Two regression cases cover
decimal timing and nontermination; frozen historical reports are unchanged.
The current head/confirmation/budget modules cover 14 passing tests, including
simulation and rejection of mismatched non-tensor base-feedback settings.

Disk headroom recovered by compressing eleven old source archives: **3.39 GiB**
saved, each verified by bit-exact decompression before removing its raw tar.
Checkpoints, recordings, failed trials and archived source contents are retained.
Restore paths and original hashes are in ignored
`logs/recovery/trajectory-spread-v1/source-archive-compression.json`.
An additional 28.17 GiB was reclaimed from an unused July InternVL download
cache; its pinned revision is recorded for re-download. Robot assets and learned
weights are preserved.

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

**Hold-exploration pair is complete at frozen source `9d0620f9`.** All 1,585
fresh regressions pass. Ordinary / quiet are **20/24 / 19/24** new cases,
**7/10 / 6/10** existing coupled and **6/6** TAMP prefixes each. Quiet introduces
two new contact failures despite more successful stochastic holds; it is rejected.
All 120 evaluation stops confirm; no evaluation wheel lift or invalid physics.
The ordinary arm exactly reproduces the previous independent-critic weights.
The paper builds to 61 pages. [Completed report](../experiments/quiet_reference_training_20261117.md).

**Hand-priority experiment is complete.** Weight 1 → 4 changes new cases
20/24 → 19/24, existing coupled stays 7/10 and TAMP 6/6. Same three contact
failure identities, but worse worst hand error, linear jerk and contact load;
no promotion. [Completed report and paired plot](../experiments/hand_priority_training_20261117.md).

**Longer fine-tuning and fresh confirmation are complete.** 65,536 additional
decisions finish (81,920 lineage total). Reused development is 20/24 → 19/24,
existing coupled stays 7/10 and unloaded TAMP 6/6. The 24 genuinely fresh cases
improve **12/24 → 16/24**, safety flags **4 → 1**, wheel lifts **2 → 0**;
teacher is 17/24. All 72 fresh stops confirm and initial/config/reference hashes
match. Peak jerk and reused stress contact loads still worsen, so no promotion.

Goal transitions are a specific gap: longer-trained 0/4 versus source 1/4 and
teacher 2/4. Single-stage coupled is also only 4/8. Frozen source `2a2ce5c1` now
passes **1,604 full regressions**. New goal cases are all admitted: teacher 13/16,
source 9/16; all 32 stops confirm, teacher has one safety-flagged episode and
source none. Managed job `20261117_111259_6729bc` is **complete** at 131,072 new
decisions / 155,423 physical ticks (212,992 lineage). Regular tracking improves
19/24 → 22/24 and TAMP stays 6/6, but existing coupled falls 7/10 → 6/10 and
goal changes 9/16 → 6/16. All 56 candidate stops confirm; no promotion.
Only four stochastic goal episodes per train case occur, all fail (31 timeouts,
one wheel-lift stop). Next inspect the three lost goal successes and fine-tracking/
credit assignment before another budget extension. No PPO remains active; the
retained simulator viewer is restored on 8766.
[Curriculum](../experiments/goal_change_curriculum_20261117.md),
[longer run](../experiments/long_finetune_20261117.md),
[fresh cases](../experiments/fresh_tracking_confirmation_20261117.md).

**Herman console is live on localhost:8770**, job `20261117_120035_064abd`.
Both head cameras, arm/odom and hand TF work through passive SSH subscriptions;
wrist is unavailable. The new UI has base/6DoF-hand previews and masks, keyboard/
pointer control, freshness displays and training progress. It sends no motor
commands and cannot stop a robot. Local bridge/session fixes retain original
capture ages, measured base twist and all-six-component velocity checks, requiring
verified SI units/read validity. They are not deployed on Herman. Keep these
changes distinct from the frozen training receipt.
[Division of responsibility, console and deployment sequence](CONTROLLER_DEPLOYMENT.md),
[hardware readiness](MARS_HARDWARE_READINESS.md).

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

**Wider spline follow-up:** the source-feedback fit scores 18/32 splines,
15/16 hand, 8/8 base, 16/16 goal changes and 6/6 unloaded TAMP. Adding eight
level-one training identities with the same fitting budget gives 21/32
splines (14/16 nearby, 6/8 medium, 1/8 fast/wide), 14/16 hand, 8/8 base,
15/16 goal changes and 6/6 unloaded TAMP. Both fits confirm all 78 actor stops.
The wider fit has two wheel-lift events at level two. Retain the source-feedback
fit as the baseline and compare matched geometry at different speeds to isolate
the fast/wide failures. These runs use the frozen simulator overlay. Reports
are under `logs/eval/gain-{source-feedback,wider-splines}-confirmation-v1`.

The paired speed-cap sweep on those eight fast/wide geometries found teacher
success of 7/8, 8/8 and 7/8 at 100%, 80% and 65% of the original speed cap.
The wider fit reached 1/8, 2/8 and 3/8, with two, two and one wheel-lift events.
One path exceeded the 30-second duration limit at 65% for both controllers.
All 48 stops confirmed. Executed pairs match joint-waypoint and reset-state
hashes. These caps are relative to the simulator reference, and actual duration
also depends on acceleration and jerk limits. Slowing alone leaves substantial
tracking and support failures. Report: `logs/eval/gain-wider-speed-sweep-v3/report.json`.

### Tracking before dedicated hold practice

The arm-only mix spent only 2,135 of 65,536 control ticks on moving arm-only
references. Its endpoint audit tied the initializer, and the new actor failed
the matched slow wave even with zero base commands.

The arm-spline benchmark and trainer now accept `--tracking-first`. This mode
uses only the selected level's moving multi-knot splines. Each episode starts
the reference immediately and ends at path completion without endpoint dwell.
The benchmark checks the existing 3 mm / 0.02 rad peak path limits and confirms
the stop. The receipt is bound to this mode, and audit cases stay out of PPO.
Use `reference_motion_ticks` to measure exposure. Promote to the standard
hold-inclusive mix manually after the trained actor demonstrates tracking on
held-out splines. Run a separate held-out gesture rehearsal before considering
a hardware trial. The stage reuses the existing task-space controller.

### Simulator action experts

The expert router binds an unloaded base, arm, or coupled command to a
single actor for its lifetime. The simulator executor drops unused command
channels and caps requested arm-only joint speeds. An unqualified learned
route can use the teacher at command acceptance, subject to the same
command envelope. A malformed policy action raises an error.

`scripts/eval_mars_action_experts.py` compares the individually trained actors
with the routed teacher on matched audit cases. It checks terminal success,
full-path hand error, arm-only base drift and measured speed, and confirmed
stops. Its report identifies the learned, teacher, or unsupported simulator
route per family. It covers the declared audit cases, and any use on new
trajectories requires their own admission. Corrective arm fitting and
independent spline confirmation are tracked under
`logs/recovery/action-experts-v1`.
