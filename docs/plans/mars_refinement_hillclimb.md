# Mars whole-body refinement hill climb

Objective: a smooth, accurate, reactive controller that can execute the agent's
task-pose contracts on real Innate Mars. Keep one runtime actor. Retain every
attempt; promote a development candidate only using explicit physical gates.

## Current baseline

`mars-base-curriculum-legacy-v1-seed90/last/actor` passes 49/50 expanded base
tracking, 42/50 full base and 15/16 full arm cases, with 16/16 arm tracking.
Remaining failures: one larger forward reference, seven other base smoothness
misses and one small-amplitude arm jerk miss. No unsafe-contact, wheel-lift or
invalid-physics flag trips. This is a development baseline, not a hardware-ready
policy. Source robot/model, normalization and all acceptance thresholds remain
fixed for the next comparison.

The previous cold-start aligned reward removes 50 native completion points but
earns no full-contract bonuses during exploratory training. Test a gradual
transition and quieter exploration from the stronger baseline instead.

## Continuation fidelity and reward schedule

The default initial arm admission remains strict. The explicit
`--initial-admission source_fidelity` mode supports simulator repair of a known
partly qualified source. First replay the source on all 66 expanded cases under
current code and require its safety flags to remain clear. Then require the
merged actor's canonical and expanded-arm metrics, actions, rewards, physical
poses, stop codes and reset/reference identities to match that fresh source
exactly. Record existing failures without granting qualification. Missing,
duplicate, changed or stale coverage fails before optimization. Final evaluation
and hardware-promotion gates are unchanged.

For fraction `f`, use `native_reward - f * native_bonus - f * excess_costs +
f * terminal_contract_bonus`. At zero this is exactly native reward; at one it
is the previous fully aligned objective. Latch `f` per episode so changing a
global schedule cannot change reward accounting halfway through a trajectory.
Native and terminal completion credit together remain bounded by their declared
mixture. Full qualification always uses the fixed original gates.

## Next matched refinement pair

Use the same 8k baseline above, seed 90, fresh optimizer, frozen arm/context/grip
and normalization. Run sequentially on CPU14, CUDA hidden, one Torch/BLAS thread,
offline scalar W&B. Require fresh source-bound full regressions and teacher
66/66 before either run.

- Each run: 16,384 transitions, 80% base / 20% moving-arm control-step budgets,
  existing base primitive/posture mixture, LR 3e-5 and initial gSDE coefficient
  std 0.1 rather than 0.5. Keep 8-tick resampling, gamma 0.9975, rollout/batch
  256, five epochs, clip 0.1 and KL target 0.01.
- Control: native reward throughout.
- Scheduled: jerk/peak weights 0.1/0.2, full-contract bonus 10, native warmup for
  4,096 transitions, linear transition over the next 4,096, then full alignment.
  Episode boundaries can delay the applied fraction; log the actual fraction.
- Evaluate all 18 canonical cases at step zero, every 4,096 and final. Evaluate
  all 66 expanded cases at final. Score every result; do not select an earlier
  checkpoint using validation. This comparison is exploratory hill climbing on
  development cases, not an untouched generalization estimate.

## Selection and further work

Retain the source unless a fixed final candidate has no safety events, preserves
16/16 arm tracking and at least 49/50 base tracking, preserves at least 15/16
full arm and 42/50 full base passes, and strictly improves at least one of those
full-pass counts. Report all failures and numeric jerk/error changes even when
a count stays unchanged. If neither improves, diagnose the specific failed
trajectories and training signal before declaring another bounded recipe.

After these finite cases are mastered, expand nearby reachable poses, base
headings, goal changes and disturbances, then coupled and continuous loaded
TAMP trajectories. Generate new audit cases before claiming generalization.
Verify real velocity-command framing, limits, watchdog, measured stop and
telemetry-only integration before a reviewable, bounded Herman motion test.
No simulator score alone authorizes uncontrolled hardware motion.

## Refinement results and critic calibration experiment (November 6)

The fixed native and scheduled finals both improve expanded base tracking to
50/50. Native achieves 43/50 full base and 11/16 full arm; scheduled achieves
40/50 and 2/16. Both retain 16/16 arm tracking and clear the recorded safety
flags, but neither preserves the baseline's 15/16 full arm passes. Retain the
source. All checkpoint results and failures are in the
[refinement report](../experiments/mars_refinement_20261106.md).

The source also fails all four coupled cases (teacher 4/4), so separate base and
arm success cannot qualify it as a useful whole-body controller. Keep these
failures visible while repairing the simpler distribution first.

An on-policy audit of the native refined actor, using its saved exploration and
8-tick resampling, finds reset-value residuals of about 25 on a base hold and
67--69 on two arm trajectories. These include SB3's time-limit bootstrap. Two
exploration seeds per case are diagnostic evidence, not a calibrated estimate
of broad performance. Critic underfit is a testable explanation for poor policy
updates, not an established cause of the smoothness regression.

Next bounded experiment, before any further reward change:

- Restart from the retained source, same 80/20 base/arm control-step mixture,
  coefficient std 0.1, seed 90, actor LR 3e-5 and native reward throughout.
- Train for 24,576 transitions: 8,192 critic-only transitions, then 16,384 PPO
  actor transitions. Freeze both actor means and noise during critic warmup;
  verify exact tensor identity at its boundary and evaluate physical behavior.
- Use a separate critic LR multiplier of 30 (effective 9e-4). During critic
  warmup use 20 epochs and GAE lambda 1; restore five epochs and lambda 0.95
  for actor learning. Leave gamma, rollout/batch size, clipping and KL unchanged.
  The optimizer must preserve its role ratio when SB3 applies LR schedules.
- Require fresh full regression receipt and teacher 66/66 under the new source.
  Evaluate canonical 18 at zero, every 4,096 and final; expanded 66 at final.
  Repeat the noisy value diagnostic at the warmup boundary and final. Do not
  select a checkpoint by repeated validation scores.
- Use the same physical nonregression/strict-improvement selection gate above.
  This changes training only; the exported inference actor, target interface,
  limits, measured feedback and executor safety remain the same.

If this does not improve the fixed final, examine task reward/value scale and
arm-error-to-base coupling before adding more training time. If it does, add
coupled goals, mask combinations and dynamic target changes, then loaded TAMP
plans and passive Herman interface validation. Hardware readiness additionally
requires verified SI feedback, freshness, odometry/twist, watchdog and measured
whole-body stop on the actual robot.

## Coupled-goal curriculum and critic units

The critic-warmup final preserves 49/50 base tracking and 42/50 full base but
regresses to 5/16 full arm. Reject it; see the
[complete report](../experiments/mars_critic_calibration_20261106.md). Its critic
improves base-hold calibration while saturating on arm states. A fixed-target,
fixed-batch value regression improves native-unit audit MSE from 791.57 to
108.50 when rewards/critic units are scaled by 0.01. This is a training
conditioning result, not improved control.

The coupled teacher-base/learned-arm diagnostic passes 4/4, while removing the
heading selector's hand exclusion still gives 0/4. Direct hand-to-base coefficient
restoration recovers 14/16 arm cases, below the retained source's 15/16. Do not
deploy either ad hoc ablation. Add the missing coordinated data while preserving
rehearsal, rather than requiring perfect primitive jerk before exposing the
controller to simultaneous goals.

Next fixed final, starting from the same retained source:

- 36,864 transitions: 4,096 critic-only warmup (20 epochs, lambda 1), then
  32,768 ordinary PPO transitions (five epochs, lambda 0.95).
- Base/arm/coupled control-step budgets 40/20/40. All base coefficients remain
  trainable; no new hand-to-base mask, teacher action loss or runtime filter.
  Learned arm/context/grip and normalization stay fixed for this diagnostic:
  the teacher-base intervention shows that the preserved arm can execute these
  four coupled references. Unfreeze arm/context later if evidence requires it.
- Scale every training reward by constant positive 0.01, without clipping or
  changing reward shape. Rescale the initial value head into these units.
  Actor outputs, observations, physical limits and native evaluation are
  unchanged. Save critic units in bundles and restore native units when a
  generic trainer transfers them. This critic is not an infeasibility certificate.
- Actor LR 3e-5, critic multiplier 10 (3e-4), coefficient std 0.1, seed 90.
  Keep gamma 0.9975, batch/rollout 256, eight-tick resampling, clip 0.1 and KL 0.01.
- Require fresh source-bound full regressions and teacher 70/70 (50 base,
  16 arm, four coupled). Initial source fidelity now covers all 70. Record its
  coupled failures; initial fidelity does not grant qualification.
- Evaluate canonical 22 at zero, warmup boundary, every 8,192 and fixed final;
  expanded 70 at final. Require exact warmup physical replay and repeat noisy
  value audits in native units at warmup and final. Keep all results.

A development candidate must preserve source base/arm tracking and full-pass
counts, have zero recorded safety events, and strictly improve at least one
stage's full-pass count. Passing any subset of coupled cases is progress, not
completion. All 70 physical gates, dynamic masks/goal changes, load/disturbance
coverage and integrated TAMP/agent execution remain required before simulator
qualification, followed by actual Herman interface and bounded-motion gates.

## Gain-bias optimizer units: matched follow-up

The normalized coupled final is rejected: base 49/50 tracking, 38/50 full;
arm 16/16 tracking, 5/16 full; coupled 0/4 tracking/full. Fresh regressions pass
1,041 and teacher 70/70; initial full transfer and warmup are physically exact.
See the [complete report](../experiments/mars_coupled_units_20261106.md).

General yaw gain biases move from approximately 0.00457 to 0.00651/0.00699,
despite 32,768 actor transitions. The restoring-turn head's biases remain near
2.0, while context readout changes can affect the arm's base reactions. Test
gain-parameter learning units before adding policy structure or another long
unchanged run.

Repeat the preceding 36,864-transition recipe from the same retained source,
changing only `--gain-bias-lr-multiplier 1000`: effective gain-bias LR 0.03,
other actor readout/noise LR 3e-5 and critic LR 3e-4. The actor context stays
frozen. Keep the same seed, 4k warmup,
40/20/40 step budgets, reward scale 0.01, exploration, evaluation cadence and
fixed final. The optimizer preserves all LR ratios under SB3 schedules and
reload, and frozen arm rows remain unchanged. Log the full base gain-bias
vectors per rollout. Initial mean action, inference architecture and safety
limits are unchanged. Fresh full regressions and teacher 70/70 are required.

Apply the same physical nonregression selection gate and retain every failure.
Do not select an intermediate validation checkpoint. Quarter-amplitude training
parents pass 2/2 under the source; half and three-quarter parents fail tracking.
This establishes an easier curriculum starting point for a later experiment,
without changing any canonical goal or threshold.

## Rank-complete arm coverage and joint whole-body refinement

The gain-bias ×1000 trial finished with base 50/50 tracking, 33/50 full;
arm 16/16 tracking, 0/16 full; coupled 1/4 full. Reject it. Larger updates to
base feedback recover some coupled motion but introduce arm-stage base jerk.

A ground-truth public-tool replay on `left:20g:0` exposes a separate coverage
gap. The retained actor, admitted as the actual initialized step-zero checkpoint
of that experiment, makes 56 policy calls and stops during pregrasp at 10.10 mm
position error. The reference completes the first pick/place but cannot plan the
return within its planning budget. Neither completes the two-transfer program.
The learned elbow command is much weaker than the reference rate. Replacing
only the phase one-hot in the logged observations reproduces the original
56 actions and changes that elbow command modestly, insufficient to explain
the deficit. This counterfactual is an input audit, not a successful rollout.

The original four arm primitives excite only joint 2 and joint 5: their joint
displacement matrix has rank two. Recent refinements freeze the arm, including
on coupled examples. Strong scores there do not establish general arm control.

Next bounded experiment, declared before training:

- Keep all 26 frozen cases, thresholds, limits, references and reset partitions.
  Add eight **training-only** arm parents in
  `mars_tracking_mastery_full_arm_v1.json`: positive/negative joint 1, 3 and 4,
  plus two coordinated pitch-joint translation motions. All targets come from
  nearby FK joint paths, with collision/support/joint-margin admission. The
  joint displacement matrix has rank five. Use existing training reset seeds;
  no validation seed or target is mined into training. Keep 0.25/0.5/1 arm
  amplitudes. Expanded admission: 50 base, 40 arm, four coupled (94 total).
- Sample control-step budgets 30% base / 50% arm / 20% coupled. Within arm,
  half the budget rehearses original primitives, half covers new parents.
- Train all seven motion readout rows together, retaining context, gripper and
  normalization initially. Use executor-consistent action masking for means,
  sampling, likelihood and entropy. State-dependent noise uses current/future
  pose errors, with a small stationary floor; hand errors may explore base
  motion when its corresponding pose constraint is absent. Export one ordinary
  eight-output actor; no training wrapper is deployed.
- Restart from the retained source, seed 90. Run 65,536 transitions including
  4,096 critic-only warmup. Actor LR 3e-5, critic ×10, gain-bias ×300; base/arm
  exploration coefficients 0.1/0.03, resampled every eight ticks. Uniform reward
  scale 0.01; keep native reward shape, gamma, rollout/batch, clip and KL. Restore
  five PPO epochs and GAE lambda 0.95 after critic warmup (20 epochs, lambda 1).
- Require fresh complete regression receipt and teacher 94/94. Verify physical
  source fidelity, actor identity through warmup, and export/reload consistency.
  Evaluate fixed canonical 22 at zero, warmup, each 16,384 and final; expanded
  94 at the fixed final. Preserve all intermediate/final failures; no checkpoint
  selection using validation. This changes distribution and trainable motion
  rows together, and is not a matched single-factor comparison.
- Retain the source unless the final preserves its original base/arm tracking
  and full-pass counts, introduces no recorded safety events, and improves the
  new arm/coupled failures. Then replay public-tool pick/place with real checkpoint
  admission, first one transfer and then continuous multi-step programs. A
  teacher planning failure remains a planner issue, never controller success.
- Finite nearby cases still do not qualify hardware or full SE(3). Follow with
  independent translation/orientation combinations, changed goals, disturbances,
  loaded grasp/lift/transport/place and measured-stop tests. Passive Herman
  feedback/velocity/watchdog validation remains a separate prerequisite. No
  hardware motion is authorized by this experiment.

A separate diagnostic audit is frozen before this run starts:
[`mars_full_arm_audit_v1/cases.json`](../../paper/data/whole_body_control/mars_full_arm_audit_v1/cases.json),
SHA256 `63f76483b2ab7048512a268459d034ceeaaeaa898193b048a1beb4c01ef00985`.
It uses 12 new reset seeds (`34200000`–`34200011`), eight 1.15-amplitude new arm
parents and four unseen coordinated joint combinations, all converted to FK
pose paths under the same limits and thresholds. None enter training. After the
initial-fidelity audit frees CPU15, replay teacher/source and then the fixed
final, preserving every admission/execution failure and plotting targets. Do not
choose another checkpoint or tune this run from these audit outcomes. These are
extra exploratory development checks, separate from the frozen canonical gate.

## Smaller-step control after the first seven-motion trial

The fixed 65,536-step final is rejected: expanded base tracking/full 50/50 and
42/50, arm 37/40 and 4/40, coupled 0/4. The fresh 12-case arm audit regresses
from source 12/12 full to 8/12 tracking and 0/12 full. Public-tool pregrasp lasts
151 policy calls with maximum hand position error 7.33 mm, but stops at 10.3 mm
base drift; stop is confirmed. Thus arm responsiveness improves without meeting
whole-body stability. No new hardware qualification follows.

A same-batch, one-update conditioning audit from the critic-warmup checkpoint
finds exact active-action Gaussian KL 0.756 at actor LR 3e-5 / gain-bias ×300.
Changing gain-bias alone to ×1 still gives 0.056. Reducing all actor learning rates
tenfold, while retaining critic LR 3e-4 and bias ratio ×300, gives KL 0.00756.
This is a local optimization diagnostic, not demonstrated policy success.

Declare a matched smaller-step control: same source, 94-case distribution,
65,536 transitions, warmup, objectives, seeds and evaluation cadence. Actor LR
3e-6, critic multiplier 100 (same effective 3e-4), bias multiplier 300 (effective
9e-4). Use immutable detached checkout `11305c8c` under
`/tmp/mars-rate-control-11305c`; verify its complete source state against the
1,064-test receipt and teacher 94/94 before launch. The current working branch
may then develop a separate TAMP-posture curriculum. Source-fidelity admission
and exact warmup checks remain; no auxiliary CPU15 simulator is launched by this
control, leaving that worker for new-code tests. Score the fixed final, then run
public-tool pick/place and the frozen 12-case audit serially on CPU14. Retain the
source unless all declared nonregression/improvement gates pass. No automatic
hardware promotion or motion.

## Procedural TAMP-like postures and longer scheduled refinement

The first seven-motion final improves public-tool position response but fails
base drift and regresses the fresh audit; see the
[complete report](../experiments/mars_seven_motion_20261106.md). In parallel with
the immutable smaller-step control, implement a separate training-only expansion:
`mars_tracking_mastery_tamp_postures_v1.json`. It adds two nominal near-straight
postures (elbows −0.04/+0.04 rad, compensating wrist pitch), eight extra arm parents
and 0.25/0.5/1 coupled amplitudes. Half the arm budget rehearses the old families;
half uses the new postures. All original canonical cases and validation seeds
remain fixed. Expanded admission is 122: 50 base, 64 arm and eight coupled.

Posture selection occurs before ordinary reset settling and geometric/support
admission. Generic runtime/reset commands cannot inject these coordinates, and
loaded/pregrasp fixtures reject the override. Actor inputs remain measured state,
masked task poses and preview; joint target witnesses remain private.

After the matched control and its serial tests finish, declare a bounded longer
candidate from the retained source: 262,144 transitions, 4,096 critic warmup;
30/50/20 base/arm/coupled control-step budgets; actor LR 3e-6, critic ×100 and
gain-bias ×300; noise 0.1/0.03, native reward scale 0.01. Use the existing staged
contract objective: 4,096 native steps, then ramp over 32,768 steps, jerk excess
weight 0.1, incremental peak weight 0.2 and full-contract bonus 10. This is a
multi-factor development experiment, not an isolated reward claim. Context/grip
and normalization remain frozen. Evaluate canonical 22 at warmup and every
32,768; score the fixed final on all 122, original 70 and separate audits, then
real public-tool pick/place. No intermediate selection or hardware promotion.

Fresh full regressions, teacher 122/122, legacy physical replay, source-fidelity
admission, and reset/frozen-fixture tests must pass first. Keep the matched
control in its immutable checkout: new source must never change underneath a
running experiment. On this workstation, one CPU14 native training worker and
one CPU15 test/teacher worker remain the limit. Caliban is reachable, but its
previous exact frozen-fixture portability failures still block a qualified
training launch there; do not skip those gates or change their tolerances.

Also freeze [12 new near-straight audit cases](../../paper/data/whole_body_control/mars_tamp_posture_audit_v1/cases.json)
with seeds 34300000–34300011 and independent procedural starting postures before
training. These never enter the sampler. Replay teacher/source/fixed final on
both the previous 12-case audit and these new 12; preserve every failed admission.
Any optional development-baseline replacement by the smaller-step control must
preserve all original/expanded tracking and full counts, keep the old fresh audit
12/12 full, clear recorded safety flags and complete the public-tool transfer.
Otherwise the longer candidate starts from the retained native base-curriculum
source. No intermediate checkpoint participates in that decision.

## Optional context adaptation, keeping the same inference actor

The motion readout pilots keep the policy's context MLP frozen, including when
new initial postures differ from its training distribution. Add opt-in
--train-context with --train-arm so the existing motion network can adapt
end to end. Keep the observation schema, stationary mean at zero feedback error,
velocity limits, executor masks and release guards. No extra inference head,
teacher residual, or temporal action filter is introduced.

A shared context can change a gripper readout even when its weights are frozen.
Therefore this mode requires a **constant source gripper head** (zero context
weights; its bias may be nonzero), and checks that invariant during training and
export. The retained source satisfies it exactly. A context-dependent gripper
source is rejected rather than silently changing training/export behavior.
Critic warmup freezes the context as well; actual PPO, export, reload, nonzero
constant grip requests and zero-error equilibrium are covered by tests.

Next matched experiment after the current fixed-final posture trial and its
serial tests: same retained parent, 122 cases, 262,144 transitions, learning
rates, warmup, contract schedule, noise, seeds and evaluation cadence; only
enable context adaptation. Require a fresh source-bound full regression receipt
and teacher 122/122 under the new code, plus exact native replay of the prior
teacher cases. Evaluate fixed final on all 122 and original subsets, then the
same actual public-tool and old/new fresh-pose tests. Retain the better existing
candidate unless all physical nonregression and strict-improvement gates pass.
This is still nearby free-space training, not contact/hardware qualification.

Implementation validation: 1,085 fresh source-bound regressions pass, including
actual context PPO updates, critic warmup, closed constant-grip export/reload
and zero-error equilibrium. Fresh paired teacher122/122 reproduces all122 prior
native cases exactly. The matched trial stays queued behind completion of the
prior candidate's serial tests.

The prior posture trial's fixed final is rejected: base50/50 tracking40/50 full,
arm62/64 tracking7/64 full and coupled2/8 full, versus source49/50 tracking42/50
full, arm64/64 tracking59/64 full and coupled2/8 full. The base linear jerk gate fails
on55 of64 arm cases. Three actual public-tool transfers all stop in pregrasp;
their references succeed. Finish the separately declared fresh audits as well.
Do not infer that more transitions or broader resets solved the stability gap.
The context trial remains an isolated matched test, with no automatic promotion.

The current trainer's independent-worker path has separate native two-world
parity/PPO/export tests and bounded two/eight-worker ARM spawn probes. Keep
this workstation serial and qualify exact ARM fixtures before scaling the
learned checkpoint. See [protocol and evidence](../experiments/mars_motion_workers_20261106.md).

## Match execution modes before further loaded training

The explicit Cartesian/jerk-limited reference now shares the checkpoint's
admitted actuator settings and initial state. All three cases pass pregrasp
but breach retention at lift/preplace with about 2 mm hand tracking; learned
tracking still fails earlier. The joint/legacy reference's success is retained
as a separate mode. Analytic calls cannot be reported as learned-policy calls.
See [matched evidence](../experiments/mars_matched_reference_20261106.md).

Repeated fault stops also rebase the arm hold target. Latch the first arm/grip
target instead, keep release unauthorized and retain a finite grip command if
jaw feedback fails. All six new behavior tests reject the previous code;
three diagnostic loaded resets now confirm within 0.10/0.25/0.20 s under the
unchanged 0.75 s budget. This fixes stopping, not retention or full transfers.

Fresh full regressions and teacher122 replay run in immutable `e6401542`.
Then declare a **source compatibility audit**, not a new policy candidate:
256 smoke-training transitions with initialized step-zero actor preselected
before updates; require source-fidelity admission and exact replay of all56
saved source inputs. Preserve the smoke final but do not select it. Archive the
actual new-source initialization evidence before rerunning three matched
reference/source transfers. Do not rewrite old checkpoints' training provenance.

After that audit, run an explicitly labeled planning-only diagnostic on all30
frozen public-tool cases for each of three priorities: existing lower pinch,
center first, and horizontal pinch 0.25 object-height above center first.
Only proposal order changes; retain the original alternatives, IK/collision
budgets, continuous physics, objects, goals, force/retention/wheel/stop limits
and reference compilation. Use a fresh serial subprocess per episode to bound
native memory. Retain every rejection/failure and mark the in-memory planning
intervention **ineligible for canonical qualification**. A promising option
still needs a real implementation, full regressions and integrated replay.
Do not tune friction or torque simply to conceal an unstable grasp.

Keep CPU14 for the declared context trial and its serial tests; CPU15 runs the
corrected-runtime gates, compatibility audit and planning diagnostic in order.
No hardware motion or automatic promotion. Once the Cartesian loaded baseline
is reliable, expand training with actual oracle trajectory snippets and loaded
resets while retaining original nearby tracking/settling gates. More free-space
transitions alone have not established either grasp stability or whole-body
safety. The existing end-to-end context comparison keeps its fixed final and
all post-tests; no intermediate is chosen to replace the retained source.

The corrected-runtime full gate passes all1,113 source-bound regressions.
Teacher122/122 reproduces all old cases exactly; compatibility initialization
repeats all56 saved input actions exactly, and all six integrated failed stops
confirm. Retention and learned tracking still fail. Full regression success
does not qualify loaded carry or hardware execution.

## Translate while preserving hand orientation

The corrected-runtime teacher reproduces all122 previous cases exactly, and
all six integrated reference/source fault stops now confirm. Retention still
fails in the three matched Cartesian reference transfers. The initialized
source repeats all56 saved actions exactly; its256-transition smoke final is
not selected. The90-episode grasp-priority diagnostic now runs separately.

The context trial's fixed262k final reaches base50/50 tracking42/50 full,
arm63/64 tracking29/64 full and coupled4/8 tracking/full. This improves over
frozen-context refinement but regresses arm stability from the retained59/64.
All three actual public-tool attempts still fail before grasp. Keep the source;
the two frozen-pose audits finish at4/12 and3/12 full (source12/12 each).
The [complete negative report](../experiments/mars_context_20261106.md) keeps
all cases/checkpoints and actual public-tool failures.

A new reference audit finds that constant-orientation training translations
are shorter and slower than actual TAMP pregrasp. See the
[coverage expansion](../experiments/mars_tamp_translation_20261106.md).
The next fixed-final262k trial changes the declared distribution and includes
the separately tested stop latch. Keep the prior context recipe:4096 critic
warmup, actor3e-6, critic multiplier100, gain/bias multiplier300, coefficient
noise0.1 base/0.03 arm, reward scale0.01, contract ramp32768 after4096,
jerk/peak weights0.1/0.2, terminal bonus10, seed90,30/50/20 step budgets.
Use one local worker onCPU14 after the context post-tests exit. Require a fresh
source-bound full gate, physical teacher146/146 and unchanged old122 rollouts.
Require source fidelity and exact warmup behavior. Keep grip and normalization
fixed; no teacher residual, new runtime filter or relaxed physical gates.
Evaluate canonical22 every32768, expanded146 at final, three separately frozen
12-case audits and three actual Cartesian public-tool transfers. Teacher/source
must see identical inputs; compare execution modes explicitly. Retain every
failed admission. A candidate must preserve source tracking/full counts on all
prior groups and improve a full-pass count; report added-case and integration
failures separately. No automatic promotion or hardware use.

The complete90-episode height-priority diagnostic gives0/30 transfers for each
priority. Reject a height-only change; see the
[full fit diagnosis](../experiments/mars_grasp_fit_20261106.md). A static full-pad
box check finds overlap at the existing30mm axial pinch point in all24 available
measured-open grasp frames; native recorded FK verifies the analytic geometry
at machine precision. The next planning-only comparison tests36mm/38mm on all30
cases each after three exact baseline replays. Keep all guards, geometry and
force/torque/friction parameters fixed; this diagnostic is not canonical
qualification and must not silently replace the controller or grasp planner.

Translation revision2227f58a now passes all1,123 fresh regressions, teacher146/146
with old122 native cases exact, and fresh teacher12/12. Offline W&B's first
startup is denied its sandbox IPC socket before any PPO update; preserve the
failed output and retry the same recipe with local socket access and task-owned
`/tmp` caches. Reverify source-bound receipts/reports; do not skip the initial
source-fidelity or warmup identity checks. The retry is running in the immutable
checkout, with the fixed-final post-tests queued behind it. No promotion.

The pad-geometry native crosscheck passes56 recorded states at machine
precision. All three nominal30mm baseline replays are physically exact; the
36mm/38mm all-case comparison is running. Three read-only contact probes are
queued afterward to record actual solved pad/object forces, frames, friction
and contact positions. Require their physical traces to match the saved
baseline exactly; keep raw contact data ignored. Do not infer success or
change force/friction limits from static clearance.
