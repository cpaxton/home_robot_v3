# Mars supported grasp-to-lift: coverage and objective audit

Supported grasp-to-lift transitions are implemented and trained, but the learned
controller still does **not** complete the integrated delivery. Both matched
pilots finish 16,384 additional PPO steps. On eight independent supported lifts,
the teacher finishes 7/8 and source/control/treatment each finish 0/8 from identical
initial physics. Neither trained candidate is qualified or promoted.

## Distribution and physical admission

The opt-in subtype replaces 75% of the existing 12% carry reset share: **9% of
expected resets**, within the existing 70% general tracking / 30% manipulation
mixture. Ordinary base/hand/combined tracking, stationary holds, mid-motion goal
changes, disturbances, pregrasp, other carry and placement remain represented.
The treatment actually spends 69.67% of its 16,384 sampled steps on reach and
12.44% on supported lift; the control spends 75.20% on reach and zero on that
subtype. Step shares differ from reset probabilities because episode lengths
vary. Per-rollout episode fractions are logged, but their average is not an exact
whole-run reset fraction.

Supported resets physically close on a table-supported object, then hand control
over **before lift-off**. Existing ordinary carry resets lift with the teacher
before actor handoff. Admission requires bilateral contact, a quiet grasp window,
private FK/IK, collision-checked nearby lift paths and sampled support margins.
Targets lift 4.8–6 cm with fixed hand attitude. A bounded min-jerk direct shortcut
is separately checked; it avoids braking at every densely sampled planner vertex.
The actor receives measured history and masked task-pose preview, without joint
witnesses or ground-truth object state. It already receives commanded-versus-
measured actuator errors and previous actions. Architecture/history are unchanged
between the matched pilots.

The reset starts at a closed supported grasp; this subtype does **not** itself
train open-finger approach/closure. The existing pregrasp share still covers that
stage. Closure-to-lift coverage across a range of physical contact conditions is
a remaining generalization requirement.

The first subtype selects a box. Cylinders still slide and spin out under the
teacher despite quiet supported closure and depth sweeps. Those rejected/failed
probes are preserved in `grasp_lift_preflight_v1.*`; other manipulation tasks retain
the original mixed-object distribution. Grip torque, friction, force, wheel and
retention limits were not increased to hide this gap. Low randomization currently
compresses mass around 60 g; this is not evidence of uniform coverage of the
entire configured 20–100 g range or of the public 20 g fixtures.

## Matched training and independent replay

Source is an explicit zero-PPO transfer of the earlier 3x learned hand-gain
diagnostic, not a new policy result. The original bundle correctly fails runtime
source admission after controller changes and is unchanged. The named training
transfer preserves all actor/value tensors, normalization arrays and recorded
deterministic actions, while recording old/new source provenance and a fresh
optimizer. This is not permission to bypass runtime admission in general.

Both pilots use seed 84, one CPU environment, frozen copied normalization,
learning-rate decay from 1e-5 to 2e-6, offline scalar W&B and identical source
`4f8456e8`. There is no DAgger or teacher residual. Full source/config/checkpoint
manifests and persistent drivers are archived. The 801-test source-bound receipt
was verified before launch. The source actor and both final trained actors are
replayed against the same independent cases; exact admitted initial physics is
also checked against the teacher supplement.

| Independent supported lifts | Final success | Ever settled | Mean final position / rotation | First failure counts |
| --- | ---: | ---: | --- | --- |
| Teacher | 7/8 | 8/8 | 2.06 mm / 0.0010 rad | 1 terminal settling timeout |
| Source | 0/8 | 0/8 | 42.74 mm / 0.1439 rad | 6 base-contract, 2 retention |
| Control, no new subtype | 0/8 | 0/8 | 39.37 mm / 0.1249 rad | 8 base-contract |
| Treatment, new subtype | 0/8 | 0/8 | 43.86 mm / 0.1676 rad | 2 base-contract, 3 retention, 3 timeout |

Teacher reaches the settling predicate at least once on every case; one loses
terminal settling. This demonstrates useful physical feasibility, not universal
trajectory feasibility or a calibrated rejection value. A separate eight-case
teacher development preflight had 5/8 final successes, 7/8 ever settled and one
retention loss. Both populations and all failures remain in the evidence.

The small paired general checks use two cases each. Base succeeds on 1/2 for
source, 0/2 for control and 1/2 for treatment; hand, combined, retarget and push
checks all remain 0/2 for every actor. These counts cannot establish no-regression.

Each actor also runs all three unchanged public single deliveries and all three
original multi-transfer programs. Every actor remains **0/3 in both programs**.
The control stops at pregrasp; the treatment trips wheel unloading during grasp.
The source retains lift orientation/base failures plus one single-delivery
inference-deadline failure. That deadline is retained, not erased by a rerun with
a larger budget. The older source's 3/3 reference deliveries are historical
baseline evidence from the preceding report, not a newly repeated reference gate
under this source revision.

![Matched measured lift traces](../../paper/data/whole_body_control/grasp_lift_v15/matched_lift_trace.png)

The plot shows independent seed 3102000. Failed controllers stop early. Dashed
hand-error lines are terminal requirements, not trajectory tubes. Complete
measured trajectories, support loads, initial physics, per-case jerk/force,
retention and wheel outcomes remain in the compressed replay.

## Reward accounting and isolated learning studies

One fixed development lift (seed 3101000) exposes an incentive problem. The
original reward independently credits holding the fixed base while the hand
misses its moving target. Parking earns 43.02 total reward, of which 44.64 is
base credit after other terms are included; the successful teacher earns 62.85.
The teacher is still preferred: at discount 0.99, returns are 23.43 versus 18.84.
This is a large competing parking incentive, not proof that parking is the
optimal policy.

A diagnostic wrapper gates that fixed-base reward by the same coupled hand-pose
tracking score. It removes only `dt * 4 * base_score * (1 - pose_score)` on these
fixed-base, hand-active cases. Counterfactual totals are 2.67 for parking and
49.81 for the teacher (discounted 0.99: 5.42 and 18.31). Safety and success gates
are unchanged. This wrapper is **not** the production reward and has not been
validated on the broader masked-goal distribution.

Two fixed-case 8,192-step PPO learning audits compare independent versus coupled
base credit with identical initial actor weights. Independent credit increases
return while hand error worsens to about 61 mm / 0.247 rad. Coupled credit reaches
4.85 mm / 0.0238 rad after 6,144 steps without retention/wheel failure, then
regresses. Smaller-update continuation and lower arm exploration reach
4.40 mm / 0.0181 rad on this same training case. It still fails the original
3 mm endpoint and settling requirements. Some checkpoints also miss the active
0.01 rad base-heading success condition even while satisfying the broader
0.02 rad full-base safety bound.

A further 32,768-step low-noise, small-update continuation completes. Its best
training-case checkpoint reaches **4.13 mm / 0.0130 rad**; the post-final-update
actor finishes at 5.20 mm / 0.0115 rad. Every recorded fixed-case checkpoint still
fails strict success. At the best checkpoint the hand is still moving at up to
0.0204 rad/s, position error is falling over the final seconds, and base heading
error is 0.0150 rad. This is incomplete settling within the fixed 12-second
budget, not evidence of a hard 4 mm kinematic floor. The nominal lift duration is
5.26 seconds; extending the budget has not been used to turn the failure into a
pass.

The best training-case checkpoint is selected without inspecting independent
outcomes, then tested on the same eight teacher cases. It is still **0/8 final
and 0/8 ever settled**, with four full-base-contract stops and four timeouts.
Mean final errors improve to **13.48 mm / 0.0295 rad**, compared with the original
mixed treatment's 43.86 mm / 0.1676 rad. No retention loss or wheel unloading is
observed in these eight component rollouts. This is evidence of a useful training
direction, not sufficient task success or a safety qualification. Selection was
frozen before the long run completed; the complete selection snapshot is retained.
These diagnostic weights have no TAMP execution admission evidence and were not
substituted into the public delivery benchmark.

![Independent diagnostic transfer](../../paper/data/whole_body_control/grasp_lift_learning_audits_v1/heldout_transfer.png)

Across the two mixed pilots, the forced-lift smoke and all isolated learning
variants, this experiment adds **90,624 PPO steps**. That sum spans alternative
branches and is not the step count of a single policy. All runs have completed;
no unattended training process remains from this experiment.

These are single-case overfit/optimizer diagnostics. They do not replace the
70/30 production mixture, establish held-out success or issue TAMP admission
evidence. Optimizer and normalization restoration are recorded, while environment
and RNG resets prevent a claim of bitwise continuation. Every numbered checkpoint
and the separate post-final-update result are retained; a checkpoint captured
before the last rollout's optimizer update is not silently called the final
updated actor. W&B stays offline; model weights remain local.

## Evidence and next gates

- `paper/data/whole_body_control/grasp_lift_v15/summary.json` binds complete paired
  replay, teacher supplement, training metadata, reward audit and driver archives
  by SHA-256. All failures are retained, without model weight uploads.
- `grasp_lift_learning_audits_v1/summary.json` binds all fixed-case checkpoint
  histories, manifests, measured diagnostics, independent transfer and selection
  snapshots to the complete compressed audit archive.
- `grasp_lift_initialization_v1/summary.json` records zero-step transfer equality.
- `grasp_lift_preflight_v1.*` retains cylinder/depth/box probes and the 512-step
  forced-lift smoke run; `grasp_lift_targets_v1.png` shows full task-pose targets.
- `grasp-lift-v2-regressions.{xml,receipt.json}` records 801 fresh source-bound
  tests. Relevant lint, runtime parity and the paper build pass at source
  `4f8456e8`; experimental wrappers are separately preserved, not covered by a
  claim that the production regression receipt tested their learning behavior.

Next, validate a production-quality masked-goal reward option that avoids parking
credit while preserving legitimate base-only tasks, then compare mixed-distribution
training against the same control and the unchanged held-out/public gates.
If precision plateaus even on one case, inspect measured command lag, base/arm
coordination and trajectory phase coverage before spending more broad PPO steps.
Cylinder load transfer, general tracking, complete public delivery and original
multi-transfer execution remain open. Herman and real-world controller admission
remain later, independent gates. No hardware commands were sent.

The subsequent [failure-isolation report](mars_lift_failure_diagnosis_20261027.md)
separates weak arm tracking from incorrect base correction, preserves failed
higher-gain interventions and identifies world-origin sensitivity. It also
corrects an accounting interpretation: the external coupled-reward wrapper uses
the previous tick's goal, rather than the goal scored by the environment.
Historical returns above describe that wrapper; same-tick parking/teacher totals
are 2.48 / 49.34. This does not reverse the incentive finding, and a native
implementation still requires timing/mask tests and fresh mixed training.
