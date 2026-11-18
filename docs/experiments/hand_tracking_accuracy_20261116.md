# Mars hand-tracking accuracy: false equilibria and unsafe posture choices

The fixed 24-episode audit identifies two distinct problems: the learned actor
can cancel its own hand correction at a wrong pose, and both actor and teacher
can choose a self-colliding arm posture for otherwise admitted hand goals.
Increasing simulation horizon alone does not address the observed steady errors.
No controller weights, physics, reference contracts or safety thresholds change.

## Protocol and reproducibility

Managed job `20261116_175205_240a6d` uses the immutable runtime
`4b3c405960876825c57f5925300f001ee1d1ae78` on one offline CPU15 worker, numerical
threads one, CUDA hidden. The CPU14 viewer stays available. Source state and
actor tensors are verified unchanged. The existing 1,538-test source receipt
remains valid; it is not reported as a new test run.

The nine selected cases are four repeated mask handoffs and five larger coupled
cases, including all four larger failures. This is a failure-enriched diagnostic
selection, not a generalization or reliability estimate. Each is replayed with
the unchanged actor and analytic teacher. Three quiet failed cases additionally
get two fixed-weight interventions, giving **24 episodes** total:

- `base_error_zero`: remove direct current/future base-error contributions only
  for active-hand commands while the base is inside its coarse acceptance region.
  Recurrent context, hand/reference/rate terms, real targets and masks stay intact.
- `hand_gain2`: double only current/future hand-error contributions before tanh.
  Context, rate feedback, reference motion, gripper authority and native guards
  stay intact. This is not a trained or exported policy.

The observer reconstructs the actor's output from its actual gain head and
feedback signals, asserting numerical agreement. Jacobian queries and actor
attribution leave qpos, qvel, actuator ctrl, arm targets and history unchanged.
All nine actor replays exactly reproduce the preceding readout's primary
accuracy, force, wheel, time and outcome metrics. All **24 stops confirm** using
unscored physics with the unchanged 0.75 s physical budget. No hardware or PPO.

Complete cases, exact runners, source/weight/trace hashes and the four-panel
figure are in [hand_tracking_deep_dive_v1](../../paper/data/whole_body_control/hand_tracking_deep_dive_v1/README.md).
Raw state/observation traces remain ignored. Offline W&B run: `bjvggd38`.

## 1. Quiet misses are policy false equilibria

| Case | Actor position / rotation | Teacher position / rotation |
| --- | --- | --- |
| Mask 1 | 6.258 mm / 0.01708 rad | 3.083 mm / 0.000690 rad |
| Mask 3 | 11.781 mm / 0.03189 rad | 1.743 mm / 0.000396 rad |
| Larger coupled 4 | 4.606 mm / 0.01089 rad | 0.000189 mm / 0.000000376 rad |
| Larger coupled 7 | 10.089 mm / 0.02833 rad | 1.649 mm / 0.000375 rad |

The 3 mm / 0.02 rad contract is unchanged. Teacher Mask 1 remains a failure;
its 0.083 mm position miss is not rounded into success. The teacher remains a
comparator, not a performance ceiling or universal execution oracle.

In these four actor failures, late-window base/hand preactivation vectors have
cosines between **-0.99999992 and -0.99999999999**. Substantial opposing terms
cancel; the net command norm is only about **0.00026%–0.022%** of their combined
norm. Measured-rate terms are small and reference twist is zero during the hold.
The tanh output is near zero, rather than saturated. Hold and stop flags remain
false before the episode deadline. The executor is not suppressing an otherwise
useful correction.

For example, Mask 3 retains 11.74–11.79 mm error from 10 s through 25 s. Its
base- and hand-pose terms contain roughly opposite 0.2–0.4 preactivation values
in several outputs, yielding actions around 1e-6. Waiting longer is not a
substitute for correcting that learned feedback field.

These target sets are jointly consistent, not competing desired poses. Their
terminal FK witness reproduces the requested hand pose, with joint margin.
At failed actual states, the normalized whole-body Jacobian has rank six,
while the arm-only Jacobian has rank five. A bounded local solve can close the
linearized corrective twist with normalized actions below about 0.35. This is
local kinematic evidence, not a dynamics or collision-free execution certificate;
no Jacobian solution is executed or supplied to the actor.

The arm needs coordinated base assistance for some task directions. A coarse
base acceptance region does not remove that authority; hand error must still
be allowed to drive the whole body inside that region.

## 2. Actuator lag and gravity sag do not explain these quiet misses

The last five seconds show small commanded and measured joint rates, substantial
joint-limit margin (at least 1.19 rad), and less than 8.75% of the simulator's
arm force limits. Late joint target errors peak at 0.182 mrad in Mask 1 and stay below
0.0047 mrad in the other three quiet misses; gravity-compensation trim remains around 0.002 rad on the loaded arm
joints. The policy asks for almost no corrective motion while the hand is wrong.

All calculations use pre-action samples, avoiding the misleading zero target
error that a terminal stop's measured-position latch would create. This finding
is scoped to these unloaded, quiet simulator failures; loaded sag, force-window
randomization and hardware identification remain separate work.

## 3. Fixed-weight probes establish sensitivity, not a replacement controller

| Case | Unchanged actor | Hand feedback ×2 | Base-error terms removed |
| --- | --- | --- | --- |
| Mask 1 | 6.258 mm | **1.558 mm**, passes | 0.000243 mm, passes |
| Mask 3 | 11.781 mm | 3.881 mm, fails | 7.914 mm, unsafe-contact stop |
| Larger coupled 7 | 10.089 mm | 3.140 mm, fails | 18.881 mm, fails |

Doubling hand terms moves all three final poses closer, supporting a feedback
coordination/priority problem. It also increases some overshoot and jerk.
Mask 1's peak active-hand error grows **18.36 → 51.10 mm** before its improved
final pose; Mask 3 yaw jerk increases 25.76 → 71.50 rad/s³, and coupled 7 increases
19.57 → 59.20 rad/s³. Final millimetres alone would hide that regression.

Removing base terms succeeds in one case, causes a new self-contact stop in
another and worsens the third. Base information remains useful. Both probes
remain diagnostic, without export, promotion or changes to tolerances.

## 4. Contact failures are different joint paths through the robot model

Both free-base failures (`...:hand_free_base:6` and `:9`) hit **`rl_link3` against
`rl_base_link`**, under actor and teacher. The actor's scored force maxima remain
34.65 N and 31.55 N. Teacher maxima are 37.10 N and 32.43 N. The current 30 N
experimental ceiling is retained; all four episodes stop.

Read-only reconstruction of actual poses reproduces hand position and rotation.
Convex separating-axis checks in relative geometry frames give initial nominal
witness separation lower bounds of **28.10 mm and 26.23 mm** for those proxies.
Relative transforms along those initial paths stay invariant to about 5.6e-17.
Case 9's retargeted nominal witness also has positive separation (57.78 mm for
the actor's replan and 52.60 mm for the teacher's). Actual execution chooses
other arm configurations and brings the proxies into contact.

A positive projection gap proves separation; it is not exact Euclidean
clearance. Negative projection gaps alone are not used as collision proof:
MuJoCo's active contacts identify the actual penetrating pair. No collision
geometry is shrunk, excluded or retuned to improve this result. Current proxies
are conservative vendor-box/visual envelopes and a convex chassis hull;
physical fidelity remains a separate model audit.

This is a safe-allocation gap in the current model. A hand-only command permits
many joint/base solutions, and the resolved-rate teacher lacks self-clearance
awareness. Imitating those unsafe teacher paths alone will not solve it. Keep
the safety stops, train safe whole-body alternatives and compare against a
collision-aware teacher or the admitted witness as an optional reference.

The contact observer stays attached during unscored stopping. Its raw
`peak_contact` can therefore exceed the scored `metrics.peak_contact_force`;
those values are kept distinct. In actor case 9, the post-stop observer peak is
32.16 N versus the scored 31.55 N.

## 5. Audit the objective actually returned by the training wrappers

`GeneralTrackingEnv` and `CoupledTrackingEnv` discard the native environment's
scalar reward and return `RewardMeter`'s three groups. The TAMP primitive also
uses that meter. Its pose term is fixed at
`exp(-position_error/0.01 - rotation_error/0.1)`, with an additional velocity
bonus of up to 0.5. It does not consume the environment's `tracking_precision`
coarse-to-fine schedule, pose-progress weight or endpoint-motion shaping.
Changing only those environment flags will not change this wrapper objective.

At a stationary hold with perfect base tracking and 0.02 rad hand error, the
tracking density is approximately 2.50, 2.30 and 2.11 at 5, 10 and 20 mm hand
position error, versus 3.0 at an exact pose. Perfect tracking still has the
higher reward; this is not proof that a wrong pose is the mathematical optimum.
It does show why explicit contract-scale shaping and pose-gated quiet credit
should be implemented in the reward the learner actually receives.

This is a current-source objective finding, not a claim that every historical
weight was trained with this exact meter. The existing actor includes prior
imitation/training and fixed-weight control transformations. The replay/probes
establish its present failure mechanism independently of that history.

## Next implementation and intermediate tests

1. Give the wrapper an explicit serialized reward configuration. Add continuous
   coarse/fine hand terms tied to the actual pose contract, measured progress
   toward the same active reference, and pose-gated quiet/velocity credit. Test
   3/5/10/20 mm and corresponding rotation errors, moving references and goal
   changes. Keep smoothness/force terms and their curriculum explicit. Avoid
   silently tuning unused native reward flags.
2. Generate analogous feasible mask/coupled cases with different seeds, including
   independent base/hand perturbations, near-target corrections and safe free-base
   allocation. Preserve ordinary tracking and the outer 50/50 general/TAMP
   physical-tick balance. Keep the fixed audit seeds out of fitting.
3. Run a bounded direct-PPO mastery comparison before scaling. Compare full
   trajectories, settling, final pose, self-contact loads and measured jerk
   against the retained actor and teacher. Preserve all failures. A gain boost
   or a longer timeout is not the selected replacement policy.
4. Add self-clearance/posture coverage and collision-aware reference comparisons
   for the free-base cases. Preserve SE(3) hand/SE(2) base masks and task-agnostic
   tracking; loaded grasp/carry and real Herman remain separate gates.
