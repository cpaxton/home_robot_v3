# Coupled Mars/Stretch tracking and unloaded-policy diagnosis

Updated 2026-11-16. Mars remains the first hardware target; all work in this
report uses simulation. Neither full pick/place nor Herman motion is qualified.
Compact results, runner/protocol text and target plots are in
[the evidence directory](../../paper/data/whole_body_control/coupled_tracking_v1/).
Raw traces, source archives, W&B runs and weights remain ignored.

## Planning scope and achievable references

Runtime `db6b6964` adds explicit native `pregrasp` / `grasp_pose` planning scope,
with historical `full_task` unchanged. The single-use plan seals that scope.
The unloaded request checks/replans its actual motion, without requiring a
prospective loaded lift. Collision/support screening, executor grip authority,
measured hold, final pose and confirmed-stop requirements remain active.
The current prefix endpoint is a terminal simulator operation requiring reset;
it is not a continuous grasp/lift handoff or successful object transfer.

On the original six Mars cases, teacher pregrasp passes **6/6**, teacher
open-jaw grasp-pose passes **6/6**, and **12/12** stops confirm. The preceding
full-task-scope report remains **5/6 at each boundary**. This is an explicit
operation-scope correction, not a changed pose threshold or removed failure.

The coupled v1 distribution has six train and four audit cases per robot:
base-only, hand-only with free base, constrained coupled motion, partial base
masks, fixed-world-hand self-motion, and a mid-motion goal reversal. Continuous
full-robot FK witnesses are private reset/admission data. Actors receive measured
history, masked pose preview and announced twist, never those joint witnesses.
Reference speed/acceleration/jerk bounds do not certify dynamic feasibility.

Mars and Stretch each admit **10/10** and complete **10/10 teacher episodes**,
with all stops confirmed and no recorded unsafe-contact, wheel-lift, invalid-
physics or retention flags. Worst final active-hand errors are **0.125 mm /
0.0000286 rad** for Mars and **1.55 mm / 0.000371 rad** for Stretch. The partial
XY/yaw-free Stretch case required retaining coupled base authority; it previously
stalled at 5.07 mm and reaches 0.98 mm after that fix.

All three readouts share the exact `db6b6964` source digest. Its fresh full
regression run passes **1,421 tests**, zero failures/errors/skips, in 863.76 s.
These deterministic development/audit cohorts are small; repeated replay is not
independent reliability evidence. Stretch's static support polygon includes its
modeled caster, but dynamic unloading metrics cover its two drive wheels,
not separate caster unloading. Mars uses its existing three-support monitor.

## Direct PPO pilot: negative and informative

Managed job `20261116_092352_9babab` completes **4,096 PPO decisions / 4,314
physical ticks**, seed 91, from the original C1 GRU actor. Critics and optimizer
are fresh. The actor learning rate decays from 3e-5 to 1e-5; three epochs and
batch size 64 are fixed in the protocol. This initializes from the GRU candidate,
not the stronger retained legacy MLP with a different architecture.

The mixture records 2,292 general ticks and 2,022 TAMP ticks (53.1% / 46.9%).
Only five general episode starts fit, leaving the sixth train case unseen;
13 TAMP starts repeatedly encounter early failures. Whole-body general success
stays **0/10 → 0/10**, and open-jaw TAMP grasp-pose stays **0/6 → 0/6**.
Several position errors improve while orientation errors worsen. This budget
provides no evidence that more of the same PPO sampling will solve the problem.
The episode callback was not attached; no per-episode callback log is claimed.
The saved first rollout and scheduler counters provide the exposure evidence.

Two preserved setup attempts perform zero PPO updates: one admission reader
needed the checksum-bound full command rows rather than compact summaries; the
other supplied mixture weights as a dict rather than an ordered vector. Neither
extends the 4,096-decision training budget. W&B is offline (`qnl4di77`).

## Teacher-action bootstrap: hand accuracy improves, settling does not

Job `20261116_093940_2eef6a` starts again from the original GRU actor, not the
PPO final. Two rounds of teacher/actor-state collection use only six general
and four TAMP train cases. Each round performs 512 masked-action-MSE updates,
batch 64, learning rate 2e-4, with balanced general/TAMP rows. No audit labels
are used. The collection retains 6,714 general and 1,583 TAMP rows; the learner
receives history/preview/motion only. No teacher is called at final inference.
W&B is offline (`40x20y16`); this is **1,024 supervised updates**, not PPO.

TAMP open-jaw grasp-pose reaches **5/6**, with successful final errors
1.12–1.72 mm and 0.0028–0.00945 rad. Audit-right fails
`object_changed_replan`; it remains counted. Collection's oracle instrumentation
is not policy evidence; final public-tool readout uses actual actor inference.
General success remains **0/10**: seven cases lack quiet settlement and three
stop on self-contact. Accurate terminal hand poses alone are insufficient.

## Why the base hunts

Five command/measured-state replays show yaw requests reversing sign on every
50 ms tick late in the episode: **10 Hz oscillation**, with mean absolute
requests 0.043–0.057 rad/s. Signed mean measured yaw is near zero. The problem
is repeated corrective reversal, not the magnitude of a slow steady turn.
The two traced contact stops involve `base_link` / `link3`, around 31–35 N.

The actor has an explicit learned gain-times-feedback path. A fixed-weight
ablation removes instantaneous joint/base velocity columns **only from its two
base rows**. The arm rows retain all measured-rate feedback, and the GRU still
receives measured history. Base outputs retain pose errors, hand errors and
announced twist. Native base servo feedback remains unchanged.

On the same ten v1 cases for the full 25-second horizon, raw bootstrap weights
pass **0/10**, while the ablation passes **9/10**, with no self-contact flags.
The base-only case becomes worse: approximately 2.22 rad heading error. Therefore
this is an opt-in candidate, not a replacement or a generally qualified actor.
It strongly implicates the learned base rate path in hunting without proving a
single scalar gain is the entire cause. It uses no new training or smoothing.

A separate diagnostic using the existing executor's explicit terminal hold and
one bounded reacquisition passes **10/10**. That is controller-plus-hold behavior,
not raw actor stability. It must not replace either 0/10 or 9/10 in comparisons.

## Coarse base region and next checks

Runtime `57e8e73b` preserves v1 and adds v2: **5 cm XY / 5° base acceptance**,
with hand acceptance unchanged at **3 mm / 0.02 rad**. Required base paths move
15 cm, so staying still cannot pass. IDs/seeds are distinct from v1. Small
fixed-hand nullspace cases are explicitly hand-hold diagnostics; their base
motion lies inside the acceptance region and is not required-base progress.

The opt-in v2 reward has no base heading precision gradient inside that region;
hand tracking, reference-velocity, motion/jerk and force costs remain active.
Frozen TAMP tolerances and the historical reward remain unchanged. Enlarging a
pose region does not excuse chatter, drift, contact load or wheel unloading.
The actor ablation is serialized in weights and manifest, cross-checked on load;
legacy bundles preserve their original behavior.

All 20 proposed v2 paths pass sampled geometry admission. Physical teacher
readouts pass **8/10 Mars** (two reverse/retarget self-contact stops) and **9/10
Stretch** (one reverse coupled case finishes at 3.14 mm and times out). These
failures remain counted; geometry admission is not dynamic tracking admission.
The serialized fixed-weight candidate reproduces **9/10 v1**, reaches **5/10
v2**, and completes **6/6 original open-jaw TAMP grasp-pose cases**, with all
six stops confirmed. Thus the base-rate ablation also repairs the previous
bootstrap audit-right approach failure, without adding teacher inference or
changing task tolerances. This selected deterministic cohort is not broad
qualification. Base-only, larger coupled motion and reverse self-contact
remain gaps. See [v2 compact results](../../paper/data/whole_body_control/coupled_tracking_v2/).

The first readout runner completes all teacher/general episodes, then fails on
an unsupported TAMP keyword before any TAMP decision. A completion attempt
lacks a multiprocessing main guard and also executes zero actor decisions.
Both setup failures remain archived; a guarded continuation executes only the
six missing TAMP cases, preserving the original twenty general results.
Fresh full regressions are still running. No new training, budget extension,
hardware motion or promotion follows.

The [SuperMimic research handoff](../plans/supermimic_controller_training.md)
recommends physically valid phase/reset coverage before a separate dynamics-
model experiment. Keep the outer general/TAMP mix; target the underexposed
settling and grasp transitions without leaking audit cases. Base-only stability
and object-sensitive final approach remain explicit gaps, followed by loaded
retention/support, continuous task execution and hardware feedback qualification.
