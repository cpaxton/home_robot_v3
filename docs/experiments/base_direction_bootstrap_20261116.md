# Base-direction coverage and output-only training diagnostic

Started 2026-11-16, simulator only. The prior
[coupled readout](coupled_tracking_20261116.md) remains unchanged: the fixed
base-rate ablation gives 9/10 small general cases, 5/10 larger general cases,
and 6/6 original open-jaw TAMP grasp-pose completions. No loaded task or
hardware controller is qualified.

## Specific failure and hypothesis

Two train-only attribution replays decompose the candidate's output into
current/future base-error, announced-motion and measured-rate contributions.
The explicit direct rate path is disabled. With positive yaw drift, the actor
requests further positive yaw while the teacher requests a negative correction.
The base-only yaw-error gains are negative in this region. On the 30 mm case,
measured yaw reaches 0.0949 rad at tick 80 while the normalized actor yaw request
is +0.0841; the teacher correction is -0.4. This is feedback with the wrong sign,
not a demand for unnecessary millimeter base precision or an angle-wrap bug.

The original bootstrap had only one base-only train reference: straight forward
translation. It did not sufficiently identify heading correction in that mode.
This distribution explanation is a hypothesis supported by the new attribution,
not proof that a single extra case will fix general control. The archived
[plot](../../paper/data/whole_body_control/base_direction_bootstrap_v1/base-yaw-attribution.svg)
shows commanded corrections and measured yaw on both train references.

## Admitted training coverage

Eight fresh train-only base references cover forward/reverse translation,
positive/negative pure yaw, and all four translation/yaw sign combinations.
Translation is 15 cm and turn magnitude 0.24 rad; reference geometry obeys the
same native limits. Pure turns legitimately begin inside the XY region while
outside heading acceptance. Hand control is inactive for these base-only cases.
They are a separate training distribution, not relabeled v2 benchmark cases.

At runtime `57e8e73b`, all eight pass physical teacher admission, and all six
unchanged original TAMP grasp-pose teacher cases pass. All stops confirm and no
scored safety flags occur. Admission and training use the exact source bound to
the existing 1,433-test receipt, verified before launch. No runtime code changes
or environment upgrades are made for this diagnostic.

## Fixed training protocol

Managed job `20261116_112422_5b1b1f`, seed 94, performs at most 1,024 supervised
updates, batch 64, fresh Adam at learning rate 1e-3. It initializes from the
same serialized fixed-weight base-rate-ablation actor. Only its two base output
rows in the gain head are optimized. Arm/gripper output rows, GRU, encoder and
context weights are frozen and checked bitwise before export. Deployment remains
one ordinary eight-output actor with no teacher fallback or added smoothing.
This is a base-output optimization diagnostic, not PPO or separate deployed skills.

Each batch has 16 new-direction rows, 16 existing-general rows and 32 TAMP rows.
Replay chooses cases uniformly and then observed phases uniformly. General bins
are moving reference, first two seconds after its end, and later hold; TAMP bins
are announced motion/hold. All histories come from continuous teacher and fixed-
initializer rollouts on training cases only. No phase teleport/reset, synthetic
grasp, stop clearing or audit action labels are used.

This is **50/50 general/TAMP replay samples**, not a claim of 50/50 PPO physical
control ticks. It does not implement SuperMimic's adaptive 50/40/10 sampler or
checkpoint-based phase resets. It tests simpler case/phase coverage first.
Changing both coverage and the optimized parameter subset makes this an
initialization diagnostic, not an isolated causal comparison of sampling methods.

The fixed final is compared with its exact initializer on 16 fresh base audit
cases, both existing general cohorts, and the original six TAMP cases. Teacher
results on the new audits are measured separately. The initializer passes 0/16
new base audits. No failed audit is added to training or removed from denominators.
W&B logging is offline. There is no automatic budget extension or promotion.

## Status

The run completed all 1,024 updates with 15,115 labeled observations from
36 continuous training rollouts. Bitwise frozen-weight checks pass, but the
candidate regresses: small general **9/10 → 4/10**, larger general **5/10 → 2/10**,
TAMP **6/6 → 1/6**, and new base audits **0/16 → 0/16**. It is rejected; the
stronger initializer remains retained. Frozen arm weights do not preserve arm
tracking when changed base commands alter the physical trajectory.

The new base-audit teacher passes **15/16**. The remaining pure reverse case
finishes with 2.80 mm XY error but 0.1304 rad heading error, outside the unchanged
5° contract, with no unsafe-contact or wheel-lift flag. It remains counted.
This does not establish a reference geometry failure; sampled FK admission
and dynamic controller success remain separate.

The result rejects this fixed-budget output-only intervention, not the broader
coverage hypothesis. A frozen context/gain representation and imitation targets
may be insufficient here; those explanations need separate tests. Next, jointly
optimize the retained actor through dense tracking/stability/contact PPO rewards,
with no teacher action loss. Shorter, teacher-checked training episodes improve
case exposure; held-out horizons/contracts remain unchanged. W&B is offline
(`lgzm86j1`). Compact results bind all cases, runner/protocol and raw-file hashes.
Replay observation arrays were not saved; collection metadata and exact runners
are retained, so this run does not provide an offline replay dataset.

## Joint PPO continuation

Job `20261116_113644_f4890e` is training the retained actor jointly, with a
32,768-decision limit (first 512 decisions update critics only), seed 95,
actor learning rate 2e-5 decaying to 5e-6, initial normalized action standard
deviation 0.02, and unchanged grouped rewards. No teacher actions enter its
loss. All fourteen general training references pass teacher admission at the
12-second training horizon. The separate 10-second preflight rejects a case
before any PPO update and remains preserved. Held-out horizons stay 25 seconds.

The outer general/TAMP mixture balances actual physical control ticks; within
general, eight directional base cases and six original coupled cases have an
equal requested share. Shorter episodes improve case exposure without clearing
faults, advancing the task executor, or changing held-out pose/safety thresholds.
First-rollout arrays, episode identities and source/actor/protocol hashes are
logged; checkpoints are backups, not post-hoc selection candidates. W&B is
offline (`k9sg2g7l`). Judge the fixed final, not noisy training successes.

The run completes **32,768 decisions / 38,096 physical ticks**, including
512 critic-only and 32,256 actor-update decisions. Final scores remain **9/10
small general, 5/10 larger general, 0/16 new base audits, and 6/6 TAMP**. It is
not promoted: preservation is useful, but base-only performance has not improved.
All training case families are exercised; the coverage gap is addressed without
closing this control gap.

A synthetic local derivative check at the same stationary base-only posture
finds yaw-error gain **-0.1461/s** in the retained controller and **-0.1174/s** in
the PPO final. Lateral-error-to-yaw gain remains approximately **-24 rad/s/m**.
These are derivative diagnostics, not physical rollouts or stability certificates.
The wrong local correction sign persists after this bounded fine-tuning run.

A separate fixed-weight simulator diagnostic now tests base-only feedback sign
and symmetry constraints plus an explicitly declared coarse hold region. It is
an architectural intervention, **not an improved raw learned checkpoint**.
Active-hand commands retain their original mapping, native speed/slew/force/stop
limits remain active, and no teacher or private FK witness supplies actions.
The diagnostic is not exported or safety-promoted.


## Feedback representation intervention

The fixed-weight sign/symmetry and coarse-region probe completes **24/24**
trajectory base cases (eight train, sixteen audit), with all stops confirmed,
no unsafe-contact or wheel-lift flags, maximum final XY error 12.95 mm and
heading error 0.04893 rad. This is a manually specified architectural prior
using learned magnitudes, not a PPO learning improvement or global stability
certificate. The actor's parameter tensors remain unchanged.

The source `25dd4798` implementation reproduces the 24 base passes through a
serialized bundle. It also gives **10/10** small general, **6/10** larger general,
**6/6** original TAMP grasp-pose, and **4/4** base-to-hand handoffs during motion.
Handoffs keep physics and command history continuous. Active-hand commands
retain their original mapping; native safety/slew/grip authority remains active.

The deliberately harder static endpoint cohort passes **2/8**. All eight reach
the 5 cm XY region, but six lateral/diagonal cases miss final heading convergence;
there is no unsafe-contact flag in those point-goal cases. This is an important
remaining gap for interactive base markers. Timed executable paths and raw
endpoint goals must not be described as the same capability.

A compatibility review finds that old v2 loaders would ignore a new optional
feedback field. Source `823b3d1d` therefore exports configured actors as **v3**,
which older runtimes reject, and updates policy/TAMP/viewer/validator dispatch.
Options are checksum-bound in both weights and manifest; default legacy behavior
is preserved. The first full regression attempt is canceled when superseded by
this required format fix, and its verified command group is terminated. Its
partial XML is never reused. Fresh source-bound v3 regressions pass **1,451 tests**, zero failures/errors/skips,
in **990.82 s**, and the receipt verifies against the current source. The paper
builds successfully (59 pages). The complete v3 native readout reproduces
24/24 base, 10/10 small general, 6/10 larger general, 6/6 TAMP, 4/4 moving
handoffs and 2/8 static endpoint goals. The public policy loader is used and
an actual older v2 loader rejects the configured bundle before inference.
See [final compact evidence](../../paper/data/whole_body_control/base_navigation_feedback_v1/).

Next work is larger coupled trajectory coverage and a proper local reference
path for raw base endpoint commands. General training uses pyramidal contacts
while TAMP fixtures use elliptic contacts/impratio 3; any future agent-facing
checkpoint needs explicit per-plant export/admission evidence or aligned plant
configuration. Existing strict source/model qualification must not be bypassed.
No hardware motion or safety promotion follows from these development results.
