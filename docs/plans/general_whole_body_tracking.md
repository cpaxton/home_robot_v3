# General whole-body tracking and continuation audit

The controller accepts one masked SE(2) base pose, one masked SE(3) pose per
supported hand, and the robot's gripper coordinates. It can receive a timed
preview or a changing pose goal. Mars is the first physical target; Stretch
uses the same task contract with its own model and native limits. This work is
simulator-only and does not authorize Herman motion.

TAMP is a downstream command source and acceptance suite. Its approach, lift,
carry and placement paths must be executable, but the controller must also
handle ordinary base motion, hand-only targets that may require base movement,
combined constrained targets, rotations, translations, goal changes and
recovery. A grasp-planning IK witness establishes geometric admission; the actor
still receives pose targets and emits bounded velocities.

## First gate: preserve moving-goal performance during updates

The completed goal-gated settling experiment retains accurate stationary holds,
but both 16,384-step continuations regress from the initial checkpoint on moving
goals. Both start from exactly matching validation behavior. Recorded-action
counterfactuals identify larger effects from changed policy weights than from
changed normalization on the same recorded observations; these are offline
comparisons, not counterfactual physics rollouts. The continuation also restarted
a fresh optimizer at 5e-5 after the source run finished at 1e-5.

`configs/benchmarks/mars_general_continuation_v1.json` specifies four sequential
8,192-step pilots from the same original weights and seed:

| Variant | Learning-rate decay | Observation statistics |
| --- | --- | --- |
| adaptive_restart | 5e-5 to 1e-5 | adaptive |
| frozen_restart | 5e-5 to 1e-5 | frozen |
| adaptive_small | 1e-5 to 2e-6 | adaptive |
| frozen_small | 1e-5 to 2e-6 | frozen |

All variants use a fresh optimizer, goal-gated settling, the corrected current
retention-error cost, the existing 70% reach / 30% manipulation reset mixture,
identical physical limits, 0.05 randomization, exploration and 6 s training
horizon. Do not change task distribution or architecture while isolating these
factors. Frozen statistics require an initializer and do not freeze PPO weights.
The native action mapping and runtime normalization implementation stay unchanged.
No checkpoint is promoted automatically.

Launch requires a fresh, source-bound full regression receipt after merging the
latest harness. Record exact commands, source states, configuration/weight
hashes, sampled episode/step shares, scalar training metrics and all failures.
Keep old training provenance intact; a later reward/harness fix does not
requalify old reports.

## Validation beyond TAMP

Evaluate the original checkpoint and every completed pilot against:

- Six strict stationary/nearby/wrist probes at two existing development postures.
- Seeded ordinary base-only, hand-only and combined pose targets, with two cases
  per mode for the initial diagnostic. Generate target poses by FK from sampled
  reachable configurations; preserve failed admission and rollout outcomes.
- Two mid-motion retarget cases and two bounded hand-push cases, using the same
  announced-goal and wrench timing semantics as the general environment.
- The unchanged three public TAMP development programs with no interstage reset,
  plus the existing per-category teacher comparison during training.

Use independent, fixed development seeds and pair controller/teacher resets.
Report admission, conditional execution and end-to-end denominators separately.
Measure arm position/orientation, active base-region/heading error, goal changes
actually delivered, force, jerk, retention, wheel unloading, and measured quiet.
Teacher failures remain visible; they do not establish geometric infeasibility.
Two cases per mode diagnose behavior and do not qualify a robot or population.
The general hand contract stays 3 mm / 0.02 rad, while ordinary base goals retain
their declared region. Fixed-base manipulation still obeys the 1 cm / 0.02 rad
safety boundary. Benign contact is allowed within existing force/support limits.

## Next gate: expand the distribution after updates are reliable

Within the tracking share, explicitly cover near-constant-orientation Cartesian
translations in all reachable directions, near-constant-position rotations,
combined motions, and terminal holds. Retain base-only, hand-only and combined
masks, rather than conditioning every episode on a fixed base. Sample new
training configurations and paths; do not optimize against frozen public
regression trajectories. Add TAMP-shaped motions as a bounded subset and keep
ordinary goals in every validation comparison.

Increase difficulty and randomization only when accuracy and settling pass at
the preceding level. Keep smoothness/energy terms subordinate to completing the
active pose contract, with native speed/acceleration and force/support guards
unchanged. Loaded retention, public task success, observed-agent testing and
Herman remain independent subsequent gates.

## Recovery and translation-coverage follow-up

All four continuation pilots completed before the reboot. The smaller/frozen
setting preserves position better but still fails angular and public task gates.
The next matched pair adds reachable fixed-orientation translation coverage in
a bounded part of the existing nearby-reach share, retaining the 70/30 macro mix.
See the [recovery report](../experiments/mars_recovery_translation_20261026.md) for
full negative results, the corrected action audit, target export and physical
translation probes. Persistent scripts preserve completed variants and numbered
checkpoints; recovered optimizer continuation resets environment/RNG state and
is explicitly not claimed bitwise identical. No candidate can promote itself.

The matched pair has now completed with no task gain: 1/8 translation success
for source/control/treatment and 0/3 public programs each. The immediate acceptance
gate is an actual agent delivery through public tools, alongside the general
coverage above. Reference execution passes one development delivery; the learned
backend now reaches grasp and begins lift in explicit gain diagnostics but still
fails loaded orientation/base tracking. See the
[integrated report](../experiments/mars_agent_controller_20261026.md). Prioritize
loaded transition coverage and first-failure evidence over longer unmeasured
continuations; offline gain scaling adds no PPO training or qualification.
