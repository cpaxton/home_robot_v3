# Tracking and held-object training

The opt-in `tracking-retention` preset extends the existing controller objective.
It is simulator research, not public TAMP, agent, learned-policy or hardware
qualification. Existing frozen cases, stop guards, effort limits and the TAMP
readiness supervisor are unchanged. A teacher failure is not an infeasibility
label for this experiment. The older qualification reports remain tied to their
original source snapshots.

## Sampling and objective

| Reset category | Probability | Purpose |
| --- | ---: | --- |
| Reach/tracking | 70% | Existing reachable whole-body goals, retargets and disturbances |
| Pregrasp | 10% | Approach, physical closure and lift |
| Carry | 12% | Loaded tracking; 25% of these resets hold the starting pose |
| Place | 5% | Supported placement and executor-authorized release |
| Disturbance | 3% | Retention and recovery under bounded, ramped hand wrenches |

These are reset probabilities, not guaranteed rollout step shares. Log both
actual episode and step shares, initialization failures and per-task reward terms.
The preset requires at least 50% tracking resets even with custom weights.
`--task-mix legacy` preserves previous sampling and observation defaults.

Twenty percent of tracking resets request retention with an empty hand. These
remain tracking tasks with the same pose objective; the distant object is placed
by the ordinary reach reset, never removed during scored motion. They provide
closed-empty counterexamples without mislabeling grasp intent as success.

The existing tracking, progress, success, slip, force, jerk, action and endpoint
motion terms remain unchanged. The held-object estimator uses a separate BCE
optimizer, with no shared gradients or bonus added to PPO. Each category keeps
separate paired teacher validation and no-regression checkpoint selection.

## Observations and occupancy estimation

An optional, versioned history suffix contains one `retain_object` command and
measured gripper velocity divided by its speed limit. Existing gripper position,
command error and previous actions stay in the named proprioceptive prefix.
Intent means maintain a grasp, not that one exists. It does not grant release
authority. Existing bundles retain their old layout and exact resume rejects a
layout change.

The advisory occupancy estimator consumes up to ten samples of gripper-local
encoder/command history. It excludes arm motion/actions, object pose, task targets, phase one-hot, simulator contacts,
mass, friction and solver forces. Simulator contacts produce supervision through
`info`, outside actor observations: positive after 100 ms of bilateral payload
contact; negative after 100 ms clearly empty without bilateral contact. Ambiguous
and invalid-physics transitions are masked. This estimates contact occupancy,
not force closure, stable orientation or payload capacity. An object can still
be held while violating the stricter 0.1 rad retention contract.

One in five vector-slot episodes is reserved for predictor audit; its
samples never update that head. Report positive/negative counts, held recall,
empty-hand false positives and Brier score. Checkpoint evaluation also scores
these metrics on separate validation seeds. Missing classes remain unsupported.
This is not independent probability calibration or an unseen-object audit.

Auxiliary minibatches balance held and empty labels and reserve a quarter of
samples for retention-command/empty-hand examples when available. This does not
change the policy's reset distribution or reward. These altered class priors are
another reason the raw sigmoid output requires independent calibration.

`ManipulationPolicy.estimate_held_object(observation)` returns a probability,
availability and explicit `calibrated=False`, `advisory_only=True`. No threshold
changes an action, overrides a guard or authorizes release. Classifier weights
and semantics are checksummed in the policy bundle; replay buffers are not
exported.

## Randomization and compatibility

The preset samples the existing two primitive payload identities and an
episode-fixed mass in 20–100 g at full difficulty, with inertia scaled by mass,
COM offsets up to 2 mm per axis, and the existing object-only friction variation.
The randomization scale contracts the mass interval around 60 g and scales COM
and friction variation. At 0.05, mass is 58–62 g and COM offsets are at most
0.1 mm per axis. Geometry, finger friction, gains and effort limits are unchanged.
Object dimension/shape expansion and calibrated actuator randomization remain
future work. A failed physical reset remains visible in admission denominators.

Validation, not elapsed time, advances the existing curriculum. Frozen public
TAMP catalogs retain their exact original properties. The new sampler does not
change those assets or turn a failed frozen case into a success.

`--initialize-from BUNDLE` can expand a compatible legacy policy by exactly two
history columns. MLP and transformer input weights for those new columns start
at zero; old actor/critic weights and normalization are preserved. This is a
warm-start with a fresh PPO optimizer, not exact resume. Robot, model, action,
preview and architecture checks still apply; older incompatible robot contracts
are not silently upgraded. `--resume-from` continues a matching extended bundle;
the advisory head restores compatible weights but starts a fresh head optimizer/replay.
The final gripper-only head has a separate version from the development head
that included arm/action channels; a feature-set change explicitly reinitializes
that head without discarding the PPO optimizer. Legacy advisory bundles remain
readable for comparing the failed experiments.

Use the teacher-driven diagnostic to check closed-empty counterexamples and
loaded holds on identical new seeds:

```bash
taskset -c 14 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python scripts/audit_held_object.py --policy BUNDLE \
  --output /tmp/held-object-audit.json --episodes 6 --seed 510000
```

It preserves every admission failure and records contact labels, encoder values
and predictions. Its frames are correlated, and its two-second observation
windows are not full-task or probability-calibration trials.

## Run and acceptance

Start with one CPU worker on this machine. For example, a scratch diagnostic:

```bash
taskset -c 14 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m emet.rl.manipulation train \
  --robot innate_mars --scenario mixed --task-mix tracking-retention \
  --envs 1 --seeds 44 --initialization-steps 0 --steps 2048 \
  --randomization-scale 0.05 --eval-every 2048 --eval-episodes 1 \
  --wandb-mode offline --output logs/manipulation/retention-diagnostic
```

One validation episode per category and 2048 steps test execution only. They
cannot establish accuracy, useful occupancy inference or retention improvement.
Use compatible warm-start weights for a subsequent performance experiment,
paired tracking/loaded validation with adequate per-category denominators,
held-out objects/grasps, an independent calibration split and a no-regression
comparison with the original tracking checkpoint. Reject degenerate all-empty
predictors even if aggregate accuracy is high. The public 30-case suite and all
hardware gates remain separate and must pass before promotion.

The [startup audit](../experiments/mars_policy_startup_audit_20261025.md)
adds an earlier gate: two-second stationary and nearby-pose holds on the actual
TAMP reset states, followed by the first certified pregrasp segment. The
[completed hold experiments](../experiments/mars_hold_repair_20261025.md) implement
tracking transfer, categorical identity normalization, fixed-current-base
sampling, physical pregrasp holds and a learned feedback head. They retain 70%
ordinary tracking and all smoothness/load/safety terms. Near-object holds are
excluded from certified closed-empty examples. The feedback model greatly
improves position holds, but continuous six-DOF quiet, rotation tracking, the
paired no-regression gate and public TAMP execution still fail. Keep these gates
open; do not treat a short accurate hand hold as manipulation qualification.

The next [hold/Cartesian iteration](../experiments/mars_hold_cartesian_20261025.md)
isolates the wheel feedback loop, passes selected continuous stationary holds,
and adds an explicit whole-path Cartesian TAMP backend. Fixed-base training now
has matching execution boundaries, full-body quiet checks and same-target pose
progress. Nearby wrist/translation targets remain a bounded part of the reach
share; manipulation remains 30% of reset selection. Preserve negative loaded
validation and full program failures. A stationary hold or small endpoint error
cannot replace the public task, retention, force or wheel gates.


A further 65,536-step precision refinement improves selected nearby positions
to approximately 0.4 mm while missing the 0.02 rad angular bound. Current public
Cartesian execution remains 0/3 programs. Phase handoff and a frozen intermediate
reference clock are repaired without changing final thresholds. Keep the
orientation, loaded and public-task gates open. The next reward ablation enables
existing goal-gated settling so remaining pose correction does not compete with
premature quiet incentives; the mixture and safety penalties remain unchanged.
