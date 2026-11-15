# Prepared directional curriculum and fresh audit

This implements the case-preparation part of the
[ordered WBC plan](mars_wbc_next_steps.md). The production/reset validation of
current-velocity feedback remains owned by
[PR #239](https://github.com/cpaxton/home_robot_v3/pull/239). No native rollout,
teacher admission, PPO update, policy export or default change is performed by
this preparation.

## Exact inputs

The versioned [manifest](../../configs/benchmarks/mars_directional_curriculum_v1.json)
binds the original supported-mastery parent and C1 control files by SHA-256.
The [loader](../../src/emet/rl/manipulation/directional_curriculum.py) returns
the three-case control, seventeen-case candidate, unchanged twelve-case
diagnostic, and twenty-four-case audit as separate native suite dictionaries.
Use this loader rather than passing the manifest to the frozen canonical
mastery-suite parser. The existing C1 specification and guard remain unchanged.

| Candidate training group | Cases |
| --- | --- |
| Base, five | `hold`, `forward`, `reverse`, `yaw_left`, `yaw_right` |
| Joint 1, two | `shoulder_left`, `shoulder_right` |
| Joint 2, two | `pitch_up`, `pitch_down` |
| Joint 3, two | `elbow_up`, `elbow_down` |
| Joint 4, two | `wrist_up`, `wrist_down` |
| Joint 5, two | `roll_left`, `roll_right` |
| Coupled, two | `forward_turn`, `reverse_turn` |

All seventeen use existing full-amplitude, free-space training cases from the
parent. This covers signed joint-coordinate directions; it is not a certificate
of task-space rank, collision clearance or physical tracking. The FK witness
continues to construct pose references privately and is not an actor hint.

Candidate scheduling targets actual control ticks at base/arm/coupled weights
30/50/20. The control retains its declared 50/50 base/arm C1 schedule and its
original hold/two-quarter-pitch cases. The curriculum, including its declared
allocation, is the experimental variable. Both derived configurations enable
current-velocity feedback from construction; only that config field is
overridden. References, tolerances, rewards, limits and terminal-hold gates are
copied unchanged. This does not qualify the option through reset or hardware.

## Audit seal and limits

| Audit group | Count | Reset seeds | Motion family |
| --- | ---: | --- | --- |
| Base | 8 | 34820000–34820007 | Combined translation and yaw, unlike the training base primitives |
| Arm | 8 | 34820008–34820015 | Two-or-more-joint combinations; four include existing near-straight posture templates |
| Coupled | 8 | 34820016–34820023 | New base/arm combinations at smaller amplitudes |

All audit IDs and seeds are unique and disjoint from every parent admission
case. Exact motion vectors are also distinct from parent cases; bounds are the
existing nearby-motion limits. The loader rejects duplicate identities, reused
seeds/motions, missing signed axes, changed contracts, unknown fields and
nonfinite/out-of-range data. New seeds and combinations constitute a bounded
audit, not a broad robustness or reliability study.

These are proposed FK-derived goals, not admitted references. A teacher must
verify the actual reset, generated task-space motion, limits, contact and
independent guards on the corrected plant before learning. If a case is
rejected, retain that outcome and explicitly version any pre-training geometry
revision. Do not remove cases after observing a trained candidate.

The twelve-case diagnostic includes training-family cases and is not a fresh
holdout. The control/candidate/diagnostic/audit unions contain 47 distinct case
IDs; do not present repeated cohort readouts as independent trials. Report each
cohort separately and preserve its physical reset/reference hashes.

## Validation and next gate

Twenty-two pure case/contract tests pass, including parent preservation, exact
coverage and rejection of schema, identity, contract and geometric-bound drift.
The final parent-mutation check also rejects changes under Python `-O`.
Ruff passes. The new manifest is bound into the canonical regression source
state. No parent receipt is reused as full validation of this new source.

The next stages remain conditional:

1. The corrected-plant production/reset baseline in #239 is complete. Diagnose
   actual TAMP tracking and initialization-sensitive planning before selecting
   more training: general tracking improved, but no learned actor passed these
   six actual TAMP pregrasps. See the
   [completed report](../experiments/mars_current_feedback_reset_validation_20261115.md).
2. Freeze this case source and obtain its own full regression receipt and
   matching teacher admission for every required case and the selected plant.
   Publish the acceptance budget anchor separately from fresh admission.
3. Only if the general learning comparison is still justified, integrate the
   separately versioned curriculum with a training/pairing path and register
   the fixed seed90, 16,384-per-variant comparison. This loader does not add a
   training CLI or bypass the existing guards.

Preparation is complete when the manifest and strict loader are reviewable;
admission and training remain open in [TODO.md](../../TODO.md).
