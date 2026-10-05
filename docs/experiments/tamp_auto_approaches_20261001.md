# Online TAMP approach selection (2026-10-01)

The shared planner now generates 16 base poses around the live object position,
checks them with a read-only simulator endpoint collision query, and chooses the
first collision-clear candidate with a reachable grasp. Explicit approach poses
and external route validators remain supported. Query failures fail closed.
No saved approach audit or preselected-pose adapter is used in this experiment.

Source: `facb69f3`, branch `feat/tamp-agent-execution`.
Managed job: `20261001_080251_e218ec`.
Artifacts: `~/runs/emet/tamp-validated-fixtures-20260928/auto-approach-r11/`.
The runner rebuilds scene00 cleanup0, executes reference admission, freezes the
new certificate, then executes three fresh-process MCTS repeats. Admission passed;
it is recorded separately from scored replays. The old certificate was not edited
to bypass implementation fingerprint validation.

## Approach decisions

All completed trials selected these candidates (indices are zero-based):

| Object | Selected candidate | Poses rejected for collision |
| --- | ---: | ---: |
| Potato | 2/16 | 9 |
| Apple | 4/16 | 7 |
| Kettle | 0/16 | 5 |

These match the previously successful assisted alternatives, now selected online.
The regression suite passed 139 tests, including read-only state preservation,
collision rejection, trying another candidate after IK failure, explicit approach
overrides, external route validation, and command-receipt query failures. Ruff and
`git diff --check` passed.

## Execution evidence

All three repeats finished: **2/3 full tasks passed, 7/9 objects relocated**.

| Repeat | Full task | Objects relocated | Motion failures |
| --- | --- | ---: | ---: |
| 1 | pass | 3/3 | 0 |
| 2 | fail | 1/3 | 2 |
| 3 | pass | 3/3 | 0 |

Replay 1 passed all three relocations. Replay 2 relocated only the potato:
the apple failed measured pregrasp arrival at 86.026 mm against the unchanged
35 mm bound. Its server targets matched the planned joint commands. The kettle
reached pregrasp, but its next arm plan failed with `invalid_start`.
These failures are retained; selecting a clear endpoint is insufficient to claim
robust execution. The logs do not establish whether apple tracking failure was
caused by contact, settling time, or another dynamics issue. The invalid-start
predicate checks both joint bounds and any configured collision checker; the
generic error alone does not identify which predicate failed.

## Residual failure diagnosis

Replay 2's `process.log` gives concrete signals for both failures:

- **Apple (pregrasp).** The archived measurement reports `left_arm_joint1`
  at -1.025518 rad against -1.033738 rad (0.008220 rad error), and
  `left_arm_joint2` at 1.624856 rad against 1.632452 rad (0.007597 rad error).
  The largest joint error is 0.035933 rad at `torso_joint1`. The prior claim of
  a roughly 1 rad shoulder stall was a transcription error. Cartesian error
  remains 86.026 mm; these joint values alone do not establish contact or a
  joint-limit cause. Server targets match the planned commands.
- **Kettle (`invalid_start`).** Pregrasp arrival was accepted (33.7 mm), then the
  grasp plan failed on `invalid_start`. This harness runs `manip_collision="none"`,
  so the collision predicate is absent and `invalid_start` can only be a joint-bounds
  violation on the measured start posture. `dec3e26e` adds an `ArmRrtPlanResult.detail`
  field so a replay names the offending joint instead of the generic reason.

Both trials selected a collision-clear base endpoint; this does not establish
whole-arm feasibility or exclude approach-dependent execution failures. Re-diagnosis (contact reconstruction for the apple, joint-bound detail for the
kettle) and a fresh frozen-registry replay are the next steps.

## Scope

This is one fixture with three process repeats, using GT/MCTS, base teleportation
and object latch assistance. It is not a learned-agent or physical-grasp result.
The query checks declared base endpoint geometry, not swept arm/payload paths.
RBY1 arm collision coverage remains incomplete. Do not pool these trials with the
earlier three assisted successes or historical benchmark scores. Broader sweeps
remain pending execution diagnosis and physical collision coverage.
