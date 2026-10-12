# Matched TAMP regressions, October 12

Job `20261012_184512_2ba365` compares main `b7a1c22d` to metadata refresh
`0476f9e9` (#203). Two simulator workers share one exclusive managed GPU job,
with disjoint ports and CPU allocations. No large neural model is loaded; peak
observed GPU use was about 3 GB. All task outcomes come from result JSON, not
process exit status (the oracle runner exits zero for fixture rejections).

## Assisted public-tool control

Both revisions run the same RBY1 kitchen-0 bowl/open/plan/execute task. The public
API has no seed argument; this is a matched task check, not identical dynamics
or a statistical repeatability estimate. Object latch, placement assistance and
base teleport are enabled; this does not establish native-contact execution.

| Revision | Task result | Wall time | ACKs | Maximum ACK latency |
| --- | --- | --- | --- | --- |
| main b7a1c22d | pass | 573.4 s | 749 | 1.250 s |
| #203 0476f9e9 | pass | 563.2 s | 783 | 2.267 s |

Neither timed out. Both had the known teardown cancellation rejection after task
success. A separate serial candidate replay also passed in 385.8 s; concurrent
simulation therefore should not be used to claim a runtime speedup.

## Oracle fixture checks

Four historical pilot construction requests were re-run on main; old certificates
were not edited to bypass scene/implementation checks. Two were admitted:
Stretch cleanup (3 objects) and navigation (8 objects). Both Nori requests failed
before the candidate revision was tested: cleanup rejected all object approaches
for collision, navigation rejected scatter reproduction. Admission coverage is
**2/4**, not 4/4; this does not reproduce the old four-fixture success claim.

Both revisions then replayed the exact same newly admitted baseline certificates:

| Fixture | main | #203 |
| --- | --- | --- |
| Stretch cleanup | pass, 3/3 relocated | fixture_pose_mismatch before MCTS |
| Stretch navigation | pass, 8/8 relocated | scatter_not_reproduced before MCTS |

These are preserved compatibility failures. #203 must not be described as having
zero regressions until expected-versus-measured poses establish the cause. The
candidate's geometry-refresh regression proves old metadata can remain stale
after free physics, but that fact alone does not explain every live rejection.

Artifacts: `/home/cpaxton/runs/emet/tamp-regression-pairs-20261012/`, containing
protocol, exact commands, source hashes, baseline reference certificates, and
per-case results. Follow-up diagnostics are a separate run and cannot replace
these outcomes.

## Follow-up diagnostics (separate denominator)

Job `20261012_185057_8c5cae` measured the Stretch cleanup kettle center at
`[1.18934345, 1.19256044, 0.16410459]` versus certificate
`[1.18933016, 1.19132997, 0.11113207]`: approximately 5.3 cm vertical
disagreement, with essentially unchanged orientation. A fresh candidate
construction attempt relocated 2/3 objects, then failed `teleport_grasp_failed`
on the kettle. It did not produce a new admitted certificate.

The oracle workflow commands named-body origins but verifies published AABB
centers; Stretch's immediate command-side metadata patch previously wrote the
command position into the center field. This frame inconsistency is a concrete
code finding and a candidate cause, not yet a complete causal reproduction of
the live failure. Fix frame handling and measure settled state before rebuilding
certificates. Do not relax verification or restore stale metadata.

Baseline Nori diagnostics found the same `lift_top_link` / `lift_middle_link`
contact (about 4.8 cm penetration) in all 16 approach candidates, plus gripper
mechanism contacts. Some candidates also hit scene obstacles. Inspect the model
and validate narrowly scoped mechanism geometry/exclusions before changing any
collision policy; do not suppress all self-collisions.

## Merge ledger

- #204 merged as `631cfdfd`: sequential pregrasp/grasp/lift endpoint admission,
  109 final targeted checks and a full assisted RBY1 pass. Sourccey now rejects
  unreachable candidate sequences before executing; that is not task success.
- #203 has an independent approving review. Keep its oracle compatibility gate
  open pending the frame/verification fix above; matched RBY1 control passed.
- #205's first Stretch public-tool run (`20261012_185308_3ed2cc`) failed
  `base_pose_validation_unavailable`, with no timeout. The server endpoint
  lacked its Stretch client method. Commit `2e698c20` shares query transport
  across both clients; 16 endpoint/approach checks pass. Corrected live replay
  `20261012_191644_3ba157` passed the full public-tool sequence in 56.8 s,
  with 10 ACKs (maximum 0.184 s), zero command failures and no timeout. This
  uses explicit teleport-oracle manipulation/base assistance. #205 merged as
  `9097d41f`; native-contact execution is still unvalidated.

Native observed providers/execution remain unavailable. None of these assisted
controls is a native-contact or LLM-agent success claim. Remaining Sourccey
work includes measured non-arm mount-joint synchronization and establishing a
reference-solvable task before changing IK budgets or acceptance thresholds.
