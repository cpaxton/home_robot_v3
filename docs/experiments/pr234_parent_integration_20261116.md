# PR #234 parent integration — November 16

Integrated updated #231 `f5fda95b` (including main `fb9f9bf1`) on clean frozen
runtime source `bceabc97`. Expected independent reviews were not visible on
GitHub at audit time. This is author review and integration evidence.

The conflict resolution preserves main's robot-specific grasp/approach/transport
hooks, model/boot/policy cache binding, safe static-support rejection evidence,
and #234's typed verification witnesses. Shared verification uses the robot's
configured finger bodies. The parent's explicit rejection of Stretch-only
`bounded_release` on Mars remains intact; legacy Mars behavior is preserved.

Reviewed the later additions beyond the original margin-only review: geometric
candidate ordering preserves the pool and shared IK cap; measured gripper hold
uses ordinary encoder/position commands followed by settling within the same
closing budget; sampled original-reference retention rejects immediately; finite
public witnesses survive primitive/provider/task receipts without stale reuse.
The 0.5 mm per-segment reserve narrows IK targets only, not every path waypoint.
No measured state is clamped and no collision/retention threshold is relaxed.

Validation:

- **467 affected tests pass**, zero failures/errors/skips. Includes native and
  observed adapters, Mars hooks/arrival, payload departure, API/provider/task
  receipts, contact/release geometry, configuration and CHAT integration.
- Frozen familiar two-goal native control: **2/2 transfers** complete; both
  independent final object goals are in-region, supported and settled.
  **482 audited commands, zero forbidden commands**; clean source, no local GPU.
- Maximum recorded original-reference translation/rotation residuals: cube
  0.00203 m / 0.00590 rad; cylinder 0.00276 m / 0.01460 rad. These are sampled
  measurements for this run, not tracking-error guarantees.
- Ruff, incremental whitespace checks and full paper build pass.

The smoke used the predeclared
[protocol](../../configs/eval/tamp_pr234_integration_20261116.json). Raw artifacts
remain under `.tamp-runs/native-stack-20261116/pr234/`; the
[compact record](pr234_parent_integration_20261116.json) contains source/job and
artifact hashes. The 467 checks overlap the parent's suite and must not be
added to it as a disjoint total.

The historical pilot remains **2/3 tasks, 5/7 transfers**, including the
swap-right25 original-reference retention failure. This familiar single smoke
is not an updated success rate or observed/hardware qualification. No agent
cohort was rerun for #234. Merge #231 before this PR, then integrate the resulting
main after the parent squash. Expected independent current-head review remains
outstanding. #242 and later recovery behavior remain separate stack work.
