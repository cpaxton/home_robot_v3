# Native pregrasp diagnosis and ordered motion, October 17

The [grasp-order agent baseline](tamp_agent_baseline_grasp_v2_20261017.md)
exposed two pregrasp revalidation failures. This follow-up isolates that phase
through live ZMQ execution, without an LLM, cameras, or local GPU. Fixtures are
the same archived familiar/left/right Stretch scenes. These diagnostics do not
rescore the six agent episodes or establish two-object task success.

## Reproduced cause

A settled-start right probe reproduces `linear_collision_at_1`. Its actual
collision pair is **rubber_tip_right / table**, with **1.529 mm penetration**
in the revalidated path sample. The preceding measured state has lift 0.6125 m
and four arm stages near 0.0533 m; the next command combines lift to 0.6310 m,
arm stages near 0.0657 m, and wrist yaw/pitch/roll changes. Unequal tracking of
those simultaneous commands changes the next measured start relative to the
nominal geometric path. This is a real predicted collision, not missing grasp
contact evidence or an IK call-budget exhaustion.

The baseline's left `invalid_start` has **not** been reproduced with detailed
evidence. That reason alone cannot establish whether bounds or collision failed.
It is not claimed solved by the right-side diagnosis.

## Implementation

- Native pregrasp candidate planning and measured-state replanning request an
  ordered coupled-arm trajectory: lift, wrist orientation, then extension.
- Each segment retains dense joint-limit and collision checking. A rejected
  ordered route rejects that grasp candidate; it does not fall back to the
  simultaneous path. Other bounded grasp alternatives are still searched.
- Revalidation now preserves the underlying bound/collision detail. Collision
  failures retain private contact pairs and the measured/commanded configurations.
  Public TAMP JSON stays sanitized; known collision/bounds codes are preserved.
- Physics, collision tolerance, retention limits, and controller gains remain
  unchanged. No contact sensor or privileged hardware capability is introduced.

## Verification

The left, right, and familiar pregrasp-only probes all complete the phase after
the change. Separate full live cylinder transfers pass on left **1/1** and right
**1/1**, including measured lift, bilateral hold, placement, support and retreat.
Across those five ordered episodes, **5/5 pregrasps complete**.

[Machine-readable evidence](tamp_pregrasp_20261017.json) retains every attempt,
including initial rejections and the reproduced control failure. These intentionally stop before grasp with a diagnostic error;
that terminal partial envelope is **not** a complete transfer success.

The main targeted run passes **120 tests, zero failures or skips**. After a
failure-detail-only refinement, all four affected focused tests pass again
(these overlap the 120, not 124 distinct tests). New regressions verify lift
before extension when the diagonal is nominally clear, refusal to fall back
when the ordered route is blocked, and preservation of an edge collision's
underlying contact pair. Ruff and whitespace checks pass.

## Artifacts and scope

Archived sources, patch, traces, and test XMLs are under
`/home/cpaxton/runs/emet/tamp-pregrasp-20261017/`.
Original traces: `/tmp/tamp-pregrasp/<case>-<repeat>/result.json`, containing
private synchronized qpos/qvel, planner/execution events, and tool envelopes.
Every probe enables the physical actuation audit in `actuation.jsonl`.
Initial unscored attempts include a left `scene_changed_replan` and a right
planning rejection; these are retained, not counted as completed pregrasps.

Probes run via `emet jobs run --no-gpu-exclusive --cpu-safe`, with software
rendering and cameras disabled. They exercise native tools and live actuator
commands, not the `run agent` reasoning loop. Full agent regression and broader
starting placements remain separate gates.

All 1025 audited commands across the nine diagnostic attempts
have zero forbidden entries. The source patch hash is recorded in the JSON.
These successful transfers are scripted privileged checks, not a revised agent
completion rate; the latest full agent result remains 2/6.

## Subsequent agent validation

The [actual run-agent rerun](tamp_agent_baseline_pregrasp_v3_20261017.md) completes
6/6 attempted transfers without pregrasp or retention failures. Full tasks remain
2/6, with 6/12 satisfied goals. Both left cylinder transfers complete; cube/right
planning and one malformed agent argument remain. No right transfer is attempted,
so this does not establish right-side agent execution robustness.
